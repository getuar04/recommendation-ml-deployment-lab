#!/bin/sh
# Prints (stdout, ONLY on success) the host PID of the containerd DAEMON serving the
# Kubernetes (`k8s.io`) containerd namespace on the LOCAL Docker Desktop Kubernetes node
# (context `docker-desktop`) -- or exits non-zero, printing nothing to stdout, if it cannot
# identify exactly one unambiguous candidate.
#
# PROVEN TOPOLOGY (do not reintroduce the earlier, disproven assumption): a
# containerd-shim-runc-v2 process is NOT a child of the containerd daemon that spawned it.
# containerd-shim-v2 deliberately double-forks and reparents each shim to the node's own
# init/subreaper (observed here as PID 848 `/sbin/init` -> systemd), precisely so a
# containerd restart never kills already-running containers. Grouping shims by raw PPid
# therefore only ever finds that shared init PID (never a real containerd daemon) -- this is
# standard containerd-shim-v2 behavior, not a Docker Desktop fluke, and will reproduce on
# every host that uses it. An earlier version of this script made that PPid assumption and
# failed closed every time as a result (empty stdout -> the caller then had nothing to pass
# to `nsenter --target`).
#
# CORRECTED METHOD: correlate by MOUNT NAMESPACE instead of PPid. A shim and the containerd
# daemon that manages it always run in the SAME mount namespace as each other (both live
# directly on the Kubernetes node's own root filesystem), while the Kubernetes node's
# containerd (this LAB's target) and the Docker Engine's own containerd (`moby` namespace,
# what `docker build`/`docker images` populate) run in DIFFERENT mount namespaces from each
# other -- separate nested environments inside the same Docker Desktop VM (Docker Desktop
# Kubernetes is backed by KIND; see this repo's diagnosis notes). So:
#   1. Find every containerd-shim-runc-v2 process reporting `-namespace k8s.io`.
#   2. Require they all share exactly ONE mount namespace (unambiguous node/environment).
#   3. Find every process whose own executable basename is EXACTLY `containerd` -- never a
#      shim, never merely a cmdline substring match (rejects lookalikes such as
#      `containerd-stargz-grpc`).
#   4. Require exactly ONE such daemon shares that same mount namespace with the k8s.io
#      shims (corroborated, not assumed, by the daemon's own executable basename) -- that is
#      the Kubernetes containerd daemon PID.
#   5. Refuse if that same mount namespace is also shared by any moby-namespaced shim (would
#      mean Kubernetes and the Docker Engine are, unexpectedly, the same environment --
#      refuse rather than guess).
#
# Never hardcodes a PID: Docker Desktop's runtime PIDs are not guaranteed stable across a
# restart and must be re-derived on every run. Must run inside a container sharing the HOST
# pid namespace (`docker run --pid=host ...`); uses only /proc parsing (no `ps`/procps, no
# network access), so discovery never depends on a package install succeeding.
set -eu

mnt_ns_of() {
    # Prints the numeric mount-namespace id for PID $1, or nothing if unreadable.
    readlink "/proc/$1/ns/mnt" 2>/dev/null | sed -n 's/^mnt:\[\([0-9]*\)\]$/\1/p'
}

k8s_shim_mnt_ns=""
moby_shim_mnt_ns=""

for cmdline in /proc/[0-9]*/cmdline; do
    [ -r "$cmdline" ] || continue
    pid=$(basename "$(dirname "$cmdline")")
    args=$(tr '\0' ' ' < "$cmdline" 2>/dev/null) || continue
    case "$args" in
        *containerd-shim-runc-v2*"-namespace k8s.io"*)
            ns=$(mnt_ns_of "$pid" || true)
            [ -n "$ns" ] || continue
            k8s_shim_mnt_ns="$k8s_shim_mnt_ns $ns"
            ;;
        *containerd-shim-runc-v2*"-namespace moby"*)
            ns=$(mnt_ns_of "$pid" || true)
            [ -n "$ns" ] || continue
            moby_shim_mnt_ns="$moby_shim_mnt_ns $ns"
            ;;
    esac
done

k8s_ns_unique=$(printf '%s\n' $k8s_shim_mnt_ns | sort -u | grep -v '^$' || true)
moby_ns_unique=$(printf '%s\n' $moby_shim_mnt_ns | sort -u | grep -v '^$' || true)

k8s_ns_count=$(printf '%s\n' "$k8s_ns_unique" | grep -c . || true)

if [ "$k8s_ns_count" -eq 0 ]; then
    echo "REFUSING: no containerd-shim-runc-v2 process reports namespace k8s.io with a readable mount namespace -- cannot identify the Kubernetes containerd daemon on this host." >&2
    exit 1
fi

if [ "$k8s_ns_count" -ne 1 ]; then
    echo "REFUSING: k8s.io-namespaced shims span $k8s_ns_count distinct mount namespaces, expected exactly 1 (ambiguous): $(printf '%s ' $k8s_ns_unique)" >&2
    exit 1
fi

k8s_ns=$(printf '%s\n' "$k8s_ns_unique" | head -1)

if printf '%s\n' "$moby_ns_unique" | grep -qx "$k8s_ns"; then
    echo "REFUSING: the k8s.io shims' mount namespace ($k8s_ns) is also used by a moby-namespaced shim -- refusing to treat this as a distinct Kubernetes environment from the Docker Engine's own." >&2
    exit 1
fi

daemon_candidates=""
for cmdline in /proc/[0-9]*/cmdline; do
    [ -r "$cmdline" ] || continue
    pid=$(basename "$(dirname "$cmdline")")
    args=$(tr '\0' ' ' < "$cmdline" 2>/dev/null) || continue
    exe=$(printf '%s' "$args" | awk '{print $1}')
    base=$(basename -- "$exe")
    [ "$base" = "containerd" ] || continue
    ns=$(mnt_ns_of "$pid" || true)
    [ "$ns" = "$k8s_ns" ] || continue
    daemon_candidates="$daemon_candidates $pid"
done

daemon_candidates_unique=$(printf '%s\n' $daemon_candidates | sort -u | grep -v '^$' || true)
daemon_count=$(printf '%s\n' "$daemon_candidates_unique" | grep -c . || true)

if [ "$daemon_count" -eq 0 ]; then
    echo "REFUSING: no process whose executable is exactly 'containerd' shares the k8s.io shims' mount namespace ($k8s_ns) -- cannot identify the Kubernetes containerd daemon." >&2
    exit 1
fi

if [ "$daemon_count" -ne 1 ]; then
    echo "REFUSING: found $daemon_count distinct 'containerd' daemon PIDs sharing the k8s.io shims' mount namespace, expected exactly 1 (ambiguous): $(printf '%s ' $daemon_candidates_unique)" >&2
    exit 1
fi

candidate_pid=$(printf '%s\n' "$daemon_candidates_unique" | head -1)

case "$candidate_pid" in
    ''|*[!0-9]*)
        echo "REFUSING: discovered candidate '$candidate_pid' is not a valid numeric PID." >&2
        exit 1
        ;;
esac

echo "$candidate_pid"
