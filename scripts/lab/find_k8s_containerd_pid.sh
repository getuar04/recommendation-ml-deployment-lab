#!/bin/sh
# Prints (stdout, and ONLY on success) the host PID of the containerd daemon serving the
# Kubernetes (`k8s.io`) containerd namespace on the LOCAL Docker Desktop Kubernetes node
# (context `docker-desktop`) -- or exits non-zero and prints nothing to stdout if it cannot
# identify exactly one unambiguous candidate.
#
# WHY this is derived at runtime, never hardcoded: Docker Desktop Kubernetes runs a SEPARATE
# containerd instance/namespace (`k8s.io`, backed by KIND under the hood -- see kubelet's own
# `--provider-id=kind://docker/desktop/...` on this node) from the Docker Engine's own
# (`moby`, what `docker build`/`docker images` populate). Both run side by side inside the
# same Docker Desktop VM. A locally built image is therefore NOT automatically visible to
# kubectl/kubelet here (proven via `kubectl get node -o json` `.status.images` during
# diagnosis of the ErrImageNeverPull LAB failure). This script exists so a later step can
# bridge the two by importing an image directly into the k8s.io containerd's own store.
# The k8s.io containerd's PID is a runtime detail of the current Docker Desktop VM instance
# and is NOT guaranteed stable across a Docker Desktop restart -- it must be re-derived every
# run, never hardcoded.
#
# Must run inside a container sharing the HOST pid namespace (`docker run --pid=host ...`),
# since the PID being searched for belongs to a process outside that container. Uses only
# /proc parsing (no `ps`/procps dependency, no network access) so discovery never depends on
# a package install succeeding.
#
# Selection method: every `containerd-shim-runc-v2` process self-reports its own
# `-namespace` argument on its command line. This script groups shim processes by their
# parent PID (the containerd daemon that spawned them) and by namespace (`k8s.io` vs
# `moby`), then requires ALL of the following before printing a PID:
#   1. At least one k8s.io-namespaced shim exists.
#   2. All k8s.io-namespaced shims share exactly ONE parent PID (unambiguous).
#   3. That parent PID's own command line contains "containerd" and is not itself a shim.
#   4. That same parent PID is NOT also the parent of any moby-namespaced shim (which would
#      mean the "Kubernetes" containerd and the Docker Engine's own containerd are actually
#      the same process -- an unexpected topology this script refuses to guess about).
set -eu

k8s_ppids=""
moby_ppids=""

for cmdline in /proc/[0-9]*/cmdline; do
    [ -r "$cmdline" ] || continue
    pid=$(basename "$(dirname "$cmdline")")
    args=$(tr '\0' ' ' < "$cmdline" 2>/dev/null) || continue
    case "$args" in
        *containerd-shim-runc-v2*)
            ppid=$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null) || continue
            [ -n "$ppid" ] || continue
            case "$args" in
                *"-namespace k8s.io"*) k8s_ppids="$k8s_ppids $ppid" ;;
                *"-namespace moby"*) moby_ppids="$moby_ppids $ppid" ;;
            esac
            ;;
    esac
done

k8s_ppids_unique=$(printf '%s\n' $k8s_ppids | sort -u | grep -v '^$' || true)
moby_ppids_unique=$(printf '%s\n' $moby_ppids | sort -u | grep -v '^$' || true)

candidate_count=$(printf '%s\n' "$k8s_ppids_unique" | grep -c . || true)

if [ "$candidate_count" -eq 0 ]; then
    echo "REFUSING: no containerd-shim-runc-v2 process reports namespace k8s.io -- cannot identify the Kubernetes containerd daemon on this host." >&2
    exit 1
fi

if [ "$candidate_count" -ne 1 ]; then
    echo "REFUSING: found $candidate_count distinct parent PIDs for k8s.io-namespaced shims, expected exactly 1 (ambiguous): $(printf '%s ' $k8s_ppids_unique)" >&2
    exit 1
fi

candidate_pid=$(printf '%s\n' "$k8s_ppids_unique" | head -1)

candidate_cmd=$(tr '\0' ' ' < "/proc/$candidate_pid/cmdline" 2>/dev/null) || {
    echo "REFUSING: could not read /proc/$candidate_pid/cmdline for the candidate PID." >&2
    exit 1
}
case "$candidate_cmd" in
    *containerd-shim*)
        echo "REFUSING: candidate PID $candidate_pid is itself a containerd-shim, not a containerd daemon: $candidate_cmd" >&2
        exit 1
        ;;
    *containerd*) ;;
    *)
        echo "REFUSING: candidate PID $candidate_pid does not look like a containerd daemon: $candidate_cmd" >&2
        exit 1
        ;;
esac

if printf '%s\n' "$moby_ppids_unique" | grep -qx "$candidate_pid"; then
    echo "REFUSING: candidate PID $candidate_pid also parents a moby-namespaced shim -- refusing to import into what may be the Docker Engine's own containerd instead of a distinct Kubernetes one." >&2
    exit 1
fi

echo "$candidate_pid"
