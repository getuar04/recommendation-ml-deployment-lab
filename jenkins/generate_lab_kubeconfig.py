"""Generates a LAB-only copy of the host's `docker-desktop` kubeconfig for the Jenkins LAB
container, rewriting only what must change for it to be reachable from inside a separate
Linux container -- never the host's own kubeconfig, which this script only reads.

Problem (verified empirically before writing this script -- see jenkins/run-jenkins-lab.ps1
and the phase notes): Docker Desktop's kubeconfig points `server` at
`https://127.0.0.1:<port>`, a Windows-host loopback address that means "this container
itself" from inside a separate Linux container's own network namespace -- unreachable.
Docker Desktop's `host.docker.internal` DNS name reaches the same port from inside a
container, but the API server's TLS certificate does not list `host.docker.internal` as a
Subject Alternative Name (verified via `openssl x509 -text`: DNS:desktop-control-plane,
DNS:kubernetes[...], DNS:localhost, IP:127.0.0.1, IP:10.96.0.1, IP:<pod-ip> -- no
host.docker.internal), so a plain server-address rewrite alone would fail TLS hostname
verification.

Fix: dial `host.docker.internal`, but tell the client to verify the certificate against
`localhost` -- a name the certificate genuinely carries -- via kubeconfig's own
`tls-server-name` field. This is NOT `--insecure-skip-tls-verify`: `certificate-authority-
data` and the full chain/hostname check still apply, just against a hostname decoupled from
the dial address (empirically verified: `curl --cacert <ca> --resolve
localhost:<port>:<host.docker.internal IP> https://localhost:<port>/version` succeeds with
full certificate validation).

Refuses to run (exit 1, no file written) unless the host's current-context is exactly
`docker-desktop`. The generated file carries ONLY that one context/cluster/user entry --
never any other context that might exist in the host kubeconfig -- so nothing else (in
particular no company-owned context) can ever reach the Jenkins container through this file.
"""
from __future__ import annotations

import copy
import re
import sys

import yaml

EXPECTED_CONTEXT = "docker-desktop"


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <host-kubeconfig> <output-path>", file=sys.stderr)
        return 2
    src_path, dst_path = sys.argv[1], sys.argv[2]

    with open(src_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    current = cfg.get("current-context")
    if current != EXPECTED_CONTEXT:
        print(
            f"REFUSING: host kubeconfig current-context is {current!r}, expected exactly "
            f"{EXPECTED_CONTEXT!r}. Not generating a lab kubeconfig for anything else.",
            file=sys.stderr,
        )
        return 1

    context_entry = next((c for c in cfg.get("contexts", []) if c["name"] == current), None)
    if context_entry is None:
        print(f"REFUSING: no context entry named {current!r} in {src_path}.", file=sys.stderr)
        return 1

    cluster_name = context_entry["context"]["cluster"]
    user_name = context_entry["context"]["user"]

    lab_cfg = copy.deepcopy(cfg)
    rewritten_server = None
    for cluster in lab_cfg.get("clusters", []):
        if cluster["name"] != cluster_name:
            continue
        server = cluster["cluster"].get("server", "")
        if "127.0.0.1" not in server and "localhost" not in server:
            print(
                f"REFUSING: cluster {cluster_name!r} server {server!r} is not a loopback "
                "address -- refusing to rewrite what looks like a non-local target.",
                file=sys.stderr,
            )
            return 1
        rewritten_server = re.sub(
            r"://(127\.0\.0\.1|localhost)(:\d+)", r"://host.docker.internal\2", server
        )
        cluster["cluster"]["server"] = rewritten_server
        cluster["cluster"]["tls-server-name"] = "localhost"

    if rewritten_server is None:
        print(f"REFUSING: cluster {cluster_name!r} not found in {src_path}.", file=sys.stderr)
        return 1

    # Carry over ONLY the docker-desktop context/cluster/user -- nothing else the host
    # kubeconfig might contain.
    lab_cfg["contexts"] = [c for c in lab_cfg["contexts"] if c["name"] == current]
    lab_cfg["clusters"] = [c for c in lab_cfg["clusters"] if c["name"] == cluster_name]
    lab_cfg["users"] = [u for u in lab_cfg["users"] if u["name"] == user_name]
    lab_cfg["current-context"] = current

    with open(dst_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(lab_cfg, f, default_flow_style=False)

    print(
        f"OK: generated {dst_path} "
        f"(context={current}, server={rewritten_server}, tls-server-name=localhost)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
