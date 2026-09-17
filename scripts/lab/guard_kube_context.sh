#!/usr/bin/env bash
# Fail-closed safety guard for the LOCAL deployment lab.
#
# NOT executed as part of Phase 2 (config preparation only). This exists so every future
# lab script/Jenkins stage that runs kubectl or helm can `source` or invoke it FIRST and
# abort before touching a cluster, instead of relying on a developer remembering to check
# their context by hand.
#
# Refuses to continue unless ALL of the following hold:
#   1. LAB_MODE=true is explicitly set (nothing runs "by accident" outside the lab).
#   2. `kubectl config current-context` is exactly EXPECTED_KUBE_CONTEXT (default:
#      docker-desktop) -- never a company EKS context such as g2r-eks or eks-g2r-dev-eu.
#   3. If K8S_NAMESPACE is set by the caller, it is exactly EXPECTED_NAMESPACE (default:
#      rms-lab) -- never g2r-prod or g2r-dev.
#
# Usage from a future lab script:
#   LAB_MODE=true K8S_NAMESPACE=rms-lab source scripts/lab/guard_kube_context.sh
set -euo pipefail

EXPECTED_KUBE_CONTEXT="${EXPECTED_KUBE_CONTEXT:-docker-desktop}"
EXPECTED_NAMESPACE="${EXPECTED_NAMESPACE:-rms-lab}"

if [ "${LAB_MODE:-}" != "true" ]; then
  echo "REFUSING: LAB_MODE is not 'true'. This guard only permits execution inside the local deployment lab." >&2
  exit 1
fi

current_context="$(kubectl config current-context 2>/dev/null || true)"
if [ -z "${current_context}" ]; then
  echo "REFUSING: no active kubectl context (kubectl config current-context returned nothing)." >&2
  exit 1
fi

if [ "${current_context}" != "${EXPECTED_KUBE_CONTEXT}" ]; then
  echo "REFUSING: active kubectl context is '${current_context}', expected exactly '${EXPECTED_KUBE_CONTEXT}'." >&2
  echo "This guard exists precisely to stop a lab script from accidentally targeting a real cluster." >&2
  exit 1
fi

if [ -n "${K8S_NAMESPACE:-}" ] && [ "${K8S_NAMESPACE}" != "${EXPECTED_NAMESPACE}" ]; then
  echo "REFUSING: K8S_NAMESPACE is '${K8S_NAMESPACE}', expected exactly '${EXPECTED_NAMESPACE}'." >&2
  exit 1
fi

echo "OK: LAB_MODE=true, kube-context=${current_context}, namespace=${EXPECTED_NAMESPACE}."
