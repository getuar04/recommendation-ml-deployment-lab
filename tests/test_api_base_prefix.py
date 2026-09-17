"""Base-path standardization regression test.

Recommendation ML Service's public API base path follows the company-wide
`/api/v1/<service-name>/...` convention (see app.core.config.API_V1_PREFIX, the single
authoritative place app.main reads to mount every router). This file proves three things a
prefix refactor could plausibly get wrong, none of which the broader OpenAPI contract test
(test_openapi_contract.py) asserts explicitly:

1. representative routes actually resolve under the new canonical prefix;
2. the old bare `/api/v1/...` prefix no longer resolves anywhere (no accidental
   backward-compatibility alias was left in place);
3. no route was accidentally double-prefixed (e.g. .../recommendation-ml-service/api/v1/...).
"""

from app.core.config import API_V1_PREFIX

_REPRESENTATIVE_PATHS = {
    "/health",
    "/recommendations",
    "/candidates/generate",
    "/model/status",
}


def test_api_v1_prefix_constant_matches_the_company_convention():
    assert API_V1_PREFIX == "/api/v1/recommendation-ml-service"


def test_representative_routes_resolve_under_the_new_canonical_prefix(client):
    spec = client.get("/openapi.json").json()
    actual_paths = set(spec["paths"].keys())
    for suffix in _REPRESENTATIVE_PATHS:
        path = API_V1_PREFIX + suffix
        assert path in actual_paths, f"expected {path} in OpenAPI paths"

    assert client.get(f"{API_V1_PREFIX}/health").status_code == 200
    assert client.get(f"{API_V1_PREFIX}/model/status").status_code == 200


def test_old_bare_api_v1_prefix_no_longer_resolves(client):
    """No permanent dual-routing: the canonical API uses only API_V1_PREFIX (see the
    base-path refactor's backward-compatibility decision -- no existing code/deployment
    configuration was found requiring a temporary alias)."""
    for suffix in _REPRESENTATIVE_PATHS:
        response = client.get(f"/api/v1{suffix}")
        assert response.status_code == 404, f"/api/v1{suffix} should no longer resolve"


def test_no_route_is_accidentally_double_prefixed(client):
    spec = client.get("/openapi.json").json()
    for path in spec["paths"]:
        assert path.startswith(API_V1_PREFIX), f"{path} does not start with {API_V1_PREFIX}"
        remainder = path[len(API_V1_PREFIX):]
        assert "/api/v1" not in remainder, f"{path} looks double-prefixed"
