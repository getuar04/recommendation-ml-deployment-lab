from app.api import health_routes


class _BrokenEngine:
    """Stand-in for a SQLAlchemy engine whose PostgreSQL connection is unavailable --
    `.connect()` raises exactly like a real dropped/refused connection would, without
    needing an actual PostgreSQL outage to test against."""

    def connect(self):
        raise ConnectionRefusedError("simulated PostgreSQL outage")


def test_postgres_connected_returns_200_and_ok_status(client):
    response = client.get("/api/v1/recommendation-ml-service/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_postgres_connected_dependency_status_is_connected(client):
    body = client.get("/api/v1/recommendation-ml-service/health").json()
    assert body["dependencies"] == {"postgres": "connected"}


def test_postgres_disconnected_returns_503_and_degraded_status(client, monkeypatch):
    monkeypatch.setattr(health_routes, "engine", _BrokenEngine())
    response = client.get("/api/v1/recommendation-ml-service/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_postgres_disconnected_dependency_status_is_disconnected(client, monkeypatch):
    monkeypatch.setattr(health_routes, "engine", _BrokenEngine())
    body = client.get("/api/v1/recommendation-ml-service/health").json()
    assert body["dependencies"] == {"postgres": "disconnected"}


def test_health_reports_service_metadata_from_configuration(client, monkeypatch):
    monkeypatch.setattr(health_routes, "SERVICE_NAME", "custom-service-name")
    monkeypatch.setattr(health_routes, "SERVICE_VERSION", "9.9.9")
    monkeypatch.setattr(health_routes, "APP_ENV", "staging")
    monkeypatch.setattr(health_routes, "SERVICE_COMMIT", "abc1234")
    monkeypatch.setattr(health_routes, "SERVICE_BUILD_TIME", "2026-08-10T12:00:00Z")
    body = client.get("/api/v1/recommendation-ml-service/health").json()
    assert body["service"] == "custom-service-name"
    assert body["version"] == "9.9.9"
    assert body["environment"] == "staging"
    assert body["commit"] == "abc1234"
    assert body["buildTime"] == "2026-08-10T12:00:00Z"


def test_health_defaults_are_never_fabricated_commit_or_build_time(client):
    """An unconfigured checkout must report an honest "unknown", never an invented commit
    hash or timestamp."""
    body = client.get("/api/v1/recommendation-ml-service/health").json()
    assert body["commit"] == "unknown"
    assert body["buildTime"] == "unknown"


def test_health_does_not_leak_database_url_or_exception_details(client, monkeypatch):
    monkeypatch.setattr(health_routes, "engine", _BrokenEngine())
    response_text = client.get("/api/v1/recommendation-ml-service/health").text
    assert "postgresql" not in response_text.lower()
    assert "traceback" not in response_text.lower()
    assert "simulated" not in response_text.lower()


def test_health_response_shape_matches_the_documented_contract(client):
    body = client.get("/api/v1/recommendation-ml-service/health").json()
    assert set(body.keys()) == {
        "status", "service", "version", "environment", "commit", "buildTime", "message", "dependencies",
    }
    assert body["message"] == "Health check completed."
    assert set(body["dependencies"].keys()) == {"postgres"}
