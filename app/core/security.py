"""Header name for the service-to-service auth convention used by this repo's OUTBOUND
clients only (app.services.service_clients, calling User Behavior Service / Candidate
Service). The INBOUND gate that used to protect this service's own routes
(require_internal_api_key) has been removed -- the consuming backend team confirmed
X-Internal-API-Key protection is no longer part of this service's own security
architecture. `API_KEY_HEADER_NAME` stays because the outbound convention is unrelated and
unchanged: `follow-service`'s own internalAuthMiddleware still gates a real
service-to-service route with this exact header/env-var convention (see README
"Service-to-service auth"), so app.services.service_clients still needs the name to send it.
"""
from __future__ import annotations

API_KEY_HEADER_NAME = "X-Internal-API-Key"
