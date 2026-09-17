"""Demo/foundation Candidate Service for the VIDEO SOCIAL/COLLABORATIVE candidate flow.

Ownership boundary (unchanged from the Recommendation ML Service side): this package owns
follow-graph traversal, similar-user discovery, and strong-engagement filtering -- the things
Recommendation ML Service must never do. It never touches app.ml (features/model/reranker) and
never imports from app.services.recommendation_service; the only thing it produces is plain
candidate payloads shaped like app.schemas.recommendation_schemas.Candidate, which a caller
posts to the existing, unmodified POST /api/v1/recommendation-ml-service/recommendations endpoint.

Transport to the real Follow Service / User Behavior Service is intentionally undecided (REST,
Kafka, gRPC, internal SDK -- not yet agreed). Nothing in this package assumes a transport: see
candidate_service.providers.base for the Protocols real integrations would implement later.
Only demo providers (local, deterministic, no network) exist today -- see
candidate_service.providers.demo_*.
"""
