Recommendation ML Service

FastAPI/Python service responsible for personalized VIDEO and LIVE recommendations.

The current design is local-state-first and event-driven: platform services remain owners of their domain data, while Recommendation ML Service (RMS) maintains recommendation-relevant local projections. Recommendation serving should not depend on synchronous User/Content/Follow/Interaction service calls.

Current status: the serving/model pipeline is in final verification. Synthetic metrics validate pipeline functionality, not production recommendation quality. Real platform Kafka contracts, a reviewed real content-classification dataset, and final taxonomy approval are still required.

Architecture

User Service ------------------+
Content Service ---------------+
Follow Service ----------------+--> Kafka --> Recommendation ML Service
Interaction/Event services ----+ |
LIVE/Streaming events ---------+ v
Local projections
/ \
 VIDEO LIVE
candidates candidates
| |
34 features 17 features
| |
VIDEO model LIVE model
| |
reranking reranking
\ /
Top-N

Recommendation flow:

Existing platform state is bootstrapped into RMS.

Subsequent platform changes are synchronized through Kafka.

RMS resolves onboarding and behavioral context for the user.

Local active content is retrieved as candidates.

VIDEO or LIVE features are constructed.

The corresponding persisted model scores candidates.

Domain-specific reranking applies freshness, diversity, seen-content and other business rules.

RMS returns Top-N recommendations.

VIDEO and LIVE share semantic/user/creator foundations but use separate ranking pipelines because their behavioral signals and lifecycle differ.

Platform integration

Initial bootstrap / backfill

Kafka only describes changes that RMS consumes after integration begins. User Service, Content Service and Follow Service may already contain data before RMS is deployed.

RMS therefore needs an initial snapshot/backfill:

Existing User Service data --> RMS user projection
Existing Content Service data --> RMS content projection
Existing Follow Service data --> RMS follow projection

The preferred mechanism is a paginated internal endpoint, controlled export, or equivalent snapshot. Records should expose stable timestamps/version information so stale backfill data cannot overwrite newer events.

Continuous synchronization

After bootstrap:

USER_CREATED / USER_UPDATED
CONTENT_CREATED / CONTENT_UPDATED / CONTENT_STATUS_CHANGED
FOLLOW / UNFOLLOW
VIDEO interactions
LIVE lifecycle/interactions
|
v
Kafka
|
v
RMS
|
v
local projections

The recommendation hot path should use these local projections rather than synchronous sibling-service requests.

Contracts required from other services

User Service

bootstrap/snapshot mechanism for existing users;

stable userId;

recommendation-relevant onboarding/context fields such as interests, language and region where approved;

user create/update event contract.

Content Service

bootstrap/snapshot mechanism for existing active content;

contentId, creatorId, content type, title/caption, hashtags, lifecycle/status and timestamps;

content create/update/status event contracts.

Interaction/Event Tracking

real Kafka topic names;

envelope/serialization/schema version;

VIDEO interaction types and watch/completion/duration signals where available.

LIVE/Streaming

stream start/update/end contracts;

JOIN/LEAVE/WATCHED/heartbeat and relevant engagement events;

stream/creator identifiers and available semantic metadata.

Kafka/platform

brokers;

topics;

serialization/schema registry conventions;

message keys/partitioning;

consumer-group conventions;

authentication.

Follow Service is maintained in the same backend workstream and its backfill/event contract can be implemented directly.

Data ownership

RMS stores only recommendation-relevant projections; it does not need copies of entire sibling databases.

User projection

Relevant data can include:

userId

onboarding interests

language

region

approved recommendation context

locally derived behavioral state

Persisted onboarding interests can drive VIDEO preferred-category retrieval when no behavior-derived preferred categories exist. Once real behavioral category evidence exists, behavior takes precedence for this retrieval path.

Content projection

Content Service owns raw content facts. RMS locally stores the facts required for retrieval and semantic processing, then derives recommendation semantics itself.

Typical input:

contentId

creatorId

contentType (VIDEO / LIVE)

title/caption

hashtags

status

createdAt / updatedAt

Follow projection

RMS needs local user -> creator relationships for personalized retrieval, particularly LIVE's FOLLOWED_CREATOR bucket.

Interaction state

Behavior events drive category/creator affinity, seen-content state, negative feedback, completion behavior, interaction-derived popularity and LIVE dynamic state.

VIDEO pipeline

userId
|
v
onboarding + local behavior
|
v
local active VIDEO candidates
|
v
34-feature construction
|
v
VIDEO model
|
v
VIDEO reranking
|
v
Top-N

Important behavior:

hard-seen VIDEO content is excluded during local candidate generation;

fresh/new content has dedicated retrieval behavior;

onboarding interests can bootstrap preferred-category retrieval;

behavior evidence supersedes onboarding-category fallback;

negative feedback and replay/seen handling affect recommendation behavior;

popularity is interaction-derived rather than trusted from caller input;

recommendation responses can expose localBucketSource to preserve local retrieval provenance separately from external candidateSource;

search/session intent can influence VIDEO ranking.

The current VIDEO model contract contains 34 features.

LIVE pipeline

userId
|
v
local user/creator/LIVE state
|
v
LIVE candidate generation
|
v
17-feature construction
|
v
LIVE model
|
v
LIVE reranking
|
v
Top-N

Personalized LIVE retrieval includes:

FOLLOWED_CREATOR

RECENT_CREATOR_INTERACTION

PREFERRED_CATEGORY

exploration/fallback

LIVE dynamic state can use active viewers, viewer growth, likes, gifts, stream age and prior LIVE evidence.

VIDEO history can bootstrap LIVE affinities when direct LIVE evidence is absent. Real LIVE evidence takes precedence once available.

A LIVE item without a category can use sufficiently established creator history as a fallback; otherwise semantic category remains UNKNOWN.

The current LIVE model contract contains 17 features.

Content understanding

RMS contains a classifier for deriving semantic/category information from content text.

Intended flow:

Content event/backfill
|
v
title/caption/hashtags
|
v
content understanding
|
v
RMS semantic fields
|
v
local Content projection
|
v
candidate generation/ranking

Classification belongs at ingestion/update time, not on every recommendation request.

A real-data training bridge exists, but the currently available classifier evidence is still based on a small deterministic bootstrap/synthetic dataset. Therefore classifier real-world quality is LIMITED until reviewed production data is available.

Taxonomy

The established VIDEO model vocabulary currently includes:

FOOD
SPORT
MUSIC
TECH
GAMING
TRAVEL
COMEDY
NEWS
FASHION
FITNESS

LIVE/demo data has also used domain-specific values such as CHAT. UNKNOWN is supported as a semantic fallback.

The final shared platform taxonomy is not yet approved. Do not treat the current list as a final cross-service contract.

API

Canonical base path:

/api/v1/recommendation-ml-service

Local/default port:

3500

Swagger:

http://localhost:3500/docs

Health:

GET /api/v1/recommendation-ml-service/health

Users

POST /api/v1/recommendation-ml-service/users
GET /api/v1/recommendation-ml-service/users/{userId}
GET /api/v1/recommendation-ml-service/users/{userId}/behaviour-profile
GET /api/v1/recommendation-ml-service/users/{userId}/profile
POST /api/v1/recommendation-ml-service/users/{userId}/search-intent
GET /api/v1/recommendation-ml-service/users/{userId}/search-intent

Content

POST /api/v1/recommendation-ml-service/contents
GET /api/v1/recommendation-ml-service/contents/{contentId}
PATCH /api/v1/recommendation-ml-service/contents/{contentId}

Events

POST /api/v1/recommendation-ml-service/events
POST /api/v1/recommendation-ml-service/interactions

Candidate generation

POST /api/v1/recommendation-ml-service/candidates/generate
POST /api/v1/recommendation-ml-service/candidates/generate/live

Recommendations

POST /api/v1/recommendation-ml-service/recommendations
POST /api/v1/recommendation-ml-service/recommendations/live

Model lifecycle

VIDEO and LIVE expose asynchronous training, job status, model status, versions, metrics and rollback operations under:

/api/v1/recommendation-ml-service/model

Recommendation lifecycle

Responses expose recommendation strategy:

COLD_START
HYBRID
PERSONALISED_ML

and interactionCount.

VIDEO onboarding provides initial preference context before sufficient behavior exists. LIVE can use VIDEO-derived bootstrap evidence before LIVE-specific evidence becomes authoritative.

Model architecture

VIDEO

34 features

persisted sklearn pipeline

point-in-time feature construction

chronological split lifecycle

model selection

probability calibration

threshold tuning

held-out evaluation

business reranking

checksum/artifact validation

active/previous/candidate artifact lifecycle

LIVE

17 features

LIVE-specific target/behavior semantics

separate candidate generation and reranking

persisted model lifecycle

synthetic training support for functional verification

A model score is the model probability before business reranking. The API score is the adjusted score after reranking. Recommendation reason is heuristic and must not be presented as SHAP/per-request model attribution.

Model-quality status

Synthetic data is useful for proving that training, persistence, promotion/rollback, inference and ranking execute correctly. It is not production preference evidence.

Current limitations:

content-classifier real-world quality is limited;

no reviewed production classifier dataset is available yet;

LIVE model evidence is synthetic;

final platform taxonomy is unapproved;

production Kafka contracts/integration remain pending.

Serving/model pipeline readiness and model-quality readiness are separate concerns.

Database and migrations

Primary database:

PostgreSQL

Schema changes are managed through Alembic.

alembic current
alembic upgrade head
alembic history

Do not hard-code a migration revision in integration documentation; verify the current repository/runtime head.

Configuration

Use .env.example as the shareable configuration reference.

Never share the real .env file if it contains credentials/secrets.

Important configuration areas include:

APP_ENV
DATABASE_URL
INTERNAL_API_KEY
MODEL_DIR / MODEL_ARTIFACT_ROOT
Kafka settings
content-understanding settings
VIDEO/LIVE thresholds
history/candidate limits

Inbound X-Internal-API-Key protection has been removed from every route on this service.

Ordinary inbound JWT/JWKS authentication is not implemented yet.

INTERNAL_API_KEY may still be configured for this service's own OUTBOUND calls to legacy
service clients (User Behavior Service / Candidate Service) -- it no longer gates any
inbound route.

GET /health remains public for infrastructure health checks.

Local setup

Requires Python 3.12+.

python -m venv .venv

Windows:

.venv\Scripts\activate

macOS/Linux:

source .venv/bin/activate

Install dependencies and migrate:

pip install -r requirements.txt
alembic upgrade head

Run:

uvicorn app.main:app --host 0.0.0.0 --port 3500 --reload

Docker

docker compose build
docker compose up -d
docker compose ps
docker compose logs -f app

Stop while preserving data:

docker compose down

Do not use docker compose down -v unless intentionally deleting volumes.

The application deployment path applies Alembic migrations before serving and should fail when migrations cannot complete.

Kubernetes / Jenkins

The repository contains separate deployment configuration for development and G2R EKS:

Jenkinsfile
Jenkins.g2r-eks
deploy/values-dev.yaml
deploy/values-g2r-eks.yaml
deploy/datastores/postgresql-values-dev.yaml
deploy/datastores/postgresql-values-g2r-eks.yaml

The deployment pipeline performs validation/CI, image build and push, migrations, and Helm deployment.

Required secrets/infrastructure must exist in the target environment.

Model artifact persistence must be provisioned/verified independently from PostgreSQL persistence; local container filesystem must not be assumed durable across Kubernetes rollouts.

Authentication and error contract

Inbound requests are not currently authenticated: the previous X-Internal-API-Key gate has
been removed, and JWT/JWKS authentication is not yet implemented. INTERNAL_API_KEY, if
configured, is used only for this service's own outbound calls to legacy service clients.

Responses include X-Request-ID.

Canonical error shape:

{
"error": "ERROR_CODE",
"message": "Safe error message",
"errorDetails": {
"code": "ERROR_CODE",
"message": "Safe error message",
"details": {}
},
"requestId": "..."
}

Unexpected exceptions are logged server-side rather than returned as raw internal details.

Verification

Full tests:

python -m pytest

Lint:

python -m ruff check app tests scripts

Types:

python -m mypy app scripts

Diff sanity:

git diff --check

Historical test totals are intentionally not hard-coded here. The current CI/full-suite output is authoritative.

Integration checklist

Before real platform integration is considered complete:

Final full suite: 0 failures / 0 errors.

User Service historical bootstrap contract confirmed.

User Service Kafka update contract confirmed.

Content Service historical bootstrap contract confirmed.

Content Service Kafka lifecycle contract confirmed.

Follow Service historical backfill contract implemented/confirmed.

Follow/unfollow Kafka synchronization confirmed.

Interaction/Event Tracking contract confirmed.

LIVE event contract confirmed.

Kafka serialization/topic/auth configuration confirmed.

Final taxonomy approved.

Reviewed real classifier dataset available.

Deployment model-artifact persistence confirmed.

Production-data model evaluation completed before quality claims.

Summary for integrating services

Other services remain owners of their data. RMS needs two integration mechanisms:

1. INITIAL BACKFILL
   Existing recommendation-relevant state -> RMS local projections

2. CONTINUOUS SYNC
   New changes -> Kafka -> RMS local projections

Recommendation serving then stays local:

RMS local state
|
v
candidate generation
|
+--> VIDEO: 34 features --> VIDEO model --> reranking
|
+--> LIVE: 17 features --> LIVE model --> reranking
|
v
Top-N recommendations

This avoids adding sibling-service request latency and availability dependencies to every recommendation request.
