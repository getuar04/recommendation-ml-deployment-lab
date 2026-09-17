"""Optional Kafka behavioral-event consumer adapter (Event Tracking Service -> Kafka -> this
service; see README "Boundary with the real platform" and "Real-service contract
verification"), guarded entirely behind KAFKA_BEHAVIOR_ENABLED (default false -- local/demo/
test operation never touches this module at all). Reuses the EXACT same ingestion logic
POST /events already uses (`app.services.event_service.store_event`) for a Kafka-delivered
event -- this module only adapts one Kafka message into the same `EventCreate` schema and
gives it its own DB session; it does not duplicate idempotency/validation/label-derivation
logic.

Real-contract search result: no event-tracking-service repository, OpenAPI/Avro schema, or
topic list exists locally. Two REAL, confirmed (but different, and neither VIDEO-specific)
Kafka event envelope conventions were found in sibling repositories in this workspace:
  - follow-service (src/infra/kafka/consumer.ts): raw Avro (the `avsc` library), envelope
    `{base: {eventId, eventType, userId, timestamp:long, schemaVersion:"v1", region, device,
    sessionId, requestId, appVersion, deviceId}, ...domain fields}`.
  - interaction-count-service (src/infra/kafka/kafkaConsumer.ts): JSON first, falling back to
    Confluent Schema-Registry Avro decode (`@kafkajs/confluent-schema-registry`); dot-separated
    `type` values (`post.viewed`, `post.like.created`, ...) -- no watch-time/completion/
    duration field exists anywhere in its dispatch table.
  - ranking-service (src/infra/types/kafka.types.ts): JSON, CloudEvents-shaped
    `{eventId, type, target, data, occurredAt}`.
None of the three carries a VIDEO watch-percentage/completion/duration signal, and no topic
name for RMS's own domain was found anywhere -- this is a genuine, reported gap, not something
papered over with an invented topic/schema. KAFKA_BEHAVIOR_TOPIC therefore has no default
(see app.core.config) and `start_consumer_in_background` refuses to start without one
explicitly configured. The message format this module parses (`json.loads` into the existing
`EventCreate` schema) is RMS's OWN already-established local/demo event shape (the same one
POST /events already accepts), kept as a testable placeholder until the real owning team
publishes an actual topic/schema for this domain -- it is NOT a claim that any real upstream
topic is JSON-encoded (the confirmed real precedents above are predominantly Avro).

Ownership boundary (per README/task): if a User Behavior Service is the platform's source of
truth for aggregated user behavior, this consumer's job is NOT to rebuild that aggregation --
it only keeps this project's own local demo `interactions` table (the same one `POST /events`
already writes to, used by the LOCAL_DB user-behavior path) in sync with the event stream, via
the identical `store_event` idempotency/validation path. It does not attempt to replace or
duplicate UBS's own state-building.

No Kafka client library is a hard dependency of this project (see requirements.txt) -- it is
imported lazily here, exactly like the Two-Tower/torch integration already does for its own
optional dependency (see app.services.recommendation_service._try_two_tower_candidates). When
KAFKA_BEHAVIOR_ENABLED is false, nothing here is imported or executed by the rest of the app.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from pydantic import ValidationError

from app.core.config import (
    KAFKA_BEHAVIOR_ENABLED,
    KAFKA_BEHAVIOR_GROUP_ID,
    KAFKA_BEHAVIOR_TOPIC,
    KAFKA_BROKERS,
)
from app.core.logging import logger
from app.db.database import SessionLocal
from app.schemas.event_schemas import EventCreate
from app.services.event_service import (
    ContentInactiveError,
    ContentTypeMismatchError,
    store_event,
)

__all__ = ["STATUS", "process_message", "start_consumer_in_background"]

# Lightweight, in-memory status only (no persistence) -- read by app.api.health_routes so
# GET /health can report DISABLED/HEALTHY/UNAVAILABLE without holding a live connection open
# just to answer a health probe. Never holds message content, only counters/flags.
STATUS: dict[str, Any] = {
    "enabled": KAFKA_BEHAVIOR_ENABLED, "connected": False, "processed": 0, "errors": 0, "lastError": None,
}


def process_message(raw_value: bytes | str) -> bool:
    """Parses and stores one Kafka message using the SAME validation/idempotency/
    label-derivation path POST /events already uses (app.services.event_service.store_event).
    Returns True if a new interaction was stored, False for a duplicate/rejected/malformed
    message. Uses its OWN DB session (never shared across messages/threads -- session
    isolation, matching the request-scoped get_db() every HTTP request already gets).
    NEVER raises: one malformed/unsupported/out-of-order/duplicate event must never take down
    the consumer loop or the process -- handled cases:
      - duplicate eventId: store_event's own existing idempotency (returns stored=False)
      - malformed JSON / schema-invalid event / unsupported eventType: caught below, logged,
        skipped (EventType is a closed enum -- an unrecognized value fails EventCreate
        validation the same as any other malformed field)
      - out-of-order timestamp: safe by construction -- every read path this event feeds
        (recent_interactions_for_ranking, interactions()) re-sorts by timestamp at query time,
        it never depends on Kafka delivery order
      - content projection not yet locally known (out-of-order relative to content.created):
        NOT a rejection -- store_event persists the interaction with content_pending=True
        instead (RMS does not own Content; see Interaction.content_pending's own comment,
        app.db.models). Returned as stored=True, same as any other genuinely new interaction.
      - domain/content-type mismatch (a LIVE_* eventType against VIDEO content or vice versa),
        content deactivated: rejected by store_event itself (ContentTypeMismatchError/
        ContentInactiveError) only when the Content row IS locally known, caught below,
        logged, skipped
      - unexpected processing exception: caught, logged, skipped
    """
    db = SessionLocal()
    try:
        try:
            payload = json.loads(raw_value)
        except (TypeError, ValueError) as exc:
            STATUS["errors"] += 1
            STATUS["lastError"] = "MALFORMED_JSON"
            logger.info("kafka behavior event malformed json error=%s", type(exc).__name__)
            return False
        try:
            event = EventCreate.model_validate(payload)
        except ValidationError as exc:
            STATUS["errors"] += 1
            STATUS["lastError"] = "SCHEMA_INVALID"
            logger.info(
                "kafka behavior event failed schema validation eventId=%s error=%s",
                payload.get("eventId") if isinstance(payload, dict) else None, str(exc)[:200],
            )
            return False
        try:
            _row, stored = store_event(db, event)
        except (ContentInactiveError, ContentTypeMismatchError) as exc:
            STATUS["errors"] += 1
            STATUS["lastError"] = type(exc).__name__
            logger.info("kafka behavior event rejected eventId=%s reason=%s", event.event_id, type(exc).__name__)
            return False
        STATUS["processed"] += 1
        if not stored:
            logger.info("kafka behavior event duplicate ignored eventId=%s", event.event_id)
        return stored
    except Exception:  # noqa: BLE001 -- one message's unexpected failure must never crash the
        # consumer loop/process; logged with type/traceback server-side only, never the raw
        # payload (may contain arbitrary caller-supplied fields).
        STATUS["errors"] += 1
        STATUS["lastError"] = "UNEXPECTED_ERROR"
        logger.exception("kafka behavior event processing failed unexpectedly")
        return False
    finally:
        db.close()


def _consume_loop() -> None:
    if not KAFKA_BEHAVIOR_TOPIC:
        # No real topic name was found for this domain anywhere in the local workspace (see
        # module docstring) -- refuse to start against a guessed name. Fails exactly like
        # UBS_BASE_URL/CANDIDATE_SERVICE_BASE_URL being unset: clearly, observably, and
        # without crashing the app (KAFKA_BEHAVIOR_ENABLED alone never blocks startup).
        STATUS["connected"] = False
        STATUS["lastError"] = "KAFKA_BEHAVIOR_TOPIC_NOT_CONFIGURED"
        logger.warning(
            "KAFKA_BEHAVIOR_ENABLED=true but KAFKA_BEHAVIOR_TOPIC is not set; no real topic "
            "name for this domain was found in the local workspace, so none is guessed. "
            "Behavioral event consumption stays disabled until a real topic is configured."
        )
        return

    try:
        from kafka import KafkaConsumer  # type: ignore[import-not-found]
    except ImportError:
        STATUS["connected"] = False
        STATUS["lastError"] = "KAFKA_CLIENT_LIBRARY_NOT_INSTALLED"
        logger.warning(
            "KAFKA_BEHAVIOR_ENABLED=true but no Kafka client library is installed; behavioral "
            "event consumption stays disabled. Local/demo/test operation is unaffected."
        )
        return

    try:
        consumer = KafkaConsumer(
            KAFKA_BEHAVIOR_TOPIC, bootstrap_servers=KAFKA_BROKERS,
            group_id=KAFKA_BEHAVIOR_GROUP_ID, enable_auto_commit=True,
        )
    except Exception as exc:  # noqa: BLE001 -- an unreachable broker at startup must not crash
        # the app; this consumer simply never comes up (see STATUS for observability).
        STATUS["connected"] = False
        STATUS["lastError"] = type(exc).__name__
        logger.warning("Kafka behavior consumer failed to connect: %s: %s", type(exc).__name__, exc)
        return

    STATUS["connected"] = True
    logger.info("Kafka behavior consumer connected topic=%s groupId=%s", KAFKA_BEHAVIOR_TOPIC, KAFKA_BEHAVIOR_GROUP_ID)
    try:
        for message in consumer:
            process_message(message.value)
    except Exception as exc:  # noqa: BLE001 -- the consume loop itself must degrade to
        # "disconnected", never take the process down.
        STATUS["connected"] = False
        STATUS["lastError"] = type(exc).__name__
        logger.warning("Kafka behavior consumer loop stopped: %s: %s", type(exc).__name__, exc)


def start_consumer_in_background() -> threading.Thread | None:
    """No-op when KAFKA_BEHAVIOR_ENABLED is false (the default): returns None, starts nothing
    -- local/demo/test operation is completely unaffected. Otherwise starts the consume loop
    on a daemon thread so it can never block process shutdown; any connection failure is
    caught inside `_consume_loop` and only updates STATUS, never raised here (so a bad/absent
    broker never prevents the app itself from starting)."""
    if not KAFKA_BEHAVIOR_ENABLED:
        return None
    thread = threading.Thread(target=_consume_loop, name="kafka-behavior-consumer", daemon=True)
    thread.start()
    return thread
