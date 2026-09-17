"""Optional Kafka `user.registered` consumer adapter (User Service -> Kafka -> this service;
Task: continuous user-projection ingestion). Guarded entirely behind
KAFKA_USER_REGISTERED_ENABLED (default false) -- local/demo/test operation never touches this
module at all. Mirrors app.services.kafka_behavior_consumer's lazy-import/no-hard-dependency
structure, but deliberately does NOT reuse its enable_auto_commit=True commit policy: see the
module-level commit-semantics note on `_consume_loop` below for why.

Scope: NEW registrations only. Users that already existed in User Service before this
integration is turned on are a separate, not-yet-implemented historical snapshot/backfill
task (see README "Initial bootstrap / backfill") -- this module intentionally does not
attempt that.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from pydantic import ValidationError

from app.core.config import (
    KAFKA_BROKERS,
    KAFKA_USER_REGISTERED_ENABLED,
    KAFKA_USER_REGISTERED_GROUP_ID,
    KAFKA_USER_REGISTERED_TOPIC,
)
from app.core.logging import logger
from app.db.database import SessionLocal
from app.schemas.user_events import UserRegisteredEvent
from app.services.user_registration_service import apply_user_registered

__all__ = ["STATUS", "process_message", "start_consumer_in_background"]

# Same lightweight, in-memory-only status convention as app.services.kafka_behavior_consumer.
# STATUS -- never holds message content, only counters/flags.
STATUS: dict[str, Any] = {
    "enabled": KAFKA_USER_REGISTERED_ENABLED, "connected": False,
    "processed": 0, "ignored": 0, "errors": 0, "lastError": None,
}


def process_message(raw_value: bytes | str) -> bool:
    """Parses and applies one user.registered message. Returns True if the local user/
    onboarding projection was written, False for a malformed/schema-invalid/wrong-type
    message (all PERMANENTLY unprocessable -- safe for the caller to commit past). A genuine
    persistence failure (apply_user_registered itself raising) is deliberately NOT caught
    here -- it propagates to _consume_loop, which must not commit the Kafka offset for it
    (see that function's own docstring): a DB failure must never be silently treated as
    successful processing."""
    db = SessionLocal()
    try:
        try:
            payload = json.loads(raw_value)
        except (TypeError, ValueError) as exc:
            STATUS["errors"] += 1
            STATUS["lastError"] = "MALFORMED_JSON"
            logger.info("user.registered event malformed json error=%s", type(exc).__name__)
            return False
        try:
            event = UserRegisteredEvent.model_validate(payload)
        except ValidationError as exc:
            STATUS["errors"] += 1
            STATUS["lastError"] = "SCHEMA_INVALID"
            logger.info(
                "user.registered event failed schema validation eventId=%s error=%s",
                payload.get("eventId") if isinstance(payload, dict) else None, str(exc)[:200],
            )
            return False
        # UserRegisteredEvent's own `type` validator already rejects anything other than
        # "user.registered" as SCHEMA_INVALID above -- this is unreachable in practice, kept
        # only as an explicit, defensive statement of intent (never silently reprocess a
        # different platform event type as a user registration).
        if event.type != "user.registered":
            STATUS["ignored"] += 1
            logger.info("ignored non-user.registered event type=%s", event.type)
            return False
        apply_user_registered(db, event)
        STATUS["processed"] += 1
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _consume_loop() -> None:
    if not KAFKA_USER_REGISTERED_TOPIC:
        # No real topic name for User Service's user.registered stream was found anywhere in
        # this workspace -- refuse to start against a guessed name, exactly like
        # KAFKA_BEHAVIOR_TOPIC's own equivalent gate.
        STATUS["connected"] = False
        STATUS["lastError"] = "KAFKA_USER_REGISTERED_TOPIC_NOT_CONFIGURED"
        logger.warning(
            "KAFKA_USER_REGISTERED_ENABLED=true but KAFKA_USER_REGISTERED_TOPIC is not set; no "
            "confirmed real topic name exists for this stream, so none is guessed. User "
            "registration consumption stays disabled until a real topic is configured."
        )
        return

    try:
        from kafka import KafkaConsumer  # type: ignore[import-not-found]
    except ImportError:
        STATUS["connected"] = False
        STATUS["lastError"] = "KAFKA_CLIENT_LIBRARY_NOT_INSTALLED"
        logger.warning(
            "KAFKA_USER_REGISTERED_ENABLED=true but no Kafka client library is installed; user "
            "registration consumption stays disabled. Local/demo/test operation is unaffected."
        )
        return

    try:
        # enable_auto_commit=False (deliberately different from
        # app.services.kafka_behavior_consumer's auto-commit): the offset for a message is
        # advanced ONLY after process_message returns without raising (see below) -- a DB
        # persistence failure must never be silently acknowledged as processed. This is a
        # narrower, correctness-motivated policy scoped to this new consumer only; it does not
        # change kafka_behavior_consumer's own existing behavior.
        consumer = KafkaConsumer(
            KAFKA_USER_REGISTERED_TOPIC, bootstrap_servers=KAFKA_BROKERS,
            group_id=KAFKA_USER_REGISTERED_GROUP_ID, enable_auto_commit=False,
        )
    except Exception as exc:  # noqa: BLE001 -- an unreachable broker at startup must not crash
        # the app; this consumer simply never comes up (see STATUS for observability).
        STATUS["connected"] = False
        STATUS["lastError"] = type(exc).__name__
        logger.warning("Kafka user.registered consumer failed to connect: %s: %s", type(exc).__name__, exc)
        return

    STATUS["connected"] = True
    logger.info(
        "Kafka user.registered consumer connected topic=%s groupId=%s",
        KAFKA_USER_REGISTERED_TOPIC, KAFKA_USER_REGISTERED_GROUP_ID,
    )
    try:
        for message in consumer:
            try:
                process_message(message.value)
            except Exception as exc:  # noqa: BLE001 -- a persistence failure must not crash this
                # consumer loop/process, but it must ALSO not advance the offset below: the
                # failed message's offset is never committed. No manual retry/sleep loop is
                # introduced here -- whether/when this message is redelivered depends on the
                # Kafka consumer's own lifecycle (this process's next poll may or may not
                # re-yield it, depending on the client's internal buffering) and on
                # rebalance/restart semantics (a fresh consumer for this group always resumes
                # from the last COMMITTED offset, so a restart/rebalance after a failure is
                # guaranteed to redeliver it). Never a busy loop -- it does not block other
                # partitions/consumers.
                STATUS["errors"] += 1
                STATUS["lastError"] = type(exc).__name__
                logger.exception("user.registered persistence failed -- offset not committed, will retry on redelivery")
                continue
            consumer.commit()
    except Exception as exc:  # noqa: BLE001 -- the consume loop itself must degrade to
        # "disconnected", never take the process down.
        STATUS["connected"] = False
        STATUS["lastError"] = type(exc).__name__
        logger.warning("Kafka user.registered consumer loop stopped: %s: %s", type(exc).__name__, exc)


def start_consumer_in_background() -> threading.Thread | None:
    """No-op when KAFKA_USER_REGISTERED_ENABLED is false (the default): returns None, starts
    nothing. Otherwise starts the consume loop on its own daemon thread (independent of
    app.services.kafka_behavior_consumer's own thread) so it can never block process
    shutdown; any connection failure is caught inside `_consume_loop` and only updates STATUS."""
    if not KAFKA_USER_REGISTERED_ENABLED:
        return None
    thread = threading.Thread(target=_consume_loop, name="kafka-user-registered-consumer", daemon=True)
    thread.start()
    return thread
