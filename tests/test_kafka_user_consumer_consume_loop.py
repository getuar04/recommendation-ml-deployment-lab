"""Gap-closure for app.services.kafka_user_consumer._consume_loop/start_consumer_in_background
-- mirrors tests/test_kafka_consume_loop.py's own proven fake-Kafka-module pattern exactly
(kafka-python is genuinely not installed in this environment, confirmed) for the sibling
user.registered consumer. tests/test_kafka_user_registered_consumer.py already fully covers
process_message() itself; this file covers only _consume_loop's own branches: missing topic,
missing client library, broker-connection failure, a successful connect/process/commit cycle,
the offset-not-committed-on-persistence-failure guarantee (this consumer's one behavioral
difference from kafka_behavior_consumer's auto-commit), the loop's own exception handling,
and start_consumer_in_background's enabled path.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest

from app.services import kafka_user_consumer


@pytest.fixture(autouse=True)
def _reset_status():
    before = dict(kafka_user_consumer.STATUS)
    yield
    kafka_user_consumer.STATUS.clear()
    kafka_user_consumer.STATUS.update(before)


def test_consume_loop_reports_missing_topic_configuration(monkeypatch):
    monkeypatch.setattr(kafka_user_consumer, "KAFKA_USER_REGISTERED_TOPIC", None)
    kafka_user_consumer._consume_loop()
    assert kafka_user_consumer.STATUS["connected"] is False
    assert kafka_user_consumer.STATUS["lastError"] == "KAFKA_USER_REGISTERED_TOPIC_NOT_CONFIGURED"


def test_consume_loop_reports_missing_client_library(monkeypatch):
    """kafka-python is confirmed not installed here -- this exercises the REAL ImportError
    path, not a simulated one."""
    monkeypatch.setattr(kafka_user_consumer, "KAFKA_USER_REGISTERED_TOPIC", "user-registered-events")
    assert "kafka" not in sys.modules or not hasattr(sys.modules.get("kafka"), "KafkaConsumer")
    kafka_user_consumer._consume_loop()
    assert kafka_user_consumer.STATUS["connected"] is False
    assert kafka_user_consumer.STATUS["lastError"] == "KAFKA_CLIENT_LIBRARY_NOT_INSTALLED"


def _install_fake_kafka_module(monkeypatch, *, consumer_factory):
    fake_module = types.ModuleType("kafka")
    fake_module.KafkaConsumer = consumer_factory
    monkeypatch.setitem(sys.modules, "kafka", fake_module)


def test_consume_loop_reports_broker_connection_failure(monkeypatch):
    monkeypatch.setattr(kafka_user_consumer, "KAFKA_USER_REGISTERED_TOPIC", "user-registered-events")

    def _raising_constructor(*args, **kwargs):
        raise ConnectionError("no route to broker")

    _install_fake_kafka_module(monkeypatch, consumer_factory=_raising_constructor)
    kafka_user_consumer._consume_loop()
    assert kafka_user_consumer.STATUS["connected"] is False
    assert kafka_user_consumer.STATUS["lastError"] == "ConnectionError"


class _FakeMessage:
    def __init__(self, value):
        self.value = value


def test_consume_loop_connects_processes_and_commits_each_message(monkeypatch):
    monkeypatch.setattr(kafka_user_consumer, "KAFKA_USER_REGISTERED_TOPIC", "user-registered-events")
    processed: list[bytes] = []
    monkeypatch.setattr(kafka_user_consumer, "process_message", lambda value: processed.append(value) or True)

    instances = []

    class _FakeConsumer:
        def __init__(self, topic, *, bootstrap_servers, group_id, enable_auto_commit):
            self.topic = topic
            self.enable_auto_commit = enable_auto_commit
            self.commit_calls = 0
            instances.append(self)

        def __iter__(self):
            return iter([_FakeMessage(b'{"eventId": "e1"}'), _FakeMessage(b'{"eventId": "e2"}')])

        def commit(self):
            self.commit_calls += 1

    _install_fake_kafka_module(monkeypatch, consumer_factory=_FakeConsumer)
    kafka_user_consumer._consume_loop()
    assert kafka_user_consumer.STATUS["connected"] is True
    assert processed == [b'{"eventId": "e1"}', b'{"eventId": "e2"}']
    assert instances[0].commit_calls == 2  # one commit per successfully processed message
    # enable_auto_commit must be explicitly False for this consumer (the one deliberate
    # behavioral difference from kafka_behavior_consumer.py's own auto-commit).
    assert instances[0].enable_auto_commit is False


def test_consume_loop_does_not_commit_offset_when_persistence_fails(monkeypatch):
    """The core STEP 6 guarantee: a message whose processing raises (simulating a DB
    persistence failure) must never have its offset committed -- proven directly by asserting
    commit() was never called for it, not merely by inspecting STATUS."""
    monkeypatch.setattr(kafka_user_consumer, "KAFKA_USER_REGISTERED_TOPIC", "user-registered-events")

    def _boom(value):
        raise RuntimeError("simulated database outage")

    monkeypatch.setattr(kafka_user_consumer, "process_message", _boom)
    instances = []

    class _FakeConsumer:
        def __init__(self, topic, *, bootstrap_servers, group_id, enable_auto_commit):
            self.commit_calls = 0
            instances.append(self)

        def __iter__(self):
            return iter([_FakeMessage(b'{"eventId": "e-fails"}')])

        def commit(self):
            self.commit_calls += 1

    _install_fake_kafka_module(monkeypatch, consumer_factory=_FakeConsumer)
    kafka_user_consumer._consume_loop()  # must not raise -- the per-message failure is caught
    assert instances[0].commit_calls == 0
    assert kafka_user_consumer.STATUS["errors"] == 1
    assert kafka_user_consumer.STATUS["lastError"] == "RuntimeError"
    # The loop itself keeps running (this consumer's connection is not torn down by one
    # message's persistence failure) -- distinct from a broken __iter__/connection, which does
    # degrade STATUS["connected"] (see the next test).
    assert kafka_user_consumer.STATUS["connected"] is True


def test_consume_loop_degrades_to_disconnected_when_iteration_fails(monkeypatch):
    monkeypatch.setattr(kafka_user_consumer, "KAFKA_USER_REGISTERED_TOPIC", "user-registered-events")

    class _BrokenConsumer:
        def __init__(self, *args, **kwargs):
            pass

        def __iter__(self):
            raise RuntimeError("connection dropped mid-stream")

    _install_fake_kafka_module(monkeypatch, consumer_factory=_BrokenConsumer)
    kafka_user_consumer._consume_loop()  # must not raise -- degrades to STATUS only
    assert kafka_user_consumer.STATUS["connected"] is False
    assert kafka_user_consumer.STATUS["lastError"] == "RuntimeError"


def test_start_consumer_in_background_spawns_a_daemon_thread_when_enabled(monkeypatch):
    monkeypatch.setattr(kafka_user_consumer, "KAFKA_USER_REGISTERED_ENABLED", True)
    started = threading.Event()
    monkeypatch.setattr(kafka_user_consumer, "_consume_loop", started.set)

    thread = kafka_user_consumer.start_consumer_in_background()

    assert isinstance(thread, threading.Thread)
    assert thread.daemon is True
    assert started.wait(timeout=2), "background thread never invoked _consume_loop"
    thread.join(timeout=2)
