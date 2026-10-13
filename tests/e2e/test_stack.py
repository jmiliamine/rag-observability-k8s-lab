"""Against a running observability stack: the telemetry of one question must reach Prometheus
(metrics), Tempo (trace) and Loki (logs).

The same test runs against both deployments, only URLs change (env vars set by the task):
  task test:compose                              the whole Compose level, PostgreSQL included
  task up && task test:stack                     against the cluster's platform
The app runs in this process and exports OTLP to OTEL_EXPORTER_OTLP_ENDPOINT (localhost:4318
in both setups). With OBSLAB_PROVIDER=fake (the task default) no model is needed.

Vector store: `task test:compose` indexes the sample notes into the Compose PostgreSQL and sets
OBSLAB_STACK_STORE=pgvector, so the question goes through pgvector and its SQL span must show
up in Tempo. Against the cluster the database is not reachable from the host, so the index is
built in memory from the sample notes.
"""

import dataclasses
import os
import time
import urllib.parse
import uuid
from pathlib import Path

import pytest

from lab_http import request, unavailable, wait_up

pytestmark = pytest.mark.stack

PROM = os.environ.get("PROM_URL", "http://localhost:9090")
TEMPO = os.environ.get("TEMPO_URL", "http://localhost:3200")
LOKI = os.environ.get("LOKI_URL", "http://localhost:3100")
NOTES = Path(__file__).resolve().parents[2] / "datalake"
STORE = os.environ.get("OBSLAB_STACK_STORE", "memory")
# One id per test run: queries only match this run's telemetry, not an earlier one.
RUN_ID = uuid.uuid4().hex[:12]


@pytest.fixture(scope="module")
def asked():
    from langchain_core.vectorstores import InMemoryVectorStore

    from obslab.app import build_components
    from obslab.config import Settings
    from obslab.rag import corpus
    from obslab.rag.providers import embeddings
    from obslab.telemetry import GenAIMetrics, init_telemetry

    settings = Settings()
    needed = [f"{PROM}/-/ready", f"{TEMPO}/ready", f"{LOKI}/ready"]
    if settings.provider == "ollama":
        needed.append(f"{settings.ollama_url}/api/version")
    for url in needed:
        if not wait_up(url):
            unavailable(f"stack not reachable ({url})")

    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"service.instance.id={RUN_ID}"
    settings = dataclasses.replace(settings, telemetry="otlp", environment="stack-test",
                                   min_score=0.2 if settings.provider == "fake" else settings.min_score)
    tel = init_telemetry(settings, set_global=True)
    store = None                # None: the index in PostgreSQL
    if STORE == "memory":
        store = InMemoryVectorStore(embeddings(settings, tel, GenAIMetrics(tel.meter)))
        store.add_documents(corpus.split(corpus.load_folder(NOTES)))
    c = build_components(settings, tel, store=store)
    status = c.status()
    if not status["ready"]:
        unavailable(f"index not ready: {status.get('error')}")
    with tel.tracer.start_as_current_span("stack-test") as span:
        result = c.ask("How do you back up and restore etcd?")
        trace_id = format(span.get_span_context().trace_id, "032x")
        result["follow_up"] = c.ask("How often should that be done?", result["conversation_id"])
    c.close()
    tel.shutdown()          # flushes spans, metrics and logs
    return result, trace_id


def _poll(fn, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            value = fn()
        except Exception:
            value = None
        if value:
            return value
        time.sleep(3)
    return None


def test_question_is_answered(asked):
    result, _ = asked
    assert result["route"] == "answered", result
    assert any("etcd" in s for s in result["sources"])


def test_a_follow_up_continues_the_conversation(asked):
    follow_up = asked[0]["follow_up"]
    assert follow_up["turn"] == 2 and follow_up["conversation_id"] == asked[0]["conversation_id"]
    if os.environ.get("OBSLAB_PROVIDER", "ollama") == "fake":      # deterministic with the fake model
        assert follow_up["route"] == "answered" and any("etcd" in s for s in follow_up["sources"])


def test_metrics_reach_prometheus(asked):
    # The collector's Prometheus exporter maps service.instance.id to the `instance` label.
    q = urllib.parse.quote('sum(gen_ai_client_operation_duration_seconds_count'
                           f'{{gen_ai_operation_name="chat",instance="{RUN_ID}"}})')
    res = _poll(lambda: request(f"{PROM}/api/v1/query?query={q}")["data"]["result"])
    assert res and float(res[0]["value"][1]) >= 1, f"no chat metric for service.instance.id={RUN_ID}"


def test_admission_wait_reaches_prometheus(asked):
    q = urllib.parse.quote(f'sum(rag_admission_wait_seconds_count{{instance="{RUN_ID}"}})')
    res = _poll(lambda: request(f"{PROM}/api/v1/query?query={q}")["data"]["result"])
    assert res and float(res[0]["value"][1]) >= 1, "no admission wait recorded for the model calls of this run"


def test_trace_reaches_tempo(asked):
    _, trace_id = asked

    def complete_trace():
        # Tempo serves a trace as soon as its first spans land; batches keep arriving for a
        # few seconds, so wait for the full tree rather than the first fragment.
        text = str(request(f"{TEMPO}/api/traces/{trace_id}"))
        return text if "invoke_workflow rag" in text and "chat " in text else None

    text = _poll(complete_trace)
    assert text is not None, f"complete trace {trace_id} not found in Tempo"
    assert asked[0]["conversation_id"] in text, "gen_ai.conversation.id is missing from the workflow span"
    assert "rag.node condense" in text
    assert "rag.admission.wait_s" in text, "the wait for a model slot is missing from the node spans"
    if STORE == "pgvector":
        assert "SELECT rag.chunks" in text, "the vector query span is missing from the trace"


def test_logs_reach_loki_with_the_trace_id(asked):
    _, trace_id = asked
    q = urllib.parse.quote(f'{{service_name="obslab-rag"}} | trace_id="{trace_id}"')
    start = int((time.time() - 900) * 1e9)
    res = _poll(lambda: request(f"{LOKI}/loki/api/v1/query_range?query={q}&start={start}&limit=5")["data"]["result"])
    assert res, f"no log line carrying trace_id={trace_id} in Loki"
    # The same lines, as events: a name and its fields, usable as filters (telemetry/events.py).
    q = urllib.parse.quote(f'{{service_name="obslab-rag"}} |= "rag.answered" | trace_id="{trace_id}"'
                           ' | rag_route="answered"')
    res = request(f"{LOKI}/loki/api/v1/query_range?query={q}&start={start}&limit=5")["data"]["result"]
    assert res, "the log line of the answer is not the rag.answered event, or its fields are missing"
