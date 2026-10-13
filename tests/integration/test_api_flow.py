"""End-to-end through HTTP: request -> FastAPI -> LangGraph -> models, and the
telemetry that comes out of it (one trace, server span as root, metrics)."""

from fastapi.testclient import TestClient

from obslab.api import create_app

from conftest import metrics_by_name


def test_ask_produces_one_trace_rooted_at_http_span(components, otel):
    _, spans, reader = otel
    client = TestClient(create_app(components))
    r = client.post("/ask", json={"question": "What does cosine similarity measure?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "answered" and body["trace_id"]

    finished = spans.get_finished_spans()
    assert {format(s.context.trace_id, "032x") for s in finished} == {body["trace_id"]}
    server = next(s for s in finished if s.name == "POST /ask")
    workflow = next(s for s in finished if s.name == "invoke_workflow rag")
    assert server.parent is None
    # the workflow hangs under the HTTP server span (possibly via ASGI internal spans)
    by_id = {s.context.span_id: s for s in finished}
    p = workflow.parent
    while p is not None and p.span_id != server.context.span_id:
        p = by_id[p.span_id].parent
    assert p is not None

    got = metrics_by_name(reader)
    assert "http.server.request.duration" in got or "http.server.duration" in got
    assert "gen_ai.client.operation.duration" in got


def test_an_announced_timeout_reaches_the_request_span(components, otel):
    _, spans, _ = otel
    client = TestClient(create_app(components))
    r = client.post("/ask", json={"question": "What does cosine similarity measure?"},
                    headers={"X-Request-Timeout": "20"})
    assert r.status_code == 200, r.text
    server = next(s for s in spans.get_finished_spans() if s.name == "POST /ask")
    assert server.attributes["rag.request.deadline_s"] == 20


def test_validation_and_provider_errors(components, otel):
    _, spans, _ = otel
    client = TestClient(create_app(components))
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    r = client.post("/ask", json={"question": "cosine similarity FAIL vectors"})
    assert r.status_code == 502
    workflow = next(s for s in spans.get_finished_spans() if s.name == "invoke_workflow rag")
    assert workflow.status.status_code.name == "ERROR"
    assert client.get("/healthz").json()["ok"] is True
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["chunks"] == 3


def test_not_ready_when_the_index_was_built_with_another_model(components):
    components.status = lambda: {
        "ready": False, "store": "pgvector", "provider": "ollama", "chunks": 3,
        "error": "index built with embed model 'fake-embed', but 'nomic-embed-text' is configured"}
    client = TestClient(create_app(components))
    r = client.get("/readyz")
    assert r.status_code == 503
    assert "nomic-embed-text" in r.json()["error"]
    assert client.get("/healthz").status_code == 200       # alive, just not serving



def test_fastapi_does_not_add_a_second_exporter(components, otel, monkeypatch):
    # FastAPI >= 0.142 adds its own OTLP exporter at startup when this variable is set (it is,
    # in the cluster). The app configures export itself, so there must still be one processor.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    tel, _, _ = otel
    before = len(tel.tracer_provider._active_span_processor._span_processors)
    with TestClient(create_app(components)):            # runs the lifespan startup
        pass
    assert len(tel.tracer_provider._active_span_processor._span_processors) == before


def test_a_conversation_through_the_api(components):
    client = TestClient(create_app(components))
    first = client.post("/ask", json={"question": "How do you back up etcd?"}).json()
    assert first["turn"] == 1 and first["conversation_id"]

    second = client.post("/ask", json={"question": "How often should that be done?",
                                       "conversation_id": first["conversation_id"]})
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["conversation_id"] == first["conversation_id"] and body["turn"] == 2
    assert body["route"] == "answered" and body["sources"] == ["etcd.md"]
    assert "etcd" in body["standalone_question"]


def test_conversation_guardrails_through_the_api(settings, otel, store):
    import dataclasses

    from obslab.app import build_components
    client = TestClient(create_app(build_components(dataclasses.replace(settings, max_turns=1), otel[0], store=store)))
    # only a UUID is accepted: the value becomes a database key and a span attribute
    for bad in ("42", "../../etc/passwd", "x" * 5000, "'; DROP TABLE checkpoints; --"):
        assert client.post("/ask", json={"question": "How do you back up etcd?",
                                         "conversation_id": bad}).status_code == 422
    first = client.post("/ask", json={"question": "How do you back up etcd?"}).json()
    full = client.post("/ask", json={"question": "How often should that be done?",
                                     "conversation_id": first["conversation_id"]})
    assert full.status_code == 409 and "new conversation" in full.json()["detail"]
