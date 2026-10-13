"""The admission gate in front of the model: a short, explicit waiting line instead of a hidden one."""

import threading
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from obslab.api import create_app
from obslab.app import build_components
from obslab.config import Settings
from obslab.rag.admission import Admission, DeadlineExceeded, Overloaded
from obslab.rag.providers import FakeChat
from obslab.telemetry import RagMetrics

from conftest import metrics_by_name, spans_by_name

QUESTION = "What does cosine similarity measure between embedding vectors?"


def _in_threads(n, fn):
    """Run fn(i) in n threads; returns the results in order, exceptions included."""
    out = [None] * n

    def run(i):
        try:
            out[i] = fn(i)
        except Exception as e:
            out[i] = e
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
        time.sleep(0.02)                        # keep the arrival order
    for t in threads:
        t.join(10)
    return out


def test_a_full_line_refuses_at_once_and_a_free_slot_is_reused(otel):
    tel, _, reader = otel
    gate = Admission(concurrency=1, queue=1, metrics=RagMetrics(tel.meter))
    release = threading.Event()

    def call(i):
        with gate.slot("generate"):
            release.wait(5)
        return "served"

    def third(i):
        if i < 2:
            return call(i)
        start = time.monotonic()                # one running, one waiting: no room
        with pytest.raises(Overloaded):
            with gate.slot("generate"):
                pass
        release.set()
        return time.monotonic() - start

    served_1, served_2, refused_in = _in_threads(3, third)
    assert (served_1, served_2) == ("served", "served")
    assert refused_in < 0.1                     # refused without waiting
    with gate.slot("generate") as waited:       # the line is empty again
        assert waited < 0.1
    got = metrics_by_name(reader)
    assert [(dict(p.attributes), p.value) for p in got["rag.admission.rejections"]] == [
        ({"rag.admission.reason": "queue_full"}, 1)]
    assert got["rag.admission.waiting"][0].value == 0          # nobody left waiting
    assert got["rag.admission.wait"][0].count == 3             # the three calls that were admitted


def test_a_call_still_waiting_at_its_deadline_never_gets_the_slot(otel):
    tel, _, reader = otel
    gate = Admission(concurrency=1, queue=4, metrics=RagMetrics(tel.meter))
    release, entered = threading.Event(), []

    def call(i):
        if i == 0:
            with gate.slot("generate"):
                release.wait(5)
            return "served"
        try:
            with gate.slot("generate", deadline=time.monotonic() + 0.15):
                entered.append(i)               # must not happen
        finally:
            release.set()

    first, late = _in_threads(2, call)
    assert first == "served"
    assert isinstance(late, DeadlineExceeded) and entered == []
    got = metrics_by_name(reader)
    assert [(dict(p.attributes), p.value) for p in got["rag.admission.rejections"]] == [
        ({"rag.admission.reason": "deadline"}, 1)]


class SlowChat(FakeChat):
    """Answers in chunks, slowly: stands in for a busy local model."""
    chunk_s: float = 0.05
    started: list = []
    chunks_sent: list = []

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.started.append(time.monotonic())
        for i in range(40):
            time.sleep(self.chunk_s)
            self.chunks_sent.append(i)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=f"word{i} "))
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


def _app(otel, store, **settings):
    tel, spans, reader = otel
    components = build_components(Settings(provider="fake", telemetry="none", top_k=3, min_score=0.2, **settings),
                                  tel, store=store)
    return components, spans, reader


def test_a_burst_is_bounded_and_the_wait_is_measured_apart_from_the_model(otel, store):
    components, spans, reader = _app(otel, store, model_concurrency=1, model_queue=1)
    components.rag.llm = SlowChat(chunk_s=0.005, started=[], chunks_sent=[])
    client = TestClient(create_app(components))

    replies = _in_threads(4, lambda i: client.post("/ask", json={"question": QUESTION}))
    codes = sorted(r.status_code for r in replies)
    assert codes == [200, 200, 503, 503]                       # one running, one waiting, two refused
    refused = next(r for r in replies if r.status_code == 503)
    assert refused.headers["retry-after"] == "5" and "queue is full" in refused.json()["detail"]
    assert len(components.rag.llm.started) == 2                # the refused ones never reached the model

    waits = sorted(s.attributes["rag.admission.wait_s"] for s in spans.get_finished_spans()
                   if s.name == "rag.node generate" and "rag.admission.wait_s" in s.attributes)
    assert len(waits) == 2 and waits[0] < 0.05 and waits[1] > 0.1     # the second one queued behind the first
    got = metrics_by_name(reader)
    assert {p.attributes["rag.admission.reason"]: p.value for p in got["rag.admission.rejections"]} == {"queue_full": 2}
    assert sum(p.count for p in got["rag.admission.wait"]) == 2


def test_a_request_waiting_past_its_deadline_never_reaches_the_model(otel, store):
    components, _, reader = _app(otel, store, model_concurrency=1, model_queue=4, request_deadline_s=1)
    components.rag.llm = SlowChat(chunk_s=0.005, started=[], chunks_sent=[])
    client = TestClient(create_app(components))
    busy = threading.Event()

    def hold_the_model():
        with components.rag.admission.slot("generate"):        # another answer is being written
            busy.set()
            time.sleep(2.0)
    holder = threading.Thread(target=hold_the_model)
    holder.start()
    busy.wait(2)
    reply = client.post("/ask", json={"question": QUESTION})
    holder.join(5)

    assert reply.status_code == 504 and "deadline" in reply.json()["detail"]
    assert components.rag.llm.started == []                    # dropped while waiting: no inference at all
    got = metrics_by_name(reader)
    assert {p.attributes["rag.admission.reason"]: p.value for p in got["rag.admission.rejections"]} == {"deadline": 1}
    assert client.post("/ask", json={"question": QUESTION}).status_code == 200      # and the gate is free again


def test_an_answer_still_being_written_at_the_deadline_is_cut(otel, store):
    components, _, reader = _app(otel, store, request_deadline_s=1)
    components.rag.llm = SlowChat(chunk_s=0.1, started=[], chunks_sent=[])       # a full answer takes 4 s
    reply = TestClient(create_app(components)).post("/ask", json={"question": QUESTION})

    assert reply.status_code == 504
    assert 0 < len(components.rag.llm.chunks_sent) < 40        # the stream was closed, not drained
    got = metrics_by_name(reader)
    assert {p.attributes["rag.admission.reason"]: p.value for p in got["rag.admission.rejections"]} == {"deadline": 1}


def test_a_caller_that_announces_its_timeout_is_not_served_after_it(otel, store):
    components, spans, reader = _app(otel, store, request_deadline_s=30)
    components.rag.llm = SlowChat(chunk_s=0.1, started=[], chunks_sent=[])       # a full answer takes 4 s
    start = time.monotonic()
    reply = TestClient(create_app(components)).post("/ask", json={"question": QUESTION},
                                                    headers={"X-Request-Timeout": "1"})

    assert reply.status_code == 504 and time.monotonic() - start < 3       # cut at 1 s, not at the server's 30
    assert 0 < len(components.rag.llm.chunks_sent) < 40
    got = metrics_by_name(reader)
    assert {p.attributes["rag.admission.reason"]: p.value for p in got["rag.admission.rejections"]} == {"deadline": 1}
    assert spans_by_name(spans)["POST /ask"].attributes["rag.request.deadline_s"] == 1


def test_a_caller_waiting_behind_a_busy_model_is_dropped_at_its_own_timeout(otel, store):
    components, _, _ = _app(otel, store, model_concurrency=1, model_queue=4, request_deadline_s=30)
    components.rag.llm = SlowChat(chunk_s=0.005, started=[], chunks_sent=[])
    client = TestClient(create_app(components))
    busy = threading.Event()

    def hold_the_model():
        with components.rag.admission.slot("generate"):
            busy.set()
            time.sleep(2.0)
    holder = threading.Thread(target=hold_the_model)
    holder.start()
    busy.wait(2)
    reply = client.post("/ask", json={"question": QUESTION}, headers={"X-Request-Timeout": "1"})
    holder.join(5)

    assert reply.status_code == 504
    assert components.rag.llm.started == []                    # the model was never asked for this caller


def test_the_server_deadline_stays_the_upper_bound(otel, store):
    components, spans, _ = _app(otel, store, request_deadline_s=1)
    components.rag.llm = SlowChat(chunk_s=0.1, started=[], chunks_sent=[])
    reply = TestClient(create_app(components)).post("/ask", json={"question": QUESTION},
                                                    headers={"X-Request-Timeout": "600"})
    assert reply.status_code == 504                            # a patient caller does not extend the limit
    assert spans_by_name(spans)["POST /ask"].attributes["rag.request.deadline_s"] == 1


@pytest.mark.parametrize("value", ["soon", "0", "-5", "0.5", "nan", "inf"])
def test_an_invalid_announced_timeout_is_refused_before_any_work(otel, store, value):
    components, _, _ = _app(otel, store)
    components.rag.llm = SlowChat(chunk_s=0.005, started=[], chunks_sent=[])
    reply = TestClient(create_app(components)).post("/ask", json={"question": QUESTION},
                                                    headers={"X-Request-Timeout": value})
    assert reply.status_code == 422 and components.rag.llm.started == []


def test_without_an_announced_timeout_the_server_deadline_applies(otel, store):
    components, spans, _ = _app(otel, store, request_deadline_s=45)
    assert TestClient(create_app(components)).post("/ask", json={"question": QUESTION}).status_code == 200
    assert spans_by_name(spans)["POST /ask"].attributes["rag.request.deadline_s"] == 45


def test_the_command_line_client_announces_its_timeout(monkeypatch):
    import io
    import json
    import urllib.request

    from obslab import cli
    sent = {}

    def fake_urlopen(req, timeout):
        sent.update(headers=dict(req.header_items()), timeout=timeout)
        return io.BytesIO(json.dumps({"answer": "ok"}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cli._post_ask("http://api.test", QUESTION, timeout=20)
    assert sent["timeout"] == 20 and sent["headers"]["X-request-timeout"] == "20"


def test_a_refused_question_leaves_the_conversation_untouched(otel, store):
    components, _, _ = _app(otel, store, model_concurrency=1, model_queue=0)
    first = components.ask(QUESTION)
    gate = components.rag.admission
    with gate.slot("generate"):                                # the model is busy
        with pytest.raises(Overloaded):
            components.ask("And how is it used for ranking?", first["conversation_id"])
    again = components.ask("And how is it used for ranking?", first["conversation_id"])
    assert again["turn"] == 2                                  # the refused attempt did not count


def test_settings_are_validated():
    for bad in ({"model_concurrency": 0}, {"model_queue": -1}, {"request_deadline_s": 0}):
        with pytest.raises(ValueError):
            Settings(provider="fake", telemetry="none", **bad)


def test_the_wait_is_on_the_node_span_of_a_normal_question(components, otel):
    _, spans, _ = otel
    components.ask(QUESTION)
    assert spans_by_name(spans)["rag.node generate"].attributes["rag.admission.wait_s"] < 0.05
