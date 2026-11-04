"""Signals of the model call itself: the pace of a streamed answer, the oldest question in flight,
and a model server that stops sending."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from obslab.api import create_app
from obslab.app import build_components
from obslab.config import Settings
from obslab.rag.providers import FakeChat, chat_model

from conftest import metrics_by_name, spans_by_name

QUESTION = "What does cosine similarity measure between embedding vectors?"


class PacedChat(FakeChat):
    """Streams `chunks` chunks, `chunk_s` seconds apart, like a model writing an answer."""
    chunks: int = 10
    chunk_s: float = 0.02

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for i in range(self.chunks):
            time.sleep(self.chunk_s)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=f"word{i} "))
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


def _components(otel, store, **settings):
    tel, spans, reader = otel
    c = build_components(Settings(provider="fake", telemetry="none", top_k=3, min_score=0.2, **settings), tel,
                         store=store)
    return c, spans, reader


def test_the_pace_of_an_answer_is_measured_between_chunks(otel, store):
    components, spans, reader = _components(otel, store)
    components.rag.llm = PacedChat(chunks=10, chunk_s=0.02)
    components.ask(QUESTION)

    (point,) = metrics_by_name(reader)["gen_ai.client.operation.time_per_output_chunk"]
    assert point.count == 9                                   # one gap per chunk after the first
    assert 0.015 < point.sum / point.count < 0.1
    assert dict(point.attributes)["gen_ai.operation.name"] == "chat"
    mean = spans_by_name(spans)["chat fake-chat"].attributes["obslab.time_per_output_chunk_s"]
    assert 0.015 < mean < 0.1


def test_a_single_chunk_answer_records_no_pace(otel, store):
    components, spans, reader = _components(otel, store)
    components.rag.llm = PacedChat(chunks=1, chunk_s=0.0)
    components.ask(QUESTION)
    assert "gen_ai.client.operation.time_per_output_chunk" not in metrics_by_name(reader)
    assert "obslab.time_per_output_chunk_s" not in spans_by_name(spans)["chat fake-chat"].attributes


def _oldest_age(reader) -> float:
    (point,) = metrics_by_name(reader)["rag.request.oldest.age"]
    return point.value


def test_the_oldest_question_in_flight_is_reported_and_back_to_zero_when_idle(otel, store):
    components, _, reader = _components(otel, store)
    assert _oldest_age(reader) == 0.0                         # idle reports 0, not nothing
    components.rag.llm = PacedChat(chunks=20, chunk_s=0.05)   # an answer of about one second
    worker = threading.Thread(target=components.ask, args=(QUESTION,))
    worker.start()
    time.sleep(0.5)
    during = _oldest_age(reader)
    worker.join(5)
    assert 0.3 < during < 2
    assert _oldest_age(reader) == 0.0


def test_a_failed_question_does_not_stay_in_flight(otel, store):
    components, _, reader = _components(otel, store)
    with pytest.raises(RuntimeError):
        components.ask("cosine similarity FAIL vectors")
    assert _oldest_age(reader) == 0.0


class _StallingOllama(BaseHTTPRequestHandler):
    """Sends the first chunk of an answer, then nothing: a model server that stopped."""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        line = {"model": "llama3.2:3b", "created_at": "2026-11-04T00:00:00Z",
                "message": {"role": "assistant", "content": "Cosine"}, "done": False}
        self.wfile.write((json.dumps(line) + "\n").encode())
        self.wfile.flush()
        time.sleep(4)

    def log_message(self, *args):
        pass


@pytest.fixture
def stalling_ollama():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StallingOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_a_model_that_stops_sending_is_cut_at_the_read_timeout(stalling_ollama):
    llm = chat_model(Settings(provider="ollama", telemetry="none", ollama_url=stalling_ollama,
                              model_read_timeout_s=1))
    start = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        for _ in llm.stream([HumanMessage(content="hello")]):
            pass
    assert time.monotonic() - start < 3                      # not the 4 s the server would hold it


def test_the_api_answers_504_and_frees_the_slot_when_the_model_stops(otel, store, stalling_ollama):
    components, _, reader = _components(otel, store, model_read_timeout_s=1, model_concurrency=1, model_queue=0)
    components.rag.llm = chat_model(Settings(provider="ollama", telemetry="none", ollama_url=stalling_ollama,
                                             model_read_timeout_s=1))
    client = TestClient(create_app(components))
    reply = client.post("/ask", json={"question": QUESTION})
    assert reply.status_code == 504 and "stopped responding" in reply.json()["detail"]
    with components.rag.admission.slot("generate") as waited:  # the slot was given back
        assert waited < 0.1
    assert _oldest_age(reader) == 0.0


def test_the_read_timeout_is_validated():
    for bad in (0, 601):
        with pytest.raises(ValueError, match="OBSLAB_MODEL_READ_TIMEOUT_S"):
            Settings(provider="fake", telemetry="none", model_read_timeout_s=bad)
