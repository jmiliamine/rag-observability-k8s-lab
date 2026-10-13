"""Log events: fixed names, fixed fields, one JSON line, and nothing a log must not hold."""

import ast
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from obslab.api import create_app
from obslab.app import build_components
from obslab.config import Settings
from obslab.telemetry import emit, init_telemetry
from obslab.telemetry.events import EVENT, EVENTS, JsonFormatter

SRC = Path(__file__).resolve().parents[2] / "src" / "obslab"
QUESTION = "What does cosine similarity measure between embedding vectors?"
SAMPLE = {"obslab.provider": "fake", "rag.route": "answered", "rag.rewrites": 1, "rag.groundedness": 0.8,
          "rag.turn": 2, "gen_ai.conversation.id": "0b0e7c0e-7d1a-4a43-9d7b-0d5f3c3f6a11",
          "rag.refusal.reason": "queue_full", "error.type": "ConnectError", "obslab.chunks": 47,
          "obslab.duration_s": 4.2, "obslab.ingest.reason": "no_notes", "obslab.conversations.removed": 3,
          "obslab.conversations.kept": 5, "obslab.conversations.ttl_days": 7.0}


class Capture(logging.Handler):
    """Keeps each event twice: the record, and the JSON line as the stream handler would print it
    (formatted when the event is written, so inside its span)."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.setFormatter(JsonFormatter("obslab-rag", "0.0.0"))
        self.records, self.lines = [], []

    def emit(self, record):
        if hasattr(record, EVENT):
            self.records.append(record)
            self.lines.append(json.loads(self.format(record)))


def events(capture) -> list[logging.LogRecord]:
    return capture.records


def lines(capture) -> list[dict]:
    return capture.lines


@pytest.fixture
def logs(otel):
    capture, app_logger = Capture(), logging.getLogger("obslab")
    app_logger.addHandler(capture)
    yield capture
    app_logger.removeHandler(capture)


@pytest.mark.parametrize("event", sorted(EVENTS))
def test_every_event_is_written_with_its_fields_and_its_level(logs, event):
    level, allowed = EVENTS[event]
    fields = {name: SAMPLE[name] for name in allowed}
    emit(event, fields)
    (record,) = events(logs)
    assert record.levelno == level and record.name == "obslab.events"
    (line,) = lines(logs)
    assert line == line | {"event": event, "service": "obslab-rag", "version": "0.0.0", **fields}
    assert line["level"] == logging.getLevelName(level) and line["time"].endswith("+00:00")


@pytest.mark.parametrize("event, fields, message", [
    ("rag.shrugged", {}, "unknown log event"),
    ("rag.answered", {"question": "how?"}, "has no field"),
    ("rag.refused", {"rag.refusal.reason": "tired"}, "is not one of"),
    ("rag.failed", {"error.type": "x" * 65}, "longer than"),
    ("rag.answered", {"rag.route": ["answered"]}, "must be a string"),
])
def test_an_unknown_event_field_or_value_is_refused(logs, event, fields, message):
    with pytest.raises(ValueError, match=message):
        emit(event, fields)
    assert events(logs) == []


def test_a_line_written_inside_a_span_carries_its_ids(logs, otel):
    tel, _, _ = otel
    with tel.tracer.start_as_current_span("work") as span:
        emit("app.stopping")
    emit("app.stopping")
    inside, outside = lines(logs)
    assert inside["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert inside["span_id"] == format(span.get_span_context().span_id, "016x")
    assert "trace_id" not in outside and "span_id" not in outside


def test_an_answer_is_one_event_without_the_question_or_the_answer(logs, components):
    result = components.ask(QUESTION)
    (line,) = lines(logs)
    assert line == line | {"event": "rag.answered", "level": "INFO", "rag.route": "answered", "rag.rewrites": 0,
                           "rag.turn": 1, "gen_ai.conversation.id": result["conversation_id"]}
    text = json.dumps(line)
    assert QUESTION not in text and result["answer"] not in text
    assert "cosine" not in text.lower()


def test_the_api_reports_start_stop_refusal_and_failure(logs, otel, store):
    tel, _, _ = otel
    components = build_components(Settings(provider="fake", telemetry="none", top_k=3, min_score=0.2,
                                           model_concurrency=1, model_queue=0), tel, store=store)
    with TestClient(create_app(components)) as client:
        with components.rag.admission.slot("generate"):            # the model is busy, nobody may wait
            assert client.post("/ask", json={"question": QUESTION}).status_code == 503
        assert client.post("/ask", json={"question": "cosine similarity FAIL vectors"}).status_code == 502
    got = [(line["event"], line["level"]) for line in lines(logs)]
    assert got == [("app.started", "INFO"), ("rag.refused", "WARNING"), ("rag.failed", "ERROR"),
                   ("app.stopping", "INFO")]
    started, refused, failed, _ = lines(logs)
    assert started["obslab.provider"] == "fake"
    assert refused["rag.refusal.reason"] == "queue_full" and refused["trace_id"]
    assert set(failed) - {"trace_id", "span_id"} == {"time", "level", "event", "service", "version", "error.type"}
    assert all(r.exc_info is None for r in events(logs))           # no traceback, so no exception message


def test_the_stream_copy_is_one_json_object_per_line(capsys):
    app_logger = logging.getLogger("obslab")
    before = list(app_logger.handlers)
    tel = init_telemetry(Settings(provider="fake", telemetry="none", log_stdout=True))
    try:
        emit("app.started", {"obslab.provider": "fake"})
        printed = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
        assert printed and printed[-1] == printed[-1] | {"event": "app.started", "obslab.provider": "fake",
                                                         "service": "obslab-rag", "level": "INFO"}
    finally:
        for handler in app_logger.handlers[:]:
            if handler not in before:
                app_logger.removeHandler(handler)
        tel.shutdown()


def test_nothing_else_in_the_app_logs_or_prints():
    """Events are written through emit(); logging is configured in telemetry/ only; print() is
    for what the command line shows to a person."""
    logging_users, printers = set(), set()
    for path in SRC.rglob("*.py"):
        name = path.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import) and any(a.name == "logging" for a in node.names):
                logging_users.add(name)
            if isinstance(node, ast.ImportFrom) and node.module == "logging":
                logging_users.add(name)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
                printers.add(name)
    assert logging_users == {"telemetry/events.py", "telemetry/setup.py"}
    assert printers == {"cli.py"}
