"""Log events: the only place that names them, lists their fields and formats them.

A log line of the lab is an event with a fixed name and a fixed set of fields, not a sentence.
The same event goes to two places:

  OTLP    -> collector -> Loki, with the trace and span ids (telemetry/setup.py wires the handler)
  stdout  -> one JSON object per line, for `kubectl logs`, `docker logs` and a CI log

Rules, enforced by `emit()` and by the tests:

- an event or a field that is not listed here is refused;
- no question or answer text, no connection string, no exception message: a field holds a
  number, a boolean, an id or a value from a short fixed list;
- the level belongs to the event, not to the call site.

Levels: ERROR a request or a job failed; WARNING the service protected itself (refused request);
INFO the normal trail, one line per request or per job.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

EVENT = "event.name"
_logger = logging.getLogger("obslab.events")

# name -> (level, fields allowed on this event)
EVENTS: dict[str, tuple[int, frozenset[str]]] = {
    "app.started": (logging.INFO, frozenset({"obslab.provider"})),
    "app.stopping": (logging.INFO, frozenset()),
    "rag.answered": (logging.INFO, frozenset({"rag.route", "rag.rewrites", "rag.groundedness", "rag.turn",
                                              "gen_ai.conversation.id"})),
    "rag.refused": (logging.WARNING, frozenset({"rag.refusal.reason"})),
    "rag.failed": (logging.ERROR, frozenset({"error.type"})),
    "ingest.finished": (logging.INFO, frozenset({"obslab.chunks", "obslab.duration_s"})),
    "ingest.failed": (logging.ERROR, frozenset({"obslab.ingest.reason"})),
    "conversations.purged": (logging.INFO, frozenset({"obslab.conversations.removed", "obslab.conversations.kept",
                                                      "obslab.conversations.ttl_days"})),
}
# Values a reason may take, so that it stays usable as a filter.
REFUSAL_REASONS = frozenset({"queue_full", "deadline", "conversation_too_long"})
INGEST_REASONS = frozenset({"source_missing", "no_notes"})
_ENUMS = {"rag.refusal.reason": REFUSAL_REASONS, "obslab.ingest.reason": INGEST_REASONS}
MAX_TEXT = 64               # an id or an error type fits; a sentence does not


def emit(event: str, fields: dict[str, Any] | None = None) -> None:
    """Write one event. Raises on an unknown event, field or value: a programming error, caught
    by the unit tests, never a runtime condition."""
    fields = fields or {}
    if event not in EVENTS:
        raise ValueError(f"unknown log event {event!r}")
    level, allowed = EVENTS[event]
    if unknown := set(fields) - allowed:
        raise ValueError(f"log event {event!r} has no field {sorted(unknown)}")
    for name, value in fields.items():
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"field {name!r} of {event!r} must be a string, a number or a boolean")
        if isinstance(value, str) and len(value) > MAX_TEXT:
            raise ValueError(f"field {name!r} of {event!r} is longer than {MAX_TEXT} characters")
        if name in _ENUMS and value not in _ENUMS[name]:
            raise ValueError(f"{name}={value!r} is not one of {sorted(_ENUMS[name])}")
    # The event name is the body of the record; `extra` becomes record attributes, which the OTLP
    # handler exports as log attributes (Loki: structured metadata, `rag.route` -> `rag_route`).
    _logger.log(level, event, extra={EVENT: event, **fields})


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, event, service, version, trace and span ids when
    the line is written inside a span, then the fields of the event."""

    def __init__(self, service: str, version: str):
        super().__init__()
        self.service, self.version = service, version

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, EVENT, None)
        line: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": event or record.getMessage(),
            "service": self.service,
            "version": self.version,
        }
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            line["trace_id"] = format(ctx.trace_id, "032x")
            line["span_id"] = format(ctx.span_id, "016x")
        if event:
            _, allowed = EVENTS[event]
            line.update({name: getattr(record, name) for name in sorted(allowed) if hasattr(record, name)})
        return json.dumps(line, ensure_ascii=False)
