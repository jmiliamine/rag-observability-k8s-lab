"""LangChain/LangGraph callback handler -> OpenTelemetry spans + GenAI metrics.

Span tree produced for one question:

  invoke_workflow rag                  (graph run, INTERNAL, carries gen_ai.conversation.id)
  ├── rag.node condense                (a chat span under it on follow-up questions only)
  ├── rag.node retrieve                (one span per LangGraph node)
  │   ├── embeddings nomic-embed-text  (InstrumentedEmbeddings, CLIENT)
  │   └── retrieval index              (explicit span in the node)
  ├── rag.node generate
  │   └── chat llama3.2:3b             (this handler, CLIENT, tokens + TTFC)
  └── rag.node grade

A node retried by a LangGraph RetryPolicy runs again inside the same step: each attempt gets its
own `rag.node` span, numbered by `langgraph.attempt`, and the failed ones carry the error.

LangChain emits many internal runnables (sequences, channel writes); they get
no span of their own and are mapped to their nearest instrumented ancestor.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .genai import ERROR_TYPE, OP, PROVIDER, REQ_MODEL, RESP_MODEL, GenAIMetrics, RagMetrics
from .setup import Telemetry


class _Run:
    __slots__ = ("span", "start", "attrs", "first_chunk", "kind", "last_chunk", "gaps", "gap_sum")

    def __init__(self, span, attrs: dict[str, str], kind: str):
        self.span, self.attrs, self.kind = span, attrs, kind
        self.start = time.perf_counter()
        self.first_chunk: float | None = None
        self.last_chunk: float | None = None
        self.gaps, self.gap_sum = 0, 0.0


class OTelCallbackHandler(BaseCallbackHandler):
    run_inline = True          # keep callbacks on the caller's thread (context propagation)
    raise_error = False        # telemetry must never break the app

    def __init__(self, telemetry: Telemetry, genai: GenAIMetrics, rag: RagMetrics, workflow_name: str = "rag"):
        self.tracer = telemetry.tracer
        self.capture = telemetry.capture_content
        self.genai, self.rag = genai, rag
        self.workflow_name = workflow_name
        self._runs: dict[UUID, _Run] = {}
        self._alias: dict[UUID, UUID | None] = {}   # uninstrumented run -> instrumented ancestor
        self._attempts: dict[tuple[UUID, str, int], int] = {}   # (graph run, node, step) -> executions
        self._lock = threading.Lock()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _owner(self, run_id: UUID | None) -> UUID | None:
        while run_id is not None and run_id not in self._runs:
            run_id = self._alias.get(run_id)
        return run_id

    def _parent_ctx(self, parent_run_id: UUID | None):
        owner = self._owner(parent_run_id)
        if owner is None:
            return otel_context.get_current()
        return trace.set_span_in_context(self._runs[owner].span)

    def context_for(self, run_id: UUID | None):
        """OTel context of the span owning `run_id` (used by nodes for manual spans)."""
        with self._lock:
            return self._parent_ctx(run_id)

    def _start(self, run_id: UUID, parent_run_id: UUID | None, name: str, kind: SpanKind,
               attrs: dict[str, Any], run_kind: str) -> None:
        with self._lock:
            span = self.tracer.start_span(name, context=self._parent_ctx(parent_run_id), kind=kind, attributes=attrs)
            metric_attrs = {k: v for k, v in attrs.items() if k in (OP, PROVIDER, REQ_MODEL)}
            self._runs[run_id] = _Run(span, metric_attrs, run_kind)

    def _pop(self, run_id: UUID) -> _Run | None:
        with self._lock:
            self._alias.pop(run_id, None)
            run = self._runs.pop(run_id, None)
            if run is not None and run.kind == "workflow":     # the graph run is over: forget its attempts
                for key in [k for k in self._attempts if k[0] == run_id]:
                    del self._attempts[key]
            return run

    @staticmethod
    def _fail(run: _Run, error: BaseException) -> str:
        etype = type(error).__name__
        run.span.set_status(Status(StatusCode.ERROR, str(error)[:200]))
        run.span.set_attribute(ERROR_TYPE, etype)
        run.span.record_exception(error)
        return etype

    # ── chains: the graph itself and its nodes ───────────────────────────────
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kw):
        name = kw.get("name") or (serialized or {}).get("name", "chain")
        metadata = metadata or {}
        if parent_run_id is None:
            attrs = {OP: "invoke_workflow", "gen_ai.workflow.name": self.workflow_name}
            if metadata.get("thread_id"):       # on the span only: an id is never a metric label
                attrs["gen_ai.conversation.id"] = str(metadata["thread_id"])
            self._start(run_id, None, f"invoke_workflow {self.workflow_name}", SpanKind.INTERNAL, attrs, "workflow")
        elif metadata.get("langgraph_node") == name and not name.startswith("__"):
            # A retry policy runs the node again in the same step (a loop in the graph is a new
            # step): count the executions so that each attempt can be told apart in the trace.
            step = int(metadata.get("langgraph_step", 0))
            with self._lock:
                attempt = self._attempts[parent_run_id, name, step] = self._attempts.get(
                    (parent_run_id, name, step), 0) + 1
            if attempt > 1:
                self.rag.node_retries.add(1, {"langgraph.node": name})
            self._start(run_id, parent_run_id, f"rag.node {name}", SpanKind.INTERNAL,
                        {"langgraph.node": name, "langgraph.step": step, "langgraph.attempt": attempt}, "node")
        else:
            with self._lock:
                self._alias[run_id] = parent_run_id

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kw):
        run = self._pop(run_id)
        if run is None:
            return
        elapsed = time.perf_counter() - run.start
        if run.kind == "workflow":
            self.genai.workflow_duration.record(elapsed, {"gen_ai.workflow.name": self.workflow_name})
        elif run.kind == "node":
            self.rag.node_duration.record(elapsed, {"langgraph.node": run.span.name.split(" ", 1)[1]})
        run.span.end()

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kw):
        run = self._pop(run_id)
        if run is None:
            return
        etype = self._fail(run, error)
        if run.kind == "workflow":
            self.genai.workflow_duration.record(time.perf_counter() - run.start,
                                                {"gen_ai.workflow.name": self.workflow_name, ERROR_TYPE: etype})
        run.span.end()

    # ── chat models ──────────────────────────────────────────────────────────
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kw):
        metadata = metadata or {}
        params = kw.get("invocation_params") or {}
        provider = metadata.get("ls_provider") or "unknown"
        model = metadata.get("ls_model_name") or params.get("model") or params.get("model_name") or "unknown"
        attrs: dict[str, Any] = {OP: "chat", PROVIDER: provider, REQ_MODEL: model}
        if metadata.get("ls_temperature") is not None:
            attrs["gen_ai.request.temperature"] = float(metadata["ls_temperature"])
        if self.capture:
            attrs["gen_ai.input.messages"] = json.dumps(
                [{"role": m.type, "content": str(m.content)} for m in (messages[0] if messages else [])])
        self._start(run_id, parent_run_id, f"chat {model}", SpanKind.CLIENT, attrs, "llm")

    def on_llm_new_token(self, token, *, chunk=None, run_id, parent_run_id=None, **kw):
        run = self._runs.get(run_id)
        if run is None or not token:
            # An empty chunk carries no text: LangChain ends a stream with one, and so does
            # Ollama's final "done" message. Counting it would add a gap of nearly zero.
            return
        now = time.perf_counter()
        if run.first_chunk is None:
            run.first_chunk = now - run.start
        else:
            gap = now - run.last_chunk
            run.gaps, run.gap_sum = run.gaps + 1, run.gap_sum + gap
            self.genai.time_per_chunk.record(gap, run.attrs)
        run.last_chunk = now

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kw):
        run = self._pop(run_id)
        if run is None:
            return
        elapsed = time.perf_counter() - run.start
        usage, resp_model, finish, text = {}, None, None, ""
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            text = gen.text
            if msg is not None:
                usage = getattr(msg, "usage_metadata", None) or {}
                meta = getattr(msg, "response_metadata", {}) or {}
                resp_model = meta.get("model") or meta.get("model_name")
                finish = meta.get("done_reason") or meta.get("finish_reason") or meta.get("stop_reason")
        except (IndexError, AttributeError):
            pass
        in_tok, out_tok = usage.get("input_tokens"), usage.get("output_tokens")
        attrs = dict(run.attrs)
        if resp_model:
            attrs[RESP_MODEL] = resp_model
            run.span.set_attribute(RESP_MODEL, resp_model)
        if in_tok is not None:
            run.span.set_attribute("gen_ai.usage.input_tokens", in_tok)
        if out_tok is not None:
            run.span.set_attribute("gen_ai.usage.output_tokens", out_tok)
        if finish:
            run.span.set_attribute("gen_ai.response.finish_reasons", [str(finish)])
        if run.first_chunk is not None:
            run.span.set_attribute("obslab.time_to_first_chunk_s", round(run.first_chunk, 4))
        if run.gaps:
            run.span.set_attribute("obslab.time_per_output_chunk_s", round(run.gap_sum / run.gaps, 4))
        if self.capture:
            run.span.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "content": text}]))
        self.genai.record_inference(attrs, elapsed, in_tok, out_tok, run.first_chunk)
        run.span.end()

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kw):
        run = self._pop(run_id)
        if run is None:
            return
        etype = self._fail(run, error)
        self.genai.duration.record(time.perf_counter() - run.start, {**run.attrs, ERROR_TYPE: etype})
        run.span.end()
