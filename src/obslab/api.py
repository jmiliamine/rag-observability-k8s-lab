"""HTTP entry point.

- FastAPI auto-instrumentation adds the server span (http.server.request.duration)
  that parents the whole RAG trace.
- Two probe endpoints, because Kubernetes asks two different questions:
    /healthz  "is the process alive?"        -> livenessProbe (restart if not)
    /readyz   "can it serve a question now?" -> readinessProbe (remove from Service if not)
  Readiness checks what the app owns: the graph, and an index in its own database built
  with the configured embedding model. It does NOT call Ollama: if a shared dependency
  is down, every pod would turn unready at once and the Service would have zero
  endpoints -- errors are better surfaced as 502s and alerts than as a silent outage.
- On shutdown (SIGTERM from Kubernetes) the lifespan hook flushes spans, metrics and
  logs still buffered in the batch processors, so the last requests are not lost.
- /ask answers 503 when the model's waiting line is full and 504 when the request deadline
  passes (rag/admission.py). The probes do not look at that line: a busy model is not an
  unready pod.
- A caller can say how long it will wait, in seconds, with the X-Request-Timeout header. The
  deadline is then the smaller of that value and OBSLAB_REQUEST_DEADLINE_S: nothing is computed
  for a caller who has already given up.
- /ask takes an optional conversation_id (returned by the previous answer). The history
  lives in PostgreSQL, so any replica can serve the next question of a conversation.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Response
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field

from .app import Components
from .rag.admission import DeadlineExceeded, Overloaded
from .rag.graph import ConversationTooLongError
from .telemetry import emit


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # Returned by a previous answer; omit it to start a new conversation. A UUID and nothing else:
    # the value ends up in a database key and on spans.
    conversation_id: UUID | None = None


def create_app(components: Components) -> FastAPI:
    # Emit the *stable* HTTP semantic conventions (http.server.request.duration, seconds)
    # instead of the legacy http.server.duration in ms. Read by the instrumentation at
    # instrument time, so it must be set before instrument_app(). Env can still override.
    os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "http")
    tel = components.telemetry

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        emit("app.started", {"obslab.provider": components.settings.provider})
        yield
        emit("app.stopping")
        components.close()      # database connections
        tel.shutdown()          # flush telemetry buffers before the process exits

    # FastAPI >= 0.142 has its own telemetry, which adds an OTLP exporter when
    # OTEL_EXPORTER_OTLP_ENDPOINT is set. The app already configures export, so turn that off
    # (otherwise every signal is sent twice) and hand FastAPI the app's providers.
    app = FastAPI(title="obslab RAG", lifespan=lifespan, telemetry={
        "auto_configure": False,
        "tracer_provider": tel.tracer_provider,
        "meter_provider": tel.meter_provider,
        "logger_provider": tel.logger_provider,
    })

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/readyz")
    def readyz(response: Response):
        body = components.status()
        if not body["ready"]:
            response.status_code = 503
        return body

    @app.post("/ask")
    def ask(body: Ask, x_request_timeout: Annotated[float | None, Header(ge=1, allow_inf_nan=False)] = None):
        try:
            result = components.ask(body.question, str(body.conversation_id) if body.conversation_id else None,
                                    deadline_s=x_request_timeout)
        except ConversationTooLongError as e:
            emit("rag.refused", {"rag.refusal.reason": "conversation_too_long"})
            raise HTTPException(status_code=409, detail=f"{e}: start a new conversation") from e
        except Overloaded as e:
            # Refused before any work: the model already has a full waiting line. 503 and not 429,
            # the limit is the service's capacity, not this client's quota.
            emit("rag.refused", {"rag.refusal.reason": "queue_full"})
            raise HTTPException(status_code=503, detail="busy: the model queue is full, retry shortly",
                                headers={"Retry-After": "5"}) from e
        except DeadlineExceeded as e:
            emit("rag.refused", {"rag.refusal.reason": "deadline"})
            raise HTTPException(status_code=504, detail="the request deadline passed before an answer was ready",
                                headers={"Retry-After": "5"}) from e
        except Exception as e:
            # 502: a dependency (model, vector database) failed. Only the error type goes back to the
            # client and into the log; the full message, which can name hosts or users, stays on
            # the trace.
            emit("rag.failed", {"error.type": type(e).__name__[:64]})
            raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}") from e
        span_ctx = trace.get_current_span().get_span_context()
        result["trace_id"] = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
        return result

    FastAPIInstrumentor.instrument_app(app, tracer_provider=tel.tracer_provider, meter_provider=tel.meter_provider,
                                       excluded_urls="healthz,readyz")   # probes would drown real traffic
    return app
