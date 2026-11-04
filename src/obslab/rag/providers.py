"""Model providers behind one factory: `ollama` (local, default) or `fake`
(deterministic, zero-dependency, used by tests and the no-GPU demo). Adding
another provider only touches this file.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import Settings
from ..telemetry.genai import ERROR_TYPE, OP, PROVIDER, REQ_MODEL, GenAIMetrics
from ..telemetry.setup import Telemetry

WORD = re.compile(r"[a-zA-Zà-ÿ0-9]{2,}")
# Words that carry no topic: without this list, "what is the..." questions all look alike.
STOPWORDS = frozenset("""a an and are as at be by can do does for from has have how i if in into is it its of on
or that the their then there these this to was what when where which who why will with you your not no only than
so all any each more most other some such use used using get gets out up about after before over under also just
very""".split())


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


# ── Fake models (tests / offline demos) ──────────────────────────────────────

class HashingEmbeddings(Embeddings):
    """Bag-of-words hashed into a fixed vector: texts sharing topic words get similar vectors.

    Crude on purpose (no model download, deterministic), but good enough for the demo corpus:
    on the sample notes every load-test question retrieves its note with a score >= 0.22, while
    off-topic questions stay under 0.15 (hence OBSLAB_MIN_SCORE=0.2 in fake mode).
    """

    def __init__(self, dim: int = 1024, model: str = "fake-embed"):
        self.dim, self.model = dim, model

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in WORD.findall(text.lower()):
            if w not in STOPWORDS:
                v[int(hashlib.md5(_stem(w).encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeChat(BaseChatModel):
    """Answers from the prompt itself, reports usage, supports streaming.

    - the condense prompt ("Follow-up:") returns the follow-up, completed with the previous
      question when it refers to it ("it", "that", "what about"...) or is very short
    - prompts containing "Rewrite" return a reformulated query
    - prompts containing "FAIL" raise (to test error paths)
    - otherwise returns the first sentence of the context block
    """

    model_name: str = "fake-chat"
    latency_s: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any):
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "obslab.fake"
        params["ls_model_name"] = self.model_name
        return params

    def _reply(self, messages: list[BaseMessage]) -> str:
        prompt = "\n".join(str(m.content) for m in messages)
        if "FAIL" in prompt:
            raise RuntimeError("fake provider failure")
        if "Follow-up:" in prompt:
            follow_up = prompt.rsplit("Follow-up:", 1)[-1].split("Standalone question:", 1)[0].strip()
            asked = re.findall(r"^User: (.*)$", prompt, flags=re.MULTILINE)
            words = set(re.findall(r"[a-z']+", follow_up.lower()))
            dependent = len(words) < 5 or words & {"it", "its", "they", "them", "their", "that", "this",
                                                    "those", "these", "about", "also", "same"}
            if not (dependent and asked):
                return follow_up
            subject = asked[-1].replace(follow_up, "").strip()      # the same follow-up asked twice
            return f"{follow_up} {subject}".strip()
        if "Rewrite" in prompt:
            q = prompt.rsplit("Question:", 1)[-1].strip()
            return f"{q} definition explanation"
        ctx = prompt.split("Context:", 1)[-1].split("Question:", 1)[0]
        first = re.split(r"(?<=[.!?])\s", ctx.strip(), maxsplit=1)[0].strip()
        return first or "I don't know."

    def _usage(self, messages: list[BaseMessage], text: str):
        n_in = sum(len(str(m.content).split()) for m in messages)
        return {"input_tokens": n_in, "output_tokens": len(text.split()), "total_tokens": n_in + len(text.split())}

    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs):
        time.sleep(self.latency_s)
        text = self._reply(messages)
        msg = AIMessage(content=text, usage_metadata=self._usage(messages, text),
                        response_metadata={"model_name": self.model_name, "finish_reason": "stop"})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs) -> Iterator[ChatGenerationChunk]:
        time.sleep(self.latency_s)
        text = self._reply(messages)
        words = text.split(" ")
        for i, w in enumerate(words):
            last = i == len(words) - 1
            chunk = ChatGenerationChunk(message=AIMessageChunk(
                content=w + ("" if last else " "),
                usage_metadata=self._usage(messages, text) if last else None,
                response_metadata={"model_name": self.model_name, "finish_reason": "stop"} if last else {}))
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


# ── Instrumented embeddings (LangChain has no callbacks for embeddings) ──────

class InstrumentedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, telemetry: Telemetry, genai: GenAIMetrics, provider: str, model: str):
        self.inner, self.tracer, self.genai = inner, telemetry.tracer, genai
        self.attrs = {OP: "embeddings", PROVIDER: provider, REQ_MODEL: model}

    def _call(self, fn, arg, count: int):
        start = time.perf_counter()
        with self.tracer.start_as_current_span(f"embeddings {self.attrs[REQ_MODEL]}", kind=SpanKind.CLIENT,
                                               attributes={**self.attrs, "obslab.embeddings.count": count},
                                               record_exception=False) as span:
            try:
                out = fn(arg)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)[:200]))
                span.set_attribute(ERROR_TYPE, type(e).__name__)
                self.genai.duration.record(time.perf_counter() - start, {**self.attrs, ERROR_TYPE: type(e).__name__})
                raise
            if out:
                dim = len(out[0]) if isinstance(out[0], list) else len(out)
                span.set_attribute("gen_ai.embeddings.dimension.count", dim)
        self.genai.duration.record(time.perf_counter() - start, self.attrs)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(self.inner.embed_documents, texts, len(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._call(self.inner.embed_query, text, 1)


# ── Factory ──────────────────────────────────────────────────────────────────

def chat_model(settings: Settings) -> BaseChatModel:
    if settings.provider == "fake":
        return FakeChat(model_name="fake-chat")
    from langchain_ollama import ChatOllama
    return ChatOllama(model=settings.chat_model, base_url=settings.ollama_url, temperature=0.1,
                      client_kwargs={"timeout": model_timeout(settings)})


def model_timeout(settings: Settings):
    """How long the Ollama clients wait for a byte. The request deadline is checked between two
    chunks; a model that sends nothing at all would hold the request, and its admission slot,
    forever without this. It bounds a silence, not an answer: a slow model still streams."""
    import httpx
    return httpx.Timeout(settings.model_read_timeout_s, connect=5.0)


def embeddings(settings: Settings, telemetry: Telemetry, genai: GenAIMetrics) -> Embeddings:
    if settings.provider == "fake":
        inner: Embeddings = HashingEmbeddings()
        provider, model = "obslab.fake", "fake-embed"
    else:
        from langchain_ollama import OllamaEmbeddings
        inner = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_url,
                                 client_kwargs={"timeout": model_timeout(settings)})
        provider, model = "ollama", settings.embed_model
    return InstrumentedEmbeddings(inner, telemetry, genai, provider, model)
