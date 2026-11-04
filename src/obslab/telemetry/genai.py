"""Instruments following the OpenTelemetry GenAI semantic conventions
(github.com/open-telemetry/semantic-conventions-genai, status: development),
plus a few RAG-specific metrics under the `rag.` namespace.

Cardinality rule: attributes are bounded enums (operation, provider, model,
node, reason, error.type). Never put prompts, questions or ids in metric
attributes — that belongs on spans (and content only when opt-in).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry.metrics import CallbackOptions, Meter, Observation

# Bucket boundaries recommended by the conventions.
DURATION_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]
TOKEN_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
WORKFLOW_BUCKETS = [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300]
SCORE_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Finer than DURATION_BUCKETS: a small local model writes a chunk every 20 to 300 ms, which the
# duration buckets would squeeze into three bins.
CHUNK_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 2.5]

# attribute keys
OP = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
REQ_MODEL = "gen_ai.request.model"
RESP_MODEL = "gen_ai.response.model"
ERROR_TYPE = "error.type"


class GenAIMetrics:
    def __init__(self, meter: Meter, price_in: float = 0.0, price_out: float = 0.0):
        self.price_in, self.price_out = price_in, price_out
        self.duration = meter.create_histogram(
            "gen_ai.client.operation.duration", unit="s", description="GenAI operation duration.",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS)
        self.ttfc = meter.create_histogram(
            "gen_ai.client.operation.time_to_first_chunk", unit="s",
            description="Time to receive the first chunk of a streamed response.",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS)
        # One value per chunk after the first, from the end of the previous one: the pace of the
        # answer once it has started. Long answers weigh more in its percentiles; the per-answer
        # mean is on the chat span (obslab.time_per_output_chunk_s).
        self.time_per_chunk = meter.create_histogram(
            "gen_ai.client.operation.time_per_output_chunk", unit="s",
            description="Time between two chunks of a streamed response, after the first one.",
            explicit_bucket_boundaries_advisory=CHUNK_BUCKETS)
        self.input_tokens = meter.create_counter(
            "gen_ai.client.inference.usage.input_tokens", unit="{token}", description="Input tokens used.")
        self.output_tokens = meter.create_counter(
            "gen_ai.client.inference.usage.output_tokens", unit="{token}", description="Output tokens used.")
        self.op_input_tokens = meter.create_histogram(
            "gen_ai.client.inference.operation.input_tokens", unit="{token}",
            description="Input tokens per inference operation.", explicit_bucket_boundaries_advisory=TOKEN_BUCKETS)
        self.op_output_tokens = meter.create_histogram(
            "gen_ai.client.inference.operation.output_tokens", unit="{token}",
            description="Output tokens per inference operation.", explicit_bucket_boundaries_advisory=TOKEN_BUCKETS)
        self.workflow_duration = meter.create_histogram(
            "gen_ai.invoke_workflow.duration", unit="s", description="Duration of a GenAI workflow invocation.",
            explicit_bucket_boundaries_advisory=WORKFLOW_BUCKETS)
        # Not in the conventions: an API-equivalent cost, useful even with a free local model.
        self.cost = meter.create_counter(
            "obslab.llm.cost.usd", unit="{USD}", description="Estimated cost at the configured price list.")

    def record_inference(self, attrs: dict[str, str], duration_s: float, input_tokens: int | None,
                         output_tokens: int | None, ttfc_s: float | None = None) -> None:
        self.duration.record(duration_s, attrs)
        if ttfc_s is not None:
            self.ttfc.record(ttfc_s, attrs)
        if input_tokens is not None:
            self.input_tokens.add(input_tokens, attrs)
            self.op_input_tokens.record(input_tokens, attrs)
        if output_tokens is not None:
            self.output_tokens.add(output_tokens, attrs)
            self.op_output_tokens.record(output_tokens, attrs)
        cost = ((input_tokens or 0) * self.price_in + (output_tokens or 0) * self.price_out) / 1_000_000
        if cost:
            self.cost.add(cost, attrs)


class RagMetrics:
    def __init__(self, meter: Meter):
        # Questions being answered in this process, by start time: the oldest one is the signal
        # for a stuck request. 0 when idle, so "idle" and "not reporting" stay apart.
        self._in_flight: dict[int, float] = {}
        self._in_flight_lock = threading.Lock()
        meter.create_observable_gauge(
            "rag.request.oldest.age", callbacks=[self._oldest_age], unit="s",
            description="Age of the oldest question still being answered in this process (0 when idle).")
        self.documents = meter.create_histogram(
            "rag.retrieval.documents", unit="{document}", description="Documents returned by the retriever.",
            explicit_bucket_boundaries_advisory=[0, 1, 2, 3, 4, 5, 8, 10, 20])
        self.top_score = meter.create_histogram(
            "rag.retrieval.top_score", unit="{score}", description="Similarity of the best retrieved chunk.",
            explicit_bucket_boundaries_advisory=SCORE_BUCKETS)
        self.relevant = meter.create_histogram(
            "rag.retrieval.relevant_documents", unit="{document}",
            description="Documents above the relevance threshold.",
            explicit_bucket_boundaries_advisory=[0, 1, 2, 3, 4, 5, 8, 10, 20])
        self.rewrites = meter.create_counter(
            "rag.query.rewrites", unit="{rewrite}", description="Query rewrites triggered by poor retrieval.")
        self.fallbacks = meter.create_counter(
            "rag.fallbacks", unit="{request}",
            description="Questions answered with I-don't-know: no relevant context, or the model declined.")
        self.groundedness = meter.create_histogram(
            "rag.answer.groundedness", unit="{score}",
            description="Share of answer words found in the retrieved context (0-1): vocabulary, not correctness.",
            explicit_bucket_boundaries_advisory=SCORE_BUCKETS)
        self.turn = meter.create_histogram(
            "rag.conversation.turn", unit="{turn}",
            description="Position of each question in its conversation (1 = first question).",
            explicit_bucket_boundaries_advisory=[1, 2, 3, 5, 8, 13, 21])
        self.node_duration = meter.create_histogram(
            "rag.graph.node.duration", unit="s", description="Duration of each LangGraph node.",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS)
        self.admission_wait = meter.create_histogram(
            "rag.admission.wait", unit="s",
            description="Time a model call waited for a slot, apart from the model's own execution time.",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS)
        self.admission_waiting = meter.create_up_down_counter(
            "rag.admission.waiting", unit="{call}", description="Model calls waiting for a slot right now.")
        self.admission_rejections = meter.create_counter(
            "rag.admission.rejections", unit="{request}",
            description="Requests dropped before the model was asked: waiting line full, or deadline passed.")
        self.node_retries = meter.create_counter(
            "rag.graph.node.retries", unit="{attempt}",
            description="Node executions repeated by a retry policy (attempts after the first).")

    @contextmanager
    def in_flight(self) -> Iterator[None]:
        """Count a question as in flight for the duration of the block."""
        key = object.__hash__(object())
        with self._in_flight_lock:
            self._in_flight[key] = time.monotonic()
        try:
            yield
        finally:
            with self._in_flight_lock:
                self._in_flight.pop(key, None)

    def _oldest_age(self, _options: CallbackOptions) -> Iterator[Observation]:
        with self._in_flight_lock:
            oldest = min(self._in_flight.values(), default=None)
        yield Observation(0.0 if oldest is None else round(time.monotonic() - oldest, 3))


class HostMetrics:
    """Host signals read by the poller (telemetry/host.py), as observable instruments: the SDK asks
    for their values at each export. GPU names follow the OpenTelemetry hardware conventions
    (development); the Ollama ones are the lab's own, Ollama has no conventions or endpoint."""

    def __init__(self, meter: Meter, sampler) -> None:
        self.sampler = sampler
        meter.create_observable_gauge(
            "hw.gpu.utilization", callbacks=[self._utilization], unit="1",
            description="Fraction of time the GPU was busy over the last sample period.")
        meter.create_observable_up_down_counter(
            "hw.gpu.memory.usage", callbacks=[self._memory_used], unit="By", description="GPU memory used.")
        meter.create_observable_up_down_counter(
            "hw.gpu.memory.limit", callbacks=[self._memory_total], unit="By", description="GPU memory available.")
        meter.create_observable_gauge(
            "obslab.host.gpu.temperature", callbacks=[self._temperature], unit="Cel",
            description="GPU temperature: laptops slow the GPU down when it is hot.")
        meter.create_observable_up_down_counter(
            "obslab.ollama.model.size", callbacks=[self._model_size], unit="By",
            description="Memory a loaded Ollama model needs, with its context (KV cache included).")
        meter.create_observable_up_down_counter(
            "obslab.ollama.model.vram", callbacks=[self._model_vram], unit="By",
            description="Part of a loaded Ollama model held in GPU memory; the rest runs on the CPU.")
        meter.create_observable_gauge(
            "obslab.ollama.models.loaded", callbacks=[self._models_loaded], unit="{model}",
            description="Models Ollama holds in memory; 0 means the next question pays a cold load.")

    @staticmethod
    def _gpu_attrs(gpu) -> dict[str, str]:
        return {"hw.id": f"gpu{gpu.index}", "hw.type": "gpu", "hw.name": gpu.name}

    def _gpu_values(self, field: str) -> Iterator[Observation]:
        for gpu in self.sampler.sample()[0]:
            value = getattr(gpu, field)
            if value is not None:
                yield Observation(value, self._gpu_attrs(gpu))

    def _utilization(self, _options: CallbackOptions) -> Iterator[Observation]:
        for o in self._gpu_values("utilization"):
            yield Observation(o.value, {**o.attributes, "hw.gpu.task": "general"})

    def _memory_used(self, _options: CallbackOptions) -> Iterator[Observation]:
        yield from self._gpu_values("memory_used")

    def _memory_total(self, _options: CallbackOptions) -> Iterator[Observation]:
        yield from self._gpu_values("memory_total")

    def _temperature(self, _options: CallbackOptions) -> Iterator[Observation]:
        yield from self._gpu_values("temperature")

    def _model_size(self, _options: CallbackOptions) -> Iterator[Observation]:
        for m in self.sampler.sample()[1]:
            yield Observation(m.size, {REQ_MODEL: m.model})

    def _model_vram(self, _options: CallbackOptions) -> Iterator[Observation]:
        for m in self.sampler.sample()[1]:
            yield Observation(m.size_vram, {REQ_MODEL: m.model})

    def _models_loaded(self, _options: CallbackOptions) -> Iterator[Observation]:
        yield Observation(len(self.sampler.sample()[1]))

