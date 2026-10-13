"""Instruments following the OpenTelemetry GenAI semantic conventions
(github.com/open-telemetry/semantic-conventions-genai, status: development),
plus a few RAG-specific metrics under the `rag.` namespace.

Cardinality rule: attributes are bounded enums (operation, provider, model,
node, reason, error.type). Never put prompts, questions or ids in metric
attributes — that belongs on spans (and content only when opt-in).
"""

from __future__ import annotations

from opentelemetry.metrics import Meter

# Bucket boundaries recommended by the conventions.
DURATION_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]
TOKEN_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
WORKFLOW_BUCKETS = [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300]
SCORE_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
