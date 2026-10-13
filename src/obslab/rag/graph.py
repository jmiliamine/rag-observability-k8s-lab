"""The RAG workflow as an explicit LangGraph state machine.

            ┌──────────┐    ┌──────────┐  relevant docs  ┌──────────┐    ┌───────┐
  START ──▶ │ condense │ ─▶ │ retrieve │ ──────────────▶ │ generate │ ─▶ │ grade │ ─▶ END
            └──────────┘    └──────────┘                 └──────────┘    └───────┘
                               │   ▲ none relevant, 1st time
                               ▼   │
                            ┌─────────┐            none relevant again ┌──────────┐
                            │ rewrite │  (LLM reformulates the query)  │ fallback │ ─▶ END
                            └─────────┘ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▶└──────────┘

Why a graph and not a single chain: every branch (rewrite, fallback) is a
named node, so it becomes a span and a metric label — the "why was this answer
bad" question gets an observable answer.

Conversations. A conversation is a LangGraph thread: the graph is compiled with a checkpointer
(rag/memory.py) and the last turns come back with the thread id. The history is used for one
thing: `condense` turns a follow-up ("and what about NoExecute?") into a standalone question,
which is what gets searched and answered. The answer itself is still written from the retrieved
notes only, so a follow-up costs one short model call and the prompt does not grow with the
conversation. The first question of a conversation skips that call.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from ..config import Settings
from ..telemetry.callbacks import OTelCallbackHandler
from ..telemetry.genai import OP, RagMetrics
from ..telemetry.setup import Telemetry
from .admission import REASON, Admission, DeadlineExceeded

log = logging.getLogger("obslab.rag")
MAX_REWRITES = 1
STOP = set("the a an of to in and or is are was what how why which with for on "
           "de la le les des un une et est que qui pour dans".split())
FALLBACK_ANSWER = "I don't know: nothing relevant in my notes for this question."

SYSTEM = ("You answer questions using ONLY the provided context from the user's notes. "
          "If the context is insufficient, say you don't know. Answer in 3-6 sentences.")
CONDENSE = ("Rewrite the follow-up question so that it can be understood without the conversation.\n"
            "- Replace each pronoun or vague reference (it, they, them, that, this...) with the exact thing "
            "it refers to in the conversation.\n"
            "- Keep the rest of the wording. Do not answer the question.\n"
            "- If the follow-up has no reference to the conversation, copy it unchanged.\n"
            "Example: after a conversation about the Eiffel Tower, \"How tall is it?\" becomes "
            "\"How tall is the Eiffel Tower?\"\n\n"
            "Conversation:\n{history}\n\nFollow-up: {question}\nStandalone question:")
ANSWER_CLIP = 400            # characters of each past answer shown to the condense step


class ConversationTooLongError(RuntimeError):
    """The conversation reached OBSLAB_MAX_TURNS: the client has to start a new one."""


class RagState(TypedDict, total=False):
    # Kept between the turns of a conversation (saved by the checkpointer):
    messages: Annotated[list[AnyMessage], add_messages]     # the last turns, trimmed on each answer
    turn: int
    # One turn only; ask() resets them on each question:
    question: str            # what the user typed
    standalone: str          # the question once references to earlier turns are resolved
    query: str               # what was searched (the standalone question, or its rewrite)
    hits: list[tuple[Document, float]]
    relevant: list[tuple[Document, float]]
    sources: list[str]
    rewrites: int
    answer: str
    route: str               # answered | fallback
    groundedness: float


# What a new turn starts from, whatever the previous turn of the conversation left behind.
NEW_TURN: RagState = {"standalone": "", "query": "", "hits": [], "relevant": [], "sources": [], "rewrites": 0,
                      "answer": "", "route": "", "groundedness": 0.0}


REFERENCES = set("it its they them their that this those these there one ones same also and but so then "
                 "done do does did about".split())


def standalone_question(follow_up: str, candidate: str) -> str:
    """The model's rewrite of a follow-up, or the follow-up itself when the rewrite lost it.

    A small model sometimes answers with a question about the previous subject instead of the one
    that was asked. A rewrite only adds what "it" or "that" stood for, so it must still contain
    the words of the follow-up; if most of them are gone, the follow-up is used as typed.
    """
    lines = candidate.strip().splitlines()
    candidate = lines[0].strip().strip('"').strip()[:500] if lines else ""
    if not candidate:
        return follow_up
    words = {w for w in re.findall(r"[a-zà-ÿ0-9]{3,}", follow_up.lower()) if w not in STOP | REFERENCES}
    if not words:
        return candidate
    kept = sum(w in candidate.lower() for w in words) / len(words)
    return candidate if kept >= 0.5 else follow_up


def groundedness(answer: str, context: str) -> float:
    """Cheap, deterministic proxy: share of answer content words present in the context."""
    words = [w for w in re.findall(r"[a-zà-ÿ0-9]{3,}", answer.lower()) if w not in STOP]
    if not words:
        return 0.0
    ctx = set(re.findall(r"[a-zà-ÿ0-9]{3,}", context.lower()))
    return round(sum(w in ctx for w in words) / len(words), 3)


@dataclass
class RagApp:
    settings: Settings
    telemetry: Telemetry
    store: VectorStore
    llm: BaseChatModel
    handler: OTelCallbackHandler
    rag_metrics: RagMetrics
    index_name: str = "notes"
    admission: Admission | None = None

    def __post_init__(self) -> None:
        if self.admission is None:
            self.admission = Admission(self.settings.model_concurrency, self.settings.model_queue,
                                       self.rag_metrics)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _node_ctx(self, config: RunnableConfig):
        cb = (config or {}).get("callbacks")
        return self.handler.context_for(getattr(cb, "parent_run_id", None))

    def _remember(self, state: RagState, answer: str) -> list[AnyMessage]:
        """This turn, plus the removal of the turns that fall out of the history window."""
        window = self.settings.history_turns
        old = state.get("messages", [])
        keep = max(window - 1, 0) * 2
        drop = old[:len(old) - keep] if keep else old
        # The question is kept as it was understood ("Why does a burn rate alert use two windows?",
        # not "Why does it use two windows?"): the subject stays in the history even when the turn
        # that introduced it has left the window.
        asked = state.get("standalone") or state["question"]
        new = [HumanMessage(content=asked), AIMessage(content=answer)] if window else []
        return [*(RemoveMessage(id=m.id) for m in drop), *new]

    @staticmethod
    def _deadline(config: RunnableConfig) -> float | None:
        return ((config or {}).get("configurable") or {}).get("deadline")

    @contextmanager
    def _model_slot(self, step: str, config: RunnableConfig):
        """One slot of the admission gate for a model call of this node. The wait is written on
        the node's span, so a trace shows waiting and model execution as two numbers."""
        with self.admission.slot(step, self._deadline(config)) as waited:
            trace.get_current_span(self._node_ctx(config)).set_attribute("rag.admission.wait_s", round(waited, 4))
            yield waited

    # ── nodes ────────────────────────────────────────────────────────────────
    def condense(self, state: RagState, config: RunnableConfig) -> dict[str, Any]:
        turn = state.get("turn", 0) + 1
        if turn > self.settings.max_turns:
            raise ConversationTooLongError(f"conversation limited to {self.settings.max_turns} turns")
        self.rag_metrics.turn.record(turn, {"gen_ai.data_source.id": self.index_name})
        history = state.get("messages", [])
        if not history:                         # first question: nothing to resolve, no model call
            return {"standalone": state["question"]}
        lines = [f"{'User' if m.type == 'human' else 'Assistant'}: {str(m.content)[:ANSWER_CLIP]}" for m in history]
        with self._model_slot("condense", config):
            msg = self.llm.invoke([HumanMessage(content=CONDENSE.format(
                history="\n".join(lines), question=state["question"]))], config)
        return {"standalone": standalone_question(state["question"], str(msg.content))}

    def retrieve(self, state: RagState, config: RunnableConfig) -> dict[str, Any]:
        query = state.get("query") or state.get("standalone") or state["question"]
        token = otel_context.attach(self._node_ctx(config))
        try:
            with self.telemetry.tracer.start_as_current_span(
                    f"retrieval {self.index_name}", kind=SpanKind.INTERNAL,
                    attributes={OP: "retrieval", "gen_ai.data_source.id": self.index_name,
                                "gen_ai.retrieval.top_k": self.settings.top_k}) as span:
                hits = self.store.similarity_search_with_score(query, k=self.settings.top_k)
                relevant = [(d, s) for d, s in hits if s >= self.settings.min_score]
                top = max((s for _, s in hits), default=0.0)
                span.set_attribute("rag.retrieval.documents", len(hits))
                span.set_attribute("rag.retrieval.relevant_documents", len(relevant))
                span.set_attribute("rag.retrieval.top_score", round(float(top), 4))
                if self.telemetry.capture_content:
                    span.set_attribute("gen_ai.retrieval.query.text", query)
        finally:
            otel_context.detach(token)
        attrs = {"gen_ai.data_source.id": self.index_name}
        self.rag_metrics.documents.record(len(hits), attrs)
        self.rag_metrics.relevant.record(len(relevant), attrs)
        if hits:
            self.rag_metrics.top_score.record(float(top), attrs)
        return {"query": query, "hits": hits, "relevant": relevant}

    def route_after_retrieve(self, state: RagState) -> str:
        if state.get("relevant"):
            return "generate"
        return "rewrite" if state.get("rewrites", 0) < MAX_REWRITES else "fallback"

    def rewrite(self, state: RagState, config: RunnableConfig) -> dict[str, Any]:
        with self._model_slot("rewrite", config):
            msg = self.llm.invoke([HumanMessage(
                content="Rewrite this question as a short search query with key technical terms.\n"
                        f"Question: {state.get('standalone') or state['question']}")], config)
        self.rag_metrics.rewrites.add(1, {"gen_ai.data_source.id": self.index_name})
        return {"query": str(msg.content).strip()[:300], "rewrites": state.get("rewrites", 0) + 1}

    def generate(self, state: RagState, config: RunnableConfig) -> dict[str, Any]:
        context = "\n\n---\n\n".join(d.page_content for d, _ in state["relevant"])
        messages = [SystemMessage(content=SYSTEM),
                    HumanMessage(content=f"Context:\n{context}\n\n"
                                         f"Question: {state.get('standalone') or state['question']}")]
        # Streaming on purpose: it is what makes time-to-first-chunk measurable, and what lets a
        # generation be cut when the request deadline passes (closing the stream stops the model).
        deadline, parts = self._deadline(config), []
        with self._model_slot("generate", config):
            stream = self.llm.stream(messages, config)
            try:
                for chunk in stream:
                    parts.append(str(chunk.content))
                    if deadline is not None and time.monotonic() > deadline:
                        self.rag_metrics.admission_rejections.add(1, {REASON: "deadline"})
                        raise DeadlineExceeded("the answer was still being written at the request deadline")
            finally:
                stream.close()
        answer = "".join(parts).strip()
        sources = sorted({d.metadata.get("source", "?") for d, _ in state["relevant"]})
        # The turn only counts once it is answered: a failed run leaves the conversation as it was.
        return {"answer": answer, "route": "answered", "sources": sources, "turn": state.get("turn", 0) + 1,
                "messages": self._remember(state, answer)}

    def fallback(self, state: RagState) -> dict[str, Any]:
        self.rag_metrics.fallbacks.add(1, {"reason": "no_relevant_context"})
        # The retrieved chunks are dropped before the state is saved: a checkpoint holds a few
        # turns of text, not documents.
        return {"answer": FALLBACK_ANSWER, "route": "fallback", "groundedness": 0.0, "sources": [],
                "hits": [], "relevant": [], "turn": state.get("turn", 0) + 1,
                "messages": self._remember(state, FALLBACK_ANSWER)}

    def grade(self, state: RagState) -> dict[str, Any]:
        context = " ".join(d.page_content for d, _ in state.get("relevant", []))
        score = groundedness(state.get("answer", ""), context)
        self.rag_metrics.groundedness.record(score, {"gen_ai.data_source.id": self.index_name})
        return {"groundedness": score, "hits": [], "relevant": []}

    # ── graph ────────────────────────────────────────────────────────────────
    def build(self, checkpointer: BaseCheckpointSaver | None = None):
        g = StateGraph(RagState)
        g.add_node("condense", self.condense)
        g.add_node("retrieve", self.retrieve)
        g.add_node("rewrite", self.rewrite)
        g.add_node("generate", self.generate)
        g.add_node("fallback", self.fallback)
        g.add_node("grade", self.grade)
        g.add_edge(START, "condense")
        g.add_edge("condense", "retrieve")
        g.add_conditional_edges("retrieve", self.route_after_retrieve,
                                {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"})
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", "grade")
        g.add_edge("grade", END)
        g.add_edge("fallback", END)
        return g.compile(name="rag", checkpointer=checkpointer)

    def ask(self, question: str, graph=None, conversation_id: str | None = None,
            deadline_s: float | None = None) -> dict[str, Any]:
        """One turn. Without `conversation_id` a new conversation starts; its id is returned so
        that the next question can continue it. `deadline_s` is how long the caller says it will
        wait; the server's own limit stays the upper bound."""
        graph = graph or self.build()
        conversation_id = conversation_id or str(uuid.uuid4())
        # One checkpoint per turn, written when the run succeeds (instead of one per node).
        saving = {"durability": "exit"} if graph.checkpointer else {}
        budget = min(deadline_s or self.settings.request_deadline_s, self.settings.request_deadline_s)
        deadline = time.monotonic() + budget
        trace.get_current_span().set_attribute("rag.request.deadline_s", budget)   # on the request span
        state = graph.invoke(
            {**NEW_TURN, "question": question},
            config={"callbacks": [self.handler],
                    "configurable": {"thread_id": conversation_id, "deadline": deadline}}, **saving)
        log.info("rag answered", extra={"rag.route": state.get("route"), "rag.rewrites": state.get("rewrites", 0),
                                        "rag.groundedness": state.get("groundedness", 0.0),
                                        "rag.turn": state.get("turn", 1),
                                        "gen_ai.conversation.id": conversation_id})
        return {"answer": state.get("answer", ""), "route": state.get("route"),
                "sources": state.get("sources", []), "rewrites": state.get("rewrites", 0),
                "groundedness": state.get("groundedness", 0.0),
                "conversation_id": conversation_id, "turn": state.get("turn", 1),
                "standalone_question": state.get("standalone") or question}
