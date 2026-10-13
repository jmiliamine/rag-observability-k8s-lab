"""Composition root: builds telemetry, models, vector store and the RAG graph once."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from .config import Settings
from .rag import memory
from .rag.graph import RagApp
from .rag.pgvector import PgVectorStore, open_pool
from .rag.providers import chat_model, embeddings
from .telemetry import GenAIMetrics, RagMetrics, Telemetry
from .telemetry.callbacks import OTelCallbackHandler

EMPTY_INDEX = "index is empty: run the ingest (task app:ingest, or obslab ingest)"


@dataclass
class Components:
    settings: Settings
    telemetry: Telemetry
    genai: GenAIMetrics
    rag: RagApp
    graph: object
    conversations: BaseCheckpointSaver | None = None

    def ask(self, question: str, conversation_id: str | None = None, deadline_s: float | None = None):
        if isinstance(self.conversations, memory.ConversationStore):
            self.conversations.ensure_tables()
        return self.rag.ask(question, self.graph, conversation_id, deadline_s)

    def status(self) -> dict:
        """What /readyz reports. The vector store is checked on every call: the index can be
        rebuilt, or the database restarted, while the API keeps running."""
        store = self.rag.store
        kind = "pgvector" if isinstance(store, PgVectorStore) else "memory"
        if isinstance(store, PgVectorStore):
            try:
                meta, chunks = store.status()
            except Exception as e:      # the message may carry connection details: keep the type only
                return {"ready": False, "store": kind, "chunks": 0,
                        "error": f"vector database unreachable ({type(e).__name__})"}
            error = check_index_meta(self.settings, meta) if meta else EMPTY_INDEX
            if error is None and isinstance(self.conversations, memory.ConversationStore):
                try:                # a no-op once the tables exist
                    self.conversations.ensure_tables()
                except Exception as e:
                    error = f"conversation store unreachable ({type(e).__name__})"
        else:                       # the in-memory double the tests pass in
            chunks = len(store.store)
            error = None if chunks else EMPTY_INDEX
        body = {"ready": self.graph is not None and chunks > 0 and error is None,
                "store": kind, "provider": self.settings.provider, "chunks": chunks}
        if error:
            body["error"] = error
        return body

    def close(self) -> None:
        for holder in (self.rag.store, self.conversations):
            pool = getattr(holder, "pool", None)
            if pool is not None:
                pool.close()


def open_store(settings: Settings, telemetry: Telemetry, emb) -> PgVectorStore:
    """The index in PostgreSQL. Connection settings come from the libpq PG* variables."""
    pool = open_pool(timeout_ms=settings.db_timeout_ms, name=settings.service_name)
    return PgVectorStore(pool, emb, tracer=telemetry.tracer)


def build_components(settings: Settings, telemetry: Telemetry, store: VectorStore | None = None,
                     conversations: BaseCheckpointSaver | None = None) -> Components:
    """Without arguments: the index and the conversations in PostgreSQL. The tests pass an
    in-memory store, and then get in-memory conversations unless they pass their own."""
    genai = GenAIMetrics(telemetry.meter, settings.price_in, settings.price_out)
    rag_metrics = RagMetrics(telemetry.meter)
    emb = embeddings(settings, telemetry, genai)
    if conversations is None:
        conversations = (memory.ConversationStore(memory.open_pool(timeout_ms=settings.db_timeout_ms))
                         if store is None else InMemorySaver())
    if store is None:
        store = open_store(settings, telemetry, emb)
    elif isinstance(store, InMemoryVectorStore):
        store.embedding = emb
    handler = OTelCallbackHandler(telemetry, genai, rag_metrics)
    rag = RagApp(settings, telemetry, store, chat_model(settings), handler, rag_metrics)
    return Components(settings, telemetry, genai, rag, rag.build(conversations), conversations)


def check_index_meta(settings: Settings, meta: dict) -> str | None:
    """None if the index can be queried with the configured embedding model, else why not."""
    built_with = meta.get("embed_model")
    if built_with != settings.embed_model_id:
        return (f"index built with embed model {built_with!r}, but {settings.embed_model_id!r} is configured: "
                "run the ingest again")
    return None
