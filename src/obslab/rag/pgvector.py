"""Vector store on PostgreSQL + pgvector.

Two roles, two entry points:
  - the API reads with `PgVectorStore` (role obslab_reader: SELECT only),
  - the ingest job writes with `rebuild()` (role obslab_writer: owns the tables).
Connection settings come from the standard libpq variables (PGHOST, PGDATABASE, PGUSER,
PGPASSWORD...), so no password ever goes through this code or its error messages.

Guardrails, in the order a request meets them:
  - statement_timeout on every connection: a slow query fails fast instead of holding a request;
  - k is bounded (1..MAX_K) whatever the caller asks;
  - the query vector must have the dimension recorded in rag.index_meta, otherwise the error
    says which model built the index instead of a cryptic pgvector message;
  - `rebuild()` fills a staging table and swaps it in one short transaction, so readers see the
    old index or the new one, never a half-built one;
  - every statement is parameterized; table names are constants, never input.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from opentelemetry.trace import SpanKind, Status, StatusCode
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

from . import corpus

SCHEMA = "rag"
CHUNKS = f"{SCHEMA}.chunks"
STAGING = f"{SCHEMA}.chunks_build"
META = f"{SCHEMA}.index_meta"
MAX_K = 20
HNSW_MAX_DIM = 2000          # pgvector's limit for an HNSW index on `vector`


class IndexMismatchError(RuntimeError):
    """The query embedding does not fit the stored index (other model, other dimension)."""


def _configure(timeout_ms: int):
    def configure(conn: Connection) -> None:
        register_vector(conn)
        conn.execute(f"SET statement_timeout = {int(timeout_ms)}")
        conn.commit()
    return configure


def open_pool(dsn: str = "", *, max_size: int = 4, timeout_ms: int = 5000, connect_timeout_s: int = 3,
              name: str = "obslab") -> ConnectionPool:
    """A small pool; `dsn` may be empty to rely on PG* environment variables only."""
    # check_connection: a connection is tested (a cheap round trip) before each use, so after a
    # database restart the pool replaces dead connections instead of failing one request each.
    return ConnectionPool(dsn, min_size=0, max_size=max_size, open=True, name=name, timeout=connect_timeout_s,
                          kwargs={"connect_timeout": connect_timeout_s, "application_name": name},
                          configure=_configure(timeout_ms), check=ConnectionPool.check_connection,
                          max_idle=300)


class PgVectorStore(VectorStore):
    """Read side: cosine similarity search over rag.chunks, scores in [-1, 1] like the in-memory store."""

    def __init__(self, pool: ConnectionPool, embedding: Embeddings, tracer=None, db_name: str = "obslab"):
        self.pool, self.embedding, self.tracer, self.db_name = pool, embedding, tracer, db_name

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    # ── status (readiness) ───────────────────────────────────────────────────
    def status(self) -> tuple[dict | None, int]:
        """(metadata of the current index or None, number of chunks). Raises if the DB is unreachable."""
        with self.pool.connection() as conn:
            if conn.execute("SELECT to_regclass(%s)", (META,)).fetchone()[0] is None:
                return None, 0
            row = conn.execute(f"SELECT provider, embed_model, dim, chunks FROM {META} WHERE id = 1").fetchone()
        if row is None:
            return None, 0
        provider, embed_model, dim, chunks = row
        return {"provider": provider, "embed_model": embed_model, "dim": dim, "chunks": chunks}, chunks

    # ── search ───────────────────────────────────────────────────────────────
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]:
        k = max(1, min(int(k), MAX_K))
        vector = self.embedding.embed_query(query)
        sql = (f"SELECT content, source, chunk, 1 - (embedding <=> %(q)s::vector) AS score "
               f"FROM {CHUNKS} ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s")
        span = None
        if self.tracer is not None:
            # Database client span, OpenTelemetry database semantic conventions.
            span = self.tracer.start_span(f"SELECT {CHUNKS}", kind=SpanKind.CLIENT, attributes={
                "db.system.name": "postgresql", "db.namespace": self.db_name,
                "db.collection.name": CHUNKS, "db.operation.name": "SELECT",
                "db.query.summary": f"SELECT {CHUNKS}", "db.response.returned_rows": 0})
        try:
            with self.pool.connection() as conn:
                dim, index_id = self._index(conn)
                if dim is not None and dim != len(vector):
                    raise IndexMismatchError(
                        f"query embedding has {len(vector)} dimensions, the index has {dim}: "
                        "it was built with another embedding model, run the ingest again")
                rows = conn.execute(sql, {"q": vector, "k": k}).fetchall()
            if span is not None:
                span.set_attribute("db.response.returned_rows", len(rows))
        except Exception as e:
            if span is not None:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, type(e).__name__))
                span.set_attribute("error.type", type(e).__name__)
            raise
        finally:
            if span is not None:
                span.end()
        # The id of the index travels with each result (read in the same connection as the rows), so
        # the caller can say which index answered without a shared field that another request overwrites.
        return [(Document(page_content=content, metadata={"source": source, "chunk": chunk, "index_id": index_id}),
                 float(score)) for content, source, chunk, score in rows]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k, **kwargs)]

    @staticmethod
    def _index(conn: Connection) -> tuple[int | None, str]:
        """(dimension, id) of the current index. The id is read through to_jsonb: an index built
        before the column existed has none, and must still be served."""
        if conn.execute("SELECT to_regclass(%s)", (META,)).fetchone()[0] is None:
            return None, ""
        row = conn.execute(f"SELECT dim, to_jsonb(m) ->> 'index_id' FROM {META} m WHERE id = 1").fetchone()
        return (row[0], row[1] or "") if row else (None, "")

    # The read side never writes: indexing goes through rebuild() with the writer role.
    def add_texts(self, texts: Iterable[str], metadatas: list[dict] | None = None, **kwargs: Any) -> list[str]:
        raise NotImplementedError("PgVectorStore is read-only; use obslab.rag.pgvector.rebuild()")

    @classmethod
    def from_texts(cls, texts: list[str], embedding: Embeddings, metadatas: list[dict] | None = None,
                   **kwargs: Any) -> PgVectorStore:
        raise NotImplementedError("use obslab.rag.pgvector.rebuild()")


def rebuild(pool: ConnectionPool, docs: Sequence[Document], embedding: Embeddings, meta: dict,
            batch: int = 64) -> int:
    """Write side: embed `docs` into a staging table, then swap it in atomically. Returns the chunk count."""
    if not docs:
        raise ValueError("no documents to index")
    vectors: list[list[float]] = []
    for i in range(0, len(docs), batch):
        vectors += embedding.embed_documents([d.page_content for d in docs[i:i + batch]])
    dim = len(vectors[0])
    if not 0 < dim <= HNSW_MAX_DIM or any(len(v) != dim for v in vectors):
        raise ValueError(f"unexpected embedding dimension {dim} (HNSW supports up to {HNSW_MAX_DIM})")

    with pool.connection() as conn:
        conn.execute("SET statement_timeout = 0")        # building the HNSW index can take a while
        with conn.transaction():
            conn.execute(f"DROP TABLE IF EXISTS {STAGING}")
            conn.execute(f"""CREATE TABLE {STAGING} (
                id integer NOT NULL,
                source text NOT NULL,
                chunk integer NOT NULL,
                content text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}',
                embedding vector({dim}) NOT NULL)""")
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {STAGING} (id, source, chunk, content, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [(i, d.metadata.get("source", "?"), int(d.metadata.get("chunk", i)), d.page_content,
                      json.dumps(d.metadata), v) for i, (d, v) in enumerate(zip(docs, vectors, strict=True))])
            # Index names are unique per schema and survive a table rename: build under staging
            # names, rename them with the table during the swap.
            conn.execute(f"ALTER TABLE {STAGING} ADD CONSTRAINT chunks_build_pkey PRIMARY KEY (id)")
            conn.execute(f"CREATE INDEX chunks_build_embedding ON {STAGING} USING hnsw (embedding vector_cosine_ops)")
        # The swap: readers block for a few milliseconds at most, then see the complete new index.
        with conn.transaction():
            conn.execute(f"""CREATE TABLE IF NOT EXISTS {META} (
                id integer PRIMARY KEY CHECK (id = 1),
                provider text NOT NULL, embed_model text NOT NULL, dim integer NOT NULL,
                chunks integer NOT NULL, built_at timestamptz NOT NULL DEFAULT now())""")
            # A table from before the id existed gets the column once. Checked first: ALTER TABLE takes
            # an exclusive lock even when there is nothing to add, and readiness reads this table.
            if conn.execute("SELECT 1 FROM information_schema.columns WHERE table_schema = %s "
                            "AND table_name = 'index_meta' AND column_name = 'index_id'", (SCHEMA,)).fetchone() is None:
                conn.execute(f"ALTER TABLE {META} ADD COLUMN index_id text")
            conn.execute(f"DROP TABLE IF EXISTS {CHUNKS}")
            conn.execute(f"ALTER TABLE {STAGING} RENAME TO chunks")
            conn.execute(f"ALTER INDEX {SCHEMA}.chunks_build_pkey RENAME TO chunks_pkey")
            conn.execute(f"ALTER INDEX {SCHEMA}.chunks_build_embedding RENAME TO chunks_embedding")
            conn.execute(
                f"""INSERT INTO {META} (id, provider, embed_model, dim, chunks, index_id)
                    VALUES (1, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET provider = EXCLUDED.provider, embed_model = EXCLUDED.embed_model,
                    dim = EXCLUDED.dim, chunks = EXCLUDED.chunks, index_id = EXCLUDED.index_id, built_at = now()""",
                (meta["provider"], meta["embed_model"], dim, len(docs),
                 corpus.index_id(docs, meta["embed_model"])))
    return len(docs)
