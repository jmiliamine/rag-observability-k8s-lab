"""The pgvector store against a real PostgreSQL, guardrails included.

Run with `task test:pgvector` (starts the compose `postgres` service). Connection strings come
from PG_WRITER_DSN / PG_READER_DSN; the defaults match the local-only compose passwords.
Uses the fake embeddings: no model needed, and scores can be compared with the in-memory store.
"""

import os
import threading
import time
from pathlib import Path

import pytest
from langchain_core.vectorstores import InMemoryVectorStore

from lab_http import unavailable

pytestmark = pytest.mark.pgvector

WRITER = os.environ.get("PG_WRITER_DSN", "postgresql://obslab_writer:writer-local@127.0.0.1:5432/obslab")
READER = os.environ.get("PG_READER_DSN", "postgresql://obslab_reader:reader-local@127.0.0.1:5432/obslab")
NOTES = Path(__file__).resolve().parents[2] / "datalake"
META = {"provider": "fake", "embed_model": "fake-embed"}
QUESTIONS = {
    "How do taints and tolerations work?": "kubernetes/taints-and-tolerations.md",
    "How do you back up and restore etcd?": "kubernetes/etcd-backup.md",
    "What is a multi-window burn rate alert?": "observability/slo-and-burn-rate.md",
}


@pytest.fixture(scope="module")
def docs():
    from obslab.rag import corpus
    return corpus.split(corpus.load_folder(NOTES))


@pytest.fixture(scope="module")
def pools():
    import psycopg

    from obslab.rag.pgvector import open_pool
    try:
        psycopg.connect(WRITER, connect_timeout=3).close()
    except psycopg.OperationalError as e:
        unavailable(f"PostgreSQL not reachable ({type(e).__name__}): task stack:up, or task test:pgvector")
    writer, reader = open_pool(WRITER, name="test-writer"), open_pool(READER, name="test-reader", timeout_ms=1000)
    yield writer, reader
    writer.close()
    reader.close()


@pytest.fixture(scope="module")
def store(pools, docs):
    from obslab.rag.pgvector import PgVectorStore, rebuild
    from obslab.rag.providers import HashingEmbeddings
    writer, reader = pools
    assert rebuild(writer, docs, HashingEmbeddings(), META) == len(docs)
    return PgVectorStore(reader, HashingEmbeddings())


def test_index_metadata_is_recorded(store, docs):
    meta, chunks = store.status()
    assert chunks == len(docs)
    assert meta == {"provider": "fake", "embed_model": "fake-embed", "dim": 1024, "chunks": len(docs)}


@pytest.mark.parametrize("question,expected", QUESTIONS.items())
def test_same_ranking_and_scores_as_the_in_memory_store(store, docs, question, expected):
    from obslab.rag.providers import HashingEmbeddings
    memory = InMemoryVectorStore(HashingEmbeddings())
    memory.add_documents(docs)
    pg_hits = store.similarity_search_with_score(question, k=4)
    mem_hits = memory.similarity_search_with_score(question, k=4)
    assert pg_hits[0][0].metadata["source"] == expected
    assert [round(s, 4) for _, s in pg_hits] == [round(s, 4) for _, s in mem_hits]


def test_k_is_bounded(store):
    from obslab.rag.pgvector import MAX_K
    assert len(store.similarity_search_with_score("kubernetes", k=10_000)) == MAX_K
    assert len(store.similarity_search_with_score("kubernetes", k=0)) == 1


def test_every_result_says_which_index_it_comes_from(store, docs):
    from obslab.rag import corpus
    hits = store.similarity_search_with_score("etcd backup", k=3)
    assert {d.metadata["index_id"] for d, _ in hits} == {corpus.index_id(docs, META["embed_model"])}


def test_an_index_built_before_the_id_existed_is_still_served(pools, docs):
    """A database volume from an earlier version: rag.index_meta has no index_id column. The
    API reads it without failing, and the next ingest adds the column."""
    from obslab.rag import corpus
    from obslab.rag.pgvector import PgVectorStore, rebuild
    from obslab.rag.providers import HashingEmbeddings
    writer, reader = pools
    with writer.connection() as conn:
        conn.execute("ALTER TABLE rag.index_meta DROP COLUMN index_id")
    try:
        old = PgVectorStore(reader, HashingEmbeddings())
        hits = old.similarity_search_with_score("etcd backup", k=2)
        assert len(hits) == 2 and hits[0][0].metadata["index_id"] == ""
        assert old.status()[1] == len(docs)
    finally:
        rebuild(writer, docs, HashingEmbeddings(), META)
    (doc, _), = old.similarity_search_with_score("etcd backup", k=1)
    assert doc.metadata["index_id"] == corpus.index_id(docs, META["embed_model"])


def test_question_text_is_never_sql(store, docs):
    hits = store.similarity_search_with_score("x'; DROP TABLE rag.chunks; --", k=2)
    assert len(hits) == 2
    assert store.status()[1] == len(docs)


@pytest.mark.parametrize("statement", [
    "INSERT INTO rag.chunks (id, source, chunk, content, embedding) VALUES (-1, 'x', 0, 'x', NULL)",
    "DELETE FROM rag.chunks",
    "DROP TABLE rag.chunks",
    "CREATE TABLE public.intruder (id int)",
])
def test_the_api_role_cannot_change_anything(pools, statement):
    import psycopg
    _, reader = pools
    with pytest.raises((psycopg.errors.InsufficientPrivilege, psycopg.errors.ReadOnlySqlTransaction)):
        with reader.connection() as conn:
            conn.execute(statement)


def test_slow_queries_are_cut_by_the_statement_timeout(pools):
    import psycopg
    _, reader = pools                        # opened with timeout_ms=1000
    start = time.perf_counter()
    with pytest.raises(psycopg.errors.QueryCanceled):
        with reader.connection() as conn:
            conn.execute("SELECT pg_sleep(10)")
    assert time.perf_counter() - start < 5


def test_an_index_from_another_model_is_refused_with_a_clear_message(store):
    from obslab.rag.pgvector import IndexMismatchError, PgVectorStore
    from obslab.rag.providers import HashingEmbeddings
    other_model = PgVectorStore(store.pool, HashingEmbeddings(dim=768))
    with pytest.raises(IndexMismatchError, match="another embedding model"):
        other_model.similarity_search_with_score("etcd backup")


def test_readiness_reports_the_model_mismatch(store):
    from obslab.app import check_index_meta
    from obslab.config import Settings
    meta, _ = store.status()
    assert check_index_meta(Settings(provider="fake"), meta) is None
    assert "run the ingest" in check_index_meta(Settings(provider="ollama", embed_model="nomic-embed-text"), meta)


def test_readers_never_see_a_half_built_index(pools, store, docs):
    """Rebuild twice (a small index, then the full one) while a reader counts rows non-stop:
    it must only ever see one complete index or the other, and never fail."""
    from obslab.rag.pgvector import rebuild
    from obslab.rag.providers import HashingEmbeddings
    writer, reader = pools
    seen, errors, stop = set(), [], threading.Event()

    def watch():
        while not stop.is_set():
            try:
                with reader.connection() as conn:
                    seen.add(conn.execute("SELECT count(*) FROM rag.chunks").fetchone()[0])
            except Exception as e:     # noqa: BLE001 - any error is a failure of the guarantee
                errors.append(repr(e))

    t = threading.Thread(target=watch)
    t.start()
    try:
        rebuild(writer, docs[:5], HashingEmbeddings(), META)
        deadline = time.time() + 5
        while 5 not in seen and time.time() < deadline:     # the reader saw the small index...
            time.sleep(0.01)
        rebuild(writer, docs, HashingEmbeddings(), META)     # ...then this one swaps under it
        time.sleep(0.2)
    finally:
        stop.set()
        t.join()
    assert not errors, errors[:3]
    assert seen == {5, len(docs)}, seen


def test_the_api_serves_from_pgvector(store, docs, monkeypatch):
    from obslab.app import build_components
    from obslab.config import Settings
    from obslab.telemetry import init_telemetry
    for key, value in (("PGHOST", "127.0.0.1"), ("PGPORT", "5432"), ("PGDATABASE", "obslab"),
                       ("PGUSER", "obslab_reader"), ("PGPASSWORD", READER.split(":")[2].split("@")[0])):
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OBSLAB_CHAT_PASSWORD", os.environ.get("OBSLAB_CHAT_PASSWORD", "chat-local"))
    settings = Settings(provider="fake", telemetry="none", min_score=0.2)
    tel = init_telemetry(settings)
    c = build_components(settings, tel)
    try:
        assert c.status() == {"ready": True, "store": "pgvector", "provider": "fake", "chunks": len(docs)}
        r = c.ask("How do taints and tolerations work?")
        assert r["route"] == "answered" and "kubernetes/taints-and-tolerations.md" in r["sources"]
    finally:
        c.close()
        tel.shutdown()


def test_the_pool_recovers_when_the_server_drops_connections(store):
    """Stands for a database restart: the server kills every API connection. The next query must
    succeed on a fresh connection, not fail once per dead connection left in the pool."""
    import psycopg
    admin = os.environ.get("PG_ADMIN_DSN", "postgresql://postgres:postgres-local@127.0.0.1:5432/obslab")
    for _ in range(3):                                   # fill the pool with a few connections
        store.similarity_search_with_score("etcd backup", k=1)
    with psycopg.connect(admin, autocommit=True) as conn:
        killed = conn.execute("SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity "
                              "WHERE usename = 'obslab_reader'").fetchone()[0]
    assert killed >= 1
    for _ in range(3):
        assert store.similarity_search_with_score("etcd backup", k=1)
