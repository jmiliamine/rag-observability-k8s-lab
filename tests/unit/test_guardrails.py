"""Guardrails that do not need a database: settings validation, what /readyz and a failed /ask
reveal to a client when the vector database misbehaves."""

import pytest
from fastapi.testclient import TestClient

from obslab.api import create_app
from obslab.config import Settings
from obslab.rag.pgvector import MAX_K, PgVectorStore

SECRET = "postgresql://obslab_reader:s3cret@db.internal:5432/obslab"


@pytest.mark.parametrize("kwargs,message", [
    ({"top_k": 0}, "OBSLAB_TOP_K"),
    ({"top_k": MAX_K + 1}, "OBSLAB_TOP_K"),
    ({"min_score": 1.5}, "OBSLAB_MIN_SCORE"),
    ({"provider": "openai"}, "OBSLAB_PROVIDER"),
    ({"db_timeout_ms": 0}, "OBSLAB_DB_TIMEOUT_MS"),
])
def test_invalid_settings_fail_at_startup(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Settings(**kwargs)


def test_each_invalid_setting_is_reported_once():
    with pytest.raises(ValueError) as error:
        Settings(history_turns=99, max_turns=0, conversation_ttl_days=0)
    for name in ("OBSLAB_HISTORY_TURNS", "OBSLAB_MAX_TURNS", "OBSLAB_CONVERSATION_TTL_DAYS"):
        assert str(error.value).count(name) == 1


class BrokenStore(PgVectorStore):
    """A pgvector store whose database is down; errors carry connection details on purpose."""

    def __init__(self, embedding):
        super().__init__(pool=None, embedding=embedding)

    def status(self):
        raise ConnectionError(f"connection failed: {SECRET}")

    def similarity_search_with_score(self, query, k=4, **kwargs):
        raise ConnectionError(f"connection failed: {SECRET}")


@pytest.fixture
def broken(components):
    components.rag.store = BrokenStore(components.rag.store.embedding)
    components.graph = components.rag.build()
    return components


def test_readiness_fails_without_leaking_connection_details(broken):
    r = TestClient(create_app(broken)).get("/readyz")
    assert r.status_code == 503
    assert r.json()["error"] == "vector database unreachable (ConnectionError)"
    assert "s3cret" not in r.text and "db.internal" not in r.text


def test_a_failed_question_returns_the_error_type_only(broken):
    r = TestClient(create_app(broken)).post("/ask", json={"question": "How do taints work?"})
    assert r.status_code == 502
    assert r.json()["detail"] == "upstream error: ConnectionError"
    assert "s3cret" not in r.text


def test_questions_are_bounded(components):
    client = TestClient(create_app(components))
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
