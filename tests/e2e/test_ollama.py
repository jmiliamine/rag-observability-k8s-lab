"""The RAG with the real local models: nomic-embed-text for retrieval, llama3.2:3b for answers.

Checks what the fakes cannot: the calibrated threshold (OBSLAB_MIN_SCORE=0.6) still separates
in-domain from off-topic questions on the sample notes, and the model answers from the context.
Run with `task test:ollama` (Ollama running, both models pulled). Telemetry is not exported.
"""

import dataclasses
from pathlib import Path

import pytest

from lab_http import is_up, unavailable

pytestmark = pytest.mark.ollama

NOTES = Path(__file__).resolve().parents[2] / "datalake"


@pytest.fixture(scope="module")
def rag():
    from langchain_core.vectorstores import InMemoryVectorStore

    from obslab.app import build_components
    from obslab.config import Settings
    from obslab.rag import corpus
    from obslab.rag.providers import embeddings
    from obslab.telemetry import GenAIMetrics, init_telemetry

    settings = dataclasses.replace(Settings(), provider="ollama", telemetry="none")
    if not is_up(f"{settings.ollama_url}/api/version"):
        unavailable(f"Ollama not reachable at {settings.ollama_url}")
    tel = init_telemetry(settings)
    docs = corpus.split(corpus.load_folder(NOTES))
    store = InMemoryVectorStore(embeddings(settings, tel, GenAIMetrics(tel.meter)))
    store.add_documents(docs)
    yield build_components(settings, tel, store=store)
    tel.shutdown()


def test_in_domain_question_is_answered_from_the_right_note(rag):
    r = rag.ask("How do taints and tolerations work?")
    assert r["route"] == "answered", r
    assert "kubernetes/taints-and-tolerations.md" in r["sources"]
    assert "toleration" in r["answer"].lower()


def test_a_follow_up_is_understood_from_the_previous_turn(rag):
    first = rag.ask("How do taints and tolerations work?")
    second = rag.ask("And what does the NoExecute effect do?", first["conversation_id"])
    assert second["turn"] == 2
    # Whether the question was rewritten is not asserted: a real model leaves a follow-up as it is
    # when it already names its subject. The rewriting itself is tested with the fake model.
    assert second["route"] == "answered", second
    assert "kubernetes/taints-and-tolerations.md" in second["sources"]
    assert "evict" in second["answer"].lower(), second["answer"]


def test_a_new_subject_in_the_same_conversation_is_not_dragged_back(rag):
    first = rag.ask("How do taints and tolerations work?")
    second = rag.ask("How do you back up and restore etcd?", first["conversation_id"])
    assert second["route"] == "answered" and "kubernetes/etcd-backup.md" in second["sources"], second


def test_off_topic_question_falls_back(rag):
    assert rag.ask("What is the best recipe for a chocolate cake?")["route"] == "fallback"
