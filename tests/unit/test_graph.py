import pytest

from obslab.rag.graph import FALLBACK_ANSWER, declined, groundedness
from obslab.rag.providers import FakeChat

from conftest import metrics_by_name, spans_by_name


def test_in_domain_question_is_answered_with_sources(components):
    r = components.ask("What does cosine similarity measure between embedding vectors?")
    assert r["route"] == "answered"
    assert r["sources"] == ["vectors.md"] or "vectors.md" in r["sources"]
    assert "Cosine similarity" in r["answer"]
    assert r["groundedness"] > 0.8


def test_span_tree_workflow_nodes_and_clients(components, otel):
    _, spans, _ = otel
    components.ask("How do I back up etcd with a snapshot?")
    s = spans_by_name(spans)
    root = s["invoke_workflow rag"]
    assert root.parent is None
    for node in ("retrieve", "generate", "grade"):
        assert s[f"rag.node {node}"].parent.span_id == root.context.span_id
    retrieve = s["rag.node retrieve"]
    assert s["retrieval notes"].parent.span_id == retrieve.context.span_id
    assert s["embeddings fake-embed"].parent.span_id == s["retrieval notes"].context.span_id
    assert s["chat fake-chat"].parent.span_id == s["rag.node generate"].context.span_id
    trace_ids = {sp.context.trace_id for sp in spans.get_finished_spans()}
    assert len(trace_ids) == 1
    assert s["retrieval notes"].attributes["gen_ai.operation.name"] == "retrieval"
    assert s["retrieval notes"].attributes["gen_ai.retrieval.top_k"] == 3


def test_off_topic_rewrites_once_then_falls_back(components, otel):
    _, spans, reader = otel
    r = components.ask("Best chocolate cake recipe for a birthday party?")
    assert r["route"] == "fallback"
    assert r["answer"] == FALLBACK_ANSWER
    assert r["rewrites"] == 1
    names = [sp.name for sp in spans.get_finished_spans()]
    assert names.count("rag.node retrieve") == 2
    assert "rag.node rewrite" in names and "rag.node fallback" in names
    got = metrics_by_name(reader)
    assert got["rag.fallbacks"][0].value == 1
    assert got["rag.query.rewrites"][0].value == 1


def test_rag_metrics_are_recorded(components, otel):
    _, _, reader = otel
    components.ask("readiness probe endpoints traffic")
    got = metrics_by_name(reader)
    for name in ("rag.retrieval.documents", "rag.retrieval.top_score", "rag.answer.groundedness",
                 "rag.graph.node.duration", "gen_ai.invoke_workflow.duration", "gen_ai.client.operation.duration"):
        assert name in got, name
    nodes = {dict(p.attributes)["langgraph.node"] for p in got["rag.graph.node.duration"]}
    assert {"retrieve", "generate", "grade"} <= nodes
    ops = {dict(p.attributes)["gen_ai.operation.name"] for p in got["gen_ai.client.operation.duration"]}
    assert ops == {"chat", "embeddings"}


def test_groundedness_heuristic():
    assert groundedness("Etcd snapshots restore data", "etcd snapshot restore data directory") == 0.75
    assert groundedness("", "anything") == 0.0
    assert groundedness("completely unrelated banana", "etcd backup") == 0.0


class DecliningChat(FakeChat):
    """Finds the context insufficient, the way a real model sometimes does."""

    def _reply(self, messages):
        prompt = "\n".join(str(m.content) for m in messages)
        if "Context:" in prompt and "Follow-up:" not in prompt:
            return "I don't know. The provided context does not mention that."
        return super()._reply(messages)


def test_a_model_that_says_it_does_not_know_is_a_fallback(components, otel):
    _, _, reader = otel
    components.rag.llm = DecliningChat()
    first = components.ask("What does cosine similarity measure?")
    assert first["route"] == "fallback" and first["answer"].startswith("I don't know")
    assert first["sources"] == [] and first["groundedness"] == 0.0 and first["turn"] == 1
    got = metrics_by_name(reader)
    assert [(dict(p.attributes), p.value) for p in got["rag.fallbacks"]] == [({"reason": "model_declined"}, 1)]
    assert "rag.answer.groundedness" not in got                # nothing to score
    assert components.ask("And for ranking?", first["conversation_id"])["turn"] == 2     # the turn counted


@pytest.mark.parametrize("answer, expected", [
    ("I don't know. The provided context does not mention the NoExecute effect.", True),
    ("I don’t know.", True),
    ("  \"I do not know based on the context.\"", True),
    ("i don't know", True),
    ("Cosine similarity measures the angle between two vectors.", False),
    ("The notes say what to do when you don't know the node name.", False),
    ("I don't knowingly skip steps.", False),
    ("", False),
])
def test_declined_recognises_the_model_saying_it_does_not_know(answer, expected):
    assert declined(answer) is expected


def test_a_trace_says_which_prompts_and_which_index_answered(components, otel):
    from obslab.rag.graph import ANSWER, CONDENSE, PROMPT_VERSION, REWRITE, SYSTEM
    _, spans, _ = otel
    for chunk in components.rag.store.store.values():         # as the pgvector store does, from rag.index_meta
        chunk["metadata"]["index_id"] = "26c8aa843dc1"
    with components.telemetry.tracer.start_as_current_span("request"):
        components.ask("What does cosine similarity measure between embedding vectors?")
    by_name = spans_by_name(spans)
    assert by_name["request"].attributes["obslab.prompt.version"] == PROMPT_VERSION
    assert by_name["retrieval notes"].attributes["obslab.index.id"] == "26c8aa843dc1"
    assert len(PROMPT_VERSION) == 8 and all(text for text in (SYSTEM, ANSWER, CONDENSE, REWRITE))


def test_a_search_without_an_index_id_leaves_the_attribute_out(components, otel):
    _, spans, _ = otel
    components.ask("What does cosine similarity measure between embedding vectors?")
    assert "obslab.index.id" not in spans_by_name(spans)["retrieval notes"].attributes


def test_the_index_id_follows_the_model_the_chunking_and_the_text():
    from langchain_core.documents import Document

    from obslab.rag.corpus import index_id, split
    notes = [Document(page_content="A taint repels pods. " * 30, metadata={"source": "taints.md"})]
    base = index_id(split(notes), "nomic-embed-text")
    assert base == index_id(split(notes), "nomic-embed-text") and len(base) == 12
    edited = [Document(page_content="A taint repels a pod. " * 30, metadata={"source": "taints.md"})]
    moved = [Document(page_content=notes[0].page_content, metadata={"source": "other.md"})]
    others = {index_id(split(notes), "embeddinggemma:300m"), index_id(split(notes, 400, 60), "nomic-embed-text"),
              index_id(split(edited), "nomic-embed-text"), index_id(split(moved), "nomic-embed-text")}
    assert base not in others and len(others) == 4
