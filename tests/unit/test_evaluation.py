"""The retrieval evaluation: the golden set itself, the metrics, the comparison of two variants,
and the gate. The last one is what stops a change that makes the search worse: it runs with the
fake embeddings on the sample notes, offline, and compares with evals/baseline.json."""

import json
from collections import Counter

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from test_repo_hygiene import ROOT

from obslab import cli, evaluation
from obslab.evaluation import Question, RetrievalReport, Row, compare, evaluate_retrieval, load_golden
from obslab.rag import corpus
from obslab.rag.providers import HashingEmbeddings

GOLDEN = ROOT / "evals" / "golden.jsonl"
BASELINE = ROOT / "evals" / "baseline.json"
NOTES = ROOT / "datalake"


# ── the golden set ───────────────────────────────────────────────────────────

def test_the_golden_set_covers_every_note_and_the_three_kinds_of_question():
    questions = load_golden(GOLDEN, NOTES)
    kinds = Counter(q.kind for q in questions)
    assert kinds["answerable"] >= 60                 # fewer, and a difference between two variants is noise
    assert kinds["no_answer"] >= 10 and kinds["follow_up"] >= 8
    per_note = Counter(s for q in questions if q.kind == "answerable" for s in q.sources)
    notes = {f.relative_to(NOTES).as_posix() for f in NOTES.rglob("*.md")}
    assert set(per_note) == notes
    assert min(per_note.values()) >= 4, per_note
    # Both ways of having no answer: nothing to do with the notes, and close to them.
    assert {q.style for q in questions if q.kind == "no_answer"} == {"off_topic", "near_domain"}


def _line(**fields) -> str:
    return json.dumps({"id": "q1", "kind": "answerable", "question": "What is a taint?",
                       "sources": ["kubernetes/taints-and-tolerations.md"]} | fields)


@pytest.mark.parametrize("lines,message", [
    ([_line(), _line()], "duplicate id"),
    ([_line(kind="easy")], "kind 'easy'"),
    ([_line(question=" ")], "empty question"),
    ([_line(sources=[])], "has no source"),
    ([_line(kind="no_answer")], "has no source"),
    ([_line(kind="follow_up")], "`previous`"),
    ([_line(previous="What is a node?")], "`previous`"),
    ([_line(expect=[[]])], "`expect` must be"),
    ([_line(expect=["evict", "taint"])], "`expect` must be"),       # a flat list would be read letter by letter
    ([_line(sources="kubernetes/taints-and-tolerations.md")], "`sources` must be"),
    (["{not json"], "line 1: not a JSON object"),
    (['{"kind": "answerable", "question": "no id"}'], "line 1: not a JSON object"),
    ([_line(sources=["kubernetes/no-such-note.md"])], "no such note"),
    ([""], "no question"),
])
def test_a_malformed_golden_set_is_refused_with_the_line(tmp_path, lines, message):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_golden(path, NOTES)


def test_the_fingerprint_does_not_depend_on_line_endings(tmp_path):
    unix, windows = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    unix.write_bytes(b'{"id": "q1"}\n')
    windows.write_bytes(b'{"id": "q1"}\r\n')
    assert evaluation.fingerprint(unix) == evaluation.fingerprint(windows)


# ── the metrics ──────────────────────────────────────────────────────────────

class FixedStore:
    """Returns, for each question, the (source, score) list written in the test."""

    def __init__(self, answers: dict[str, list[tuple[str, float]]]):
        self.answers = answers

    def similarity_search_with_score(self, query: str, k: int = 4):
        return [(Document(page_content="", metadata={"source": s}), score) for s, score in self.answers[query]][:k]


QUESTIONS = [
    Question("first", "answerable", "q first", ("a.md",), style="direct"),
    Question("third", "answerable", "q third", ("a.md", "b.md"), style="direct"),
    Question("below", "answerable", "q below", ("a.md",), style="keyword"),
    Question("deep", "answerable", "q deep", ("a.md",), style="keyword"),
    Question("refused", "no_answer", "q refused"),
    Question("leaks", "no_answer", "q leaks"),
    Question("later", "follow_up", "q later", ("a.md",), previous="q first"),
]
OTHER = [("x.md", 0.9), ("y.md", 0.8)]
STORE = FixedStore({
    "q first": [("a.md", 0.9), ("x.md", 0.5)],
    "q third": [*OTHER, ("b.md", 0.7), ("a.md", 0.6)],
    "q below": [("a.md", 0.4), ("x.md", 0.3)],                   # found, but under the threshold
    "q deep": [*OTHER, ("z.md", 0.7), ("w.md", 0.7), ("a.md", 0.65)],   # found at rank 5, k is 4
    "q refused": [("x.md", 0.2)],
    "q leaks": [("x.md", 0.8)],
})


def test_ranks_threshold_and_refusals_are_counted():
    report = evaluate_retrieval(STORE, QUESTIONS, k=4, min_score=0.5)
    rows = {r.id: r for r in report.rows}
    assert "later" not in rows                                   # a follow-up needs a model first
    assert [rows[i].rank for i in ("first", "third", "below", "deep")] == [1, 3, 1, 5]
    assert [rows[i].in_context for i in ("first", "third", "below", "deep")] == [True, True, False, False]
    assert report.metrics() == {
        "questions": 4, "recall_at_1": 0.5, "recall_at_k": 0.75,
        "mrr": round((1 + 1 / 3 + 1 + 1 / 5) / 4, 4), "in_context": 0.5,
        "no_answer_questions": 2, "no_answer_refused": 0.5}
    assert [r.id for r in report.misses()] == ["below", "deep"]
    assert report.by_style() == {"direct": {"questions": 2, "recall_at_k": 1.0, "mrr": round((1 + 1 / 3) / 2, 4)},
                                 "keyword": {"questions": 2, "recall_at_k": 0.5, "mrr": 0.6}}


def test_a_note_missing_from_the_results_counts_for_zero():
    store = FixedStore({"q first": OTHER})
    report = evaluate_retrieval(store, QUESTIONS[:1], k=4, min_score=0.5)
    assert report.rows[0].rank is None and report.metrics()["mrr"] == 0.0


def test_recall_at_k_is_not_capped_when_k_is_larger_than_the_depth_of_the_reciprocal_rank():
    deep = [(f"x{i}.md", 0.9 - i / 100) for i in range(11)] + [("a.md", 0.7)]      # the note comes 12th
    report = evaluate_retrieval(FixedStore({"q first": deep}), QUESTIONS[:1], k=15, min_score=0.5)
    assert report.rows[0].rank == 12
    assert report.metrics() | {"questions": 1} == report.metrics()
    assert (report.metrics()["recall_at_k"], report.metrics()["in_context"], report.metrics()["mrr"]) == (1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="k=0"):
        evaluate_retrieval(STORE, QUESTIONS, k=0, min_score=0.5)


def _report(ranks: dict[str, int | None]) -> RetrievalReport:
    return RetrievalReport(4, 0.5, tuple(Row(i, "answerable", "direct", rank, bool(rank), 1, 0.9)
                                         for i, rank in ranks.items()))


@pytest.mark.parametrize("after,expected", [
    ({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1},        # 5 better, none worse: 2 * (1/2)^5
     {"better": 5, "worse": 0, "same": 1, "p_value": 0.0625}),
    ({"a": 1, "b": 3, "c": 2, "d": 2, "e": 2, "f": 1},        # one each way: no evidence at all
     {"better": 1, "worse": 1, "same": 4, "p_value": 1.0}),
    ({"a": 2, "b": 2, "c": 2, "d": 2, "e": 2, "f": 1},
     {"better": 0, "worse": 0, "same": 6, "p_value": 1.0}),
])
def test_two_variants_are_compared_question_by_question(after, expected):
    before = _report({"a": 2, "b": 2, "c": 2, "d": 2, "e": 2, "f": 1})
    assert compare(before, _report(after)) == {"questions": 6} | expected


def test_two_runs_on_different_questions_are_not_compared():
    with pytest.raises(ValueError, match="do not cover the same questions"):
        compare(_report({"a": 1, "b": 1}), _report({"a": 1}))


# ── the gate ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_report() -> RetrievalReport:
    store = InMemoryVectorStore(HashingEmbeddings())
    store.add_documents(corpus.split(corpus.load_folder(NOTES)))
    return evaluate_retrieval(store, load_golden(GOLDEN, NOTES), k=4, min_score=0.2)


def test_the_search_on_the_sample_notes_is_not_below_the_baseline(sample_report):
    baseline = evaluation.read_baseline(BASELINE, "fake-embed")
    assert baseline is not None
    assert (baseline["k"], baseline["min_score"], baseline["golden"]) == (4, 0.2, evaluation.fingerprint(GOLDEN)), \
        "the golden set or the setup changed: task eval UPDATE=1, and commit evals/baseline.json"
    assert evaluation.check(sample_report.metrics(), baseline) == []


def test_a_drop_of_a_gated_metric_is_reported(sample_report):
    metrics = sample_report.metrics()
    worse = metrics | {"recall_at_k": metrics["recall_at_k"] - 0.02, "no_answer_refused": 0.0}
    problems = evaluation.check(worse, metrics)
    assert len(problems) == 1 and problems[0].startswith("recall_at_k ")       # `refused` is reported, not gated
    older = {name: value for name, value in metrics.items() if name != "in_context"}
    assert evaluation.check(metrics, older) == ["in_context is not in the baseline: record it again"]


@pytest.fixture
def eval_cli(monkeypatch, tmp_path):
    for name, value in {"OBSLAB_PROVIDER": "fake", "OBSLAB_MIN_SCORE": "0.2", "OBSLAB_TOP_K": "4",
                        "OBSLAB_TELEMETRY": "none"}.items():
        monkeypatch.setenv(name, value)
    baseline = tmp_path / "baseline.json"

    def run(*args: str) -> int:
        return cli.main(["eval", "--golden", str(GOLDEN), "--source", str(NOTES), "--baseline", str(baseline), *args])
    return run, baseline


def test_the_command_records_a_baseline_then_passes_against_it(eval_cli, capsys, tmp_path):
    run, baseline = eval_cli
    assert run("--check") == 1 and "no baseline for fake-embed" in capsys.readouterr().err
    assert run("--update-baseline") == 0
    assert json.loads(baseline.read_text(encoding="utf-8")) == {
        "fake-embed": json.loads(BASELINE.read_text(encoding="utf-8"))["fake-embed"]}     # the committed one is current
    rows = tmp_path / "rows.json"
    assert run("--check", "--json", str(rows)) == 0
    assert "baseline of fake-embed: pass" in capsys.readouterr().out
    written = json.loads(rows.read_text(encoding="utf-8"))
    assert written["config"]["embed_model"] == "fake-embed" and len(written["rows"]) == 99 + 12


def test_the_command_fails_below_the_baseline_and_on_another_setup(eval_cli, capsys, monkeypatch):
    run, baseline = eval_cli
    assert run("--update-baseline") == 0
    data = json.loads(baseline.read_text(encoding="utf-8"))
    data["fake-embed"]["mrr"] = 0.99
    baseline.write_text(json.dumps(data), encoding="utf-8")
    capsys.readouterr()
    assert run("--check") == 1
    assert "mrr 0.8333 is below the baseline 0.9900" in capsys.readouterr().err
    monkeypatch.setenv("OBSLAB_MIN_SCORE", "0.3")              # the numbers are not comparable any more
    assert run("--check") == 1
    assert "recorded for another setup: min_score=0.3 (baseline: 0.2)" in capsys.readouterr().err
