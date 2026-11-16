"""The evaluation on the golden set: the set itself, the retrieval metrics, the judgement of
answers, the comparison of two variants, and the gate. The last one is what stops a change that
makes the search worse: it runs with the fake embeddings on the sample notes, offline, and
compares with evals/baseline.json."""

import json
import logging
import re
from collections import Counter

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from test_repo_hygiene import ROOT

from obslab import cli, evaluation
from obslab.config import Settings
from obslab.evaluation import (
    Question,
    RetrievalReport,
    Row,
    compare,
    evaluate_answers,
    evaluate_retrieval,
    load_golden,
)
from obslab.rag import corpus
from obslab.rag.providers import HashingEmbeddings

GOLDEN = ROOT / "evals" / "golden.jsonl"
_DEFAULTS = Settings(provider="fake", telemetry="none")
CHUNK = (_DEFAULTS.chunk_size, _DEFAULTS.chunk_overlap)
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
    ([_line(id="q" * 65)], "`id` must be a string of 1 to 64"),         # an id is a field of a log event
    ([_line(id=7)], "`id` must be a string of 1 to 64"),
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
    """Returns, for each question, the (source, score) list written in the test. The text of a
    chunk is "text of <source>"."""

    def __init__(self, answers: dict[str, list[tuple[str, float]]]):
        self.answers = answers

    def similarity_search_with_score(self, query: str, k: int = 4):
        return [(Document(page_content=f"text of {s}", metadata={"source": s}), score)
                for s, score in self.answers[query]][:k]


QUESTIONS = [
    Question("first", "answerable", "q first", ("a.md",), style="direct", expect=(("of a.md",),)),
    # The note is found, through b.md, but the text that holds the answer is the one of a.md at
    # rank 4... which is in the context too. The second group asks for a chunk that never comes.
    Question("third", "answerable", "q third", ("a.md", "b.md"), style="direct",
             expect=(("of a.md",), ("of c.md", "of d.md"))),
    Question("below", "answerable", "q below", ("a.md",), style="keyword", expect=(("of a.md",),)),
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
        "facts_in_context": round(1 / 3, 4),            # 3 questions have terms; only `first` gets its text
        "no_answer_questions": 2, "no_answer_refused": 0.5}
    assert [rows[i].facts for i in ("first", "third", "below", "deep")] == [True, False, False, None]
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
    assert compare(before.scores(), _report(after).scores()) == {"questions": 6} | expected


def test_two_runs_on_different_questions_are_not_compared():
    with pytest.raises(ValueError, match="do not cover the same questions"):
        compare(_report({"a": 1, "b": 1}).scores(), _report({"a": 1}).scores())


# ── the gate ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_report() -> RetrievalReport:
    store = InMemoryVectorStore(HashingEmbeddings())
    store.add_documents(corpus.split(corpus.load_folder(NOTES), *CHUNK))
    return evaluate_retrieval(store, load_golden(GOLDEN, NOTES), k=4, min_score=0.2)


def test_the_search_on_the_sample_notes_is_not_below_the_baseline(sample_report):
    baseline = evaluation.read_baseline(BASELINE, "fake-embed")
    assert baseline is not None
    setup = {"k": 4, "min_score": 0.2, "chunk_size": CHUNK[0], "chunk_overlap": CHUNK[1],
             "golden": evaluation.fingerprint(GOLDEN)}
    assert {name: baseline.get(name) for name in setup} == setup, \
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
    assert re.search(r"mrr 0\.\d{4} is below the baseline 0\.9900", capsys.readouterr().err)
    monkeypatch.setenv("OBSLAB_MIN_SCORE", "0.3")              # the numbers are not comparable any more
    assert run("--check") == 1
    assert "recorded for another setup: min_score=0.3 (baseline: 0.2)" in capsys.readouterr().err


def test_two_runs_are_compared_from_their_files(eval_cli, capsys, tmp_path, monkeypatch):
    run, _ = eval_cli
    first = tmp_path / "first.json"
    assert run("--json", str(first)) == 0
    monkeypatch.setenv("OBSLAB_CHUNK_SIZE", "400")
    monkeypatch.setenv("OBSLAB_CHUNK_OVERLAP", "60")
    capsys.readouterr()
    assert run("--against", str(first)) == 0
    out = capsys.readouterr().out
    assert "chunks of 400 characters (60 shared)" in out
    assert f"against {first}: " in out and "(sign test p = " in out


def test_a_file_that_cannot_be_compared_is_refused_before_anything_runs(eval_cli, capsys, tmp_path, monkeypatch):
    run, _ = eval_cli
    first, old, other = tmp_path / "first.json", tmp_path / "old.json", tmp_path / "other.json"
    assert run("--json", str(first)) == 0
    old.write_text('{"metrics": {}}', encoding="utf-8")
    scores = json.loads(first.read_text(encoding="utf-8"))["scores"]
    other.write_text(json.dumps({"mode": "retrieval", "scores": dict(list(scores.items())[:-1])}), encoding="utf-8")
    monkeypatch.setattr(evaluation, "evaluate_answers", lambda *a, **k: pytest.fail("the models were called"))
    capsys.readouterr()
    for args, message in [
        (("--answers", "--against", str(first)), "is a retrieval run, this one is a answers run"),
        (("--answers", "--against", str(tmp_path / "missing.json")), "cannot be read (FileNotFoundError)"),
        (("--answers", "--against", str(old)), "not a --json file of this command"),
        (("--answers", "--check"), "--answers is reported, never gated"),
        (("--answers", "--update-baseline"), "--answers is reported, never gated"),
    ]:
        assert run(*args) == 2
        assert message in capsys.readouterr().err
    kept = tmp_path / "kept.json"                # another question set: found at the end, the run is kept
    assert run("--against", str(other), "--json", str(kept)) == 1
    assert "do not cover the same questions" in capsys.readouterr().err
    assert json.loads(kept.read_text(encoding="utf-8"))["mode"] == "retrieval"


# ── the answers ──────────────────────────────────────────────────────────────

def test_answers_are_judged_with_the_terms_the_source_and_the_route():
    asked = []

    def ask(question, conversation_id=None):
        asked.append((question, conversation_id))
        reply = {
            "q terms": {"route": "answered", "answer": "Running pods are EVICTED.", "sources": ["x.md"]},
            "q wrong": {"route": "answered", "answer": "Running pods stay.", "sources": ["a.md"]},
            "q source": {"route": "answered", "answer": "Anything.", "sources": ["a.md"]},
            "q declined": {"route": "fallback", "answer": "I don't know.", "sources": []},
            "q none": {"route": "fallback", "answer": "I don't know.", "sources": []},
            "q leaks": {"route": "answered", "answer": "Paris.", "sources": ["a.md"]},
            "q before": {"route": "answered", "answer": "", "sources": []},
            "q after": {"route": "answered", "answer": "It is throttled.", "sources": ["a.md"], "rewrites": 1,
                        "standalone_question": "q after, about q before"},
        }[question]
        return {"conversation_id": "c1", **reply}

    questions = [
        Question("terms", "answerable", "q terms", ("a.md",), expect=(("evict",),)),
        Question("wrong", "answerable", "q wrong", ("a.md",), expect=(("evict",),)),
        Question("source", "answerable", "q source", ("a.md",)),             # no terms: the source decides
        Question("declined", "answerable", "q declined", ("a.md",), expect=(("evict",),)),
        Question("boom", "answerable", "q boom", ("a.md",)),                 # not in the table: raises
        Question("none", "no_answer", "q none"),
        Question("leaks", "no_answer", "q leaks"),
        Question("after", "follow_up", "q after", ("a.md",), previous="q before", expect=(("throttl",),)),
        Question("orphan", "follow_up", "q after", ("a.md",), previous="q missing", expect=(("throttl",),)),
    ]
    seen = []
    report = evaluate_answers(ask, questions, progress=seen.append)
    rows = {r.id: r for r in report.rows}
    assert {i: r.right for i, r in rows.items()} == {
        "terms": True, "wrong": False, "source": True, "declined": False, "boom": False,
        "none": True, "leaks": False, "after": True, "orphan": False}
    assert (rows["boom"].route, rows["boom"].error) == ("error", "KeyError") and rows["terms"].right_source is False
    assert (rows["orphan"].route, rows["orphan"].seconds) == ("error", 0.0)   # the turn before failed: not asked
    assert asked[-3:-1] == [("q before", None), ("q after", "c1")]   # the turn before, then the follow-up in it
    assert rows["after"].standalone == "q after, about q before" and rows["after"].rewrites == 1
    assert [r.id for r in seen] == [q.id for q in questions]
    metrics = report.metrics()
    assert {name: metrics[name] for name in ("right", "answered", "right_source", "declined", "follow_up_right",
                                             "errors")} == {
        "right": 0.4, "answered": 0.6, "right_source": 0.4, "declined": 0.5, "follow_up_right": 0.5, "errors": 2}
    assert report.scores()["wrong"] == 0.0 and report.scores()["after"] == 1.0
    only_unanswerable = evaluate_answers(ask, questions[5:7]).metrics()         # no question with an answer
    assert (only_unanswerable["right"], only_unanswerable["answered"], only_unanswerable["declined"]) == (0.0, 0.0, 0.5)


def test_the_command_asks_every_question_to_the_models(eval_cli, capsys, tmp_path):
    run, baseline = eval_cli
    out_file = tmp_path / "answers.json"
    assert run("--answers", "--json", str(out_file)) == 0
    assert not baseline.exists()
    out = capsys.readouterr().out
    assert "answers     fake-chat" in out and "follow-up right" in out
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written["mode"] == "answers" and written["config"]["chat_model"] == "fake-chat"
    assert len(written["rows"]) == 121 and set(written["scores"].values()) <= {0.0, 1.0}
    assert all(row["answer"] for row in written["rows"] if row["route"] == "answered")


# ── results as events ────────────────────────────────────────────────────────

@pytest.fixture
def eval_events():
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            ctx = getattr(record, "otelTraceID", None)
            records.append(({k: v for k, v in vars(record).items() if k.startswith(("gen_ai.", "obslab.", "rag."))
                             or k == "event.name"}, ctx))
    handler, logger = Keep(logging.INFO), logging.getLogger("obslab")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


def test_a_retrieval_run_is_logged_question_by_question_with_what_produced_it(eval_cli, eval_events):
    run, _ = eval_cli
    assert run() == 0
    events = [fields for fields, _ in eval_events]
    results = [e for e in events if e["event.name"] == "gen_ai.evaluation.result"]
    assert len(results) == 99 and {e["gen_ai.evaluation.name"] for e in results} == {"retrieval"}
    one = next(e for e in results if e["obslab.eval.question"] == "etcd-03")
    assert one["gen_ai.evaluation.score.value"] == 1.0 and one["gen_ai.evaluation.score.label"] == "hit"
    # The score and the label describe the same thing: the note is in the first 4, or not.
    assert {e["gen_ai.evaluation.score.label"]: e["gen_ai.evaluation.score.value"] >= 0.25 for e in results} == {
        "hit": True, "miss": False}
    (finished,) = [e for e in events if e["event.name"] == "eval.finished"]
    committed = json.loads(BASELINE.read_text(encoding="utf-8"))["fake-embed"]
    assert finished == finished | {
        "obslab.eval.mode": "retrieval", "obslab.eval.questions": 99, "obslab.eval.golden": committed["golden"],
        "obslab.eval.recall_at_k": committed["recall_at_k"], "obslab.eval.mrr": committed["mrr"],
        "obslab.embed_model": "fake-embed", "rag.retrieval.top_k": 4, "rag.retrieval.min_score": 0.2}
    assert len(finished["obslab.index.id"]) == 12 and len(finished["obslab.eval.run"]) == len("20261116T101500-") + 6
    assert {e["obslab.eval.run"] for e in events} == {finished["obslab.eval.run"]}       # one id ties the run


def test_an_answers_run_logs_each_verdict_in_the_trace_of_its_answer(eval_cli, eval_events, monkeypatch):
    from opentelemetry import trace

    from obslab.rag.graph import PROMPT_VERSION
    traces = []
    real = logging.getLogger("obslab.events").log

    def log(level, event, extra):                       # the trace that is current when the event is written
        traces.append((event, trace.get_current_span().get_span_context().trace_id))
        real(level, event, extra=extra)
    monkeypatch.setattr(logging.getLogger("obslab.events"), "log", log)
    run, _ = eval_cli
    assert run("--answers") == 0
    events = [fields for fields, _ in eval_events]
    results = [e for e in events if e["event.name"] == "gen_ai.evaluation.result"]
    assert len(results) == 121 and {e["gen_ai.evaluation.name"] for e in results} == {"answer"}
    assert {e["gen_ai.evaluation.score.label"] for e in results} == {"right", "wrong"}
    (finished,) = [e for e in events if e["event.name"] == "eval.finished"]
    assert finished == finished | {"obslab.eval.mode": "answers", "gen_ai.request.model": "fake-chat",
                                   "obslab.prompt.version": PROMPT_VERSION, "obslab.eval.questions": 121}
    verdict_traces = [t for event, t in traces if event == "gen_ai.evaluation.result"]
    assert all(verdict_traces) and len(set(verdict_traces)) == 121       # one trace per question, never none
