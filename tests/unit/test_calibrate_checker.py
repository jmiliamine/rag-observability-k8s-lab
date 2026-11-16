"""The calibration of a grounding checker: the labelled answers, the bar, and how a model's
Yes or No becomes a score. The checkers themselves are models: they are not run here."""

import importlib.util
import json
from collections import Counter

import pytest
from test_repo_hygiene import ROOT

from obslab.evaluation import load_golden

SCRIPT = ROOT / "scripts" / "calibrate_checker.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("calibrate_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_labelled_answers_are_well_formed_and_tied_to_the_golden_set(checker):
    golden = {q.id: q for q in load_golden(ROOT / "evals" / "golden.jsonl", ROOT / "datalake")}
    rows = [json.loads(line) for line in checker.ANSWERS.read_text(encoding="utf-8").splitlines()]
    assert len({r["id"] for r in rows}) == len(rows)
    labels = Counter(r["label"] for r in rows)
    assert set(labels) == {"correct", "wrong"} and labels["wrong"] >= 25 and labels["correct"] >= 75
    for r in rows:
        question = golden[r["question_id"]]
        assert question.kind != "no_answer" and r["sources"] == list(question.sources), r["id"]
        assert r["answer"].strip() and all((checker.NOTES / s).is_file() for s in r["sources"]), r["id"]
    # Not one question repeated: no question gives more than two answers of the same label.
    assert max(Counter((r["question_id"], r["label"]) for r in rows).values()) <= 2


def test_the_bar_is_four_wrong_answers_in_five_and_two_false_alarms_in_twenty_three(checker):
    wrong, correct = [0.1] * 4 + [0.9], [0.9] * 21 + [0.1] * 2          # exactly at the bar
    assert checker.verdict(wrong, correct)[0] is True
    assert checker.verdict([0.1] * 3 + [0.9] * 2, correct)[0] is False           # one wrong answer too few
    assert checker.verdict(wrong, [0.9] * 20 + [0.1] * 3)[0] is False            # one false alarm too many
    # A checker that separates at another threshold passes too; one that scores everything alike never does.
    assert checker.verdict([0.55] * 5, [0.65] * 23)[0] is True
    assert checker.verdict([0.5] * 5, [0.5] * 23)[0] is False


def test_the_table_counts_what_is_flagged_under_each_threshold(checker):
    _, table = checker.verdict([0.1, 0.3, 0.8], [0.2, 0.9])
    by_threshold = {threshold: (caught, alarms) for threshold, caught, alarms in table}
    assert by_threshold[0.25] == (1, 1) and by_threshold[0.5] == (2, 1) and by_threshold[1.0] == (3, 2)


@pytest.mark.parametrize("reply,expected", [
    # log-probabilities of the first token: the share of Yes between Yes and No
    ({"message": {"content": "Yes"}, "logprobs": [{"token": "Yes", "logprob": -0.1, "top_logprobs": [
        {"token": "Yes", "logprob": -0.105}, {"token": "No", "logprob": -2.303}]}]}, 0.9),
    ({"message": {"content": " No"}, "logprobs": [{"token": " No", "logprob": -0.1, "top_logprobs": [
        {"token": " No", "logprob": -0.105}, {"token": " Yes", "logprob": -2.303}]}]}, 0.1),
    # one word, several tokens: "Yes", " Yes" and "yes" add up, the rare spelling does not replace the common one
    ({"message": {"content": "Yes"}, "logprobs": [{"token": "Yes", "logprob": -0.22, "top_logprobs": [
        {"token": "Yes", "logprob": -0.223}, {"token": "No", "logprob": -1.609}, {"token": " Yes", "logprob": -11.4},
        {"token": "The", "logprob": -11.8}, {"token": "yes", "logprob": -12.0}]}]}, 0.8),
    # a server that returns no log-probabilities: the text decides
    ({"message": {"content": "Yes."}}, 1.0),
    ({"message": {"content": "no"}}, 0.0),
    ({"message": {"content": "Maybe"}, "logprobs": []}, 0.0),
])
def test_a_yes_or_no_reply_becomes_a_score(checker, reply, expected):
    assert checker._yes_probability(reply, "yes", "no") == pytest.approx(expected, abs=0.005)


def test_an_unknown_checker_is_refused(checker):
    with pytest.raises(SystemExit, match="unknown checker"):
        checker.scorer("vibes", "http://localhost:11434")


def test_a_run_is_read_only_where_an_answer_was_given_and_a_note_exists(checker):
    run = {"rows": [
        {"id": "a-1", "route": "answered", "answer": "Use etcdctl snapshot save.", "right": True},
        {"id": "a-2", "route": "fallback", "answer": "I don't know.", "right": False},      # declined: nothing to check
        {"id": "na-1", "route": "answered", "answer": "Paris.", "right": False},             # no note for this question
        {"id": "a-3", "route": "answered", "answer": "", "right": False},                    # failed call
    ]}
    rows = checker.to_check(run, {"a-1": ["k/etcd.md"], "a-2": ["k/etcd.md"], "na-1": [], "a-3": ["k/etcd.md"]})
    assert [(r["id"], r["sources"]) for r in rows] == [("a-1", ["k/etcd.md"])]


def test_the_flagged_answers_are_under_the_threshold_least_supported_first(checker):
    rows = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.2}, {"id": "c", "score": 0.49}, {"id": "d", "score": 0.5}]
    assert [r["id"] for r in checker.flagged(rows, 0.5)] == ["b", "c"]
    assert checker.flagged(rows, 0.1) == []


def test_a_run_file_is_scored_against_the_notes_of_the_golden_set(checker, tmp_path, capsys):
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"rows": [
        {"id": "etcd-01", "route": "answered", "answer": "With etcdctl snapshot save.", "right": True},
        {"id": "etcd-02", "route": "answered", "answer": "It goes to the moon.", "right": False},
        {"id": "etcd-03", "route": "fallback", "answer": "I don't know.", "right": False},
    ]}), encoding="utf-8")
    seen = []

    def score(note, answer):                    # a checker that knows one sentence
        seen.append(note)
        return 0.05 if "moon" in answer else 0.95

    assert checker.check_run("stub", score, run, 0.5) == 0
    out = capsys.readouterr().out
    assert "2 answers read against their note, 1 flagged" in out
    assert "0 that the term check counts right, 1 wrong" in out
    assert "0.05  wrong  etcd-02" in out and "etcd-01" not in out
    assert len(seen) == 2 and all("etcd" in note.lower() for note in seen)       # the whole note of the question


def test_the_command_line_calibrates_by_default_and_reads_a_run_on_request(checker, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(checker, "scorer", lambda name, url: calls.append((name, url)) or "scorer")
    monkeypatch.setattr(checker, "calibrate", lambda name, score: calls.append(("calibrate", name, score)) or 1)
    monkeypatch.setattr(checker, "check_run", lambda name, score, path, under: calls.append(("run", path, under)) or 0)
    assert checker.main(["ollama:m"]) == 1                 # the exit code says whether the bar is passed
    assert calls == [("ollama:m", "http://127.0.0.1:11434"), ("calibrate", "ollama:m", "scorer")]
    calls.clear()
    run = str(tmp_path / "r.json")
    assert checker.main(["ollama:m", "--run", run, "--threshold", "0.3", "--url", "http://h:1"]) == 0
    assert calls == [("ollama:m", "http://h:1"), ("run", tmp_path / "r.json", 0.3)]
    calls.clear()
    assert checker.main(["ollama:m", "--run", str(tmp_path / "r.json")]) == 0 and calls[-1][-1] == checker.FLAG_UNDER
