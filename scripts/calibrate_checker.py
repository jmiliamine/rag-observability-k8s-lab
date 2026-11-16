"""Calibrate a grounding checker on the labelled answers (evals/answers.jsonl).

A grounding checker reads a note and an answer and says whether the note supports the answer.
Before one is trusted, it is measured on answers whose label is known: 111 answers of the
answering model to questions of the golden set, read by two readers and kept only where they
agree (82 correct, 29 wrong).

The bar was fixed before any run, when the answering model was tried as its own judge: catch at
least 4 wrong answers in 5 with at most 2 false alarms in 23 correct answers. On this set: at
least 24 of the 29 wrong answers, at most 7 of the 82 correct ones. A checker that has no
threshold meeting both is not used.

    python scripts/calibrate_checker.py ollama:bespoke-minicheck
    python scripts/calibrate_checker.py ollama:granite3-guardian:2b
    uv run --with torch --with transformers --with sentencepiece --with protobuf \\
        python scripts/calibrate_checker.py hhem

The checkers are not dependencies of the project: the Ollama ones are models to pull, the
Vectara one needs PyTorch, installed for this run only by the `uv run --with` line.

A checker that passes can then read the answers of an evaluation run, saved by
`obslab eval --answers --json run.json`, and list the ones the note does not support:

    python scripts/calibrate_checker.py ollama:bespoke-minicheck --run run.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES, ANSWERS, GOLDEN = ROOT / "datalake", ROOT / "evals" / "answers.jsonl", ROOT / "evals" / "golden.jsonl"
CATCH, ALARMS = 4 / 5, 2 / 23                   # the bar: share of wrong caught, share of correct flagged
# Where bespoke-minicheck passes the bar on the labelled answers (25 of 29 wrong caught, 6 of 82
# correct flagged). Another checker needs its own value: --threshold.
FLAG_UNDER = 0.5
# Pinned: the model card says to load it with trust_remote_code. The code at this revision is a
# 70-line wrapper around T5ForTokenClassification (modeling_hhem_v2.py), read before use.
HHEM_REVISION = "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8"

Scorer = Callable[[str, str], float]            # (note, answer) -> score in [0, 1], 1 = supported


def hhem() -> Scorer:
    """Vectara HHEM-2.1-open: a 0.1B classifier, on CPU."""
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model", revision=HHEM_REVISION, trust_remote_code=True)
    return lambda note, answer: float(model.predict([(note, answer)])[0])


def _ollama(url: str, body: dict) -> dict:
    request = urllib.request.Request(f"{url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as reply:
        return json.loads(reply.read())


def _yes_probability(reply: dict, yes: str, no: str) -> float:
    """Probability of the `yes` word, from the log-probabilities of the first token when the
    server returns them, else 1 or 0 from the text."""
    text = reply["message"]["content"].strip().lower()
    for token in reply.get("logprobs") or []:
        # "Yes", "yes" and " Yes" are three tokens for one word: their probabilities add up.
        mass = {yes: 0.0, no: 0.0}
        for option in token.get("top_logprobs", []):
            word = option["token"].strip().lower()
            if word in mass:
                mass[word] += math.exp(option["logprob"])
        if mass[yes] + mass[no] > 0:
            return mass[yes] / (mass[yes] + mass[no])
    return 1.0 if text.startswith(yes) else 0.0


def minicheck(model: str, url: str) -> Scorer:
    """Bespoke-MiniCheck: "Document: ... Claim: ..." answered by Yes or No. Each sentence of the
    answer is one claim, as the model was trained; the answer gets its least supported sentence."""
    def score(note: str, answer: str) -> float:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15] or [answer]
        return min(_yes_probability(_ollama(url, {
            "model": model, "stream": False, "logprobs": True, "top_logprobs": 5,
            "options": {"temperature": 0, "num_ctx": 4096},
            "messages": [{"role": "user", "content": f"Document: {note}\nClaim: {s}"}]}), "yes", "no")
            for s in sentences)
    return score


def guardian(model: str, url: str) -> Scorer:
    """IBM Granite Guardian, groundedness risk: answers Yes when the reply is NOT grounded."""
    def score(note: str, answer: str) -> float:
        reply = _ollama(url, {
            "model": model, "stream": False, "logprobs": True, "top_logprobs": 5,
            "options": {"temperature": 0, "num_ctx": 4096},
            "messages": [{"role": "system", "content": "groundedness"},
                         {"role": "context", "content": note},
                         {"role": "assistant", "content": answer}]})
        return 1.0 - _yes_probability(reply, "yes", "no")
    return score


def scorer(name: str, url: str) -> Scorer:
    if name == "hhem":
        return hhem()
    if name.startswith("ollama:"):
        model = name.removeprefix("ollama:")
        return guardian(model, url) if "guardian" in model else minicheck(model, url)
    raise SystemExit(f"unknown checker {name!r}: hhem, or ollama:<model>")


def verdict(wrong: list[float], correct: list[float]) -> tuple[bool, list[tuple[float, int, int]]]:
    """(passes the bar, [(threshold, wrong caught, false alarms)]): an answer is flagged when
    its score is under the threshold."""
    table = [(t / 20, sum(s < t / 20 for s in wrong), sum(s < t / 20 for s in correct)) for t in range(1, 21)]
    ok = any(caught >= math.ceil(CATCH * len(wrong)) and alarms <= math.floor(ALARMS * len(correct))
             for _, caught, alarms in table)
    return ok, table


def note(sources: list[str]) -> str:
    return "\n\n".join((NOTES / s).read_text(encoding="utf-8") for s in sources)


def calibrate(name: str, score: Scorer) -> int:
    """Scores the labelled answers and says whether the checker passes the bar."""
    rows = [json.loads(line) for line in ANSWERS.read_text(encoding="utf-8").splitlines() if line.strip()]
    times = []
    for row in rows:
        start = time.perf_counter()
        row["score"] = score(note(row["sources"]), row["answer"])
        times.append(time.perf_counter() - start)
    wrong = [r["score"] for r in rows if r["label"] == "wrong"]
    correct = [r["score"] for r in rows if r["label"] == "correct"]
    ok, table = verdict(wrong, correct)
    times.sort()
    need, allowed = math.ceil(CATCH * len(wrong)), math.floor(ALARMS * len(correct))
    print(f"{name}: {len(correct)} correct and {len(wrong)} wrong answers, "
          f"{times[len(times) // 2]:.2f} s per answer (p95 {times[int(len(times) * 0.95)]:.2f} s)")
    print(f"bar: at least {need} of {len(wrong)} wrong answers caught, at most {allowed} of {len(correct)} "
          "correct ones flagged\n\nflag under   wrong caught   correct flagged")
    seen = set()
    for threshold, caught, alarms in table:
        if (caught, alarms) not in seen:        # one line per distinct outcome
            seen.add((caught, alarms))
            print(f"   {threshold:.2f}        {caught:>2} of {len(wrong)}       {alarms:>2} of {len(correct)}")
    print(f"\n{'PASSES' if ok else 'FAILS'} the bar")
    return 0 if ok else 1


def to_check(run: dict, sources: dict[str, list[str]]) -> list[dict]:
    """The answers of a run that a checker can read: given (not "I don't know") to a question that
    has a note. An answer to a question without note is already counted by the run, as not declined."""
    return [row | {"sources": sources[row["id"]]} for row in run["rows"]
            if row.get("route") != "fallback" and row.get("answer") and sources.get(row["id"])]


def flagged(rows: list[dict], threshold: float) -> list[dict]:
    """The answers under the threshold, least supported first."""
    return sorted((r for r in rows if r["score"] < threshold), key=lambda r: r["score"])


def check_run(name: str, score: Scorer, path: Path, threshold: float) -> int:
    """Scores the answers of an evaluation run against the note of their question."""
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = to_check(json.loads(path.read_text(encoding="utf-8")), {q["id"]: q.get("sources", []) for q in golden})
    for row in rows:
        row["score"] = score(note(row["sources"]), row["answer"])
    hits = flagged(rows, threshold)
    right = sum(bool(r.get("right")) for r in hits)
    print(f"{name} on {path.name}: {len(rows)} answers read against their note, {len(hits)} flagged "
          f"(score under {threshold:.2f}): {right} that the term check counts right, {len(hits) - right} wrong\n")
    for row in hits:
        answer = " ".join(row["answer"].split())
        print(f"  {row['score']:.2f}  {'right' if row.get('right') else 'wrong'}  {row['id']:<12} {answer[:100]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate a grounding checker on the labelled answers, "
                                                 "or read the answers of an evaluation run with it.")
    parser.add_argument("checker", help="hhem, or ollama:<model>")
    # 127.0.0.1 and not localhost: each call opens a connection, and where localhost resolves to
    # ::1 first (Windows) every one of them waits two seconds before falling back.
    parser.add_argument("--url", default="http://127.0.0.1:11434", help="Ollama server (default %(default)s)")
    parser.add_argument("--run", type=Path, metavar="FILE",
                        help="read the answers of this run (obslab eval --answers --json FILE) instead of calibrating")
    parser.add_argument("--threshold", type=float, default=FLAG_UNDER,
                        help="with --run: flag an answer whose score is under this value (default %(default)s)")
    args = parser.parse_args(argv)
    score = scorer(args.checker, args.url)
    if args.run:
        return check_run(args.checker, score, args.run, args.threshold)
    return calibrate(args.checker, score)


if __name__ == "__main__":
    sys.exit(main())
