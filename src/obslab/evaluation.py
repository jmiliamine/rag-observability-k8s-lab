"""Retrieval and answers measured on questions whose source is known (evals/golden.jsonl).

The telemetry of the lab says how fast an answer came. It cannot say whether the search found
the right note, or whether the answer is right. This module can, on the golden set: each
question names the note that answers it and the terms a right answer contains.

Retrieval, with no model call (`evaluate_retrieval`):

  recall@1, recall@k   the expected note is the first result, or among the first k
  MRR                  mean of 1 / rank of the expected note (0 when it is not in the first DEPTH)
  in context           the expected note is among the first k AND above the score threshold:
                       what the answering model really receives
  facts in context     the expected terms are in the text of those chunks. A note can be found
                       through a chunk that does not hold the answer: this one sees it
  refused              for a question the notes cannot answer: nothing is above the threshold

A note is "found" when any chunk of it is returned: the golden set is labelled per note, so a
change of chunking does not invalidate it.

Answers, through the whole graph with the configured models (`evaluate_answers`): a question
with an answer is right when it is answered and holds the expected terms, a question without
answer when the reply is "I don't know", a follow-up when it is right after the turn before.
The term check is crude on purpose: the same verdict on every run, no judge model.

Two variants are compared question by question (`compare`), not on their averages: with a
hundred questions, one or two points are within the noise.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from math import comb
from pathlib import Path

from langchain_core.vectorstores import VectorStore

KINDS = ("answerable", "no_answer", "follow_up")
DEPTH = 10                   # ranks looked at for the reciprocal rank
GATED = ("recall_at_k", "mrr", "in_context", "facts_in_context")   # a drop of one of these fails the check


@dataclass(frozen=True)
class Question:
    id: str
    kind: str                # answerable | no_answer | follow_up
    question: str
    sources: tuple[str, ...] = ()      # any of these notes answers it
    previous: str = ""       # follow_up: the question asked just before, in the same conversation
    expect: tuple[tuple[str, ...], ...] = ()   # terms a right answer contains: one of each group
    style: str = ""          # direct | paraphrase | keyword | off_topic | near_domain | follow_up


def load_golden(path: Path, notes: Path | None = None) -> list[Question]:
    """The golden set, checked: a malformed line fails here, not as a silent zero in a metric.
    With `notes`, every expected source must be a file of that folder."""
    def strings(value) -> bool:
        return isinstance(value, list) and all(isinstance(v, str) and v for v in value)

    questions, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name} line {n}"
        try:
            raw = json.loads(line)
            ident, kind, text = raw["id"], raw["kind"], raw["question"]
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{where}: not a JSON object with id, kind and question ({type(e).__name__})") from e
        if not isinstance(ident, str) or not 0 < len(ident) <= 64:      # an id is also a field of a log event
            raise ValueError(f"{where}: `id` must be a string of 1 to 64 characters")
        where = f"{where} ({ident})"
        sources, expect = raw.get("sources", []), raw.get("expect", [])
        # A string where a list is expected would be read letter by letter, and match anything.
        if not strings(sources):
            raise ValueError(f"{where}: `sources` must be a list of note paths")
        if not isinstance(expect, list) or not all(strings(group) and group for group in expect):
            raise ValueError(f"{where}: `expect` must be a list of non-empty lists of terms")
        q = Question(id=ident, kind=kind, question=str(text).strip(), sources=tuple(sources),
                     previous=raw.get("previous", ""), style=raw.get("style", ""),
                     expect=tuple(tuple(t.lower() for t in group) for group in expect))
        if q.id in seen:
            raise ValueError(f"{where}: duplicate id")
        seen.add(q.id)
        if q.kind not in KINDS:
            raise ValueError(f"{where}: kind {q.kind!r} is not one of {KINDS}")
        if not q.question:
            raise ValueError(f"{where}: empty question")
        if (q.kind == "no_answer") == bool(q.sources):
            raise ValueError(f"{where}: a no_answer question has no source, the others have at least one")
        if (q.kind == "follow_up") != bool(q.previous):
            raise ValueError(f"{where}: `previous` is for follow_up questions, and required there")
        if notes is not None and (missing := [s for s in q.sources if not (notes / s).is_file()]):
            raise ValueError(f"{where}: no such note {missing}")
        questions.append(q)
    if not questions:
        raise ValueError(f"{path}: no question")
    return questions


def has_terms(text: str, expect: Sequence[Sequence[str]]) -> bool:
    """One term of each group is in the text. Substrings, lower case."""
    text = text.lower()
    return all(any(term in text for term in group) for group in expect)


def fingerprint(path: Path) -> str:
    """Short identity of a golden set file, printed with every result computed from it."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


@dataclass(frozen=True)
class Row:
    """What the search returned for one question."""
    id: str
    kind: str
    style: str
    rank: int | None         # rank of the first chunk of an expected note in the results, from 1
    in_context: bool         # an expected note among the first k, above the threshold
    kept: int                # chunks among the first k above the threshold
    top_score: float
    facts: bool | None = None    # the expected terms are in the kept chunks; None: no terms to look for

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.rank if self.rank and self.rank <= DEPTH else 0.0


@dataclass(frozen=True)
class RetrievalReport:
    k: int
    min_score: float
    rows: tuple[Row, ...]

    def _answerable(self) -> list[Row]:
        return [r for r in self.rows if r.kind == "answerable"]

    def recall_at(self, n: int, rows: Sequence[Row] | None = None) -> float:
        rows = self._answerable() if rows is None else rows
        return sum(1 for r in rows if r.rank is not None and r.rank <= n) / len(rows) if rows else 0.0

    def metrics(self) -> dict[str, float | int]:
        answerable = self._answerable()
        unanswerable = [r for r in self.rows if r.kind == "no_answer"]
        with_terms = [r for r in answerable if r.facts is not None]
        share = lambda hits, rows: round(hits / len(rows), 4) if rows else 0.0      # noqa: E731
        return {
            "questions": len(answerable),
            "recall_at_1": round(self.recall_at(1), 4),
            "recall_at_k": round(self.recall_at(self.k), 4),
            "mrr": share(sum(r.reciprocal_rank for r in answerable), answerable),
            "in_context": share(sum(r.in_context for r in answerable), answerable),
            "facts_in_context": share(sum(bool(r.facts) for r in with_terms), with_terms),
            "no_answer_questions": len(unanswerable),
            "no_answer_refused": share(sum(r.kept == 0 for r in unanswerable), unanswerable),
        }

    def by_style(self) -> dict[str, dict[str, float | int]]:
        out = {}
        for style in sorted({r.style for r in self._answerable()}):
            rows = [r for r in self._answerable() if r.style == style]
            out[style] = {"questions": len(rows), "recall_at_k": round(self.recall_at(self.k, rows), 4),
                          "mrr": round(sum(r.reciprocal_rank for r in rows) / len(rows), 4)}
        return out

    def misses(self) -> list[Row]:
        """Answerable questions the answering model would not get the right note for."""
        return [r for r in self._answerable() if not r.in_context]

    def scores(self) -> dict[str, float]:
        """What `compare` works on: the reciprocal rank of the expected note, per question."""
        return {r.id: r.reciprocal_rank for r in self._answerable()}


def evaluate_retrieval(store: VectorStore, questions: Sequence[Question], k: int, min_score: float) -> RetrievalReport:
    """Score the search of `store` on the questions that need no conversation. Follow-up
    questions are skipped: they only make sense once a model has rewritten them."""
    if k < 1:
        raise ValueError(f"k={k}: at least one result is needed")
    rows = []
    for q in questions:
        if q.kind == "follow_up":
            continue
        hits = store.similarity_search_with_score(q.question, k=max(k, DEPTH))
        expected = [i for i, (doc, _) in enumerate(hits, 1) if doc.metadata.get("source") in q.sources]
        context = " ".join(doc.page_content for doc, score in hits[:k] if score >= min_score)
        rows.append(Row(
            id=q.id, kind=q.kind, style=q.style, facts=has_terms(context, q.expect) if q.expect else None,
            rank=expected[0] if expected else None,
            in_context=any(i <= k and hits[i - 1][1] >= min_score for i in expected),
            kept=sum(1 for _, score in hits[:k] if score >= min_score),
            top_score=round(max((float(score) for _, score in hits), default=0.0), 4)))
    return RetrievalReport(k=k, min_score=min_score, rows=tuple(rows))


def compare(base: Mapping[str, float], other: Mapping[str, float]) -> dict[str, float | int]:
    """`other` against `base`, question by question. Each maps a question id to a score: the
    reciprocal rank of the expected note (`RetrievalReport.scores`) or right/wrong
    (`AnswerReport.scores`).

    Returns how many questions got better, worse or stayed, and the p-value of an exact two-sided
    sign test on those that changed: the probability of a split at least this uneven if the two
    variants were equally good. A variant is better when the p-value is small, not when its
    average is higher."""
    if base.keys() != other.keys():
        # A p-value on the questions two runs happen to share would look like a result.
        raise ValueError(f"the two runs do not cover the same questions: {sorted(base.keys() ^ other.keys())[:5]}")
    pairs = [(score, other[i]) for i, score in base.items()]
    better = sum(1 for a, b in pairs if b > a)
    worse = sum(1 for a, b in pairs if b < a)
    changed = better + worse
    tail = sum(comb(changed, i) for i in range(min(better, worse) + 1))
    return {"questions": len(pairs), "better": better, "worse": worse, "same": len(pairs) - changed,
            "p_value": round(min(1.0, 2 * tail / 2 ** changed), 4) if changed else 1.0}


# ── baseline ─────────────────────────────────────────────────────────────────

def check(metrics: dict[str, float | int], baseline: dict[str, float | int]) -> list[str]:
    """The gated metrics that are below the committed baseline, as readable lines. Empty: pass."""
    problems = [f"{name} is not in the baseline: record it again" for name in GATED if name not in baseline]
    return problems + [f"{name} {metrics[name]:.4f} is below the baseline {baseline[name]:.4f}"
                       for name in GATED if name in baseline and metrics[name] < baseline[name] - 1e-9]


def read_baseline(path: Path, key: str) -> dict | None:
    """The baseline of one configuration (the embedding model), or None when it has none yet."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(key)


def write_baseline(path: Path, key: str, entry: dict) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data[key] = entry
    path.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n", encoding="utf-8", newline="\n")


# ── answers ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Answer:
    """What the graph answered to one question of the golden set."""
    id: str
    kind: str
    route: str               # answered | fallback | error
    right: bool              # see `evaluate_answers`
    right_source: bool       # an expected note is among the sources of the answer
    seconds: float
    rewrites: int = 0
    answer: str = ""
    standalone: str = ""     # a follow-up as the model rewrote it
    error: str = ""          # the type of the exception, when the question raised


@dataclass(frozen=True)
class AnswerReport:
    rows: tuple[Answer, ...]

    def metrics(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {}
        for kind, name in (("answerable", "right"), ("no_answer", "declined"), ("follow_up", "follow_up_right")):
            rows = [r for r in self.rows if r.kind == kind]
            out[f"{kind}_questions"] = len(rows)
            out[name] = round(sum(r.right for r in rows) / len(rows), 4) if rows else 0.0
        answerable = [r for r in self.rows if r.kind == "answerable"]
        share = lambda hits: round(hits / len(answerable), 4) if answerable else 0.0      # noqa: E731
        out["answered"] = share(sum(r.route == "answered" for r in answerable))
        out["right_source"] = share(sum(r.right_source for r in answerable))
        times = sorted(r.seconds for r in self.rows)
        out["p50_s"] = round(times[len(times) // 2], 2) if times else 0.0
        out["p95_s"] = round(times[min(len(times) - 1, int(len(times) * 0.95))], 2) if times else 0.0
        out["errors"] = sum(r.route == "error" for r in self.rows)
        return out

    def scores(self) -> dict[str, float]:
        return {r.id: float(r.right) for r in self.rows}


def evaluate_answers(ask: Callable[..., dict], questions: Sequence[Question],
                     progress: Callable[[Answer], None] | None = None,
                     around: Callable[[Question], AbstractContextManager] | None = None) -> AnswerReport:
    """Ask every question through `ask(question, conversation_id)` (`Components.ask`) and judge
    the reply with what the golden set says:

      a question with an answer   right = answered, with the expected terms (or, when the line
                                  has no terms, from an expected note)
      a question without answer   right = the reply is the fallback, whichever step declined
      a follow-up                 `previous` is asked first in the same conversation, then as above

    A question that raises is counted wrong, with the route `error`. `around(question)` is a
    context that stays open while the question is asked and `progress` is called: the command
    opens a span there, so the verdict is logged in the trace of the answer it judges."""
    rows = []
    for q in questions:
        with around(q) if around else nullcontext():
            start, error = None, ""
            try:
                conversation = ask(q.previous)["conversation_id"] if q.kind == "follow_up" else None
                start = time.perf_counter()
                reply = ask(q.question, conversation)
            except Exception as e:   # one failed question must not lose the run; it shows as an error
                reply, error = {"route": "error"}, type(e).__name__
            seconds = time.perf_counter() - start if start else 0.0     # no time when the turn before failed
            route, answer = reply.get("route") or "", reply.get("answer", "")
            right_source = bool(set(q.sources) & set(reply.get("sources", ())))
            if q.kind == "no_answer":
                right = route == "fallback"
            else:
                right = route == "answered" and (has_terms(answer, q.expect) if q.expect else right_source)
            row = Answer(id=q.id, kind=q.kind, route=route, right=right, right_source=right_source,
                         seconds=round(seconds, 3), rewrites=reply.get("rewrites", 0), answer=answer,
                         standalone=reply.get("standalone_question", ""), error=error)
            rows.append(row)
            if progress:
                progress(row)
    return AnswerReport(rows=tuple(rows))
