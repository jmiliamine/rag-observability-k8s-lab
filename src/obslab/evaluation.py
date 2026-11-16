"""Retrieval measured on questions whose source is known (evals/golden.jsonl).

The telemetry of the lab says how fast an answer came. It cannot say whether the search found
the right note. This module can, with no model call: each question of the golden set names the
note that answers it, and the search is scored on where that note comes back.

  recall@1, recall@k   the expected note is the first result, or among the first k
  MRR                  mean of 1 / rank of the expected note (0 when it is not in the first DEPTH)
  in context           the expected note is among the first k AND above the score threshold:
                       what the answering model really receives
  refused              for a question the notes cannot answer: nothing is above the threshold

A note is "found" when any chunk of it is returned: the golden set is labelled per note, so a
change of chunking does not invalidate it.

Two variants are compared question by question (`compare`), not on their averages: with a
hundred questions, one or two points of recall are within the noise.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path

from langchain_core.vectorstores import VectorStore

KINDS = ("answerable", "no_answer", "follow_up")
DEPTH = 10                   # ranks looked at for the reciprocal rank
GATED = ("recall_at_k", "mrr", "in_context")      # a drop of one of these fails the check


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
        share = lambda hits, rows: round(hits / len(rows), 4) if rows else 0.0      # noqa: E731
        return {
            "questions": len(answerable),
            "recall_at_1": round(self.recall_at(1), 4),
            "recall_at_k": round(self.recall_at(self.k), 4),
            "mrr": share(sum(r.reciprocal_rank for r in answerable), answerable),
            "in_context": share(sum(r.in_context for r in answerable), answerable),
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
        rows.append(Row(
            id=q.id, kind=q.kind, style=q.style,
            rank=expected[0] if expected else None,
            in_context=any(i <= k and hits[i - 1][1] >= min_score for i in expected),
            kept=sum(1 for _, score in hits[:k] if score >= min_score),
            top_score=round(max((float(score) for _, score in hits), default=0.0), 4)))
    return RetrievalReport(k=k, min_score=min_score, rows=tuple(rows))


def compare(base: RetrievalReport, other: RetrievalReport) -> dict[str, float | int]:
    """`other` against `base`, question by question, on the reciprocal rank of the expected note.

    Returns how many questions got better, worse or stayed, and the p-value of an exact two-sided
    sign test on those that changed: the probability of a split at least this uneven if the two
    variants were equally good. A variant is better when the p-value is small, not when its
    average is higher."""
    ours = {r.id: r for r in base.rows if r.kind == "answerable"}
    theirs = {r.id: r for r in other.rows if r.kind == "answerable"}
    if ours.keys() != theirs.keys():
        # A p-value on the questions two runs happen to share would look like a result.
        raise ValueError(f"the two runs do not cover the same questions: {sorted(ours.keys() ^ theirs.keys())[:5]}")
    pairs = [(r.reciprocal_rank, theirs[i].reciprocal_rank) for i, r in ours.items()]
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
