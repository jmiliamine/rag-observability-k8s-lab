# evals

What the lab measures its answers against.

- `golden.jsonl`: questions about the sample notes in `datalake/`, each with the note that
  answers it. Written by hand.
- `baseline.json`: the retrieval metrics of the current `main`, one entry per embedding model.
  `task eval` fails when a run is below them.
- `answers.jsonl`: 111 answers of the answering model to questions of the golden set, labelled
  `correct` or `wrong` against the note. The test set for anything that claims to detect a wrong
  answer.

Results and decisions are in [docs/results.md](../docs/results.md).

## Running it

```
task eval                    # fake embeddings, offline: the same check as in `task test`
task eval PROVIDER=ollama    # nomic-embed-text, needs Ollama; no chat model is called
task eval UPDATE=1           # record the run as the new baseline, then commit baseline.json
task eval PROVIDER=ollama ANSWERS=1    # ask every question to the chat model, count the right answers
```

Two levels:

- **Retrieval** (the default): no model call, the same numbers on every run, compared with the
  baseline. The output lists the questions whose note did not reach the answering model, with
  the rank of the note and the best score: start there when a number drops.
- **Answers** (`ANSWERS=1`): every question goes through the whole graph. A question with an
  answer is right when it is answered and the reply holds the expected terms; a question without
  answer is right when the reply is "I don't know"; a follow-up is asked after its previous
  turn. A few minutes on a GPU. The answers of a real model change from run to run, so this
  level is reported and never gated.

To compare two variants, keep the first run and give it to the second:

```
task eval PROVIDER=ollama ANSWERS=1 -- --json before.json
OBSLAB_CHUNK_SIZE=400 OBSLAB_CHUNK_OVERLAP=60 task eval PROVIDER=ollama ANSWERS=1 -- --against before.json
```

The last line says how many questions got better, how many worse, and the p-value of a sign
test on those. Run the same variant twice first: the difference between two identical runs is
the noise any real difference has to exceed.

With `TELEMETRY=otlp`, each result is also sent as a `gen_ai.evaluation.result` log event and
the run as one `eval.finished` event with the configuration that produced it; in answers mode
the event carries the trace ID of the answer it judges.

## The golden set

One JSON object per line.

| Field | Meaning |
|---|---|
| `id` | unique, `<note>-<number>` |
| `kind` | `answerable`, `no_answer` or `follow_up` |
| `question` | as a user would type it |
| `sources` | the notes that answer it, relative to `datalake/`. Any one of them counts. Empty for `no_answer` |
| `previous` | `follow_up` only: the question asked just before, in the same conversation |
| `expect` | optional. Groups of terms: a right answer contains one term of each group. A crude check: the same verdict on every run, and it cannot tell an answer that uses the right words to say the wrong thing. Also used on the retrieved text ("facts in context") |
| `style` | `direct` (the words of the note), `paraphrase` (other words), `keyword` (an identifier, a number, a command); `off_topic` or `near_domain` for questions without answer |

Three kinds, because they fail differently:

- **answerable**: the search must bring back the note.
- **no_answer**: the right output is "I don't know". Some have nothing to do with the notes;
  others are close to them (a Kubernetes object the notes never mention), which is the hard case.
- **follow_up**: only makes sense after `previous`. Retrieval is not scored on these: a model
  rewrites them first, so they are scored with the answers.

## Adding or changing a question

1. Add the line. The note must contain the answer; if two notes do, list both.
2. `task test`: the format is checked, and the baseline test fails because the set changed.
3. `task eval UPDATE=1`, and `task eval PROVIDER=ollama UPDATE=1` if Ollama is running.
4. Commit `golden.jsonl` and `baseline.json` together, with the before and after numbers in the
   pull request. A baseline that goes down needs a reason.

The labels are per note, not per chunk, so the set stays valid when the chunking changes.

## The labelled answers

`answers.jsonl` holds real answers of `llama3.2:3b`, collected from evaluation runs: for each
question, one answer that the term check calls right, and up to two it calls wrong. Each was
read against its note by two readers, independently. They agreed on 111 of 116; the five others
were dropped. 82 are `correct` (every claim is supported by the note), 29 are `wrong` (a claim
is contradicted by the note, or is not in it and matters for the question).

It is used to calibrate a grounding checker, a model that reads a note and an answer and says
whether the note supports the answer:

```
task eval:checker CHECKER=ollama:bespoke-minicheck
task eval:checker CHECKER=hhem
```

A checker that passes can then read the answers of an evaluation run and list the ones their
note does not support:

```
task eval PROVIDER=ollama ANSWERS=1 -- --json run.json
task eval:checker RUN=run.json
```

The bar is written in `scripts/calibrate_checker.py` and was fixed before any checker was run:
catch at least 4 wrong answers in 5, with at most 2 false alarms in 23 correct answers. On this
file: at least 24 of the 29, at most 7 of the 82. A checker that has no threshold meeting both
is not used; the results are in [docs/results.md](../docs/results.md).

A new answer is added with its `question_id`, the `sources` of that question, and a label that a
second reader confirms.
