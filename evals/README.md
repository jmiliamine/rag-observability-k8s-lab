# evals

What the lab measures its answers against.

- `golden.jsonl`: questions about the sample notes in `datalake/`, each with the note that
  answers it. Written by hand.
- `baseline.json`: the retrieval metrics of the current `main`, one entry per embedding model.
  `task eval` fails when a run is below them.

Results and decisions are in [docs/results.md](../docs/results.md).

## Running it

```
task eval                    # fake embeddings, offline: the same check as in `task test`
task eval PROVIDER=ollama    # nomic-embed-text, needs Ollama; no chat model is called
task eval UPDATE=1           # record the run as the new baseline, then commit baseline.json
```

The output lists the questions whose note did not reach the answering model, with the rank of
the note and the best score: start there when a number drops.

## The golden set

One JSON object per line.

| Field | Meaning |
|---|---|
| `id` | unique, `<note>-<number>` |
| `kind` | `answerable`, `no_answer` or `follow_up` |
| `question` | as a user would type it |
| `sources` | the notes that answer it, relative to `datalake/`. Any one of them counts. Empty for `no_answer` |
| `previous` | `follow_up` only: the question asked just before, in the same conversation |
| `expect` | optional. Groups of terms: a right answer contains one term of each group. A crude check, used when answers are evaluated, not for retrieval |
| `style` | `direct` (the words of the note), `paraphrase` (other words), `keyword` (an identifier, a number, a command); `off_topic` or `near_domain` for questions without answer |

Three kinds, because they fail differently:

- **answerable**: the search must bring back the note.
- **no_answer**: the right output is "I don't know". Some have nothing to do with the notes;
  others are close to them (a Kubernetes object the notes never mention), which is the hard case.
- **follow_up**: only makes sense after `previous`. Retrieval is not scored on these: a model
  rewrites them first, so they belong to the evaluation of answers.

## Adding or changing a question

1. Add the line. The note must contain the answer; if two notes do, list both.
2. `task test`: the format is checked, and the baseline test fails because the set changed.
3. `task eval UPDATE=1`, and `task eval PROVIDER=ollama UPDATE=1` if Ollama is running.
4. Commit `golden.jsonl` and `baseline.json` together, with the before and after numbers in the
   pull request. A baseline that goes down needs a reason.

The labels are per note, not per chunk, so the set stays valid when the chunking changes.
