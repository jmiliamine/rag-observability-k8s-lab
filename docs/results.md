# Results

What was measured on the sample notes, and what was decided from it. A change to the search, to
a prompt or to a model is kept when it shows here, and dropped otherwise; what was dropped is
written down too.

Every number comes from `task eval` and the golden set in [`evals/`](../evals/README.md).

## The yardstick

121 questions written by hand over the 20 sample notes, then reviewed line by line against the
notes:

| Kind | Questions | What is expected |
|---|---|---|
| With an answer in one note | 99 | the search brings that note back |
| Without answer in the notes | 12 | "I don't know": 4 are off topic, 8 are close to the notes |
| Follow-up of a previous question | 10 | understood from the turn before (answers only, not scored on retrieval) |

The 99 are split by how they are worded: 52 use the words of the note, 26 paraphrase it, 21 ask
about an identifier, a number or a command.

Retrieval is scored with no model call, so a run gives the same numbers every time:

- **recall@1, recall@4**: the expected note is the first result, or among the first 4 (the
  number of chunks the application retrieves).
- **MRR**: mean of 1 / rank of the expected note.
- **in context**: the expected note is among the first 4 *and* above the score threshold. This
  is what the answering model receives.
- **refused**: for a question without answer, nothing is above the threshold.

With 99 questions, one question is one point. Two variants are compared question by question
(a sign test on the questions whose rank changed), not on their averages.

## Retrieval baseline

Vector search only, chunks of 800 characters with 120 of overlap (47 chunks), first 4 results.

| | `nomic-embed-text`, threshold 0.6 | fake embeddings, threshold 0.2 |
|---|---|---|
| recall@1 | 0.879 | 0.727 |
| recall@4 | 0.970 | 0.949 |
| MRR | 0.915 | 0.833 |
| in context | 0.879 | 0.768 |
| refused (12 questions) | 0.833 | 0.583 |

The fake embeddings are a bag of words hashed into a vector. Their column is not a result: it
is the baseline that the offline tests compare with, on every push. The `nomic-embed-text`
column is checked after each merge, in the real-model workflow.

By wording, with `nomic-embed-text`:

| Wording | Questions | recall@4 | MRR |
|---|---|---|---|
| Words of the note | 52 | 0.981 | 0.900 |
| Paraphrase | 26 | 1.000 | 0.974 |
| Identifier, number or command | 21 | 0.905 | 0.878 |

What the baseline shows:

- **The threshold costs more than the ranking.** The note is among the first 4 results for 96
  questions out of 99, and reaches the model for 87: nine questions find their note and lose it
  to the 0.6 threshold. "What does the NoExecute effect do?" scores 0.59.
- **The threshold cannot separate the hard cases.** Off-topic questions score 0.38 to 0.45 at
  best. Questions close to the notes but without answer score 0.51 to 0.70, and 18 answerable
  questions have a best score under 0.62: the two groups overlap, so no value of the threshold
  refuses one and keeps the other.
- **Questions about an exact term are the weak spot of vector search.** "What does exit code 137
  mean?" finds its note at rank 9, "What is fsGroup for?" not in the first 10.

## Decisions

None yet beyond the baseline. Each change tried against it is added here with its numbers,
kept or not.
