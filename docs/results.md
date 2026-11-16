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
- **facts in context**: the terms a right answer must contain are in the text of those chunks
  (82 of the 99 questions have such terms). A note can come back through a chunk that does not
  hold the answer; this number sees it, the previous ones do not.
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
| facts in context | 0.805 | 0.683 |
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

## Answers baseline

`task eval PROVIDER=ollama ANSWERS=1`: every question through the whole graph, with
`llama3.2:3b` on a laptop GPU. A question with an answer is right when it is answered and the
reply holds the expected terms; a question without answer is right when the reply is "I don't
know"; a follow-up is asked after its previous turn.

Three runs of the same configuration:

| | Mean | Lowest and highest run |
|---|---|---|
| Right, 99 questions with an answer | 0.825 | 0.818 to 0.828 |
| Declined, 12 questions without answer | 0.889 | 0.833 to 0.917 |
| Right, 10 follow-ups | 0.833 | 0.700 to 0.900 |
| Time per question | p50 2.5 s | |

What it shows:

- **The model does not answer the same way twice.** Between two identical runs, 5 of the 121
  verdicts change. A difference of that size between two variants is noise; with 10 questions,
  the follow-up line alone can move by 20 points.
- **About 17 answerable questions out of 99 are counted wrong on every run.** Six of them never
  get their note past the threshold and end as "I don't know". The others are answered: most
  wrongly, from the right note (the liveness probe given where the note says startup probe) or
  from a chunk of another note that came along, and a few correctly in other words than the
  expected terms.
- **A question close to the notes can be answered from the model's own knowledge.** "How do I
  fine-tune an embedding model on my own data?" passes the threshold, the notes say nothing
  about it, and the model answers anyway.
- **The term check has two blind spots, both seen in these runs.** It calls right an answer
  that holds the terms next to a false statement, and it calls wrong a correct answer worded
  differently. Read by hand, about 5 answers in 100 are in the first case.

## Decisions

None yet beyond the baselines. Each change tried against them is added here with its numbers,
kept or not.
