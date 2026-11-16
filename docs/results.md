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

## Current baseline

What the gate compares with today (`evals/baseline.json`): vector search only, chunks of 500
characters with 100 of overlap (83 chunks), first 4 results. How it got there is in the two
sections that follow.

| | `nomic-embed-text`, threshold 0.6 | fake embeddings, threshold 0.2 |
|---|---|---|
| recall@1 | 0.919 | 0.768 |
| recall@4 | 0.980 | 0.939 |
| MRR | 0.947 | 0.850 |
| in context | 0.949 | 0.848 |
| facts in context | 0.866 | 0.720 |
| refused (12 questions) | 0.833 | 0.583 |

The fake embeddings are a bag of words hashed into a vector. Their column is not a result: it
is the baseline that the offline tests compare with, on every push. The `nomic-embed-text`
column is checked after each merge, in the real-model workflow.

By wording, with `nomic-embed-text`:

| Wording | Questions | recall@4 | MRR |
|---|---|---|---|
| Words of the note | 52 | 0.981 | 0.951 |
| Paraphrase | 26 | 1.000 | 0.933 |
| Identifier, number or command | 21 | 0.952 | 0.952 |

Answers, `task eval PROVIDER=ollama ANSWERS=1` with `llama3.2:3b` on a laptop GPU, three runs:

| | Mean | Lowest and highest run |
|---|---|---|
| Right, 99 questions with an answer | 0.869 | 0.869 on each run |
| Declined, 12 questions without answer | 0.889 | 0.833 to 0.917 |
| Right, 10 follow-ups | 0.900 | 0.900 on each run |
| Time per question | p50 2.0 s | |

What is left:

- **Five notes out of 99 do not reach the model.** Two are not among the first 4 results ("What
  does exit code 137 mean?" finds its note at rank 12, "What does USE stand for?" at rank 7),
  three are found and lost to the threshold ("What is an exemplar?" scores 0.52).
- **13 answerable questions out of 99 are still counted wrong on a run**, most of them answered
  from the right note.
- **Declining is the model's job for the questions close to the notes.** The off-topic
  questions score 0.41 to 0.49 at best and all stop at the threshold. The 8 questions close to
  the notes score 0.52 to 0.73: two pass, and the model has to say "I don't know" by itself.

## Starting point

The same measurements before the decisions below, with chunks of 800 characters and 120 of
overlap (47 chunks). The decisions started from what they show.

### Retrieval

| | `nomic-embed-text`, threshold 0.6 | fake embeddings, threshold 0.2 |
|---|---|---|
| recall@1 | 0.879 | 0.727 |
| recall@4 | 0.970 | 0.949 |
| MRR | 0.915 | 0.833 |
| in context | 0.879 | 0.768 |
| facts in context | 0.805 | 0.683 |
| refused (12 questions) | 0.833 | 0.583 |

By wording, with `nomic-embed-text`:

| Wording | Questions | recall@4 | MRR |
|---|---|---|---|
| Words of the note | 52 | 0.981 | 0.900 |
| Paraphrase | 26 | 1.000 | 0.974 |
| Identifier, number or command | 21 | 0.905 | 0.878 |

- **The threshold costs more than the ranking.** The note is among the first 4 results for 96
  questions out of 99, and reaches the model for 87: nine questions find their note and lose it
  to the 0.6 threshold. "What does the NoExecute effect do?" scores 0.59.
- **The threshold cannot separate the hard cases.** Off-topic questions score 0.38 to 0.45 at
  best. Questions close to the notes but without answer score 0.51 to 0.70, and 18 answerable
  questions have a best score under 0.62: the two groups overlap, so no value of the threshold
  refuses one and keeps the other.
- **Questions about an exact term are the weak spot of vector search.** "What does exit code 137
  mean?" finds its note at rank 9, "What is fsGroup for?" not in the first 10.

### Answers

Every question through the whole graph, with `llama3.2:3b`. A question with an answer is right
when it is answered and the reply holds the expected terms; a question without answer is right
when the reply is "I don't know"; a follow-up is asked after its previous turn.

Three runs of the same configuration:

| | Mean | Lowest and highest run |
|---|---|---|
| Right, 99 questions with an answer | 0.825 | 0.818 to 0.828 |
| Declined, 12 questions without answer | 0.889 | 0.833 to 0.917 |
| Right, 10 follow-ups | 0.833 | 0.700 to 0.900 |
| Time per question | p50 2.5 s | |

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

Each change was tried against the starting point above. The retrieval numbers are exact. The answer
numbers are means over three runs for the chunk sizes and the threshold, and a single run for
the prompt and the chat models: read them with the noise in mind, about 5 verdicts in 121.
"Better" and "worse" count questions, and p is the sign test on them.

| Change | Decision | Why, in one number |
|---|---|---|
| Chunks of 500 characters instead of 800 | **adopted** | 94 notes reach the model instead of 87 |
| Threshold 0.5 instead of 0.6 | not adopted | questions without answer declined: 0.889 to 0.694 |
| A stricter prompt | not adopted | right answers: 0.879 to 0.717 |
| Hybrid search | not adopted | at most 2 questions left to gain |
| A reranker on CPU | not adopted | 6 questions better, 3 worse, half a second more |
| Another embedding model | `nomic-embed-text` kept | better ranking, fewer facts past the threshold |
| Another chat model | `llama3.2:3b` kept | as many right answers, fewer refusals, 2 to 4 times slower |
| A grounding checker | `bespoke-minicheck` passes the bar, kept as an offline tool | 25 of 29 wrong answers caught, 6 of 82 correct ones flagged |

### Chunk size: 500 characters with 100 of overlap, adopted

| | 800 / 120 (before) | 500 / 100 | 400 / 60 |
|---|---|---|---|
| Chunks | 47 | 83 | 104 |
| recall@1 | 0.879 | 0.919 | 0.960 |
| recall@4 | 0.970 | 0.980 | 1.000 |
| MRR | 0.915 | 0.947 | 0.980 |
| in context | 0.879 | 0.949 | 0.939 |
| facts in context | 0.805 | 0.866 | 0.878 |
| refused (12 questions) | 0.833 | 0.833 | 0.750 |
| Answers right, 3 runs | 0.825 | 0.869 | 0.842 |
| Declined, 3 runs | 0.889 | 0.889 | 1.000 |
| Follow-ups right, 3 runs | 0.833 | 0.900 | 0.700 |

Against 800 / 120, question by question: with 500 / 100 the note reaches the model for 9 more
questions and 2 fewer (p = 0.07), and over three runs 10 questions are answered better and 4
worse (p = 0.18). With 400 / 60 the ranking is the best of the three, and the answers are not:
12 better, 10 worse (p = 0.83), and the follow-ups drop.

Not a proof at the usual 5 %: 99 questions cannot give one for a gain of this size. It is kept
because the two levels agree, the exact one (retrieval) and the noisy one (answers), and nothing
gets worse. Neighbouring sizes (450 / 90, 500 / 150) give the same picture: the gain comes from
the range, not from one value. Two effects pull in opposite directions. Smaller chunks rank
better: one chunk, one subject. Larger chunks hold the answer more often once they are found:
with no threshold, the expected facts are in the first 4 chunks for 0.976 of the questions at
800 characters, 0.951 at 500 and 0.927 at 300.

### Score threshold: 0.6 kept

At 0.5, on the chunks of 800 characters, three runs each:

| | 0.6 | 0.5 |
|---|---|---|
| Right, questions with an answer | 0.825 | 0.869 |
| Declined, questions without answer | 0.889 | 0.694 |
| Right, follow-ups | 0.833 | 0.967 |

9 answerable questions better, 5 worse (p = 0.42). The threshold trades one error for another:
fewer "I don't know" where the notes had the answer, more answers where they had none, and the
second kind is the worse one for a tool that answers from notes. With the chunks of 500
characters the threshold costs less anyway: 94 notes out of 99 pass it.

No value of the threshold separates the questions close to the notes from the answerable ones,
with any of the three embedding models tried. The threshold can only stop what is off topic;
declining the rest is the model's job.

### A stricter prompt: not adopted

A system prompt that insists on declining ("reply exactly I don't know and nothing else",
"never use knowledge that is not in the context"), at threshold 0.5, one run:

| | Current prompt | Stricter prompt |
|---|---|---|
| Right, questions with an answer | 0.879 | 0.717 |
| Declined, questions without answer | 0.750 | 0.917 |
| Right, follow-ups | 0.900 | 0.300 |

No question better, 16 worse (p < 0.001). `llama3.2:3b` does not decline more precisely, it
declines more: 21 answerable questions end as "I don't know". The same on the chunks of 500
characters at threshold 0.6, one run: 0.758 right against 0.869 with the current prompt.

### Hybrid search (PostgreSQL full text and vectors, reciprocal rank fusion): not adopted

Built-in `tsvector`, the words of the question joined by OR, `ts_rank`, fused with the vector
ranking by reciprocal rank (k = 60).

| Chunks | | recall@1 | recall@4 | MRR | Better / worse than vectors alone |
|---|---|---|---|---|---|
| 800 / 120 | vectors | 0.879 | 0.970 | 0.915 | |
| 800 / 120 | hybrid | 0.950 | 1.000 | 0.972 | 11 / 1 (p = 0.006) |
| 400 / 60 | vectors | 0.960 | 1.000 | 0.980 | |
| 400 / 60 | hybrid | 0.939 | 1.000 | 0.968 | 1 / 4 (p = 0.38) |

On the chunks of 800 characters hybrid search ranks better, and it is the one result of this
page that passes the 5 % line. It still does not change what the model receives: the gate is the
score threshold, and "in context" stays at 0.879. With smaller chunks the ranking gain is gone.
On the chunks adopted here, 97 notes out of 99 are already in the first 4: two questions is the
most a better ranking can add, for a second index, a second query and a fusion to maintain. It
was not measured on the 500 / 100 chunks. Worth trying again on a corpus where questions about
exact identifiers are common and the notes are long.

### A reranker on CPU: not adopted

A cross-encoder (`ms-marco-MiniLM-L-6-v2`, ONNX, no GPU) reorders the first 8 results, on the
chunks of 500 characters:

| | recall@1 | recall@4 | MRR |
|---|---|---|---|
| Vector order | 0.919 | 0.980 | 0.947 |
| Reranked | 0.939 | 0.990 | 0.965 |

6 questions better, 3 worse (p = 0.51), for 0.5 second more per question at the median and 1.2
at the 95th percentile on a laptop CPU, where an answer takes 2 seconds. Used as the gate
instead of the cosine threshold, its score does not separate better either (facts in context
0.854 with 11 of 12 questions without answer refused, against 0.866 with 10). It would also be a
new runtime dependency and a model file to ship in the image.

### Embedding model: `nomic-embed-text` kept

On the chunks of 500 characters. "With prefixes" means the task prefixes each model card
documents, which Ollama does not add by itself.

| Model | recall@1 | recall@4 | MRR | Better / worse than `nomic-embed-text` | Facts in context, 10 of 12 refused |
|---|---|---|---|---|---|
| `nomic-embed-text` | 0.919 | 0.980 | 0.947 | | 0.866 |
| `nomic-embed-text`, with prefixes | 0.919 | 0.990 | 0.955 | 5 / 3 (p = 0.73) | 0.817 |
| `embeddinggemma:300m` | 0.960 | 1.000 | 0.976 | 8 / 4 (p = 0.39) | 0.573 |
| `embeddinggemma:300m`, with prefixes | 0.960 | 1.000 | 0.978 | 8 / 4 (p = 0.39) | 0.805 |
| `qwen3-embedding:0.6b` | 0.869 | 1.000 | 0.925 | 7 / 12 (p = 0.36) | 0.707 |
| `qwen3-embedding:0.6b`, with prefixes | 0.950 | 1.000 | 0.975 | 7 / 4 (p = 0.55) | 0.756 |

The last column is the one that decides: each model at the threshold that refuses 10 of the 12
questions without answer, as the current setup does. `embeddinggemma` ranks a little better, not
significantly, and its scores separate less well: at equal refusal fewer expected facts reach
the model. A change would also mean a new index and a new threshold for an unproven gain. The
prefixes matter for the two newer models and change nothing measurable for `nomic-embed-text`.

### Chat model: `llama3.2:3b` kept

On the chunks of 500 characters, threshold 0.6. One run per candidate, against the three runs of
`llama3.2:3b`:

| Model | Right, with an answer | Declined, without answer | Right, follow-ups | Time per question, p50 and p95 |
|---|---|---|---|---|
| `llama3.2:3b` | 0.869 | 0.889 | 0.900 | 1.6 s and 5.8 s |
| `qwen3.5:4b`, thinking off | 0.869 | 0.750 | 0.900 | 7.0 s and 10.2 s |
| `phi4-mini:3.8b` | 0.859 | 0.500 | 1.000 | 4.0 s and 8.5 s |

The times come from the same 30 questions asked to each model alone, fully loaded on the GPU.

Neither candidate answers more questions right: against one run of `llama3.2:3b`, `qwen3.5:4b`
is better on 2 questions and worse on 2, `phi4-mini` better on 2 and worse on 3. Both decline
less often when the notes have no answer (9 and 6 of the 12 questions, against 10 or 11), and
both are slower. 10 of the 13 questions `llama3.2:3b` gets wrong are wrong with all three
models: what is left to gain is mostly not in the answering model. Not tried: `qwen3.5:2b`, and
`qwen3.5:4b` with thinking on.

### Grounding checker: `bespoke-minicheck` passes the bar, and stays an offline tool

A grounding checker reads a note and an answer and says whether the note supports the answer.
Three were scored on the 111 labelled answers of `evals/answers.jsonl` (82 correct, 29 wrong)
with `task eval:checker`, against a bar fixed before any run: at least 24 of the 29 wrong
answers caught, at most 7 of the 82 correct ones flagged. For each, the best threshold within
the allowed false alarms:

| Checker | Size | Wrong answers caught | Correct answers flagged | Time per answer | Bar |
|---|---|---|---|---|---|
| Vectara HHEM-2.1-open | 0.1B parameters, on CPU | 15 of 29 | 5 of 82 | 1 s | fails |
| `granite3-guardian:2b` | 2.7 GB | 15 of 29 | 6 of 82 | 0.4 s | fails |
| `bespoke-minicheck` | 4.7 GB, 7.7B parameters | 26 of 29 | 7 of 82 | 2.0 s | passes |

`bespoke-minicheck` passes at three neighbouring thresholds (24 and 6, 25 and 6, 26 and 7), so
with one answer to spare on each side at the middle one, 0.5. The threshold is chosen on the
answers it is judged on: on new answers, expect a little less. Each sentence of an answer is
checked alone, as the model card asks, and the answer gets the score of its least supported
sentence.

What it is used for: `task eval:checker RUN=run.json` reads the answers of an evaluation run
and lists the ones their note does not support. On one run of the current baseline, 104 answers
read in 5 minutes, 20 flagged: 7 that the term check counts wrong and 13 that it counts right.
Those 13 were not read by hand; at the false-alarm rate measured above, about 7 are expected by
chance among the 95 answers counted right.

What it is not used for: a check at request time, or a step of the post-merge workflow. Loaded,
the model takes 5.2 GB: it does not fit on a laptop GPU next to the answering model, and the
hosted runner has no GPU. Its licence is CC BY-NC 4.0: fine for a lab, not for a commercial use
without an agreement with its authors. The two others are under Apache 2.0.
