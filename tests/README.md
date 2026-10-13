# Tests

Five levels, from "runs anywhere in two seconds" to "needs the whole cluster". Each level is one
Task command, and each one needs a bit more than the previous.

| Level | Command | Needs | Runs in CI |
|---|---|---|---|
| Unit and integration | `task test` | nothing (fake models, in-memory exporters, no network) | every push |
| Vector store | `task test:pgvector` | Docker | every push |
| Compose end to end | `task test:compose` | Docker | every push |
| Cluster end to end | `task test:k8s`, `task test:stack` | the k3d cluster (`task up`) | on main, weekly, and on pull requests labelled `cluster-e2e` |
| Real models | `task test:compose PROVIDER=ollama`, `task test:ollama` | Ollama with both models pulled | after each merge on main and weekly, in a separate workflow that does not block a merge |

## What each level checks

**Unit and integration** (`unit/`, `integration/`). The RAG graph and its branches, the spans and
metrics it emits, the HTTP API, the sample notes against the load-test questions, input limits
and error messages. The telemetry is captured in memory, so the tests can assert on exact span
names and attributes.

**Vector store** (`e2e/test_pgvector.py`). The pgvector store against a real PostgreSQL: same
ranking as the in-memory store, a read-only API role, a re-index that readers never see half
done, an index from another embedding model refused, slow queries cut off, a pool that recovers
after the database drops its connections.

**Data lake** (`e2e/test_datalake.py`, same command). Runs the real `obslab ingest` on a copy of
the sample notes: a note added to the folder is answered after the next ingest, a removed note
is forgotten, files that are not `.md` or `.txt` are ignored, and an empty or missing folder
fails without touching the index that is being served.

**Conversations** (`unit/test_conversation.py` with in-memory checkpoints, `e2e/test_conversations.py`
with PostgreSQL, same command as the vector store). A follow-up is answered thanks to the
previous turn and not without it; the first question makes no extra model call; another
replica, or the same one after a restart, continues the conversation; a failed turn changes
nothing; the saved state is a few turns of text with one checkpoint per question; the length
limit and the purge work; the conversation role cannot read the index and the index role
cannot read the conversations.

**Compose end to end** (`e2e/test_stack.py`). Starts Prometheus, Tempo, Loki, the
OpenTelemetry Collector and PostgreSQL with Docker Compose, indexes the sample notes, then asks
one question and follows it:

1. the answer comes from the right note,
2. its metrics are in Prometheus, the wait for a model slot included,
3. its trace is complete in Tempo, down to the SQL query on `rag.chunks`,
4. its log line is in Loki, findable by trace ID, as the `rag.answered` event with its fields.

This is the cheapest test that proves the whole telemetry path, and it needs no model: the app
runs with the fake provider.

**Cluster end to end** (`e2e/test_cluster.py`, plus `test_stack.py` against the cluster). The
same question through the Gateway: both API replicas answer, the trace carries the pod
identity, metric labels stay bounded, the SLO rules are evaluated. A conversation of five
turns is served by both replicas, and the purge job runs. A note added to the data lake
is served after the ingest job, with the same image and the same API pods, and only that job
mounts the folder.

**Real models** (`e2e/test_ollama.py`). With `llama3.2:3b` and `nomic-embed-text`: a question
about the notes is answered from the right note, a follow-up is understood from the previous
turn, a new subject in the same conversation is not dragged back to the old one, an off-topic
question is declined. The Compose end-to-end test also runs with these models
(`PROVIDER=ollama`): the index is built with the real embedding model, and the checks on
metrics, trace and logs are the same.

## Conventions

- Tests that need infrastructure carry a marker (`pgvector`, `stack`, `k8s`, `ollama`) and are
  deselected by default; the Task commands select them.
- When a Task command runs them, a missing dependency is a failure, not a skip
  (`OBSLAB_REQUIRE_STACK=1`). Run `pytest -m stack` by hand and they skip instead.
- `conftest.py` holds the shared fixtures: fake models, a three-document corpus, in-memory
  span and metric exporters.
