# Architecture

![Architecture diagram](img/architecture.svg)

## Who owns what

The cluster is split the way a small company would split it:

| Namespace | Owner | Contents |
|---|---|---|
| `gateway` | platform | the shared `Gateway` listening on `*.localhost` |
| `monitoring` | platform | Prometheus Operator, Prometheus, Grafana, Alertmanager (kube-prometheus-stack) |
| `observability` | platform | OpenTelemetry Collector, Tempo, Loki |
| `obslab` | app team | the RAG API (2 replicas), the ingest job, PostgreSQL + pgvector, its `HTTPRoute` |

The app team ships its own route, its own alert rules (a `PrometheusRule`) and its own
dashboard (a labelled `ConfigMap`). The platform decides which namespaces may attach routes
to the Gateway (`obslab.dev/gateway-access=true`).

## One question, step by step

1. `POST /ask` arrives through the Gateway (Traefik) and reaches the API pod.
2. FastAPI auto-instrumentation opens the root span `POST /ask`.
3. The LangGraph flow runs. A callback handler turns each node into a span
   (`rag.node retrieve`, `rag.node generate`...), all under `invoke_workflow rag`, which carries
   the conversation id.
4. `condense` loads the last turns of the conversation. On a first question it does nothing; on
   a follow-up, one short model call rewrites it as a standalone question.
5. Retrieval embeds that question (`embeddings nomic-embed-text` span) and asks PostgreSQL for
   the closest chunks by cosine distance (`SELECT rag.chunks` database span, HNSW index).
   If nothing scores above `OBSLAB_MIN_SCORE`, the question is rewritten once, then the
   flow falls back to "I don't know".
6. Generation streams the answer (`chat llama3.2:3b` span, with time to first chunk and
   token usage), then a cheap groundedness score is computed.
7. The turn (question and answer) is saved with the conversation, and the API returns the
   answer, its sources, the conversation id and the trace ID.

Everything the pod emits goes over OTLP/HTTP to the Collector, which adds Kubernetes metadata
(pod, deployment, node) and sends traces to Tempo, logs to Loki, and exposes metrics that
Prometheus scrapes. The same trace ID is on the spans, on the log records and on histogram
exemplars, so Grafana can jump between the three.

## Conversations

`POST /ask` accepts a `conversation_id`. Without one, a new conversation starts and its id comes
back with the answer; sending it with the next question continues the conversation.

```
POST /ask {"question": "How do taints and tolerations work?"}
  -> {"answer": "...", "conversation_id": "6f0c...", "turn": 1}
POST /ask {"question": "And what about NoExecute?", "conversation_id": "6f0c..."}
  -> {"answer": "...", "turn": 2, "standalone_question": "What does the NoExecute taint effect do?"}
```

A conversation is a LangGraph thread. The graph is compiled with a PostgreSQL checkpointer, so
the state of a thread is loaded when its id comes back, on whichever replica gets the request.

The history has one job: turning a follow-up into a standalone question. That question is what
gets searched, and the answer is written from the retrieved notes only, as for a first question.
So the answer stays grounded in the notes, and the generation prompt does not grow with the
conversation.

A small model sometimes rewrites a follow-up into a question about the previous subject. A
rewrite should only add what "it" or "that" stood for, so it is accepted only if it still
contains the words of the follow-up; otherwise the question is searched as typed. The answer
returns the question that was searched (`standalone_question`).

What keeps the cost bounded:

| Rule | Setting | Effect |
|---|---|---|
| No model call on a first question | | a one-off question costs what it did before |
| A few turns of history, answers clipped | `OBSLAB_HISTORY_TURNS` (3) | the condense prompt has a fixed upper size |
| One checkpoint per answered question | | one database write per turn, not one per graph node |
| No retrieved chunk in a checkpoint | | a saved turn is a few hundred bytes of text |
| Maximum length | `OBSLAB_MAX_TURNS` (20) | the API answers 409 and asks for a new conversation |
| Retention | `OBSLAB_CONVERSATION_TTL_DAYS` (7) | a nightly job deletes idle conversations |

A turn counts once it is answered: when a model call fails, the history and the turn count stay
as they were, and the client can retry.

The conversation id is a UUID. It is written on the workflow span (`gen_ai.conversation.id`) and
on the log line, never on a metric: one label value per conversation would be unbounded.

In the database the conversations have their own schema (`chat`) and their own role
(`obslab_chat`), which cannot read the index. The API keeps its read-only role for the index.

## The index

The ingest job (a suspended CronJob, triggered by `task app:ingest`) reads the notes from the data
lake, embeds them with the configured model and rebuilds `rag.chunks` in a staging table.
The new table and its metadata (model, dimension, chunk count) replace the old ones in a single
transaction, so the API answers from the old index until the new one is complete. The API only
has a read-only role; `/readyz` reports 503 when the database is unreachable, empty, or holds an
index built with another model.

What protects the index:

- The API connects with a role that can only read. Only the ingest job can write.
- A re-index is swapped in atomically, so a reader never sees half of it.
- The index records which embedding model built it. Vectors from two models are not comparable,
  so a mismatch makes the API unready, with the reason in `/readyz`.
- Every query has a timeout, `k` is capped, and the SQL is parameterized.
- Connections are checked before use, so the API recovers by itself after a database restart.
- Only the API and the ingest job can reach the database (NetworkPolicy).
- Errors returned to clients carry an error type, never connection details.

`tests/e2e/test_pgvector.py` exercises each of these against a real PostgreSQL.

## The signals

| Signal | Examples | Where |
|---|---|---|
| GenAI metrics | `gen_ai.client.operation.duration`, `gen_ai.client.inference.usage.{input,output}_tokens`, time to first chunk | Prometheus |
| RAG metrics | retrieval top score, relevant documents, rewrites, fallbacks, groundedness, node duration, node retries, admission wait, waiting line, dropped requests | Prometheus |
| HTTP metrics | `http.server.request.duration` (stable semantic conventions) | Prometheus |
| Traces | one tree per question, GenAI attributes on each model call, a database span per vector query | Tempo |
| Logs | one record per answer with route, rewrites and groundedness, plus the trace ID | Loki |

Metric labels are bounded (operation, model, node, reason, error type). Prompts and answers
only appear on spans when `OBSLAB_CAPTURE_CONTENT=true`.

Two kinds of retry exist in a LangGraph workflow, and both are observable here:

| Retry | What happens | In a trace | In metrics |
|---|---|---|---|
| A loop in the graph (`rewrite`, then `retrieve` again) | the node runs again in a new step | a second `rag.node retrieve` span, `langgraph.attempt` 1 | `rag.query.rewrites` |
| A `RetryPolicy` on a node | the node runs again inside the same step | one `rag.node` span per attempt, numbered by `langgraph.attempt`; failed attempts carry the error | `rag.graph.node.retries` |

The node spans come from the LangChain callbacks, which fire for every attempt. LangGraph's task
stream reports the start and the final result of a retried node only, so it cannot be used to see
the attempts. No node of the lab uses a `RetryPolicy` today; a unit test covers the case.

## Deployment options

| | Cluster, `local` overlay | Cluster, `fake` overlay |
|---|---|---|
| Models | Ollama on the host | built-in fakes |
| RAM | ~4 GB + Ollama | ~4 GB |
| Start | `task up` | `task up OVERLAY=fake` |

The lab runs on the cluster. `deploy/compose/` is a test harness: the Collector, Prometheus,
Tempo, Loki and PostgreSQL without Kubernetes, so that the telemetry path and the database code
are tested in about a minute on every push. It has no Grafana and is not meant to be used as
a second way to run the lab. Both sides read the same SLO rules and database setup from
`deploy/shared/`.

## Design choices

**OpenTelemetry only, with the GenAI conventions.** One standard for the app, the platform and
the model calls, and any backend can read it. The GenAI conventions are still evolving, so every
metric and span name lives in one module (`src/obslab/telemetry/genai.py`).

**A Collector between the app and the backends.** The app only knows one OTLP endpoint. Routing,
batching, Kubernetes metadata and attribute filtering happen in the Collector, and changing a
backend does not touch the app.

**LangGraph instead of a single chain.** Each branch (retrieve, rewrite, generate, fallback,
grade) is a named node, so it gets its own span and its own latency series. "Why was this answer
bad" becomes a query.

**Hand-written instrumentation.** A callback handler maps LangChain and LangGraph events to spans
and metrics. It is about 200 lines, and every attribute in a trace can be traced back to a line
of code.

**Bounded metric labels, content off by default.** Labels are small enumerations (operation,
model, node, error type). Question text and identifiers never become labels, because every new
value is a new time series. In the cluster, the Collector also drops per-pod identifiers that
change on each rollout from metrics, while traces and logs keep them. Prompts and answers are
only recorded on spans when `OBSLAB_CAPTURE_CONTENT=true`.

**PostgreSQL + pgvector.** The index is shared, so the API is stateless: two replicas, rolling
updates without downtime, re-indexing without a restart. It is the only copy of the index: nothing is kept in a local file. The unit tests replace it
with an in-memory double.

**Conversation memory in PostgreSQL, used for the question only.** The LangGraph checkpointer
with PostgreSQL makes the history available to every replica with no new service. The history
resolves the follow-up and is not passed to the answer: passing it would grow every prompt
with the conversation and let the model answer from its own earlier answers instead of the
notes. The trade-off: a request such as "repeat your last answer" is searched again in the notes
instead of being read back from the history.

**Compose for tests, Kubernetes for the lab.** Every push needs an answer in a minute or two,
and a cluster with the whole platform takes several. So the fast end-to-end test runs against
the same backends started with Compose, and the cluster suite runs less often. To limit what
has to be kept in step, the Compose side holds only what the tests query, and the rules and the
database setup are shared files.

**A data lake for the notes.** The notes are data, so they stay out of the image: a folder on
the host (`DATALAKE`, default `datalake/`), mounted read-only into the cluster nodes and handed
to the ingest job through a PersistentVolumeClaim. Changing notes means re-running the ingest
job, with the same image and the same API pods. Only the ingest job mounts the folder. An empty
or missing folder fails the job before the database is touched, so the current index keeps
serving. In production the same job would read from object storage; a folder keeps the lab to
one less service.

**The platform starts in the background.** `task up` submits the four Helm releases without
waiting, builds and deploys the app while their images are pulled, and checks at the end that
everything is Ready (`task platform:wait`). Grafana keeps its own database in memory: its
dashboards and datasources are code, and its first-start migrations take seconds there instead
of minutes on a virtual disk. The image is built after the cluster exists, not during its
creation: on a small machine the two compete for the Docker engine and node creation times out.

**k3d for the cluster.** k3s in Docker gives a real control-plane and worker split for about
500 MB per node, with Traefik, storage and a load balancer included. `task stop` gives the
memory back in seconds.

**Pinned Helm charts for the platform, Kustomize for the app.** Charts are how these components
are usually consumed, and a pinned version makes an upgrade a one-line change. The app is small
enough for plain manifests with two overlays (`local`, `fake`).

**Gateway API instead of Ingress.** The platform owns the listener, each app owns its route.
Traefik implements it; the CRDs are the ones bundled with k3s.

**Ollama on the host.** The GPU stays where its drivers are. The cluster reaches Ollama through
an `ExternalName` Service and calls it by its full cluster name, which Ollama's host check
accepts.

**Restricted pods.** Every pod in the app namespace, PostgreSQL included, runs under the
restricted Pod Security Standard: non-root, read-only root filesystem, no capabilities. Memory
limits but no CPU limits, since throttling hurts latency more than it protects.

**Published ports on 127.0.0.1 only.** The API has no authentication and the UIs are plain
HTTP, which is fine on one machine and not on a shared network. Docker publishes a port on
every interface unless it is given a host address, so the cluster load balancer, the registry
and the Compose stack all name 127.0.0.1. A unit test fails if a published port does not.

**A bounded line in front of the model.** Two API replicas accept more questions than one local
model can answer in time, and the readiness probe says nothing about that backlog. Without a
bound the surplus queues inside the model server, unmeasured, and a question whose caller gave
up still gets answered. So each replica lets `OBSLAB_MODEL_CONCURRENCY` model calls run and
`OBSLAB_MODEL_QUEUE` wait; one more is refused at once with a 503 and `Retry-After`. A question
has `OBSLAB_REQUEST_DEADLINE_S` in total: still waiting at that point it is dropped before the
model is asked, and an answer still being written is cut by closing the stream (504). The wait
is its own number, on the node span (`rag.admission.wait_s`) and in `rag.admission.wait`, next
to the model's execution time. The probes are unchanged, since a busy model is not an unready
pod.

**The caller says how long it will wait.** A fixed deadline is either too long for an impatient
caller or too short for the first question after a cold start. So a caller can send
`X-Request-Timeout`, in seconds, and the deadline of that question is the smaller of this value
and `OBSLAB_REQUEST_DEADLINE_S`. The server value stays the upper bound; a value that is not a
number of at least one second is a 422. The command-line client sends its own timeout. The
deadline in use is on the request span (`rag.request.deadline_s`). Sizing rule for a caller that
sends nothing: queue x typical answer time < deadline < the caller's timeout.

**The bound stays per replica.** Two replicas admit twice as much in front of one model. A
shared limit would need a round trip to PostgreSQL for every model call, and a slot held by a
replica that crashed would have to expire. With a fixed number of replicas the per-replica value
is simply set with that number in mind (`deployment.yaml`, `configmap.yaml`); to be revisited if
the API is ever scaled automatically.

**A caller that disconnects is not detected.** `/ask` is a synchronous endpoint and the deadline
is its only way out. An asynchronous endpoint would free the model as soon as the connection
closes, at the price of moving the graph, the database calls and the telemetry context to async.
A caller that announces its timeout is already never served beyond it, which covers most of the
case for much less.

**Readiness without Ollama.** `/readyz` checks what the app owns: its graph and its index. It
does not call Ollama, because a shared dependency being down would empty the Service on every
replica at once; that case shows up as errors and alerts instead.

**A mode without models.** `OVERLAY=fake` swaps the models for deterministic fakes, so the whole
pipeline runs on a machine without a GPU, and in CI.
