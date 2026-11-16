# <img src="docs/img/logo.svg" width="44" align="top" alt=""> rag-observability-k8s-lab

[![CI](https://github.com/jmiliamine/rag-observability-k8s-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/jmiliamine/rag-observability-k8s-lab/actions/workflows/ci.yml)
[![Real model](https://github.com/jmiliamine/rag-observability-k8s-lab/actions/workflows/real-model.yml/badge.svg?branch=main)](https://github.com/jmiliamine/rag-observability-k8s-lab/actions/workflows/real-model.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![OpenTelemetry GenAI conventions](https://img.shields.io/badge/OpenTelemetry-GenAI%20conventions-f5a800?logo=opentelemetry&logoColor=white)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
[![Security scans](https://img.shields.io/badge/scans-zizmor%20%C2%B7%20gitleaks%20%C2%B7%20trivy-2f7366)](.github/SECURITY.md)
[![LinkedIn: Amine Jmili](https://img.shields.io/badge/LinkedIn-Amine%20Jmili-0a66c2)](https://www.linkedin.com/in/jmiliamine/)

A RAG over your own notes, built with LangGraph, instrumented with the
[OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/),
and observed with Prometheus, Tempo, Loki and Grafana. The notes are searched with pgvector.
Everything runs in a two-node k3d cluster set up like a small company platform: Gateway API,
pinned Helm charts, restricted pods, SLO alerts.

The design is explained step by step in an article:
[Private LangGraph RAG on Kubernetes with Full Observability](https://medium.com/@jmiliamine/private-langgraph-rag-on-kubernetes-with-full-observability-fc3176d87292).

One question, followed from the terminal to its trace:

1. **Ask the RAG a question.** The answer comes back with its sources, a groundedness
   score and a `trace_id`.
2. **Monitor it in Grafana.** Latency per LangGraph node, tokens per second, time to first
   chunk, similarity score of the retrieved notes, groundedness of the answers.
3. **Follow its trace.** The cascade of spans for that same question: retrieval, embedding,
   SQL query, model call, with the token counts on the model span.

![A question asked from a terminal, then its latency, token and retrieval metrics on the Grafana dashboard, then the same question as a trace in Tempo with the token counts on the model call](docs/img/demo.gif)

Most LLM demos stop at the answer. This lab is about what happens around it. For every
question you can see how long each step took, how many tokens it used, why it ended in
"I don't know", and which pod served it. The same trace ID links the metric, the trace and the log line.

## Architecture

![A question goes through the Gateway to the RAG API. The API searches the pgvector index in PostgreSQL, keeps the conversation history there too, calls Ollama for embeddings and answers, and sends traces, metrics and logs to the OpenTelemetry Collector, which forwards them to Tempo, Loki and Prometheus. Grafana reads all three. The ingest job reads the notes from the data lake, a host folder mounted read-only, and writes the index.](docs/img/architecture.svg)

More detail, including what each namespace owns and how a request is traced, in
[docs/architecture.md](docs/architecture.md).

## What is in the box

- **The app.** A LangGraph flow: `condense → retrieve → generate → grade`, with one query rewrite
  before giving up. Served by FastAPI (two replicas), local models through Ollama.
- **Conversations.** A follow-up such as "and what about NoExecute?" is understood from the
  previous turns. The history is a LangGraph thread checkpointed in PostgreSQL, so either
  replica can continue a conversation. It is bounded: a few turns kept, one write per question,
  a maximum length, and a nightly purge.
- **The vector store.** PostgreSQL with pgvector. The API connects with a read-only role,
  and the ingest job rebuilds the index and swaps it in atomically. An index built with another
  embedding model is refused.
- **The telemetry.** One trace per question, from the HTTP request down to each model call,
  named after the GenAI conventions (`chat llama3.2:3b`, `gen_ai.client.operation.duration`...).
  Prompts stay off the traces unless you opt in.

  ![One question as a trace in Tempo: the HTTP request, the LangGraph nodes, the embedding call and the llama3.2:3b call](docs/img/trace.png)
- **The platform.** kube-prometheus-stack, Tempo, Loki and the OpenTelemetry Collector
  from pinned Helm charts, Traefik as the Gateway API implementation, a local image registry.
- **The operations side.** SLO recording rules and alerts,
  a dashboard generated from code, liveness and readiness probes that answer different
  questions.

## Quick start

Tested on Windows with Docker Desktop (WSL2 backend). Linux and macOS should work (every command
goes through [Task](https://taskfile.dev)), but have not been tested yet.

You need Docker Desktop with about **10 GB of memory** for its VM, and these tools:

```bash
winget install Docker.DockerDesktop Task.Task k3d.k3d Kubernetes.kubectl Helm.Helm astral-sh.uv
```

Ollama is optional. Without it, the lab runs with built-in fake models (see below).

```bash
winget install Ollama.Ollama
```

```bash
ollama pull llama3.2:3b
```

```bash
ollama pull nomic-embed-text
```

Then, from the repository:

```bash
task doctor
```

```bash
task setup
```

```bash
task up
```

`task up` takes 15 to 20 minutes the first time, most of it image downloads: it creates the cluster, installs the
platform, builds the image, starts PostgreSQL, indexes the notes of the data lake (`datalake/`)
into it and deploys the API.
No GPU or no Ollama? Use `task up OVERLAY=fake` instead. Everything is the same except the
answers, which become extracts of the notes instead of generated text.

Send some traffic and open Grafana:

```bash
task load
```

| What | Where |
|---|---|
| Ask a question | `task ask Q="How do taints and tolerations work?"` |
| Have a conversation | `task chat`, then type questions one after the other |
| Grafana | http://grafana.localhost:8080 (user `admin`, password from `task grafana:password`) |
| Prometheus | http://prometheus.localhost:8080 |

The dashboard is in the *LLM Observability* folder. Click a point on a latency panel to jump
to an example trace (exemplars), then from a span to its logs.

![Grafana dashboard of the RAG: request rate, latency percentiles, tokens, retrieval scores](docs/img/dashboard.png)

`task stop` pauses the cluster and gives the memory back; `task start` resumes it with all
its data. `task cluster:down` deletes it.

## Your own notes

The notes live in a data lake: a folder of `.md` or `.txt` files, `datalake/` by default, which
ships with the sample notes. The image holds code only. The ingest job mounts the folder
read-only and writes the index to PostgreSQL; the API reads the index and never sees the files.

Add, edit or remove notes in the folder, then re-index:

```bash
task app:ingest
```

No image is rebuilt and no pod restarts. The API keeps answering from the previous index until
the new one is complete.

To use another folder, give it when the cluster is created (the mount is set at creation):

```bash
task up DATALAKE=D:/path/to/notes
```

The notes are only read from that folder and indexed into a local database. Nothing leaves your
machine, since the models run locally too.

## Tests

The first three run in CI on every push, the cluster level on `main` and every week. The real
models answer the same questions after each merge on `main`, on a hosted runner without a GPU,
in about four minutes. Details in [tests/README.md](tests/README.md).

| Command | What it proves | Needs |
|---|---|---|
| `task test` | The graph, the telemetry (in-memory exporters), the API, the sample corpus, input and error guardrails. Outbound network is blocked. | nothing |
| `task test:pgvector` | Same ranking as the in-memory store, read-only API role, atomic re-index, model mismatch refused, query timeout. The data lake drives the index. A conversation continues on another replica and after a restart, stays small, is purged | Docker (starts PostgreSQL) |
| `task lint` | Ruff, generated files in sync, both Kustomize overlays render | kubectl |
| `task scan` | Security scans: workflow audit, secrets, dependency and image vulnerabilities, manifest misconfigurations | Docker |
| `task test:compose` | Starts the Compose stack, indexes the sample notes, then follows one question: answer, metrics in Prometheus, full trace in Tempo (SQL span included), log line in Loki | Docker |
| `task test:stack` | The same telemetry checks against the cluster's platform | the cluster |
| `task test:k8s` | Through the Gateway: answer from pgvector, both replicas serving, pod identity on the trace, clean metric labels, SLO rules evaluated | the cluster |
| `task eval` | The search finds the right note for the questions of the golden set: recall and MRR not below the committed baseline ([results](docs/results.md)). `PROVIDER=ollama` scores the real embedding model | nothing |
| `task test:ollama` | The real models answer from the right note, understand a follow-up from the previous turn, and off-topic questions still fall back | Ollama |

## Layout

Each code folder has its own README.

```
src/obslab/      the app: RAG graph, models, vector store, telemetry, API, CLI
tests/           unit, integration and end-to-end suites
deploy/k8s/      the cluster: k3d definition, platform (Helm values), app (Kustomize)
deploy/compose/  the backends the tests need, with Docker Compose
deploy/shared/   alert rules, dashboard and database setup used by both
datalake/        the notes to index: the sample notes, or point DATALAKE at your own folder
scripts/         helpers behind the Task commands
docs/            architecture, results and troubleshooting
evals/           the golden set of questions and the retrieval baseline
```

## Limits

- One PostgreSQL instance, no replication and no scheduled backups. Fine for a lab.
- Questions and answers are kept in PostgreSQL for the conversation history, 7 days by default
  (`OBSLAB_CONVERSATION_TTL_DAYS`).
- Local use only. The UIs have no TLS.
- A 3B model answers like a 3B model. The point is the telemetry around it, not the answers.

Having trouble? See [docs/troubleshooting.md](docs/troubleshooting.md).

## License

MIT, see [LICENSE](LICENSE). The models are not part of this repository and have their own
licenses: Llama 3.2 (Llama 3.2 Community License) and nomic-embed-text (Apache 2.0).
