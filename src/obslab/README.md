# obslab

The application: about 1,400 lines of Python. A good reading order is `rag/graph.py`, then
`telemetry/callbacks.py`, then `api.py`.

```
obslab/
├── cli.py            the `obslab` command: ingest, ask, chat, serve, load, eval, conversations purge
├── api.py            FastAPI app: POST /ask, /healthz, /readyz
├── app.py            wires everything together once (models, vector store, graph)
├── config.py         every setting, read from environment variables and validated
├── evaluation.py     scores the search and the answers on questions with a known source
├── rag/
│   ├── graph.py          the LangGraph flow: condense, retrieve, rewrite, generate, fallback, grade
│   ├── memory.py         conversation history: LangGraph checkpoints in PostgreSQL, purge
│   ├── providers.py      models: Ollama, or deterministic fakes for tests
│   ├── pgvector.py       vector store on PostgreSQL (search for the API, rebuild for the ingest job)
│   └── corpus.py         loads a folder of notes and splits it into chunks
└── telemetry/
    ├── setup.py          OpenTelemetry providers and exporters (traces, metrics, logs)
    ├── genai.py          every metric and attribute name, in one place
    └── callbacks.py      turns LangChain and LangGraph events into spans and metrics
```

## How a question flows through the code

1. `api.py` receives `POST /ask` and calls `Components.ask()` (`app.py`).
2. `rag/graph.py` runs the graph. `retrieve` embeds the question and searches the vector store;
   if nothing is relevant enough the question is rewritten once, then the graph falls back to
   "I don't know". Otherwise `generate` streams an answer and `grade` scores how much of it
   comes from the retrieved text.
3. While the graph runs, `telemetry/callbacks.py` opens a span per node and per model call and
   records durations, token counts and time to first chunk, using the names in
   `telemetry/genai.py`.
4. `telemetry/setup.py` exports all of it over OTLP.

## Running it without the cluster

```bash
task stack:up
```

```bash
task stack:ingest PROVIDER=fake
```

Then, with `PGHOST=127.0.0.1 PGDATABASE=obslab PGUSER=obslab_reader PGPASSWORD=reader-local`
and `OBSLAB_PROVIDER=fake OBSLAB_MIN_SCORE=0.2` in the environment:

```bash
uv run obslab ask "How do taints and tolerations work?"
```
