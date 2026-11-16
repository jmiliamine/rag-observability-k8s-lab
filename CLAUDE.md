# Notes for coding agents

This file tells an AI coding agent (Claude Code, Codex, Cursor...) how to install, run and
change this repository. Humans can read it too; the README is the friendlier version.

## Installing the lab for a user

Work through these steps in order and stop at the first failure. Show the user the failing
output instead of guessing.

1. `task doctor`: checks the tools, Docker memory (about 10 GB needed), free ports
   (8080, 4318, 5001) and Ollama. If a tool is missing on Windows, suggest the `winget`
   line from the README. Do not change Docker or WSL settings yourself: tell the user
   what to change.
2. `task setup`: uv installs Python 3.14 and the dependencies into `.venv`.
3. `task test`: must pass with no network. If it fails here, the problem is in the code,
   not the infrastructure. `task test:pgvector` then checks the vector store against a real
   PostgreSQL, and `task test:compose` the whole telemetry path with Docker Compose (Docker only).
4. **Ask the user before creating the cluster.** `task up` uses about 4 GB of RAM and
   15 to 20 minutes, and binds the ports above. Use `task up OVERLAY=fake` when Ollama is not
   installed or the models are not pulled (`task doctor` says so).
5. `task test:k8s`, then `task load` to feed the dashboards. Grafana:
   http://grafana.localhost:8080, password from `task grafana:password`. Never print the
   password in a summary or commit it anywhere.

`task stop` frees the memory when the user is done; `task cluster:down` deletes everything
and asks for confirmation. Do not run it without the user's explicit request.

## Changing the code

- Entry points are Taskfile tasks (`task --list`). Keep them idempotent and cross-platform
  (no bash-only syntax in `cmds`; small Python scripts in `scripts/` instead).
- Metric and span names follow the OpenTelemetry GenAI conventions and are defined only in
  `src/obslab/telemetry/genai.py`.
- Never add unbounded metric attributes (question text, ids, pod UIDs). Content goes on spans
  only, behind `OBSLAB_CAPTURE_CONTENT` (off by default).
- Dashboard and SLO rules: edit `scripts/gen_observability.py` or `deploy/shared/`, then run
  `task gen`. Never edit `deploy/k8s/platform/generated/` or the dashboard JSON by hand;
  `task lint` fails when they are out of sync.
- Kubernetes: platform = pinned Helm charts (versions in `Taskfile.yml`, values in
  `deploy/k8s/platform/values/`); app = Kustomize in `deploy/k8s/app` (overlays
  `local` and `fake`; `_build` is generated and git-ignored). Every manifest explains *why*
  in comments; keep that.
- Security baseline: restricted Pod Security, non-root, read-only root
  filesystem, requests on every container, memory limits, no CPU limits.
- Vector store: PostgreSQL + pgvector, code in `src/obslab/rag/pgvector.py`.
  The API uses the read-only role, only the ingest job writes. Keep SQL parameterized, table
  names constant, and every guardrail covered by `tests/e2e/test_pgvector.py`. Never print or
  log a connection string with a password; connection settings come from the `PG*` variables.
  Database passwords live in the `obslab-db` Secret (`task db:secret`, generated, never printed).
- A new component or trade-off gets a short paragraph in `docs/architecture.md` (Design choices).
- If you change the sample notes in `datalake/` or the questions in `src/obslab/cli.py`, run `task test`:
  `tests/unit/test_sample_corpus.py` checks that every load-test question is still answered.
- A change to the search (chunking, embedding model, threshold, ranking) is judged on the golden
  set: `task eval`, and `task eval PROVIDER=ollama` when Ollama runs. Record the before and after
  in `docs/results.md`; update `evals/baseline.json` with `UPDATE=1` only with a reason
  (`evals/README.md`).

## Before you finish

Run `task lint` and `task test`. Both must pass. If you touched the Dockerfile, a manifest,
a dependency or a workflow, run `task scan` too (Docker needed). Run the e2e tasks only if the infrastructure
is up, and say which ones you ran.

## Windows specifics

- `*.localhost` does not resolve outside browsers on Windows: the tests and the CLI connect
  to 127.0.0.1 and send the `Host` header (`tests/e2e/lab_http.py`).
- If Python behaves oddly, clear `PYTHONHOME` and `PYTHONPATH` in the shell.
- Files are LF (`.gitattributes`); write them with `newline="\n"` from Python.
- More in `docs/troubleshooting.md`.
