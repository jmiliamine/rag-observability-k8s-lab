# Troubleshooting

Start with `task doctor`. Most problems below show up there first.

## Setup

**`winget install` with several packages fails.** Older winget versions take one package per
command. Run them one by one.

**`uv` or `task` is not found right after installing it.** Open a new terminal: winget
updates the PATH for new shells only.

**Python picks up the wrong packages.** Clear `PYTHONHOME` and `PYTHONPATH` in the shell
(`$env:PYTHONHOME=$null; $env:PYTHONPATH=$null` in PowerShell).

## Cluster

**`task doctor` says Docker has less than 10 GB.** Docker Desktop > Settings > Resources. On
the WSL2 backend the limit comes from `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=10GB
processors=2
```

then `wsl --shutdown` and restart Docker Desktop.

**Port 8080, 4318 or 5001 is already in use.** Another cluster or the Compose stack is
running. `task stack:down`, or `k3d cluster list` to find the other cluster.

**`kubectl` behaves like an old version on Windows.** Windows searches `C:\Windows` before the
PATH when a program starts `kubectl`. `task doctor` warns when a `kubectl.exe` sits there;
remove it from an elevated prompt.

**`http://rag.localhost:8080` works in the browser but not in scripts.** Windows only resolves
`*.localhost` in browsers. Use `127.0.0.1:8080` with a `Host: rag.localhost:8080` header, which
is what `task ask` and the tests do.

**The API pods are Running but not Ready.** Their `/readyz` answer says why:

```bash
kubectl -n obslab port-forward deploy/obslab-api 8000:8000
```

then open http://127.0.0.1:8000/readyz. The usual reasons:

- *index is empty*: the ingest has not run yet. `task app:ingest`, then check the job logs with
  `kubectl -n obslab logs job/<name>`.
- *index built with embed model ...*: the index comes from another embedding model, for
  example after switching overlays. Re-index with `task app:ingest`.
- *vector database unreachable*: `kubectl -n obslab get pods -l app.kubernetes.io/name=postgres`
  and its logs.

**The ingest job fails with an authentication error.** The `obslab-db` Secret and the database
disagree, usually because the Secret was recreated after the database volume was initialised.
Either restore the old Secret, or delete both (`kubectl -n obslab delete secret obslab-db` and
`kubectl -n obslab delete pvc data-postgres-0`, which deletes the index) and run `task app`.

**The ingest job stays in `ContainerCreating`, or fails with "no .md/.txt notes".** The job
reads the data lake, a host folder mounted into the nodes when the cluster was created.
`kubectl -n obslab describe pod -l app.kubernetes.io/name=obslab-ingest` shows a hostPath error
when the nodes have no such mount: the cluster predates the data lake or was created by hand.
Recreate it with `task cluster:down` and `task up DATALAKE=...`. When the job runs but finds no
notes, the folder is empty or holds no `.md` or `.txt` file; the previous index keeps serving.

**`/readyz` reports "conversation store unreachable", or the API logs "password authentication
failed for user obslab_chat".**
The database volume was initialised before conversations existed, so the `obslab_chat` role and
the `chat` schema are missing (the init script only runs on an empty volume). With Compose:
`docker compose -f deploy/compose/docker-compose.yml down -v`, then `task stack:up` and
`task stack:ingest`. In the cluster: delete the `obslab-db` Secret and the `data-postgres-0`
claim as described above, then `task app`.

**`/ask` returns 409.** The conversation reached `OBSLAB_MAX_TURNS`. Send the next question
without `conversation_id` to start a new one.

**Inspect the conversations.** `kubectl -n obslab exec -it postgres-0 -- psql -U postgres -d obslab`,
then `SELECT thread_id, count(*) FROM chat.checkpoints GROUP BY 1;`. `obslab conversations purge
--days 0` (with the chat role) deletes them all.

**Inspect the index.** `kubectl -n obslab exec -it postgres-0 -- psql -U postgres -d obslab`,
then `SELECT * FROM rag.index_meta;` or `SELECT source, count(*) FROM rag.chunks GROUP BY 1;`.

**Every answer is "I don't know".** `OBSLAB_MIN_SCORE` probably does not fit
the embedding model. With `nomic-embed-text` the default is 0.6, with the fakes 0.2.

**Pods are OOMKilled.** `kubectl describe pod` shows the reason. Limits are sized for the
sample notes; a much larger notes folder needs more memory for the ingest job and PostgreSQL
(see `deploy/k8s/app/base/`).

## Ollama

**Pods get 403 from Ollama.** Ollama rejects unknown `Host` headers. The config calls it as
`ollama.obslab.svc.cluster.local`, which it accepts. If you changed
`OLLAMA_BASE_URL`, keep a `.local` or `.internal` name.

**The first answer takes a minute.** Ollama loads the model on the first call. Later calls
are much faster.

**A question is stuck (alert `RagRequestStuck`).** The oldest question in flight is close to
the request deadline. Look in this order. The model server: `ollama ps` shows whether the model
is loaded, and whether part of it runs on the CPU. GPU memory: other programs on the machine
can push the model out of it, and answers then slow down a lot. The waiting line: the
"Model admission" row of the dashboard. A model server that stops sending entirely is cut
after `OBSLAB_MODEL_READ_TIMEOUT_S` (60 s) and the question gets a 504.

## Telemetry

**No data in Grafana.** Send traffic first (`task load`). Metrics arrive after about 30
seconds (10 s export + 15 s scrape).

**`task test:compose` or `task test:stack` fails right after the stack starts.** Tempo answers 503 for about 15 seconds
after its container is healthy. The test waits up to a minute; if it still fails, check
`docker compose -f deploy/compose/docker-compose.yml logs tempo`.

**`task test:pgvector` cannot connect.** Port 5432 may already be taken by another PostgreSQL
on the machine. Stop it, or change the published port in `deploy/compose/docker-compose.yml`
and set `PG_WRITER_DSN` / `PG_READER_DSN` accordingly.
