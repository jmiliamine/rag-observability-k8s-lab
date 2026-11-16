# scripts

Small helpers called by the Task commands. Plain Python so they behave the same on Windows,
macOS and Linux.

| Script | Called by | What it does |
|---|---|---|
| `doctor.py` | `task doctor` | Checks the tools, Docker memory, free ports, Ollama and the kubectl/server version gap |
| `gen_observability.py` | `task gen`, `task lint` | Builds the Grafana dashboard and wraps the alert rules and the database init script into Kubernetes objects. `--check` fails if the generated files are out of date |
| `datalake_path.py` | `task cluster:up` | Prints the absolute path of the data lake folder for the k3d volume mount, and fails if the folder does not exist |
| `calibrate_checker.py` | `task eval:checker` | Scores a grounding checker (a model that says whether a note supports an answer) on the labelled answers of `evals/answers.jsonl`, and says whether it passes the bar fixed for it. With `RUN=file.json`, lists the answers of an evaluation run that their note does not support |
| `set_image_tag.py` | `task app:build` | Writes the git-ignored overlay that pins the image tag of the last build |
