"""Observability as code: the dashboard, the alert rules and the database setup, from one source.

  python scripts/gen_observability.py            write the files below
  python scripts/gen_observability.py --check    fail if they are out of date (CI, `task lint`)

Writes:
  deploy/shared/grafana/llm-rag-overview.json                  dashboard, readable and diffable
  deploy/k8s/platform/generated/grafana-dashboard.yaml         same JSON in a ConfigMap that the
                                                               Grafana sidecar picks up (label grafana_dashboard=1)
  deploy/k8s/platform/generated/prometheusrule-rag-slo.yaml    deploy/shared/prometheus/rag-slo.yml wrapped
                                                               in a PrometheusRule CR for the Prometheus Operator
  deploy/k8s/app/base/postgres-init.generated.yaml     deploy/shared/postgres/init-obslab.sh in a ConfigMap
                                                               (Kustomize cannot read files outside its root)

Why generate instead of hand-writing: the dashboard JSON is ~1 000 lines nobody should edit
by hand, and the rules and database setup are shared with the Compose test harness.

Dashboard rows follow the questions an on-call asks, in order: is it up and fast (RED),
what does the model cost (tokens), is retrieval working, are answers good.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy/shared/grafana/llm-rag-overview.json"
RULES = ROOT / "deploy/shared/prometheus/rag-slo.yml"
GEN = ROOT / "deploy/k8s/platform/generated"
PG_INIT = ROOT / "deploy/shared/postgres/init-obslab.sh"
APP_BASE = ROOT / "deploy/k8s/app/base"
PROM = {"type": "prometheus", "uid": "prometheus"}
LOKI = {"type": "loki", "uid": "loki"}

panels = []
y = 0


def row(title: str) -> None:
    global y
    panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}})
    y += 1


def ts(title: str, targets, unit: str = "short", x: int = 0, w: int = 8, h: int = 8, desc: str = "", kind="timeseries"):
    panels.append({
        "type": kind, "title": title, "description": desc, "datasource": PROM,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}} if kind == "timeseries" else
                   {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value"},
        "targets": [{"expr": e, "legendFormat": legend, "refId": chr(65 + i), "exemplar": kind == "timeseries"}
                    for i, (e, legend) in enumerate(targets)],
    })


def advance(h: int = 8) -> None:
    global y
    y += h


HQ = "histogram_quantile({q}, sum by (le{by}) (rate({m}_bucket[$__rate_interval])))"

row("Service health (RED)")
ts("Questions / s", [("rag:requests:rate5m", "workflow"),
                     ("sum(rate(http_server_request_duration_seconds_count[$__rate_interval]))", "http")], "reqps", 0, 6, 5, kind="stat")
ts("Error ratio", [("rag:errors:ratio5m", "errors")], "percentunit", 6, 6, 5, kind="stat")
ts("p95 end-to-end latency", [("rag:latency:p95_5m", "p95")], "s", 12, 6, 5, kind="stat")
ts("Fallback ratio (no context)", [("rag:fallback:ratio15m", "fallback")], "percentunit", 18, 6, 5, kind="stat")
advance(5)
ts("Workflow latency percentiles", [(HQ.format(q=q, by="", m="gen_ai_invoke_workflow_duration_seconds"), f"p{int(q*100)}")
                                     for q in (0.5, 0.95, 0.99)], "s", 0, 12)
ts("Latency per LangGraph node (p95)", [(HQ.format(q=0.95, by=", langgraph_node", m="rag_graph_node_duration_seconds"),
                                         "{{langgraph_node}}")], "s", 12, 12)
advance()

row("Model calls (OpenTelemetry GenAI conventions)")
ts("LLM / embeddings duration p95", [(HQ.format(q=0.95, by=", gen_ai_operation_name, gen_ai_request_model",
                                                m="gen_ai_client_operation_duration_seconds"),
                                      "{{gen_ai_operation_name}} {{gen_ai_request_model}}")], "s", 0, 8)
ts("Time to first chunk", [(HQ.format(q=q, by="", m="gen_ai_client_operation_time_to_first_chunk_seconds"), f"p{int(q*100)}")
                           for q in (0.5, 0.95)], "s", 8, 8,
   desc="What the user feels before text starts streaming.")
ts("Tokens / s", [("sum by (gen_ai_request_model) (rate(gen_ai_client_inference_usage_input_tokens_total[$__rate_interval]))", "in {{gen_ai_request_model}}"),
                  ("sum by (gen_ai_request_model) (rate(gen_ai_client_inference_usage_output_tokens_total[$__rate_interval]))", "out {{gen_ai_request_model}}")],
   "short", 16, 8)
advance()
ts("Input tokens per call (p50 / p95)", [(HQ.format(q=q, by="", m="gen_ai_client_inference_operation_input_tokens"), f"p{int(q*100)}")
                                         for q in (0.5, 0.95)], "short", 0, 8,
   desc="Prompt size: context stuffing shows up here first.")
ts("API-equivalent cost (last 24h)", [("sum(increase(obslab_llm_cost_usd_total[24h])) or vector(0)", "USD")], "currencyUSD", 8, 8, kind="stat",
   desc="Local model is free; set OBSLAB_PRICE_IN/OUT to see what the same traffic would cost on an API.")
ts("Model errors / s by type", [("sum by (error_type) (rate(gen_ai_client_operation_duration_seconds_count{error_type!=\"\"}[$__rate_interval]))",
                                 "{{error_type}}"),
                                ("vector(0)", "no error")], "reqps", 16, 8)
advance()

row("Retrieval & answer quality")
ts("Top similarity score", [(HQ.format(q=q, by="", m="rag_retrieval_top_score"), f"p{int(q*100)}") for q in (0.1, 0.5, 0.9)],
   "short", 0, 8, desc="A drop means the corpus no longer covers what people ask.")
ts("Relevant documents per question (avg)", [("sum(rate(rag_retrieval_relevant_documents_sum[$__rate_interval])) / sum(rate(rag_retrieval_relevant_documents_count[$__rate_interval]))", "relevant"),
                                              ("sum(rate(rag_retrieval_documents_sum[$__rate_interval])) / sum(rate(rag_retrieval_documents_count[$__rate_interval]))", "retrieved")],
   "short", 8, 8)
ts("Groundedness", [(HQ.format(q=q, by="", m="rag_answer_groundedness"), f"p{int(q*100)}") for q in (0.1, 0.5)], "percentunit", 16, 8,
   desc="Share of answer words found in the retrieved context. It shows answers drifting away from the notes. "
        "It does not show an answer that uses the same words to say the opposite.")
advance()
ts("Query rewrites & fallbacks / min", [("sum(rate(rag_query_rewrites_total[$__rate_interval])) * 60", "rewrites"),
                                        ("sum(rate(rag_fallbacks_total[$__rate_interval])) * 60", "fallbacks")], "short", 0, 12)
# One line per event (telemetry/events.py): its name, which is the body of the log record, then
# the field that says what happened.
EVENT_LINE = '{{__line__}} {{.rag_route}}{{.rag_refusal_reason}}{{.error_type}}{{.obslab_ingest_reason}}'
panels.append({"type": "logs", "title": "Application events (click a trace_id to open the trace)", "datasource": LOKI,
               "gridPos": {"h": 8, "w": 12, "x": 12, "y": y},
               "targets": [{"expr": f'{{service_name="obslab-rag"}} | line_format "{EVENT_LINE}"', "refId": "A"}],
               "options": {"showTime": True, "wrapLogMessage": True}})
advance()

row("Conversations")
ts("Questions / min: first vs follow-up", [
    ('sum(rate(rag_conversation_turn_bucket{le="1.0"}[$__rate_interval])) * 60', "first question"),
    ('(sum(rate(rag_conversation_turn_count[$__rate_interval])) '
     '- sum(rate(rag_conversation_turn_bucket{le="1.0"}[$__rate_interval]))) * 60', "follow-up")], "short", 0, 8,
   desc="A follow-up costs one more short model call (the condense step) before the search.")
ts("Conversation depth (turn number)", [(HQ.format(q=q, by="", m="rag_conversation_turn"), f"p{int(q*100)}")
                                        for q in (0.5, 0.95)], "short", 8, 8,
   desc="Position of the questions in their conversation. Capped by OBSLAB_MAX_TURNS.")
ts("Condense step duration (p50 / p95)", [
    (HQ.format(q=q, by="", m='rag_graph_node_duration_seconds').replace(
        "rag_graph_node_duration_seconds_bucket", 'rag_graph_node_duration_seconds_bucket{langgraph_node="condense"}'),
     f"p{int(q*100)}") for q in (0.5, 0.95)], "s", 16, 8,
   desc="Near zero on a first question (no model call); on a follow-up, the time to rewrite it as a standalone question.")
advance()

row("Model admission (overload)")
ts("Wait for a model slot (p50 / p95)", [(HQ.format(q=q, by="", m="rag_admission_wait_seconds"), f"p{int(q*100)}")
                                           for q in (0.5, 0.95)], "s", 0, 8,
   desc="Time a model call spent waiting for a slot, apart from the model's own execution time. "
        "Near zero unless more questions arrive than the model can answer.")
ts("Model calls waiting now", [("sum(rag_admission_waiting)", "waiting")], "short", 8, 8,
   desc="Length of the waiting line, all replicas together. Bounded by OBSLAB_MODEL_QUEUE per replica.")
ts("Requests dropped / min, by reason", [
    ("sum by (rag_admission_reason) (rate(rag_admission_rejections_total[$__rate_interval])) * 60",
     "{{rag_admission_reason}}")], "short", 16, 8,
   desc="queue_full: refused at once, the waiting line was full (HTTP 503). "
        "deadline: dropped after OBSLAB_REQUEST_DEADLINE_S (HTTP 504).")
advance()

dashboard = {
    "uid": "obslab-rag", "title": "LLM / RAG overview", "tags": ["llm", "rag", "opentelemetry"],
    "timezone": "browser", "schemaVersion": 39, "refresh": "10s",
    "time": {"from": "now-1h", "to": "now"}, "panels": panels,
}
HEADER = "# GENERATED by scripts/gen_observability.py — do not edit, edit the generator or deploy/shared.\n"


def indent(text: str, n: int) -> str:
    """Indent every non-empty line by n spaces (to nest a document inside YAML)."""
    return "\n".join((" " * n + line) if line.strip() else "" for line in text.splitlines())


outputs = {OUT: json.dumps(dashboard, indent=2) + "\n"}

# ConfigMap: the kube-prometheus-stack Grafana sidecar loads every ConfigMap labelled
# grafana_dashboard=1 (any namespace) and the folder annotation groups it in the UI.
outputs[GEN / "grafana-dashboard.yaml"] = HEADER + f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: obslab-dashboard-llm-rag
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
  annotations:
    grafana_folder: LLM Observability
data:
  llm-rag-overview.json: |
{indent(json.dumps(dashboard, indent=2), 4)}
"""

# PrometheusRule: same `groups:` as the compose rule file, wrapped in the Operator's CRD.
rules_body = RULES.read_text(encoding="utf-8")
rules_body = rules_body[rules_body.index("groups:"):]       # drop the file's leading comments
outputs[GEN / "prometheusrule-rag-slo.yaml"] = HEADER + f"""apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: obslab-rag-slo
  namespace: monitoring
spec:
{indent(rules_body, 2)}
"""

# Postgres first-start script, shared with compose, as a ConfigMap mounted in initdb.d.
outputs[APP_BASE / "postgres-init.generated.yaml"] = HEADER + f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-init
data:
  10-obslab.sh: |
{indent(PG_INIT.read_text(encoding="utf-8"), 4)}
"""

if "--check" in sys.argv:
    stale = [p.relative_to(ROOT).as_posix() for p, text in outputs.items()
             if not p.exists() or p.read_text(encoding="utf-8") != text]
    if stale:
        sys.exit(f"out of date, run `task gen`: {', '.join(stale)}")
    print("generated files are up to date")
else:
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:     # LF on Windows too
            f.write(text)
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
