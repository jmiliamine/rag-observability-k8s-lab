"""obslab <command>

  ingest [--source DIR]          chunk + embed a folder of .md/.txt notes into the index
  ask "question" [--url ...]     one question, prints answer + trace id (in-process, or a running API);
                                 --conversation ID continues a previous answer
  chat [--url ...]               several questions in one conversation, typed one after the other
  conversations purge [--days N] delete the conversations idle for more than N days
  serve [--port 8000]            HTTP API (POST /ask)
  load [--n 30] [--url ...] [--host-header H]   traffic generator (in-domain + off-topic questions);
                                 --concurrency N sends N at once, --timeout S is how long a caller waits
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

QUESTIONS_IN = [
    "What is cosine similarity used for in vector search?",
    "How does a Kubernetes liveness probe differ from a readiness probe?",
    "How do you back up and restore etcd?",
    "How do taints and tolerations work?",
    "What does chunk overlap change in a RAG pipeline?",
    "What is a multi-window burn rate alert?",
    "Why should metric labels have low cardinality?",
    "What does the OpenTelemetry Collector do?",
]
# Follow-ups that only make sense after the question they are attached to: the load generator
# sends them in the same conversation, so the dashboards show the condense step too.
FOLLOW_UPS = {
    "How do taints and tolerations work?": "And what effect does NoExecute have on them?",
    "How do you back up and restore etcd?": "How often should that be done?",
    "What is a multi-window burn rate alert?": "Why does it use two windows?",
}
QUESTIONS_OFF = [
    "What is the best recipe for a chocolate cake?",
    "Who won the 1998 football world cup?",
]


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")


def _components(settings=None, telemetry=None):
    from .app import build_components
    from .config import Settings
    from .telemetry import init_telemetry
    settings = settings or Settings()
    telemetry = telemetry or init_telemetry(settings, set_global=True)
    return build_components(settings, telemetry)


def cmd_ingest(a) -> int:
    from .config import Settings
    from .rag import corpus, pgvector
    from .rag.providers import embeddings
    from .telemetry import GenAIMetrics, emit, init_telemetry
    settings = Settings()
    tel = init_telemetry(settings, set_global=True)

    def failed(reason: str, message: str) -> int:
        emit("ingest.failed", {"obslab.ingest.reason": reason})
        tel.shutdown()
        print(message, file=sys.stderr)
        return 1
    source = Path(a.source)
    if not source.is_dir():
        return failed("source_missing", f"data lake not found: {a.source} is not a folder")
    docs = corpus.split(corpus.load_folder(source))
    if not docs:        # checked before the database is touched: the current index keeps serving
        return failed("no_notes", f"no .md/.txt notes under {a.source}: the index is left as it is")
    meta = {"provider": settings.provider, "embed_model": settings.embed_model_id}
    emb = embeddings(settings, tel, GenAIMetrics(tel.meter))
    start = time.perf_counter()
    with tel.tracer.start_as_current_span("ingest", attributes={"obslab.chunks": len(docs)}):
        with pgvector.open_pool(name="obslab-ingest") as pool:
            pgvector.rebuild(pool, docs, emb, meta)
        seconds = time.perf_counter() - start
        emit("ingest.finished", {"obslab.chunks": len(docs), "obslab.duration_s": round(seconds, 2)})
    tel.shutdown()
    print(f"indexed {len(docs)} chunks in {seconds:.1f}s -> PostgreSQL (rag.chunks)")
    return 0


def _post_ask(url: str, question: str, host_header: str = "", timeout: float = 180,
              conversation_id: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if host_header:
        # Behind the cluster Gateway, routing is by Host header (rag.localhost); Windows does
        # not resolve *.localhost, so we hit 127.0.0.1 and send the Host header ourselves.
        headers["Host"] = host_header
    body = {"question": question} | ({"conversation_id": conversation_id} if conversation_id else {})
    req = urllib.request.Request(f"{url}/ask", data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def cmd_ask(a) -> int:
    if a.url:                       # ask the deployed API instead of running the graph here
        print(json.dumps(_post_ask(a.url, a.question, a.host_header, conversation_id=a.conversation),
                         indent=2, ensure_ascii=False))
        return 0
    c = _components()
    with c.telemetry.tracer.start_as_current_span("cli.ask") as span:
        result = c.ask(a.question, a.conversation)
        result["trace_id"] = format(span.get_span_context().trace_id, "032x")
    c.close()
    c.telemetry.shutdown()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_chat(a) -> int:
    """Questions typed one after the other, in one conversation."""
    conversation = a.conversation
    print("Ask a question (empty line or Ctrl+C to leave).")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        try:
            r = _post_ask(a.url, question, a.host_header, conversation_id=conversation)
        except urllib.error.HTTPError as e:
            print(f"error {e.code}: {e.read().decode('utf-8', 'replace')[:300]}", file=sys.stderr)
            continue
        conversation = r["conversation_id"]
        if r["standalone_question"] != question:
            print(f"  (understood as: {r['standalone_question']})")
        print(f"{r['answer']}\n  [{r['route']}, turn {r['turn']}, sources: {', '.join(r['sources']) or 'none'}]")
    if conversation:
        print(f"conversation {conversation}")
    return 0


def cmd_conversations(a) -> int:
    """Housekeeping of the stored conversations (run daily by a CronJob in the cluster)."""
    from .config import Settings
    from .rag import memory
    from .telemetry import emit, init_telemetry
    settings = Settings()
    tel = init_telemetry(settings, set_global=True)     # the job's event reaches Loki like the API's
    store = memory.ConversationStore(memory.open_pool(name="obslab-purge"))
    try:
        days = a.days if a.days is not None else settings.conversation_ttl_days
        removed, kept = store.purge(days), store.count()
        emit("conversations.purged", {"obslab.conversations.removed": removed, "obslab.conversations.kept": kept,
                                      "obslab.conversations.ttl_days": float(days)})
        print(f"purged {removed} conversations idle for more than {days:g} days, {kept} left")
    finally:
        store.close()
        tel.shutdown()
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    from .api import create_app
    uvicorn.run(create_app(_components()), host=a.host, port=a.port)
    return 0


def _load_one(a, i: int) -> list[tuple[str, float]]:
    """One simulated caller: a question, then its follow-up when it has one. Returns what happened
    to each request as (outcome, seconds); the outcome is ok, busy (503), deadline (504) or error."""
    q = random.choice(QUESTIONS_OFF if random.random() < a.off_topic else QUESTIONS_IN)
    out, conversation = [], None
    for question in (q, FOLLOW_UPS.get(q)):         # a follow-up, when the question has one
        if question is None:
            break
        t = time.perf_counter()
        try:
            body = _post_ask(a.url, question, a.host_header, timeout=a.timeout, conversation_id=conversation)
            conversation = body.get("conversation_id")
            outcome, note = "ok", body.get("route")
        except urllib.error.HTTPError as e:
            outcome, note = {503: "busy", 504: "deadline"}.get(e.code, "error"), f"HTTP {e.code}"
        except Exception as e:                      # the caller gave up (timeout) or the API is down
            outcome, note = "error", type(e).__name__
        elapsed = time.perf_counter() - t
        out.append((outcome, elapsed))
        print(f"{i + 1:>3} {elapsed:5.1f}s {outcome:<8} {note:<9} "
              f"{'  + ' if question != q else ''}{question[:60]}", flush=True)
        if outcome != "ok":
            break
        time.sleep(a.pause)
    return out


def cmd_load(a) -> int:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results = [r for rs in pool.map(lambda i: _load_one(a, i), range(a.n)) for r in rs]
    count = {k: sum(1 for o, _ in results if o == k) for k in ("ok", "busy", "deadline", "error")}
    times = sorted(t for o, t in results if o == "ok")
    p50 = times[len(times) // 2] if times else 0.0
    print(f"done: {count['ok']} ok (p50 {p50:.1f}s, max {times[-1] if times else 0.0:.1f}s), "
          f"{count['busy']} busy, {count['deadline']} past deadline, {count['error']} errors")
    return 0 if count["error"] == 0 else 1


def main(argv=None) -> int:
    _utf8()
    ap = argparse.ArgumentParser(prog="obslab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest")
    s.add_argument("--source", default=os.environ.get("OBSLAB_DATALAKE", "datalake"),
                   help="the data lake: a folder of .md/.txt notes (default: $OBSLAB_DATALAKE, else datalake)")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("ask")
    s.add_argument("question")
    s.add_argument("--url", default="", help="ask a running API (e.g. http://127.0.0.1:8080) instead of in-process")
    s.add_argument("--host-header", default="", help="e.g. rag.localhost:8080 when --url is the Gateway IP")
    s.add_argument("--conversation", default=None, help="conversation_id of a previous answer, to continue it")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("chat", help="ask several questions in one conversation")
    s.add_argument("--url", default="http://127.0.0.1:8000")
    s.add_argument("--host-header", default="", help="e.g. rag.localhost:8080 when --url is the Gateway IP")
    s.add_argument("--conversation", default=None, help="continue an existing conversation")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("conversations", help="housekeeping of the stored conversations")
    s.add_argument("action", choices=["purge"])
    s.add_argument("--days", type=float, default=None,
                   help="delete conversations idle for more than this (default: OBSLAB_CONVERSATION_TTL_DAYS)")
    s.set_defaults(fn=cmd_conversations)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("load")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--url", default="http://127.0.0.1:8000")
    s.add_argument("--off-topic", type=float, default=0.2)
    s.add_argument("--pause", type=float, default=0.5)
    s.add_argument("--concurrency", type=int, default=1, help="callers at the same time (a burst)")
    s.add_argument("--timeout", type=float, default=180, help="seconds a caller waits before giving up")
    s.add_argument("--host-header", default="", help="e.g. rag.localhost:8080 when --url is the Gateway IP")
    s.set_defaults(fn=cmd_load)

    a = ap.parse_args(argv)
    return a.fn(a)

if __name__ == "__main__":
    sys.exit(main())
