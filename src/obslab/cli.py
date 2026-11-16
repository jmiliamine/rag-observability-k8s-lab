"""obslab <command>

  ingest [--source DIR]          chunk + embed a folder of .md/.txt notes into the index
  ask "question" [--url ...]     one question, prints answer + trace id (in-process, or a running API);
                                 --conversation ID continues a previous answer
  chat [--url ...]               several questions in one conversation, typed one after the other
  conversations purge [--days N] delete the conversations idle for more than N days
  eval [--check] [--answers]     score the search on the golden set (evals/golden.jsonl): recall, MRR;
                                 --check fails when a metric is below the committed baseline;
                                 --answers asks every question to the models and counts the right answers
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
    docs = corpus.split(corpus.load_folder(source), settings.chunk_size, settings.chunk_overlap)
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
    # The API drops the question once this caller has stopped waiting for it.
    headers = {"Content-Type": "application/json", "X-Request-Timeout": f"{timeout:g}"}
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


def cmd_eval(a) -> int:
    """Score the search, or with --answers the answers, on the golden set (evaluation.py)."""
    import uuid

    from langchain_core.vectorstores import InMemoryVectorStore

    from . import evaluation
    from .app import build_components
    from .config import Settings
    from .rag import corpus
    from .rag.graph import PROMPT_VERSION
    from .rag.providers import embeddings
    from .telemetry import GenAIMetrics, emit, init_telemetry
    settings = Settings()
    mode = "answers" if a.answers else "retrieval"
    if a.answers and (a.check or a.update_baseline):
        # Said before a model is called: the answers of a real model change from run to run.
        print("the baseline is for retrieval: --answers is reported, never gated (drop --check or "
              "--update-baseline)", file=sys.stderr)
        return 2
    golden, notes, baseline_file = Path(a.golden), Path(a.source), Path(a.baseline)
    questions = evaluation.load_golden(golden, notes)
    previous = None
    if a.against:               # read now: a wrong file must not be found out after a run of several minutes
        try:
            previous = json.loads(Path(a.against).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"--against {a.against}: cannot be read ({type(e).__name__})", file=sys.stderr)
            return 2
        if not isinstance(previous, dict) or "scores" not in previous or "mode" not in previous:
            print(f"--against {a.against}: not a --json file of this command (no mode or no scores)", file=sys.stderr)
            return 2
        if previous["mode"] != mode:
            print(f"--against {a.against} is a {previous['mode']} run, this one is a {mode} run", file=sys.stderr)
            return 2
    docs = corpus.split(corpus.load_folder(notes), settings.chunk_size, settings.chunk_overlap)
    model, k = settings.embed_model_id, settings.top_k
    index = corpus.index_id(docs, model)
    for d in docs:              # as the pgvector store does: each result says which index it comes from
        d.metadata["index_id"] = index
    config = {"k": k, "min_score": settings.min_score, "chunk_size": settings.chunk_size,
              "chunk_overlap": settings.chunk_overlap, "golden": evaluation.fingerprint(golden)}
    kinds = [q.kind for q in questions]
    # Each result is also a log event (telemetry/events.py): with OBSLAB_TELEMETRY=otlp a run can
    # be read in Loki next to the traces of its questions. `run` ties the events of one run.
    run = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    print(f"run         {run}")
    print(f"golden set  {golden.as_posix()} ({config['golden']}): {kinds.count('answerable')} questions with a "
          f"source, {kinds.count('no_answer')} without answer, {kinds.count('follow_up')} follow-ups")
    print(f"search      {model}, {len(docs)} chunks of {settings.chunk_size} characters "
          f"({settings.chunk_overlap} shared), first {k} results, threshold {settings.min_score:g}, index {index}")
    # A field of an event holds 64 characters at most: a longer model name is cut, not refused.
    identity = {"obslab.eval.run": run, "obslab.eval.golden": config["golden"], "obslab.embed_model": model[:64],
                "obslab.index.id": index, "rag.retrieval.top_k": k, "rag.retrieval.min_score": settings.min_score}

    def result(name: str, question: str, score: float, label: str) -> None:
        emit("gen_ai.evaluation.result", {
            "gen_ai.evaluation.name": name, "gen_ai.evaluation.score.value": float(score),
            "gen_ai.evaluation.score.label": label, "obslab.eval.question": question, "obslab.eval.run": run})

    tel = init_telemetry(settings)
    try:
        store = InMemoryVectorStore(embeddings(settings, tel, GenAIMetrics(tel.meter)))
        store.add_documents(docs)
        if a.answers:
            chat = "fake-chat" if settings.provider == "fake" else settings.chat_model
            print(f"answers     {chat}, prompts {PROMPT_VERSION}\n")
            components = build_components(settings, tel, store=store)

            def judged(r) -> None:
                result("answer", r.id, float(r.right), "right" if r.right else "wrong")
                print(f"  {r.id:<11} {r.route:<9} {'right' if r.right else 'WRONG':<6} {r.seconds:5.1f}s", flush=True)
            report = evaluation.evaluate_answers(
                components.ask, questions, progress=judged,
                around=lambda q: tel.tracer.start_as_current_span("eval.question", attributes={
                    "obslab.eval.question": q.id, "obslab.eval.run": run}))
            config |= {"chat_model": chat, "prompts": PROMPT_VERSION}
            metrics = report.metrics()
            emit("eval.finished", identity | {
                "obslab.eval.mode": "answers", "obslab.eval.questions": len(report.rows),
                "obslab.eval.right": metrics["right"], "obslab.eval.declined": metrics["declined"],
                "obslab.eval.follow_up_right": metrics["follow_up_right"], "gen_ai.request.model": chat[:64],
                "obslab.prompt.version": PROMPT_VERSION})
        else:
            report = evaluation.evaluate_retrieval(store, questions, k, settings.min_score)
            for r in report.rows:
                if r.kind == "answerable":      # the score and the label both describe the ranking
                    result("retrieval", r.id, r.reciprocal_rank, "hit" if r.rank and r.rank <= k else "miss")
            metrics = report.metrics()
            emit("eval.finished", identity | {
                "obslab.eval.mode": "retrieval", "obslab.eval.questions": metrics["questions"],
                "obslab.eval.recall_at_k": metrics["recall_at_k"], "obslab.eval.mrr": metrics["mrr"],
                "obslab.eval.in_context": metrics["in_context"],
                "obslab.eval.facts_in_context": metrics["facts_in_context"]})
    finally:
        tel.shutdown()
    def line(label: str, value: float, note: str = "") -> None:
        print(f"  {label:<17} {value:.4f}   {note}".rstrip())
    print()
    if a.answers:
        line("right", metrics["right"], f"{metrics['answerable_questions']} questions with an answer: answered, "
                                        "with the expected terms")
        line("answered", metrics["answered"], "of them, not a fallback")
        line("declined", metrics["declined"], f"{metrics['no_answer_questions']} questions without answer: the "
                                              "reply is the fallback")
        line("follow-up right", metrics["follow_up_right"], f"{metrics['follow_up_questions']} follow-ups, after "
                                                            "the turn before")
        errors = sorted({r.error for r in report.rows if r.error})
        print(f"  time              p50 {metrics['p50_s']:.1f} s, p95 {metrics['p95_s']:.1f} s, "
              f"{metrics['errors']} errors{' (' + ', '.join(errors) + ')' if errors else ''}")
        if wrong := [r.id for r in report.rows if not r.right]:
            print(f"\nwrong ({len(wrong)}): " + ", ".join(wrong))
        rows = [vars(r) for r in report.rows]
    else:
        line("recall@1", metrics["recall_at_1"])
        line(f"recall@{k}", metrics["recall_at_k"])
        line("MRR", metrics["mrr"])
        line("in context", metrics["in_context"], f"expected note in the first {k} and above the threshold")
        line("facts in context", metrics["facts_in_context"], "the expected terms are in the text of those chunks")
        line("refused", metrics["no_answer_refused"], "questions without answer: nothing above the threshold")
        print()
        for style, m in report.by_style().items():
            print(f"  {style:<11} {m['questions']:>3} questions   recall@{k} {m['recall_at_k']:.4f}   "
                  f"MRR {m['mrr']:.4f}")
        if misses := report.misses():
            print(f"\nnot in context ({len(misses)}): " + ", ".join(
                f"{r.id} (rank {r.rank or '-'}, top score {r.top_score:.2f})" for r in misses))
        rows = [{"id": r.id, "kind": r.kind, "rank": r.rank, "in_context": r.in_context, "facts": r.facts,
                 "top_score": r.top_score} for r in report.rows]
    if a.json:                  # written first: whatever fails below, the run is kept
        out = {"mode": mode, "run": run, "config": {**config, "embed_model": model, "chunks": len(docs)},
               "metrics": metrics, "scores": report.scores(), "rows": rows}
        Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if previous is not None:
        try:
            c = evaluation.compare(previous["scores"], report.scores())
        except ValueError as e:         # another golden set: no pair of questions to compare
            print(f"\n--against {a.against}: {e}", file=sys.stderr)
            return 1
        print(f"\nagainst {a.against}: {c['better']} questions better, {c['worse']} worse, {c['same']} the same "
              f"(sign test p = {c['p_value']})")
    if not (a.check or a.update_baseline):
        return 0
    if a.update_baseline:
        evaluation.write_baseline(baseline_file, model, {**config, **metrics})
        print(f"\nbaseline of {model} written to {baseline_file.as_posix()}")
        return 0
    baseline = evaluation.read_baseline(baseline_file, model)
    if baseline is None:
        print(f"\nno baseline for {model} in {baseline_file.as_posix()}: run again with --update-baseline",
              file=sys.stderr)
        return 1
    if stale := [f"{name}={value} (baseline: {baseline.get(name)})" for name, value in config.items()
                 if baseline.get(name) != value]:
        # Another k, threshold, chunking or question set: the numbers are not comparable any more.
        print("\nthe baseline was recorded for another setup: " + ", ".join(stale)
              + "; run again with --update-baseline and commit the file", file=sys.stderr)
        return 1
    if problems := evaluation.check(metrics, baseline):
        print("\nFAILED against the baseline:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"\nbaseline of {model}: pass")
    return 0


def cmd_host_metrics(a) -> int:
    """GPU and Ollama readings from the host, sent to the collector until Ctrl+C (telemetry/host.py)."""
    import dataclasses

    from .config import Settings
    from .telemetry import init_telemetry
    from .telemetry.genai import HostMetrics
    from .telemetry.host import HostSampler, read_gpus
    # A fixed identity: these values become labels, and they describe the host, not a replica.
    os.environ.setdefault("OTEL_RESOURCE_ATTRIBUTES", "host.name=lab-host")
    settings = dataclasses.replace(Settings(), service_name="obslab-host")
    tel = init_telemetry(settings, set_global=True)
    HostMetrics(tel.meter, HostSampler(settings.ollama_url))
    gpus = read_gpus()
    print(f"host metrics -> {settings.otlp_endpoint} every 10 s: "
          f"{len(gpus)} GPU(s){'' if gpus else ' (nvidia-smi not found or no GPU: GPU values skipped)'}, "
          f"Ollama at {settings.ollama_url}. Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
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

    s = sub.add_parser("eval", help="score the search, or the answers, on the golden set")
    s.add_argument("--golden", default="evals/golden.jsonl", help="questions and the note that answers each")
    s.add_argument("--source", default="datalake", help="the notes the golden set was written for")
    s.add_argument("--baseline", default="evals/baseline.json")
    s.add_argument("--check", action="store_true", help="fail when a metric is below the baseline")
    s.add_argument("--update-baseline", action="store_true", help="record this run as the baseline")
    s.add_argument("--answers", action="store_true",
                   help="ask every question through the graph and count the right answers (calls the models)")
    s.add_argument("--json", default="", help="also write the result, question by question, to this file")
    s.add_argument("--against", default="", help="a --json file of an earlier run: compare question by question")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("host-metrics", help="send the host's GPU and Ollama readings to the collector")
    s.set_defaults(fn=cmd_host_metrics)

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
