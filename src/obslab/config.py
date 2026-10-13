"""All settings come from environment variables (12-factor), with local defaults.

OBSLAB_PROVIDER          ollama | fake            (fake = deterministic, for tests/CI)
OBSLAB_CHAT_MODEL        default llama3.2:3b
OBSLAB_EMBED_MODEL       default nomic-embed-text
OLLAMA_BASE_URL          default http://localhost:11434
PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD    PostgreSQL connection (standard libpq variables)
OBSLAB_DB_TIMEOUT_MS     statement timeout for vector queries, default 5000
OBSLAB_TELEMETRY         otlp | console | none    (tests inject in-memory exporters directly)
OTEL_EXPORTER_OTLP_ENDPOINT  default http://localhost:4318 (the collector)
OBSLAB_CAPTURE_CONTENT   true to record prompts/answers on spans (opt-in, off by default)
OBSLAB_HISTORY_TURNS     past turns used to understand a follow-up question, default 3 (0..10)
OBSLAB_MAX_TURNS         questions allowed in one conversation, default 20
OBSLAB_CONVERSATION_TTL_DAYS  idle conversations older than this are purged, default 7
OBSLAB_MODEL_CONCURRENCY model calls running at once in this process, default 1
OBSLAB_MODEL_QUEUE       model calls allowed to wait behind them, default 8 (more are refused at once)
OBSLAB_REQUEST_DEADLINE_S  seconds a question may take before it is dropped, default 120
OBSLAB_TOP_K / OBSLAB_MIN_SCORE   retrieval tuning (validated: 1..20 and 0..1)
OBSLAB_PRICE_IN / OBSLAB_PRICE_OUT  USD per 1M tokens, to chart an "API-equivalent" cost
OBSLAB_ENVIRONMENT       deployment.environment.name resource attribute (local, k3d, ...)
OBSLAB_LOG_STDOUT        true to also write each log event as a JSON line on the process output
                         (containers: `kubectl logs`); events are listed in telemetry/events.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PROVIDERS = ("ollama", "fake")
MAX_TOP_K = 20


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: os.environ.get("OBSLAB_PROVIDER", "ollama"))
    chat_model: str = field(default_factory=lambda: os.environ.get("OBSLAB_CHAT_MODEL", "llama3.2:3b"))
    embed_model: str = field(default_factory=lambda: os.environ.get("OBSLAB_EMBED_MODEL", "nomic-embed-text"))
    ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    db_timeout_ms: int = field(default_factory=lambda: int(os.environ.get("OBSLAB_DB_TIMEOUT_MS", "5000")))
    telemetry: str = field(default_factory=lambda: os.environ.get("OBSLAB_TELEMETRY", "otlp"))
    otlp_endpoint: str = field(default_factory=lambda: os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"))
    service_name: str = field(default_factory=lambda: os.environ.get("OTEL_SERVICE_NAME", "obslab-rag"))
    capture_content: bool = field(default_factory=lambda: _bool("OBSLAB_CAPTURE_CONTENT"))
    top_k: int = field(default_factory=lambda: int(os.environ.get("OBSLAB_TOP_K", "4")))
    min_score: float = field(default_factory=lambda: float(os.environ.get("OBSLAB_MIN_SCORE", "0.6")))
    history_turns: int = field(default_factory=lambda: int(os.environ.get("OBSLAB_HISTORY_TURNS", "3")))
    max_turns: int = field(default_factory=lambda: int(os.environ.get("OBSLAB_MAX_TURNS", "20")))
    conversation_ttl_days: float = field(
        default_factory=lambda: float(os.environ.get("OBSLAB_CONVERSATION_TTL_DAYS", "7")))
    model_concurrency: int = field(default_factory=lambda: int(os.environ.get("OBSLAB_MODEL_CONCURRENCY", "1")))
    model_queue: int = field(default_factory=lambda: int(os.environ.get("OBSLAB_MODEL_QUEUE", "8")))
    request_deadline_s: float = field(
        default_factory=lambda: float(os.environ.get("OBSLAB_REQUEST_DEADLINE_S", "120")))
    price_in: float = field(default_factory=lambda: float(os.environ.get("OBSLAB_PRICE_IN", "0")))
    price_out: float = field(default_factory=lambda: float(os.environ.get("OBSLAB_PRICE_OUT", "0")))
    environment: str = field(default_factory=lambda: os.environ.get("OBSLAB_ENVIRONMENT", "local"))
    log_stdout: bool = field(default_factory=lambda: _bool("OBSLAB_LOG_STDOUT"))

    def __post_init__(self) -> None:
        # Fail at startup with a clear message rather than serve odd answers later.
        problems = []
        if self.provider not in PROVIDERS:
            problems.append(f"OBSLAB_PROVIDER={self.provider!r} (expected one of {PROVIDERS})")
        if not 1 <= self.top_k <= MAX_TOP_K:
            problems.append(f"OBSLAB_TOP_K={self.top_k} (expected 1..{MAX_TOP_K})")
        if not 0.0 <= self.min_score <= 1.0:
            problems.append(f"OBSLAB_MIN_SCORE={self.min_score} (expected 0..1)")
        if not 0 <= self.history_turns <= 10:
            problems.append(f"OBSLAB_HISTORY_TURNS={self.history_turns} (expected 0..10)")
        if not 1 <= self.max_turns <= 200:
            problems.append(f"OBSLAB_MAX_TURNS={self.max_turns} (expected 1..200)")
        if self.conversation_ttl_days <= 0:
            problems.append(f"OBSLAB_CONVERSATION_TTL_DAYS={self.conversation_ttl_days} (expected > 0)")
        if not 0 <= self.history_turns <= 10:
            problems.append(f"OBSLAB_HISTORY_TURNS={self.history_turns} (expected 0..10)")
        if not 1 <= self.max_turns <= 200:
            problems.append(f"OBSLAB_MAX_TURNS={self.max_turns} (expected 1..200)")
        if self.conversation_ttl_days <= 0:
            problems.append(f"OBSLAB_CONVERSATION_TTL_DAYS={self.conversation_ttl_days} (expected > 0)")
        if not 1 <= self.model_concurrency <= 64:
            problems.append(f"OBSLAB_MODEL_CONCURRENCY={self.model_concurrency} (expected 1..64)")
        if not 0 <= self.model_queue <= 1000:
            problems.append(f"OBSLAB_MODEL_QUEUE={self.model_queue} (expected 0..1000)")
        if not 1 <= self.request_deadline_s <= 3600:
            problems.append(f"OBSLAB_REQUEST_DEADLINE_S={self.request_deadline_s} (expected 1..3600)")
        if not 100 <= self.db_timeout_ms <= 60_000:
            problems.append(f"OBSLAB_DB_TIMEOUT_MS={self.db_timeout_ms} (expected 100..60000)")
        if problems:
            raise ValueError("invalid settings: " + "; ".join(problems))

    @property
    def embed_model_id(self) -> str:
        """Identity of the embedding model, as written in the index metadata."""
        return "fake-embed" if self.provider == "fake" else self.embed_model
