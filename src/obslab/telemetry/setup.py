"""OpenTelemetry bootstrap: traces, metrics and logs share one Resource.

Nothing in the app reaches for global providers: everything receives a
`Telemetry` object. That keeps tests isolated (in-memory exporters) and makes
the export target a pure configuration choice:

  otlp     -> OTLP/HTTP to the collector (docker compose or the k3d cluster)
  console  -> stdout, handy to learn what a span/metric looks like
  none     -> SDK wired but nothing exported
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.instrumentation.logging.handler import LoggingHandler  # replaces the deprecated SDK handler
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogRecordExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter

from .. import __version__
from ..config import Settings
from .events import JsonFormatter

SCOPE = "obslab"


@dataclass
class Telemetry:
    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    logger_provider: LoggerProvider | None
    tracer: trace.Tracer
    meter: metrics.Meter
    capture_content: bool = False

    def shutdown(self) -> None:
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()
        if self.logger_provider:
            self.logger_provider.shutdown()


def build_resource(settings: Settings) -> Resource:
    # Resource.create() also merges OTEL_RESOURCE_ATTRIBUTES from the environment: in Kubernetes the
    # Deployment injects k8s.pod.name / k8s.namespace.name / k8s.node.name through the downward API,
    # so every span, metric and log says which pod produced it (the collector adds more).
    return Resource.create({
        "service.name": settings.service_name,
        "service.version": __version__,
        "deployment.environment.name": settings.environment,
    })


def init_telemetry(settings: Settings, span_exporter: SpanExporter | None = None,
                   metric_reader: MetricReader | None = None, set_global: bool = False) -> Telemetry:
    """Build providers. Passing an exporter/reader (tests) overrides settings.telemetry."""
    resource = build_resource(settings)
    tp = TracerProvider(resource=resource)
    readers = []
    logger_provider = None

    if span_exporter is not None or metric_reader is not None:
        if span_exporter is not None:
            tp.add_span_processor(SimpleSpanProcessor(span_exporter))
        if metric_reader is not None:
            readers.append(metric_reader)
    elif settings.telemetry == "otlp":
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        base = settings.otlp_endpoint.rstrip("/")
        tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces")))
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{base}/v1/metrics"),
                                                     export_interval_millis=10_000))
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{base}/v1/logs")))
    elif settings.telemetry == "console":
        tp.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=30_000))
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogRecordExporter()))

    mp = MeterProvider(resource=resource, metric_readers=readers)
    app_logger = logging.getLogger("obslab")
    app_logger.setLevel(logging.INFO)
    if logger_provider is not None:
        # OTLP copy of every event (telemetry/events.py) -> collector -> Loki, with trace/span ids.
        app_logger.addHandler(LoggingHandler(level=logging.INFO, logger_provider=logger_provider))
    if settings.log_stdout and not any(getattr(h, "_obslab_stdout", False) for h in app_logger.handlers):
        # stdout copy for `kubectl logs` / `docker logs` / a CI log: one JSON object per event, with
        # the trace id so a line can be pasted into Tempo. Still there when the collector is down.
        stream = logging.StreamHandler()
        stream.setFormatter(JsonFormatter(settings.service_name, __version__))
        stream._obslab_stdout = True  # type: ignore[attr-defined]
        app_logger.addHandler(stream)
    if set_global:
        trace.set_tracer_provider(tp)
        metrics.set_meter_provider(mp)
        if logger_provider is not None:
            set_logger_provider(logger_provider)
    return Telemetry(tp, mp, logger_provider, tp.get_tracer(SCOPE, __version__),
                     mp.get_meter(SCOPE, __version__), settings.capture_content)
