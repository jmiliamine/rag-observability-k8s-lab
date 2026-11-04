"""The host poller: what it reads from nvidia-smi and Ollama, and what it does when they are absent."""

import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from test_repo_hygiene import ROOT

from obslab.telemetry.genai import HostMetrics
from obslab.telemetry.host import (
    GpuSample,
    HostSampler,
    ModelSample,
    parse_nvidia_smi,
    parse_ollama_ps,
    read_gpus,
    read_ollama,
)

# A laptop GPU, then a second GPU that reports nothing but its memory.
NVIDIA_SMI = ("0, Example Laptop GPU, 17, 3007, 6144, 54\n"
              "1, Some Datacenter GPU, [N/A], 512, 81920, [Not Supported]\n")
OLLAMA_PS = {"models": [{"name": "llama3.2:3b", "model": "llama3.2:3b", "size": 2340883004,
                         "size_vram": 2340883004, "context_length": 4096}]}


def test_nvidia_smi_lines_are_read_and_missing_fields_stay_missing():
    first, second = parse_nvidia_smi(NVIDIA_SMI)
    assert first == GpuSample(0, "Example Laptop GPU", 0.17, 3007 * 2**20, 6144 * 2**20, 54.0)
    assert (second.utilization, second.temperature) == (None, None)
    assert second.memory_total == 81920 * 2**20
    assert parse_nvidia_smi("garbage\n\n") == []


def test_no_nvidia_smi_or_a_failing_one_gives_no_gpu():
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    def failing(*args, **kwargs):
        return subprocess.CompletedProcess(args, 9, stdout="", stderr="NVIDIA-SMI has failed")
    assert read_gpus(run=missing) == []
    assert read_gpus(run=failing) == []


def test_ollama_models_are_read():
    assert parse_ollama_ps(OLLAMA_PS) == [ModelSample("llama3.2:3b", 2340883004, 2340883004)]
    assert parse_ollama_ps({"models": None}) == []


def test_an_unreachable_ollama_gives_no_model():
    assert read_ollama("http://127.0.0.1:9", timeout=0.5) == []


class _Ollama(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"models": [{"name": "llama3.2:3b", "size": 2340883004, "size_vram": 1500000000}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_a_model_partly_on_the_cpu_is_read_from_the_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Ollama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (model,) = read_ollama(f"http://127.0.0.1:{server.server_address[1]}")
    finally:
        server.shutdown()
    assert model.size - model.size_vram == 840883004            # the part that runs on the CPU


def _points(sampler) -> dict[str, list]:
    reader = InMemoryMetricReader()
    HostMetrics(MeterProvider(metric_readers=[reader]).get_meter("test"), sampler)
    data = reader.get_metrics_data()
    return {m.name: (m.unit, list(m.data.data_points)) for rm in data.resource_metrics
            for sm in rm.scope_metrics for m in sm.metrics}


def test_names_units_and_attributes_of_the_host_signals():
    sampler = HostSampler("unused", gpus=lambda: parse_nvidia_smi(NVIDIA_SMI),
                          models=lambda url: parse_ollama_ps(OLLAMA_PS))
    got = _points(sampler)
    units = {name: unit for name, (unit, _) in got.items()}
    assert units == {"hw.gpu.utilization": "1", "hw.gpu.memory.usage": "By", "hw.gpu.memory.limit": "By",
                     "obslab.host.gpu.temperature": "Cel", "obslab.ollama.model.size": "By",
                     "obslab.ollama.model.vram": "By", "obslab.ollama.models.loaded": "{model}"}
    util = got["hw.gpu.utilization"][1]
    assert [(dict(p.attributes), p.value) for p in util] == [
        ({"hw.id": "gpu0", "hw.type": "gpu", "hw.name": "Example Laptop GPU",
          "hw.gpu.task": "general"}, 0.17)]                    # the second GPU reports none: no point
    assert len(got["hw.gpu.memory.limit"][1]) == 2
    assert [dict(p.attributes) for p in got["obslab.ollama.model.vram"][1]] == [{"gen_ai.request.model": "llama3.2:3b"}]
    assert got["obslab.ollama.models.loaded"][1][0].value == 1


def test_nothing_to_read_gives_no_gpu_values_and_zero_models():
    got = _points(HostSampler("unused", gpus=lambda: [], models=lambda url: []))
    assert all(not points for name, (_, points) in got.items() if name != "obslab.ollama.models.loaded")
    assert got["obslab.ollama.models.loaded"][1][0].value == 0


def test_one_reading_serves_every_instrument_of_an_export():
    calls = []
    sampler = HostSampler("unused", gpus=lambda: calls.append("gpu") or [], models=lambda url: [])
    _points(sampler)
    assert calls == ["gpu"]


@pytest.mark.parametrize("path", ["deploy/k8s/platform/values/otel-collector.yaml",
                                  "deploy/compose/otel-collector.yaml"])
def test_the_collector_forgets_a_series_soon_after_its_last_value(path):
    # Otherwise a stopped host poller would show its last reading as if it were live, for 5 minutes.
    text = (ROOT / path).read_text(encoding="utf-8")
    assert "metric_expiration: 1m" in text
    yaml.safe_load(text)
