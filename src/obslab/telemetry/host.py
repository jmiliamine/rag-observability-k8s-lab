"""What the host knows and the cluster cannot see: the GPU, and the memory Ollama gives each model.

Ollama and the GPU run on the host, outside the cluster, and Ollama has no metrics endpoint. A
small poller on the host (`obslab host-metrics`, `task host:metrics`) reads two sources and sends
the values through the same collector as the app, over the OTLP port the cluster publishes on
127.0.0.1:

  nvidia-smi        utilisation, memory used and total, temperature of each GPU
  Ollama /api/ps    the models loaded, their size, and how much of each sits in GPU memory

Two derived readings carry most of the diagnosis on a laptop: a model whose size is larger than
its GPU part runs partly on the CPU, and GPU memory used minus Ollama's share is what other
programs hold (on Windows, nvidia-smi cannot attribute memory per process).

Everything is optional: no GPU, no nvidia-smi or no Ollama gives no values, never an error. The
metric names are in genai.py, like every other name of the lab.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

GPU_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
MIB = 1024 * 1024


@dataclass(frozen=True)
class GpuSample:
    index: int
    name: str
    utilization: float | None       # 0..1
    memory_used: int | None         # bytes
    memory_total: int | None        # bytes
    temperature: float | None       # °C


@dataclass(frozen=True)
class ModelSample:
    model: str
    size: int                       # bytes, the whole model with its context
    size_vram: int                  # bytes of it in GPU memory


def _number(field: str) -> float | None:
    """nvidia-smi writes "[N/A]" or "[Not Supported]" for what a GPU does not report."""
    field = field.strip()
    try:
        return float(field)
    except ValueError:
        return None


def parse_nvidia_smi(text: str) -> list[GpuSample]:
    """Lines of `nvidia-smi --query-gpu=GPU_QUERY --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6 or _number(parts[0]) is None:
            continue
        util, used, total, temp = (_number(p) for p in parts[2:])
        gpus.append(GpuSample(
            index=int(_number(parts[0])), name=parts[1][:64],
            utilization=None if util is None else util / 100,
            memory_used=None if used is None else int(used * MIB),
            memory_total=None if total is None else int(total * MIB),
            temperature=temp))
    return gpus


def read_gpus(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[GpuSample]:
    if shutil.which("nvidia-smi") is None and run is subprocess.run:
        return []
    try:
        done = run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_nvidia_smi(done.stdout) if done.returncode == 0 else []


def parse_ollama_ps(payload: dict) -> list[ModelSample]:
    models = []
    for m in payload.get("models") or []:
        name = str(m.get("name") or m.get("model") or "")[:64]
        if name:
            models.append(ModelSample(model=name, size=int(m.get("size") or 0), size_vram=int(m.get("size_vram") or 0)))
    return models


def read_ollama(base_url: str, timeout: float = 3.0) -> list[ModelSample]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/ps", timeout=timeout) as r:
            return parse_ollama_ps(json.loads(r.read()))
    except (OSError, urllib.error.URLError, ValueError):
        return []


class HostSampler:
    """One reading of both sources per collection: the metric callbacks of one export all ask
    for it, so it is kept for `ttl` seconds instead of running nvidia-smi once per instrument."""

    def __init__(self, ollama_url: str, ttl: float = 2.0,
                 gpus: Callable[[], list[GpuSample]] = read_gpus,
                 models: Callable[[str], list[ModelSample]] = read_ollama):
        self.ollama_url, self.ttl = ollama_url, ttl
        self._gpus, self._models = gpus, models
        self._at = float("-inf")
        self._last: tuple[list[GpuSample], list[ModelSample]] = ([], [])

    def sample(self) -> tuple[list[GpuSample], list[ModelSample]]:
        now = time.monotonic()
        if now - self._at >= self.ttl:
            self._last = (self._gpus(), self._models(self.ollama_url))
            self._at = now
        return self._last
