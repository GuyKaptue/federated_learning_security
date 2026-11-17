"""Resource and communication monitoring.

Metric definitions (all MEASURED on the executing machine; see docs/fl_perf_002/metrics_definitions.md):

* wall_s            = t_end - t_start (perf_counter)
* cpu_time_s        = (user+system CPU seconds of this process, children included) at end minus start
                      -> cumulative resource consumption
* cpu_util_mean_pct = mean over samples of process CPU% (100% = one fully used core; may exceed 100)
* cpu_util_peak_pct = max over samples
* rss_mean_mb / rss_peak_mb = mean / max resident set size of the process over samples (MiB)
* comm bytes        = logical serialized payload bytes counted by the protocol code (not NIC traffic,
                      since all virtual clients share one machine)
* throughput_samples_per_s = training samples processed by clients / client compute seconds
* GPU               = recorded as 'unavailable' when no NVIDIA tooling is present (never estimated)
"""
from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import psutil


def gpu_snapshot() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"gpu_available": False, "gpu_util_pct": None, "gpu_mem_mb": None}
    try:
        out = subprocess.check_output(  # nosec B603 B607
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            timeout=5).decode().strip().splitlines()[0]
        u, m = [float(x) for x in out.split(",")]
        return {"gpu_available": True, "gpu_util_pct": u, "gpu_mem_mb": m}
    except Exception:  # pragma: no cover - depends on host
        return {"gpu_available": False, "gpu_util_pct": None, "gpu_mem_mb": None}

class ResourceMonitor:
    """Background sampler. Use as a context manager; call .summary() afterwards."""

    def __init__(self, interval: float = 0.05):
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu: list[float] = []
        self.rss: list[float] = []
        self.t0 = self.t1 = 0.0
        self._cpu0 = self._cpu1 = 0.0

    def _cpu_time(self) -> float:
        t = self.proc.cpu_times()
        return t.user + t.system + getattr(t, "children_user", 0.0) + getattr(t, "children_system", 0.0)

    def _run(self):
        self.proc.cpu_percent(None)
        while not self._stop.wait(self.interval):
            self.cpu.append(self.proc.cpu_percent(None))
            self.rss.append(self.proc.memory_info().rss / 2 ** 20)

    def __enter__(self):
        self.t0, self._cpu0 = time.perf_counter(), self._cpu_time()
        self.rss.append(self.proc.memory_info().rss / 2 ** 20)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.rss.append(self.proc.memory_info().rss / 2 ** 20)
        self.t1, self._cpu1 = time.perf_counter(), self._cpu_time()
        return False

    def summary(self) -> dict:
        cpu = self.cpu or [0.0]
        rss = self.rss or [self.proc.memory_info().rss / 2 ** 20]
        s = dict(wall_s=self.t1 - self.t0, cpu_time_s=self._cpu1 - self._cpu0,
                 cpu_util_mean_pct=float(np.mean(cpu)), cpu_util_peak_pct=float(np.max(cpu)),
                 rss_mean_mb=float(np.mean(rss)), rss_peak_mb=float(np.max(rss)),
                 n_samples=len(self.cpu), sample_interval_s=self.interval)
        s.update(gpu_snapshot())
        return s

@dataclass
class CommCounter:
    up_bytes: int = 0
    down_bytes: int = 0

    def add_up(self, n: int):
        self.up_bytes += int(n)

    def add_down(self, n: int):
        self.down_bytes += int(n)

    @property
    def total_mb(self) -> float:
        return (self.up_bytes + self.down_bytes) / 2 ** 20

@dataclass
class Timer:
    totals: dict = field(default_factory=dict)

    @contextmanager
    def section(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t

def overhead_pct(secure: float, baseline: float) -> float:
    """Relative overhead (secure - baseline) / baseline * 100."""
    if baseline <= 0:
        raise ValueError("baseline must be > 0")
    return 100.0 * (secure - baseline) / baseline

def accuracy_loss_pct(acc_secure: float, acc_baseline: float) -> float:
    """Absolute accuracy loss in percentage points: 100*(acc_baseline - acc_secure)."""
    return 100.0 * (acc_baseline - acc_secure)
