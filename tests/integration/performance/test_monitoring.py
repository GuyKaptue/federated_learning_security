import time

import numpy as np
import pytest

from federated_learning.monitoring.resources import CommCounter, ResourceMonitor, Timer, accuracy_loss_pct, overhead_pct


def test_cpu_accounting_tracks_busy_loop():
    with ResourceMonitor(0.02) as m:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.5:
            sum(i * i for i in range(1000))
    s = m.summary()
    assert 0.35 < s["cpu_time_s"] < 0.9 and s["wall_s"] >= 0.5 and s["cpu_util_mean_pct"] > 50


def test_peak_memory_reflects_allocation():
    with ResourceMonitor(0.01) as m:
        base = m.rss[0]
        a = np.ones(60 * 2 ** 20 // 8); time.sleep(0.1); del a
    assert m.summary()["rss_peak_mb"] - base > 40 and m.summary()["rss_peak_mb"] >= m.summary()["rss_mean_mb"]


def test_gpu_absent_reported_not_estimated():
    s = ResourceMonitor().summary()
    assert s["gpu_available"] in (True, False) and (s["gpu_available"] or s["gpu_util_pct"] is None)


def test_overhead_formulas():
    assert overhead_pct(130, 100) == pytest.approx(30.0) and accuracy_loss_pct(0.985, 0.99) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        overhead_pct(1, 0)


def test_comm_counter_and_timer():
    c = CommCounter(); c.add_up(1024 ** 2); c.add_down(1024 ** 2); assert c.total_mb == 2.0
    t = Timer()
    with t.section("a"):
        time.sleep(0.01)
    assert t.totals["a"] > 0.005


def test_invalid_interval():
    with pytest.raises(ValueError):
        ResourceMonitor(0)
