"""Measure the machine, then choose the worker count. Instruction 86.

    "Add a small benchmark command that records throughput and peak RAM/VRAM
     for representative fields, making worker-count defaults specific to the
     current machine."

The defaults it replaces are arithmetic on the core count. Cores are the one
thing that is never the binding constraint on a measurement run -- a field is
hundreds of megabytes decompressed, so eight workers on a 16 GB laptop swap
long before they saturate the CPU.
"""
from __future__ import annotations

import time

import pytest

from spacr.benchmark import (
    Measurement,
    Recommendation,
    available_memory_bytes,
    benchmark,
    format_report,
    recommend_workers,
)


GIB = 1024 ** 3


def _measurement(**over):
    base = dict(items=4, seconds=2.0, peak_rss_bytes=5 * GIB,
                baseline_rss_bytes=1 * GIB)
    base.update(over)
    return Measurement(**base)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def test_it_times_the_work_and_reports_throughput():
    seen = []
    out = benchmark(lambda i: seen.append(i), [0, 1, 2, 3], warmup=0)
    assert out.items == 4
    assert seen == [0, 1, 2, 3]
    assert out.items_per_second > 0


def test_the_warm_up_item_is_run_but_not_counted():
    """The first field pays for imports, CUDA context and page faults that no
    later field pays again; counting it makes a short run look far slower
    than the plate it is predicting."""
    seen = []
    out = benchmark(lambda i: seen.append(i), [0, 1, 2], warmup=1)
    assert seen == [0, 1, 2], "the warm-up item still runs"
    assert out.items == 2, "but it is not in the timing"
    assert any("warm-up" in note for note in out.notes)


def test_a_benchmark_over_nothing_is_refused():
    """Otherwise the per-item cost is NaN and a recommendation is built on it."""
    with pytest.raises(ValueError):
        benchmark(lambda i: i, [])


def test_the_warm_up_never_eats_every_item():
    out = benchmark(lambda i: i, [0], warmup=5)
    assert out.items == 1


def test_the_work_footprint_excludes_the_interpreter():
    """What the WORK needed, not what the process happened to be sitting at."""
    out = _measurement(peak_rss_bytes=5 * GIB, baseline_rss_bytes=1 * GIB)
    assert out.work_rss_bytes == 4 * GIB


def test_a_shrinking_footprint_does_not_go_negative():
    out = _measurement(peak_rss_bytes=1 * GIB, baseline_rss_bytes=2 * GIB)
    assert out.work_rss_bytes == 0


def test_no_cuda_reports_none_rather_than_zero():
    """"not measured" and "measured, and it used none" are different answers."""
    out = benchmark(lambda i: i, [0, 1], warmup=0)
    assert out.peak_vram_bytes is None or out.peak_vram_bytes >= 0
    if out.peak_vram_bytes is None:
        assert any("VRAM" in note for note in out.notes)


# ---------------------------------------------------------------------------
# The recommendation
# ---------------------------------------------------------------------------

def test_memory_is_the_bound_when_a_worker_is_large():
    """The term the core-count defaults omit, and usually the binding one."""
    out = recommend_workers(
        _measurement(peak_rss_bytes=9 * GIB, baseline_rss_bytes=1 * GIB),
        cores=32, available_bytes=34 * GIB, reserve_bytes=2 * GIB)
    # 32 GiB budget / 8 GiB per worker = 4
    assert out.workers == 4
    assert "memory-bound" in out.reason


def test_cores_are_the_bound_when_a_worker_is_small():
    out = recommend_workers(
        _measurement(peak_rss_bytes=1200 * 1024 ** 2,
                     baseline_rss_bytes=1024 ** 3),
        cores=4, available_bytes=64 * GIB, reserve_bytes=2 * GIB)
    assert out.workers == 4
    assert "core-bound" in out.reason


def test_a_worker_that_does_not_fit_still_gets_one():
    """One at a time is what fits; zero workers would run nothing."""
    out = recommend_workers(
        _measurement(peak_rss_bytes=41 * GIB, baseline_rss_bytes=1 * GIB),
        cores=16, available_bytes=8 * GIB, reserve_bytes=2 * GIB)
    assert out.workers == 1
    assert "one worker at a time" in out.reason


def test_the_reserve_is_actually_subtracted():
    """Sizing against total memory is how a run is OOM-killed at field 900."""
    big = recommend_workers(
        _measurement(peak_rss_bytes=2 * GIB, baseline_rss_bytes=1 * GIB),
        cores=64, available_bytes=10 * GIB, reserve_bytes=0)
    small = recommend_workers(
        _measurement(peak_rss_bytes=2 * GIB, baseline_rss_bytes=1 * GIB),
        cores=64, available_bytes=10 * GIB, reserve_bytes=8 * GIB)
    assert big.workers > small.workers


def test_without_a_measurement_it_says_so_rather_than_inventing_one():
    out = recommend_workers(None, cores=8, available_bytes=32 * GIB)
    assert out.workers == 8
    assert "no usable memory measurement" in out.reason


def test_work_too_small_to_register_falls_back_to_cores():
    out = recommend_workers(
        _measurement(peak_rss_bytes=1 * GIB, baseline_rss_bytes=1 * GIB),
        cores=6, available_bytes=32 * GIB)
    assert out.workers == 6


def test_it_never_recommends_zero_or_a_negative_number():
    for cores in (0, -4):
        out = recommend_workers(None, cores=cores, available_bytes=GIB)
        assert out.workers >= 1


def test_the_maximum_is_honoured():
    out = recommend_workers(None, cores=256, available_bytes=999 * GIB,
                            maximum=8)
    assert out.workers == 8


def test_the_recommendation_carries_its_evidence():
    """A number a user disagrees with should be arguable, not just overridable."""
    measurement = _measurement()
    out = recommend_workers(measurement, cores=8, available_bytes=32 * GIB)
    assert out.measurement is measurement
    assert out.cores == 8 and out.available_bytes == 32 * GIB
    assert str(out).startswith(f"{out.workers} worker(s):")


# ---------------------------------------------------------------------------
# It measures; it does not tune
# ---------------------------------------------------------------------------

def test_nothing_here_writes_a_setting():
    """A benchmark that changed n_jobs would make a run's speed depend on
    when the benchmark last ran, which is the opposite of reproducible."""
    import inspect

    from spacr import benchmark as module

    source = inspect.getsource(module)
    for forbidden in ("set_default", "settings[", "save_settings",
                      "os.environ["):
        assert forbidden not in source, (
            f"the benchmark writes state via {forbidden!r}")


def test_available_memory_is_a_number():
    assert available_memory_bytes() >= 0


def test_the_report_names_every_figure_it_used():
    measurement = _measurement()
    out = recommend_workers(measurement, cores=8, available_bytes=32 * GIB)
    text = format_report(measurement, out)
    for expected in ("throughput", "peak RSS", "VRAM", "recommended workers",
                     "because"):
        assert expected in text


def test_the_module_needs_no_display():
    import subprocess
    import sys

    code = ("import sys, spacr.benchmark; "
            "assert not [m for m in sys.modules if m.startswith('PySide6')]")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
