"""Measuring one worker, and turning that into a worker count with a reason.

:mod:`spacr.benchmark` runs the work serially so the figure it reports is
what ONE worker needs, then divides free memory by it. The paths driven here
are the ones a healthy laptop never takes: no CUDA at all, an unreadable
``/proc/meminfo``, a per-worker footprint bigger than the free memory, and a
core count so high the fixed maximum becomes the binding term.
"""
from __future__ import annotations

import builtins

import pytest

from spacr import benchmark as B


def test_vram_is_not_measured_when_torch_is_not_installed(monkeypatch):
    """``None`` means "not measured"; ``0`` would mean "measured, used none"."""
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert B._vram_bytes() is None
    B._reset_vram()  # must not raise either


def test_vram_is_not_measured_without_a_cuda_device(monkeypatch):
    """A torch build with no device reports nothing rather than zero."""
    import torch

    class NoDevices:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(torch, "cuda", NoDevices(), raising=False)
    assert B._vram_bytes() is None
    B._reset_vram()


def test_a_cuda_device_reports_its_peak_allocation(monkeypatch):
    """The peak is read back in bytes, and the counter is reset before the run."""
    import torch

    reset = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def max_memory_allocated():
            return 1536

        @staticmethod
        def reset_peak_memory_stats():
            reset.append(True)

    monkeypatch.setattr(torch, "cuda", FakeCuda(), raising=False)
    assert B._vram_bytes() == 1536
    B._reset_vram()
    assert reset == [True]


def test_a_cuda_query_that_raises_is_read_as_not_measured(monkeypatch):
    """A driver that errors mid-question has not measured anything."""
    import torch

    class BrokenCuda:
        @staticmethod
        def is_available():
            raise RuntimeError("the CUDA driver is not loaded")

        @staticmethod
        def reset_peak_memory_stats():
            raise RuntimeError("the CUDA driver is not loaded")

    monkeypatch.setattr(torch, "cuda", BrokenCuda(), raising=False)
    assert B._vram_bytes() is None
    B._reset_vram()


def test_unreadable_meminfo_reports_no_free_memory_rather_than_a_guess(
        monkeypatch):
    """Zero available means "the memory bound cannot be used", not "no memory"."""
    def no_meminfo(*args, **kwargs):
        raise OSError("/proc is not mounted")

    monkeypatch.setattr(builtins, "open", no_meminfo)
    assert B.available_memory_bytes() == 0


def test_a_benchmark_over_nothing_is_refused():
    """A per-item cost of NaN would poison every number built on it."""
    with pytest.raises(ValueError, match="at least one item"):
        B.benchmark(lambda item: item, [])


def test_the_warm_up_item_is_excluded_from_the_timing_and_said_to_be():
    """The first field pays for imports that no later field pays again."""
    seen = []
    measurement = B.benchmark(seen.append, [1, 2, 3], warmup=1)
    assert seen == [1, 2, 3]
    assert measurement.items == 2
    assert any("1 warm-up item(s) excluded" in note
               for note in measurement.notes)


def test_a_warm_up_larger_than_the_run_still_measures_one_item():
    """There is always at least one item left to time."""
    measurement = B.benchmark(lambda item: None, [1, 2], warmup=99)
    assert measurement.items == 1


def test_a_worker_that_does_not_fit_gets_one_worker_and_the_arithmetic():
    """The refusal shows both figures so it can be argued with."""
    measurement = B.Measurement(items=4, seconds=2.0,
                                peak_rss_bytes=40 * 1024 ** 3,
                                baseline_rss_bytes=0)
    recommendation = B.recommend_workers(
        measurement, cores=16, available_bytes=8 * 1024 ** 3)
    assert recommendation.workers == 1
    assert "one worker at a time is what fits" in recommendation.reason
    assert "40.00 GiB" in recommendation.reason


def test_a_machine_with_more_cores_than_the_maximum_is_capped_and_says_so():
    """Above the fixed ceiling, neither cores nor memory is the reason."""
    measurement = B.Measurement(items=4, seconds=2.0,
                                peak_rss_bytes=1024 ** 3,
                                baseline_rss_bytes=0)
    recommendation = B.recommend_workers(
        measurement, cores=256, available_bytes=512 * 1024 ** 3, maximum=8)
    assert recommendation.workers == 8
    assert recommendation.reason == "capped at the 8-worker maximum"


def test_a_memory_bound_machine_names_memory_and_a_core_bound_one_names_cores():
    """The reason is the binding term, not a restatement of the answer."""
    measurement = B.Measurement(items=4, seconds=2.0,
                                peak_rss_bytes=4 * 1024 ** 3,
                                baseline_rss_bytes=0)
    memory_bound = B.recommend_workers(
        measurement, cores=64, available_bytes=18 * 1024 ** 3)
    assert memory_bound.reason.startswith("memory-bound:")

    core_bound = B.recommend_workers(
        measurement, cores=2, available_bytes=512 * 1024 ** 3)
    assert core_bound.workers == 2
    assert core_bound.reason.startswith("core-bound:")


def test_a_run_with_no_measurable_footprint_falls_back_to_the_core_count():
    """A memory bound invented from a missing number would be worse than none."""
    tiny = B.Measurement(items=1, seconds=1.0, peak_rss_bytes=10,
                         baseline_rss_bytes=10)
    recommendation = B.recommend_workers(tiny, cores=4,
                                         available_bytes=64 * 1024 ** 3)
    assert recommendation.workers == 4
    assert "no usable memory measurement" in recommendation.reason
    assert str(recommendation).startswith("4 worker(s): ")


def test_the_report_prints_the_vram_it_measured_and_the_notes_it_collected():
    """Every number the recommendation used appears in the pasteable report."""
    measurement = B.Measurement(items=4, seconds=2.0,
                                peak_rss_bytes=8 * 1024 ** 3,
                                peak_vram_bytes=3 * 1024 ** 3,
                                baseline_rss_bytes=2 * 1024 ** 3,
                                notes=["1 warm-up item(s) excluded"])
    recommendation = B.recommend_workers(
        measurement, cores=4, available_bytes=64 * 1024 ** 3)
    report = B.format_report(measurement, recommendation)
    assert "peak VRAM           3.00 GiB" in report
    assert "attributable to it  6.00 GiB" in report
    assert "note                1 warm-up item(s) excluded" in report
    assert f"recommended workers {recommendation.workers}" in report


def test_the_report_says_when_there_was_no_device_to_measure():
    """"Not measured" is printed as such, never as a zero."""
    measurement = B.Measurement(items=2, seconds=1.0,
                                peak_rss_bytes=1024 ** 3)
    report = B.format_report(measurement, B.recommend_workers(measurement))
    assert "not measured (no CUDA device)" in report
