"""The machine-specific corners of the worker-count benchmark.

Three of them decide a number rather than decorating one: the kilobyte /
byte difference in ``ru_maxrss`` between Linux and macOS, which is a factor
of 1024 in a memory budget; the platform that has no ``resource`` module at
all; and the CUDA peak counter, which reports the previous run's high-water
mark unless it is reset before the clock starts.
"""
from __future__ import annotations

import builtins
import io
import sys
import types

import pytest

from spacr import benchmark as B


GIB = 1024 ** 3


class _FakeRusage:
    def __init__(self, ru_maxrss):
        self.ru_maxrss = ru_maxrss


def _fake_resource(ru_maxrss):
    module = types.ModuleType("resource")
    module.RUSAGE_SELF = 0
    module.getrusage = lambda who: _FakeRusage(ru_maxrss)
    return module


class _BlockedImport:
    """A meta-path finder that makes one module name unimportable."""

    def __init__(self, blocked):
        self.blocked = blocked

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.blocked:
            raise ImportError(f"No module named {self.blocked!r}")
        return None


# ---------------------------------------------------------------------------
# ru_maxrss is kilobytes on Linux and bytes on macOS
# ---------------------------------------------------------------------------

def test_linux_reads_ru_maxrss_as_kilobytes(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", _fake_resource(2048))
    monkeypatch.setattr(sys, "platform", "linux")

    assert B._rss_bytes() == 2048 * 1024


def test_macos_reads_the_same_number_as_bytes(monkeypatch):
    """Getting this backwards recommends one worker on a large machine or
    forty on a small one."""
    monkeypatch.setitem(sys.modules, "resource", _fake_resource(2048))
    monkeypatch.setattr(sys, "platform", "darwin")

    assert B._rss_bytes() == 2048


def test_a_platform_without_the_resource_module_reports_no_footprint(
        monkeypatch):
    """Windows has no `resource`. Zero means "not measured", and
    `recommend_workers` falls back to the core count rather than dividing by
    a number that is not there."""
    monkeypatch.delitem(sys.modules, "resource", raising=False)
    monkeypatch.setattr(sys, "meta_path",
                        [_BlockedImport("resource")] + list(sys.meta_path))

    assert B._rss_bytes() == 0


def test_no_resource_module_leaves_the_core_count_as_the_only_bound(
        monkeypatch):
    """End to end: the whole benchmark still returns, and the recommendation
    says why it could not use memory."""
    monkeypatch.delitem(sys.modules, "resource", raising=False)
    monkeypatch.setattr(sys, "meta_path",
                        [_BlockedImport("resource")] + list(sys.meta_path))

    measurement = B.benchmark(lambda item: item * 2, [1, 2, 3], warmup=0)
    assert measurement.peak_rss_bytes == 0
    assert measurement.work_rss_bytes == 0

    out = B.recommend_workers(measurement, cores=6, available_bytes=64 * GIB)
    assert out.workers == 6
    assert "no usable memory measurement" in out.reason


# ---------------------------------------------------------------------------
# The CUDA peak counter has to be reset before the clock starts
# ---------------------------------------------------------------------------

class _FakeCuda:
    def __init__(self, peak=0, available=True):
        self.peak = peak
        self._available = available

    def is_available(self):
        return self._available

    def max_memory_allocated(self):
        return self.peak

    def reset_peak_memory_stats(self):
        self.peak = 0


def _install_fake_torch(monkeypatch, cuda):
    torch = types.ModuleType("torch")
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_the_peak_from_an_earlier_run_is_not_reported_as_this_ones(
        monkeypatch):
    """`max_memory_allocated` is a high-water mark for the whole process. A
    previous segmentation leaves 12 GiB standing there, and reporting it as
    one field's requirement recommends a single worker on a machine that
    could feed eight."""
    cuda = _FakeCuda(peak=12 * GIB)
    _install_fake_torch(monkeypatch, cuda)

    def work(item):
        cuda.peak = max(cuda.peak, 3 * GIB)

    measurement = B.benchmark(work, [1, 2, 3], warmup=0)

    assert measurement.peak_vram_bytes == 3 * GIB
    assert measurement.notes == []


def test_a_torch_without_a_device_is_reported_as_not_measured(monkeypatch):
    """None and 0 are different answers, and the note says which this is."""
    _install_fake_torch(monkeypatch, _FakeCuda(peak=99, available=False))

    measurement = B.benchmark(lambda item: None, [1, 2], warmup=0)

    assert measurement.peak_vram_bytes is None
    assert "no CUDA device" in measurement.notes[0]


def test_a_driver_that_raises_on_reset_does_not_end_the_benchmark(
        monkeypatch):
    """A CUDA call can fail on a busy or half-initialised device. Losing the
    reset costs accuracy; raising costs the whole measurement."""
    class _Exploding(_FakeCuda):
        def reset_peak_memory_stats(self):
            raise RuntimeError("CUDA driver is shutting down")

    _install_fake_torch(monkeypatch, _Exploding(peak=5))

    measurement = B.benchmark(lambda item: None, [1, 2, 3], warmup=0)

    assert measurement.items == 3
    assert measurement.peak_vram_bytes == 5


def test_reset_is_a_no_op_when_there_is_no_device(monkeypatch):
    cuda = _FakeCuda(peak=7, available=False)
    _install_fake_torch(monkeypatch, cuda)

    B._reset_vram()

    assert cuda.peak == 7


# ---------------------------------------------------------------------------
# MemAvailable
# ---------------------------------------------------------------------------

def _meminfo(text, monkeypatch):
    def fake_open(path, *args, **kwargs):
        assert str(path) == "/proc/meminfo"
        return io.StringIO(text)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_meminfo_without_a_memavailable_line_reports_zero(monkeypatch):
    """Old kernels have MemFree but no MemAvailable. Falling back to MemFree
    would size workers against memory the page cache is already using."""
    _meminfo("MemTotal:       32000000 kB\nMemFree:         1000000 kB\n",
             monkeypatch)

    assert B.available_memory_bytes() == 0


def test_a_memavailable_line_that_is_not_a_number_reports_zero(monkeypatch):
    _meminfo("MemAvailable:   not-a-number kB\n", monkeypatch)

    assert B.available_memory_bytes() == 0


def test_a_truncated_memavailable_line_reports_zero(monkeypatch):
    """A /proc read torn mid-line has the key and no value."""
    _meminfo("MemAvailable:\n", monkeypatch)

    assert B.available_memory_bytes() == 0


def test_memavailable_is_read_as_kilobytes(monkeypatch):
    _meminfo("MemTotal:       32000000 kB\nMemAvailable:   16000000 kB\n",
             monkeypatch)

    assert B.available_memory_bytes() == 16000000 * 1024


def test_the_recommendation_uses_meminfo_when_it_is_not_told_a_number(
        monkeypatch):
    """`available_bytes=None` is the real call: nothing passes the number in."""
    _meminfo("MemAvailable:   8388608 kB\n", monkeypatch)  # 8 GiB
    measurement = B.Measurement(items=2, seconds=1.0,
                                peak_rss_bytes=3 * GIB,
                                baseline_rss_bytes=1 * GIB)

    out = B.recommend_workers(measurement, cores=16, reserve_bytes=2 * GIB)

    assert out.available_bytes == 8 * GIB
    # (8 - 2) GiB budget / 2 GiB per worker
    assert out.workers == 3
    assert "memory-bound" in out.reason


# ---------------------------------------------------------------------------
# The arithmetic that has no measurement to divide by
# ---------------------------------------------------------------------------

def test_a_measurement_over_no_items_reports_nan_rather_than_dividing_by_zero():
    empty = B.Measurement(items=0, seconds=0.0, peak_rss_bytes=0)

    assert empty.per_item_seconds != empty.per_item_seconds  # NaN
    assert empty.items_per_second != empty.items_per_second


def test_a_run_that_took_no_measurable_time_reports_nan_throughput():
    instant = B.Measurement(items=4, seconds=0.0, peak_rss_bytes=GIB)

    assert instant.items_per_second != instant.items_per_second
    assert instant.per_item_seconds == 0.0


def test_the_core_count_is_read_from_the_machine_when_none_is_given(
        monkeypatch):
    monkeypatch.setattr(B.os, "cpu_count", lambda: 12)

    out = B.recommend_workers(None, available_bytes=64 * GIB)

    assert out.cores == 12
    assert out.workers == 12


def test_a_machine_that_will_not_say_how_many_cores_it_has_gets_one(
        monkeypatch):
    """`os.cpu_count()` returns None in some containers."""
    monkeypatch.setattr(B.os, "cpu_count", lambda: None)

    out = B.recommend_workers(None, available_bytes=64 * GIB)

    assert out.cores == 1
    assert out.workers == 1


def test_a_negative_core_count_is_still_one_worker():
    """`cpu_count() - 4` on a two-core machine is the default this replaces."""
    out = B.recommend_workers(None, cores=-2, available_bytes=64 * GIB)

    assert out.workers == 1
    assert out.cores == 1


def test_a_recommendation_prints_its_number_and_its_reason():
    out = B.Recommendation(workers=4, reason="core-bound: 4 core(s)")

    assert str(out) == "4 worker(s): core-bound: 4 core(s)"


def test_a_warm_up_note_is_absent_when_nothing_was_warmed_up():
    seen = []
    measurement = B.benchmark(seen.append, [1, 2, 3], warmup=0)

    assert seen == [1, 2, 3]
    assert measurement.items == 3
    assert not any("warm-up" in note for note in measurement.notes)


def test_the_warm_up_item_is_run_before_the_clock_and_named_in_the_notes():
    seen = []
    measurement = B.benchmark(seen.append, ["a", "b", "c"], warmup=1)

    assert seen == ["a", "b", "c"]
    assert measurement.items == 2
    assert "1 warm-up item(s) excluded from the timing" in measurement.notes
