"""What a resource reading does when the machine will not answer.

The happy paths -- `/proc/self/statm` on Linux, no Torch in `sys.modules` --
are what an ordinary test run measures, so they are the only ones the rest
of the suite reaches. The readings that matter for a failure report are the
other ones: a container with no `/proc`, a build with no psutil, a process
that has already imported Torch and is holding GPU memory, and a Torch whose
CUDA layer raises when asked.

Every one of them is produced by INJECTION rather than by asserting a mock
was called: the file-open that `host_rss` makes is made to fail, the psutil
import is blocked at `builtins.__import__`, and a Torch stand-in is placed in
`sys.modules` -- which is the exact lookup `gpu_allocated` documents itself as
doing, precisely so it never imports Torch to take a measurement.
"""

import builtins
import sys
import types

from spacr.fit_resources import (RESOURCE_KEY, STAGE_KEY, describe_resources,
                                 gpu_allocated, host_rss, peak, readable,
                                 record_stage)


# ---------------------------------------------------------------------------
# resident memory
# ---------------------------------------------------------------------------

def _break_statm(monkeypatch, error=OSError("no /proc here")):
    """Make only `/proc/self/statm` unopenable; everything else is untouched."""
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        if str(file) == "/proc/self/statm":
            raise error
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)


def test_a_container_without_proc_falls_back_to_psutil(monkeypatch):
    """With `/proc/self/statm` unreadable the reading still comes back.

    A number, not None: psutil is the documented second source and it is
    installed here, so "nobody measured" would be the wrong answer.
    """
    _break_statm(monkeypatch)
    reading = host_rss()
    assert isinstance(reading, int)
    assert reading > 0


def test_a_malformed_statm_is_a_failure_not_a_wrong_number(monkeypatch):
    """A `/proc/self/statm` that does not parse falls through, not to zero."""
    _break_statm(monkeypatch, error=ValueError("not a number"))
    assert host_rss() > 0


def test_with_no_proc_and_no_psutil_nobody_measured(monkeypatch):
    """Both sources gone reports None -- never zero.

    Zero would read as "nothing was using memory", which is the opposite
    finding from "nobody measured", and a failure report that confuses the
    two blames the wrong thing.
    """
    _break_statm(monkeypatch)
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)

    assert host_rss() is None


def test_an_unmeasured_reading_is_spelled_not_measured():
    """`readable(None)` says so in words rather than printing 0.0 B."""
    assert readable(None) == "not measured"
    assert readable(0) == "0.0 B"
    assert readable(2048) == "2.0 KB"


# ---------------------------------------------------------------------------
# GPU high-water mark
# ---------------------------------------------------------------------------

def _fake_torch(monkeypatch, *, available=True, allocated=0, peak_bytes=0,
                raises=None):
    """A Torch stand-in placed where `gpu_allocated` looks: `sys.modules`."""
    cuda = types.SimpleNamespace()

    def is_available():
        if raises is not None:
            raise raises
        return available

    cuda.is_available = is_available
    cuda.memory_allocated = lambda: allocated
    cuda.max_memory_allocated = lambda: peak_bytes
    module = types.ModuleType("torch")
    module.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_the_gpu_reading_is_the_high_water_mark_not_the_current_one(
        monkeypatch):
    """The larger of current and peak allocation is reported.

    Fit tensors are often already released by the time a stage boundary is
    recorded, so the current allocation would report a fit that filled the
    card as having used nothing.
    """
    _fake_torch(monkeypatch, allocated=4 * 1024, peak_bytes=900 * 1024 * 1024)
    assert gpu_allocated() == 900 * 1024 * 1024


def test_a_card_that_is_not_there_is_not_a_measurement(monkeypatch):
    """Torch imported but no CUDA reports None rather than zero bytes."""
    _fake_torch(monkeypatch, available=False, peak_bytes=1234)
    assert gpu_allocated() is None


def test_a_torch_that_raises_when_asked_reports_nothing(monkeypatch):
    """A CUDA layer that throws does not throw out of the diagnostic."""
    _fake_torch(monkeypatch, raises=RuntimeError("driver mismatch"))
    assert gpu_allocated() is None


def test_torch_is_never_imported_to_take_a_measurement(monkeypatch):
    """With no Torch in `sys.modules` the answer is None and no import runs."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def refuse_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(
                "gpu_allocated imported torch to take a measurement")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_torch)
    assert gpu_allocated() is None


# ---------------------------------------------------------------------------
# the per-stage table
# ---------------------------------------------------------------------------

def test_a_stage_is_recorded_with_both_readings(monkeypatch):
    """`record_stage` writes the stage name and appends one reading."""
    _fake_torch(monkeypatch, allocated=64, peak_bytes=128)
    settings = {}
    reading = record_stage(settings, "design matrix")

    assert settings[STAGE_KEY] == "design matrix"
    assert settings[RESOURCE_KEY] == [reading]
    assert reading["stage"] == "design matrix"
    assert reading["gpu"] == 128


def test_an_immutable_settings_object_does_not_break_the_fit():
    """A settings object that refuses assignment still yields a reading.

    Diagnostics are best effort; a mapping that cannot be written to is not
    a reason to stop the regression that was being measured.
    """
    class Frozen(dict):
        def __setitem__(self, key, value):
            raise TypeError("frozen")

    settings = Frozen()
    reading = record_stage(settings, "fit")
    assert reading["stage"] == "fit"
    assert settings == {}


def test_the_peak_of_nothing_is_empty_not_zero():
    """No readings gives {} -- there is no measured peak to report."""
    assert peak({}) == {}
    assert peak({RESOURCE_KEY: []}) == {}
    assert describe_resources({}) == ""


def test_a_settings_object_that_cannot_be_read_reports_nothing():
    """A `.get` that raises is reported as no readings, not as a crash."""
    class Hostile:
        def get(self, key, default=None):
            raise RuntimeError("no")

    assert peak(Hostile()) == {}
    assert describe_resources(Hostile()) == ""


def test_the_peak_names_the_stage_it_was_reached_in(monkeypatch):
    """The table ends with the largest reading and where it was taken."""
    settings = {RESOURCE_KEY: [
        {"stage": "load", "rss": 100, "gpu": None},
        {"stage": "fit", "rss": 900, "gpu": 4096},
        {"stage": "report", "rss": 200, "gpu": 2048},
    ]}
    high = peak(settings)
    assert high == {"rss": 900, "rss_stage": "fit",
                    "gpu": 4096, "gpu_stage": "fit"}

    table = describe_resources(settings)
    lines = table.splitlines()
    assert lines[0].split() == ["stage", "resident", "GPU"]
    assert "load" in table and "report" in table
    # a stage with no GPU reading says so rather than claiming zero
    assert "not measured" in table
    assert "PEAK resident 900.0 B at 'fit'" in table
    assert "PEAK GPU      4.0 KB at 'fit'" in table


def test_a_stage_name_too_long_for_the_column_is_truncated_not_wrapped():
    """The stage column stays one line per stage, however long the name."""
    long_name = "a" * 80
    table = describe_resources({RESOURCE_KEY: [
        {"stage": long_name, "rss": 1, "gpu": None}]})
    lines = table.splitlines()
    # header, the one stage, and the peak line
    assert len(lines) == 3
    stage_row = lines[1]
    assert stage_row.strip().startswith("a" * 34)
    assert "a" * 35 not in stage_row
    assert stage_row.endswith("not measured")
