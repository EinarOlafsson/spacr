"""Which memory figure was read, and the refusal to report one it could not.

The module docstring gives the rule: "USS where the platform gives it, then
PSS, then RSS -- and every record NAMES the measure it used, because RSS
double-counts the pages a fork shares and would overstate a sweep badly. A
number whose definition is unrecorded cannot be compared between two machines."

Every branch here is that preference order or the honesty about it.
"""
from __future__ import annotations

import pytest


class _Full:
    """A ``memory_full_info`` result carrying whichever measures are given."""

    def __init__(self, **values):
        for name, value in values.items():
            setattr(self, name, value)


class _Info:
    def __init__(self, rss=None):
        if rss is not None:
            self.rss = rss


class _Psutil:
    """The minimum of psutil ``_memory`` reaches for: one exception class."""

    class NoSuchProcess(Exception):
        pass


class _Process:
    """A stand-in psutil Process with controllable memory calls."""

    pid = 4321

    def __init__(self, full=None, rss=None, full_raises=False):
        self._full = full
        self._rss = rss
        self._full_raises = full_raises

    def memory_full_info(self):
        if self._full_raises:
            raise PermissionError("private memory needs more rights")
        return self._full

    def memory_info(self):
        return _Info(self._rss)


def test_uss_is_preferred_when_the_platform_gives_it():
    """First in the order, and the least misleading figure of the three."""
    from spacr.resource_log import _memory

    value, measure = _memory(_Psutil, _Process(full=_Full(uss=100, pss=200,
                                                       rss=300)))

    assert (value, measure) == (100, "uss")


def test_pss_is_next_when_uss_is_absent():
    """Linux without USS still reports PSS, which at least shares fairly."""
    from spacr.resource_log import _memory

    value, measure = _memory(_Psutil, _Process(full=_Full(pss=200, rss=300)))

    assert (value, measure) == (200, "pss")


def test_rss_is_last_and_is_still_named():
    """The fallback the docstring warns about.

    RSS double-counts the pages a fork shares, so a sweep of eight children
    reads far larger than it is. Reporting it is fine; reporting it WITHOUT
    saying so is what makes two machines incomparable.
    """
    from spacr.resource_log import _memory

    value, measure = _memory(_Psutil, _Process(full=_Full(rss=300)))

    assert (value, measure) == (300, "rss")


def test_a_non_integer_measure_is_passed_over():
    """The ``isinstance(value, int)`` check.

    A mock, a None from a platform that declares the attribute without
    filling it, or a float from a future psutil would otherwise be recorded as
    a byte count.
    """
    from spacr.resource_log import _memory

    value, measure = _memory(_Psutil, _Process(full=_Full(uss=None, pss="200",
                                                       rss=300)))

    assert (value, measure) == (300, "rss")


def test_private_figures_that_are_refused_fall_back_to_resident():
    """The except: private memory needs permissions the resident one does not.

    Reading another user's process, or a hardened container, raises here --
    and the sample must still carry a number rather than nothing.
    """
    from spacr.resource_log import _memory

    value, measure = _memory(_Psutil, _Process(full_raises=True, rss=300))

    assert (value, measure) == (300, "rss")


def test_a_process_that_reports_no_resident_size_reports_nothing():
    """Both None, which is the documented "could not read this one".

    A zero would read as a process using no memory, which is never true and
    would drag a peak down.
    """
    from spacr.resource_log import _memory

    value, measure = _memory(_Psutil, _Process(full_raises=True, rss=None))

    assert (value, measure) == (None, None)


# ---------------------------------------------------------------------------
# summarise — the largest process across a run
# ---------------------------------------------------------------------------

def _sample(time, members, *, measure="uss", missed=0, total=None):
    """One record as ``tree_sample`` writes it."""
    row = {"time": time, "processes": list(members),
           "measure": measure, "missed": missed}
    if total is not None:
        row["total"] = total
    return row


def test_the_largest_process_is_found_across_every_sample():
    """The scan the report is built from."""
    from spacr.resource_log import summarise

    out = summarise([
        _sample(1.0, [{"pid": 1, "name": "a", "memory": 100, "measure": "uss"},
                      {"pid": 2, "name": "b", "memory": 900, "measure": "uss"}]),
        _sample(2.0, [{"pid": 1, "name": "a", "memory": 300, "measure": "uss"}]),
    ])

    assert out["peak_process"]["pid"] == 2
    assert out["peak_process"]["memory"] == 900


def test_a_member_with_no_readable_memory_is_passed_over():
    """The ``isinstance(figure, int)`` skip.

    A child that exited between being enumerated and being read has no figure,
    and the module docstring calls that "an expected outcome and not an
    error". Comparing None against an int would raise mid-summary.
    """
    from spacr.resource_log import summarise

    out = summarise([
        _sample(1.0, [{"pid": 1, "name": "gone", "memory": None},
                      {"pid": 2, "name": "b", "memory": 500, "measure": "uss"}]),
    ])

    assert out["peak_process"]["pid"] == 2


def test_no_readable_member_anywhere_names_no_peak_process():
    """Every member skipped, so there is nothing to name.

    The key is ABSENT rather than None, matching the docstring's rule that an
    empty summary is empty: "nothing was using memory" and "nobody measured"
    are opposite findings.
    """
    from spacr.resource_log import summarise

    out = summarise([_sample(1.0, [{"pid": 1, "name": "gone", "memory": None}])])

    assert "peak_process" not in out
