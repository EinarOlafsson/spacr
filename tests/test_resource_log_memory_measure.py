"""Which memory figure ``spacr.resource_log._memory`` reports, and how it labels it.

USS first, then PSS, then RSS: a platform that will not give the private
figures still gives the resident one, and the row says which it is. A figure
that is not an integer (``None``, a string, a float) is not a measurement, and
when no figure is available at all the answer is ``(None, None)`` -- never a
zero that would read as "this process used no memory".
"""
from __future__ import annotations

import pytest

from spacr.resource_log import _memory


class _Psutil:
    """The one psutil attribute ``_memory`` reads."""

    class NoSuchProcess(Exception):
        pass


def _figures(**values):
    """An object carrying exactly the named attributes."""
    return type("Figures", (), values)()


class _Process:
    """A process whose full and resident figures are given, or raise."""

    pid = 4242

    def __init__(self, full=None, rss="absent", full_raises=None):
        self.full = full
        self.rss = rss
        self.full_raises = full_raises

    def memory_full_info(self):
        if self.full_raises is not None:
            raise self.full_raises
        return self.full

    def memory_info(self):
        if self.rss == "absent":
            raise AssertionError("the private figures were available")
        return _figures(rss=self.rss)


def test_memory_takes_the_first_integer_figure_then_resident_then_nothing(
        subtests):
    """Every combination of available figures maps to one exact ``(bytes, measure)``."""
    cases = [
        ("an integer uss wins", _Process(_figures(uss=111, pss=222, rss=333)),
         (111, "uss")),
        ("no uss, an integer pss", _Process(_figures(uss=None, pss=222)),
         (222, "pss")),
        ("float private figures fall through to rss",
         _Process(_figures(uss=1.5, pss=2.5), rss=7_340_032),
         (7_340_032, "rss")),
        ("None and a string fall through to rss",
         _Process(_figures(uss=None, pss="not a number"), rss=4096),
         (4096, "rss")),
        ("full info refused, rss read",
         _Process(rss=2048, full_raises=PermissionError("denied")),
         (2048, "rss")),
        ("full info is None, rss read", _Process(None, rss=1024),
         (1024, "rss")),
        ("no resident figure either reports nothing",
         _Process(rss=None, full_raises=RuntimeError("permission denied")),
         (None, None)),
        ("a process that exited is not reported as rss",
         _Process(rss=1024, full_raises=_Psutil.NoSuchProcess(4242)),
         _Psutil.NoSuchProcess),
    ]
    for case_id, process, expected in cases:
        with subtests.test(msg=case_id):
            if isinstance(expected, tuple):
                assert _memory(_Psutil, process) == expected
            else:
                with pytest.raises(expected):
                    _memory(_Psutil, process)
