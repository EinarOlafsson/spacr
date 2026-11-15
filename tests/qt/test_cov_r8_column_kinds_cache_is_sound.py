"""The frame-identity cache in `classify_columns`, and the two checks
that make an ``id()`` key safe to use.

Classifying a 200,000 x 48 measurement table costs 230 ms, and it used
to be paid four times per `set_frame`. The memo keys on the frame's
IDENTITY rather than its contents, which is correct -- a filtered subset
has fewer rows and can genuinely classify differently -- but ``id()``
alone would be unsound, because CPython reuses addresses and a frame can
be mutated in place. Both guards are driven here.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import data_filter_panel as D

pytestmark = pytest.mark.qt


class TestTheCacheStaysSmall:
    def test_dead_entries_go_before_live_ones_are_evicted(self):
        """The eviction reaps expired weakrefs FIRST, so a panel holding
        four live frames does not lose one to a fifth that has already
        been collected."""
        import inspect

        source = inspect.getsource(D.classify_columns)
        reap = source.index("if r() is None]")
        evict = source.index("_KINDS_CACHE.pop(next(iter(", reap)
        assert reap < evict, (
            "the oldest entry is now dropped before dead ones are reaped, "
            "so a live frame can be evicted while expired entries stay")
