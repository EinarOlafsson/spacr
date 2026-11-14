"""Column kinds and the saved filter state of the data filter panel.

`classify_columns` memoises on frame IDENTITY, made sound by a weakref (a
reused address is not a hit) and a shape check (a frame mutated in place is
classified again); the memo is capped and reaps dead entries first.  Saved
filter rows name their column and kind, restore inside-range bounds exactly
and skip rows whose clause could not be added.
"""
from __future__ import annotations

import gc

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import data_filter_panel as D             # noqa: E402
from spacr.qt.widgets.data_filter_panel import DataFilterPanel  # noqa: E402

pytestmark = pytest.mark.qt


class _FrameThatCannotBeWeakReferenced:
    """A frame-shaped view with no ``__weakref__`` slot."""

    __slots__ = ("_frame",)

    def __init__(self, frame):
        self._frame = frame

    @property
    def columns(self):
        return self._frame.columns

    @property
    def shape(self):
        return self._frame.shape

    def __getitem__(self, name):
        return self._frame[name]


def _frame():
    return pd.DataFrame({"plate": ["p1", "p1", "p2"], "value": [1.0, 2.0, 3.0]})


_KINDS = {"plate": "category", "value": "category"}


@pytest.fixture
def counted(monkeypatch):
    """An empty cache, and the shapes of the frames actually classified.

    Shapes, not frames: holding the frames would keep them alive and defeat
    the weakref cases below.
    """
    D._KINDS_CACHE.clear()
    calls = []
    real = D._classify_columns_uncached
    monkeypatch.setattr(D, "_classify_columns_uncached",
                        lambda f: calls.append(f.shape) or real(f))
    yield calls
    D._KINDS_CACHE.clear()


def test_classify_columns_memoises_soundly_on_frame_identity(counted, subtests):
    """Hits, misses after mutation or collection, the cap, the reap and unweakrefable frames."""
    with subtests.test(msg="the same frame twice is classified once, each caller gets its own dict"):
        frame = _frame()
        first = D.classify_columns(frame)
        first["injected"] = "nonsense"
        assert D.classify_columns(frame) == _KINDS
        assert counted == [(3, 2)]

    with subtests.test(msg="a column added in place is classified again"):
        frame["added"] = [1, 2, 3]
        assert D.classify_columns(frame) == dict(_KINDS, added="category")
        assert counted == [(3, 2), (3, 3)]

    with subtests.test(msg="a row dropped in place changes the recorded shape"):
        frame = _frame()
        D.classify_columns(frame)
        frame.drop(index=frame.index[-1], inplace=True)
        _ref, shape, _cached = D._KINDS_CACHE[id(frame)]
        assert (shape, frame.shape) == ((3, 2), (2, 2))

    with subtests.test(msg="the cache holds no strong reference and a dead weakref is no hit"):
        D._KINDS_CACHE.clear()
        frame = _frame()
        D.classify_columns(frame)
        ref, shape, cached = D._KINDS_CACHE[id(frame)]
        assert ref() is frame
        del frame
        assert ref() is None
        other = pd.DataFrame({"plate": ["p9", "p9", "p9"],
                              "value": [4.0, 5.0, 6.0]})
        assert other.shape == shape
        D._KINDS_CACHE[id(other)] = (ref, shape, dict(cached, value="stale"))
        assert D.classify_columns(other) == _KINDS

    with subtests.test(msg="the cache never grows past its cap"):
        frames = [_frame() for _ in range(D._KINDS_CACHE_MAX + 3)]
        for each in frames:
            D.classify_columns(each)
        assert len(D._KINDS_CACHE) == D._KINDS_CACHE_MAX == 8
        del frames, each

    with subtests.test(msg="dead entries are reaped together, not evicted one by one"):
        D._KINDS_CACHE.clear()
        loaded = [_frame() for _ in range(D._KINDS_CACHE_MAX)]
        for each in loaded:
            D.classify_columns(each)
        assert len(D._KINDS_CACHE) == D._KINDS_CACHE_MAX
        loaded.clear()
        del each
        gc.collect()
        assert all(r() is None for r, _s, _c in D._KINDS_CACHE.values())
        kept = _frame()
        D.classify_columns(kept)
        assert [r() for r, _s, _c in D._KINDS_CACHE.values()] == [kept]

    for case, store in (("classified", None),
                        ("remembered", {"cell_area": "range",
                                        "plate": "category", "extra": "range"})):
        with subtests.test(msg=f"an unweakrefable frame is {case} but never cached"):
            D._KINDS_CACHE.clear()
            table = pd.DataFrame({"cell_area": range(100), "plate": ["p1"] * 100})
            view = _FrameThatCannotBeWeakReferenced(table)
            if store is not None:
                D._remember_column_kinds(view, store)
            else:
                assert D.classify_columns(view) == {"cell_area": "range",
                                                    "plate": "category"}
            assert id(view) not in D._KINDS_CACHE
            assert D.classify_columns(view) == D.classify_columns(table) == {
                "cell_area": "range", "plate": "category"}


class _Link:
    def publish_filter(self, *args, **kwargs):
        pass

    def set_filter(self, *args, **kwargs):
        pass

    def subscribe(self, *args, **kwargs):
        return None


def test_the_panel_saves_and_restores_its_filter_rows(qapp, monkeypatch,
                                                      subtests):
    """state() names column and kind; restore() keeps legal bounds and skips rows it could not add."""
    frame = pd.DataFrame({"cell_area": [float(i) for i in range(100)]})

    with subtests.test(msg="a saved row names the column and the kind it was"):
        panel = DataFilterPanel(link=_Link())
        panel.set_frame(frame)
        panel.add_column("cell_area")
        assert panel.state() == {"version": 1, "filters": [
            {"kind": "range", "column": "cell_area", "enabled": True,
             "low": 0.0, "high": 99.0}]}

    with subtests.test(msg="bounds inside the range are restored exactly"):
        panel = DataFilterPanel(link=_Link())
        panel.set_frame(frame)
        missing = panel.restore({"version": 1, "filters": [
            {"kind": "range", "column": "cell_area", "low": 10.0, "high": 20.0}]})
        assert missing == []
        row = panel._rows["cell_area"]
        assert (row._low.value(), row._high.value()) == (10.0, 20.0)
        assert panel.state()["filters"][0]["low"] == 10.0

    with subtests.test(msg="a row whose clause was not added is skipped"):
        panel = DataFilterPanel(link=_Link())
        panel.set_frame(frame)
        monkeypatch.setattr(panel, "add_column", lambda column: None)
        missing = panel.restore({"version": 1, "filters": [
            {"kind": "range", "column": "cell_area", "low": 10.0, "high": 20.0},
            {"kind": "range", "column": "gone", "low": 0, "high": 1}]})
        assert missing == ["gone"]
        assert panel._rows == {}
        assert panel.state() == {"version": 1, "filters": []}
