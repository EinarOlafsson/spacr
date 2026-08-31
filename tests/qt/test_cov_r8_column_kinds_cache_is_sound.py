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

import gc

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import data_filter_panel as D

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _an_empty_cache():
    """Each test starts from a known cache, and leaves one behind."""
    D._KINDS_CACHE.clear()
    yield
    D._KINDS_CACHE.clear()


def _frame():
    return pd.DataFrame({
        "plate": ["p1", "p1", "p2"],
        "value": [1.0, 2.0, 3.0],
    })


class TestTheHit:

    def test_the_same_frame_twice_is_classified_once(self, monkeypatch):
        frame = _frame()
        calls = []
        real = D._classify_columns_uncached
        monkeypatch.setattr(D, "_classify_columns_uncached",
                            lambda f: calls.append(f) or real(f))

        first = D.classify_columns(frame)
        second = D.classify_columns(frame)

        assert first == second
        assert len(calls) == 1, (
            "the second call re-derived the classification, so the memo "
            "this function exists for is not working")

    def test_each_caller_gets_its_own_dict(self):
        """``GraphSpec.kinds_for`` updates the result in place, so handing
        back the cached object would let one caller corrupt the next."""
        frame = _frame()

        first = D.classify_columns(frame)
        first["injected"] = "nonsense"

        assert "injected" not in D.classify_columns(frame)


class TestTheShapeCheck:

    def test_a_frame_mutated_in_place_is_classified_again(self,
                                                          monkeypatch):
        """THE UNCOVERED ARC: the entry is found and does NOT match.

        A frame can gain a column without becoming a different object, so
        the identity check alone would answer with a classification that
        is missing the new column -- and the panel would offer a filter
        list that does not mention a column the user can see in the table.
        """
        frame = _frame()
        calls = []
        real = D._classify_columns_uncached
        monkeypatch.setattr(D, "_classify_columns_uncached",
                            lambda f: calls.append(f) or real(f))

        before = D.classify_columns(frame)
        assert "added" not in before

        frame["added"] = [1, 2, 3]
        after = D.classify_columns(frame)

        assert len(calls) == 2, (
            "a frame that gained a column in place was answered from the "
            "cache, so the shape check is not doing anything")
        assert "added" in after

    def test_a_row_dropped_in_place_is_classified_again(self):
        """The other axis of the shape, and the one that can change the
        ANSWER rather than the column list: fewer rows means fewer
        distinct values, which is what the kinds are derived from."""
        frame = _frame()
        D.classify_columns(frame)

        frame.drop(index=frame.index[-1], inplace=True)

        entry = D._KINDS_CACHE.get(id(frame))
        assert entry is not None
        _ref, shape, _cached = entry
        assert shape != frame.shape, (
            "dropping a row did not change the recorded shape, so the "
            "stale entry would be served")


class TestTheWeakref:

    def test_a_collected_frame_cannot_produce_a_false_hit(self):
        """THE OTHER HALF of the same arc: the weakref is dead.

        This is why ``id()`` alone is unsound. CPython reuses addresses,
        so a new frame can land on the id of one that has gone -- and the
        cache would then answer for a table it has never seen.
        """
        frame = _frame()
        D.classify_columns(frame)
        key = id(frame)

        entry = D._KINDS_CACHE[key]
        ref, _shape, _cached = entry
        assert ref() is frame

        del frame
        assert ref() is None, (
            "the cache holds a strong reference to the frame, so a loaded "
            "measurement table cannot be freed while the panel is open")

    def test_an_entry_whose_frame_is_gone_is_not_served(self):
        """Driven by planting the exact state a reused address produces:
        a live entry under a key whose weakref has expired."""
        frame = _frame()
        D.classify_columns(frame)
        key = id(frame)
        ref, shape, cached = D._KINDS_CACHE[key]
        del frame

        assert ref() is None

        # A different frame answering to the same key, WITH THE SAME
        # SHAPE. That last part is what makes this a test of the weakref
        # rather than of the shape: an address is reused by an object of
        # a similar size, so the cheap check agrees and only `is` can
        # tell the two frames apart.
        other = pd.DataFrame({
            "plate": ["p9", "p9", "p9"],
            "value": [4.0, 5.0, 6.0],
        })
        assert other.shape == shape, (
            "the planted entry no longer matches the new frame's shape, "
            "so the shape check would reject it and the weakref is not "
            "what this test is standing on")
        poisoned = dict(cached)
        poisoned["value"] = "this came from the frame that is gone"
        D._KINDS_CACHE[id(other)] = (ref, shape, poisoned)

        kinds = D.classify_columns(other)

        assert kinds["value"] != "this came from the frame that is gone", (
            "a dead weakref was treated as a hit, so a new frame was "
            "classified as the one that used to live at its address")


class TestTheCacheStaysSmall:

    def test_it_does_not_grow_past_its_cap(self):
        frames = [_frame() for _ in range(D._KINDS_CACHE_MAX + 3)]
        for frame in frames:
            D.classify_columns(frame)

        assert len(D._KINDS_CACHE) <= D._KINDS_CACHE_MAX

    def test_a_dead_entry_is_reaped_rather_than_counted(self):
        """THE ARC: the reaping loop with something to reap.

        Frames are the largest objects the panel holds, and the cap is
        four. An entry whose frame has been collected still occupies a
        slot, so without the reap a panel that had loaded and closed four
        tables would evict a LIVE one to make room -- paying the 230 ms
        classification again for a table that is still on screen.
        """
        # HELD WHILE THE CACHE FILLS, then dropped together. Classifying
        # four temporaries in a row does NOT fill it: CPython hands the
        # next frame the address the last one just released, so they
        # collapse onto two keys and the state under test never exists.
        loaded = [_frame() for _ in range(D._KINDS_CACHE_MAX)]
        for frame in loaded:
            D.classify_columns(frame)

        assert len(D._KINDS_CACHE) == D._KINDS_CACHE_MAX
        loaded.clear()          # the panel closed all four tables
        del frame               # ...and so does the loop variable
        gc.collect()            # pandas frames sit in reference cycles

        assert all(ref() is None
                   for ref, _shape, _cached in D._KINDS_CACHE.values()), (
            "the frames outlived the panel, so this test is not standing "
            "on the state it says it is")

        kept = _frame()
        D.classify_columns(kept)

        live = [ref for ref, _s, _c in D._KINDS_CACHE.values()
                if ref() is not None]
        assert len(live) == 1 and live[0]() is kept
        assert len(D._KINDS_CACHE) == 1, (
            "the dead entries were evicted one at a time by the cap "
            "instead of being reaped together")

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
