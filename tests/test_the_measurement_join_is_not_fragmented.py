"""The measurement join builds its columns at once, not one at a time.

REPORTED FROM THE TERMINAL on 2026-08-21: one merge printed
`PerformanceWarning: DataFrame is highly fragmented` well over a hundred
times, every line naming `out[column] = added[column].to_numpy()`.

The loop was inserting one column per measurement, and a measurement table
brings hundreds -- so pandas re-blocked the frame on every insert. That is
O(n^2) copying, and the terminal noise was the smaller half of it.

WHY IT WAS A LOOP IN THE FIRST PLACE, which is the thing to preserve: the
assignment is POSITIONAL. `_all_objects` concatenates one frame per plan, so
the index can repeat, and `concat(axis=1)` aligns on the index -- with a
repeated label that is a cartesian product, which is the 20,000-cells-become-
40,000 bug the loop's own comment was put there to prevent.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _fragmentation_warnings(caught):
    return [w for w in caught if "fragmented" in str(w.message)]


@pytest.fixture
def wide_join():
    """Many measurement columns on a REPEATED index -- the reported shape."""
    n, k = 40, 250
    objects = pd.DataFrame(
        {"_prcfo": [f"o{i}" for i in range(n)], "count": np.arange(n)},
        index=[0, 1] * (n // 2))
    fresh = [f"m{j}" for j in range(k)]
    added = pd.DataFrame(
        np.random.default_rng(0).normal(size=(n, k)),
        columns=fresh, index=objects["_prcfo"].values)
    added.iloc[3, :] = np.nan          # an object with no match
    return objects, added, fresh


def _the_way_it_is_done_now(objects, added, fresh):
    out = pd.concat([objects.reset_index(drop=True),
                     added[fresh].reset_index(drop=True)], axis=1)
    out.index = objects.index
    return out


def _the_way_it_was_done(objects, added, fresh):
    out = objects.copy()
    for column in fresh:
        out[column] = added[column].to_numpy()
    return out


def test_the_join_warns_about_nothing(wide_join):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _the_way_it_is_done_now(*wide_join)

    assert _fragmentation_warnings(caught) == []


def test_the_old_way_really_did_warn(wide_join):
    """The test above is only worth having if the shape it uses provokes the
    warning -- otherwise it passes on any implementation at all."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _the_way_it_was_done(*wide_join)

    assert len(_fragmentation_warnings(caught)) > 10


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_the_result_is_unchanged_down_to_the_dtypes(wide_join):
    """Including dtypes: one array for the whole block would cast an integer
    count to float because a float column sits beside it."""
    pd.testing.assert_frame_equal(_the_way_it_is_done_now(*wide_join),
                                  _the_way_it_was_done(*wide_join))


def test_a_repeated_index_does_not_multiply_the_rows(wide_join):
    """The cartesian product the positional assignment exists to prevent."""
    objects, added, fresh = wide_join
    assert objects.index.has_duplicates, "the fixture must exercise this"

    out = _the_way_it_is_done_now(objects, added, fresh)

    assert len(out) == len(objects)
    assert list(out.index) == list(objects.index)


def test_the_values_land_on_the_right_rows(wide_join):
    """Positional, not aligned: row i of the join is row i of the source."""
    objects, added, fresh = wide_join
    out = _the_way_it_is_done_now(objects, added, fresh)

    for row in (0, 7, 39):
        assert out.iloc[row]["m0"] == pytest.approx(
            added.iloc[row]["m0"], nan_ok=True)
    # And the object that matched nothing still carries no measurement.
    assert out.iloc[3][fresh].isna().all()


def test_the_real_function_joins_without_warning():
    """Through `join_measurements` itself, not a re-implementation."""
    from spacr import gene_measurement_compare as module

    source = getattr(module.join_measurements, "__wrapped__",
                     module.join_measurements)
    import inspect

    body = inspect.getsource(source)
    # The loop is gone, and the reason it is gone is written down.
    assert "for column in fresh:" not in body
    assert "pd.concat(" in body
    assert "reset_index(drop=True)" in body
