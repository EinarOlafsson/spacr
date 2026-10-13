"""An in-memory prediction join that refuses rather than inventing numbers.

:func:`spacr.predictions.attach_predictions` is what a montage uses when the
database has no score column of its own. Three ways it can come up empty --
nothing to join, a score table missing the column that was asked for, and a
key that lands on no row at all -- each have to return the ORIGINAL frame and
a matched count of 0, because a caller that sees columns full of NaN cannot
tell "no scores here" from "every score is missing".
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.predictions import CV_CLASS_COLUMN, CV_SCORE_COLUMN, attach_predictions


def _objects(n=3):
    return pd.DataFrame([
        {"png_path": f"/crops/plate1_r1_c{i}_f1_{i}.png",
         "file_name": f"plate1_r1_c{i}_f1_{i}.png",
         "plateID": "plate1", "rowID": "r1", "columnID": f"c{i}",
         "fieldID": "f1", "object_label": i}
        for i in range(n)])


def _scores(n=3, **extra):
    rows = []
    for i in range(n):
        row = {"path": f"/crops/plate1_r1_c{i}_f1_{i}.png",
               "pred": 0.1 * (i + 1), "cv_predictions": i % 2}
        row.update({k: v[i] for k, v in extra.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Nothing to join
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("objects,results", [
    (None, _scores()),
    (_objects(), None),
    (_objects().iloc[:0], _scores()),
    (_objects(), _scores().iloc[:0]),
])
def test_an_empty_side_joins_nothing_and_says_zero(objects, results):
    """An empty frame on either side is a real answer, not an exception.

    The caller decides what to do about zero; it must not have to catch a
    TypeError from a join it never should have started.
    """
    frame, matched = attach_predictions(objects, results)

    assert matched == 0
    assert frame is objects


def test_a_frame_with_no_usable_key_comes_back_untouched():
    """No key means no join -- and the caller still gets its frame back."""
    objects = pd.DataFrame({"nothing_recognisable": [1, 2, 3]})
    results = pd.DataFrame({"also_nothing": [1, 2, 3], "pred": [0.1, 0.2, 0.3]})

    frame, matched = attach_predictions(objects, results)

    assert matched == 0
    assert frame is objects
    assert CV_SCORE_COLUMN not in frame.columns


# ---------------------------------------------------------------------------
# A join that can only carry half of what was asked for
# ---------------------------------------------------------------------------

def test_a_missing_source_column_is_skipped_not_invented():
    """A score table without `cv_predictions` still delivers its scores.

    The class column is simply absent from the result, rather than present
    and full of NaN, so a montage can tell it has no class to colour by.
    """
    results = _scores().drop(columns=["cv_predictions"])

    frame, matched = attach_predictions(_objects(), results)

    assert matched == 3
    assert CV_SCORE_COLUMN in frame.columns
    assert CV_CLASS_COLUMN not in frame.columns


def test_a_column_that_matches_no_row_is_not_added():
    """A source column present but joining nowhere adds nothing.

    `cv_predictions` here is all-NA, so every joined value is missing and the
    column would carry no information -- it stays off the frame, and the
    matched count reports only the column that really landed.
    """
    results = _scores()
    results["cv_predictions"] = [None, None, None]

    frame, matched = attach_predictions(_objects(), results)

    assert matched == 3
    assert CV_SCORE_COLUMN in frame.columns
    assert CV_CLASS_COLUMN not in frame.columns


def test_a_join_that_lands_on_no_row_returns_the_original_frame():
    """Keys that agree with nothing must not leave an all-NaN score column."""
    results = _scores()
    results["path"] = [f"/elsewhere/other_x1_y1_f9_{i}.png" for i in range(3)]
    objects = _objects().drop(columns=["file_name", "plateID", "rowID",
                                       "columnID", "fieldID", "object_label"])

    frame, matched = attach_predictions(objects, results)

    assert matched == 0
    assert frame is objects
    assert CV_SCORE_COLUMN not in frame.columns
