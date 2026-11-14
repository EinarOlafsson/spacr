"""The in-memory prediction join refuses rather than inventing numbers.

:func:`spacr.predictions.attach_predictions` is what a montage uses when the
database has no score column of its own. When there is nothing to join, no
usable key, a source column that is missing or all-NA, or keys that land on
no row, it returns the ORIGINAL frame (the same object) and a matched count
of 0 -- a caller that sees columns full of NaN cannot tell "no scores here"
from "every score is missing". A real join adds exactly the columns that
landed, with the joined values.
"""
from __future__ import annotations

import pandas as pd

from spacr.predictions import (CV_CLASS_COLUMN, CV_SCORE_COLUMN,
                               attach_predictions)


def _objects(n=3):
    return pd.DataFrame([
        {"png_path": f"/crops/plate1_r1_c{i}_f1_{i}.png",
         "file_name": f"plate1_r1_c{i}_f1_{i}.png",
         "plateID": "plate1", "rowID": "r1", "columnID": f"c{i}",
         "fieldID": "f1", "object_label": i}
        for i in range(n)])


def _scores(n=3):
    return pd.DataFrame([
        {"path": f"/crops/plate1_r1_c{i}_f1_{i}.png",
         "pred": 0.1 * (i + 1), "cv_predictions": i % 2}
        for i in range(n)])


def _with(frame, **columns):
    frame = frame.copy()
    for name, values in columns.items():
        frame[name] = values
    return frame


def test_attach_predictions_joins_what_lands_and_nothing_else(subtests):
    """Untouched frame and 0 for every empty join; exact columns otherwise."""
    objects = _objects()
    path_only = objects.drop(columns=["file_name", "plateID", "rowID",
                                      "columnID", "fieldID", "object_label"])
    untouched = [
        ("no objects", None, _scores()),
        ("no results", objects, None),
        ("an empty object frame", objects.iloc[:0], _scores()),
        ("an empty score frame", objects, _scores().iloc[:0]),
        ("no usable key", pd.DataFrame({"nothing_recognisable": [1, 2, 3]}),
         pd.DataFrame({"also_nothing": [1, 2, 3], "pred": [0.1, 0.2, 0.3]})),
        ("keys that land on no row", path_only,
         _with(_scores(), path=[f"/elsewhere/other_x1_y1_f9_{i}.png"
                                for i in range(3)])),
        ("every source column all-NA", objects,
         _with(_scores(), pred=[None] * 3, cv_predictions=[None] * 3)),
    ]
    for case_id, left, right in untouched:
        with subtests.test(msg=case_id):
            before = None if left is None else left.copy()
            frame, matched = attach_predictions(left, right)
            assert matched == 0
            assert frame is left
            if left is not None:
                pd.testing.assert_frame_equal(left, before)

    joined = [
        ("both columns land", _scores(), [0.1, 0.2, 0.30000000000000004],
         [0, 1, 0]),
        ("a missing class column is skipped, not invented",
         _scores().drop(columns=["cv_predictions"]),
         [0.1, 0.2, 0.30000000000000004], None),
        ("an all-NA class column is not added",
         _with(_scores(), cv_predictions=[None] * 3),
         [0.1, 0.2, 0.30000000000000004], None),
        ("a partial join counts only the rows that matched",
         _scores().iloc[:2], [0.1, 0.2, None], [0, 1, None]),
    ]
    for case_id, results, scores, classes in joined:
        with subtests.test(msg=case_id):
            frame, matched = attach_predictions(objects, results)
            assert frame is not objects
            assert matched == sum(v is not None for v in scores)
            expected = list(objects.columns) + [CV_SCORE_COLUMN] + (
                [] if classes is None else [CV_CLASS_COLUMN])
            assert list(frame.columns) == expected
            got = [None if pd.isna(v) else v for v in frame[CV_SCORE_COLUMN]]
            assert got == scores
            if classes is not None:
                got = [None if pd.isna(v) else v
                       for v in frame[CV_CLASS_COLUMN]]
                assert got == classes
            assert CV_SCORE_COLUMN not in objects.columns
