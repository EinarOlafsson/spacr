"""An `ml_analysis` score CSV joins to `png_list` like a CV one does.

Reported from a real screen: four plates of XGBoost scores read as "No
per-object classification score was found in the attached databases", about
files that held one score per cell. Three separate reasons, each enough on
its own:

  1. the score column is `prediction_probability_class_1`, not `pred`;
  2. the file carries no `path`, so the key had to be rebuilt from the
     plate metadata -- which `_db_keys` could do and `_result_keys` could
     not;
  3. the rebuilt key skipped `_clean_prcfo`, so `pplate1_...` met
     `plate1_...` and nothing matched.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.predictions import (
    SCORE_SOURCE_COLUMNS, attach_predictions, first_present,
)


def _png_list(n=6):
    """`png_list` as spaCR writes it, plate id in the doubled-prefix form."""
    return pd.DataFrame({
        "plateID": ["pplate1"] * n,
        "rowID": [f"r{i + 1}" for i in range(n)],
        "columnID": ["c3"] * n,
        "fieldID": ["f1"] * n,
        "cell_id": [f"o{i + 1}" for i in range(n)],
        "prcfo": [f"pplate1_r{i + 1}_c3_f1_o{i + 1}" for i in range(n)],
        "png_path": [f"/plate1/cell/r{i + 1}.png" for i in range(n)],
    })


def _xgboost_scores(n=6):
    """`ml_analysis` output: plain spellings, no path, probability column."""
    return pd.DataFrame({
        "plate": ["pplate1"] * n,
        "row": [f"r{i + 1}" for i in range(n)],
        "column_name": ["c3"] * n,
        "field": ["f1"] * n,
        "object": [f"o{i + 1}" for i in range(n)],
        "prediction_probability_class_1": np.linspace(0.1, 0.9, n),
    })


def test_the_xgboost_score_column_is_recognised():
    assert first_present(_xgboost_scores(),
                         SCORE_SOURCE_COLUMNS) == \
        "prediction_probability_class_1"


def test_pred_still_wins_when_both_are_present():
    """An explicit `pred` is the first choice, so a CV file is unchanged."""
    frame = _xgboost_scores()
    frame["pred"] = 0.5
    assert first_present(frame, SCORE_SOURCE_COLUMNS) == "pred"


def test_every_row_joins(caplog):
    """The whole bug in one number: this was 0."""
    joined, matched = attach_predictions(_png_list(), _xgboost_scores())
    assert matched == 6
    assert "pred" in joined.columns
    assert joined["pred"].notna().all()


def test_the_scores_land_on_the_right_objects():
    """Matching the right COUNT is not matching the right cells."""
    joined, _matched = attach_predictions(_png_list(), _xgboost_scores())
    by_object = dict(zip(joined["cell_id"], joined["pred"]))
    assert by_object["o1"] == pytest.approx(0.1)
    assert by_object["o6"] == pytest.approx(0.9)


def test_the_doubled_plate_prefix_does_not_break_the_join():
    """`pplate1` in the database against `plate1` in the scores."""
    scores = _xgboost_scores()
    scores["plate"] = "plate1"
    _joined, matched = attach_predictions(_png_list(), scores)
    assert matched == 6


def test_a_cv_score_file_still_joins_by_path():
    """The route that already worked must keep working."""
    cv = pd.DataFrame({
        "path": [f"/plate1/cell/r{i + 1}.png" for i in range(6)],
        "pred": np.linspace(0.2, 0.8, 6),
        "cv_predictions": [0, 1, 0, 1, 0, 1],
    })
    joined, matched = attach_predictions(_png_list(), cv)
    assert matched == 6
    assert joined["pred"].notna().all()


def test_a_frame_with_no_score_at_all_matches_nothing():
    """Refusing needs a real number behind it, not an exception."""
    scores = _xgboost_scores().drop(columns=["prediction_probability_class_1"])
    joined, matched = attach_predictions(_png_list(), scores)
    assert matched == 0
    assert "pred" not in joined.columns


def test_scores_for_another_experiment_match_nothing():
    """The join must stay strict: a different plate is not a match."""
    scores = _xgboost_scores()
    scores["plate"] = "pplate9"
    _joined, matched = attach_predictions(_png_list(), scores)
    assert matched == 0


def test_first_present_survives_a_frame_with_no_columns():
    assert first_present(None, SCORE_SOURCE_COLUMNS) is None
    assert first_present(pd.DataFrame(), SCORE_SOURCE_COLUMNS) is None


def test_the_rebuilt_key_matches_the_stored_one():
    """The two builders must agree by construction, not by luck."""
    from spacr.predictions import _db_keys, _result_keys

    png = _png_list()
    stored = _db_keys("prcfo", png)
    rebuilt = _db_keys("prcfo", png.drop(columns=["prcfo"]))
    assert list(stored) == list(rebuilt)


def test_a_partial_row_gets_no_key_rather_than_a_colliding_one():
    """An empty component would make two different objects share a key."""
    from spacr.predictions import _prcfo_from_metadata

    png = _png_list(3)
    png.loc[1, "fieldID"] = None
    keys = _prcfo_from_metadata(png.drop(columns=["prcfo"]))
    assert keys[1] is None
    assert keys[0] is not None
