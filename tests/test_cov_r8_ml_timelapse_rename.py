"""The timepoint rename in `interpret_vision_model`, and why it is live.

`read_and_preprocess_data` joins a scores CSV onto the measurements
database, and the timepoint is part of the join key -- without it every
frame's object matches every frame's score and the frame is multiplied
by the number of frames.

The two sides can spell the timepoint differently. A scores file is read
through `tabular.read_table`, whose vocabulary rewrites `time_id` to the
canonical `timeID`, so the scores side is always canonical. The DATABASE
side is not guaranteed to be: `spacr.io`'s own note says
`rename_columns_in_db` repairs an old database in place, but "a database
opened read-only, or one carrying both spellings, still reads correctly
here". A read-only legacy database therefore arrives spelled `time_id`
while its scores say `timeID`, and the rename is what makes them join.

That is the branch, and it is the one case in this area that is NOT dead
-- the neighbouring row/col alias assignments are, for the opposite
reason, and `tests/test_cov_r6_ml.py` records why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import ml
from tests.test_predictions_merge import vision_settings


def _legacy_measurements(time_column: str) -> pd.DataFrame:
    """One plate, one well, two frames, two objects -- spelled as asked."""
    rows = []
    for frame in (1, 2):
        for obj in (1, 2):
            rows.append({
                "plateID": "plate1", "rowID": "r1", "columnID": "c1",
                "fieldID": "f1", "object_label": f"o{obj}",
                time_column: frame,
                "cell_area": 100.0 * frame + obj,
                "cell_channel_0_mean_intensity": 10.0 * frame + obj,
            })
    return pd.DataFrame(rows)


def _scores(tmp_path, time_column: str):
    """A canonical scores file: one score per object per frame."""
    rows = []
    for frame in (1, 2):
        for obj in (1, 2):
            rows.append({
                "plateID": "plate1", "rowID": "r1", "columnID": "c1",
                "fieldID": "f1", "object": str(obj),
                time_column: frame,
                "cv_predictions": (frame + obj) % 2,
            })
    path = tmp_path / f"scores_{time_column}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture()
def legacy_db(monkeypatch):
    """Make the DB side answer with whichever spelling a test asks for."""
    def install(time_column):
        import spacr.io as io_mod

        frame = _legacy_measurements(time_column)

        def fake_read(*_a, **_k):
            return frame.copy(), None

        monkeypatch.setattr(io_mod, "_read_and_merge_data", fake_read)
        return frame
    return install


def case_a_legacy_database_and_canonical_scores_still_join(tmp_path,
                                                           legacy_db):
    """THE UNCOVERED BRANCH: the two spellings differ, so one is renamed.

    Four rows out, not eight: the timepoint really is in the join key.
    Sixteen would mean every frame matched every frame.
    """
    _seen = []
    legacy_db("time_id")
    scores = _scores(tmp_path, "timeID")
    merged = ml.interpret_vision_model(
        vision_settings(str(tmp_path / "src"), scores))
    _seen.append(keep(len(merged)))
    _seen.append(bool(
        set(merged["cv_predictions"]) <= {0, 1}))
    return _seen


def case_a_canonical_database_needs_no_rename(tmp_path, legacy_db):
    """The other side of the same branch, so the rename is visibly a rename."""
    _seen = []
    legacy_db("timeID")
    scores = _scores(tmp_path, "timeID")
    merged = ml.interpret_vision_model(
        vision_settings(str(tmp_path / "src"), scores))
    _seen.append(keep(len(merged)))
    return _seen


def case_a_timelapse_database_and_flat_scores_are_refused(tmp_path,
                                                          legacy_db):
    """Only one side carrying a timepoint is a mismatch, not a join.

    Joining without it would match every frame's object to every frame's
    score, so the code raises rather than returning a silently multiplied
    frame -- a wrong answer that looks like a right one.
    """
    _seen = []
    legacy_db("timeID")
    rows = [{"plateID": "plate1", "rowID": "r1", "columnID": "c1",
             "fieldID": "f1", "object": str(obj), "cv_predictions": obj % 2}
            for obj in (1, 2)]
    path = tmp_path / "flat_scores.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(Exception) as caught:
        ml.interpret_vision_model(
            vision_settings(str(tmp_path / "src"), path))
    _seen.append(('Exception' if isinstance(caught.value, Exception) else type(caught.value).__name__))
    _seen.append(('timepoint' if 'timepoint' in (str(caught.value).lower()) else ("missing", 'timepoint')))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_a_legacy_database_and_canonical_scores_still_join, [4, True], id="a_legacy_database_and_canonical_scores_still_join"),
    pytest.param(case_a_canonical_database_needs_no_rename, [4], id="a_canonical_database_needs_no_rename"),
    pytest.param(case_a_timelapse_database_and_flat_scores_are_refused, ['Exception', 'timepoint'], id="a_timelapse_database_and_flat_scores_are_refused"),
])
def test_r8_ml_timelapse_rename_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
