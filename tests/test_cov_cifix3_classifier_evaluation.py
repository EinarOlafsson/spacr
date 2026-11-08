"""Held behaviour for plate-map well identities and empty treatment matches in spacr.classifier_evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.classifier_evaluation import (
    _frame_well_keys,
    _write_prediction_scores,
)


def test_plate_map_without_a_well_identity_is_refused_with_its_columns():
    frame = pd.DataFrame({"plateID": ["p1"], "compound": ["x"]})
    with pytest.raises(ValueError) as caught:
        _frame_well_keys(frame, "well_metadata_path map.csv")
    message = str(caught.value)
    assert message.startswith("well_metadata_path map.csv has no well identity")
    assert "['compound', 'plateID']" in message


def test_plate_map_with_well_ids_gives_well_keys():
    frame = pd.DataFrame({"plateID": ["p1", "p1"], "wellID": ["A01", "B02"]})
    assert _frame_well_keys(frame, "map") == ["p1_r1_c1", "p1_r2_c2"]


def test_per_treatment_table_is_skipped_when_no_object_has_a_treatment(tmp_path, capsys):
    plate_map = tmp_path / "map.csv"
    pd.DataFrame({"plate": ["p9"], "well": ["A01"], "compound": ["x"]}).to_csv(
        plate_map, index=False)
    out = _write_prediction_scores(
        tmp_path / "scores", np.array([[0.8, 0.2], [0.3, 0.7]]), np.array([0, 1]),
        ["p1_r1_c1", "p1_r1_c2"], classes=["nc", "pc"],
        treatment_setting="compound", metadata_path=str(plate_map))
    assert out["per_treatment"] is None
    assert out["per_well"].endswith("scores_per_well.csv")
    assert "2 of 2 objects have no compound" in capsys.readouterr().out
    assert not (tmp_path / "scores_per_treatment.csv").exists()
