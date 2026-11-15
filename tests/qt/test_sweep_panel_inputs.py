"""Which well every measurement and score lands on in ``sweep_inputs``.

:func:`spacr.qt.widgets.sweep_panel.sweep_inputs` has several ways in: the
measurements may already carry ``prc`` or need it composed, and the score
table may be keyed by ``prc``, by plate/row/column, by nothing at all, or be
superseded by a score column in the measurements.  A wrong choice is silent --
the join matches the wrong wells or nothing -- so every case asserts the exact
wells and values that come out.

One general test walking a case table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.sweep_panel import sweep_inputs  # noqa: E402

pytestmark = pytest.mark.qt

WELLS = ["plate1_r1_c1", "plate1_r1_c2", "plate2_r3_c4"]


def _cells():
    """Measurements whose ``prc`` disagrees with its own plate/row/column."""
    return pd.DataFrame({
        "prc": ["pplate1_r1_c1", "pplate1_r1_c2", "plate2_r3_c4"],
        "plateID": ["plateX"] * 3,
        "rowID": ["r9"] * 3,
        "columnID": ["c9"] * 3,
        "area": [1.0, 2.0, 3.0],
    })


def _counts():
    """Guide fractions keyed the way the count CSVs key them."""
    return pd.DataFrame([
        {"prc": "plate1_r1_c1", "grna": "A", "fraction": 0.5},
        {"prc": "plate1_r1_c2", "grna": "A", "fraction": 0.25},
        {"prc": "plate2_r3_c4", "grna": "B", "fraction": 1.0},
    ])


def test_sweep_inputs_joins_measurements_counts_and_scores_by_well(subtests):
    """Own ``prc`` wins and is un-doubled; scores join by key or not at all."""
    with subtests.test(msg="a frame with prc keeps its own wells"):
        wells, fractions, plates, found = sweep_inputs(_cells(), _counts())
        assert list(wells.index) == WELLS
        assert list(wells["area"]) == [1.0, 2.0, 3.0]
        assert list(plates) == ["plate1", "plate1", "plate2"]
        assert fractions.loc["plate1_r1_c1", "A"] == pytest.approx(0.5)
        assert fractions.loc["plate1_r1_c2", "A"] == pytest.approx(0.25)
        assert fractions.loc["plate2_r3_c4", "B"] == pytest.approx(1.0)
        assert found is None

    with subtests.test(msg="without prc the key is composed"):
        composed, _f, plates, _s = sweep_inputs(
            _cells().drop(columns=["prc"]), _counts())
        assert list(composed.index) == ["plateX_r9_c9"]
        assert list(plates) == ["plateX"]
        assert composed.loc["plateX_r9_c9", "area"] == pytest.approx(2.0)

    cases = [
        # id, measurements, scores, expected score series (None: refused)
        ("scores keyed by prc are joined on it", _cells(),
         pd.DataFrame({"prc": ["pplate1_r1_c1", "pplate1_r1_c2",
                               "plate2_r3_c4"],
                       "plateID": ["plateX"] * 3, "rowID": ["r9"] * 3,
                       "columnID": ["c9"] * 3, "pred": [0.1, 0.2, 0.3]}),
         [0.1, 0.2, 0.3]),
        ("scores keyed by plate/row/column get their key built", _cells(),
         pd.DataFrame({"plateID": ["pplate1", "pplate1", "plate2"],
                       "rowID": ["r1", "r1", "r3"],
                       "columnID": ["c1", "c2", "c4"],
                       "pred": [0.4, 0.5, 0.6]}),
         [0.4, 0.5, 0.6]),
        ("a score table without the score column gives no scores", _cells(),
         pd.DataFrame({"prc": ["pplate1_r1_c1", "pplate1_r1_c2",
                               "plate2_r3_c4"],
                       "recall": [0.9, 0.8, 0.7]}),
         None),
        ("the same table under the score name gives scores", _cells(),
         pd.DataFrame({"prc": ["pplate1_r1_c1", "pplate1_r1_c2",
                               "plate2_r3_c4"],
                       "pred": [0.9, 0.8, 0.7]}),
         [0.9, 0.8, 0.7]),
        ("scores with no well key are refused", _cells(),
         pd.DataFrame({"pred": [0.1, 0.2, 0.3]}), None),
        ("a score column in the measurements beats the CSV",
         _cells().assign(pred=[0.11, 0.22, 0.33]),
         pd.DataFrame({"prc": WELLS, "pred": [9.0, 9.0, 9.0]}),
         [0.11, 0.22, 0.33]),
    ]
    for case_id, cells, scores, expected in cases:
        with subtests.test(msg=case_id):
            wells, _f, _p, found = sweep_inputs(cells, _counts(),
                                                scores=scores)
            if expected is None:
                assert found is None
            else:
                assert list(found.index) == list(wells.index) == WELLS
                assert list(found) == pytest.approx(expected)
