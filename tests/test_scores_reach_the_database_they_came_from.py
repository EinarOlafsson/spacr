"""An ML run scored 45,777 real cells and wrote none of them back.

Driving classify's ML family on the tsg101 screen fitted, scored, explained
with SHAP and plotted, then finished with:

    Merged predictions, ml_pred into png_list on 'prcfo': 0/60816 rows matched
    !! NOTHING MATCHED. 45777 result row(s) and 60816 png_list row(s) share
    no 'prcfo' value ... The results probably come from a different
    experiment than png_list.

It was the same database it had just read.

THE PLATE IS THE HALF THAT DISAGREES. The screen on disk stamps its plate
``pplate1``; everything computed since stamps it ``plate1``.
``schema.canonical_plate_id`` is the one rule that collapses the doubled
prefix, and ``PLATE_BEARING_COLUMNS`` includes ``prcfo`` -- but
``normalise_plate_columns`` has two callers and neither is on this path.

Both sides of the join are normalised now, and only for the ``prcfo`` key: a
png_path or a file_name is not a plate id and must not have its first two
characters rewritten.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.predictions import _clean_key, _clean_prcfo


@pytest.mark.parametrize("stored,computed", [
    ("pplate1_r8_c19_f11_o84", "plate1_r8_c19_f11_o84"),
    ("plate1_r8_c19_f11_o84", "plate1_r8_c19_f11_o84"),
    ("pplate1_r1_c1_f1_t2_o7", "plate1_r1_c1_f1_t2_o7"),
])
def test_the_two_spellings_of_a_plate_meet(stored, computed):
    assert _clean_prcfo(stored) == _clean_prcfo(computed)


def test_a_key_that_is_not_a_plate_is_left_alone():
    """Only the prcfo key is normalised."""
    assert _clean_key("/nas/pp_data/x.png") == "/nas/pp_data/x.png"
    assert _clean_key("pp_thing.png") == "pp_thing.png"


@pytest.mark.parametrize("value", [None, "", "   ", float("nan")])
def test_an_absent_key_stays_absent(value):
    assert _clean_prcfo(value) is None


def test_the_join_matches_when_the_plates_are_spelled_differently():
    """The shape of the failure: one side written, the other computed."""
    from spacr.predictions import _db_keys, _result_keys

    on_disk = pd.DataFrame({"prcfo": ["pplate1_r8_c19_f11_o84",
                                      "pplate1_r8_c19_f11_o85"]})
    scored = pd.DataFrame({"prcfo": ["plate1_r8_c19_f11_o84"],
                           "ml_pred": [0.8]})

    db_keys = set(_db_keys("prcfo", on_disk).dropna())
    result_keys = set(_result_keys("prcfo", scored, False).dropna())

    assert result_keys & db_keys, "the scores still meet no row"
    assert len(result_keys & db_keys) == 1


def test_a_genuinely_different_plate_still_does_not_match():
    """Normalising may not become matching anything to anything."""
    from spacr.predictions import _db_keys, _result_keys

    on_disk = pd.DataFrame({"prcfo": ["pplate1_r8_c19_f11_o84"]})
    scored = pd.DataFrame({"prcfo": ["plate9_r8_c19_f11_o84"],
                           "ml_pred": [0.8]})

    db_keys = set(_db_keys("prcfo", on_disk).dropna())
    result_keys = set(_result_keys("prcfo", scored, False).dropna())

    assert not (result_keys & db_keys)
