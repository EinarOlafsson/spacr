"""The five settings proposed in instruction 237 (P3-P7), now built.

Each was written up as a proposal because the module could not do the thing
at all, and each is a setting whose DEFAULT is what every existing run
already does -- so nothing changes for a run that does not ask.

P3 ``fields``            re-run one field without redoing the plate
P4 ``*_max_size``        an upper bound, not only a lower one
P5 ``barcode_mismatches`` a budget, instead of exact-match-or-nothing
P6 ``holdout_plate``     train without a plate and score on it
P7 settings.json         the run's settings, with their types intact
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# P3: fields
# ---------------------------------------------------------------------------

NAMES = ["plate1_E01_1_1.npy", "plate1_E01_9_1.npy",
         "plate1_E01_10_1.npy", "plate1_E01_11_1.npy"]


def _fields_of(kept):
    return [n.split("_")[2] for n in kept]


@pytest.mark.parametrize("asked,expected", [
    (None, ["1", "9", "10", "11"]),
    ([], ["1", "9", "10", "11"]),
    ("f9", ["9"]),
    ([9], ["9"]),
    ("F009", ["9"]),
    (["f1", "f10"], ["1", "10"]),
    ("f1,f9", ["1", "9"]),
    ("f1*", ["1", "10", "11"]),
])
def test_a_run_can_be_limited_to_named_fields(asked, expected):
    """Every spelling of a field id spaCR accepts, plus a glob."""
    from spacr.io import select_fields

    assert _fields_of(select_fields(NAMES, asked)) == expected


def test_a_field_that_is_not_there_selects_nothing():
    from spacr.io import select_fields

    assert select_fields(NAMES, "f99") == []


def test_the_default_processes_every_field():
    """The setting must be invisible to a run that does not set it."""
    from spacr.io import select_fields

    assert select_fields(NAMES, None) == NAMES


# ---------------------------------------------------------------------------
# P4: an upper size bound
# ---------------------------------------------------------------------------

def _three_objects():
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[0:2, 0:2] = 1        # 4 px   -- debris
    mask[5:9, 5:9] = 2        # 16 px  -- a real object
    mask[10:20, 10:20] = 3    # 100 px -- a segmentation blow-up
    return mask


def _labels(mask):
    return sorted(int(v) for v in np.unique(mask) if v)


def test_a_minimum_alone_keeps_the_blow_up():
    """Which is the reason the maximum exists."""
    from spacr.utils import _filter_object

    assert _labels(_filter_object(_three_objects(), 10)) == [2, 3]


def test_a_maximum_removes_it():
    from spacr.utils import _filter_object

    assert _labels(_filter_object(_three_objects(), 10, max_value=50)) == [2]


def test_no_bounds_removes_nothing():
    from spacr.utils import _filter_object

    assert _labels(_filter_object(_three_objects(), 0)) == [1, 2, 3]


def test_the_background_is_never_an_object():
    """Label 0 is not a small object to be filtered away."""
    from spacr.utils import _filter_object

    mask = _filter_object(_three_objects(), 10, max_value=50)
    assert (mask == 0).any(), "the background was removed"


# ---------------------------------------------------------------------------
# P5: a mismatch budget
# ---------------------------------------------------------------------------

@pytest.fixture
def library(tmp_path):
    path = tmp_path / "lib.csv"
    pd.DataFrame({"name": ["g1", "g2"],
                  "sequence": ["AAAACCCCGGGGTTTT",
                               "TTTTGGGGCCCCAAAA"]}).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("budget,expected", [
    (0, [None, None]),
    (1, ["g1", None]),
    (2, ["g1", "g1"]),
])
def test_the_budget_is_what_decides(library, budget, expected):
    from spacr.sequencing import map_sequences_to_names

    reads = ["AAAACCCCGGGGTTTA",     # one base wrong
             "AAAACCCCGGGGTTAA"]     # two bases wrong
    got = map_sequences_to_names(library, reads, rc=False, mismatches=budget)
    assert [None if pd.isna(v) else v for v in got] == expected


def test_an_exact_read_matches_at_every_budget(library):
    from spacr.sequencing import map_sequences_to_names

    for budget in (0, 1, 2):
        got = map_sequences_to_names(library, ["AAAACCCCGGGGTTTT"],
                                     rc=False, mismatches=budget)
        assert got[0] == "g1"


def test_a_read_within_reach_of_two_barcodes_is_unassigned(tmp_path):
    """Giving it to whichever was found first would misattribute a count."""
    from spacr.sequencing import map_sequences_to_names

    path = tmp_path / "close.csv"
    pd.DataFrame({"name": ["g1", "g2"],
                  "sequence": ["AAAACCCC", "AAAACCCG"]}).to_csv(path,
                                                                index=False)
    got = map_sequences_to_names(str(path), ["AAAACCCA"], rc=False,
                                 mismatches=1)
    assert pd.isna(got[0])


# ---------------------------------------------------------------------------
# P6: a held-out plate
# ---------------------------------------------------------------------------

@pytest.fixture
def three_plates():
    groups = np.array(["plate1"] * 20 + ["plate2"] * 20 + ["plate3"] * 20)
    labels = np.array(([0] * 10 + [1] * 10) * 3)
    return groups, labels


def test_the_named_plate_is_the_whole_test_side(three_plates):
    from spacr.classifier_evaluation import grouped_split

    groups, labels = three_plates
    train, test, report = grouped_split(groups, labels, 0.2, group_by="plate",
                                        hold_out_groups=["plate3"])

    assert set(groups[test]) == {"plate3"}
    assert set(groups[train]) == {"plate1", "plate2"}
    assert "plate3" in report.rule


def test_holding_out_a_plate_that_is_not_there_is_refused(three_plates):
    from spacr.classifier_evaluation import grouped_split

    groups, labels = three_plates
    with pytest.raises(ValueError, match="appear in the data"):
        grouped_split(groups, labels, 0.2, group_by="plate",
                      hold_out_groups=["plate9"])


def test_holding_out_everything_is_refused(three_plates):
    from spacr.classifier_evaluation import grouped_split

    groups, labels = three_plates
    with pytest.raises(ValueError, match="nothing left to train"):
        grouped_split(groups, labels, 0.2, group_by="plate",
                      hold_out_groups=["plate1", "plate2", "plate3"])


def test_a_holdout_missing_a_class_is_refused():
    """A score on one class does not mean what it says."""
    from spacr.classifier_evaluation import grouped_split

    groups = np.array(["p1"] * 10 + ["p2"] * 10)
    labels = np.array([0] * 10 + [1] * 10)      # p2 is entirely class 1

    with pytest.raises(ValueError, match="without class"):
        grouped_split(groups, labels, 0.2, group_by="plate",
                      hold_out_groups=["p2"])


def test_without_a_holdout_the_split_is_unchanged(three_plates):
    """The default must behave exactly as it did."""
    from spacr.classifier_evaluation import grouped_split

    groups, labels = three_plates
    train, test, _r = grouped_split(groups, labels, 0.34, seed=0,
                                    group_by="plate")

    assert len(train) and len(test)
    assert not set(groups[train]) & set(groups[test])


# ---------------------------------------------------------------------------
# P7: settings.json
# ---------------------------------------------------------------------------

def test_the_run_records_its_settings_with_their_types(tmp_path):
    """The CSV makes everything text; the JSON keeps the shape."""
    from spacr.utils import save_settings

    settings = {"src": str(tmp_path), "channels": [0, 1, 2, 3],
                "png_size": [[224, 224]], "holdout_plate": None,
                "plot": False, "cell_min_size": 8000}
    save_settings(settings, name="demo")

    written = json.load(open(tmp_path / "settings" / "demo.json"))
    assert written["channels"] == [0, 1, 2, 3]
    assert written["png_size"] == [[224, 224]]
    assert written["holdout_plate"] is None
    assert written["plot"] is False
    assert written["cell_min_size"] == 8000


def test_the_csv_is_still_written(tmp_path):
    """Every existing loader reads it; the JSON is a sibling, not a swap."""
    from spacr.utils import save_settings

    save_settings({"src": str(tmp_path), "plot": True}, name="demo")

    assert os.path.exists(tmp_path / "settings" / "demo.csv")
    assert os.path.exists(tmp_path / "settings" / "demo.json")


def test_a_value_json_cannot_hold_does_not_lose_the_file(tmp_path):
    """A settings copy that cannot be written is a note, not a lost run."""
    from spacr.utils import save_settings

    class Odd:
        def __repr__(self):
            return "<an object>"

    save_settings({"src": str(tmp_path), "thing": Odd()}, name="demo")

    written = json.load(open(tmp_path / "settings" / "demo.json"))
    assert written["thing"] == "<an object>"
