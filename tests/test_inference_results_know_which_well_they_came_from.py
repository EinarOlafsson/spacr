"""An inference run scored 553 crops and could not say where any of them was.

Driving classify's ``apply_model_to_dataset`` on a real plate printed this
once per crop --

    Error processing filename: plate1_E01_18_1_250.png
    Error: cannot identify a field from 'plate1_E01_18_1_250': expected
    plate_well_field (3 parts, or that plus the timepoint spacr.io names
    every stack with), got 5.

-- and then reported success. Every row of the results CSV came out with
``plateID``, ``rowID``, ``columnID`` and ``fieldID`` set to ``'error'`` and
``prc`` set to ``'error_error_error'``.

Which means the scores could not be joined back to a well: no per-well
aggregate, no regression on a CV model's output, nothing downstream at all.
The only sign was a screenful of errors the run scrolled past.

``process_vision_results`` called ``_map_wells``, which parses a FIELD stem
-- three parts, or four with the timepoint spacr.io writes. Its input is a
CROP name, which is ``plate_well_field_time_object`` and therefore five.
``_map_wells_png`` has always sat directly beneath it and parses exactly
that shape.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.utils import process_vision_results


CROPS = ["plate1_E01_18_1_250.png", "plate1_E01_9_1_517.png",
         "/some/where/plate1_E01_1_1_106.png"]


@pytest.fixture
def scored():
    return process_vision_results(
        pd.DataFrame({"path": CROPS, "pred": [0.9, 0.2, 0.6]}), threshold=0.5)


def test_no_row_comes_back_as_an_error(scored):
    for column in ("plateID", "rowID", "columnID", "fieldID"):
        assert not (scored[column] == "error").any(), (
            f"{column} could not be parsed from a crop name")
    assert not scored["prc"].str.contains("error").any()


def test_the_identity_is_the_one_the_object_tables_carry(scored):
    """E01 is row 5, column 1 -- what `png_list` and `cell` hold."""
    assert list(scored["plateID"]) == ["plate1"] * 3
    assert list(scored["rowID"]) == ["r5"] * 3
    assert list(scored["columnID"]) == ["c1"] * 3
    assert list(scored["fieldID"]) == ["f18", "f9", "f1"]


def test_the_object_id_is_the_last_component(scored):
    """Not the fourth: on a timelapse crop the fourth is the timepoint."""
    assert list(scored["object"]) == ["250", "517", "106"]


def test_a_full_path_parses_the_same_as_a_bare_name(scored):
    """A directory must not leak into plateID."""
    assert scored.iloc[2]["plateID"] == "plate1"
    assert scored.iloc[2]["fieldID"] == "f1"


def test_the_well_key_is_composed_not_pasted(scored):
    """`compose_prc_column` escapes a plate id that contains the separator."""
    assert set(scored["prc"]) == {"plate1_r5_c1"}


def test_the_threshold_still_binarises(scored):
    assert list(scored["cv_predictions"]) == [1, 0, 1]


def test_a_timelapse_crop_keeps_its_field(scored):
    """plate_well_field_time_object: the field is the third part, not the fourth."""
    out = process_vision_results(
        pd.DataFrame({"path": ["plate1_E01_7_3_42.png"], "pred": [0.8]}))
    assert out.iloc[0]["fieldID"] == "f7"
    assert out.iloc[0]["object"] == "42"


def test_the_field_parser_is_not_used_on_a_crop_name():
    """The next person to touch this must not reach for _map_wells again."""
    import inspect

    from spacr import utils

    body = inspect.getsource(utils.process_vision_results)
    assert "_map_wells_png(x)" in body
    assert "_map_wells(x)" not in body.replace("_map_wells_png(x)", "")
