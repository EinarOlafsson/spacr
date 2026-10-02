"""A plate name with underscores validates, and crop dilation ratios are optional when dilation is off."""
import pandas as pd

from spacr import schema
from spacr.measure import _per_crop_mode


def test_prcf_of_a_plate_with_underscores_validates():
    plate = "AssayPlate_PerkinElmer_CellCarrier-384"
    prcf = schema.compose_prcf(plate, "r10", "c7", "f3")
    frame = pd.DataFrame({
        schema.PLATE_KEY: [plate], schema.ROW_KEY: ["r10"],
        schema.COLUMN_KEY: ["c7"], schema.FIELD_KEY: ["f3"],
        schema.PRCF_KEY: [prcf]})
    expected = (frame[schema.PLATE_KEY].map(schema.escape_filename_component)
                + "_r10_c7_f3")
    assert frame[schema.PRCF_KEY].eq(expected).all()


def test_empty_ratios_are_still_refused_when_asked_for():
    try:
        _per_crop_mode([], 1, "dialate_png_ratios")
    except ValueError:
        return
    raise AssertionError("an empty ratio list must be refused when used")
