"""Branch coverage for the crop / object-settings / recruitment helpers in ``spacr.utils``.

Covers the defensive and rarely-taken paths of:

    _get_percentiles       - the "no non-zero pixels" fallback
    _crop_center           - label masks, mask-outside zeroing, empty mask
    _masks_to_masks_stack  - order/identity preservation
    _get_object_settings   - the unsupported-object-type branch ('cell_large')
    _get_cellpose_channels - the organelle channel
    annotate_conditions    - loc values that are neither 'r*' nor 'c*'
    _split_data            - missing well columns, all-object frames, all-numeric frames
    _calculate_recruitment - the full set of ratio columns

Everything here is CPU-only, offline and sub-second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Never let a figure leak out of a test (nothing here plots, but be safe)."""
    yield
    import sys
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")


# ---------------------------------------------------------------------------
# _get_percentiles
# ---------------------------------------------------------------------------

def test_get_percentiles_ignores_zero_pixels_on_populated_channel():
    """Percentiles are computed over the > 0 pixels only."""
    from spacr.utils import _get_percentiles

    arr = np.zeros((8, 8, 1), dtype=np.float32)
    arr[:, :, 0] = np.arange(64, dtype=np.float32).reshape(8, 8)

    (lo, hi), = _get_percentiles(arr, p1=2, p2=98)

    non_zero = np.arange(1, 64, dtype=np.float32)
    assert lo == pytest.approx(float(np.percentile(non_zero, 2)))
    assert hi == pytest.approx(float(np.percentile(non_zero, 98)))
    # The zero pixel must not have dragged the low percentile down to 0.
    assert lo > 0.0


def test_get_percentiles_all_zero_channel_falls_back_to_raw_image():
    """A channel with no positive pixels uses the raw image -> [0, 0]."""
    from spacr.utils import _get_percentiles

    arr = np.zeros((8, 8, 2), dtype=np.float32)
    arr[:, :, 0] = np.arange(64, dtype=np.float32).reshape(8, 8)
    # channel 1 stays all-zero -> exercises the `non_zero_img.size == 0` branch

    out = _get_percentiles(arr, p1=2, p2=98)

    assert len(out) == 2
    assert out[1] == [0.0, 0.0]
    assert out[0][0] > 0.0 and out[0][1] > out[0][0]


def test_get_percentiles_all_negative_channel_uses_raw_percentiles():
    """Negative pixels are not `> 0`, so the fallback reports the real range."""
    from spacr.utils import _get_percentiles

    arr = np.empty((4, 4, 1), dtype=np.float64)
    arr[:, :, 0] = np.linspace(-10.0, -1.0, 16).reshape(4, 4)

    (lo, hi), = _get_percentiles(arr, p1=0, p2=100)

    assert lo == pytest.approx(-10.0)
    assert hi == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# _crop_center
# ---------------------------------------------------------------------------

def test_crop_center_zeroes_outside_label_mask_and_centers_on_centroid():
    """A non-binary label mask is honoured; pixels outside it come back as 0."""
    from spacr.utils import _crop_center

    img = np.full((40, 40, 2), 9, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[16:24, 16:24] = 7  # a single label, centroid at (19.5, 19.5)

    out = _crop_center(img, mask, new_width=12, new_height=12)

    assert out.shape == (12, 12, 2)
    assert out.dtype == np.uint8
    # 8x8 masked pixels survive, everything else in the 12x12 window is zeroed.
    assert int((out[:, :, 0] == 9).sum()) == 64
    assert int((out[:, :, 0] == 0).sum()) == 12 * 12 - 64
    # And the surviving block is centred: it must not touch the crop border.
    ys, xs = np.nonzero(out[:, :, 0])
    assert ys.min() > 0 and ys.max() < 11
    assert xs.min() > 0 and xs.max() < 11


def test_crop_center_empty_mask_returns_zero_crop_of_requested_size():
    """An empty mask zeroes the whole image, so any crop is all zeros."""
    from spacr.utils import _crop_center

    img = np.full((20, 20, 3), 7, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)

    with np.errstate(invalid="ignore"):
        out = _crop_center(img, mask, new_width=8, new_height=6)

    assert out.shape == (6, 8, 3)
    assert out.dtype == np.uint8
    assert not out.any()


def test_crop_center_larger_than_image_pads_with_zeros():
    """Requesting a window bigger than the object pads rather than clipping."""
    from spacr.utils import _crop_center

    img = np.full((16, 16, 1), 5, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[6:10, 6:10] = 1

    out = _crop_center(img, mask, new_width=32, new_height=32)

    assert out.shape == (32, 32, 1)
    assert int((out == 5).sum()) == 16  # only the 4x4 masked block is non-zero


# ---------------------------------------------------------------------------
# _masks_to_masks_stack
# ---------------------------------------------------------------------------

def test_masks_to_masks_stack_preserves_order_and_object_identity():
    from spacr.utils import _masks_to_masks_stack

    masks = [np.zeros((2, 2), dtype=np.int32) + i for i in range(3)]
    out = _masks_to_masks_stack(masks)

    assert isinstance(out, list)
    assert len(out) == 3
    # The helper must not copy: the same array objects come back, in order.
    assert all(out[i] is masks[i] for i in range(3))


def test_masks_to_masks_stack_consumes_generators():
    from spacr.utils import _masks_to_masks_stack

    out = _masks_to_masks_stack(np.zeros((3, 2, 2), dtype=np.int32))
    assert len(out) == 3
    assert all(m.shape == (2, 2) for m in out)


# ---------------------------------------------------------------------------
# _get_object_settings — the unsupported object-type branch
# ---------------------------------------------------------------------------

def test_get_object_settings_cell_large_reports_unsupported(capsys):
    """'cell_large' has a diameter but no settings branch -> warning, bare dict."""
    from spacr.utils import _get_object_settings

    out = _get_object_settings("cell_large", {"magnification": 20, "verbose": False})

    printed = capsys.readouterr().out
    assert "cell_large not supported" in printed

    assert out["diameter"] == 2 * 20 + 120
    assert out["minimum_size"] == (160 ** 2) / 4
    assert out["maximum_size"] == (160 ** 2) * 10
    assert out["model_name"] == "cpsam"      # Cellpose 4 ships only cpsam
    assert out["resample"] is True
    assert out["merge"] is False
    # The per-object keys are only set inside the supported branches.
    assert "filter_size" not in out
    assert "filter_intensity" not in out
    assert "restore_type" not in out


def test_get_object_settings_unsupported_type_raises_before_settings():
    """A type that isn't even a known diameter fails loudly in _get_diam."""
    from spacr.utils import _get_object_settings

    with pytest.raises(ValueError, match="unsupported object type"):
        _get_object_settings("mitochondria", {"magnification": 20, "verbose": False})


# ---------------------------------------------------------------------------
# _get_cellpose_channels — organelle
# ---------------------------------------------------------------------------

def test_get_cellpose_channels_remaps_organelle_channel():
    from spacr.utils import _get_cellpose_channels

    settings = {
        "cellpose_nucleus_channel": 2,
        "cellpose_cell_channel": 5,
        "cellpose_pathogen_channel": 1,
        "cellpose_organelle_channel": 7,
    }
    extract, channels = _get_cellpose_channels(settings)

    assert extract == [1, 2, 5, 7]
    assert channels == {
        "nucleus": [1],
        "cell": [2, 1],      # cell first, nucleus second
        "pathogen": [0],
        "organelle": [3],
    }


def test_get_cellpose_channels_organelle_only():
    """Organelle alone still gets remapped to index 0 of the extracted stack."""
    from spacr.utils import _get_cellpose_channels

    extract, channels = _get_cellpose_channels({"cellpose_organelle_channel": 4})

    assert extract == [4]
    assert channels == {"organelle": [0]}


def test_get_cellpose_channels_deduplicates_shared_channel():
    """The same source channel used twice is extracted once."""
    from spacr.utils import _get_cellpose_channels

    extract, channels = _get_cellpose_channels(
        {"cellpose_cell_channel": 3, "cellpose_organelle_channel": 3}
    )

    assert extract == [3]
    assert channels == {"cell": [0], "organelle": [0]}


# ---------------------------------------------------------------------------
# annotate_conditions — loc values that map to neither rowID nor columnID
# ---------------------------------------------------------------------------

def _cond_df():
    return pd.DataFrame({"rowID": ["r1", "r2", "r3"], "columnID": ["c1", "c2", "c3"]})


def test_annotate_conditions_skips_loc_values_without_row_or_column_prefix():
    """'x1' is neither a row nor a column id, so nothing is assigned for it."""
    from spacr.utils import annotate_conditions

    out = annotate_conditions(
        _cond_df(),
        cells=["HeLa", "Vero"],
        cell_loc=[["x1"], ["r2"]],
    )

    assert pd.isna(out.loc[0, "host_cells"])   # 'x1' was silently ignored
    assert out.loc[1, "host_cells"] == "Vero"
    assert pd.isna(out.loc[2, "host_cells"])
    # condition is built only from the non-NaN annotations
    assert out.loc[1, "condition"] == "Vero"
    assert pd.isna(out.loc[0, "condition"])
    assert pd.isna(out.loc[2, "condition"])


def test_annotate_conditions_skips_non_string_loc_values():
    """Non-string loc entries hit the same `return None` guard."""
    from spacr.utils import annotate_conditions

    out = annotate_conditions(
        _cond_df(),
        pathogens=["wt", "ko"],
        pathogen_loc=[[1], ["c3"]],
    )

    assert out["pathogen"].isna().tolist() == [True, True, False]
    assert out.loc[2, "pathogen"] == "ko"
    assert out.loc[2, "condition"] == "ko"


def test_annotate_conditions_unmatched_loc_leaves_condition_na():
    """No loc value matches -> every condition is NA, not an empty string."""
    from spacr.utils import annotate_conditions

    out = annotate_conditions(
        _cond_df(),
        treatments=["drug"],
        treatment_loc=[["zz"]],
    )

    assert out["treatment"].isna().all()
    assert out["condition"].isna().all()
    assert (out["condition"] == "").sum() == 0


# ---------------------------------------------------------------------------
# _split_data
# ---------------------------------------------------------------------------

def test_split_data_reports_missing_well_columns_and_reuses_existing_prcf(capsys):
    """Both prcft and prcf construction fail; an existing prcf column is reused."""
    from spacr.utils import _split_data

    df = pd.DataFrame({
        "prcf": ["p1_r1_c1_f1"] * 2 + ["p1_r2_c1_f1"] * 2,
        "plateID": ["p1"] * 4,
        "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1"] * 4,          # no fieldID and no timeID
        "object_label": [1, 2, 3, 4],
        "cell_area": [100.0, 200.0, 300.0, 400.0],
    })

    numeric, non_numeric = _split_data(df, "prcf", "object_label")

    printed = capsys.readouterr().out
    # One Exception line, from the prcf attempt tripping over the missing
    # fieldID. The prcft attempt no longer prints anything: it now asks whether
    # a timepoint column (timeID or the legacy time_id) is present rather than
    # hard-coding 'timeID' inside a bare try/except, and this frame has none —
    # so there is nothing to build and nothing to report. That distinction is
    # the point of the change: "not a timelapse frame" and "the metadata is
    # broken" used to print the same line.
    assert printed.splitlines() == ["Exception 'fieldID'"]

    assert "prcft" not in numeric.columns and "prcft" not in non_numeric.columns
    assert numeric.index.tolist() == ["p1_r1_c1_f1", "p1_r2_c1_f1"]
    # area is summed; object_label is CARRIED, not averaged. It used to be
    # averaged -- objects 1 and 2 rolled up to 1.5, a label for an object
    # that does not exist, sitting in a numeric column indistinguishable from
    # a measurement. `aggregation_for` matches identity ahead of every other
    # rule now, so the label arrives verbatim and the count is what says how
    # many objects went into the row.
    assert numeric.loc["p1_r1_c1_f1", "cell_area"] == 300.0
    assert numeric.loc["p1_r2_c1_f1", "cell_area"] == 700.0
    assert numeric.loc["p1_r1_c1_f1", "object_label"] == 1
    assert numeric.loc["p1_r2_c1_f1", "object_label"] == 3
    assert non_numeric.loc["p1_r1_c1_f1", "prcfo"] == "p1_r1_c1_f1_1"
    # the caller's frame must not be mutated
    assert "prcfo" not in df.columns


def test_split_data_all_object_columns_gives_empty_numeric_frame(capsys):
    """No numeric columns -> an empty numeric frame indexed by the group keys."""
    from spacr.utils import _split_data

    df = pd.DataFrame({
        "plateID": ["p1"] * 4,
        "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1"] * 4,
        "fieldID": ["f1"] * 4,
        "object_label": ["1", "2", "3", "4"],
        "note": ["a", "b", "c", "d"],
    })

    numeric, non_numeric = _split_data(df, "prcf", "object_label")

    # Nothing is printed: prcf builds fine, and the absence of a timepoint
    # column is now a silent "no prcft" rather than a printed exception.
    assert capsys.readouterr().out.count("Exception") == 0
    assert isinstance(numeric, pd.DataFrame)
    assert numeric.shape == (2, 0)
    assert sorted(numeric.index.tolist()) == ["p1_r1_c1_f1", "p1_r2_c1_f1"]
    # non-numeric side keeps the first row of every group
    assert non_numeric.loc["p1_r1_c1_f1", "note"] == "a"
    assert non_numeric.loc["p1_r2_c1_f1", "note"] == "c"
    assert non_numeric.loc["p1_r2_c1_f1", "prcfo"] == "p1_r2_c1_f1_3"


def test_split_data_all_numeric_columns_gives_empty_non_numeric_frame():
    """Grouping on both string columns leaves nothing non-numeric behind."""
    from spacr.utils import _split_data

    df = pd.DataFrame({
        "plateID": [1, 1, 1, 1],
        "rowID": [1, 1, 2, 2],
        "columnID": [1, 1, 1, 1],
        "fieldID": [1, 1, 1, 1],
        "object_label": [1, 2, 3, 4],
        "cell_area": [100.0, 200.0, 300.0, 400.0],
        "cell_channel_0_mean_intensity": [10.0, 20.0, 30.0, 40.0],
    })

    numeric, non_numeric = _split_data(df, ["prcf", "prcfo"], "object_label")

    assert isinstance(non_numeric, pd.DataFrame)
    assert non_numeric.shape == (4, 0)
    assert non_numeric.index.tolist() == [
        ("1_1_1_1", "1_1_1_1_1"),
        ("1_1_1_1", "1_1_1_1_2"),
        ("1_2_1_1", "1_2_1_1_3"),
        ("1_2_1_1", "1_2_1_1_4"),
    ]
    # every group holds exactly one object, so sums == means == the raw values
    assert numeric["cell_area"].tolist() == [100.0, 200.0, 300.0, 400.0]
    assert numeric["cell_channel_0_mean_intensity"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_split_data_sums_size_columns_and_means_the_rest():
    """sum_keywords columns are summed per group; everything else is averaged.

    The frame carries ``timeID``, so the grouping key is the timepoint key
    ``p1_r1_c1_f1_<time>_<object>``. This test used to expect
    ``p1_r1_c1_f1_1`` — plate/row/column/field/object with the timepoint
    dropped — and that expectation was the bug: ``_map_wells(timelapse=True)``
    writes the timepoint into ``prcf`` and ``_split_data`` rebuilt it without,
    so every object collapsed across all of its frames. All four rows here
    share one timepoint, which is why the old key still looked plausible.
    """
    from spacr.utils import _split_data

    df = pd.DataFrame({
        "plateID": ["p1"] * 4,
        "rowID": ["r1"] * 4,
        "columnID": ["c1"] * 4,
        "fieldID": ["f1"] * 4,
        "timeID": [1] * 4,
        "object_label": [1, 1, 1, 1],
        "cell_area": [1.0, 2.0, 3.0, 4.0],
        "cell_perimeter": [1.0, 1.0, 1.0, 1.0],
        "cell_equivalent_diameter": [2.0, 2.0, 2.0, 2.0],
        "cell_channel_0_mean_intensity": [10.0, 20.0, 30.0, 40.0],
    })

    numeric, non_numeric = _split_data(df, "prcfo", "object_label")

    assert numeric.shape[0] == 1
    key = "p1_r1_c1_f1_1_1"
    # AREAS SUM, LENGTHS AVERAGE (maintainer's call, 2026-08-11). Four
    # objects rolled onto one parent occupy the sum of their areas -- a real
    # quantity of the parent. They have no combined perimeter or diameter:
    # those describe an INDIVIDUAL object, so the parent gets the typical
    # one. perimeter and equivalent_diameter were summed here and are now
    # averaged, which is why the expected values changed from 4.0 and 8.0.
    assert numeric.loc[key, "cell_area"] == 10.0                 # summed
    assert numeric.loc[key, "cell_perimeter"] == 1.0             # averaged
    assert numeric.loc[key, "cell_equivalent_diameter"] == 2.0   # averaged
    assert numeric.loc[key, "cell_channel_0_mean_intensity"] == 25.0  # averaged
    # timeID was present, so prcft got built and lands on the non-numeric side
    assert non_numeric.loc[key, "prcft"] == "p1_r1_c1_f1_1"
    # prcf now agrees with prcft: same key, from the same components.
    assert non_numeric.loc[key, "prcf"] == "p1_r1_c1_f1_1"


# ---------------------------------------------------------------------------
# _calculate_recruitment
# ---------------------------------------------------------------------------

def _recruitment_df(channel=1):
    """Two-row frame carrying every intensity column _calculate_recruitment reads."""
    return pd.DataFrame({
        f"pathogen_channel_{channel}_mean_intensity": [100.0, 200.0],
        f"cell_channel_{channel}_mean_intensity": [10.0, 50.0],
        f"cytoplasm_channel_{channel}_mean_intensity": [5.0, 25.0],
        f"nucleus_channel_{channel}_mean_intensity": [4.0, 8.0],
        f"pathogen_channel_{channel}_percentile_75": [300.0, 400.0],
        f"pathogen_channel_{channel}_outside_mean": [20.0, 30.0],
        f"pathogen_channel_{channel}_outside_75_percentile": [60.0, 90.0],
        f"pathogen_channel_{channel}_periphery_mean": [8.0, 16.0],
    })


def test_calculate_recruitment_computes_every_ratio_column():
    from spacr.utils import _calculate_recruitment

    df = _recruitment_df(channel=1)
    out = _calculate_recruitment(df, channel=1)

    assert out is df  # annotates in place and hands the same frame back

    assert out["pathogen_cell_mean_mean"].tolist() == [10.0, 4.0]
    assert out["pathogen_cytoplasm_mean_mean"].tolist() == [20.0, 8.0]
    assert out["pathogen_nucleus_mean_mean"].tolist() == [25.0, 25.0]

    assert out["pathogen_cell_q75_mean"].tolist() == [30.0, 8.0]
    assert out["pathogen_cytoplasm_q75_mean"].tolist() == [60.0, 16.0]
    assert out["pathogen_nucleus_q75_mean"].tolist() == [75.0, 50.0]

    assert out["pathogen_outside_cell_mean_mean"].tolist() == [2.0, 0.6]
    assert out["pathogen_outside_cytoplasm_mean_mean"].tolist() == [4.0, 1.2]
    assert out["pathogen_outside_nucleus_mean_mean"].tolist() == [5.0, 3.75]

    assert out["pathogen_outside_cell_q75_mean"].tolist() == [6.0, 1.8]
    assert out["pathogen_outside_cytoplasm_q75_mean"].tolist() == [12.0, 3.6]
    assert out["pathogen_outside_nucleus_q75_mean"].tolist() == [15.0, 11.25]

    assert out["pathogen_periphery_cell_mean_mean"].tolist() == [0.8, 0.32]
    assert out["pathogen_periphery_cytoplasm_mean_mean"].tolist() == [1.6, 0.64]
    assert out["pathogen_periphery_nucleus_mean_mean"].tolist() == [2.0, 2.0]


def test_calculate_recruitment_adds_placeholder_slope_columns():
    from spacr.utils import _calculate_recruitment

    out = _calculate_recruitment(_recruitment_df(channel=0), channel=0)

    for obj in ("pathogen", "nucleus"):
        for chan in (0, 1, 2, 3):
            col = f"{obj}_slope_channel_{chan}"
            assert col in out.columns
            assert out[col].tolist() == [1, 1]
    # cell/cytoplasm slopes are deliberately not emitted
    assert "cell_slope_channel_0" not in out.columns


def test_calculate_recruitment_propagates_zero_denominator_as_inf():
    from spacr.utils import _calculate_recruitment

    df = _recruitment_df(channel=2)
    df["nucleus_channel_2_mean_intensity"] = [0.0, 8.0]

    out = _calculate_recruitment(df, channel=2)

    assert np.isinf(out.loc[0, "pathogen_nucleus_mean_mean"])
    assert out.loc[1, "pathogen_nucleus_mean_mean"] == 25.0


def test_calculate_recruitment_missing_column_raises_key_error():
    """A frame lacking a required intensity column must fail loudly."""
    from spacr.utils import _calculate_recruitment

    df = _recruitment_df(channel=1).drop(columns=["pathogen_channel_1_periphery_mean"])

    with pytest.raises(KeyError, match="periphery_mean"):
        _calculate_recruitment(df, channel=1)


def test_calculate_recruitment_wrong_channel_raises_key_error():
    from spacr.utils import _calculate_recruitment

    with pytest.raises(KeyError):
        _calculate_recruitment(_recruitment_df(channel=1), channel=3)


# ---------------------------------------------------------------------------
# _group_by_well
# ---------------------------------------------------------------------------

def test_group_by_well_means_numeric_and_takes_first_non_numeric():
    from spacr.utils import _group_by_well

    df = pd.DataFrame({
        "plateID": ["p1"] * 4,
        "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1"] * 4,
        "cell_area": [100.0, 300.0, 10.0, 30.0],
        "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r2_c1", "p1_r2_c1"],
    })

    out = _group_by_well(df)

    assert out.index.names == ["plateID", "rowID", "columnID"]
    assert out.loc[("p1", "r1", "c1"), "cell_area"] == 200.0
    assert out.loc[("p1", "r2", "c1"), "cell_area"] == 20.0
    # non-numeric column keeps the first value of the group
    assert out.loc[("p1", "r2", "c1"), "prc"] == "p1_r2_c1"
    assert len(out) == 2
