"""Branch coverage for the plate-heatmap plotters in :mod:`spacr.plot`.

Covers :func:`spacr.plot.generate_plate_heatmap` (prc parsing, metadata
back-fill, min_count filtering, every ``min_max`` colour-scale mode and the
degenerate/empty guards), :func:`spacr.plot.plot_plates` (grid layout, unused
axis removal, PDF export) and :func:`spacr.plot.print_mask_and_flows`
(down-sampling, contour overlay, flow panel).

Everything is CPU-only, offline and rendered on the Agg backend.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_figure_leak():
    """Never let Agg figures accumulate between tests."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def show_recorder(monkeypatch):
    """Replace ``plt.show`` with a call counter (spacr.plot uses ``plt.show``)."""
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    return calls


def _prc_df(prc, values):
    """Fresh long-format frame — generate_plate_heatmap mutates its input."""
    return pd.DataFrame({"prc": list(prc), "value": np.asarray(values, dtype=float)})


# ---------------------------------------------------------------------------
# generate_plate_heatmap — min_count sanitising and filtering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_min_count", [None, "two", object()])
def test_generate_plate_heatmap_non_numeric_min_count_is_treated_as_zero(bad_min_count):
    """A non-numeric ``min_count`` is coerced to 0, so nothing is filtered."""
    from spacr.plot import generate_plate_heatmap

    prc = ["p1_r1_c1"] * 3 + ["p1_r2_c1"] + ["p1_r3_c1"] * 2
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df(prc, [1, 2, 3, 100, 5, 6]), "p1", "value", "count",
        "all", bad_min_count,
    )
    # The singleton well r2 survives -> all three rows present, count 1 kept.
    assert plate_map.index.tolist() == ["r1", "r2", "r3"]
    assert plate_map.loc["r2", "c1"] == 1.0
    assert (vmin, vmax) == (1.0, 3.0)


def test_generate_plate_heatmap_min_count_drops_sparse_wells():
    """``min_count`` removes wells whose true row count is below the threshold."""
    from spacr.plot import generate_plate_heatmap

    prc = ["p1_r1_c1"] * 3 + ["p1_r2_c1"] + ["p1_r3_c1"] * 2
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df(prc, [1, 2, 3, 100, 5, 6]), "p1", "value", "count",
        "all", 2,
    )
    assert plate_map.index.tolist() == ["r1", "r3"]
    assert "r2" not in plate_map.index
    assert plate_map.loc["r1", "c1"] == 3.0
    assert plate_map.loc["r3", "c1"] == 2.0
    assert (vmin, vmax) == (2.0, 3.0)


def test_generate_plate_heatmap_min_count_filtering_everything_returns_empty_map():
    """Every well filtered out -> empty pivot and the neutral [0, 1] colour range."""
    from spacr.plot import generate_plate_heatmap

    prc = [f"p1_r{i % 4 + 1}_c{i % 6 + 1}" for i in range(12)]
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df(prc, np.arange(12)), "p1", "value", "count", "all", 999,
    )
    assert plate_map.values.size == 0
    assert (vmin, vmax) == (0.0, 1.0)


def test_generate_plate_heatmap_unknown_plate_returns_empty_map():
    """Selecting a plate that is not in the frame yields an empty heatmap."""
    from spacr.plot import generate_plate_heatmap

    plate_map, limits = generate_plate_heatmap(
        _prc_df(["p1_r1_c1", "p1_r2_c2"], [1.0, 2.0]),
        "plate_that_does_not_exist", "value", "mean", "all", 0,
    )
    assert plate_map.empty
    assert plate_map.values.size == 0
    assert limits == (0.0, 1.0)


# ---------------------------------------------------------------------------
# generate_plate_heatmap — 4-part prc identifiers
# ---------------------------------------------------------------------------

def test_generate_plate_heatmap_four_part_prc_is_rebuilt_with_plate_number():
    """A 4-token prc (``exp_plate_row_col``) is rewritten as ``<plate_number>_row_col``."""
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(
        ["exp1_plateA_r1_c1", "exp1_plateA_r1_c1", "exp1_plateA_r2_c3"],
        [2.0, 4.0, 9.0],
    )
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        df, "p9", "value", "mean", "all", 0,
    )
    # Rows/columns come from tokens 3 and 4; the plate token is replaced.
    assert plate_map.index.tolist() == ["r1", "r2"]
    assert plate_map.columns.tolist() == ["c1", "c3"]
    assert plate_map.loc["r1", "c1"] == 3.0        # mean(2, 4)
    assert plate_map.loc["r2", "c3"] == 9.0
    assert plate_map.loc["r1", "c3"] == 0.0        # missing well -> filled with 0
    assert (vmin, vmax) == (0.0, 9.0)


def test_generate_plate_heatmap_four_part_prc_does_not_mutate_caller_prc():
    """The 4-part branch copies the frame, so the caller's ``prc`` is untouched."""
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["exp1_plateA_r1_c1", "exp1_plateA_r2_c2"], [1.0, 2.0])
    before = df["prc"].tolist()
    generate_plate_heatmap(df, "p9", "value", "mean", "all", 0)
    assert df["prc"].tolist() == before


# ---------------------------------------------------------------------------
# generate_plate_heatmap — metadata back-fill from legacy column names
# ---------------------------------------------------------------------------

def test_generate_plate_heatmap_backfills_columnid_from_column():
    """A legacy ``column`` column seeds ``columnID`` before prc parsing."""
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["p1_r1_c1", "p1_r1_c1", "p1_r2_c2"], [1.0, 3.0, 7.0])
    df["column"] = ["c1", "c1", "c2"]
    assert "column_name" not in df.columns

    plate_map, (vmin, vmax) = generate_plate_heatmap(df, "p1", "value", "mean", "all", 0)
    # The seeded columnID is superseded by the prc tokens (prc is the truth).
    assert df["columnID"].tolist() == ["c1", "c1", "c2"]
    assert plate_map.loc["r1", "c1"] == 2.0
    assert plate_map.loc["r2", "c2"] == 7.0
    assert (vmin, vmax) == (0.0, 7.0)


def test_generate_plate_heatmap_backfills_plateid_from_plate():
    """A legacy ``plate`` column seeds ``plateID`` when ``plateID`` is absent."""
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["p1_r1_c1", "p1_r2_c2"], [4.0, 6.0])
    df["plate"] = "p1"
    assert "plateID" not in df.columns

    plate_map, _ = generate_plate_heatmap(df, "p1", "value", "mean", "all", 0)
    assert df["plateID"].tolist() == ["p1", "p1"]
    assert plate_map.loc["r1", "c1"] == 4.0
    assert plate_map.loc["r2", "c2"] == 6.0


def test_generate_plate_heatmap_backfills_plateid_from_plate_name():
    """``plate_name`` is the second fallback for ``plateID``."""
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["p1_r1_c1", "p1_r2_c2"], [4.0, 6.0])
    df["plate_name"] = "ignored_by_prc_parsing"
    assert "plateID" not in df.columns and "plate" not in df.columns

    plate_map, _ = generate_plate_heatmap(df, "p1", "value", "sum", "all", 0)
    # prc parsing overwrites the seeded plateID, so the plate filter still works.
    assert df["plateID"].tolist() == ["p1", "p1"]
    assert plate_map.loc["r1", "c1"] == 4.0


def test_generate_plate_heatmap_default_plateid_when_no_metadata_columns():
    """With no plate metadata at all the seed is 'p1', then prc parsing wins."""
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["pX_r1_c1", "pX_r2_c2"], [1.0, 5.0])
    plate_map, _ = generate_plate_heatmap(df, "pX", "value", "mean", "all", 0)
    assert df["plateID"].tolist() == ["pX", "pX"]
    assert plate_map.loc["r2", "c2"] == 5.0


# ---------------------------------------------------------------------------
# generate_plate_heatmap — aggregation modes
# ---------------------------------------------------------------------------

def test_generate_plate_heatmap_sum_grouping_adds_values_per_well():
    """``grouping='sum'`` totals the variable inside each well."""
    from spacr.plot import generate_plate_heatmap

    prc = ["p1_r1_c1"] * 3 + ["p1_r2_c1"] + ["p1_r3_c1"] * 2
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df(prc, [1, 2, 3, 100, 5, 6]), "p1", "value", "sum", "all", 0,
    )
    assert plate_map.loc["r1", "c1"] == 6.0     # 1 + 2 + 3
    assert plate_map.loc["r2", "c1"] == 100.0
    assert plate_map.loc["r3", "c1"] == 11.0    # 5 + 6
    assert (vmin, vmax) == (6.0, 100.0)


def test_generate_plate_heatmap_sum_coerces_non_numeric_values_to_nan():
    """Non-numeric cells are coerced with ``errors='coerce'`` and sum as 0."""
    from spacr.plot import generate_plate_heatmap

    df = pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r2_c1"],
        "value": ["1.5", "not_a_number", "4"],
    })
    plate_map, _ = generate_plate_heatmap(df, "p1", "value", "sum", "all", 0)
    assert plate_map.loc["r1", "c1"] == 1.5
    assert plate_map.loc["r2", "c1"] == 4.0


def test_generate_plate_heatmap_sum_missing_variable_raises_keyerror():
    from spacr.plot import generate_plate_heatmap

    with pytest.raises(KeyError):
        generate_plate_heatmap(
            _prc_df(["p1_r1_c1"], [1.0]), "p1", "absent", "sum", "all", 0,
        )


# ---------------------------------------------------------------------------
# generate_plate_heatmap — min_max colour-scale modes
# ---------------------------------------------------------------------------

def _sum_map_df():
    prc = ["p1_r1_c1"] * 3 + ["p1_r2_c1"] + ["p1_r3_c1"] * 2
    return _prc_df(prc, [1, 2, 3, 100, 5, 6])


def test_generate_plate_heatmap_min_max_int_pair_is_used_verbatim():
    """An explicit non-float [vmin, vmax] pair is taken literally."""
    from spacr.plot import generate_plate_heatmap

    _, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", [2, 10], 0,
    )
    assert (vmin, vmax) == (2.0, 10.0)
    assert isinstance(vmin, float) and isinstance(vmax, float)


def test_generate_plate_heatmap_min_max_tuple_pair_is_accepted():
    """Tuples work as well as lists."""
    from spacr.plot import generate_plate_heatmap

    _, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", (0, 42), 0,
    )
    assert (vmin, vmax) == (0.0, 42.0)


def test_generate_plate_heatmap_min_max_float_pair_is_treated_as_quantiles():
    """A pair of floats selects quantiles of the plate map rather than limits."""
    from spacr.plot import generate_plate_heatmap

    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", [0.1, 0.9], 0,
    )
    expected = np.quantile(plate_map.values, [0.1, 0.9])
    assert vmin == pytest.approx(expected[0])
    assert vmax == pytest.approx(expected[1])
    # Quantiles must sit strictly inside the raw data range.
    assert vmin > plate_map.values.min() and vmax < plate_map.values.max()


def test_generate_plate_heatmap_min_max_allq_clips_to_2_98_quantiles():
    """``'allq'`` clamps the colour range to the 2nd/98th percentiles."""
    from spacr.plot import generate_plate_heatmap

    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", "allq", 0,
    )
    expected = np.quantile(plate_map.values, [0.02, 0.98])
    assert vmin == pytest.approx(expected[0])
    assert vmax == pytest.approx(expected[1])
    # The 100.0 outlier well must no longer set the top of the scale.
    assert vmax < plate_map.values.max()


def test_generate_plate_heatmap_unknown_min_max_falls_back_to_full_range():
    """An unrecognised ``min_max`` spec falls back to nanmin/nanmax."""
    from spacr.plot import generate_plate_heatmap

    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", "auto", 0,
    )
    assert vmin == float(np.nanmin(plate_map.values))
    assert vmax == float(np.nanmax(plate_map.values))
    assert (vmin, vmax) == (6.0, 100.0)


def test_generate_plate_heatmap_three_element_min_max_falls_back_to_full_range():
    """A malformed 3-element spec also lands in the fallback branch."""
    from spacr.plot import generate_plate_heatmap

    _, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", [1, 2, 3], 0,
    )
    assert (vmin, vmax) == (6.0, 100.0)


def test_generate_plate_heatmap_degenerate_range_is_nudged():
    """A constant plate map would give vmin == vmax, so vmax is bumped."""
    from spacr.plot import generate_plate_heatmap

    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df([f"p1_r{i % 3 + 1}_c1" for i in range(6)], [0.0] * 6),
        "p1", "value", "sum", "all", 0,
    )
    assert plate_map.values.min() == plate_map.values.max() == 0.0
    assert vmin == 0.0
    assert vmax == pytest.approx(1e-6)
    assert vmax > vmin


def test_generate_plate_heatmap_rejects_unknown_grouping():
    from spacr.plot import generate_plate_heatmap

    with pytest.raises(ValueError, match="grouping must be"):
        generate_plate_heatmap(
            _prc_df(["p1_r1_c1"], [1.0]), "p1", "value", "median", "all", 0,
        )


# ---------------------------------------------------------------------------
# plot_plates
# ---------------------------------------------------------------------------

def test_plot_plates_lays_out_one_axis_per_plate_and_saves_pdf(tmp_path, show_recorder):
    """Five plates -> 2x4 grid, three unused axes deleted, PDF written to dst."""
    from spacr.plot import plot_plates

    prc = []
    values = []
    for p in range(1, 6):
        for r in range(1, 4):
            prc.append(f"p{p}_r{r}_c{r}")
            values.append(float(p * r))
    df = _prc_df(prc, values)

    fig = plot_plates(df, "value", "mean", "all", "viridis",
                      min_count=0, verbose=False, dst=str(tmp_path))

    # 5 heatmap axes + 5 seaborn colorbar axes; the 3 spare grid slots are gone.
    heat_axes = [a for a in fig.axes if a.get_title()]
    assert sorted(a.get_title() for a in heat_axes) == ["p1", "p2", "p3", "p4", "p5"]
    assert os.path.isfile(tmp_path / "plate_heatmap_0.pdf")
    assert (tmp_path / "plate_heatmap_0.pdf").stat().st_size > 0
    assert show_recorder == []          # verbose=False must not call plt.show


def test_plot_plates_autoincrements_output_filename(tmp_path, show_recorder):
    """A second export next to an existing file gets the next free index."""
    from spacr.plot import plot_plates

    (tmp_path / "plate_heatmap_0.pdf").write_bytes(b"placeholder")
    df = _prc_df(["p1_r1_c1", "p1_r2_c2"], [1.0, 2.0])
    plot_plates(df, "value", "mean", "all", "viridis",
                min_count=0, verbose=False, dst=str(tmp_path))

    assert (tmp_path / "plate_heatmap_1.pdf").is_file()
    assert (tmp_path / "plate_heatmap_0.pdf").read_bytes() == b"placeholder"


def test_plot_plates_verbose_calls_show_and_skips_saving(tmp_path, show_recorder):
    from spacr.plot import plot_plates

    df = _prc_df(["p1_r1_c1", "p2_r2_c2"], [1.0, 2.0])
    fig = plot_plates(df, "value", "count", "all", "viridis",
                      min_count=0, verbose=True, dst=None)

    assert show_recorder == [1]
    assert list(tmp_path.iterdir()) == []
    assert fig is plt.figure(fig.number)


# ---------------------------------------------------------------------------
# print_mask_and_flows
# ---------------------------------------------------------------------------

def _blob_image(shape=(24, 18), value=0.9):
    img = np.zeros(shape, dtype=np.float32)
    img[6:14, 5:12] = value
    return img


def _blob_mask(shape=(24, 18)):
    mask = np.zeros(shape, dtype=np.int32)
    mask[6:14, 5:12] = 1
    return mask


def _only_figure():
    nums = plt.get_fignums()
    assert len(nums) == 1, f"expected exactly one figure, got {nums}"
    return plt.figure(nums[0])


def test_print_mask_and_flows_without_flows_uses_two_panels(show_recorder):
    """``flows=None`` -> 2-panel figure; a 2D stack is displayed as-is."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image()
    mask = _blob_mask()
    assert print_mask_and_flows(stack, mask, None, overlay=True) is None

    fig = _only_figure()
    assert len(fig.axes) == 2
    assert [a.get_title() for a in fig.axes] == ["Original Image", "Mask with Overlay"]
    shown = np.asarray(fig.axes[0].images[0].get_array())
    assert shown.shape == stack.shape
    assert np.allclose(shown, stack)
    assert show_recorder == [1]


def test_print_mask_and_flows_overlay_draws_red_contours(show_recorder):
    """The overlay panel is an RGB uint8 image carrying pure-red contour pixels."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image()
    mask = _blob_mask()
    print_mask_and_flows(stack, mask, None, overlay=True, thickness=1)

    fig = _only_figure()
    overlaid = np.asarray(fig.axes[1].images[0].get_array())
    assert overlaid.shape == stack.shape + (3,)
    assert overlaid.dtype == np.uint8
    red = (overlaid[..., 0] == 255) & (overlaid[..., 1] == 0) & (overlaid[..., 2] == 0)
    assert red.any(), "no red contour pixels were drawn"
    # Contours hug the blob, so red pixels live near the mask border only.
    assert red.sum() < mask.sum()


def test_print_mask_and_flows_without_overlay_shows_raw_mask(show_recorder):
    """``overlay=False`` shows the label mask itself in the second panel."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image()
    mask = _blob_mask()
    mask[16:20, 2:6] = 2
    print_mask_and_flows(stack, mask, None, overlay=False)

    fig = _only_figure()
    assert fig.axes[1].get_title() == "Mask"
    shown = np.asarray(fig.axes[1].images[0].get_array())
    assert shown.shape == mask.shape
    assert np.array_equal(shown, mask)


def test_print_mask_and_flows_squeezes_singleton_channel(show_recorder):
    """An ``(H, W, 1)`` stack is squeezed to 2D before display."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image()[..., None]
    print_mask_and_flows(stack, _blob_mask(), None, overlay=True)

    fig = _only_figure()
    shown = np.asarray(fig.axes[0].images[0].get_array())
    assert shown.ndim == 2
    assert shown.shape == stack.shape[:2]


def test_print_mask_and_flows_multichannel_stack_uses_first_channel(show_recorder):
    """A 3-channel stack falls back to channel 0 for the 'Original Image' panel."""
    from spacr.plot import print_mask_and_flows

    stack = np.zeros((24, 18, 3), dtype=np.float32)
    stack[..., 0] = _blob_image(value=0.8)
    stack[..., 1] = 0.25
    stack[..., 2] = 0.5
    print_mask_and_flows(stack, _blob_mask(), None, overlay=False)

    fig = _only_figure()
    shown = np.asarray(fig.axes[0].images[0].get_array())
    assert shown.shape == (24, 18)
    assert np.allclose(shown, stack[..., 0])


def test_print_mask_and_flows_downsamples_large_inputs(show_recorder):
    """Anything larger than ``max_size`` is resized, channels preserved."""
    from spacr.plot import print_mask_and_flows

    stack = np.zeros((24, 12, 3), dtype=np.float32)
    stack[..., 0] = _blob_image(shape=(24, 12), value=0.7)
    mask = _blob_mask(shape=(24, 12))
    flows = [np.linspace(0, 1, 24 * 12, dtype=np.float32).reshape(24, 12)]

    print_mask_and_flows(stack, mask, flows, overlay=True, max_size=8)

    fig = _only_figure()
    assert len(fig.axes) == 3
    # scale = 8/24 -> spatial shape (8, 4); the channel axis is preserved by
    # resize_if_needed, so channel 0 of the resized stack is what gets shown.
    from skimage.transform import resize as sk_resize
    expected_stack = sk_resize(stack, (8, 4, 3), preserve_range=True,
                               anti_aliasing=True).astype(stack.dtype)
    shown = np.asarray(fig.axes[0].images[0].get_array())
    assert shown.shape == (8, 4)
    assert np.allclose(shown, expected_stack[..., 0])

    expected_flow = sk_resize(flows[0], (8, 4), preserve_range=True,
                              anti_aliasing=True).astype(flows[0].dtype)
    shown_flow = np.asarray(fig.axes[2].images[0].get_array())
    assert shown_flow.shape == (8, 4)
    assert np.allclose(shown_flow, expected_flow)
    assert fig.axes[2].get_title() == "Flows"


def test_print_mask_and_flows_small_input_is_not_resized(show_recorder):
    """Below ``max_size`` the arrays are passed through untouched."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image(shape=(20, 16))
    flows = [np.zeros((20, 16), dtype=np.float32)]
    print_mask_and_flows(stack, _blob_mask(shape=(20, 16)), flows,
                         overlay=False, max_size=1000)

    fig = _only_figure()
    assert np.asarray(fig.axes[0].images[0].get_array()).shape == (20, 16)
    assert np.asarray(fig.axes[2].images[0].get_array()).shape == (20, 16)


def test_print_mask_and_flows_three_dimensional_flow_uses_first_channel(show_recorder):
    """A (H, W, C) flow array is reduced to its first channel for display."""
    from spacr.plot import print_mask_and_flows

    flow = np.zeros((20, 16, 3), dtype=np.float32)
    flow[..., 0] = 0.75
    flow[..., 1] = -1.0
    print_mask_and_flows(_blob_image((20, 16)), _blob_mask((20, 16)), [flow],
                         overlay=True)

    fig = _only_figure()
    shown = np.asarray(fig.axes[2].images[0].get_array())
    assert shown.shape == (20, 16)
    assert np.allclose(shown, 0.75)


def test_print_mask_and_flows_empty_flow_list_raises(show_recorder):
    """An empty flow list is structurally invalid and must raise."""
    from spacr.plot import print_mask_and_flows

    with pytest.raises(ValueError, match="flow dimensionality"):
        print_mask_and_flows(_blob_image((20, 16)), _blob_mask((20, 16)), [],
                             overlay=False)
    assert show_recorder == []


def test_print_mask_and_flows_rejects_four_dimensional_stack(show_recorder):
    """A 4D stack is not displayable."""
    from spacr.plot import print_mask_and_flows

    stack = np.zeros((4, 20, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="stack dimensionality"):
        print_mask_and_flows(stack, _blob_mask((20, 16)), None, overlay=False)
    assert show_recorder == []
