"""Branch coverage for the plate-heatmap plotters in :mod:`spacr.plot`.

Covers :func:`spacr.plot.generate_plate_heatmap` (prc parsing, metadata
back-fill, min_count filtering, every ``min_max`` colour-scale mode and the
degenerate/empty guards), :func:`spacr.plot.plot_plates` (grid layout, unused
axis removal, PDF export) and :func:`spacr.plot.print_mask_and_flows`
(down-sampling, contour overlay, flow panel).

Everything is CPU-only, offline and rendered on the Agg backend.

Each contract is one parametrised test.  The rows name a helper that does the
setup and call (keeping its secondary checks) and returns what it observed;
the row carries the expected value and the test body compares the two.
"""
from __future__ import annotations

import functools
import inspect
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


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


def _prc_df(prc, values):
    """Fresh long-format frame — generate_plate_heatmap mutates its input."""
    return pd.DataFrame({"prc": list(prc), "value": np.asarray(values, dtype=float)})


# ---------------------------------------------------------------------------
# generate_plate_heatmap — min_count sanitising and filtering
# ---------------------------------------------------------------------------

def _case_generate_plate_heatmap_non_numeric_min_count_is_treated_as_zero(bad_min_count):
    """A non-numeric ``min_count`` is coerced to 0, so nothing is filtered."""
    from spacr.plot import generate_plate_heatmap

    prc = ["p1_r1_c1"] * 3 + ["p1_r2_c1"] + ["p1_r3_c1"] * 2
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df(prc, [1, 2, 3, 100, 5, 6]), "p1", "value", "count",
        "all", bad_min_count,
    )
    # The singleton well r2 survives -> all three rows present, count 1 kept.
    return plate_map.index.tolist(), plate_map.loc["r2", "c1"], (vmin, vmax)


def _case_generate_plate_heatmap_min_count_drops_sparse_wells():
    """``min_count`` removes wells whose true row count is below the threshold."""
    from spacr.plot import generate_plate_heatmap

    prc = ["p1_r1_c1"] * 3 + ["p1_r2_c1"] + ["p1_r3_c1"] * 2
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df(prc, [1, 2, 3, 100, 5, 6]), "p1", "value", "count",
        "all", 2,
    )
    assert "r2" not in plate_map.index
    return (plate_map.index.tolist(), plate_map.loc["r1", "c1"],
            plate_map.loc["r3", "c1"], (vmin, vmax))


def _case_generate_plate_heatmap_min_count_filtering_everything_returns_empty_map():
    """Every well filtered out -> empty pivot and the neutral [0, 1] colour range."""
    from spacr.plot import generate_plate_heatmap

    prc = [f"p1_r{i % 4 + 1}_c{i % 6 + 1}" for i in range(12)]
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df(prc, np.arange(12)), "p1", "value", "count", "all", 999,
    )
    return plate_map.values.size, (vmin, vmax)


def _case_generate_plate_heatmap_unknown_plate_returns_empty_map():
    """Selecting a plate that is not in the frame yields an empty heatmap."""
    from spacr.plot import generate_plate_heatmap

    plate_map, limits = generate_plate_heatmap(
        _prc_df(["p1_r1_c1", "p1_r2_c2"], [1.0, 2.0]),
        "plate_that_does_not_exist", "value", "mean", "all", 0,
    )
    return plate_map.empty, plate_map.values.size, limits


# ---------------------------------------------------------------------------
# generate_plate_heatmap — 4-part prc identifiers
# ---------------------------------------------------------------------------

def _case_generate_plate_heatmap_four_part_prc_is_rebuilt_with_plate_number():
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
    # r1/c1 is mean(2, 4); the missing well r1/c3 is filled with 0.
    return (plate_map.index.tolist(), plate_map.columns.tolist(),
            plate_map.loc["r1", "c1"], plate_map.loc["r2", "c3"],
            plate_map.loc["r1", "c3"], (vmin, vmax))


def _case_generate_plate_heatmap_four_part_prc_does_not_mutate_caller_prc():
    """The 4-part branch copies the frame, so the caller's ``prc`` is untouched."""
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["exp1_plateA_r1_c1", "exp1_plateA_r2_c2"], [1.0, 2.0])
    generate_plate_heatmap(df, "p9", "value", "mean", "all", 0)
    return df["prc"].tolist()


# ---------------------------------------------------------------------------
# generate_plate_heatmap — metadata back-fill from legacy column names
# ---------------------------------------------------------------------------

def _case_generate_plate_heatmap_reads_the_column_from_prc_not_from_a_column_column():
    """A legacy ``column`` column does not get a vote on where a well is drawn.

    Renamed from the "backfills columnID" spelling it had while the function
    carried a disagreeing derivation: it read ``column`` / ``column_name``
    into ``columnID`` and then overwrote that from the prc tokens a few lines
    later, so the derivation could never change an answer. What is worth
    pinning is the property that survived: prc is the only authority on the
    well, so a frame carrying a second spelling of the column is placed
    exactly where the same frame without it is.
    """
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["p1_r1_c1", "p1_r1_c1", "p1_r2_c2"], [1.0, 3.0, 7.0])
    df["column"] = ["c9", "c9", "c9"]

    plate_map, (vmin, vmax) = generate_plate_heatmap(df, "p1", "value", "mean", "all", 0)
    return (df["columnID"].tolist(), plate_map.columns.tolist(),
            plate_map.loc["r1", "c1"], plate_map.loc["r2", "c2"], (vmin, vmax))


def _case_generate_plate_heatmap_takes_the_plate_from_prc_over_a_plate_column():
    """``plateID`` comes from the identifier even when a ``plate`` column exists.

    Renamed from "backfills plateID from plate" for the reason given above:
    the seeding branch it named was overwritten before anything read it. The
    plate filter has to agree with the identifier or a well lands on the
    wrong plate's grid, and that is what is asserted here.
    """
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["p1_r1_c1", "p1_r2_c2"], [4.0, 6.0])
    df["plate"] = "some_other_plate"

    plate_map, _ = generate_plate_heatmap(df, "p1", "value", "mean", "all", 0)
    return (df["plateID"].tolist(), plate_map.loc["r1", "c1"],
            plate_map.loc["r2", "c2"])


def _case_generate_plate_heatmap_ignores_a_plate_name_column_entirely():
    """``plate_name`` names nothing this function reads.

    Renamed from "backfills plateID from plate_name": that fallback was
    overwritten from the prc tokens in every case. A disagreeing
    ``plate_name`` must not move the well, which is the assertion left.
    """
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["p1_r1_c1", "p1_r2_c2"], [4.0, 6.0])
    df["plate_name"] = "ignored_by_prc_parsing"

    plate_map, _ = generate_plate_heatmap(df, "p1", "value", "sum", "all", 0)
    return df["plateID"].tolist(), plate_map.loc["r1", "c1"]


def _case_generate_plate_heatmap_names_the_plate_in_prc_with_no_metadata_columns():
    """With no plate metadata at all the plate is still the one in the prc.

    Renamed from "default plateID when no metadata columns": the ``'p1'``
    default it named was overwritten from the prc token before it was read,
    so a frame of ``pX_`` wells was never labelled ``p1``. That is what is
    checked now.
    """
    from spacr.plot import generate_plate_heatmap

    df = _prc_df(["pX_r1_c1", "pX_r2_c2"], [1.0, 5.0])
    plate_map, _ = generate_plate_heatmap(df, "pX", "value", "mean", "all", 0)
    return df["plateID"].tolist(), plate_map.loc["r2", "c2"]


# ---------------------------------------------------------------------------
# generate_plate_heatmap — aggregation modes
# ---------------------------------------------------------------------------

def _case_generate_plate_heatmap_sum_grouping_adds_values_per_well():
    """``grouping='sum'`` totals the variable inside each well."""
    from spacr.plot import generate_plate_heatmap

    prc = ["p1_r1_c1"] * 3 + ["p1_r2_c1"] + ["p1_r3_c1"] * 2
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df(prc, [1, 2, 3, 100, 5, 6]), "p1", "value", "sum", "all", 0,
    )
    # r1 is 1 + 2 + 3, r3 is 5 + 6.
    return (plate_map.loc["r1", "c1"], plate_map.loc["r2", "c1"],
            plate_map.loc["r3", "c1"], (vmin, vmax))


def _case_generate_plate_heatmap_sum_coerces_non_numeric_values_to_nan():
    """Non-numeric cells are coerced with ``errors='coerce'`` and sum as 0."""
    from spacr.plot import generate_plate_heatmap

    df = pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r2_c1"],
        "value": ["1.5", "not_a_number", "4"],
    })
    plate_map, _ = generate_plate_heatmap(df, "p1", "value", "sum", "all", 0)
    return plate_map.loc["r1", "c1"], plate_map.loc["r2", "c1"]


def _case_generate_plate_heatmap_sum_missing_variable_raises_keyerror():
    from spacr.plot import generate_plate_heatmap

    with pytest.raises(KeyError) as excinfo:
        generate_plate_heatmap(
            _prc_df(["p1_r1_c1"], [1.0]), "p1", "absent", "sum", "all", 0,
        )
    return type(excinfo.value).__name__


# ---------------------------------------------------------------------------
# generate_plate_heatmap — min_max colour-scale modes
# ---------------------------------------------------------------------------

def _sum_map_df():
    prc = ["p1_r1_c1"] * 3 + ["p1_r2_c1"] + ["p1_r3_c1"] * 2
    return _prc_df(prc, [1, 2, 3, 100, 5, 6])


def _case_generate_plate_heatmap_min_max_int_pair_is_used_verbatim():
    """An explicit non-float [vmin, vmax] pair is taken literally."""
    from spacr.plot import generate_plate_heatmap

    _, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", [2, 10], 0,
    )
    assert isinstance(vmin, float) and isinstance(vmax, float)
    return vmin, vmax


def _case_generate_plate_heatmap_min_max_tuple_pair_is_accepted():
    """Tuples work as well as lists."""
    from spacr.plot import generate_plate_heatmap

    _, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", (0, 42), 0,
    )
    return vmin, vmax


def _case_generate_plate_heatmap_min_max_float_pair_is_treated_as_quantiles():
    """A pair of floats selects quantiles of the plate map rather than limits."""
    from spacr.plot import generate_plate_heatmap

    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", [0.1, 0.9], 0,
    )
    # Quantiles must sit strictly inside the raw data range.
    assert vmin > plate_map.values.min() and vmax < plate_map.values.max()
    return vmin, vmax


def _case_generate_plate_heatmap_min_max_allq_clips_to_2_98_quantiles():
    """``'allq'`` clamps the colour range to the 2nd/98th percentiles."""
    from spacr.plot import generate_plate_heatmap

    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", "allq", 0,
    )
    # The 100.0 outlier well must no longer set the top of the scale.
    assert vmax < plate_map.values.max()
    return vmin, vmax


def _case_generate_plate_heatmap_unknown_min_max_falls_back_to_full_range():
    """An unrecognised ``min_max`` spec falls back to nanmin/nanmax."""
    from spacr.plot import generate_plate_heatmap

    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", "auto", 0,
    )
    assert vmin == float(np.nanmin(plate_map.values))
    assert vmax == float(np.nanmax(plate_map.values))
    return vmin, vmax


def _case_generate_plate_heatmap_three_element_min_max_falls_back_to_full_range():
    """A malformed 3-element spec also lands in the fallback branch."""
    from spacr.plot import generate_plate_heatmap

    _, (vmin, vmax) = generate_plate_heatmap(
        _sum_map_df(), "p1", "value", "sum", [1, 2, 3], 0,
    )
    return vmin, vmax


def _case_generate_plate_heatmap_degenerate_range_is_nudged():
    """A constant plate map would give vmin == vmax, so vmax is bumped."""
    from spacr.plot import generate_plate_heatmap

    plate_map, (vmin, vmax) = generate_plate_heatmap(
        _prc_df([f"p1_r{i % 3 + 1}_c1" for i in range(6)], [0.0] * 6),
        "p1", "value", "sum", "all", 0,
    )
    assert plate_map.values.min() == plate_map.values.max() == 0.0
    assert vmax > vmin
    return vmin, vmax


def _case_generate_plate_heatmap_rejects_unknown_grouping():
    from spacr.plot import generate_plate_heatmap

    with pytest.raises(ValueError, match="grouping must be") as excinfo:
        generate_plate_heatmap(
            _prc_df(["p1_r1_c1"], [1.0]), "p1", "value", "median", "all", 0,
        )
    return type(excinfo.value).__name__


# ---------------------------------------------------------------------------
# plot_plates
# ---------------------------------------------------------------------------

def _case_plot_plates_lays_out_one_axis_per_plate_and_saves_pdf(tmp_path, show_recorder):
    """Five plates -> one small multiple, one shared colour bar, one file."""
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

    # 5 plate axes, each an image, plus the ONE colour bar they share.
    heat_axes = [a for a in fig.axes if a.images]
    assert (tmp_path / "plate_heatmap_value.pdf").stat().st_size > 0
    # verbose=False must not call plt.show (show_recorder stays empty).
    return (sorted(a.get_title() for a in heat_axes),
            len([a for a in fig.axes if not a.images]),
            os.path.isfile(tmp_path / "plate_heatmap_value.pdf"),
            show_recorder)


def _case_plot_plates_rewrites_its_own_file_rather_than_numbering_a_new_one(
        tmp_path, show_recorder):
    """The export is named for the measurement it draws, and replaced.

    The old loop took the first free ``plate_heatmap_<n>.pdf`` and never
    overwrote, so the maintainer's screen accumulated twelve byte-identical
    copies of one figure from twelve runs -- and the figure grid showed all
    twelve. A run already gets its own results folder
    (``spacr.ml._next_results_folder``), so one file per measurement in it is
    the whole of what belongs there. The old loop also probed for a ``.pdf``
    that ``save_figure`` may write as ``.png``, in which case it never found
    its own previous output at all.
    """
    from spacr.plot import plot_plates

    df = _prc_df(["p1_r1_c1", "p1_r2_c2"], [1.0, 2.0])
    for _ in range(3):
        plot_plates(df, "value", "mean", "all", "viridis",
                    min_count=0, verbose=False, dst=str(tmp_path))

    assert (tmp_path / "plate_heatmap_value.pdf").stat().st_size > 0
    return [p.name for p in sorted(tmp_path.iterdir())]


def _case_plot_plates_verbose_calls_show_and_skips_saving(tmp_path, show_recorder):
    from spacr.plot import plot_plates

    df = _prc_df(["p1_r1_c1", "p2_r2_c2"], [1.0, 2.0])
    fig = plot_plates(df, "value", "count", "all", "viridis",
                      min_count=0, verbose=True, dst=None)

    return show_recorder, list(tmp_path.iterdir()), fig is plt.figure(fig.number)


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


def _case_print_mask_and_flows_without_flows_uses_two_panels(show_recorder):
    """``flows=None`` -> 2-panel figure; a 2D stack is displayed as-is."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image()
    mask = _blob_mask()
    returned = print_mask_and_flows(stack, mask, None, overlay=True)

    fig = _only_figure()
    shown = np.asarray(fig.axes[0].images[0].get_array())
    assert np.allclose(shown, stack)
    return (returned, len(fig.axes), [a.get_title() for a in fig.axes],
            shown.shape, show_recorder)


def _case_print_mask_and_flows_overlay_draws_red_contours(show_recorder):
    """The overlay panel is an RGB uint8 image carrying pure-red contour pixels."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image()
    mask = _blob_mask()
    print_mask_and_flows(stack, mask, None, overlay=True, thickness=1)

    fig = _only_figure()
    overlaid = np.asarray(fig.axes[1].images[0].get_array())
    red = (overlaid[..., 0] == 255) & (overlaid[..., 1] == 0) & (overlaid[..., 2] == 0)
    # Contours hug the blob, so red pixels live near the mask border only.
    assert red.sum() < mask.sum()
    return overlaid.shape, str(overlaid.dtype), bool(red.any())


def _case_print_mask_and_flows_without_overlay_shows_raw_mask(show_recorder):
    """``overlay=False`` shows the label mask itself in the second panel."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image()
    mask = _blob_mask()
    mask[16:20, 2:6] = 2
    print_mask_and_flows(stack, mask, None, overlay=False)

    fig = _only_figure()
    shown = np.asarray(fig.axes[1].images[0].get_array())
    return (fig.axes[1].get_title(), shown.shape, bool(np.array_equal(shown, mask)))


def _case_print_mask_and_flows_squeezes_singleton_channel(show_recorder):
    """An ``(H, W, 1)`` stack is squeezed to 2D before display."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image()[..., None]
    print_mask_and_flows(stack, _blob_mask(), None, overlay=True)

    fig = _only_figure()
    shown = np.asarray(fig.axes[0].images[0].get_array())
    assert shown.shape == stack.shape[:2]
    return shown.ndim, shown.shape


def _case_print_mask_and_flows_multichannel_stack_uses_first_channel(show_recorder):
    """A 3-channel stack falls back to channel 0 for the 'Original Image' panel."""
    from spacr.plot import print_mask_and_flows

    stack = np.zeros((24, 18, 3), dtype=np.float32)
    stack[..., 0] = _blob_image(value=0.8)
    stack[..., 1] = 0.25
    stack[..., 2] = 0.5
    print_mask_and_flows(stack, _blob_mask(), None, overlay=False)

    fig = _only_figure()
    shown = np.asarray(fig.axes[0].images[0].get_array())
    return shown.shape, bool(np.allclose(shown, stack[..., 0]))


def _case_print_mask_and_flows_downsamples_large_inputs(show_recorder):
    """Anything larger than ``max_size`` is resized, channels preserved."""
    from spacr.plot import print_mask_and_flows

    stack = np.zeros((24, 12, 3), dtype=np.float32)
    stack[..., 0] = _blob_image(shape=(24, 12), value=0.7)
    mask = _blob_mask(shape=(24, 12))
    flows = [np.linspace(0, 1, 24 * 12, dtype=np.float32).reshape(24, 12)]

    print_mask_and_flows(stack, mask, flows, overlay=True, max_size=8)

    fig = _only_figure()
    # scale = 8/24 -> spatial shape (8, 4); the channel axis is preserved by
    # resize_if_needed, so channel 0 of the resized stack is what gets shown.
    from skimage.transform import resize as sk_resize
    expected_stack = sk_resize(stack, (8, 4, 3), preserve_range=True,
                               anti_aliasing=True).astype(stack.dtype)
    shown = np.asarray(fig.axes[0].images[0].get_array())
    assert np.allclose(shown, expected_stack[..., 0])

    expected_flow = sk_resize(flows[0], (8, 4), preserve_range=True,
                              anti_aliasing=True).astype(flows[0].dtype)
    shown_flow = np.asarray(fig.axes[2].images[0].get_array())
    assert np.allclose(shown_flow, expected_flow)
    return (len(fig.axes), shown.shape, shown_flow.shape, fig.axes[2].get_title())


def _case_print_mask_and_flows_small_input_is_not_resized(show_recorder):
    """Below ``max_size`` the arrays are passed through untouched."""
    from spacr.plot import print_mask_and_flows

    stack = _blob_image(shape=(20, 16))
    flows = [np.zeros((20, 16), dtype=np.float32)]
    print_mask_and_flows(stack, _blob_mask(shape=(20, 16)), flows,
                         overlay=False, max_size=1000)

    fig = _only_figure()
    return (np.asarray(fig.axes[0].images[0].get_array()).shape,
            np.asarray(fig.axes[2].images[0].get_array()).shape)


def _case_print_mask_and_flows_three_dimensional_flow_uses_first_channel(show_recorder):
    """A (H, W, C) flow array is reduced to its first channel for display."""
    from spacr.plot import print_mask_and_flows

    flow = np.zeros((20, 16, 3), dtype=np.float32)
    flow[..., 0] = 0.75
    flow[..., 1] = -1.0
    print_mask_and_flows(_blob_image((20, 16)), _blob_mask((20, 16)), [flow],
                         overlay=True)

    fig = _only_figure()
    shown = np.asarray(fig.axes[2].images[0].get_array())
    return shown.shape, bool(np.allclose(shown, 0.75))


def _case_print_mask_and_flows_empty_flow_list_raises(show_recorder):
    """An empty flow list is structurally invalid and must raise."""
    from spacr.plot import print_mask_and_flows

    with pytest.raises(ValueError, match="flow dimensionality") as excinfo:
        print_mask_and_flows(_blob_image((20, 16)), _blob_mask((20, 16)), [],
                             overlay=False)
    return type(excinfo.value).__name__, show_recorder


def _case_print_mask_and_flows_rejects_four_dimensional_stack(show_recorder):
    """A 4D stack is not displayable."""
    from spacr.plot import print_mask_and_flows

    stack = np.zeros((4, 20, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="stack dimensionality") as excinfo:
        print_mask_and_flows(stack, _blob_mask((20, 16)), None, overlay=False)
    return type(excinfo.value).__name__, show_recorder


_P = pytest.param

_HEATMAP_ROWS = [
    _P(functools.partial(_case_generate_plate_heatmap_non_numeric_min_count_is_treated_as_zero, None),
       (["r1", "r2", "r3"], 1.0, (1.0, 3.0)), id="min-count-none-is-zero"),
    _P(functools.partial(_case_generate_plate_heatmap_non_numeric_min_count_is_treated_as_zero, "two"),
       (["r1", "r2", "r3"], 1.0, (1.0, 3.0)), id="min-count-string-is-zero"),
    _P(functools.partial(_case_generate_plate_heatmap_non_numeric_min_count_is_treated_as_zero, object()),
       (["r1", "r2", "r3"], 1.0, (1.0, 3.0)), id="min-count-object-is-zero"),
    _P(_case_generate_plate_heatmap_min_count_drops_sparse_wells,
       (["r1", "r3"], 3.0, 2.0, (2.0, 3.0)), id="min-count-drops-sparse-wells"),
    _P(_case_generate_plate_heatmap_min_count_filtering_everything_returns_empty_map,
       (0, (0.0, 1.0)), id="min-count-filtering-everything-is-empty"),
    _P(_case_generate_plate_heatmap_unknown_plate_returns_empty_map,
       (True, 0, (0.0, 1.0)), id="unknown-plate-is-empty"),
    _P(_case_generate_plate_heatmap_four_part_prc_is_rebuilt_with_plate_number,
       (["r1", "r2"], ["c1", "c3"], 3.0, 9.0, 0.0, (0.0, 9.0)),
       id="four-part-prc-rebuilt"),
    _P(_case_generate_plate_heatmap_four_part_prc_does_not_mutate_caller_prc,
       ["exp1_plateA_r1_c1", "exp1_plateA_r2_c2"], id="four-part-prc-leaves-caller-alone"),
    _P(_case_generate_plate_heatmap_reads_the_column_from_prc_not_from_a_column_column,
       (["c1", "c1", "c2"], ["c1", "c2"], 2.0, 7.0, (0.0, 7.0)),
       id="column-comes-from-prc"),
    _P(_case_generate_plate_heatmap_takes_the_plate_from_prc_over_a_plate_column,
       (["p1", "p1"], 4.0, 6.0), id="plate-comes-from-prc"),
    _P(_case_generate_plate_heatmap_ignores_a_plate_name_column_entirely,
       (["p1", "p1"], 4.0), id="plate-name-column-ignored"),
    _P(_case_generate_plate_heatmap_names_the_plate_in_prc_with_no_metadata_columns,
       (["pX", "pX"], 5.0), id="plate-named-from-prc-without-metadata"),
    _P(_case_generate_plate_heatmap_sum_grouping_adds_values_per_well,
       (6.0, 100.0, 11.0, (6.0, 100.0)), id="sum-grouping-adds-per-well"),
    _P(_case_generate_plate_heatmap_sum_coerces_non_numeric_values_to_nan,
       (1.5, 4.0), id="sum-coerces-non-numeric"),
    _P(_case_generate_plate_heatmap_min_max_int_pair_is_used_verbatim,
       (2.0, 10.0), id="min-max-int-pair-verbatim"),
    _P(_case_generate_plate_heatmap_min_max_tuple_pair_is_accepted,
       (0.0, 42.0), id="min-max-tuple-pair"),
    _P(_case_generate_plate_heatmap_min_max_float_pair_is_treated_as_quantiles,
       pytest.approx((7.0, 82.2)), id="min-max-float-pair-is-quantiles"),
    _P(_case_generate_plate_heatmap_min_max_allq_clips_to_2_98_quantiles,
       pytest.approx((6.2, 96.44)), id="min-max-allq-is-2-98-quantiles"),
    _P(_case_generate_plate_heatmap_unknown_min_max_falls_back_to_full_range,
       (6.0, 100.0), id="min-max-unknown-is-full-range"),
    _P(_case_generate_plate_heatmap_three_element_min_max_falls_back_to_full_range,
       (6.0, 100.0), id="min-max-three-elements-is-full-range"),
    _P(_case_generate_plate_heatmap_degenerate_range_is_nudged,
       (0.0, pytest.approx(1e-6)), id="degenerate-range-is-nudged"),
]


@pytest.mark.parametrize("helper, expected", _HEATMAP_ROWS)
def test_generate_plate_heatmap_maps_wells_and_colour_range(request, helper, expected):
    """Well placement, filtering, aggregation and colour limits of the heatmap."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_generate_plate_heatmap_sum_missing_variable_raises_keyerror,
       "KeyError", id="heatmap-missing-variable-refused"),
    _P(_case_generate_plate_heatmap_rejects_unknown_grouping,
       "ValueError", id="heatmap-unknown-grouping-refused"),
    _P(_case_print_mask_and_flows_empty_flow_list_raises,
       ("ValueError", []), id="mask-flows-empty-flow-list-refused"),
    _P(_case_print_mask_and_flows_rejects_four_dimensional_stack,
       ("ValueError", []), id="mask-flows-four-dimensional-stack-refused"),
])
def test_invalid_plot_input_is_refused_by_name(request, helper, expected):
    """Bad variables, groupings and array shapes raise the named error (and draw nothing)."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_plot_plates_lays_out_one_axis_per_plate_and_saves_pdf,
       (["p1", "p2", "p3", "p4", "p5"], 1, True, []), id="one-axis-per-plate-and-pdf"),
    _P(_case_plot_plates_rewrites_its_own_file_rather_than_numbering_a_new_one,
       ["plate_heatmap_value.pdf"], id="rewrites-own-file"),
    _P(_case_plot_plates_verbose_calls_show_and_skips_saving,
       ([1], [], True), id="verbose-shows-and-skips-saving"),
])
def test_plot_plates_draws_small_multiples_and_exports(request, helper, expected):
    """plot_plates lays out one axis per plate and exports (or shows) the figure."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_print_mask_and_flows_without_flows_uses_two_panels,
       (None, 2, ["Original Image", "Mask with Overlay"], (24, 18), [1]),
       id="no-flows-two-panels"),
    _P(_case_print_mask_and_flows_overlay_draws_red_contours,
       ((24, 18, 3), "uint8", True), id="overlay-draws-red-contours"),
    _P(_case_print_mask_and_flows_without_overlay_shows_raw_mask,
       ("Mask", (24, 18), True), id="no-overlay-shows-raw-mask"),
    _P(_case_print_mask_and_flows_squeezes_singleton_channel,
       (2, (24, 18)), id="singleton-channel-squeezed"),
    _P(_case_print_mask_and_flows_multichannel_stack_uses_first_channel,
       ((24, 18), True), id="multichannel-uses-first-channel"),
    _P(_case_print_mask_and_flows_downsamples_large_inputs,
       (3, (8, 4), (8, 4), "Flows"), id="large-input-downsampled"),
    _P(_case_print_mask_and_flows_small_input_is_not_resized,
       ((20, 16), (20, 16)), id="small-input-not-resized"),
    _P(_case_print_mask_and_flows_three_dimensional_flow_uses_first_channel,
       ((20, 16), True), id="3d-flow-uses-first-channel"),
])
def test_print_mask_and_flows_panels(request, helper, expected):
    """print_mask_and_flows shows the image, mask (or overlay) and flow panels."""
    observed = _call(request, helper)
    assert observed == expected
