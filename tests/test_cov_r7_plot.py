"""Round-7 ``spacr.plot``: the guarantees behind seven guards that no caller
in the module can make false.

Not one of this round's targets can be reached. They are all the same two
shapes that round 6 catalogued for this module, and almost every one of them
turns out to be a NESTED helper whose only call site is a few dozen lines
below its ``def``:

* a parameter the sole caller always passes -- ``find_files``' ``extensions``
  (normalised by the enclosing function first), ``plot_from_file_dict``'s
  ``save=False``, ``join_measurments_and_annotation``'s ``tables=[...]``,
  ``random_color_cmap``'s ``seed=``;
* a re-check of something the enclosing function has already established --
  ``apply_contours_on_image``'s ``image.ndim``, the magenta overlay's
  per-channel outlines, ``create_grouped_plot``'s trailing ``graph_type``
  test, ``spacrGraph._get_positions``' trailing ``graph_type`` test, and
  ``preprocess_data``'s ``grouping_column in df.columns``.

Where the guaranteed value is worth holding on to, it is pinned from the
public API rather than argued in prose, so that the day one of these
guarantees stops holding a test says so. Nothing is excluded from coverage.

CPU-only and offline throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import tifffile  # noqa: E402

import spacr.plot as P  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


def _titles():
    """Every axes title currently on screen, in figure order."""
    return [axes.get_title()
            for number in plt.get_fignums()
            for axes in plt.figure(number).axes]


# ---------------------------------------------------------------------------
# find_files -- the extension default, applied one scope up
# ---------------------------------------------------------------------------

def _pair(root, stem, suffix):
    """One intensity/mask pair of ``stem`` under ``root/images`` and
    ``root/masks``, written in whichever format ``suffix`` names."""
    rng = np.random.default_rng(abs(hash(stem)) % 1000)
    intensity = rng.integers(0, 4096, size=(48, 48)).astype(np.uint16)
    mask = np.zeros((48, 48), dtype=np.uint16)
    mask[4:20, 4:20] = 1
    mask[28:40, 28:40] = 2
    for folder, array in (("images", intensity), ("masks", mask)):
        directory = root / folder
        directory.mkdir(exist_ok=True)
        path = directory / f"{stem}{suffix}"
        if suffix == ".npy":
            np.save(path, array)
        else:
            tifffile.imwrite(str(path), array)
    return root / "images", root / "masks"


def test_a_bare_call_walks_npy_and_tif_and_nothing_else(tmp_path):
    """plot.py:1723->1724 -- why the nested default is never the one applied.

    ``find_files``' ``if extensions is None`` cannot be true. The enclosing
    ``plot_images_and_arrays`` replaces ``None`` with the SAME four
    extensions at plot.py:1691-1692, before either nested function is even
    defined, and plot.py:1816 is ``find_files``' only call site. The nested
    default is a duplicate of a substitution that has already happened.

    The list itself is what matters and is pinned here. It has to cover both
    of the formats a spaCR project actually stores -- the ``.npy`` merged
    arrays and the ``.tif`` masks -- because a name missing from ONE of the
    folders is dropped from the result entirely, so a default naming only
    one format would silently show nothing at all.
    """
    _pair(tmp_path, "arrays", ".npy")
    _pair(tmp_path, "tiffs", ".tif")
    folders = [str(tmp_path / "images"), str(tmp_path / "masks")]

    P.plot_images_and_arrays(folders, randomize=False)
    assert sorted(t for t in _titles() if t.endswith("- Mask")) == [
        "arrays - Mask", "tiffs - Mask"]

    # An explicit list beats the default, which is what makes the pair above
    # the default's doing and not everything that happens to be on disk.
    plt.close("all")
    P.plot_images_and_arrays(folders, extensions=[".npy"], randomize=False)
    assert sorted(t for t in _titles() if t.endswith("- Mask")) == [
        "arrays - Mask"]

    plt.close("all")
    P.plot_images_and_arrays(folders, extensions=[".jpg"], randomize=False)
    assert _titles() == []


# ---------------------------------------------------------------------------
# The magenta overlay's per-channel outlines
# ---------------------------------------------------------------------------

SHAPE = (48, 48)
MAGENTA = (1.0, 0.0, 1.0)


def _mask(slices, label):
    array = np.zeros(SHAPE, dtype=np.uint16)
    array[slices] = label
    return array


def _stack(masks, n_intensity=3):
    rng = np.random.default_rng(11)
    planes = [rng.random(SHAPE).astype(np.float32) for _ in range(n_intensity)]
    return np.dstack([*planes, *[m.astype(np.float32) for m in masks]])


def test_every_channel_that_has_an_object_gets_that_object_outlined(tmp_path):
    """plot.py:1149->1152 and 1152->1179 -- why neither can be reached.

    ``cell_outlines``/``nucleus_outlines``/``pathogen_outlines`` are each
    assigned by ``np.take`` in the same ``if <role>_channel is not None``
    block that appends that channel to ``channels_with_outlines``
    (plot.py:1310-1341). So a channel that reaches the per-channel branch
    matches exactly one of the three equality tests AND has a non-None
    outline beside it: the chain cannot fall through, and ``outline`` cannot
    still be None below it.

    The guarantee is what this asserts: each channel is outlined with its
    OWN object and no other's. A fall-through would draw an unannotated
    panel that looks like a channel in which nothing was segmented.
    """
    cell = _mask(np.s_[4:20, 4:20], 1)
    nucleus = _mask(np.s_[8:14, 8:14], 1)
    pathogen = _mask(np.s_[30:38, 30:38], 1)
    stack = _stack([cell, nucleus, pathogen])
    np.save(tmp_path / "fov.npy", stack)

    figure = P.plot_image_mask_overlay_magenta_outlines(
        str(tmp_path / "fov.npy"), [0, 1, 2], cell_channel=0,
        nucleus_channel=1, pathogen_channel=2, figuresize=2, thickness=1,
        save_pdf=False, mode="outlines")

    def magenta_pixels(index):
        panel = np.asarray(figure.axes[index].images[0].get_array())
        return np.all(np.isclose(panel, MAGENTA, atol=1e-6), axis=-1)

    # Every object channel is outlined, and each panel's magenta sits on its
    # own object's bounding box rather than on a neighbour's.
    for index, region in enumerate((np.s_[4:20, 4:20], np.s_[8:14, 8:14],
                                    np.s_[30:38, 30:38])):
        drawn = magenta_pixels(index)
        assert drawn.any(), f"channel {index} was not outlined"
        assert drawn.sum() == drawn[region].sum()


# ---------------------------------------------------------------------------
# apply_contours_on_image -- the image is always the grayscale base
# ---------------------------------------------------------------------------

def test_a_multichannel_stack_is_outlined_on_its_first_channel_alone():
    """plot.py:3358->3362 -- the ``else`` that copies a colour image is dead.

    ``print_mask_and_flows`` reduces its input to a single plane before it
    overlays anything: a 2-D stack is used as-is and a 3-D stack becomes
    ``stack[..., 0]`` (plot.py:3403-3408), with anything else refused
    outright. So ``apply_contours_on_image``'s only caller always hands it a
    2-D array and the ``image.copy()`` branch cannot run.

    That is the behaviour, not an accident of it: the overlay panel is a
    GRAYSCALE base with coloured contours on it, so the contour colour means
    "object boundary" and cannot be confused with channel 2 being bright.
    """
    base = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    stack = np.dstack([base, np.ones_like(base), np.zeros_like(base)])
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[8:24, 8:24] = 1

    P.print_mask_and_flows(stack, mask, flows=None, overlay=True)
    panel = np.asarray(plt.gcf().axes[1].images[0].get_array())

    assert panel.shape == (32, 32, 3)
    red = np.all(panel == (255, 0, 0), axis=-1)
    assert red.any()                     # the contour really was drawn
    # Everywhere else the three channels are equal: the base is channel 0
    # replicated, not the green and blue planes of the stack.
    grey = panel[~red]
    assert np.array_equal(grey[:, 0], grey[:, 1])
    assert np.array_equal(grey[:, 1], grey[:, 2])
    expected = np.clip(base, 0, 1) * 255
    assert panel[0, 0, 0] == expected.astype(np.uint8)[0, 0]
    assert panel[31, 0, 0] == expected.astype(np.uint8)[31, 0]

    # The 2-D form takes the same branch and produces the same panel, which
    # is why the else has no caller of its own.
    plt.close("all")
    P.print_mask_and_flows(base, mask, flows=None, overlay=True)
    assert np.array_equal(
        np.asarray(plt.gcf().axes[1].images[0].get_array()), panel)


# ---------------------------------------------------------------------------
# create_grouped_plot -- the trailing graph_type test
# ---------------------------------------------------------------------------

def _grouped_frame():
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "grp": ["a"] * 12 + ["b"] * 12,
        "val": np.concatenate([rng.normal(1.0, 0.2, 12),
                               rng.normal(1.6, 0.2, 12)]),
    })


@pytest.mark.parametrize("graph_type", ["bar", "violin", "jitter", "box",
                                        "jitter_box", "jitter_bar", "line"])
def test_every_graph_type_the_error_message_offers_actually_draws(graph_type):
    """plot.py:4550->4557 -- why the trailing ``not in`` is never False.

    The ``elif`` chain above it already consumes ``bar``, ``violin``,
    ``jitter``, ``box``, ``jitter_box``, ``jitter_bar``, ``line`` and
    ``line_std`` (plot.py:4444-4524). The five names retested at 4550 are a
    strict subset of those, so anything that reaches 4550 matched none of
    them and the test is true by construction.

    What matters is that the seven names its own message offers all draw
    something: ``line`` and ``jitter_bar`` used to fall out of this chain
    with no branch at all, and ``plt.gcf()`` below handed back a blank
    figure with no error.
    """
    figure, results = P.create_grouped_plot(
        df=_grouped_frame(), grouping_column="grp", data_column="val",
        graph_type=graph_type, save=False)
    assert figure.axes
    assert any(axes.has_data() for axes in figure.axes)
    assert isinstance(results, pd.DataFrame)


def test_a_graph_type_the_menu_does_not_offer_is_refused_by_name():
    """The complement of the parametrisation above: an unknown type raises.

    Reaching plot.py:4550 at all takes a name none of the eight branches
    matched, and then the ``not in`` can only be true -- so the raise is the
    branch's whole behaviour.
    """
    with pytest.raises(ValueError, match="graph_type='swarm' is not one of"):
        P.create_grouped_plot(df=_grouped_frame(), grouping_column="grp",
                              data_column="val", graph_type="swarm",
                              save=False)


# ---------------------------------------------------------------------------
# spacrGraph.preprocess_data -- the grouping column is always still there
# ---------------------------------------------------------------------------

def test_a_grouping_column_that_is_not_in_the_frame_never_reaches_the_ordering():
    """plot.py:4828->4836 -- the ``elif`` can only ever be true.

    ``preprocess_data`` opens with ``df.dropna(subset=[self.grouping_column]
    + self.data_column)`` (plot.py:4784), which raises ``KeyError`` on a
    grouping column the frame does not have, and every aggregation below it
    groups BY that column, so ``reset_index`` puts it back as a column. By
    the time the ordering runs it is always present.

    The ordering it therefore always applies is the observable half: groups
    come out as an ordered Categorical, so a plot's x-axis is in a declared
    order rather than in whatever order the rows happened to arrive.
    """
    frame = _grouped_frame()

    with pytest.raises(KeyError):
        P.spacrGraph(frame, "no_such_column", "val")

    default = P.spacrGraph(frame, "grp", "val").df
    assert isinstance(default["grp"].dtype, pd.CategoricalDtype)
    assert list(default["grp"].cat.categories) == ["a", "b"]

    ordered = P.spacrGraph(frame, "grp", "val", order=["b", "a"]).df
    assert list(ordered["grp"].cat.categories) == ["b", "a"]

    # Aggregating to one row per well still leaves the grouping column in
    # the frame, which is the other half of why the test cannot be false.
    wells = frame.assign(prc=["p1_r1_c1"] * 12 + ["p1_r1_c2"] * 12)
    per_well = P.spacrGraph(wells, "grp", "val", representation="well").df
    assert "grp" in per_well.columns
    assert list(per_well["grp"].cat.categories) == ["a", "b"]


# ---------------------------------------------------------------------------
# volcano_plot -- the threshold helper's unknown-transform raise
# ---------------------------------------------------------------------------

def test_an_unknown_x_transform_is_refused_before_any_threshold_is_converted():
    """plot.py:7304->7306 -- why the second "Unknown x_transform" is dead.

    ``_transform_x`` is applied to the data at plot.py:7332, eight lines
    before ``_threshold_x_in_plot_units`` is called at 7340, and it raises on
    exactly the same set of unknown names. The two messages are spelled
    differently, which is what this test reads: ``_transform_x`` lower-cases
    the mode first, so a mixed-case name comes back lower-cased. A message
    carrying the original spelling would mean the dead raise had run.
    """
    frame = pd.DataFrame({"fc": [1.5, 2.0, 0.5], "p": [0.01, 0.2, 0.04]})

    with pytest.raises(ValueError) as failure:
        P.volcano_plot(frame, fold_change_col="fc", p_value_col="p",
                       x_transform="LogE", fold_change_threshold=1.5,
                       show=False)
    assert str(failure.value) == "Unknown x_transform: loge"

    # A transform that IS known gets as far as converting the threshold into
    # plot units, so the refusal above is about the name and not about the
    # threshold being set.
    figure, axes, _texts = P.volcano_plot(
        frame, fold_change_col="fc", p_value_col="p", x_transform="log2",
        fold_change_threshold=2.0, show=False)
    assert figure is not None
    # log2(2.0) == 1.0, drawn as a pair of vertical threshold lines either
    # side of the axis' own zero line.
    verticals = sorted({round(float(line.get_xdata()[0]), 6)
                        for line in axes.lines
                        if len(set(line.get_xdata())) == 1})
    assert verticals == [-1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Proved unreachable, argued in place
#
# The remaining targets on this round's list have no cheap executable pin,
# and every one of them is a nested helper called from exactly one place in
# the function that defines it:
#
# * plot.py:1039 ``if seed is not None`` in the magenta variant's
#   ``random_color_cmap``. Its three call sites -- 1137, 1159, 1175 -- all
#   pass ``random.randint(0, 100)`` positionally, and 1190 passes another,
#   so ``seed`` is never None. (The all_on_all variant's copy at plot.py:608
#   is the same shape: 724, 745 and 760 pass ``seed=`` explicitly.) The
#   default exists for the signature, not for a caller.
#
# * plot.py:1809-1811 ``if save:`` in ``plot_from_file_dict``. Its only call
#   site, plot.py:1824, passes ``save=False`` as a literal --
#   ``plot_images_and_arrays`` has no ``save`` parameter of its own to
#   forward -- so the branch cannot be entered from any public API. Note
#   that it would also save to ``folder``, the loop variable left over from
#   the reading loop above, which is the LAST folder walked rather than the
#   one the image came from.
#
# * plot.py:4118-4119 ``if tables is None`` in
#   ``join_measurments_and_annotation``. Its only call site, plot.py:4141,
#   passes ``tables=['cell', 'nucleus', 'pathogen', 'cytoplasm']`` -- the
#   same four names the default spells -- so the default is never applied.
#
# * plot.py:5295->5298 ``elif self.graph_type in ['line', 'line_std']`` in
#   ``spacrGraph._get_positions``. ``create_plot`` raises "Unknown graph
#   type" at plot.py:5397 for anything outside the eight it draws, and that
#   happens before ``_get_positions`` is called at 5420. The eight are
#   partitioned exactly by the four branches above this one plus this one,
#   so the last test can never be false -- and if it were, ``x_positions``
#   would be unbound and the return would raise UnboundLocalError.
#
# * plot.py:5339->5341 ``if not self.results_df.empty`` in ``create_plot``.
#   ``results_df`` is empty only when ``perform_normality_tests`` appended
#   no row, i.e. when the preprocessed frame has no groups at all
#   (``data_column=[]`` is refused earlier, at plot.py:4718). A frame with
#   no groups does reach this line -- and then dies eleven lines later in
#   ``_standerdize_figure_format`` at plot.py:5586, dividing 1.5 by
#   ``num_groups`` of zero. The guard is defensive against a state the
#   function cannot survive, so it is left proved rather than pinned: a test
#   for it would have to assert a ZeroDivisionError as though it were the
#   intended behaviour.
#
# Round 6 already proved four more of this round's targets, in
# tests/test_cov_r6_plot.py, and re-measuring them here agrees:
# plot.py:734->767 and 739->767 (``channel_to_outline`` is built in the same
# function that consumes it, one non-None entry per non-None channel),
# 3905->3908 and 3908->3911 (``x_lim``/``y_lim`` replaced at the top of
# ``plot_lorenz_curves``), and 4077->4079 (``y_lim`` replaced at the top of
# ``read_and_plot__vision_results``). All three of those are pinned
# executably there; nothing is duplicated here.
# ---------------------------------------------------------------------------
