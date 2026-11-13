"""CPU-only coverage for the Cellpose/organelle *output display* block of
``spacr.plot`` (plot.py lines ~715-960).

The functions in this block are pure matplotlib side-effect renderers -- they
return ``None`` and communicate only through the figures they build.  Every
test here therefore asserts on the *drawn artefacts*: how many figures were
created, how many axes each one has, the axis titles, the pixel data handed to
each ``imshow`` and the per-object label annotations.  Nothing is a smoke test.

Covered here:

  * ``plot_cellpose4_output`` -- the full channel/mask/flow panel row, the
    ``print_object_number=False`` short-circuit and the ``nr`` cap that stops
    plotting after N images (the ``index < nr`` false branch + trailing
    ``return``).
  * ``plot_organelle_output`` -- the whole body: the three-panel layout, the
    ``0 in mask`` / no-background object count, the annotation branch, the
    ``min(len(masks), nr, img_batch.shape[0])`` cap and the empty-masks
    early-out.
  * ``plot_masks`` -- the ``file_type='png'`` branch that unwraps ``f[0]`` out
    of each flow entry, plus the 3-D-batch and non-list masks/flows coercions.
  * ``_plot_4D_arrays`` -- the ``.npz`` discovery + ``random.sample`` cap, the
    ``num_channels == 1`` axes-wrapping branch, the multi-channel branch and
    the empty-directory early-out.

The renderers must label every real object even when the mask has no
background pixels (the ``*_when_mask_has_no_background`` rows).

Each contract is one parametrised test per function.  The rows name a helper
that does the setup and call (keeping its secondary checks) and returns what
it observed; the row carries the expected value and the test body compares
the two.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


@pytest.fixture(autouse=True)
def _no_lingering_figures():
    plt.close("all")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _figures():
    """Every open figure, in creation order."""
    return [plt.figure(n) for n in plt.get_fignums()]


def _label_mask(size=32, n=3):
    """Label mask with ``n`` disjoint horizontal bars and real 0 background."""
    m = np.zeros((size, size), dtype=np.int32)
    h = size // (n + 1)
    for i in range(n):
        m[i * h + 1:(i + 1) * h - 1, 2:size - 2] = i + 1
    assert (m == 0).any(), "helper must leave background"
    return m


def _full_mask(size=16):
    """Label mask with NO background pixels (labels 1 and 2 only)."""
    m = np.ones((size, size), dtype=np.int32)
    m[size // 2:] = 2
    return m


def _rgb_flow(size=32, seed=0):
    """Cellpose-style RGB flow image, uint8 (H, W, 3)."""
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def _batch(n=1, size=32, chans=2, seed=1):
    return np.random.default_rng(seed).random((n, size, size, chans)).astype(np.float32)


def _titles(fig):
    return [a.get_title() for a in fig.axes]


def _img_array(ax):
    return np.asarray(ax.get_images()[0].get_array())


# ---------------------------------------------------------------------------
# plot_cellpose4_output
# ---------------------------------------------------------------------------

def _case_plot_cellpose4_output_builds_channel_mask_and_flow_panels():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=1, size=32, chans=2)
    mask = _label_mask(32, n=3)
    flow = _rgb_flow(32)

    out = plot_cellpose4_output(batch, [mask], [flow], figuresize=2, nr=1)

    figs = _figures()
    fig = figs[0]
    np.testing.assert_allclose(_img_array(fig.axes[0]), batch[0][..., 0])
    np.testing.assert_allclose(_img_array(fig.axes[1]), batch[0][..., 1])
    np.testing.assert_array_equal(_img_array(fig.axes[2]), mask)
    np.testing.assert_array_equal(_img_array(fig.axes[3]), flow)
    # font size follows figuresize / 2
    assert fig.axes[2].texts[0].get_fontsize() == pytest.approx(1.0)
    assert fig.axes[2].texts[0].get_color() == "white"
    # 2 image channels + mask + flow, each panel with exactly one image; one
    # annotation per foreground label, drawn on the mask panel only
    return (out, len(figs), _titles(fig), [len(a.get_images()) for a in fig.axes],
            sorted(t.get_text() for t in fig.axes[2].texts),
            [len(a.texts) for a in fig.axes])


def _case_plot_cellpose4_output_without_object_numbers_draws_no_text():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=1, size=32, chans=1)
    mask = _label_mask(32, n=3)

    plot_cellpose4_output(batch, [mask], [_rgb_flow(32)], figuresize=2, nr=1,
                          print_object_number=False)

    fig = _figures()[0]
    # 1 channel + mask + flow
    return len(fig.axes), _titles(fig), [len(a.texts) for a in fig.axes]


def _case_plot_cellpose4_output_nr_caps_the_number_of_figures():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=4, size=16, chans=1)
    masks = [_label_mask(16, n=2) for _ in range(4)]
    flows = [_rgb_flow(16, seed=i) for i in range(4)]

    # nr=2 -> the third and fourth images fall through the `index < nr` guard
    out = plot_cellpose4_output(batch, masks, flows, figuresize=2, nr=2)

    figs = _figures()
    for i, fig in enumerate(figs):
        np.testing.assert_array_equal(_img_array(fig.axes[2]), flows[i])
    return out, len(figs)


def _case_plot_cellpose4_output_accepts_2d_flow_field():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=1, size=16, chans=1)
    flow2d = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)

    plot_cellpose4_output(batch, [_label_mask(16, n=2)], [flow2d], figuresize=2, nr=1)

    fig = _figures()[0]
    np.testing.assert_allclose(_img_array(fig.axes[2]), flow2d)
    return fig.axes[2].get_title(), fig.axes[2].get_images()[0].get_cmap().name


def _case_plot_cellpose4_output_labels_every_object_when_mask_has_no_background():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=1, size=16, chans=2)
    mask = _full_mask(16)  # labels {1, 2}, no zeros anywhere
    assert set(np.unique(mask)) == {1, 2}

    plot_cellpose4_output(batch, [mask], [_rgb_flow(16)], figuresize=2, nr=1)

    fig = _figures()[0]
    return sorted(t.get_text() for t in fig.axes[2].texts)


# ---------------------------------------------------------------------------
# plot_organelle_output
# ---------------------------------------------------------------------------

def _organelle_batch(n=1, size=48, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.random((n, size, size)).astype(np.float32) * 0.1
    # a couple of bright spots so the top-hat diagnostic is not degenerate
    img[:, 10:16, 10:16] = 0.9
    img[:, 30:36, 30:36] = 0.8
    return img


def _case_plot_organelle_output_spots_otsu_three_panels():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=1, size=48)
    mask = _label_mask(48, n=3)
    settings = {
        "organelle_morphology": "spots",
        "organelle_method": "otsu",
        "organelle_tophat_radius": 3,
    }

    out = plot_organelle_output(imgs, [mask], settings, figuresize=2, nr=1)

    figs = _figures()
    fig = figs[0]
    np.testing.assert_allclose(_img_array(fig.axes[0]), imgs[0])
    np.testing.assert_array_equal(_img_array(fig.axes[1]), mask)
    # diagnostic panel is a same-shaped 2-D image, not the raw channel
    diag = _img_array(fig.axes[2])
    assert diag.shape == imgs[0].shape
    assert not np.allclose(diag, imgs[0])
    # every foreground label annotated, background never annotated; all three
    # panels have their frames/ticks switched off
    return (out, len(figs), len(fig.axes), _titles(fig),
            sorted(t.get_text() for t in fig.axes[1].texts),
            [len(a.texts) for a in fig.axes], [a.axison for a in fig.axes])


def _case_plot_organelle_output_defaults_to_spots_otsu_when_settings_empty():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=1, size=48)
    plot_organelle_output(imgs, [_label_mask(48, n=2)], {}, figuresize=2, nr=1)

    fig = _figures()[0]
    # default organelle_tophat_radius is 5
    return _titles(fig)


def _case_plot_organelle_output_counts_all_labels_when_mask_has_no_background():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=1, size=16)
    mask = _full_mask(16)

    plot_organelle_output(imgs, [mask], {"organelle_morphology": "irregular"},
                          figuresize=2, nr=1)

    fig = _figures()[0]
    # `0 in mask` is False here -> no background subtraction from the count
    # irregular morphology routes to the gaussian diagnostic
    return (fig.axes[1].get_title(),
            sorted(t.get_text() for t in fig.axes[1].texts),
            fig.axes[0].get_title(),
            fig.axes[2].get_title().startswith("Gaussian smoothed"))


def _case_plot_organelle_output_without_object_numbers_draws_no_text():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=1, size=48)
    plot_organelle_output(imgs, [_label_mask(48, n=3)],
                          {"organelle_morphology": "spots", "organelle_method": "otsu"},
                          figuresize=2, nr=1, print_object_number=False)

    fig = _figures()[0]
    return fig.axes[1].get_title(), [len(a.texts) for a in fig.axes]


def _case_plot_organelle_output_stops_at_the_smallest_of_masks_nr_and_batch():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=2, size=32)          # batch limit = 2
    masks = [_label_mask(32, n=2) for _ in range(3)]  # mask limit = 3
    plot_organelle_output(imgs, masks, {}, figuresize=2, nr=5)  # nr limit = 5

    figs = _figures()
    for fig, img in zip(figs, imgs):
        np.testing.assert_allclose(_img_array(fig.axes[0]), img)
    return len(figs)  # min(3, 5, 2)


def _case_plot_organelle_output_with_no_masks_draws_nothing():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=2, size=16)
    out = plot_organelle_output(imgs, [], {}, figuresize=2, nr=3)
    return out, plt.get_fignums()


# ---------------------------------------------------------------------------
# plot_masks
# ---------------------------------------------------------------------------

def _case_plot_masks_png_file_type_unwraps_the_first_flow_element():
    from spacr.plot import plot_masks

    batch = _batch(n=1, size=32, chans=2)
    mask = _label_mask(32, n=3)
    flow_rgb = _rgb_flow(32, seed=7)
    # cellpose PNG runs hand back [[ [rgb, dP, cellprob] ]]: the outer list is
    # stripped by `flows = flows[0]`, then file_type='png' takes f[0].
    flows = [[[flow_rgb, np.zeros((2, 32, 32), np.float32), "not-an-array"]]]

    out = plot_masks(batch, [mask], flows, figuresize=2, nr=1, file_type="png")

    fig = _figures()[0]
    # the flow panel shows f[0]; without the unwrap imshow would have raised
    np.testing.assert_array_equal(_img_array(fig.axes[3]), flow_rgb)
    return out, _titles(fig), sorted(t.get_text() for t in fig.axes[2].texts)


def _case_plot_masks_expands_a_single_3d_image_and_wraps_bare_arrays():
    from spacr.plot import plot_masks

    single = _batch(n=1, size=16, chans=1)[0]       # (H, W, C) -> expanded
    mask = _label_mask(16, n=2)                      # bare ndarray -> wrapped
    flow = _rgb_flow(16, seed=9)                     # bare ndarray -> wrapped

    plot_masks(single, mask, flow, figuresize=2, nr=1)

    figs = _figures()
    fig = figs[0]
    np.testing.assert_allclose(_img_array(fig.axes[0]), single[..., 0])
    np.testing.assert_array_equal(_img_array(fig.axes[1]), mask)
    np.testing.assert_array_equal(_img_array(fig.axes[2]), flow)
    return (len(figs), len(fig.axes),
            sorted(t.get_text() for t in fig.axes[1].texts))


def _case_plot_masks_nr_zero_creates_no_figure():
    from spacr.plot import plot_masks

    batch = _batch(n=1, size=16, chans=1)
    out = plot_masks(batch, _label_mask(16, n=2), _rgb_flow(16), nr=0)
    return out, plt.get_fignums()


# ---------------------------------------------------------------------------
# _plot_4D_arrays
# ---------------------------------------------------------------------------

def _write_npz(path, arr):
    np.savez(path, data=arr)


def _case_plot_4D_arrays_multichannel_plots_one_panel_per_channel(tmp_path):
    from spacr.plot import _plot_4D_arrays

    stack = np.random.default_rng(5).random((2, 16, 16, 3)).astype(np.float32)
    _write_npz(tmp_path / "field.npz", stack)

    out = _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=1, nr=2)

    figs = _figures()
    panels = []
    for i, fig in enumerate(figs):
        panels.append((len(fig.axes), _titles(fig), [a.axison for a in fig.axes]))
        for c, ax in enumerate(fig.axes):
            np.testing.assert_allclose(_img_array(ax), stack[i][:, :, c])
            # THE TITLE SCALES WITH THE PANEL NOW. It was pinned at 24 pt whatever
    # the panel size was, so on the 2-inch panels a browse loop draws, the
    # title was taller than the image it labelled. `_montage_type_size` keeps
    # the house tiers' ratios and scales them by the canvas.
    from spacr.plot import _montage_type_size

    assert ax.title.get_size() == pytest.approx(_montage_type_size(2))
    # nr=2 images out of the 2 in the stack
    return out, len(figs), panels


def _case_plot_4D_arrays_single_channel_uses_one_axes(tmp_path):
    from spacr.plot import _plot_4D_arrays

    stack = np.random.default_rng(6).random((1, 16, 16, 1)).astype(np.float32)
    _write_npz(tmp_path / "one_channel.npz", stack)

    _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=1, nr=1, cmap="gray")

    figs = _figures()
    fig = figs[0]
    np.testing.assert_allclose(_img_array(fig.axes[0]), stack[0][:, :, 0])
    # axs was wrapped in a list, not indexed as an array (one axes)
    return (len(figs), len(fig.axes), fig.axes[0].get_title(),
            fig.axes[0].get_images()[0].get_cmap().name)


def _case_plot_4D_arrays_nr_npz_caps_files_and_ignores_non_npz(tmp_path):
    from spacr.plot import _plot_4D_arrays

    stacks = {}
    for i in range(3):
        arr = np.full((1, 8, 8, 1), float(i), dtype=np.float32)
        stacks[f"s{i}.npz"] = arr
        _write_npz(tmp_path / f"s{i}.npz", arr)
    # decoys that must never be opened by np.load
    (tmp_path / "readme.txt").write_text("not a stack", encoding="utf-8")
    (tmp_path / "field.tif").write_bytes(b"\x00\x01")

    _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=2, nr=1)

    figs = _figures()
    drawn = {float(_img_array(f.axes[0])[0, 0]) for f in figs}
    assert drawn.issubset({0.0, 1.0, 2.0})
    # 2 of the 3 .npz files, 1 image each; two *different* files were sampled
    return len(figs), len(drawn)


def _case_plot_4D_arrays_empty_directory_plots_nothing(tmp_path):
    from spacr.plot import _plot_4D_arrays

    (tmp_path / "only.txt").write_text("nothing to plot", encoding="utf-8")
    out = _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=3, nr=1)
    return out, plt.get_fignums()


def _case_plot_4D_arrays_nr_larger_than_stack_is_clamped(tmp_path):
    from spacr.plot import _plot_4D_arrays

    stack = np.zeros((2, 8, 8, 2), dtype=np.float32)
    _write_npz(tmp_path / "small.npz", stack)

    _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=1, nr=99)

    # min(99, num_images=2) figures; the source file is left in place
    return len(_figures()), os.path.exists(tmp_path / "small.npz")


_P = pytest.param
_FLOW_TITLES = ["Image - Channel0", "Image - Channel1", "Mask", "Flow"]


@pytest.mark.parametrize("helper, expected", [
    _P(_case_plot_cellpose4_output_builds_channel_mask_and_flow_panels,
       (None, 1, _FLOW_TITLES, [1, 1, 1, 1], ["1", "2", "3"], [0, 0, 3, 0]),
       id="builds-channel-mask-and-flow-panels"),
    _P(_case_plot_cellpose4_output_without_object_numbers_draws_no_text,
       (3, ["Image - Channel0", "Mask", "Flow"], [0, 0, 0]),
       id="no-object-numbers-draws-no-text"),
    _P(_case_plot_cellpose4_output_nr_caps_the_number_of_figures,
       (None, 2), id="nr-caps-number-of-figures"),
    _P(_case_plot_cellpose4_output_accepts_2d_flow_field,
       ("Flow", "viridis"), id="accepts-2d-flow-field"),
    _P(_case_plot_cellpose4_output_labels_every_object_when_mask_has_no_background,
       ["1", "2"], id="labels-every-object-without-background"),
])
def test_plot_cellpose4_output_draws_panels_and_labels(request, helper, expected):
    """plot_cellpose4_output draws channel, mask and flow panels with one label per object."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_plot_organelle_output_spots_otsu_three_panels,
       (None, 1, 3,
        ["Organelle channel (spots/otsu)", "Mask (3 objects)",
         "Top-hat filtered (r=3)"],
        ["1", "2", "3"], [0, 3, 0], [False, False, False]),
       id="spots-otsu-three-panels"),
    _P(_case_plot_organelle_output_defaults_to_spots_otsu_when_settings_empty,
       ["Organelle channel (spots/otsu)", "Mask (2 objects)",
        "Top-hat filtered (r=5)"], id="defaults-to-spots-otsu"),
    _P(_case_plot_organelle_output_counts_all_labels_when_mask_has_no_background,
       ("Mask (2 objects)", ["1", "2"], "Organelle channel (irregular/otsu)", True),
       id="counts-all-labels-without-background"),
    _P(_case_plot_organelle_output_without_object_numbers_draws_no_text,
       ("Mask (3 objects)", [0, 0, 0]), id="no-object-numbers-draws-no-text"),
    _P(_case_plot_organelle_output_stops_at_the_smallest_of_masks_nr_and_batch,
       2, id="stops-at-smallest-of-masks-nr-batch"),
    _P(_case_plot_organelle_output_with_no_masks_draws_nothing,
       (None, []), id="no-masks-draws-nothing"),
])
def test_plot_organelle_output_draws_panels_and_counts_objects(request, helper, expected):
    """plot_organelle_output titles its panels, counts objects and caps the figures."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_plot_masks_png_file_type_unwraps_the_first_flow_element,
       (None, _FLOW_TITLES, ["1", "2", "3"]), id="png-file-type-unwraps-first-flow"),
    _P(_case_plot_masks_expands_a_single_3d_image_and_wraps_bare_arrays,
       (1, 3, ["1", "2"]), id="expands-single-3d-image-and-wraps-arrays"),
    _P(_case_plot_masks_nr_zero_creates_no_figure,
       (None, []), id="nr-zero-creates-no-figure"),
])
def test_plot_masks_coerces_inputs_and_draws(request, helper, expected):
    """plot_masks unwraps png flows, wraps bare arrays and honours nr=0."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_plot_4D_arrays_multichannel_plots_one_panel_per_channel,
       (None, 2, [(3, ["Channel 0", "Channel 1", "Channel 2"], [False] * 3)] * 2),
       id="multichannel-one-panel-per-channel"),
    _P(_case_plot_4D_arrays_single_channel_uses_one_axes,
       (1, 1, "Channel 0", "gray"), id="single-channel-one-axes"),
    _P(_case_plot_4D_arrays_nr_npz_caps_files_and_ignores_non_npz,
       (2, 2), id="nr-npz-caps-files-and-ignores-non-npz"),
    _P(_case_plot_4D_arrays_empty_directory_plots_nothing,
       (None, []), id="empty-directory-plots-nothing"),
    _P(_case_plot_4D_arrays_nr_larger_than_stack_is_clamped,
       (2, True), id="nr-larger-than-stack-clamped"),
])
def test_plot_4d_arrays_draws_one_figure_per_image(request, helper, expected):
    """_plot_4D_arrays draws a panel per channel for each sampled npz image, capped by nr."""
    observed = _call(request, helper)
    assert observed == expected
