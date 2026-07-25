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

One strict xfail records a real defect: the two Cellpose renderers annotate
``np.unique(mask)[1:]``, which silently drops a *real* label whenever the mask
has no background pixels (``plot_organelle_output`` gets this right by
filtering ``!= 0`` explicitly).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


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

def test_plot_cellpose4_output_builds_channel_mask_and_flow_panels():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=1, size=32, chans=2)
    mask = _label_mask(32, n=3)
    flow = _rgb_flow(32)

    out = plot_cellpose4_output(batch, [mask], [flow], figuresize=2, nr=1)

    assert out is None
    figs = _figures()
    assert len(figs) == 1
    fig = figs[0]
    # 2 image channels + mask + flow
    assert len(fig.axes) == 4
    assert _titles(fig) == [
        "Image - Channel0", "Image - Channel1", "Mask", "Flow",
    ]
    # every panel got exactly one image
    assert [len(a.get_images()) for a in fig.axes] == [1, 1, 1, 1]
    np.testing.assert_allclose(_img_array(fig.axes[0]), batch[0][..., 0])
    np.testing.assert_allclose(_img_array(fig.axes[1]), batch[0][..., 1])
    np.testing.assert_array_equal(_img_array(fig.axes[2]), mask)
    np.testing.assert_array_equal(_img_array(fig.axes[3]), flow)
    # one annotation per foreground label, drawn on the mask panel only
    assert sorted(t.get_text() for t in fig.axes[2].texts) == ["1", "2", "3"]
    assert [len(a.texts) for a in fig.axes] == [0, 0, 3, 0]
    # font size follows figuresize / 2
    assert fig.axes[2].texts[0].get_fontsize() == pytest.approx(1.0)
    assert fig.axes[2].texts[0].get_color() == "white"


def test_plot_cellpose4_output_without_object_numbers_draws_no_text():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=1, size=32, chans=1)
    mask = _label_mask(32, n=3)

    plot_cellpose4_output(batch, [mask], [_rgb_flow(32)], figuresize=2, nr=1,
                          print_object_number=False)

    fig = _figures()[0]
    assert len(fig.axes) == 3  # 1 channel + mask + flow
    assert _titles(fig) == ["Image - Channel0", "Mask", "Flow"]
    assert all(len(a.texts) == 0 for a in fig.axes)


def test_plot_cellpose4_output_nr_caps_the_number_of_figures():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=4, size=16, chans=1)
    masks = [_label_mask(16, n=2) for _ in range(4)]
    flows = [_rgb_flow(16, seed=i) for i in range(4)]

    # nr=2 -> the third and fourth images fall through the `index < nr` guard
    assert plot_cellpose4_output(batch, masks, flows, figuresize=2, nr=2) is None

    figs = _figures()
    assert len(figs) == 2
    for i, fig in enumerate(figs):
        np.testing.assert_array_equal(_img_array(fig.axes[2]), flows[i])


def test_plot_cellpose4_output_accepts_2d_flow_field():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=1, size=16, chans=1)
    flow2d = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)

    plot_cellpose4_output(batch, [_label_mask(16, n=2)], [flow2d], figuresize=2, nr=1)

    fig = _figures()[0]
    assert fig.axes[2].get_title() == "Flow"
    np.testing.assert_allclose(_img_array(fig.axes[2]), flow2d)
    assert fig.axes[2].get_images()[0].get_cmap().name == "viridis"


def test_plot_cellpose4_output_labels_every_object_when_mask_has_no_background():
    from spacr.plot import plot_cellpose4_output

    batch = _batch(n=1, size=16, chans=2)
    mask = _full_mask(16)  # labels {1, 2}, no zeros anywhere
    assert set(np.unique(mask)) == {1, 2}

    plot_cellpose4_output(batch, [mask], [_rgb_flow(16)], figuresize=2, nr=1)

    fig = _figures()[0]
    assert sorted(t.get_text() for t in fig.axes[2].texts) == ["1", "2"]


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


def test_plot_organelle_output_spots_otsu_three_panels():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=1, size=48)
    mask = _label_mask(48, n=3)
    settings = {
        "organelle_morphology": "spots",
        "organelle_method": "otsu",
        "organelle_tophat_radius": 3,
    }

    out = plot_organelle_output(imgs, [mask], settings, figuresize=2, nr=1)

    assert out is None
    figs = _figures()
    assert len(figs) == 1
    fig = figs[0]
    assert len(fig.axes) == 3
    assert _titles(fig) == [
        "Organelle channel (spots/otsu)",
        "Mask (3 objects)",
        "Top-hat filtered (r=3)",
    ]
    np.testing.assert_allclose(_img_array(fig.axes[0]), imgs[0])
    np.testing.assert_array_equal(_img_array(fig.axes[1]), mask)
    # diagnostic panel is a same-shaped 2-D image, not the raw channel
    diag = _img_array(fig.axes[2])
    assert diag.shape == imgs[0].shape
    assert not np.allclose(diag, imgs[0])
    # every foreground label annotated, background never annotated
    assert sorted(t.get_text() for t in fig.axes[1].texts) == ["1", "2", "3"]
    assert [len(a.texts) for a in fig.axes] == [0, 3, 0]
    # all three panels have their frames/ticks switched off
    assert [a.axison for a in fig.axes] == [False, False, False]


def test_plot_organelle_output_defaults_to_spots_otsu_when_settings_empty():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=1, size=48)
    plot_organelle_output(imgs, [_label_mask(48, n=2)], {}, figuresize=2, nr=1)

    fig = _figures()[0]
    assert fig.axes[0].get_title() == "Organelle channel (spots/otsu)"
    # default organelle_tophat_radius is 5
    assert fig.axes[2].get_title() == "Top-hat filtered (r=5)"
    assert fig.axes[1].get_title() == "Mask (2 objects)"


def test_plot_organelle_output_counts_all_labels_when_mask_has_no_background():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=1, size=16)
    mask = _full_mask(16)

    plot_organelle_output(imgs, [mask], {"organelle_morphology": "irregular"},
                          figuresize=2, nr=1)

    fig = _figures()[0]
    # `0 in mask` is False here -> no background subtraction from the count
    assert fig.axes[1].get_title() == "Mask (2 objects)"
    assert sorted(t.get_text() for t in fig.axes[1].texts) == ["1", "2"]
    # irregular morphology routes to the gaussian diagnostic
    assert fig.axes[0].get_title() == "Organelle channel (irregular/otsu)"
    assert fig.axes[2].get_title().startswith("Gaussian smoothed")


def test_plot_organelle_output_without_object_numbers_draws_no_text():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=1, size=48)
    plot_organelle_output(imgs, [_label_mask(48, n=3)],
                          {"organelle_morphology": "spots", "organelle_method": "otsu"},
                          figuresize=2, nr=1, print_object_number=False)

    fig = _figures()[0]
    assert fig.axes[1].get_title() == "Mask (3 objects)"
    assert all(len(a.texts) == 0 for a in fig.axes)


def test_plot_organelle_output_stops_at_the_smallest_of_masks_nr_and_batch():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=2, size=32)          # batch limit = 2
    masks = [_label_mask(32, n=2) for _ in range(3)]  # mask limit = 3
    plot_organelle_output(imgs, masks, {}, figuresize=2, nr=5)  # nr limit = 5

    figs = _figures()
    assert len(figs) == 2  # min(3, 5, 2)
    for fig, img in zip(figs, imgs):
        np.testing.assert_allclose(_img_array(fig.axes[0]), img)


def test_plot_organelle_output_with_no_masks_draws_nothing():
    from spacr.plot import plot_organelle_output

    imgs = _organelle_batch(n=2, size=16)
    assert plot_organelle_output(imgs, [], {}, figuresize=2, nr=3) is None
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# plot_masks
# ---------------------------------------------------------------------------

def test_plot_masks_png_file_type_unwraps_the_first_flow_element():
    from spacr.plot import plot_masks

    batch = _batch(n=1, size=32, chans=2)
    mask = _label_mask(32, n=3)
    flow_rgb = _rgb_flow(32, seed=7)
    # cellpose PNG runs hand back [[ [rgb, dP, cellprob] ]]: the outer list is
    # stripped by `flows = flows[0]`, then file_type='png' takes f[0].
    flows = [[[flow_rgb, np.zeros((2, 32, 32), np.float32), "not-an-array"]]]

    assert plot_masks(batch, [mask], flows, figuresize=2, nr=1, file_type="png") is None

    fig = _figures()[0]
    assert _titles(fig) == ["Image - Channel0", "Image - Channel1", "Mask", "Flow"]
    # the flow panel shows f[0]; without the unwrap imshow would have raised
    np.testing.assert_array_equal(_img_array(fig.axes[3]), flow_rgb)
    assert sorted(t.get_text() for t in fig.axes[2].texts) == ["1", "2", "3"]


def test_plot_masks_expands_a_single_3d_image_and_wraps_bare_arrays():
    from spacr.plot import plot_masks

    single = _batch(n=1, size=16, chans=1)[0]       # (H, W, C) -> expanded
    mask = _label_mask(16, n=2)                      # bare ndarray -> wrapped
    flow = _rgb_flow(16, seed=9)                     # bare ndarray -> wrapped

    plot_masks(single, mask, flow, figuresize=2, nr=1)

    figs = _figures()
    assert len(figs) == 1
    fig = figs[0]
    assert len(fig.axes) == 3
    np.testing.assert_allclose(_img_array(fig.axes[0]), single[..., 0])
    np.testing.assert_array_equal(_img_array(fig.axes[1]), mask)
    np.testing.assert_array_equal(_img_array(fig.axes[2]), flow)
    assert sorted(t.get_text() for t in fig.axes[1].texts) == ["1", "2"]


def test_plot_masks_nr_zero_creates_no_figure():
    from spacr.plot import plot_masks

    batch = _batch(n=1, size=16, chans=1)
    assert plot_masks(batch, _label_mask(16, n=2), _rgb_flow(16), nr=0) is None
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# _plot_4D_arrays
# ---------------------------------------------------------------------------

def _write_npz(path, arr):
    np.savez(path, data=arr)


def test_plot_4D_arrays_multichannel_plots_one_panel_per_channel(tmp_path):
    from spacr.plot import _plot_4D_arrays

    stack = np.random.default_rng(5).random((2, 16, 16, 3)).astype(np.float32)
    _write_npz(tmp_path / "field.npz", stack)

    assert _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=1, nr=2) is None

    figs = _figures()
    assert len(figs) == 2  # nr=2 images out of the 2 in the stack
    for i, fig in enumerate(figs):
        assert len(fig.axes) == 3
        assert _titles(fig) == ["Channel 0", "Channel 1", "Channel 2"]
        assert [a.axison for a in fig.axes] == [False, False, False]
        for c, ax in enumerate(fig.axes):
            np.testing.assert_allclose(_img_array(ax), stack[i][:, :, c])
            assert ax.title.get_size() == pytest.approx(24)


def test_plot_4D_arrays_single_channel_uses_one_axes(tmp_path):
    from spacr.plot import _plot_4D_arrays

    stack = np.random.default_rng(6).random((1, 16, 16, 1)).astype(np.float32)
    _write_npz(tmp_path / "one_channel.npz", stack)

    _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=1, nr=1, cmap="gray")

    figs = _figures()
    assert len(figs) == 1
    fig = figs[0]
    assert len(fig.axes) == 1  # axs was wrapped in a list, not indexed as an array
    assert fig.axes[0].get_title() == "Channel 0"
    np.testing.assert_allclose(_img_array(fig.axes[0]), stack[0][:, :, 0])
    assert fig.axes[0].get_images()[0].get_cmap().name == "gray"


def test_plot_4D_arrays_nr_npz_caps_files_and_ignores_non_npz(tmp_path):
    from spacr.plot import _plot_4D_arrays

    stacks = {}
    for i in range(3):
        arr = np.full((1, 8, 8, 1), float(i), dtype=np.float32)
        stacks[f"s{i}.npz"] = arr
        _write_npz(tmp_path / f"s{i}.npz", arr)
    # decoys that must never be opened by np.load
    (tmp_path / "readme.txt").write_text("not a stack")
    (tmp_path / "field.tif").write_bytes(b"\x00\x01")

    _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=2, nr=1)

    figs = _figures()
    assert len(figs) == 2  # 2 of the 3 .npz files, 1 image each
    drawn = {float(_img_array(f.axes[0])[0, 0]) for f in figs}
    assert drawn.issubset({0.0, 1.0, 2.0})
    assert len(drawn) == 2  # two *different* files were sampled


def test_plot_4D_arrays_empty_directory_plots_nothing(tmp_path):
    from spacr.plot import _plot_4D_arrays

    (tmp_path / "only.txt").write_text("nothing to plot")
    assert _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=3, nr=1) is None
    assert plt.get_fignums() == []


def test_plot_4D_arrays_nr_larger_than_stack_is_clamped(tmp_path):
    from spacr.plot import _plot_4D_arrays

    stack = np.zeros((2, 8, 8, 2), dtype=np.float32)
    _write_npz(tmp_path / "small.npz", stack)

    _plot_4D_arrays(str(tmp_path), figuresize=2, nr_npz=1, nr=99)

    assert len(_figures()) == 2  # min(99, num_images=2)
    assert os.path.exists(tmp_path / "small.npz")
