"""Coverage for spacr.plot's merged-stack overlay + image-grid helpers.

Covers ``_normalize_and_outline``, ``_plot_merged_plot``, ``plot_merged`` and
``_plot_images_on_grid`` (spacr/plot.py ~1239-1498).

Everything here is CPU-only, offline and synthetic: 48x48 uint16 "merged"
stacks laid out the way ``preprocess_generate_masks`` writes them
(intensity channels first, then cell/nucleus/pathogen label masks) and tiny
PNG/TIFF files read back through OpenCV.

``plt.show`` is replaced with a recorder in an autouse fixture so no call can
ever block, and every figure is closed after each test.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import matplotlib.pyplot as plt

# The functions under test import spacr.utils lazily inside their bodies; pull
# it in once at collection time so no single test pays that import cost.
import spacr.utils  # noqa: F401


H = W = 48


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_blocking_show(monkeypatch):
    """Replace plt.show with a recorder and close all figures afterwards."""
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    plt.close("all")  # start from a clean slate: figure counts are asserted
    yield calls
    plt.close("all")


@pytest.fixture
def show_calls(_no_blocking_show):
    """The list plt.show appends to (one entry per call)."""
    return _no_blocking_show


# ---------------------------------------------------------------------------
# synthetic stack builders
# ---------------------------------------------------------------------------

def _merged_stack(big_second_cell: bool = False, n_pathogens: int = 1):
    """A (H, W, 6) uint16 merged stack.

    Channels 0-2 are intensity gradients (never constant, so the per-channel
    min/max normalisation in ``_plot_merged_plot`` is well defined).
    Channel 3 = cell mask (2 labels), 4 = nucleus mask (1 per cell),
    5 = pathogen mask (only inside cell 1, so cell 2 is "uninfected").
    """
    yy, xx = np.mgrid[:H, :W]
    stack = np.zeros((H, W, 6), dtype=np.uint16)
    stack[:, :, 0] = (yy * 37 + xx * 11 + 5).astype(np.uint16)
    stack[:, :, 1] = (xx * 53 + 7).astype(np.uint16)
    stack[:, :, 2] = ((yy + xx) * 23 + 3).astype(np.uint16)

    cell = np.zeros((H, W), dtype=np.uint16)
    nucleus = np.zeros((H, W), dtype=np.uint16)
    pathogen = np.zeros((H, W), dtype=np.uint16)

    cell[(yy - 12) ** 2 + (xx - 12) ** 2 <= 9 ** 2] = 1
    nucleus[(yy - 12) ** 2 + (xx - 12) ** 2 <= 4 ** 2] = 1

    r2 = 13 if big_second_cell else 9
    cell[(yy - 34) ** 2 + (xx - 34) ** 2 <= r2 ** 2] = 2
    nucleus[(yy - 34) ** 2 + (xx - 34) ** 2 <= 4 ** 2] = 2

    for pid in range(1, n_pathogens + 1):
        cy = 9 + 4 * (pid - 1)
        pathogen[(yy - cy) ** 2 + (xx - 15) ** 2 <= 2 ** 2] = pid

    stack[:, :, 3] = cell
    stack[:, :, 4] = nucleus
    stack[:, :, 5] = pathogen
    return stack


def _plot_merge_settings(**overrides):
    """Default plot_merge settings re-pointed at the 6-channel test stack."""
    from spacr.settings import set_default_plot_merge_settings

    settings = set_default_plot_merge_settings()
    settings.update(
        cell_mask_dim=3, nucleus_mask_dim=4, pathogen_mask_dim=5,
        overlay_chans=[0, 1, 2], outline_thickness=2, figuresize=4,
        verbose=False, nr=5,
    )
    settings.update(overrides)
    return settings


def _write_stacks(dirpath, n=2, **kwargs):
    for i in range(n):
        np.save(os.path.join(dirpath, f"plate1_A0{i + 1}_1.npy"),
                _merged_stack(**kwargs))
    return dirpath


def _titles(fig):
    return [ax.get_title() for ax in fig.axes]


# ---------------------------------------------------------------------------
# _normalize_and_outline
# ---------------------------------------------------------------------------

def test_normalize_and_outline_overlay_returns_rgb_image_and_one_outline_per_mask():
    from spacr.plot import _normalize_and_outline, _get_colours_merged

    colours = _get_colours_merged("rgb")
    stack = _merged_stack(n_pathogens=2)
    overlayed, image, outlines = _normalize_and_outline(
        image=stack, remove_background=False, normalize=True,
        normalization_percentiles=[2, 98], overlay=True,
        overlay_chans=[0, 1, 2], mask_dims=[3, 4, 5],
        outline_colors=colours, outline_thickness=2)

    # overlay=True keeps every channel (mask dims included) in `image`.
    assert image.shape == (H, W, 6)
    assert overlayed.shape == (H, W, 3)
    assert overlayed.dtype == np.float32
    assert len(outlines) == 3
    for outline in outlines:
        assert outline.shape == (H, W)
        assert outline.dtype == np.uint8
        assert set(np.unique(outline)) <= {0, 255}
    # The cell mask (2 labels) really produced an outline.
    assert outlines[0].max() == 255

    # Cell-outline pixels not shared with the nucleus/pathogen outlines are
    # painted with the first outline colour (red for 'rgb').
    only_cell = (outlines[0] > 0) & (outlines[1] == 0) & (outlines[2] == 0)
    assert only_cell.any()
    assert np.allclose(overlayed[only_cell], np.array(colours[0], dtype=np.float32))


def test_normalize_and_outline_without_overlay_drops_mask_channels():
    from spacr.plot import _normalize_and_outline, _get_colours_merged
    from spacr.utils import normalize_to_dtype

    stack = _merged_stack()
    expected = normalize_to_dtype(array=stack.copy(), p1=0, p2=100)[:, :, [0, 1, 2]]

    overlayed, image, outlines = _normalize_and_outline(
        image=stack, remove_background=False, normalize=False,
        normalization_percentiles=[2, 98], overlay=False,
        overlay_chans=[0, 1, 2], mask_dims=[3, 4, 5],
        outline_colors=_get_colours_merged("rgb"), outline_thickness=2)

    assert overlayed == []
    assert outlines == []
    assert image.shape == (H, W, 3)
    assert image.dtype == np.uint16
    np.testing.assert_array_equal(image, expected)


@pytest.mark.parametrize("normalize,expected", [(True, (2, 98)), (False, (0, 100))])
def test_normalize_and_outline_percentile_selection(monkeypatch, normalize, expected):
    """normalize=True forwards the configured percentiles; False forces 0/100."""
    import spacr.utils as U
    from spacr.plot import _normalize_and_outline, _get_colours_merged

    seen = {}
    real = U.normalize_to_dtype

    def spy(array, p1=2, p2=98, **kwargs):
        seen["p"] = (p1, p2)
        return real(array=array, p1=p1, p2=p2, **kwargs)

    monkeypatch.setattr(U, "normalize_to_dtype", spy)

    _normalize_and_outline(
        image=_merged_stack(), remove_background=False, normalize=normalize,
        normalization_percentiles=[2, 98], overlay=False,
        overlay_chans=[0, 1, 2], mask_dims=[3, 4, 5],
        outline_colors=_get_colours_merged("rgb"), outline_thickness=2)

    assert seen["p"] == expected


def test_normalize_and_outline_remove_background_zeroes_only_intensity_channels(monkeypatch):
    """remove_background zeroes sub-1st-percentile pixels, masks untouched."""
    import spacr.utils as U
    from spacr.plot import _normalize_and_outline, _get_colours_merged

    captured = {}

    def spy(array, p1=2, p2=98, **kwargs):
        captured["array"] = array.copy()
        return array

    monkeypatch.setattr(U, "normalize_to_dtype", spy)

    stack = _merged_stack()
    original = stack.copy()
    mask_dims = [3, 4, 5]

    _normalize_and_outline(
        image=stack, remove_background=True, normalize=True,
        normalization_percentiles=[2, 98], overlay=False,
        overlay_chans=[0, 1, 2], mask_dims=mask_dims,
        outline_colors=_get_colours_merged("rgb"), outline_thickness=2)

    backgrounds = np.percentile(original, 1, axis=(0, 1))
    expected = original.copy()
    for chan in range(original.shape[-1]):
        if chan not in mask_dims:
            below = original[:, :, chan] < backgrounds[chan]
            expected[:, :, chan][below] = 0

    got = captured["array"]
    np.testing.assert_array_equal(got, expected)
    # Non-trivial: some intensity pixels really were knocked out ...
    assert np.count_nonzero(got[:, :, 0] == 0) > 0
    assert np.count_nonzero(original[:, :, 0] == 0) == 0
    # ... while the label masks came through byte-identical.
    for dim in mask_dims:
        np.testing.assert_array_equal(got[:, :, dim], original[:, :, dim])


@pytest.mark.xfail(strict=True, reason=(
    "BUG: _normalize_and_outline percentile-normalises the mask channels too, "
    "so a mask holding exactly one object collapses to a constant image and "
    "_outline_and_overlay finds no contour -> that object is never outlined"))
def test_normalize_and_outline_single_object_mask_is_outlined():
    from spacr.plot import _normalize_and_outline, _get_colours_merged

    stack = _merged_stack(n_pathogens=1)
    assert len(np.unique(stack[:, :, 5])) == 2  # exactly one pathogen + background

    _, _, outlines = _normalize_and_outline(
        image=stack, remove_background=False, normalize=True,
        normalization_percentiles=[2, 98], overlay=True,
        overlay_chans=[0, 1, 2], mask_dims=[5],
        outline_colors=_get_colours_merged("rgb"), outline_thickness=2)

    assert outlines[0].max() > 0, "single-object mask produced an empty outline"


# ---------------------------------------------------------------------------
# _plot_merged_plot
# ---------------------------------------------------------------------------

def _gradient_image(c=3, h=24, w=24):
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros((h, w, c), dtype=np.uint16)
    for k in range(c):
        img[:, :, k] = (yy * (3 + k) + xx * (5 + k) + 1).astype(np.uint16)
    return img


def _label_stack(h=24, w=24):
    """(h, w, 2) stack: channel 0 has 2 objects, channel 1 has 1 object."""
    stack = np.zeros((h, w, 2), dtype=np.uint16)
    stack[2:8, 2:8, 0] = 1
    stack[12:18, 12:18, 0] = 2
    stack[3:6, 3:6, 1] = 1
    return stack


def _ring_outline(h=24, w=24, box=(2, 8, 2, 8)):
    out = np.zeros((h, w), dtype=np.uint8)
    r0, r1, c0, c1 = box
    out[r0:r1, c0:c1] = 255
    out[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = 0
    return out


def test_plot_merged_plot_overlay_axis_layout_and_titles():
    from spacr.plot import _plot_merged_plot, _get_colours_merged

    image = _gradient_image(c=3)
    stack = _label_stack()
    fig = _plot_merged_plot(
        overlay=True, image=image, stack=stack, mask_dims=[0, 1], figuresize=3,
        overlayed_image=np.zeros((24, 24, 3), dtype=np.float32),
        outlines=[_ring_outline()], cmap="inferno",
        outline_colors=_get_colours_merged("rgb"), print_object_number=False,
        mask_names=["Cell Mask", "Nucleus Mask"])

    # 3 channels + 2 masks + 1 overlay panel
    assert len(fig.axes) == 6
    assert _titles(fig) == [
        "Overlayed Image", "Channel 1", "Channel 2", "Channel 3",
        "Cell Mask - 2 objects", "Nucleus Mask - 1 object",
    ]


def test_plot_merged_plot_without_overlay_has_no_overlay_panel():
    from spacr.plot import _plot_merged_plot, _get_colours_merged

    fig = _plot_merged_plot(
        overlay=False, image=_gradient_image(c=2), stack=_label_stack(),
        mask_dims=[0, 1], figuresize=3, overlayed_image=[], outlines=[],
        cmap="inferno", outline_colors=_get_colours_merged("rgb"),
        print_object_number=False, mask_names=["Cell Mask", "Nucleus Mask"])

    assert len(fig.axes) == 4
    assert _titles(fig)[0] == "Channel 1"
    assert "Overlayed Image" not in _titles(fig)


def test_plot_merged_plot_falls_back_to_generic_mask_names():
    from spacr.plot import _plot_merged_plot, _get_colours_merged

    common = dict(overlay=False, image=_gradient_image(c=1), stack=_label_stack(),
                  mask_dims=[0, 1], figuresize=3, overlayed_image=[], outlines=[],
                  cmap="inferno", outline_colors=_get_colours_merged("rgb"),
                  print_object_number=False)

    fig_none = _plot_merged_plot(mask_names=None, **common)
    assert _titles(fig_none)[1:] == ["Mask 1 - 2 objects", "Mask 2 - 1 object"]

    fig_short = _plot_merged_plot(mask_names=["Cell Mask"], **common)
    assert _titles(fig_short)[1:] == ["Cell Mask - 2 objects", "Mask 2 - 1 object"]


def test_plot_merged_plot_paints_outlines_onto_each_channel():
    from spacr.plot import _plot_merged_plot, _get_colours_merged

    colours = _get_colours_merged("rgb")
    outline = _ring_outline()
    fig = _plot_merged_plot(
        overlay=False, image=_gradient_image(c=2), stack=_label_stack(),
        mask_dims=[0], figuresize=3, overlayed_image=[], outlines=[outline],
        cmap="inferno", outline_colors=colours, print_object_number=False,
        mask_names=["Cell Mask"])

    for ax in fig.axes[:2]:  # the two channel panels
        rgb = np.asarray(ax.images[0].get_array())
        assert rgb.shape == (24, 24, 3)
        painted = rgb[outline > 0]
        assert painted.shape == (20, 3)  # the 6x6 ring minus its 4x4 interior
        np.testing.assert_allclose(
            painted, np.broadcast_to(np.array(colours[0], dtype=float), painted.shape))
        # Off-outline pixels stay grey (r == g == b) and inside [0, 1].
        off = rgb[outline == 0]
        assert np.allclose(off[:, 0], off[:, 1]) and np.allclose(off[:, 1], off[:, 2])
        assert off.min() >= 0.0 and off.max() <= 1.0


def test_plot_merged_plot_print_object_number_labels_every_object():
    from spacr.plot import _plot_merged_plot, _get_colours_merged

    common = dict(overlay=False, image=_gradient_image(c=1), stack=_label_stack(),
                  mask_dims=[0, 1], figuresize=3, overlayed_image=[], outlines=[],
                  cmap="inferno", outline_colors=_get_colours_merged("rgb"),
                  mask_names=["Cell Mask", "Nucleus Mask"])

    fig = _plot_merged_plot(print_object_number=True, **common)
    cell_ax, nucleus_ax = fig.axes[1], fig.axes[2]
    assert sorted(t.get_text() for t in cell_ax.texts) == ["1", "2"]
    assert [t.get_text() for t in nucleus_ax.texts] == ["1"]
    # Labels are placed at the object centroids (6x6 block at rows/cols 2:8).
    x, y = cell_ax.texts[0].get_position()
    assert x == pytest.approx(4.5) and y == pytest.approx(4.5)

    fig_off = _plot_merged_plot(print_object_number=False, **common)
    assert len(fig_off.axes[1].texts) == 0


def test_plot_merged_plot_calls_show_once_and_returns_the_figure(show_calls):
    from spacr.plot import _plot_merged_plot, _get_colours_merged

    fig = _plot_merged_plot(
        overlay=False, image=_gradient_image(c=1), stack=_label_stack(),
        mask_dims=[0], figuresize=3, overlayed_image=[], outlines=[],
        cmap="inferno", outline_colors=_get_colours_merged("rgb"),
        print_object_number=False)

    assert len(show_calls) == 1
    assert fig in [plt.figure(n) for n in plt.get_fignums()]


# ---------------------------------------------------------------------------
# plot_merged
# ---------------------------------------------------------------------------

def test_plot_merged_plots_every_file_and_returns_none(tmp_path, show_calls):
    from spacr.plot import plot_merged

    src = _write_stacks(tmp_path, n=2)
    settings = _plot_merge_settings(nr=5)

    assert plot_merged(str(src), settings) is None
    assert len(show_calls) == 2
    assert len(plt.get_fignums()) == 2


def test_plot_merged_returns_last_figure_once_nr_is_exceeded(tmp_path, show_calls):
    from spacr.plot import plot_merged

    src = _write_stacks(tmp_path, n=3)
    settings = _plot_merge_settings(nr=1)

    fig = plot_merged(str(src), settings)

    assert fig is not None
    # Only `nr` figures were drawn even though 3 files were on disk.
    assert len(show_calls) == 1
    # 6 image channels + 3 masks + overlay panel.
    assert len(fig.axes) == 10
    assert _titles(fig)[0] == "Overlayed Image"


def test_plot_merged_removes_uninfected_cells_when_pathogen_limit_positive(tmp_path):
    from spacr.plot import plot_merged

    src = _write_stacks(tmp_path, n=2)
    settings = _plot_merge_settings(nr=1, pathogen_limit=10)

    fig = plot_merged(str(src), settings)

    # Cell 2 carries no pathogen -> _remove_noninfected drops it and its nucleus.
    titles = _titles(fig)
    assert "Cell Mask - 1 object" in titles
    assert "Nucleus Mask - 1 object" in titles
    assert "Pathogen Mask - 1 object" in titles


def test_plot_merged_keeps_uninfected_cells_when_pathogen_limit_zero(tmp_path):
    from spacr.plot import plot_merged

    src = _write_stacks(tmp_path, n=2)
    settings = _plot_merge_settings(nr=1, pathogen_limit=0)

    titles = _titles(plot_merged(str(src), settings))

    assert "Cell Mask - 2 objects" in titles
    assert "Nucleus Mask - 2 objects" in titles


def test_plot_merged_filter_min_max_drops_out_of_range_objects(tmp_path):
    from spacr.plot import plot_merged

    src = _write_stacks(tmp_path, n=2, big_second_cell=True)
    # Cell 1 ~253 px, cell 2 ~529 px -> only cell 1 survives the size window.
    settings = _plot_merge_settings(
        nr=1, pathogen_limit=0,
        filter_min_max=[[100, 400], [0, 100000], [0, 100000]])

    titles = _titles(plot_merged(str(src), settings))

    assert "Cell Mask - 1 object" in titles
    assert "Nucleus Mask - 2 objects" in titles


def test_plot_merged_without_pathogen_mask_dim_forces_pathogen_limit(tmp_path):
    from spacr.plot import plot_merged

    src = _write_stacks(tmp_path, n=2)
    settings = _plot_merge_settings(
        nr=1, pathogen_mask_dim=None, pathogen_limit=0,
        nuclei_limit=True, filter_min_max=None)

    fig = plot_merged(str(src), settings)

    assert settings["pathogen_limit"] is True
    titles = _titles(fig)
    # Only cell + nucleus masks are plotted now: 6 channels + 2 masks + overlay.
    assert len(fig.axes) == 9
    assert titles[-2:] == ["Cell Mask - 2 objects", "Nucleus Mask - 2 objects"]
    assert not any("Pathogen" in t for t in titles)


def test_plot_merged_without_overlay_drops_mask_channels_from_the_grid(tmp_path):
    from spacr.plot import plot_merged

    src = _write_stacks(tmp_path, n=2)
    settings = _plot_merge_settings(nr=1, overlay=False, print_object_number=False)

    fig = plot_merged(str(src), settings)

    # 3 intensity channels (mask dims removed) + 3 mask panels, no overlay panel.
    assert len(fig.axes) == 6
    assert _titles(fig)[:3] == ["Channel 1", "Channel 2", "Channel 3"]


@pytest.mark.parametrize("verbose", [True, False])
def test_plot_merged_verbose_controls_settings_display(tmp_path, monkeypatch, verbose):
    import spacr.plot as P

    shown = []
    monkeypatch.setattr(P, "display", lambda obj: shown.append(obj))

    src = _write_stacks(tmp_path, n=1)
    settings = _plot_merge_settings(nr=1, verbose=verbose)
    P.plot_merged(str(src), settings)

    assert shown == ([settings] if verbose else [])


@pytest.mark.xfail(strict=True, reason=(
    "BUG: plot_merged with nr=0 hits `return fig` before `fig` is ever "
    "assigned -> UnboundLocalError instead of simply plotting nothing"))
def test_plot_merged_with_nr_zero_plots_nothing(tmp_path):
    from spacr.plot import plot_merged

    src = _write_stacks(tmp_path, n=1)
    settings = _plot_merge_settings(nr=0)

    assert plot_merged(str(src), settings) is None
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# _plot_images_on_grid
# ---------------------------------------------------------------------------

def _write_rgb_png(path, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)
    # cv2 wrote BGR; _plot_images_on_grid converts back to RGB on read.
    return img[:, :, ::-1]


def _write_gray16_tif(path):
    import cv2

    img = (np.mgrid[:16, :16][0] * 4000).astype(np.uint16)
    assert cv2.imwrite(str(path), img)
    return img


def _write_three_rgb(tmp_path):
    paths, expected = [], []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        expected.append(_write_rgb_png(p, seed=i))
        paths.append(str(p))
    return paths, expected


def test_plot_images_on_grid_rgb_titles_scalebar_and_padding(tmp_path):
    from spacr.plot import _plot_images_on_grid

    paths, expected = _write_three_rgb(tmp_path)

    fig = _plot_images_on_grid(paths, channel_indices=[0, 1, 2], um_per_pixel=0.1,
                               scale_bar_length_um=5, fontsize=8,
                               show_filename=True)

    # ceil(sqrt(3)) = 2 cols, ceil(3/2) = 2 rows -> 4 axes, last one blank.
    assert len(fig.axes) == 4
    assert [ax.axison for ax in fig.axes] == [False, False, False, False]
    assert len(fig.axes[3].images) == 0
    assert fig.texts == []

    assert [ax.get_title() for ax in fig.axes[:3]] == [
        "img_0.png", "img_1.png", "img_2.png"]

    # uint8 -> /255 float32, BGR->RGB applied.
    arr = np.asarray(fig.axes[0].images[0].get_array())
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, expected[0].astype(np.float32) / 255.0)

    # scale bar: 5 um / 0.1 um-per-px = 50 px starting at x=10.
    (line,) = fig.axes[0].lines
    np.testing.assert_array_equal(line.get_xdata(), [10, 60])
    np.testing.assert_array_equal(line.get_ydata(), [6, 6])  # 16 px tall - 10


def test_plot_images_on_grid_channel_name_banner(tmp_path):
    from spacr.plot import _plot_images_on_grid

    paths, _ = _write_three_rgb(tmp_path)

    fig = _plot_images_on_grid(paths, channel_indices=[0, 1, 2], um_per_pixel=0.1,
                               fontsize=9,
                               channel_names=["red", "green", "blue", "extra"])

    # The 4th name falls off the end of channel_colors and defaults to white.
    assert [t.get_text() for t in fig.texts] == ["red", "green", "blue", "extra"]
    assert [t.get_color() for t in fig.texts] == ["red", "green", "blue", "white"]
    assert [round(t.get_position()[0], 3) for t in fig.texts] == [0.02, 0.07, 0.12, 0.17]
    assert all(t.get_position()[1] == 0.99 for t in fig.texts)
    assert all(t.get_fontsize() == 9 for t in fig.texts)


@pytest.mark.xfail(strict=True, reason=(
    "BUG: _plot_images_on_grid reuses the loop variable `i` for the "
    "channel_names loop, so `range(i + 1, len(axes))` is computed from the "
    "number of channel names instead of the number of images and the unused "
    "grid cells keep their visible axes"))
def test_plot_images_on_grid_pads_unused_cells_with_channel_names(tmp_path):
    from spacr.plot import _plot_images_on_grid

    paths, _ = _write_three_rgb(tmp_path)

    fig = _plot_images_on_grid(paths, channel_indices=[0, 1, 2], um_per_pixel=0.1,
                               channel_names=["c1", "c2", "c3", "c4"])

    assert len(fig.axes) == 4
    assert fig.axes[3].axison is False


def test_plot_images_on_grid_single_channel_index_is_grayscale(tmp_path):
    from spacr.plot import _plot_images_on_grid

    paths, expected = [], []
    for i in range(2):
        p = tmp_path / f"img_{i}.png"
        expected.append(_write_rgb_png(p, seed=10 + i))
        paths.append(str(p))

    fig = _plot_images_on_grid(paths, channel_indices=[1], um_per_pixel=0.5,
                               show_filename=False)

    assert len(fig.axes) == 2
    img = fig.axes[0].images[0]
    arr = np.asarray(img.get_array())
    assert arr.ndim == 2
    assert img.get_cmap().name == "gray"
    np.testing.assert_allclose(arr, expected[0][:, :, 1].astype(np.float32) / 255.0)
    assert fig.axes[0].get_title() == ""
    # 5 um default / 0.5 um-per-px = 10 px.
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [10, 20])


def test_plot_images_on_grid_two_channel_indices_are_averaged(tmp_path):
    from spacr.plot import _plot_images_on_grid

    p = tmp_path / "dual.png"
    expected = _write_rgb_png(p, seed=3)

    fig = _plot_images_on_grid([str(p), str(p)], channel_indices=[0, 2],
                               um_per_pixel=0.1)

    arr = np.asarray(fig.axes[0].images[0].get_array())
    assert arr.ndim == 2
    assert fig.axes[0].images[0].get_cmap().name == "gray"
    # np.mean produces float64 -> the uint8/uint16 rescale branches are skipped,
    # so the data stays on the 0-255 scale.
    np.testing.assert_allclose(
        arr, np.mean(expected[:, :, [0, 2]].astype(float), axis=2))
    assert arr.max() > 1.0


def test_plot_images_on_grid_uint16_grayscale_without_channel_indices(tmp_path):
    from spacr.plot import _plot_images_on_grid

    p = tmp_path / "gray16.tif"
    expected = _write_gray16_tif(p)

    fig = _plot_images_on_grid([str(p), str(p)], channel_indices=None,
                               um_per_pixel=0.1, show_filename=False)

    img = fig.axes[0].images[0]
    arr = np.asarray(img.get_array())
    assert arr.dtype == np.float32
    assert img.get_cmap().name == "gray"
    np.testing.assert_allclose(arr, expected.astype(np.float32) / 65535.0)
    assert fig.texts == []


def test_plot_images_on_grid_rgb_without_channel_indices_keeps_colour(tmp_path):
    from spacr.plot import _plot_images_on_grid

    p = tmp_path / "rgb.png"
    expected = _write_rgb_png(p, seed=7)

    fig = _plot_images_on_grid([str(p), str(p)], channel_indices=None,
                               um_per_pixel=0.1)

    img = fig.axes[0].images[0]
    arr = np.asarray(img.get_array())
    assert arr.shape == (16, 16, 3)
    assert img.cmap.name == plt.rcParams["image.cmap"]  # cmap=None -> default
    np.testing.assert_allclose(arr, expected.astype(np.float32) / 255.0)


@pytest.mark.parametrize("plot", [True, False])
def test_plot_images_on_grid_plot_flag_controls_show(tmp_path, show_calls, plot):
    from spacr.plot import _plot_images_on_grid

    p = tmp_path / "img.png"
    _write_rgb_png(p, seed=1)

    fig = _plot_images_on_grid([str(p), str(p)], channel_indices=[0, 1, 2],
                               um_per_pixel=0.1, plot=plot)

    assert len(show_calls) == (1 if plot else 0)
    assert fig.get_facecolor()[:3] == (0.0, 0.0, 0.0)


@pytest.mark.xfail(strict=True, reason=(
    "BUG: _plot_images_on_grid with a single image builds a 1x1 subplot grid, "
    "so plt.subplots returns a bare Axes and axes.flatten() raises "
    "AttributeError"))
def test_plot_images_on_grid_handles_a_single_image(tmp_path):
    from spacr.plot import _plot_images_on_grid

    p = tmp_path / "solo.png"
    _write_rgb_png(p, seed=2)

    fig = _plot_images_on_grid([str(p)], channel_indices=[0, 1, 2],
                               um_per_pixel=0.1)

    assert len(fig.axes) == 1
    assert len(fig.axes[0].images) == 1
