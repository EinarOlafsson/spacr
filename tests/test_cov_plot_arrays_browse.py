"""Coverage for the array-browsing block of ``spacr.plot`` (lines ~960-1239).

Covers:
  * ``_get_colours_merged``      - outline colour ordering table
  * ``plot_images_and_arrays``   - multi-folder image/mask pairing + overlay
  * ``_filter_objects_in_plot``  - area filtering + multi-object cell removal
  * ``plot_arrays``              - .npy / .npz / directory browsing

Everything runs headless on the Agg backend; no file is read that the test
did not just write, and every assertion inspects real figure/axes/array
state rather than merely "it did not raise".
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Start and end every test with a clean figure manager."""
    plt.close("all")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_tif(path, arr):
    import tifffile
    tifffile.imwrite(str(path), arr)


def _gradient_image(shape=(64, 64), n_levels=4000):
    """uint16 image with many (> any test threshold) distinct values."""
    h, w = shape
    return (np.arange(h * w).reshape(h, w) % n_levels).astype(np.uint16)


def _label_image(shape=(64, 64), with_single_pixel=False):
    """Small int label mask: two round objects (+ optional 1-px speck).

    Round rather than square on purpose: ``find_contours`` is run on the
    per-region bounding-box crop, and a solid rectangle fills its own bbox
    completely, so there is no 0.75 level crossing to trace.
    """
    m = np.zeros(shape, dtype=np.uint16)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    m[(yy - 14) ** 2 + (xx - 14) ** 2 <= 8 ** 2] = 1
    m[(yy - 40) ** 2 + (xx - 40) ** 2 <= 12 ** 2] = 2
    if with_single_pixel:
        m[60, 60] = 3
    return m


def _make_pair_folders(tmp_path, names=("f1",), with_png=True,
                       with_single_pixel=False):
    """Build image/mask(/png) sibling folders sharing base filenames."""
    img_dir = tmp_path / "arrays"
    mask_dir = tmp_path / "masks"
    png_dir = tmp_path / "pngs"
    img_dir.mkdir()
    mask_dir.mkdir()
    if with_png:
        png_dir.mkdir()

    from PIL import Image
    payload = {}
    for name in names:
        img = _gradient_image()
        mask = _label_image(with_single_pixel=with_single_pixel)
        np.save(img_dir / f"{name}.npy", img)
        _write_tif(mask_dir / f"{name}.tif", mask)
        if with_png:
            Image.fromarray((mask * 60).astype(np.uint8)).save(png_dir / f"{name}.png")
        payload[name] = {"image": img, "mask": mask}

    folders = [str(img_dir), str(mask_dir)]
    if with_png:
        folders.append(str(png_dir))
    return folders, payload


def _multi_object_stack():
    """(60, 60, 3) int32 stack: cell/nucleus/pathogen label planes.

    cell 1 -> 2 nuclei, 1 pathogen   (multinucleated)
    cell 2 -> 1 nucleus,  2 pathogens (multi-infected)
    cell 3 -> 1 nucleus,  1 pathogen  (clean; must survive every filter)
    """
    size = 60
    cell = np.zeros((size, size), dtype=np.int32)
    cell[2:18, 2:18] = 1
    cell[22:38, 22:38] = 2
    cell[42:58, 42:58] = 3

    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[4:8, 4:8] = 1
    nucleus[12:16, 12:16] = 2
    nucleus[24:28, 24:28] = 3
    nucleus[44:48, 44:48] = 4

    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[10:12, 4:8] = 1
    pathogen[30:34, 24:28] = 2
    pathogen[30:34, 32:36] = 3
    pathogen[50:54, 50:54] = 4

    return np.stack([cell, nucleus, pathogen], axis=-1)


def _labels(plane):
    return set(int(v) for v in np.unique(plane))


# ---------------------------------------------------------------------------
# _get_colours_merged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order,expected", [
    ("rgb", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ("bgr", [[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    ("gbr", [[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    ("rbg", [[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    ("not-a-format", [[1, 0, 0], [0, 0, 1], [0, 1, 0]]),  # fallback == rbg
    (None, [[1, 0, 0], [0, 0, 1], [0, 1, 0]]),            # fallback == rbg
])
def test_get_colours_merged_exact_tables(order, expected):
    from spacr.plot import _get_colours_merged

    out = _get_colours_merged(order)
    assert out == expected
    # every entry is a pure primary and the three are a permutation basis.
    assert sorted(tuple(c) for c in out) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]


# ---------------------------------------------------------------------------
# plot_images_and_arrays
# ---------------------------------------------------------------------------

def test_plot_images_and_arrays_pairs_mask_with_normalised_image(tmp_path, capsys):
    """.npy intensity + .tif mask are paired; .png files are skipped."""
    from spacr.plot import plot_images_and_arrays

    folders, payload = _make_pair_folders(tmp_path, names=("f1",))
    # A file present in only one folder must be dropped by the all-folders filter.
    np.save(os.path.join(folders[0], "orphan.npy"), _gradient_image())

    out = plot_images_and_arrays(folders, lower_percentile=2, upper_percentile=98,
                                 threshold=10, randomize=False)
    assert out is None

    figs = plt.get_fignums()
    assert len(figs) == 1, "exactly one paired figure expected (orphan dropped)"
    axes = plt.figure(figs[0]).axes
    assert len(axes) == 2
    assert axes[0].get_title() == "f1 - Mask"
    assert axes[1].get_title() == "f1 - Normalized Image"
    # No overlay requested -> no contour lines drawn.
    assert axes[1].lines == [] or len(axes[1].lines) == 0

    img = payload["f1"]["image"]
    p2, p98 = np.percentile(img, (2, 98))
    expected = np.clip((img - p2) / (p98 - p2), 0, 1)
    shown = np.asarray(axes[1].images[0].get_array())
    assert np.allclose(shown, expected)
    assert shown.min() >= 0.0 and shown.max() <= 1.0

    shown_mask = np.asarray(axes[0].images[0].get_array())
    assert np.array_equal(shown_mask, payload["f1"]["mask"])
    # 'Overlay will only work...' banner is only printed when overlay=True.
    assert "Overlay will only work" not in capsys.readouterr().out


def test_plot_images_and_arrays_overlay_draws_contours_and_skips_tiny_regions(tmp_path, capsys):
    from spacr.plot import plot_images_and_arrays

    folders, _ = _make_pair_folders(tmp_path, names=("f1",), with_single_pixel=True)

    plot_images_and_arrays(folders, threshold=10, overlay=True, randomize=False)

    assert "Overlay will only work on the first two folders" in capsys.readouterr().out

    axes = plt.figure(plt.get_fignums()[0]).axes
    lines = list(axes[1].lines)
    assert len(lines) >= 2, "one contour per >=2x2 object expected"
    for ln in lines:
        assert ln.get_color() == "magenta"
        assert ln.get_linewidth() == 2

    # The 1-px speck at (60, 60) fails the `region.image.shape >= 2` guard,
    # so no contour vertex may land anywhere near it.
    for ln in lines:
        xs, ys = ln.get_xdata(), ln.get_ydata()
        assert not np.any((np.abs(np.asarray(xs) - 60) < 1.5) &
                          (np.abs(np.asarray(ys) - 60) < 1.5))


def test_plot_images_and_arrays_max_nr_limits_plotted_groups(tmp_path):
    from spacr.plot import plot_images_and_arrays

    folders, _ = _make_pair_folders(tmp_path, names=("f1", "f2", "f3"))

    plot_images_and_arrays(folders, threshold=10, max_nr=2, randomize=False)

    assert len(plt.get_fignums()) == 2
    titles = {plt.figure(n).axes[0].get_title() for n in plt.get_fignums()}
    assert titles.issubset({"f1 - Mask", "f2 - Mask", "f3 - Mask"})
    assert len(titles) == 2


def test_plot_images_and_arrays_randomize_shuffles_before_truncating(tmp_path, monkeypatch):
    """randomize=True shuffles the key order, then max_nr slices the result."""
    import spacr.plot as P

    folders, _ = _make_pair_folders(tmp_path, names=("f1", "f2", "f3"))

    seen = {}

    def fake_shuffle(seq):
        seen["before"] = [k for k, _ in seq]
        seq.reverse()

    monkeypatch.setattr(P.random, "shuffle", fake_shuffle)

    P.plot_images_and_arrays(folders, threshold=10, max_nr=1.0, randomize=True)

    assert "before" in seen, "randomize=True must call random.shuffle"
    assert sorted(seen["before"]) == ["f1", "f2", "f3"]
    assert len(plt.get_fignums()) == 1
    # after reverse() the last pre-shuffle key becomes the only survivor.
    expected_key = seen["before"][-1]
    assert plt.figure(plt.get_fignums()[0]).axes[0].get_title() == f"{expected_key} - Mask"


def test_plot_images_and_arrays_no_intensity_image_plots_nothing(tmp_path):
    """Two label-only folders never populate image_data -> no figure."""
    from spacr.plot import plot_images_and_arrays

    a = tmp_path / "m1"
    b = tmp_path / "m2"
    a.mkdir()
    b.mkdir()
    np.save(a / "f1.npy", _label_image())
    _write_tif(b / "f1.tif", _label_image())

    plot_images_and_arrays([str(a), str(b)], threshold=10, randomize=False)

    assert plt.get_fignums() == []


def test_plot_images_and_arrays_extensions_argument_filters_candidates(tmp_path):
    """Restricting extensions can leave a key present in only one folder."""
    from spacr.plot import plot_images_and_arrays

    folders, _ = _make_pair_folders(tmp_path, names=("f1",), with_png=False)

    # masks are .tif only, so '.npy'-only scanning breaks the pairing.
    plot_images_and_arrays(folders, threshold=10, extensions=[".npy"], randomize=False)
    assert plt.get_fignums() == []

    # ...while allowing both extensions restores it.
    plot_images_and_arrays(folders, threshold=10, extensions=[".npy", ".tif"],
                           randomize=False)
    assert len(plt.get_fignums()) == 1


# ---------------------------------------------------------------------------
# _filter_objects_in_plot
# ---------------------------------------------------------------------------

def test_filter_objects_in_plot_none_min_max_keeps_every_object(capsys):
    """filter_min_max=None falls back to the wide-open default range."""
    from spacr.plot import _filter_objects_in_plot

    stack = _multi_object_stack()
    out = _filter_objects_in_plot(stack.copy(), cell_mask_dim=0, nucleus_mask_dim=1,
                                  pathogen_mask_dim=2, mask_dims=[0, 1, 2],
                                  filter_min_max=None,
                                  nuclei_limit=True, pathogen_limit=True)

    assert out.shape == stack.shape
    assert np.array_equal(out, stack), "nothing may be removed with no limits"
    assert _labels(out[:, :, 0]) == {0, 1, 2, 3}
    assert _labels(out[:, :, 1]) == {0, 1, 2, 3, 4}
    assert _labels(out[:, :, 2]) == {0, 1, 2, 3, 4}

    printed = capsys.readouterr().out
    assert "removed 0 cells" in printed
    assert "removed 0 nucleus" in printed
    assert "removed 0 pathogens" in printed


def test_filter_objects_in_plot_area_filter_drops_small_cells(capsys):
    from spacr.plot import _filter_objects_in_plot

    stack = _multi_object_stack()
    # shrink cell 3 to 36 px so a min-area of 100 excludes it.
    stack[42:58, 42:58, 0] = 0
    stack[42:48, 42:48, 0] = 3
    stack[:, :, 1][stack[:, :, 1] == 4] = 0
    stack[42:44, 42:44, 1] = 4
    stack[:, :, 2][stack[:, :, 2] == 4] = 0
    stack[46:48, 46:48, 2] = 4

    out = _filter_objects_in_plot(stack.copy(), cell_mask_dim=0, nucleus_mask_dim=1,
                                  pathogen_mask_dim=2, mask_dims=[0, 1, 2],
                                  filter_min_max=[[100, 10 ** 8],
                                                  [1, 10 ** 8],
                                                  [1, 10 ** 8]],
                                  nuclei_limit=True, pathogen_limit=True)

    assert _labels(out[:, :, 0]) == {0, 1, 2}, "36-px cell 3 must be filtered out"
    # the filter only touches the plane it is applied to.
    assert 4 in _labels(out[:, :, 1])
    assert 4 in _labels(out[:, :, 2])
    assert "removed 1 cells" in capsys.readouterr().out


def test_filter_objects_in_plot_both_limits_false_removes_multiobject_cells():
    """Both limits off -> every cell holding >1 nucleus OR >1 pathogen goes."""
    from spacr.plot import _filter_objects_in_plot

    stack = _multi_object_stack()
    out = _filter_objects_in_plot(stack.copy(), cell_mask_dim=0, nucleus_mask_dim=1,
                                  pathogen_mask_dim=2, mask_dims=[0, 1, 2],
                                  filter_min_max=None,
                                  nuclei_limit=False, pathogen_limit=False)

    assert _labels(out[:, :, 0]) == {0, 3}, "only the clean cell 3 survives"
    # nuclei/pathogens of the removed cells are cleared too.
    assert _labels(out[:, :, 1]) == {0, 4}
    assert _labels(out[:, :, 2]) == {0, 4}
    # cell 3's own objects are untouched and still in place.
    assert out[44, 44, 1] == 4
    assert out[50, 50, 2] == 4


@pytest.mark.xfail(strict=True,
                   reason="BUG: nuclei_limit/pathogen_limit pass swapped "
                          "object_dim to _remove_multiobject_cells")
def test_nuclei_limit_false_removes_the_multinucleated_cell():
    """nuclei_limit=False documents 'do not keep multinucleated cells'.

    Cell 1 has two nuclei, cell 2 has two pathogens. With only nuclei_limit
    off, cell 1 is the one that must go. plot.py passes
    ``object_dim=pathogen_mask_dim`` here, so cell 2 is removed instead.
    """
    from spacr.plot import _filter_objects_in_plot

    stack = _multi_object_stack()
    out = _filter_objects_in_plot(stack.copy(), cell_mask_dim=0, nucleus_mask_dim=1,
                                  pathogen_mask_dim=2, mask_dims=[0, 1, 2],
                                  filter_min_max=None,
                                  nuclei_limit=False, pathogen_limit=True)

    assert _labels(out[:, :, 0]) == {0, 2, 3}


@pytest.mark.xfail(strict=True,
                   reason="BUG: nuclei_limit/pathogen_limit pass swapped "
                          "object_dim to _remove_multiobject_cells")
def test_pathogen_limit_false_removes_the_multiinfected_cell():
    """pathogen_limit=False must drop the multi-infected cell (cell 2).

    plot.py passes ``object_dim=nucleus_mask_dim`` here, so the
    multinucleated cell 1 is removed instead.
    """
    from spacr.plot import _filter_objects_in_plot

    stack = _multi_object_stack()
    out = _filter_objects_in_plot(stack.copy(), cell_mask_dim=0, nucleus_mask_dim=1,
                                  pathogen_mask_dim=2, mask_dims=[0, 1, 2],
                                  filter_min_max=None,
                                  nuclei_limit=True, pathogen_limit=False)

    assert _labels(out[:, :, 0]) == {0, 1, 3}


# ---------------------------------------------------------------------------
# plot_arrays
# ---------------------------------------------------------------------------

def _rand_stack(shape=(32, 32, 3), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5000, size=shape).astype(np.uint16)


def test_plot_arrays_single_npy_one_subplot_per_channel(tmp_path, capsys):
    from spacr.plot import plot_arrays
    from spacr.utils import normalize_to_dtype

    arr = _rand_stack()
    path = tmp_path / "stack3.npy"
    np.save(path, arr)

    plot_arrays(str(path), figuresize=6, cmap="gray", normalize=True, q1=1, q2=99)

    assert f"Image path: {path}" in capsys.readouterr().out
    figs = plt.get_fignums()
    assert len(figs) == 1
    axes = plt.figure(figs[0]).axes
    assert len(axes) == 3

    expected = normalize_to_dtype(array=arr, p1=1, p2=99)
    for c, ax in enumerate(axes):
        assert ax.get_title() == f"Channel {c}"
        assert not ax.axison
        shown = np.asarray(ax.images[0].get_array())
        assert shown.shape == (32, 32)
        assert np.array_equal(shown, expected[:, :, c])
    # normalisation really changed the data.
    assert not np.array_equal(expected, arr)


def test_plot_arrays_single_channel_3d_wraps_axes_in_list(tmp_path):
    """array_nr == 1 -> plt.subplots returns a bare Axes that must be wrapped."""
    from spacr.plot import plot_arrays

    arr = _rand_stack(shape=(24, 24, 1))
    path = tmp_path / "one_channel.npy"
    np.save(path, arr)

    plot_arrays(str(path), normalize=True)

    axes = plt.figure(plt.get_fignums()[0]).axes
    assert len(axes) == 1
    assert axes[0].get_title() == "Channel 0"
    assert np.asarray(axes[0].images[0].get_array()).shape == (24, 24)


def test_plot_arrays_npz_uses_first_key_and_first_batch_item(tmp_path):
    from spacr.plot import plot_arrays

    first = _rand_stack(shape=(16, 16, 2), seed=1)
    second = _rand_stack(shape=(16, 16, 2), seed=2)
    other = _rand_stack(shape=(16, 16, 2), seed=3)
    path = tmp_path / "batch.npz"
    np.savez(path, images=np.stack([first, second]), extra=other[None, ...])

    plot_arrays(str(path), normalize=False)

    with np.load(path) as data:
        expected = data[list(data.keys())[0]][0]
    assert np.array_equal(expected, first)

    axes = plt.figure(plt.get_fignums()[0]).axes
    assert len(axes) == 2
    for c, ax in enumerate(axes):
        assert np.array_equal(np.asarray(ax.images[0].get_array()), first[:, :, c])
    # the second key must not have been plotted.
    assert not np.array_equal(np.asarray(axes[0].images[0].get_array()), other[:, :, 0])


def test_plot_arrays_directory_samples_nr_arrays(tmp_path, capsys):
    from spacr.plot import plot_arrays

    src = tmp_path / "arrays"
    src.mkdir()
    names = ["a.npy", "b.npy", "c.npy"]
    for n in names:
        np.save(src / n, _rand_stack(shape=(16, 16, 2)))
    (src / "notes.txt").write_text("ignore me")
    np.savez(src / "d.npz", images=_rand_stack(shape=(16, 16, 2))[None, ...])

    plot_arrays(str(src), nr=2, normalize=False)

    printed = capsys.readouterr().out
    plotted = [ln.split("Image path: ")[1] for ln in printed.splitlines()
               if ln.startswith("Image path: ")]
    assert len(plotted) == 2
    assert len(set(plotted)) == 2
    allowed = {str(src / n) for n in names} | {str(src / "d.npz")}
    assert set(plotted).issubset(allowed)
    assert str(src / "notes.txt") not in plotted
    assert len(plt.get_fignums()) == 2


def test_plot_arrays_directory_nr_larger_than_available(tmp_path):
    from spacr.plot import plot_arrays

    src = tmp_path / "few"
    src.mkdir()
    for n in ("a.npy", "b.npy"):
        np.save(src / n, _rand_stack(shape=(12, 12, 2)))

    plot_arrays(str(src), nr=25, normalize=False)

    assert len(plt.get_fignums()) == 2


def test_plot_arrays_2d_array_falls_back_to_single_axes(tmp_path):
    """A 2-D array takes the non-3-D branch: one axes titled 'Channel 0'."""
    from spacr.plot import plot_arrays

    arr = (np.arange(32 * 32).reshape(32, 32) % 500).astype(np.uint16)
    path = tmp_path / "flat.npy"
    np.save(path, arr)

    plot_arrays(str(path), normalize=False, cmap="viridis")

    figs = plt.get_fignums()
    assert len(figs) == 1
    axes = plt.figure(figs[0]).axes
    assert len(axes) == 1
    assert axes[0].get_title() == "Channel 0"
    assert not axes[0].axison
    assert np.array_equal(np.asarray(axes[0].images[0].get_array()), arr)


@pytest.mark.xfail(strict=True,
                   reason="BUG: plot_arrays(normalize=True) crashes on 2-D "
                          "arrays - normalize_to_dtype indexes array.shape[2]")
def test_plot_arrays_2d_array_with_default_normalisation(tmp_path):
    """plot_arrays explicitly supports 2-D arrays, but normalize defaults to
    True and normalize_to_dtype requires an (H, W, C) stack, so the documented
    2-D path raises IndexError for every caller that keeps the default."""
    from spacr.plot import plot_arrays

    arr = (np.arange(32 * 32).reshape(32, 32) % 500).astype(np.uint16)
    path = tmp_path / "flat_norm.npy"
    np.save(path, arr)

    plot_arrays(str(path), normalize=True)

    assert len(plt.get_fignums()) == 1
    assert plt.figure(plt.get_fignums()[0]).axes[0].get_title() == "Channel 0"
