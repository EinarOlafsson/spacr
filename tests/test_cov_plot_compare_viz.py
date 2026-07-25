"""CPU coverage for spacr.plot's comparison / quick-look visualisers.

Covers ``plot_resize``, ``normalize_and_visualize``, ``visualize_masks``,
``visualize_cellpose_masks``, ``plot_comparison_results`` and
``plot_object_outlines``.

Every test inspects the Matplotlib artists the function actually produced
(displayed arrays, colormaps, norms, titles) or the files/calls it emitted,
so nothing here is a bare smoke test.  The backend is Agg (set by the test
runner), so ``plt.show()`` is a no-op and the created figure stays reachable
through ``plt.get_fignums()``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_leaked_figures():
    """Close every figure before and after each test."""
    plt.close("all")
    yield
    plt.close("all")


def _capture(func, *args, **kwargs):
    """Call ``func`` and return the last figure it created."""
    plt.close("all")
    result = func(*args, **kwargs)
    nums = plt.get_fignums()
    assert nums, f"{func.__name__} created no figure"
    return plt.figure(nums[-1]), result


def _shown(ax):
    """Return the array of the single AxesImage drawn on ``ax``."""
    assert len(ax.images) == 1, f"expected exactly one image on {ax}"
    return np.asarray(ax.images[0].get_array())


def _blob_mask(size=32, n=3):
    """Small deterministic int32 label mask with ``n`` disjoint square blobs."""
    mask = np.zeros((size, size), dtype=np.int32)
    step = size // (n + 1)
    for i in range(n):
        y = step * (i + 1) - 3
        x = step * (i + 1) - 3
        mask[y:y + 5, x:x + 5] = i + 1
    return mask


# ---------------------------------------------------------------------------
# plot_resize -- every branch of the nested prepare_image()
# ---------------------------------------------------------------------------

def test_plot_resize_grayscale_and_single_channel(rng):
    """2D input keeps its shape; (H, W, 1) is squeezed. Both use cmap 'gray'."""
    from spacr.plot import plot_resize
    gray = rng.random((16, 16))
    single = rng.random((8, 8, 1))
    fig, _ = _capture(plot_resize, [gray], [single], [gray], [single])

    ax = np.asarray(fig.axes).reshape(2, 2)
    assert _shown(ax[0, 0]).shape == (16, 16)
    assert ax[0, 0].images[0].get_cmap().name == "gray"
    squeezed = _shown(ax[0, 1])
    assert squeezed.shape == (8, 8)
    assert np.allclose(squeezed, single[:, :, 0])
    assert ax[0, 1].images[0].get_cmap().name == "gray"
    assert [a.get_title() for a in fig.axes] == [
        "Original Image", "Resized Image", "Original Label", "Resized Label",
    ]


def test_plot_resize_rgb_and_rgba_passed_through(rng):
    """3- and 4-channel images are handed to imshow untouched (cmap=None)."""
    from spacr.plot import plot_resize
    rgb = rng.random((12, 12, 3))
    rgba = rng.random((12, 12, 4))
    fig, _ = _capture(plot_resize, [rgb], [rgba], [rgb], [rgba])

    ax = np.asarray(fig.axes).reshape(2, 2)
    shown_rgb = _shown(ax[0, 0])
    shown_rgba = _shown(ax[0, 1])
    assert shown_rgb.shape == (12, 12, 3)
    assert np.allclose(shown_rgb, rgb)
    assert shown_rgba.shape == (12, 12, 4)
    assert np.allclose(shown_rgba, rgba)
    # RGB(A) data is not colormapped -> the default colormap is left in place.
    assert ax[0, 0].images[0].get_cmap().name != "gray"
    assert ax[0, 1].images[0].get_cmap().name != "gray"


def test_plot_resize_multichannel_falls_back_to_channel_mean(rng):
    """A 5-channel image is averaged over channels and shown as grayscale."""
    from spacr.plot import plot_resize
    multi = rng.random((10, 10, 5))
    gray = rng.random((10, 10))
    fig, _ = _capture(plot_resize, [multi], [gray], [gray], [gray])

    shown = _shown(fig.axes[0])
    assert shown.shape == (10, 10)
    assert np.allclose(shown, multi.mean(axis=-1))
    assert fig.axes[0].images[0].get_cmap().name == "gray"


def test_plot_resize_rejects_unsupported_dimensionality(rng):
    """4D input raises ValueError naming the offending shape."""
    from spacr.plot import plot_resize
    bad = rng.random((2, 8, 8, 3))
    gray = rng.random((8, 8))
    with pytest.raises(ValueError, match=r"Unsupported image shape: \(2, 8, 8, 3\)"):
        plot_resize([bad], [gray], [gray], [gray])


def test_plot_resize_rejects_1d_label(rng):
    """The ValueError branch is also reached from the label panels."""
    from spacr.plot import plot_resize
    gray = rng.random((8, 8))
    with pytest.raises(ValueError, match="Unsupported image shape"):
        plot_resize([gray], [gray], [np.arange(8.0)], [gray])


# ---------------------------------------------------------------------------
# normalize_and_visualize
# ---------------------------------------------------------------------------

def test_normalize_and_visualize_3d_original_2d_normalized(rng):
    """Multi-channel originals are mean-projected; 2D normalised shown as-is."""
    from spacr.plot import normalize_and_visualize
    image = rng.random((16, 16, 3))
    normalized = rng.random((16, 16))
    fig, _ = _capture(normalize_and_visualize, image, normalized, title="chan")

    left, right = fig.axes
    assert np.allclose(_shown(left), image.mean(axis=-1))
    assert np.allclose(_shown(right), normalized)
    assert left.get_title() == "Original chan"
    assert right.get_title() == "Normalized chan"
    assert left.axison is False and right.axison is False


def test_normalize_and_visualize_2d_original_3d_normalized(rng):
    """The mirror branch: 2D original passed through, 3D normalised averaged."""
    from spacr.plot import normalize_and_visualize
    image = rng.random((16, 16))
    normalized = rng.random((16, 16, 4))
    fig, _ = _capture(normalize_and_visualize, image, normalized)

    left, right = fig.axes
    assert np.allclose(_shown(left), image)
    assert np.allclose(_shown(right), normalized.mean(axis=-1))
    # default title is the empty string
    assert left.get_title() == "Original "
    assert right.get_title() == "Normalized "


# ---------------------------------------------------------------------------
# visualize_masks
# ---------------------------------------------------------------------------

def test_visualize_masks_binary_skips_normalisation_labelled_does_not():
    """Binary masks autoscale to [0, 1]; label masks get an explicit 0..max norm."""
    from spacr.plot import visualize_masks
    labelled = _blob_mask(32, n=4)          # ids 1..4
    binary = (labelled > 0).astype(np.uint8)
    fig, _ = _capture(visualize_masks, binary, labelled, binary.copy())

    assert len(fig.axes) == 3
    assert [a.get_title() for a in fig.axes] == ["Mask 1", "Mask 2", "Mask 3"]
    # binary panel -> no norm supplied, imshow autoscaled to the data range
    assert fig.axes[0].images[0].norm.vmax == pytest.approx(1.0)
    # labelled panel -> plt.Normalize(vmin=0, vmax=mask.max())
    assert fig.axes[1].images[0].norm.vmin == pytest.approx(0.0)
    assert fig.axes[1].images[0].norm.vmax == pytest.approx(float(labelled.max()))
    assert all(a.axison is False for a in fig.axes)
    assert np.array_equal(_shown(fig.axes[1]), labelled)


def test_visualize_masks_all_zero_masks_take_the_binary_branch():
    """An empty mask still satisfies isin([0, 1]) -> binary branch, no crash."""
    from spacr.plot import visualize_masks
    empty = np.zeros((16, 16), dtype=np.int32)
    fig, _ = _capture(visualize_masks, empty, empty.copy(), empty.copy())
    assert len(fig.axes) == 3
    assert all(_shown(a).max() == 0 for a in fig.axes)


def test_visualize_masks_uses_the_title_argument_for_the_suptitle():
    """The documented `title` parameter must end up as the figure suptitle."""
    from spacr.plot import visualize_masks
    mask = _blob_mask(16, n=2)
    fig, _ = _capture(visualize_masks, mask, mask.copy(), mask.copy(),
                      title="My Comparison")
    assert fig.get_suptitle() == "My Comparison"


# ---------------------------------------------------------------------------
# visualize_cellpose_masks
# ---------------------------------------------------------------------------

def test_visualize_cellpose_masks_default_titles_and_norm():
    """Without titles the panels are numbered and each gets a 0..max norm."""
    from spacr.plot import visualize_cellpose_masks
    m1 = _blob_mask(32, n=3)
    m2 = _blob_mask(32, n=2) * 5
    fig, ret = _capture(visualize_cellpose_masks, [m1, m2], filename="fieldA")

    assert ret is None
    assert [a.get_title() for a in fig.axes] == ["Mask 1", "Mask 2"]
    assert fig.get_suptitle() == "Masks Comparison for fieldA"
    assert fig.axes[1].images[0].norm.vmax == pytest.approx(float(m2.max()))
    assert all(a.axison is False for a in fig.axes)
    assert np.array_equal(_shown(fig.axes[0]), m1)


def test_visualize_cellpose_masks_custom_titles():
    """Explicit titles are used verbatim, in order."""
    from spacr.plot import visualize_cellpose_masks
    masks = [_blob_mask(16, n=2), _blob_mask(16, n=1), _blob_mask(16, n=3)]
    fig, _ = _capture(visualize_cellpose_masks, masks,
                      titles=["cell", "nucleus", "pathogen"])
    assert [a.get_title() for a in fig.axes] == ["cell", "nucleus", "pathogen"]
    assert fig.get_suptitle() == "Masks Comparison for None"


def test_visualize_cellpose_masks_title_count_mismatch_asserts():
    """Mismatched titles/masks raise before any figure is created."""
    from spacr.plot import visualize_cellpose_masks
    plt.close("all")
    masks = [_blob_mask(16, n=2), _blob_mask(16, n=2)]
    with pytest.raises(AssertionError, match="Number of titles and masks must match"):
        visualize_cellpose_masks(masks, titles=["only-one"])
    assert plt.get_fignums() == []


def test_visualize_cellpose_masks_saves_pdf_under_src(tmp_path, capsys):
    """save=True writes <src>/results/<filename>.pdf and prints the path."""
    from spacr.plot import visualize_cellpose_masks
    masks = [_blob_mask(16, n=2), _blob_mask(16, n=3)]
    visualize_cellpose_masks(masks, titles=["a", "b"], filename="field_001",
                             save=True, src=str(tmp_path))

    out = tmp_path / "results" / "field_001.pdf"
    assert out.is_file()
    assert out.stat().st_size > 0
    with out.open("rb") as fh:
        assert fh.read(4) == b"%PDF"
    assert str(out) in capsys.readouterr().out


def test_visualize_cellpose_masks_save_defaults_src_to_cwd(tmp_path, monkeypatch):
    """src=None falls back to os.getcwd()."""
    from spacr.plot import visualize_cellpose_masks
    monkeypatch.chdir(tmp_path)
    masks = [_blob_mask(16, n=1), _blob_mask(16, n=2)]
    visualize_cellpose_masks(masks, titles=["a", "b"], filename="cwd_field",
                             save=True, src=None)
    assert (tmp_path / "results" / "cwd_field.pdf").is_file()


def test_visualize_cellpose_masks_reuses_existing_results_dir(tmp_path):
    """An already-present results/ directory is reused (exist_ok=True)."""
    from spacr.plot import visualize_cellpose_masks
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "keep.txt").write_text("keep me")
    masks = [_blob_mask(16, n=1), _blob_mask(16, n=1)]
    visualize_cellpose_masks(masks, titles=["a", "b"], filename="second",
                             save=True, src=str(tmp_path))
    assert (tmp_path / "results" / "keep.txt").read_text() == "keep me"
    assert (tmp_path / "results" / "second.pdf").is_file()


def test_visualize_cellpose_masks_does_not_save_when_save_is_false(tmp_path):
    """save=False leaves the filesystem untouched."""
    from spacr.plot import visualize_cellpose_masks
    masks = [_blob_mask(16, n=1), _blob_mask(16, n=1)]
    visualize_cellpose_masks(masks, titles=["a", "b"], filename="nope",
                             save=False, src=str(tmp_path))
    assert not (tmp_path / "results").exists()


def test_visualize_cellpose_masks_handles_a_single_mask():
    """One mask is a legitimate input and must render one panel."""
    from spacr.plot import visualize_cellpose_masks
    fig, _ = _capture(visualize_cellpose_masks, [_blob_mask(16, n=2)],
                      titles=["only"])
    assert [a.get_title() for a in fig.axes] == ["only"]


# ---------------------------------------------------------------------------
# plot_comparison_results
# ---------------------------------------------------------------------------

def _comparison_records():
    return [
        {"filename": "f1", "jaccard_a_b": 0.80, "dice_a_b": 0.88,
         "boundary_f1_a_b": 0.70, "average_precision_a_b": 0.60},
        {"filename": "f2", "jaccard_a_b": 0.60, "dice_a_b": 0.75,
         "boundary_f1_a_b": 0.55, "average_precision_a_b": 0.45},
        {"filename": "f3", "jaccard_a_b": 0.90, "dice_a_b": 0.95,
         "boundary_f1_a_b": 0.85, "average_precision_a_b": 0.72},
    ]


def test_plot_comparison_results_builds_four_metric_panels():
    """Returns the figure; each panel is titled/labelled for its metric family."""
    from spacr.plot import plot_comparison_results
    fig = plot_comparison_results(_comparison_records())

    assert fig is plt.figure(plt.get_fignums()[-1])
    axs = fig.axes
    assert len(axs) == 4
    assert [a.get_title() for a in axs] == [
        "Jaccard Index by Comparison",
        "Dice Coefficient by Comparison",
        "Boundary F1 Score by Comparison",
        "Average Precision by Comparison",
    ]
    assert [a.get_ylabel() for a in axs] == [
        "Jaccard Index", "Dice Coefficient", "Boundary F1 Score",
        "Average Precision",
    ]
    assert [a.get_xlabel() for a in axs] == ["Comparison"] * 4
    # each panel only carries the columns of its own metric family
    assert [t.get_text() for t in axs[0].get_xticklabels()] == ["jaccard_a_b"]
    assert [t.get_text() for t in axs[1].get_xticklabels()] == ["dice_a_b"]
    assert [t.get_text() for t in axs[2].get_xticklabels()] == ["boundary_f1_a_b"]
    assert [t.get_text() for t in axs[3].get_xticklabels()] == ["average_precision_a_b"]
    assert axs[0].get_xticklabels()[0].get_rotation() == pytest.approx(45.0)


def test_plot_comparison_results_strip_points_match_the_input_rows():
    """The stripplot on the Jaccard panel holds one point per input file."""
    from spacr.plot import plot_comparison_results
    records = _comparison_records()
    fig = plot_comparison_results(records)

    pts = np.concatenate([c.get_offsets() for c in fig.axes[0].collections
                          if len(c.get_offsets())])
    assert pts.shape[0] == len(records)
    assert sorted(np.round(pts[:, 1], 2)) == sorted(
        r["jaccard_a_b"] for r in records)


def test_plot_comparison_results_handles_two_comparisons_per_metric():
    """Two comparison columns per family give two x categories per panel."""
    from spacr.plot import plot_comparison_results
    records = [
        {"filename": "f1", "jaccard_a_b": 0.5, "jaccard_a_c": 0.4,
         "dice_a_b": 0.6, "dice_a_c": 0.5,
         "boundary_f1_a_b": 0.7, "boundary_f1_a_c": 0.6,
         "average_precision_a_b": 0.8, "average_precision_a_c": 0.7},
        {"filename": "f2", "jaccard_a_b": 0.55, "jaccard_a_c": 0.45,
         "dice_a_b": 0.65, "dice_a_c": 0.55,
         "boundary_f1_a_b": 0.75, "boundary_f1_a_c": 0.65,
         "average_precision_a_b": 0.85, "average_precision_a_c": 0.75},
    ]
    fig = plot_comparison_results(records)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [
        "jaccard_a_b", "jaccard_a_c"]
    assert len(fig.axes[3].get_xticklabels()) == 2


def test_plot_comparison_results_requires_a_filename_column():
    """pd.melt needs 'filename' as the id var -> KeyError without it."""
    from spacr.plot import plot_comparison_results
    with pytest.raises(KeyError):
        plot_comparison_results([{"jaccard_a_b": 0.5, "dice_a_b": 0.5,
                                  "boundary_f1_a_b": 0.5,
                                  "average_precision_a_b": 0.5}])


# ---------------------------------------------------------------------------
# plot_object_outlines
# ---------------------------------------------------------------------------

@pytest.fixture
def record_plot_images(monkeypatch):
    """Replace spacr.plot.plot_images_and_arrays with a call recorder."""
    import spacr.plot as P
    calls = []

    def _recorder(folders, **kwargs):
        calls.append({"folders": folders, "kwargs": kwargs})

    monkeypatch.setattr(P, "plot_images_and_arrays", _recorder)
    return calls


def test_plot_object_outlines_default_objects_and_channels(record_plot_images, capsys):
    """Defaults iterate nucleus/cell/pathogen against channel folders 1/2/3."""
    from spacr.plot import plot_object_outlines
    src = os.path.join("data", "plate01")
    assert plot_object_outlines(src) is None

    assert [c["folders"] for c in record_plot_images] == [
        [os.path.join(src, "masks", "nucleus_mask_stack"), os.path.join(src, "1")],
        [os.path.join(src, "masks", "cell_mask_stack"), os.path.join(src, "2")],
        [os.path.join(src, "masks", "pathogen_mask_stack"), os.path.join(src, "3")],
    ]
    kwargs = record_plot_images[0]["kwargs"]
    assert kwargs["lower_percentile"] == 2
    assert kwargs["upper_percentile"] == 99.5
    assert kwargs["threshold"] == 1000
    assert kwargs["extensions"] == [".npy", ".tif", ".tiff", ".png"]
    assert kwargs["overlay"] is True
    assert kwargs["randomize"] is True
    # the folder pairs are echoed to stdout
    printed = capsys.readouterr().out
    assert printed.count("mask_stack") == 3


def test_plot_object_outlines_custom_objects_and_channels(record_plot_images):
    """Explicit objects/channels skip the defaults; folder = channel + 1."""
    from spacr.plot import plot_object_outlines
    plot_object_outlines("/root", objects=["cell"], channels=[3])
    assert len(record_plot_images) == 1
    assert record_plot_images[0]["folders"] == [
        os.path.join("/root", "masks", "cell_mask_stack"),
        os.path.join("/root", "4"),
    ]


def test_plot_object_outlines_zip_truncates_to_the_shorter_sequence(record_plot_images):
    """More objects than channels -> only the paired prefix is plotted."""
    from spacr.plot import plot_object_outlines
    plot_object_outlines("/root", objects=["cell", "nucleus", "pathogen"],
                         channels=[0])
    assert len(record_plot_images) == 1
    assert record_plot_images[0]["folders"][0].endswith("cell_mask_stack")


def test_plot_object_outlines_empty_objects_plots_nothing(record_plot_images):
    """An empty object list is respected rather than replaced by the defaults."""
    from spacr.plot import plot_object_outlines
    plot_object_outlines("/root", objects=[], channels=[])
    assert record_plot_images == []


def test_plot_object_outlines_forwards_max_nr(record_plot_images):
    """max_nr must reach plot_images_and_arrays."""
    from spacr.plot import plot_object_outlines
    plot_object_outlines("/root", objects=["cell"], channels=[0], max_nr=3)
    assert record_plot_images[0]["kwargs"]["max_nr"] == 3
