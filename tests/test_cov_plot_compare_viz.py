"""CPU coverage for spacr.plot's comparison / quick-look visualisers.

Covers ``plot_resize``, ``normalize_and_visualize``, ``visualize_masks``,
``visualize_cellpose_masks``, ``plot_comparison_results`` and
``plot_object_outlines``.

Every test inspects the Matplotlib artists the function actually produced
(displayed arrays, colormaps, norms, titles) or the files/calls it emitted,
so nothing here is a bare smoke test.  The backend is Agg (set by the test
runner), so ``plt.show()`` is a no-op and the created figure stays reachable
through ``plt.get_fignums()``.

Three parametrised tests carry the contracts: the resize / normalise panels,
the mask visualisers, and the comparison / outline plots. Each row names a
helper that drives one call and returns what it observed, beside the value
the contract requires.
"""
from __future__ import annotations

import inspect
import os
import io
import json
import re
import zipfile

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


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


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

def _case_plot_resize_grayscale_and_single_channel(rng):
    """2D input keeps its shape; (H, W, 1) is squeezed. Both use cmap 'gray'."""
    from spacr.plot import plot_resize
    gray = rng.random((16, 16))
    single = rng.random((8, 8, 1))
    fig, _ = _capture(plot_resize, [gray], [single], [gray], [single])

    ax = np.asarray(fig.axes).reshape(2, 2)
    squeezed = _shown(ax[0, 1])
    assert np.allclose(squeezed, single[:, :, 0])
    return (_shown(ax[0, 0]).shape, ax[0, 0].images[0].get_cmap().name,
            squeezed.shape, ax[0, 1].images[0].get_cmap().name,
            [a.get_title() for a in fig.axes])


def _case_plot_resize_zip_contains_all_four_source_arrays(tmp_path):
    """The image zip retains the source and resized image and label pixels."""
    import tifffile

    from spacr.figures.bundle import _save_zip
    from spacr.plot import plot_resize

    image = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    resized = image[::2, ::2].copy()
    labels = (image % 3).astype(np.uint16)
    resized_labels = labels[::2, ::2].copy()
    source = [image, resized, labels, resized_labels]
    fig, _ = _capture(plot_resize, [image], [resized], [labels],
                      [resized_labels])

    output = _save_zip(fig, str(tmp_path / "resize.zip"), formats=["png"])
    with zipfile.ZipFile(output) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        has_png = any(name.endswith(".png") for name in archive.namelist())
        shapes = []
        for index, expected in enumerate(source):
            record = metadata["arrays"][index]
            shapes.append(record["shape"])
            actual = tifffile.imread(io.BytesIO(archive.read(record["file"])))
            np.testing.assert_array_equal(actual, expected)
        return (has_png, len(metadata["arrays"]), shapes)


def _case_plot_resize_rgb_and_rgba_passed_through(rng):
    """3- and 4-channel images are handed to imshow untouched (cmap=None)."""
    from spacr.plot import plot_resize
    rgb = rng.random((12, 12, 3))
    rgba = rng.random((12, 12, 4))
    fig, _ = _capture(plot_resize, [rgb], [rgba], [rgb], [rgba])

    ax = np.asarray(fig.axes).reshape(2, 2)
    shown_rgb = _shown(ax[0, 0])
    shown_rgba = _shown(ax[0, 1])
    assert np.allclose(shown_rgb, rgb)
    assert np.allclose(shown_rgba, rgba)
    # RGB(A) data is not colormapped -> the default colormap is left in place.
    return (shown_rgb.shape, shown_rgba.shape,
            ax[0, 0].images[0].get_cmap().name != "gray",
            ax[0, 1].images[0].get_cmap().name != "gray")


def _case_plot_resize_multichannel_falls_back_to_channel_mean(rng):
    """A 5-channel image is averaged over channels and shown as grayscale."""
    from spacr.plot import plot_resize
    multi = rng.random((10, 10, 5))
    gray = rng.random((10, 10))
    fig, _ = _capture(plot_resize, [multi], [gray], [gray], [gray])

    shown = _shown(fig.axes[0])
    assert np.allclose(shown, multi.mean(axis=-1))
    return (shown.shape, fig.axes[0].images[0].get_cmap().name)


def _case_plot_resize_rejects_unsupported_dimensionality(rng):
    """4D input raises ValueError naming the offending shape."""
    from spacr.plot import plot_resize
    bad = rng.random((2, 8, 8, 3))
    gray = rng.random((8, 8))
    with pytest.raises(ValueError,
                       match=r"Unsupported image shape: \(2, 8, 8, 3\)") as raised:
        plot_resize([bad], [gray], [gray], [gray])
    return re.search(r"Unsupported image shape: \(.*\)",
                     str(raised.value)).group(0)


def _case_plot_resize_rejects_1d_label(rng):
    """The ValueError branch is also reached from the label panels."""
    from spacr.plot import plot_resize
    gray = rng.random((8, 8))
    with pytest.raises(ValueError, match="Unsupported image shape") as raised:
        plot_resize([gray], [gray], [np.arange(8.0)], [gray])
    return re.search(r"Unsupported image shape: \(.*\)",
                     str(raised.value)).group(0)


# ---------------------------------------------------------------------------
# normalize_and_visualize
# ---------------------------------------------------------------------------

def _case_normalize_and_visualize_3d_original_2d_normalized(rng):
    """Multi-channel originals are mean-projected; 2D normalised shown as-is."""
    from spacr.plot import normalize_and_visualize
    image = rng.random((16, 16, 3))
    normalized = rng.random((16, 16))
    fig, _ = _capture(normalize_and_visualize, image, normalized, title="chan")

    left, right = fig.axes
    return (bool(np.allclose(_shown(left), image.mean(axis=-1))),
            bool(np.allclose(_shown(right), normalized)),
            left.get_title(), right.get_title(),
            (left.axison, right.axison))


def _case_normalize_and_visualize_2d_original_3d_normalized(rng):
    """The mirror branch: 2D original passed through, 3D normalised averaged."""
    from spacr.plot import normalize_and_visualize
    image = rng.random((16, 16))
    normalized = rng.random((16, 16, 4))
    fig, _ = _capture(normalize_and_visualize, image, normalized)

    left, right = fig.axes
    # default title is the empty string
    return (bool(np.allclose(_shown(left), image)),
            bool(np.allclose(_shown(right), normalized.mean(axis=-1))),
            left.get_title(), right.get_title())


# ---------------------------------------------------------------------------
# visualize_masks
# ---------------------------------------------------------------------------

def _case_visualize_masks_binary_skips_normalisation_labelled_does_not():
    """Binary masks autoscale to [0, 1]; label masks get an explicit 0..max norm."""
    from spacr.plot import visualize_masks
    labelled = _blob_mask(32, n=4)          # ids 1..4
    binary = (labelled > 0).astype(np.uint8)
    fig, _ = _capture(visualize_masks, binary, labelled, binary.copy())

    assert np.array_equal(_shown(fig.axes[1]), labelled)
    return (len(fig.axes), [a.get_title() for a in fig.axes],
            # binary panel -> no norm supplied, autoscaled to the data range
            fig.axes[0].images[0].norm.vmax,
            # labelled panel -> plt.Normalize(vmin=0, vmax=mask.max())
            fig.axes[1].images[0].norm.vmin,
            fig.axes[1].images[0].norm.vmax,
            all(a.axison is False for a in fig.axes))


def _case_visualize_masks_all_zero_masks_take_the_binary_branch():
    """An empty mask still satisfies isin([0, 1]) -> binary branch, no crash."""
    from spacr.plot import visualize_masks
    empty = np.zeros((16, 16), dtype=np.int32)
    fig, _ = _capture(visualize_masks, empty, empty.copy(), empty.copy())
    return (len(fig.axes), [int(_shown(a).max()) for a in fig.axes])


def _case_visualize_masks_uses_the_title_argument_for_the_suptitle():
    """The documented `title` parameter must end up as the figure suptitle."""
    from spacr.plot import visualize_masks
    mask = _blob_mask(16, n=2)
    fig, _ = _capture(visualize_masks, mask, mask.copy(), mask.copy(),
                      title="My Comparison")
    return fig.get_suptitle()


# ---------------------------------------------------------------------------
# visualize_cellpose_masks
# ---------------------------------------------------------------------------

def _case_visualize_cellpose_masks_default_titles_and_norm():
    """Without titles the panels are numbered and each gets a 0..max norm."""
    from spacr.plot import visualize_cellpose_masks
    m1 = _blob_mask(32, n=3)
    m2 = _blob_mask(32, n=2) * 5
    fig, ret = _capture(visualize_cellpose_masks, [m1, m2], filename="fieldA")

    assert np.array_equal(_shown(fig.axes[0]), m1)
    return (ret is None, [a.get_title() for a in fig.axes],
            fig.get_suptitle(), fig.axes[1].images[0].norm.vmax,
            float(m2.max()), all(a.axison is False for a in fig.axes))


def _case_visualize_cellpose_masks_custom_titles():
    """Explicit titles are used verbatim, in order."""
    from spacr.plot import visualize_cellpose_masks
    masks = [_blob_mask(16, n=2), _blob_mask(16, n=1), _blob_mask(16, n=3)]
    fig, _ = _capture(visualize_cellpose_masks, masks,
                      titles=["cell", "nucleus", "pathogen"])
    return ([a.get_title() for a in fig.axes], fig.get_suptitle())


def _case_visualize_cellpose_masks_title_count_mismatch_asserts():
    """Mismatched titles/masks raise before any figure is created."""
    from spacr.plot import visualize_cellpose_masks
    plt.close("all")
    masks = [_blob_mask(16, n=2), _blob_mask(16, n=2)]
    with pytest.raises(AssertionError,
                       match="Number of titles and masks must match") as raised:
        visualize_cellpose_masks(masks, titles=["only-one"])
    return ("Number of titles and masks must match" in str(raised.value),
            plt.get_fignums())


def _case_visualize_cellpose_masks_saves_pdf_under_src(tmp_path, capsys):
    """save=True writes <src>/results/<filename>.pdf and prints the path."""
    from spacr.plot import visualize_cellpose_masks
    masks = [_blob_mask(16, n=2), _blob_mask(16, n=3)]
    visualize_cellpose_masks(masks, titles=["a", "b"], filename="field_001",
                             save=True, src=str(tmp_path))

    out = tmp_path / "results" / "field_001.pdf"
    assert out.stat().st_size > 0
    with out.open("rb") as fh:
        magic = fh.read(4)
    return (out.is_file(), magic, str(out) in capsys.readouterr().out)


def _case_visualize_cellpose_masks_save_defaults_src_to_cwd(tmp_path, monkeypatch):
    """src=None falls back to os.getcwd()."""
    from spacr.plot import visualize_cellpose_masks
    monkeypatch.chdir(tmp_path)
    masks = [_blob_mask(16, n=1), _blob_mask(16, n=2)]
    visualize_cellpose_masks(masks, titles=["a", "b"], filename="cwd_field",
                             save=True, src=None)
    return (tmp_path / "results" / "cwd_field.pdf").is_file()


def _case_visualize_cellpose_masks_reuses_existing_results_dir(tmp_path):
    """An already-present results/ directory is reused (exist_ok=True)."""
    from spacr.plot import visualize_cellpose_masks
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "keep.txt").write_text("keep me", encoding="utf-8")
    masks = [_blob_mask(16, n=1), _blob_mask(16, n=1)]
    visualize_cellpose_masks(masks, titles=["a", "b"], filename="second",
                             save=True, src=str(tmp_path))
    return ((tmp_path / "results" / "keep.txt").read_text(encoding="utf-8"),
            (tmp_path / "results" / "second.pdf").is_file())


def _case_visualize_cellpose_masks_does_not_save_when_save_is_false(tmp_path):
    """save=False leaves the filesystem untouched."""
    from spacr.plot import visualize_cellpose_masks
    masks = [_blob_mask(16, n=1), _blob_mask(16, n=1)]
    visualize_cellpose_masks(masks, titles=["a", "b"], filename="nope",
                             save=False, src=str(tmp_path))
    return (tmp_path / "results").exists()


def _case_visualize_cellpose_masks_handles_a_single_mask():
    """One mask is a legitimate input and must render one panel."""
    from spacr.plot import visualize_cellpose_masks
    fig, _ = _capture(visualize_cellpose_masks, [_blob_mask(16, n=2)],
                      titles=["only"])
    return [a.get_title() for a in fig.axes]


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


def _case_plot_comparison_results_builds_four_metric_panels():
    """Returns the figure; each panel is titled/labelled for its metric family."""
    from spacr.plot import plot_comparison_results
    fig = plot_comparison_results(_comparison_records())

    assert fig is plt.figure(plt.get_fignums()[-1])
    axs = fig.axes
    return {
        "panels": len(axs),
        "titles": [a.get_title() for a in axs],
        "ylabels": [a.get_ylabel() for a in axs],
        "xlabels": [a.get_xlabel() for a in axs],
        # each panel only carries the columns of its own metric family
        "ticks": [[t.get_text() for t in a.get_xticklabels()] for a in axs],
        "rotation": axs[0].get_xticklabels()[0].get_rotation(),
    }


def _case_plot_comparison_results_strip_points_match_the_input_rows():
    """The stripplot on the Jaccard panel holds one point per input file."""
    from spacr.plot import plot_comparison_results
    records = _comparison_records()
    fig = plot_comparison_results(records)

    pts = np.concatenate([c.get_offsets() for c in fig.axes[0].collections
                          if len(c.get_offsets())])
    assert pts.shape[0] == len(records)
    return [float(v) for v in sorted(np.round(pts[:, 1], 2))]


def _case_plot_comparison_results_handles_two_comparisons_per_metric():
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
    return ([t.get_text() for t in fig.axes[0].get_xticklabels()],
            len(fig.axes[3].get_xticklabels()))


def _case_plot_comparison_results_requires_a_filename_column():
    """pd.melt needs 'filename' as the id var -> KeyError without it."""
    from spacr.plot import plot_comparison_results
    with pytest.raises(KeyError) as raised:
        plot_comparison_results([{"jaccard_a_b": 0.5, "dice_a_b": 0.5,
                                  "boundary_f1_a_b": 0.5,
                                  "average_precision_a_b": 0.5}])
    return raised.type.__name__


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


def _case_plot_object_outlines_default_objects_and_channels(record_plot_images, capsys):
    """Defaults iterate nucleus/cell/pathogen against channel folders 1/2/3."""
    from spacr.plot import plot_object_outlines
    src = os.path.join("data", "plate01")
    returned_none = plot_object_outlines(src) is None

    kwargs = record_plot_images[0]["kwargs"]
    # the folder pairs are echoed to stdout
    printed = capsys.readouterr().out
    return {
        "returned_none": returned_none,
        "folders": [c["folders"] for c in record_plot_images],
        "lower": kwargs["lower_percentile"],
        "upper": kwargs["upper_percentile"],
        "threshold": kwargs["threshold"],
        "extensions": kwargs["extensions"],
        "overlay": kwargs["overlay"],
        "randomize": kwargs["randomize"],
        "echoed": printed.count("mask_stack"),
    }


def _case_plot_object_outlines_custom_objects_and_channels(record_plot_images):
    """Explicit objects/channels skip the defaults; folder = channel + 1."""
    from spacr.plot import plot_object_outlines
    plot_object_outlines("/root", objects=["cell"], channels=[3])
    return (len(record_plot_images), record_plot_images[0]["folders"])


def _case_plot_object_outlines_zip_truncates_to_the_shorter_sequence(record_plot_images):
    """More objects than channels -> only the paired prefix is plotted."""
    from spacr.plot import plot_object_outlines
    plot_object_outlines("/root", objects=["cell", "nucleus", "pathogen"],
                         channels=[0])
    return (len(record_plot_images),
            record_plot_images[0]["folders"][0].endswith("cell_mask_stack"))


def _case_plot_object_outlines_empty_objects_plots_nothing(record_plot_images):
    """An empty object list is respected rather than replaced by the defaults."""
    from spacr.plot import plot_object_outlines
    plot_object_outlines("/root", objects=[], channels=[])
    return record_plot_images


def _case_plot_object_outlines_forwards_max_nr(record_plot_images):
    """max_nr must reach plot_images_and_arrays."""
    from spacr.plot import plot_object_outlines
    plot_object_outlines("/root", objects=["cell"], channels=[0], max_nr=3)
    return record_plot_images[0]["kwargs"]["max_nr"]


_COMPARISON_TICKS = [["jaccard_a_b"], ["dice_a_b"], ["boundary_f1_a_b"],
                     ["average_precision_a_b"]]


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_case_plot_resize_grayscale_and_single_channel,
                 ((16, 16), "gray", (8, 8), "gray",
                  ["Original Image", "Resized Image", "Original Label",
                   "Resized Label"]),
                 id="resize-grayscale-and-single-channel"),
    pytest.param(_case_plot_resize_zip_contains_all_four_source_arrays,
                 (True, 4, [[16, 16], [8, 8], [16, 16], [8, 8]]),
                 id="resize-zip-contains-all-four-source-arrays"),
    pytest.param(_case_plot_resize_rgb_and_rgba_passed_through,
                 ((12, 12, 3), (12, 12, 4), True, True),
                 id="resize-rgb-and-rgba-passed-through"),
    pytest.param(_case_plot_resize_multichannel_falls_back_to_channel_mean,
                 ((10, 10), "gray"), id="resize-multichannel-channel-mean"),
    pytest.param(_case_plot_resize_rejects_unsupported_dimensionality,
                 "Unsupported image shape: (2, 8, 8, 3)",
                 id="resize-refuses-4d-input-by-shape"),
    pytest.param(_case_plot_resize_rejects_1d_label,
                 "Unsupported image shape: (8,)",
                 id="resize-refuses-1d-label-by-shape"),
    pytest.param(_case_normalize_and_visualize_3d_original_2d_normalized,
                 (True, True, "Original chan", "Normalized chan",
                  (False, False)),
                 id="normalize-3d-original-2d-normalized"),
    pytest.param(_case_normalize_and_visualize_2d_original_3d_normalized,
                 (True, True, "Original ", "Normalized "),
                 id="normalize-2d-original-3d-normalized"),
])
def test_resize_and_normalise_panels_show_the_prepared_image(
        request, helper, expected):
    """Each branch of the image preparation shows the array the contract
    names, with its colormap and titles, and refuses unsupported shapes."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    pytest.param(
        _case_visualize_masks_binary_skips_normalisation_labelled_does_not,
        (3, ["Mask 1", "Mask 2", "Mask 3"], pytest.approx(1.0),
         pytest.approx(0.0), pytest.approx(4.0), True),
        id="visualize-masks-binary-vs-labelled-norm"),
    pytest.param(_case_visualize_masks_all_zero_masks_take_the_binary_branch,
                 (3, [0, 0, 0]), id="visualize-masks-all-zero-binary-branch"),
    pytest.param(_case_visualize_masks_uses_the_title_argument_for_the_suptitle,
                 "My Comparison", id="visualize-masks-title-is-suptitle"),
    pytest.param(_case_visualize_cellpose_masks_default_titles_and_norm,
                 (True, ["Mask 1", "Mask 2"], "Masks Comparison for fieldA",
                  pytest.approx(10.0), 10.0, True),
                 id="cellpose-masks-default-titles-and-norm"),
    pytest.param(_case_visualize_cellpose_masks_custom_titles,
                 (["cell", "nucleus", "pathogen"],
                  "Masks Comparison for None"),
                 id="cellpose-masks-custom-titles"),
    pytest.param(_case_visualize_cellpose_masks_title_count_mismatch_asserts,
                 (True, []), id="cellpose-masks-title-count-mismatch-refused"),
    pytest.param(_case_visualize_cellpose_masks_saves_pdf_under_src,
                 (True, b"%PDF", True), id="cellpose-masks-saves-pdf-under-src"),
    pytest.param(_case_visualize_cellpose_masks_save_defaults_src_to_cwd,
                 True, id="cellpose-masks-save-defaults-src-to-cwd"),
    pytest.param(_case_visualize_cellpose_masks_reuses_existing_results_dir,
                 ("keep me", True), id="cellpose-masks-reuses-results-dir"),
    pytest.param(_case_visualize_cellpose_masks_does_not_save_when_save_is_false,
                 False, id="cellpose-masks-no-save-leaves-filesystem"),
    pytest.param(_case_visualize_cellpose_masks_handles_a_single_mask,
                 ["only"], id="cellpose-masks-single-mask"),
])
def test_mask_visualisers_title_normalise_and_save_panels(
        request, helper, expected):
    """Mask panels get their titles, colour normalisation and, on request,
    a PDF under ``<src>/results``."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    pytest.param(
        _case_plot_comparison_results_builds_four_metric_panels,
        {"panels": 4,
         "titles": ["Jaccard Index by Comparison",
                    "Dice Coefficient by Comparison",
                    "Boundary F1 Score by Comparison",
                    "Average Precision by Comparison"],
         "ylabels": ["Jaccard Index", "Dice Coefficient", "Boundary F1 Score",
                     "Average Precision"],
         "xlabels": ["Comparison"] * 4, "ticks": _COMPARISON_TICKS,
         "rotation": pytest.approx(45.0)},
        id="comparison-builds-four-metric-panels"),
    pytest.param(
        _case_plot_comparison_results_strip_points_match_the_input_rows,
        [0.6, 0.8, 0.9], id="comparison-strip-points-match-input-rows"),
    pytest.param(
        _case_plot_comparison_results_handles_two_comparisons_per_metric,
        (["jaccard_a_b", "jaccard_a_c"], 2),
        id="comparison-two-comparisons-per-metric"),
    pytest.param(
        _case_plot_comparison_results_requires_a_filename_column,
        "KeyError", id="comparison-refuses-missing-filename-column"),
    pytest.param(
        _case_plot_object_outlines_default_objects_and_channels,
        {"returned_none": True,
         "folders": [
             [os.path.join("data", "plate01", "masks", "nucleus_mask_stack"),
              os.path.join("data", "plate01", "1")],
             [os.path.join("data", "plate01", "masks", "cell_mask_stack"),
              os.path.join("data", "plate01", "2")],
             [os.path.join("data", "plate01", "masks", "pathogen_mask_stack"),
              os.path.join("data", "plate01", "3")]],
         "lower": 2, "upper": 99.5, "threshold": 1000,
         "extensions": [".npy", ".tif", ".tiff", ".png"],
         "overlay": True, "randomize": True, "echoed": 3},
        id="outlines-default-objects-and-channels"),
    pytest.param(
        _case_plot_object_outlines_custom_objects_and_channels,
        (1, [os.path.join("/root", "masks", "cell_mask_stack"),
             os.path.join("/root", "4")]),
        id="outlines-custom-objects-and-channels"),
    pytest.param(
        _case_plot_object_outlines_zip_truncates_to_the_shorter_sequence,
        (1, True), id="outlines-zip-truncates-to-shorter-sequence"),
    pytest.param(_case_plot_object_outlines_empty_objects_plots_nothing,
                 [], id="outlines-empty-objects-plots-nothing"),
    pytest.param(_case_plot_object_outlines_forwards_max_nr, 3,
                 id="outlines-forwards-max-nr"),
])
def test_comparison_and_outline_plots_follow_their_inputs(
        request, helper, expected):
    """The comparison figure and the outline browser mirror the records,
    objects and channels handed in."""
    observed = _call(request, helper)
    assert observed == expected
