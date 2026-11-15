"""CPU coverage for the single-cell-image / timelapse block of spacr.plot.

Covers:
    _save_scimg_plot                                (grid figure per channel)
    _plot_cropped_arrays                            (2-D and 3-D stacks)
    _visualize_and_save_timelapse_stack_with_tracks (gif + interactive frame)
    _display_gif

Everything runs headless on Agg with tiny synthetic PNGs / label masks.
The two IO helpers that would otherwise render 20x20 inch figures at
300 dpi (`spacr.io._save_figure`) or animate a real GIF
(`spacr.io._save_mask_timelapse_as_gif`) are replaced by recorders in most
tests so the assertions can inspect exactly what spacr.plot handed them;
one row lets the real `_save_figure` run so the emitted PDFs are checked
on disk.

Each contract is one parametrised test.  The rows name a helper that does the
setup and call (keeping its secondary checks) and returns what it observed;
the row carries the expected value and the test body compares the two.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


def _names(files):
    return [os.path.basename(f) for f in files]


@pytest.fixture(autouse=True)
def _close_figures():
    """No figure may survive a test (these functions never close their own)."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def scimg_dir(tmp_path):
    """A single-cell crop folder: 3 identically sized crops + 1 odd one.

    Layout mirrors what `_save_scimg_plot` expects:
        <root>/plate1/cell/*.png     <- src
        <root>/plate1/figure/*.pdf   <- where _save_figure writes
    """
    root = tmp_path / "plate1"
    src = root / "cell"
    src.mkdir(parents=True)
    rng = np.random.default_rng(7)

    same = []
    for i in range(3):
        arr = np.zeros((40, 40, 3), dtype=np.uint8)
        # every pixel of the block is >= 30 so the non-zero bounding box that
        # _find_similar_sized_images computes is exactly 24 x 24 for all three.
        arr[8:32, 8:32, :] = rng.integers(30, 255, size=(24, 24, 3), dtype=np.uint8)
        p = src / f"crop_{i}.png"
        assert cv2.imwrite(str(p), arr)
        same.append(str(p))

    odd = np.zeros((40, 40, 3), dtype=np.uint8)
    odd[5:35, 5:20, :] = 200  # 30 x 15 bounding box -> different size group
    odd_path = src / "odd_crop.png"
    assert cv2.imwrite(str(odd_path), odd)

    # a non-image file that _generate_filelist must ignore
    (src / "notes.txt").write_text("not an image", encoding="utf-8")

    return {
        "root": root,
        "src": str(src),
        "same": sorted(same),
        "odd": str(odd_path),
        "all": sorted(same + [str(odd_path)]),
        "figure_dir": root / "figure",
    }


@pytest.fixture
def grid_recorder(monkeypatch):
    """Replace `_plot_images_on_grid` with a recorder returning a real figure."""
    import spacr.plot

    calls = []

    def fake(image_files, channel_indices, um_per_pixel, scale_bar_length_um=5,
             fontsize=8, show_filename=True, channel_names=None, plot=False):
        calls.append({
            "files": sorted(image_files),
            "channel_indices": None if channel_indices is None else list(channel_indices),
            "um_per_pixel": um_per_pixel,
            "scale_bar_length_um": scale_bar_length_um,
            "fontsize": fontsize,
            "show_filename": show_filename,
            "channel_names": channel_names,
            "plot": plot,
        })
        fig = plt.figure(figsize=(1, 1))
        fig.add_subplot(111).plot([0, 1], [1, 0])
        return fig

    monkeypatch.setattr(spacr.plot, "_plot_images_on_grid", fake)
    return calls


@pytest.fixture
def figure_recorder(monkeypatch):
    """Replace `spacr.io._save_figure` (imported inside _save_scimg_plot)."""
    import spacr.io

    calls = []

    def fake(fig, src, text, dpi=300, i=1, all_folders=1):
        calls.append({"fig": fig, "src": src, "text": text, "dpi": dpi,
                      "n_axes": len(fig.axes)})
        plt.close(fig)

    monkeypatch.setattr(spacr.io, "_save_figure", fake)
    return calls


def _write_tiny_gif(path):
    from PIL import Image
    Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).save(path, format="GIF")


@pytest.fixture
def gif_recorder(monkeypatch):
    """Replace `spacr.io._save_mask_timelapse_as_gif` with a light real-GIF writer."""
    import spacr.io

    calls = []

    def fake(masks, tracks_df, path, cmap, norm, filenames):
        calls.append({"masks": masks, "tracks_df": tracks_df, "path": path,
                      "cmap": cmap, "norm": norm, "filenames": filenames})
        _write_tiny_gif(path)

    monkeypatch.setattr(spacr.io, "_save_mask_timelapse_as_gif", fake)
    return calls


@pytest.fixture
def timelapse_masks():
    """Two 12x12 label frames: frame 0 has label 1, frame 1 has labels 1 and 2."""
    f0 = np.zeros((12, 12), dtype=np.int32)
    f0[2:5, 2:5] = 1
    f1 = np.zeros((12, 12), dtype=np.int32)
    f1[3:6, 3:6] = 1
    f1[8:10, 8:10] = 2
    return [f0, f1]


@pytest.fixture
def timelapse_tracks():
    return pd.DataFrame({
        "track_id": [1, 1, 2, 2],
        "x": [3.0, 4.0, 8.5, 9.0],
        "y": [3.0, 4.0, 8.5, 9.0],
        "frame": [0, 1, 0, 1],
    })


# ---------------------------------------------------------------------------
# _save_scimg_plot
# ---------------------------------------------------------------------------

def _case_save_scimg_plot_defaults_to_three_channels_and_forwards_args(
    scimg_dir, grid_recorder, figure_recorder
):
    """channel_indices=None -> [0,1,2]; one combined figure + one per channel."""
    from spacr.plot import _save_scimg_plot

    out = _save_scimg_plot(
        scimg_dir["src"],
        nr_imgs=None,
        channel_indices=None,
        um_per_pixel=0.25,
        scale_bar_length_um=7,
        standardize=True,
        fontsize=11,
        show_filename=False,
        channel_names=["r", "g", "b"],
        plot=False,
    )

    # channel names are only stamped on the combined figure (returned below)
    # scalar plot arguments reach _plot_images_on_grid in the right slots
    for call in grid_recorder:
        assert call["um_per_pixel"] == 0.25
        assert call["scale_bar_length_um"] == 7
        assert call["fontsize"] == 11
        assert call["show_filename"] is False
        assert call["plot"] is False

    assert all(c["src"] == scimg_dir["src"] for c in figure_recorder)
    # every saved figure is the one the grid plotter just built
    assert all(c["n_axes"] == 1 for c in figure_recorder)
    return (out,
            [c["channel_indices"] for c in grid_recorder],
            [c["channel_names"] for c in grid_recorder],
            [c["text"] for c in figure_recorder])


def _case_save_scimg_plot_standardize_keeps_only_the_largest_size_group(
    scimg_dir, grid_recorder, figure_recorder
):
    """standardize=True drops the odd-sized crop; the .txt file is never listed."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=None, channel_indices=[1],
                     standardize=True)

    for call in grid_recorder:
        assert scimg_dir["odd"] not in call["files"]
    # combined figure + channel_1, each over the same-sized crops only
    return (len(grid_recorder), [_names(c["files"]) for c in grid_recorder],
            [c["text"] for c in figure_recorder])


def _case_save_scimg_plot_without_standardize_uses_every_image(
    scimg_dir, grid_recorder, figure_recorder
):
    """standardize=False skips _find_similar_sized_images entirely."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=None, channel_indices=[2],
                     standardize=False)

    return (len(grid_recorder), [_names(c["files"]) for c in grid_recorder],
            [c["text"] for c in figure_recorder])


def _case_save_scimg_plot_nr_imgs_subsamples_deterministically(
    scimg_dir, grid_recorder, figure_recorder
):
    """nr_imgs < len(files) -> a seeded random subset, identical for every figure."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=2, channel_indices=[0],
                     standardize=False)

    for call in grid_recorder:
        assert set(call["files"]).issubset(set(scimg_dir["all"]))
    n_figures = len(grid_recorder)
    sizes = [len(c["files"]) for c in grid_recorder]
    # random.seed(42) is re-applied per figure, so both figures show the same crops
    same_within = grid_recorder[0]["files"] == grid_recorder[1]["files"]
    first_files = grid_recorder[0]["files"]

    # ...and the seeding makes the choice reproducible across calls
    grid_recorder.clear()
    _save_scimg_plot(scimg_dir["src"], nr_imgs=2, channel_indices=[0],
                     standardize=False)
    same_within_again = grid_recorder[0]["files"] == grid_recorder[1]["files"]
    same_across_calls = grid_recorder[0]["files"] == first_files
    return n_figures, sizes, same_within, same_within_again, same_across_calls


def _case_save_scimg_plot_nr_imgs_larger_than_folder_keeps_all(
    scimg_dir, grid_recorder, figure_recorder
):
    """nr_imgs >= number of images -> no sampling, the full list is plotted."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=99, channel_indices=[0],
                     standardize=False)

    return _names(grid_recorder[0]["files"])


def _case_save_scimg_plot_writes_pdfs_to_the_figure_folder(scimg_dir):
    """End-to-end with the real grid plotter and the real _save_figure."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=2, channel_indices=[0],
                     um_per_pixel=0.5, scale_bar_length_um=2, standardize=True,
                     plot=False)

    fig_dir = scimg_dir["figure_dir"]
    for pdf in fig_dir.glob("*.pdf"):
        data = pdf.read_bytes()
        assert data.startswith(b"%PDF")
        assert len(data) > 1000
    # _save_figure closes each figure it writes (empty fignums)
    return sorted(p.name for p in fig_dir.glob("*.pdf")), plt.get_fignums()


def _case_save_scimg_plot_with_a_single_image(scimg_dir, figure_recorder):
    """A one-crop folder/sample is a legal input and should still emit figures."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=1, channel_indices=[0],
                     standardize=False)

    return [c["text"] for c in figure_recorder]


# ---------------------------------------------------------------------------
# _plot_cropped_arrays
# ---------------------------------------------------------------------------

def _case_plot_cropped_arrays_2d_label_image_gets_random_object_cmap():
    """<= threshold unique values -> random label cmap + object count in title."""
    from spacr.plot import _plot_cropped_arrays

    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[2:6, 2:6] = 1
    arr[9:13, 9:13] = 2
    fig = _plot_cropped_arrays(arr, "field_1.npy", figuresize=3)

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.images) == 1
    np.testing.assert_array_equal(np.asarray(ax.images[0].get_array()), arr)
    cmap = ax.images[0].get_cmap()
    assert isinstance(cmap, ListedColormap)
    # 2 objects (labels 1 and 2); background 0 is not an object, so cmap.N is 3
    return (len(fig.axes), ax.get_title(), ax.axison, cmap.N,
            tuple(cmap.colors[0]), fig.get_size_inches().tolist())


def _case_plot_cropped_arrays_2d_intensity_image_keeps_requested_cmap():
    """> threshold unique values -> the caller's cmap and an unannotated title."""
    from spacr.plot import _plot_cropped_arrays

    arr = np.arange(256, dtype=np.uint16).reshape(16, 16)
    fig = _plot_cropped_arrays(arr, "field_2.npy", figuresize=2, cmap="inferno",
                               threshold=10)

    ax = fig.axes[0]
    # the registered 'inferno' map is used verbatim, not a freshly drawn
    # random label map (which would be a 3-colour ListedColormap here)
    cmap = ax.images[0].get_cmap()
    assert cmap.N == plt.get_cmap("inferno").N
    np.testing.assert_allclose(cmap(0.0), plt.get_cmap("inferno")(0.0))
    # the title has no ", N (obj.)" suffix
    return ax.get_title(), cmap.name, cmap.N


def _case_plot_cropped_arrays_3d_makes_one_axis_per_channel():
    from spacr.plot import _plot_cropped_arrays

    stack = np.zeros((8, 8, 3), dtype=np.uint16)
    stack[:, :, 0] = np.arange(64).reshape(8, 8)      # 64 unique -> label cmap
    stack[2:5, 2:5, 1] = 1                            # 2 unique  -> label cmap, 1 obj
    stack[:, :, 2] = np.arange(64).reshape(8, 8) * 9  # 64 unique

    fig = _plot_cropped_arrays(stack, "field_3.npy", figuresize=4, threshold=10)

    np.testing.assert_array_equal(
        np.asarray(fig.axes[1].images[0].get_array()), stack[:, :, 1]
    )
    return (len(fig.axes), [ax.get_title() for ax in fig.axes],
            [ax.axison for ax in fig.axes])


def _case_plot_cropped_arrays_object_count_excludes_background():
    """The "(obj.)" count is the number of labels, not of distinct values.

    Regression: the title used to print ``len(np.unique(array))``, so the
    background value 0 was counted as an object and every mask panel read one
    too high -- an all-background mask claimed "1 (obj.)".
    """
    from spacr.plot import _plot_cropped_arrays

    three = np.zeros((16, 16), dtype=np.uint8)
    three[1:4, 1:4] = 1
    three[6:9, 6:9] = 2
    three[11:14, 11:14] = 3
    titles = [_plot_cropped_arrays(three, "three.npy", figuresize=2).axes[0].get_title()]

    empty = np.zeros((8, 8), dtype=np.uint8)
    titles.append(_plot_cropped_arrays(empty, "empty.npy", figuresize=2).axes[0].get_title())

    # No background pixel at all: the single label is still one object.
    full = np.ones((8, 8), dtype=np.uint8)
    titles.append(_plot_cropped_arrays(full, "full.npy", figuresize=2).axes[0].get_title())

    # Non-contiguous labels are counted as they are, not as max(label).
    sparse = np.zeros((8, 8), dtype=np.uint16)
    sparse[1:3, 1:3] = 7
    sparse[5:7, 5:7] = 91
    titles.append(_plot_cropped_arrays(sparse, "sparse.npy", figuresize=2).axes[0].get_title())
    return titles


def _case_plot_cropped_arrays_single_channel_stack():
    """A stack with exactly one channel is a valid shape and must not crash."""
    from spacr.plot import _plot_cropped_arrays

    stack = np.zeros((8, 8, 1), dtype=np.uint16)
    stack[2:5, 2:5, 0] = 1
    fig = _plot_cropped_arrays(stack, "field_4.npy", figuresize=2)

    return len(fig.axes), fig.axes[0].get_title().startswith("C. 0")


# ---------------------------------------------------------------------------
# _visualize_and_save_timelapse_stack_with_tracks
# ---------------------------------------------------------------------------

def _case_timelapse_save_builds_label_colormap_and_gif_path(
    tmp_path, timelapse_masks, timelapse_tracks, gif_recorder
):
    from spacr.plot import _visualize_and_save_timelapse_stack_with_tracks

    src = tmp_path / "plate1" / "masks"
    src.mkdir(parents=True)

    _visualize_and_save_timelapse_stack_with_tracks(
        timelapse_masks, timelapse_tracks, True, str(src), "well_A01",
        False, ["f0.tif", "f1.tif"], "cell",
    )

    call = gif_recorder[0]

    expected = tmp_path / "plate1" / "movies" / "gif" / "timelapse_masks_cell_well_A01.gif"
    assert call["path"] == str(expected)
    assert expected.parent.is_dir()
    assert expected.is_file()

    assert call["masks"] is timelapse_masks
    assert call["tracks_df"] is timelapse_tracks

    cmap = call["cmap"]
    assert isinstance(cmap, ListedColormap)
    assert np.allclose(np.asarray(cmap.colors)[:, 3], 1.0)   # fully opaque
    assert tuple(cmap.colors[0]) == (0.0, 0.0, 0.0, 1.0)     # background is black
    assert not np.allclose(cmap.colors[1], cmap.colors[0])

    norm = call["norm"]
    # cmap.N is the highest label (2) + 1; plot=False -> no figure was opened
    return (len(gif_recorder), str(expected.relative_to(tmp_path)),
            call["filenames"], cmap.N, (norm.vmin, norm.vmax), plt.get_fignums())


def _case_timelapse_without_save_or_plot_writes_nothing(
    tmp_path, timelapse_masks, timelapse_tracks, gif_recorder
):
    from spacr.plot import _visualize_and_save_timelapse_stack_with_tracks

    src = tmp_path / "plate1" / "masks"
    src.mkdir(parents=True)

    out = _visualize_and_save_timelapse_stack_with_tracks(
        timelapse_masks, timelapse_tracks, False, str(src), "n", False,
        ["f0.tif", "f1.tif"], "nucleus",
    )

    return (out, gif_recorder, (tmp_path / "plate1" / "movies").exists(),
            plt.get_fignums())


def _case_timelapse_interactive_renders_the_first_frame_with_tracks(
    tmp_path, timelapse_masks, timelapse_tracks
):
    """plot + interactive -> ipywidgets.interact draws frame 0 immediately."""
    from spacr.plot import _visualize_and_save_timelapse_stack_with_tracks

    src = tmp_path / "plate1" / "masks"
    src.mkdir(parents=True)

    _visualize_and_save_timelapse_stack_with_tracks(
        timelapse_masks, timelapse_tracks, False, str(src), "n", True,
        ["f0.tif", "f1.tif"], "cell", interactive=True,
    )

    nums = plt.get_fignums()
    assert len(nums) == 1
    ax = plt.figure(nums[0]).axes[0]

    # frame 0 carries a single object -> a single label annotation
    # the mask is drawn with the shared label cmap / norm
    assert len(ax.images) == 1
    np.testing.assert_array_equal(
        np.asarray(ax.images[0].get_array()), timelapse_masks[0]
    )

    # one black polyline per track id
    assert len(ax.lines) == timelapse_tracks["track_id"].nunique()
    for line in ax.lines:
        assert line.get_color() == "k"
        assert line.get_linewidth() == 1
    return (ax.get_title(), ax.axison, [t.get_text() for t in ax.texts],
            ax.images[0].get_cmap().N,
            (ax.images[0].norm.vmin, ax.images[0].norm.vmax),
            [float(v) for v in ax.lines[0].get_xdata()],
            [float(v) for v in ax.lines[0].get_ydata()])


def _case_timelapse_plot_non_interactive_displays_the_saved_gif(
    tmp_path, monkeypatch, timelapse_masks, timelapse_tracks, gif_recorder
):
    import spacr.plot
    from spacr.plot import _visualize_and_save_timelapse_stack_with_tracks

    shown = []
    monkeypatch.setattr(spacr.plot, "display", lambda obj: shown.append(obj))

    src = tmp_path / "plate1" / "masks"
    src.mkdir(parents=True)

    _visualize_and_save_timelapse_stack_with_tracks(
        timelapse_masks, timelapse_tracks, True, str(src), "n", True,
        ["f0.tif", "f1.tif"], "pathogen", interactive=False,
    )

    gif_path = gif_recorder[0]["path"]
    from IPython.display import Image as ipyimage
    assert isinstance(shown[0], ipyimage)
    assert shown[0].data == open(gif_path, "rb").read()
    # non-interactive -> the per-frame matplotlib figure was never built
    return len(shown), shown[0].data.startswith(b"GIF"), plt.get_fignums()


def _case_timelapse_interactive_save_does_not_display_the_gif(
    tmp_path, monkeypatch, timelapse_masks, timelapse_tracks, gif_recorder
):
    """interactive=True suppresses the inline gif in favour of the slider."""
    import spacr.plot
    from spacr.plot import _visualize_and_save_timelapse_stack_with_tracks

    shown = []
    monkeypatch.setattr(spacr.plot, "display", lambda obj: shown.append(obj))

    src = tmp_path / "plate1" / "masks"
    src.mkdir(parents=True)

    _visualize_and_save_timelapse_stack_with_tracks(
        timelapse_masks, timelapse_tracks, True, str(src), "n", True,
        ["f0.tif", "f1.tif"], "cell", interactive=True,
    )

    # shown stays empty: _display_gif was not reached
    return len(gif_recorder), os.path.isfile(gif_recorder[0]["path"]), shown


def _case_timelapse_empty_mask_list_raises(tmp_path, timelapse_tracks, gif_recorder):
    """No frames -> max() over an empty generator; nothing is written."""
    from spacr.plot import _visualize_and_save_timelapse_stack_with_tracks

    src = tmp_path / "plate1" / "masks"
    src.mkdir(parents=True)

    with pytest.raises(ValueError) as excinfo:
        _visualize_and_save_timelapse_stack_with_tracks(
            [], timelapse_tracks, True, str(src), "n", False, [], "cell",
        )

    return (type(excinfo.value).__name__, gif_recorder,
            (tmp_path / "plate1" / "movies").exists())


# ---------------------------------------------------------------------------
# _display_gif
# ---------------------------------------------------------------------------

def _case_display_gif_hands_the_raw_bytes_to_an_ipython_image(tmp_path, monkeypatch):
    import spacr.plot
    from spacr.plot import _display_gif

    gif = tmp_path / "movie.gif"
    _write_tiny_gif(gif)

    shown = []
    monkeypatch.setattr(spacr.plot, "display", lambda obj: shown.append(obj))

    returned = _display_gif(str(gif))

    from IPython.display import Image as ipyimage
    assert isinstance(shown[0], ipyimage)
    assert shown[0].data == gif.read_bytes()
    return returned, len(shown), shown[0].format


def _case_display_gif_states_the_format_rather_than_leaving_it_to_ipython(
        tmp_path, monkeypatch):
    """The format must be PASSED, not inferred from the bytes.

    `test_display_gif_hands_the_raw_bytes_to_an_ipython_image` above reads
    `.format` off the constructed Image, which is green on IPython 9 whether
    or not spaCR says anything: 9.0.0 was the release that learned to
    recognise the GIF87a/GIF89a magic bytes. On IPython 8 the same call
    yields `format == 'png'` and the animation goes out with an `image/png`
    mime type -- and IPython 9 requires Python 3.11, so on the 3.9 and 3.10
    ends of the range spaCR claims, 8.x is the only IPython there is. That is
    how this reached CI as a 3.9-only failure.

    Asserting on the keyword instead of on the sniffed result is what makes
    the check independent of the installed IPython.
    """
    import spacr.plot
    from spacr.plot import _display_gif

    gif = tmp_path / "movie.gif"
    _write_tiny_gif(gif)

    calls = []

    def _record(data, *args, **kwargs):
        calls.append((data, args, kwargs))
        return object()

    monkeypatch.setattr(spacr.plot, "ipyimage", _record)
    monkeypatch.setattr(spacr.plot, "display", lambda obj: None)

    _display_gif(str(gif))

    data, args, kwargs = calls[0]
    # A missing format keyword means IPython < 9 would guess 'png' for GIF bytes.
    return len(calls), kwargs.get("format"), data == gif.read_bytes()


def _case_display_gif_missing_file_raises(tmp_path, monkeypatch):
    import spacr.plot
    from spacr.plot import _display_gif

    shown = []
    monkeypatch.setattr(spacr.plot, "display", lambda obj: shown.append(obj))

    with pytest.raises(FileNotFoundError) as excinfo:
        _display_gif(str(tmp_path / "does_not_exist.gif"))
    return type(excinfo.value).__name__, shown


_P = pytest.param
_SAME = ["crop_0.png", "crop_1.png", "crop_2.png"]
_ALL = _SAME + ["odd_crop.png"]


@pytest.mark.parametrize("helper, expected", [
    _P(_case_save_scimg_plot_defaults_to_three_channels_and_forwards_args,
       (None, [[0, 1, 2], [0], [1], [2]], [["r", "g", "b"], None, None, None],
        ["all_channels", "channel_0", "channel_1", "channel_2"]),
       id="defaults-to-three-channels"),
    _P(_case_save_scimg_plot_standardize_keeps_only_the_largest_size_group,
       (2, [_SAME, _SAME], ["all_channels", "channel_1"]),
       id="standardize-keeps-largest-size-group"),
    _P(_case_save_scimg_plot_without_standardize_uses_every_image,
       (2, [_ALL, _ALL], ["all_channels", "channel_2"]),
       id="no-standardize-uses-every-image"),
    _P(_case_save_scimg_plot_nr_imgs_subsamples_deterministically,
       (2, [2, 2], True, True, True), id="nr-imgs-subsamples-deterministically"),
    _P(_case_save_scimg_plot_nr_imgs_larger_than_folder_keeps_all,
       _ALL, id="nr-imgs-larger-than-folder-keeps-all"),
    _P(_case_save_scimg_plot_writes_pdfs_to_the_figure_folder,
       (["cell_plate1_all_channels.pdf", "cell_plate1_channel_0.pdf"], []),
       id="writes-pdfs-to-figure-folder"),
    _P(_case_save_scimg_plot_with_a_single_image,
       ["all_channels", "channel_0"], id="single-image-still-emits-figures"),
])
def test_save_scimg_plot_selects_crops_and_saves_figures(request, helper, expected):
    """_save_scimg_plot picks the crops (size group, subsample) and saves one figure per view."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_plot_cropped_arrays_2d_label_image_gets_random_object_cmap,
       (1, "Channel one, 2 (obj.)", False, 3, (0.0, 0.0, 0.0, 1.0), [3.0, 3.0]),
       id="2d-label-image-random-cmap"),
    _P(_case_plot_cropped_arrays_2d_intensity_image_keeps_requested_cmap,
       ("Channel one", "inferno", 256), id="2d-intensity-image-keeps-cmap"),
    _P(_case_plot_cropped_arrays_3d_makes_one_axis_per_channel,
       (3, ["C. 0", "C. 1, 1 (obj.)", "C. 2"], [False, False, False]),
       id="3d-one-axis-per-channel"),
    _P(_case_plot_cropped_arrays_object_count_excludes_background,
       ["Channel one, 3 (obj.)", "Channel one, 0 (obj.)",
        "Channel one, 1 (obj.)", "Channel one, 2 (obj.)"],
       id="object-count-excludes-background"),
    _P(_case_plot_cropped_arrays_single_channel_stack,
       (1, True), id="single-channel-stack"),
])
def test_plot_cropped_arrays_titles_and_colour_maps(request, helper, expected):
    """_plot_cropped_arrays titles panels, counts objects and picks the colour map."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_timelapse_save_builds_label_colormap_and_gif_path,
       (1, "plate1/movies/gif/timelapse_masks_cell_well_A01.gif",
        ["f0.tif", "f1.tif"], 3, (0, 2), []),
       id="save-builds-label-colormap-and-gif-path"),
    _P(_case_timelapse_without_save_or_plot_writes_nothing,
       (None, [], False, []), id="no-save-no-plot-writes-nothing"),
    _P(_case_timelapse_interactive_renders_the_first_frame_with_tracks,
       ("Frame: 0", False, ["1"], 3, (0, 2), [3.0, 4.0], [3.0, 4.0]),
       id="interactive-renders-first-frame"),
    _P(_case_timelapse_plot_non_interactive_displays_the_saved_gif,
       (1, True, []), id="non-interactive-displays-gif"),
    _P(_case_timelapse_interactive_save_does_not_display_the_gif,
       (1, True, []), id="interactive-save-does-not-display-gif"),
])
def test_timelapse_with_tracks_saves_or_draws(request, helper, expected):
    """The timelapse helper saves a gif, draws the first frame, or does neither."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_display_gif_hands_the_raw_bytes_to_an_ipython_image,
       (None, 1, "gif"), id="display-gif-hands-raw-bytes"),
    _P(_case_display_gif_states_the_format_rather_than_leaving_it_to_ipython,
       (1, "gif", True), id="display-gif-states-format"),
])
def test_display_gif_passes_the_bytes_and_format(request, helper, expected):
    """_display_gif gives IPython the file bytes and states the gif format itself."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _P(_case_timelapse_empty_mask_list_raises,
       ("ValueError", [], False), id="timelapse-empty-mask-list-refused"),
    _P(_case_display_gif_missing_file_raises,
       ("FileNotFoundError", []), id="display-gif-missing-file-refused"),
])
def test_timelapse_and_gif_refuse_missing_input_by_name(request, helper, expected):
    """No frames or no gif file raises the named error and writes or shows nothing."""
    observed = _call(request, helper)
    assert observed == expected
