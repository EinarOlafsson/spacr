"""CPU coverage for the single-cell-image / timelapse block of spacr.plot.

Covers:
    _save_scimg_plot                                (grid figure per channel)
    _plot_cropped_arrays                            (2-D and 3-D stacks)
    _visualize_and_save_timelapse_stack_with_tracks (gif + interactive frame)
    _display_gif

Everything runs headless on Agg with tiny synthetic PNGs / label masks.
The two IO helpers that would otherwise render 20x20 inch figures at
300 dpi (`spacr.io._save_figure`) or animate a 4000x4000 px GIF
(`spacr.io._save_mask_timelapse_as_gif`) are replaced by recorders in most
tests so the assertions can inspect exactly what spacr.plot handed them;
one test lets the real `_save_figure` run so the emitted PDFs are checked
on disk.
"""
from __future__ import annotations

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
    (src / "notes.txt").write_text("not an image")

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

def test_save_scimg_plot_defaults_to_three_channels_and_forwards_args(
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
    assert out is None

    assert [c["channel_indices"] for c in grid_recorder] == [[0, 1, 2], [0], [1], [2]]
    # channel names are only stamped on the combined figure
    assert [c["channel_names"] for c in grid_recorder] == [["r", "g", "b"], None, None, None]
    # scalar plot arguments reach _plot_images_on_grid in the right slots
    for call in grid_recorder:
        assert call["um_per_pixel"] == 0.25
        assert call["scale_bar_length_um"] == 7
        assert call["fontsize"] == 11
        assert call["show_filename"] is False
        assert call["plot"] is False

    assert [c["text"] for c in figure_recorder] == [
        "all_channels", "channel_0", "channel_1", "channel_2",
    ]
    assert all(c["src"] == scimg_dir["src"] for c in figure_recorder)
    # every saved figure is the one the grid plotter just built
    assert all(c["n_axes"] == 1 for c in figure_recorder)


def test_save_scimg_plot_standardize_keeps_only_the_largest_size_group(
    scimg_dir, grid_recorder, figure_recorder
):
    """standardize=True drops the odd-sized crop; the .txt file is never listed."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=None, channel_indices=[1],
                     standardize=True)

    assert len(grid_recorder) == 2  # combined + channel_1
    for call in grid_recorder:
        assert call["files"] == scimg_dir["same"]
        assert scimg_dir["odd"] not in call["files"]
    assert [c["text"] for c in figure_recorder] == ["all_channels", "channel_1"]


def test_save_scimg_plot_without_standardize_uses_every_image(
    scimg_dir, grid_recorder, figure_recorder
):
    """standardize=False skips _find_similar_sized_images entirely."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=None, channel_indices=[2],
                     standardize=False)

    assert len(grid_recorder) == 2
    for call in grid_recorder:
        assert call["files"] == scimg_dir["all"]
    assert [c["text"] for c in figure_recorder] == ["all_channels", "channel_2"]


def test_save_scimg_plot_nr_imgs_subsamples_deterministically(
    scimg_dir, grid_recorder, figure_recorder
):
    """nr_imgs < len(files) -> a seeded random subset, identical for every figure."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=2, channel_indices=[0],
                     standardize=False)

    assert len(grid_recorder) == 2
    for call in grid_recorder:
        assert len(call["files"]) == 2
        assert set(call["files"]).issubset(set(scimg_dir["all"]))
    # random.seed(42) is re-applied per figure, so both figures show the same crops
    assert grid_recorder[0]["files"] == grid_recorder[1]["files"]

    # ...and the seeding makes the choice reproducible across calls
    grid_recorder.clear()
    _save_scimg_plot(scimg_dir["src"], nr_imgs=2, channel_indices=[0],
                     standardize=False)
    assert grid_recorder[0]["files"] == grid_recorder[1]["files"]


def test_save_scimg_plot_nr_imgs_larger_than_folder_keeps_all(
    scimg_dir, grid_recorder, figure_recorder
):
    """nr_imgs >= number of images -> no sampling, the full list is plotted."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=99, channel_indices=[0],
                     standardize=False)

    assert grid_recorder[0]["files"] == scimg_dir["all"]


def test_save_scimg_plot_writes_pdfs_to_the_figure_folder(scimg_dir):
    """End-to-end with the real grid plotter and the real _save_figure."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=2, channel_indices=[0],
                     um_per_pixel=0.5, scale_bar_length_um=2, standardize=True,
                     plot=False)

    fig_dir = scimg_dir["figure_dir"]
    assert sorted(p.name for p in fig_dir.glob("*.pdf")) == [
        "cell_plate1_all_channels.pdf",
        "cell_plate1_channel_0.pdf",
    ]
    for pdf in fig_dir.glob("*.pdf"):
        data = pdf.read_bytes()
        assert data.startswith(b"%PDF")
        assert len(data) > 1000
    # _save_figure closes each figure it writes
    assert plt.get_fignums() == []


def test_save_scimg_plot_with_a_single_image(scimg_dir, figure_recorder):
    """A one-crop folder/sample is a legal input and should still emit figures."""
    from spacr.plot import _save_scimg_plot

    _save_scimg_plot(scimg_dir["src"], nr_imgs=1, channel_indices=[0],
                     standardize=False)

    assert [c["text"] for c in figure_recorder] == ["all_channels", "channel_0"]


# ---------------------------------------------------------------------------
# _plot_cropped_arrays
# ---------------------------------------------------------------------------

def test_plot_cropped_arrays_2d_label_image_gets_random_object_cmap():
    """<= threshold unique values -> random label cmap + object count in title."""
    from spacr.plot import _plot_cropped_arrays

    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[2:6, 2:6] = 1
    arr[9:13, 9:13] = 2
    fig = _plot_cropped_arrays(arr, "field_1.npy", figuresize=3)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_title() == "Channel one, 3 (obj.)"  # 0, 1, 2
    assert ax.axison is False
    assert len(ax.images) == 1
    np.testing.assert_array_equal(np.asarray(ax.images[0].get_array()), arr)
    cmap = ax.images[0].get_cmap()
    assert isinstance(cmap, ListedColormap)
    assert cmap.N == 3  # background + 2 objects
    assert tuple(cmap.colors[0]) == (0.0, 0.0, 0.0, 1.0)
    assert fig.get_size_inches().tolist() == [3.0, 3.0]


def test_plot_cropped_arrays_2d_intensity_image_keeps_requested_cmap():
    """> threshold unique values -> the caller's cmap and an unannotated title."""
    from spacr.plot import _plot_cropped_arrays

    arr = np.arange(256, dtype=np.uint16).reshape(16, 16)
    fig = _plot_cropped_arrays(arr, "field_2.npy", figuresize=2, cmap="inferno",
                               threshold=10)

    ax = fig.axes[0]
    assert ax.get_title() == "Channel one"  # no ", N (obj.)" suffix
    # the registered 'inferno' map is used verbatim, not a freshly drawn
    # random label map (which would be a 3-colour ListedColormap here)
    cmap = ax.images[0].get_cmap()
    assert cmap.name == "inferno"
    assert cmap.N == plt.get_cmap("inferno").N == 256
    np.testing.assert_allclose(cmap(0.0), plt.get_cmap("inferno")(0.0))


def test_plot_cropped_arrays_3d_makes_one_axis_per_channel():
    from spacr.plot import _plot_cropped_arrays

    stack = np.zeros((8, 8, 3), dtype=np.uint16)
    stack[:, :, 0] = np.arange(64).reshape(8, 8)      # 64 unique -> label cmap
    stack[2:5, 2:5, 1] = 1                            # 2 unique  -> label cmap
    stack[:, :, 2] = np.arange(64).reshape(8, 8) * 9  # 64 unique

    fig = _plot_cropped_arrays(stack, "field_3.npy", figuresize=4, threshold=10)

    assert len(fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes] == [
        "C. 0", "C. 1, 2 (obj.)", "C. 2",
    ]
    np.testing.assert_array_equal(
        np.asarray(fig.axes[1].images[0].get_array()), stack[:, :, 1]
    )
    assert all(ax.axison is False for ax in fig.axes)


def test_plot_cropped_arrays_single_channel_stack():
    """A stack with exactly one channel is a valid shape and must not crash."""
    from spacr.plot import _plot_cropped_arrays

    stack = np.zeros((8, 8, 1), dtype=np.uint16)
    stack[2:5, 2:5, 0] = 1
    fig = _plot_cropped_arrays(stack, "field_4.npy", figuresize=2)

    assert len(fig.axes) == 1
    assert fig.axes[0].get_title().startswith("C. 0")


# ---------------------------------------------------------------------------
# _visualize_and_save_timelapse_stack_with_tracks
# ---------------------------------------------------------------------------

def test_timelapse_save_builds_label_colormap_and_gif_path(
    tmp_path, timelapse_masks, timelapse_tracks, gif_recorder
):
    from spacr.plot import _visualize_and_save_timelapse_stack_with_tracks

    src = tmp_path / "plate1" / "masks"
    src.mkdir(parents=True)

    _visualize_and_save_timelapse_stack_with_tracks(
        timelapse_masks, timelapse_tracks, True, str(src), "well_A01",
        False, ["f0.tif", "f1.tif"], "cell",
    )

    assert len(gif_recorder) == 1
    call = gif_recorder[0]

    expected = tmp_path / "plate1" / "movies" / "gif" / "timelapse_masks_cell_well_A01.gif"
    assert call["path"] == str(expected)
    assert expected.parent.is_dir()
    assert expected.is_file()

    assert call["masks"] is timelapse_masks
    assert call["tracks_df"] is timelapse_tracks
    assert call["filenames"] == ["f0.tif", "f1.tif"]

    cmap = call["cmap"]
    assert isinstance(cmap, ListedColormap)
    assert cmap.N == 3                                   # highest label (2) + 1
    assert np.allclose(np.asarray(cmap.colors)[:, 3], 1.0)   # fully opaque
    assert tuple(cmap.colors[0]) == (0.0, 0.0, 0.0, 1.0)     # background is black
    assert not np.allclose(cmap.colors[1], cmap.colors[0])

    norm = call["norm"]
    assert (norm.vmin, norm.vmax) == (0, 2)

    # plot=False -> no figure was opened
    assert plt.get_fignums() == []


def test_timelapse_without_save_or_plot_writes_nothing(
    tmp_path, timelapse_masks, timelapse_tracks, gif_recorder
):
    from spacr.plot import _visualize_and_save_timelapse_stack_with_tracks

    src = tmp_path / "plate1" / "masks"
    src.mkdir(parents=True)

    out = _visualize_and_save_timelapse_stack_with_tracks(
        timelapse_masks, timelapse_tracks, False, str(src), "n", False,
        ["f0.tif", "f1.tif"], "nucleus",
    )

    assert out is None
    assert gif_recorder == []
    assert not (tmp_path / "plate1" / "movies").exists()
    assert plt.get_fignums() == []


def test_timelapse_interactive_renders_the_first_frame_with_tracks(
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
    assert ax.get_title() == "Frame: 0"
    assert ax.axison is False

    # frame 0 carries a single object -> a single label annotation
    assert [t.get_text() for t in ax.texts] == ["1"]
    # the mask is drawn with the shared label cmap / norm
    assert len(ax.images) == 1
    np.testing.assert_array_equal(
        np.asarray(ax.images[0].get_array()), timelapse_masks[0]
    )
    assert ax.images[0].get_cmap().N == 3
    assert (ax.images[0].norm.vmin, ax.images[0].norm.vmax) == (0, 2)

    # one black polyline per track id
    assert len(ax.lines) == timelapse_tracks["track_id"].nunique()
    for line in ax.lines:
        assert line.get_color() == "k"
        assert line.get_linewidth() == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [3.0, 4.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [3.0, 4.0])


def test_timelapse_plot_non_interactive_displays_the_saved_gif(
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
    assert len(shown) == 1
    from IPython.display import Image as ipyimage
    assert isinstance(shown[0], ipyimage)
    assert shown[0].data == open(gif_path, "rb").read()
    assert shown[0].data.startswith(b"GIF")
    # non-interactive -> the per-frame matplotlib figure was never built
    assert plt.get_fignums() == []


def test_timelapse_interactive_save_does_not_display_the_gif(
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

    assert len(gif_recorder) == 1
    assert os.path.isfile(gif_recorder[0]["path"])
    assert shown == []  # _display_gif was not reached


def test_timelapse_empty_mask_list_raises(tmp_path, timelapse_tracks, gif_recorder):
    """No frames -> max() over an empty generator; nothing is written."""
    from spacr.plot import _visualize_and_save_timelapse_stack_with_tracks

    src = tmp_path / "plate1" / "masks"
    src.mkdir(parents=True)

    with pytest.raises(ValueError):
        _visualize_and_save_timelapse_stack_with_tracks(
            [], timelapse_tracks, True, str(src), "n", False, [], "cell",
        )

    assert gif_recorder == []
    assert not (tmp_path / "plate1" / "movies").exists()


# ---------------------------------------------------------------------------
# _display_gif
# ---------------------------------------------------------------------------

def test_display_gif_hands_the_raw_bytes_to_an_ipython_image(tmp_path, monkeypatch):
    import spacr.plot
    from spacr.plot import _display_gif

    gif = tmp_path / "movie.gif"
    _write_tiny_gif(gif)

    shown = []
    monkeypatch.setattr(spacr.plot, "display", lambda obj: shown.append(obj))

    assert _display_gif(str(gif)) is None

    from IPython.display import Image as ipyimage
    assert len(shown) == 1
    assert isinstance(shown[0], ipyimage)
    assert shown[0].data == gif.read_bytes()
    assert shown[0].format == "gif"


def test_display_gif_missing_file_raises(tmp_path, monkeypatch):
    import spacr.plot
    from spacr.plot import _display_gif

    shown = []
    monkeypatch.setattr(spacr.plot, "display", lambda obj: shown.append(obj))

    with pytest.raises(FileNotFoundError):
        _display_gif(str(tmp_path / "does_not_exist.gif"))
    assert shown == []
