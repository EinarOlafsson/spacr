"""The tracked-mask movie is sized from the mask, not from a wall-sized constant.

``spacr.io._save_mask_timelapse_as_gif`` opened ``plt.subplots(figsize=(50,
50))`` and wrote the animation at ``dpi=80``, so every frame came out 4000 px
on a side whatever the field was -- 64 MB of RGBA per frame for a mask that is
usually a few hundred pixels across.  It is written whenever ``save`` is true
on any of the three tracking backends, and a real timelapse is tens to hundreds
of frames, so the cost was paid on every tracking run and grew with the run's
length rather than with the field's size.

That size is also why the three ``if plot or save:`` call sites in
``spacr.timelapse`` had no test that let them run: one real movie cost seconds
and hundreds of megabytes.  The last three cases here run them for real,
because now they are cheap.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import pytest
from PIL import Image

from spacr.io import (MASK_MOVIE_DPI, MASK_MOVIE_MAX_PX, MASK_MOVIE_MIN_PX,
                      _mask_movie_frame_geometry, _save_mask_timelapse_as_gif)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _mask(height=128, width=128, labels=3):
    """A field-shaped label image with `labels` separated objects in it."""
    mask = np.zeros((height, width), dtype=np.uint16)
    step = max(1, height // (labels + 1))
    for i in range(labels):
        top = step * (i + 1) - step // 4
        mask[top:top + step // 2, step:step + step // 2] = i + 1
    return mask


def _write_gif(masks, path, filenames=None):
    _save_mask_timelapse_as_gif(
        masks, None, str(path), cmap="viridis",
        norm=mcolors.Normalize(vmin=0, vmax=int(max(m.max() for m in masks))),
        filenames=filenames or [f"plate1_A01_1_{t}.tif" for t in range(len(masks))])


# ---------------------------------------------------------------------------
# the geometry itself
# ---------------------------------------------------------------------------

def test_an_ordinary_field_keeps_its_own_resolution():
    """A field inside the band is written at the size it actually is."""
    geometry = _mask_movie_frame_geometry([np.zeros((512, 512), np.uint16)])

    assert geometry["frame_px"] == (512, 512)
    assert geometry["figsize"] == (512 / MASK_MOVIE_DPI, 512 / MASK_MOVIE_DPI)


def test_a_small_mask_is_scaled_up_far_enough_to_read_its_labels():
    """Below the floor the label numbers would be a couple of pixels tall."""
    geometry = _mask_movie_frame_geometry([np.zeros((40, 40), np.uint16)])

    assert geometry["frame_px"] == (MASK_MOVIE_MIN_PX, MASK_MOVIE_MIN_PX)


def test_a_whole_slide_field_is_capped_rather_than_written_at_full_size():
    """The cap is the whole point: 4000 px per frame is what this replaced."""
    geometry = _mask_movie_frame_geometry([np.zeros((6000, 6000), np.uint16)])

    assert geometry["frame_px"] == (MASK_MOVIE_MAX_PX, MASK_MOVIE_MAX_PX)


def test_a_non_square_field_keeps_its_aspect_ratio():
    """Scaling to the cap must not squash the field into a square."""
    geometry = _mask_movie_frame_geometry([np.zeros((2048, 1024), np.uint16)])

    width, height = geometry["frame_px"]
    assert height == MASK_MOVIE_MAX_PX
    assert width == MASK_MOVIE_MAX_PX // 2


def test_the_largest_frame_decides_so_a_grown_mask_is_not_cropped():
    """Ragged input is real: a batch can hold frames of two shapes."""
    geometry = _mask_movie_frame_geometry(
        [np.zeros((100, 700), np.uint16), np.zeros((900, 100), np.uint16)])

    assert geometry["frame_px"] == (700, 900)


def test_lettering_grows_with_the_frame_instead_of_staying_at_24_point():
    """24 pt was chosen for a 50-inch canvas; on a 5-inch one it is the picture."""
    small = _mask_movie_frame_geometry([np.zeros((320, 320), np.uint16)])
    large = _mask_movie_frame_geometry([np.zeros((1024, 1024), np.uint16)])

    assert large["label_pt"] > small["label_pt"]
    assert small["title_pt"] > small["label_pt"]     # the heading reads first
    # and neither collapses to something invisible
    assert small["label_pt"] >= 5.0


def test_a_movie_with_no_frame_to_measure_is_refused():
    """An empty animation writes a GIF no player can open; say so instead."""
    with pytest.raises(ValueError) as exc:
        _mask_movie_frame_geometry([])

    assert "no frame" in str(exc.value)


def test_a_frame_of_zero_width_is_refused_rather_than_divided_by():
    """A degenerate mask reaches the same refusal as an empty list."""
    with pytest.raises(ValueError):
        _mask_movie_frame_geometry([np.zeros((0, 8), np.uint16)])


def test_a_one_dimensional_array_is_refused_as_a_frame_not_as_an_iterable():
    """A corrupt .npy is the ordinary source of this, and it has to say so."""
    with pytest.raises(ValueError) as exc:
        _mask_movie_frame_geometry([np.zeros(8, np.uint16)])

    assert "no frame" in str(exc.value)


# ---------------------------------------------------------------------------
# the file that actually lands on disk
# ---------------------------------------------------------------------------

def test_the_gif_written_for_an_ordinary_field_is_the_field_not_4000_px(tmp_path):
    """The measurement this whole change exists for, taken on a real file."""
    out = tmp_path / "movie.gif"
    _write_gif([_mask(512, 512) for _ in range(3)], out)

    with Image.open(out) as im:
        assert im.format == "GIF"
        assert im.n_frames == 3
        assert im.size == (512, 512)
        # 512 x 512 x 4 is 1 MB of RGBA per frame; the old canvas was 64 MB.
        assert im.size[0] * im.size[1] * 4 < 4_000_000


def test_the_frame_counter_reaches_the_file(tmp_path):
    """It never used to.

    The counter was drawn by ``ax.set_title`` into a figure whose axes had been
    handed every last inch by ``subplots_adjust(top=1)``, so it was clipped
    away on every frame: zero lit pixels in the top 5 % of the canvas against
    222 in the bottom 5 % where the filename sits.  Both bands are now
    reserved, and both captions land inside the picture.
    """
    out = tmp_path / "captions.gif"
    _write_gif([_mask(256, 256)], out, filenames=["plate1_A01_1_1.tif"])

    with Image.open(out) as im:
        frame = np.asarray(im.convert("L"))
    band = max(1, frame.shape[0] // 20)

    assert (frame[:band] > 200).sum() > 0, "the frame counter is off-canvas again"
    assert (frame[-band:] > 200).sum() > 0, "the filename caption went missing"


def test_every_frame_of_a_ragged_batch_fits_in_the_canvas(tmp_path):
    """The largest frame sizes the movie, so no frame is cropped by the writer."""
    masks = [_mask(64, 200), _mask(200, 64)]
    out = tmp_path / "ragged.gif"
    _write_gif(masks, out)

    with Image.open(out) as im:
        assert im.n_frames == 2
        width, height = im.size
    assert width / height == pytest.approx(200 / 200, abs=0.02)


# ---------------------------------------------------------------------------
# the three `if plot or save:` call sites, run for real
# ---------------------------------------------------------------------------

def _moving_stack(n_frames=3, height=48, width=48):
    stack = np.zeros((n_frames, height, width), dtype=np.uint16)
    for t in range(n_frames):
        stack[t, 8 + t:16 + t, 8:16] = 1
        stack[t, 28:36, 8 + t:16 + t] = 2
    return stack


def _gif_written_under(root):
    return sorted(
        os.path.join(base, f)
        for base, _, files in os.walk(str(root))
        for f in files if f.endswith(".gif"))


def test_btrack_with_nothing_to_track_still_writes_its_movie(tmp_path):
    """The empty-segmentation early return has its own visualiser call.

    It is the branch that runs when a field segmented to nothing, which is
    exactly the run whose movie a user goes looking for.  Nothing exercised it
    with ``save=True`` before, because doing so wrote a 4000 px animation.
    """
    from spacr.timelapse import _btrack_track_cells

    src = tmp_path / "masks"
    src.mkdir()
    masks = np.zeros((3, 48, 48), dtype=np.int32)

    _btrack_track_cells(
        src=str(src), name="plate1_A01_1",
        batch_filenames=[f"plate1_A01_1_t{t}.tif" for t in range(3)],
        object_type="cell", plot=False, save=True, masks_3D=masks,
        mode="btrack", timelapse_remove_transient=False, radius=5, n_jobs=1)

    gifs = _gif_written_under(tmp_path)
    assert gifs, "save=True wrote no movie"
    with Image.open(gifs[0]) as im:
        assert im.size == (MASK_MOVIE_MIN_PX, MASK_MOVIE_MIN_PX)
        assert im.n_frames == 3


def test_trackastra_writes_its_movie_at_the_size_of_the_field(tmp_path,
                                                              monkeypatch):
    """The trackastra backend's `if plot or save:` line, driven end to end."""
    from tests.test_timelapse_trackastra import _install_stub_trackastra
    from spacr.timelapse import _trackastra_track_cells

    masks = _moving_stack()
    _install_stub_trackastra(monkeypatch, relabelled=masks)

    src = tmp_path / "run" / "batch"
    src.parent.mkdir(parents=True, exist_ok=True)
    _trackastra_track_cells(
        src=str(src), name="b1",
        batch_filenames=[f"plate1_A01_1_t{t}.tif" for t in range(3)],
        object_type="cell", masks=masks, save=True)

    gifs = _gif_written_under(tmp_path)
    assert gifs, "save=True wrote no movie"
    with Image.open(gifs[0]) as im:
        assert im.size == (MASK_MOVIE_MIN_PX, MASK_MOVIE_MIN_PX)


def test_ultrack_writes_its_movie_at_the_size_of_the_field(tmp_path, monkeypatch):
    """The ultrack backend's `if plot or save:` line, driven end to end."""
    from tests.test_timelapse_ultrack import _install_stub_ultrack
    from spacr.timelapse import _ultrack_track_cells

    masks = _moving_stack()
    _install_stub_ultrack(monkeypatch, relabelled=masks)

    src = tmp_path / "run" / "batch"
    src.parent.mkdir(parents=True, exist_ok=True)
    _ultrack_track_cells(
        src=str(src), name="b1",
        batch_filenames=[f"plate1_A01_1_t{t}.tif" for t in range(3)],
        object_type="cell", masks=masks, save=True)

    gifs = _gif_written_under(tmp_path)
    assert gifs, "save=True wrote no movie"
    with Image.open(gifs[0]) as im:
        assert im.size == (MASK_MOVIE_MIN_PX, MASK_MOVIE_MIN_PX)
