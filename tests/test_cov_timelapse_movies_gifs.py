"""CPU coverage for the movie / GIF helpers at the top of spacr.timelapse:

    _npz_to_movie          (dtype + channel-count normalisation, codec choice)
    _scmovie               (per-object single-cell movies from PNG folders)
    _masks_to_gif          (random label colormap + delegation to spacr.io)
    _timelapse_masks_to_gif(grouping .npy frames by plate/well/field)

Everything runs headless and offline.  The matplotlib animation inside ``spacr.io._save_mask_timelapse_as_gif`` is
stubbed out for most tests and exercised for real exactly once so the tests
stay fast.  It is no longer the 4000x4000 canvas it used to be -- the frame is
sized from the mask now -- but a real animation is still the slowest thing
here.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


class _FakeWriter:
    """Stand-in for cv2.VideoWriter that records everything handed to it."""

    instances: list = []

    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False
        type(self).instances.append(self)

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def isOpened(self):
        return True

    def release(self):
        self.released = True


def _fake_writer(monkeypatch):
    """Patch cv2.VideoWriter with the recorder above and hand back the registry."""
    _FakeWriter.instances = []
    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)
    return _FakeWriter


def _gif_spy(monkeypatch):
    """Replace the expensive spacr.io GIF animator with a recorder."""
    import spacr.io as sio

    calls = []

    def _spy(masks, tracks_df, path, cmap, norm, filenames):
        calls.append(
            {"masks": np.asarray(masks), "tracks_df": tracks_df, "path": path,
             "cmap": cmap, "norm": norm, "filenames": list(filenames)}
        )
        # emit a marker file so callers can assert the destination folder exists
        with open(path, "wb") as fh:
            fh.write(b"GIF89a-stub")

    monkeypatch.setattr(sio, "_save_mask_timelapse_as_gif", _spy)
    return calls


def _write_png(path, value, shape):
    img = np.full((shape[0], shape[1], 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# _npz_to_movie
# ---------------------------------------------------------------------------

def _scn_npz_to_movie_float32_is_clipped_and_scaled(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """float32 frames are clipped to [0, 1] then scaled to uint8 and greyscale-expanded."""
    from spacr.timelapse import _npz_to_movie

    frame = np.zeros((64, 64), dtype=np.float32)
    frame[0:10, 0:10] = 0.5
    frame[10:20, 0:10] = 3.0     # above 1 -> clipped to 255
    frame[20:25, 0:10] = -2.0    # below 0 -> clipped to 0

    out = str(tmp_path / "movie.avi")
    _npz_to_movie([frame, frame], ["frame_0.npy", "frame_1.npy"], out, fps=7)

    (writer,) = fake_writer.instances
    assert writer.path == out
    assert writer.fps == 7
    assert writer.size == (64, 64)
    assert writer.fourcc == cv2.VideoWriter_fourcc(*"XVID")   # non-mp4 -> XVID
    assert writer.released is True
    assert len(writer.frames) == 2

    written = writer.frames[0]
    assert written.shape == (64, 64, 3)
    assert written.dtype == np.uint8
    # grey -> BGR means all three channels are identical
    assert np.array_equal(written[..., 0], written[..., 2])
    assert written[5, 5, 0] == 127        # 0.5 * 255 truncated
    assert written[15, 5, 0] == 255       # clipped high
    assert written[22, 5, 0] == 0         # clipped low
    return [int(written[5, 5, 0]), int(written[15, 5, 0]), int(written[22, 5, 0])]


def _scn_npz_to_movie_uint16_is_scaled_to_8bit(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """uint16 frames go through convertScaleAbs (65535 -> 255)."""
    from spacr.timelapse import _npz_to_movie

    frame = np.zeros((64, 64), dtype=np.uint16)
    frame[0:10, 0:10] = 65535
    frame[10:20, 0:10] = 32768

    _npz_to_movie([frame], ["f.npy"], str(tmp_path / "m.avi"))

    written = fake_writer.instances[0].frames[0]
    assert written.dtype == np.uint8
    assert written.shape == (64, 64, 3)
    assert written[5, 5, 0] == 255
    assert written[15, 5, 0] == pytest.approx(128, abs=1)
    return [int(written[5, 5, 0]), int(written[15, 5, 0])]


def _scn_npz_to_movie_single_channel_axis_is_expanded(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """(H, W, 1) frames are RGB internally and encoded at the writer boundary."""
    from spacr.timelapse import _npz_to_movie

    frame = np.zeros((64, 64, 1), dtype=np.uint8)
    frame[0:10, 0:10, 0] = 200

    _npz_to_movie([frame], ["f.npy"], str(tmp_path / "m.avi"))

    written = fake_writer.instances[0].frames[0]
    assert written.shape == (64, 64, 3)
    assert written[5, 5].tolist() == [200, 200, 200]
    return written[5, 5].tolist()


def _scn_npz_to_movie_two_channel_becomes_red_green(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """Two RGB planes are converted only when handed to the BGR writer."""
    from spacr.timelapse import _npz_to_movie

    frame = np.zeros((64, 64, 2), dtype=np.uint8)
    frame[0:10, 0:10, 0] = 111    # "red"
    frame[0:10, 0:10, 1] = 222    # "green"

    _npz_to_movie([frame], ["f.npy"], str(tmp_path / "m.avi"))

    written = fake_writer.instances[0].frames[0]
    assert written.shape == (64, 64, 3)
    assert written.dtype == np.uint8
    assert written[5, 5, 0] == 0
    assert written[5, 5, 1] == 222
    assert written[5, 5, 2] == 111
    return written[5, 5].tolist()


def _scn_npz_to_movie_uint16_two_channel_is_scaled_then_packed(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """The dtype branch runs before the channel branch for multi-channel frames."""
    from spacr.timelapse import _npz_to_movie

    frame = np.zeros((64, 64, 2), dtype=np.uint16)
    frame[0:10, 0:10, 0] = 65535
    frame[0:10, 0:10, 1] = 65535 // 2

    _npz_to_movie([frame], ["f.npy"], str(tmp_path / "m.avi"))

    written = fake_writer.instances[0].frames[0]
    assert written.dtype == np.uint8
    assert written[5, 5, 0] == 0
    assert written[5, 5, 1] == pytest.approx(128, abs=1)
    assert written[5, 5, 2] == 255
    return written[5, 5].tolist()


def _scn_npz_to_movie_three_channel_is_rgb_to_bgr_swapped(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """RGB input is converted to the BGR order OpenCV expects."""
    from spacr.timelapse import _npz_to_movie

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[0:10, 0:10] = (10, 20, 30)   # R, G, B

    _npz_to_movie([frame], ["f.npy"], str(tmp_path / "m.mp4"))

    writer = fake_writer.instances[0]
    assert writer.fourcc == cv2.VideoWriter_fourcc(*"mp4v")   # .mp4 -> mp4v
    written = writer.frames[0]
    assert written[5, 5].tolist() == [30, 20, 10]             # B, G, R
    return written[5, 5].tolist()


def _scn_npz_to_movie_stamps_filename_on_every_frame(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """The per-frame filename is drawn in white near the bottom of the frame."""
    from spacr.timelapse import _npz_to_movie

    frames = [np.zeros((64, 200), dtype=np.uint8) for _ in range(2)]
    _npz_to_movie(frames, ["AAAA", "BBBB"], str(tmp_path / "m.avi"))

    written = fake_writer.instances[0].frames
    # nothing above the text baseline, bright (anti-aliased) pixels in the text band
    for f in written:
        assert f[:25, :, :].max() == 0
        assert f[25:, :, :].max() > 200
        # the stamp is greyscale-white: all three channels agree
        assert np.array_equal(f[..., 0], f[..., 2])
    # the two labels are different strings -> different pixel stamps
    assert not np.array_equal(written[0], written[1])
    return [int(f[:25, :, :].max()) for f in written]


def _scn_npz_to_movie_writes_real_containers(tmp_path, monkeypatch):
    """End-to-end with the real cv2 writer: an mp4 and an avi land on disk."""
    from spacr.timelapse import _npz_to_movie

    frames = [(np.full((64, 64, 3), i * 40, dtype=np.uint8)) for i in range(1, 4)]
    mp4 = str(tmp_path / "real.mp4")
    avi = str(tmp_path / "real.avi")
    _npz_to_movie(frames, ["a", "b", "c"], mp4, fps=5)
    _npz_to_movie(frames, ["a", "b", "c"], avi, fps=5)

    assert os.path.getsize(mp4) > 0 and os.path.getsize(avi) > 0
    with open(mp4, "rb") as fh:
        assert fh.read(12)[4:8] == b"ftyp"      # ISO base media (mp4v)
    with open(avi, "rb") as fh:
        head = fh.read(12)
    assert head[:4] == b"RIFF" and head[8:12] == b"AVI "

    cap = cv2.VideoCapture(mp4)
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
        counted = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    finally:
        cap.release()
    return counted


# ---------------------------------------------------------------------------
# _scmovie
# ---------------------------------------------------------------------------

def _scn_scmovie_writes_one_movie_per_object(tmp_path, monkeypatch):
    """Each (plate, well, field, object) group becomes its own mp4 under movies/."""
    from spacr.timelapse import _scmovie

    folder = tmp_path / "pngs"
    folder.mkdir()
    for t in range(3):
        for obj in (1, 2):
            _write_png(folder / f"p1_A01_f01_{t}_{obj}.png", 30 * (t + 1), (24, 24))

    _scmovie([str(folder)])

    movies = sorted(os.listdir(folder / "movies"))
    assert movies == ["p1_A01_f01_1.mp4", "p1_A01_f01_2.mp4"]
    for name in movies:
        assert os.path.getsize(folder / "movies" / name) > 0
    return movies


def _scn_scmovie_orders_frames_by_time_and_pads_to_max_size(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """Frames are time-sorted and zero-padded to the largest frame in the group."""
    from spacr.timelapse import _scmovie

    folder = tmp_path / "pngs"
    folder.mkdir()
    # deliberately different sizes; the t=1 frame is the biggest
    _write_png(folder / "p1_A01_f01_2_7.png", 50, (10, 12))
    _write_png(folder / "p1_A01_f01_0_7.png", 10, (8, 9))
    _write_png(folder / "p1_A01_f01_1_7.png", 30, (14, 20))

    _scmovie([str(folder)])

    (writer,) = fake_writer.instances
    assert writer.path == str(folder / "movies" / "p1_A01_f01_7.mp4")
    assert writer.size == (20, 14)          # (max_width, max_height)
    assert writer.fps == 10
    assert writer.released is True
    assert len(writer.frames) == 3

    # every written frame was padded to the same canvas
    assert {f.shape for f in writer.frames} == {(14, 20, 3)}
    # time order 0 -> 1 -> 2 recovered from the encoded grey values
    assert [int(f[0, 0, 0]) for f in writer.frames] == [10, 30, 50]
    # the smallest frame (8x9) is zero outside its own extent
    small = writer.frames[0]
    assert small[:8, :9].min() == 10
    assert small[8:, :].max() == 0
    assert small[:, 9:].max() == 0
    return (writer.size, writer.fps, [int(f[0, 0, 0]) for f in writer.frames])


def _scn_scmovie_deduplicates_folder_paths(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """A repeated folder path is processed exactly once."""
    from spacr.timelapse import _scmovie

    folder = tmp_path / "pngs"
    folder.mkdir()
    _write_png(folder / "p1_A01_f01_0_1.png", 20, (8, 8))
    _write_png(folder / "p1_A01_f01_1_1.png", 40, (8, 8))

    _scmovie([str(folder), str(folder), str(folder)])

    assert len(fake_writer.instances) == 1
    assert len(fake_writer.instances[0].frames) == 2
    return len(fake_writer.instances[0].frames)


def _scn_scmovie_ignores_non_png_and_unparsable_names(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """Non-png files and png names that do not match the pattern are skipped."""
    from spacr.timelapse import _scmovie

    folder = tmp_path / "pngs"
    folder.mkdir()
    _write_png(folder / "not_a_match.png", 10, (8, 8))        # too few fields
    _write_png(folder / "p1_A01_f01_x_1.png", 10, (8, 8))     # time is not digits
    (folder / "notes.txt").write_text("ignore me", encoding="utf-8")
    (folder / "array.npy").write_bytes(b"\x00\x01")

    _scmovie([str(folder)])

    assert os.path.isdir(folder / "movies")
    assert os.listdir(folder / "movies") == []
    assert fake_writer.instances == []
    return os.listdir(folder / "movies")


def _scn_scmovie_separates_wells_and_fields(tmp_path, monkeypatch):
    fake_writer = _fake_writer(monkeypatch)
    """Different well/field keys never share a movie."""
    from spacr.timelapse import _scmovie

    folder = tmp_path / "pngs"
    folder.mkdir()
    _write_png(folder / "p1_A01_f01_0_1.png", 10, (8, 8))
    _write_png(folder / "p1_A02_f01_0_1.png", 20, (8, 8))
    _write_png(folder / "p1_A02_f02_0_1.png", 30, (8, 8))

    _scmovie([str(folder)])

    names = sorted(os.path.basename(w.path) for w in fake_writer.instances)
    assert names == ["p1_A01_f01_1.mp4", "p1_A02_f01_1.mp4", "p1_A02_f02_1.mp4"]
    assert all(len(w.frames) == 1 for w in fake_writer.instances)
    return names


# ---------------------------------------------------------------------------
# _masks_to_gif
# ---------------------------------------------------------------------------

def _scn_masks_to_gif_builds_label_colormap_and_delegates(tmp_path, monkeypatch):
    gif_spy = _gif_spy(monkeypatch)
    """A random opaque colour per label, black background, and the io hand-off."""
    from spacr.timelapse import _masks_to_gif

    m0 = np.zeros((8, 8), dtype=np.uint16)
    m0[1:3, 1:3] = 1
    m1 = np.zeros((8, 8), dtype=np.uint16)
    m1[4:6, 4:6] = 4          # highest label across the whole sequence

    _masks_to_gif([m0, m1], str(tmp_path), "1_A01_1", ["f0.npy", "f1.npy"], "cell")

    (call,) = gif_spy
    assert call["path"] == str(tmp_path / "timelapse_masks_cell_1_A01_1.gif")
    assert call["tracks_df"] is None
    assert call["filenames"] == ["f0.npy", "f1.npy"]
    assert call["masks"].shape == (2, 8, 8)

    cmap = call["cmap"]
    assert cmap.N == 5                                    # highest_label + 1
    colors = np.asarray(cmap.colors)
    assert colors.shape == (5, 4)
    assert np.allclose(colors[:, 3], 1.0)                 # fully opaque
    assert cmap(0) == (0.0, 0.0, 0.0, 1.0)                # background is black
    assert colors[1:, :3].max() > 0.0                     # real labels got colour

    norm = call["norm"]
    assert (norm.vmin, norm.vmax) == (0, 4)
    assert norm(4) == pytest.approx(1.0)
    assert os.path.exists(call["path"])
    return (cmap.N, (norm.vmin, norm.vmax), os.path.basename(call["path"]))


def _scn_masks_to_gif_single_label_sequence(tmp_path, monkeypatch):
    gif_spy = _gif_spy(monkeypatch)
    """An all-background sequence still yields a 1-entry black colormap."""
    from spacr.timelapse import _masks_to_gif

    masks = np.zeros((3, 6, 6), dtype=np.uint16)
    _masks_to_gif(masks, str(tmp_path), "empty", ["a", "b", "c"], "pathogen")

    (call,) = gif_spy
    assert call["cmap"].N == 1
    assert call["cmap"](0) == (0.0, 0.0, 0.0, 1.0)
    assert (call["norm"].vmin, call["norm"].vmax) == (0, 0)
    assert call["path"].endswith("timelapse_masks_pathogen_empty.gif")
    return (call["cmap"].N, (call["norm"].vmin, call["norm"].vmax))


def _scn_masks_to_gif_writes_a_real_gif(tmp_path, monkeypatch):
    """One un-stubbed run: matplotlib really renders and pillow really writes.

    The frame comes out at the movie's own minimum rather than at the old
    fixed 4000 px: an 8 x 8 mask is scaled up to MASK_MOVIE_MIN_PX so its
    label numbers are legible, and no further.
    """
    from PIL import Image
    from spacr.io import MASK_MOVIE_MIN_PX
    from spacr.timelapse import _masks_to_gif

    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[1:3, 1:3] = 1
    mask[5:7, 5:7] = 2

    _masks_to_gif(np.array([mask]), str(tmp_path), "real", ["f0.npy"], "cell")

    gif = tmp_path / "timelapse_masks_cell_real.gif"
    assert gif.is_file() and gif.stat().st_size > 0
    with Image.open(gif) as im:
        assert im.format == "GIF"
        assert im.n_frames == 1
        assert im.size == (MASK_MOVIE_MIN_PX, MASK_MOVIE_MIN_PX)
        im_n_frames, im_size = im.n_frames, im.size
    return (im_n_frames, im_size == (MASK_MOVIE_MIN_PX, MASK_MOVIE_MIN_PX))


def _scn_masks_to_gif_display_helper_reads_and_displays_bytes(tmp_path, monkeypatch):
    """Exercise the nested _display_gif closure, which the caller leaves unused.

    It is a local function of _masks_to_gif with no external handle, so it is
    reconstructed from the parent's code constants to be callable at all.
    """
    from PIL import Image
    import spacr.timelapse as TL

    code = next(c for c in TL._masks_to_gif.__code__.co_consts
                if isinstance(c, types.CodeType) and c.co_name == "_display_gif")
    display_gif = types.FunctionType(code, TL.__dict__)

    gif_path = tmp_path / "tiny.gif"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(gif_path)
    raw = gif_path.read_bytes()

    shown = []
    monkeypatch.setattr(TL, "display", lambda *objs, **kw: shown.extend(objs))

    display_gif(str(gif_path))

    assert len(shown) == 1
    assert shown[0].data == raw
    return len(shown)


# ---------------------------------------------------------------------------
# _timelapse_masks_to_gif
# ---------------------------------------------------------------------------

def _write_npy_series(folder, plate, well, field, times, shape=(6, 6), channels=2):
    for t in times:
        arr = np.zeros((shape[0], shape[1], channels), dtype=np.uint16)
        arr[0, 0, 0] = 10 + t                       # channel-0 marker
        if channels > 1:
            arr[0, 1, 1] = 100 + t                  # channel-1 marker
        np.save(os.path.join(folder, f"{plate}_{well}_{field}_{t}.npy"), arr)


def _scn_timelapse_masks_to_gif_one_gif_per_channel(tmp_path, monkeypatch):
    gif_spy = _gif_spy(monkeypatch)
    """Each mask channel of a plate/well/field series becomes its own GIF."""
    from spacr.timelapse import _timelapse_masks_to_gif

    src = tmp_path / "masks"
    src.mkdir()
    _write_npy_series(str(src), "1", "A01", "1", times=[2, 0, 1])

    _timelapse_masks_to_gif(str(src), [0, 1], ["cell", "nucleus"])

    gif_folder = tmp_path / "movies" / "gif"
    assert gif_folder.is_dir()
    assert len(gif_spy) == 2

    cell, nucleus = gif_spy
    assert cell["path"] == str(gif_folder / "timelapse_masks_cell_1_A01_1.gif")
    assert nucleus["path"] == str(gif_folder / "timelapse_masks_nucleus_1_A01_1.gif")

    # frames are ordered by the trailing time index, not by glob order
    assert cell["filenames"] == ["1_A01_1_0.npy", "1_A01_1_1.npy", "1_A01_1_2.npy"]
    assert cell["masks"].shape == (3, 6, 6)
    assert cell["masks"][:, 0, 0].tolist() == [10, 11, 12]
    # the second call really pulled the OTHER channel
    assert nucleus["masks"][:, 0, 1].tolist() == [100, 101, 102]
    assert nucleus["masks"][:, 0, 0].max() == 0

    assert sorted(os.listdir(gif_folder)) == [
        "timelapse_masks_cell_1_A01_1.gif",
        "timelapse_masks_nucleus_1_A01_1.gif",
    ]
    return (cell["filenames"], cell["masks"][:, 0, 0].tolist(), nucleus["masks"][:, 0, 1].tolist())


def _scn_timelapse_masks_to_gif_groups_by_plate_well_field(tmp_path, monkeypatch):
    gif_spy = _gif_spy(monkeypatch)
    """Different wells / fields never end up in the same GIF."""
    from spacr.timelapse import _timelapse_masks_to_gif

    src = tmp_path / "masks"
    src.mkdir()
    _write_npy_series(str(src), "1", "A01", "1", times=[0, 1], channels=1)
    _write_npy_series(str(src), "1", "A02", "1", times=[0], channels=1)
    _write_npy_series(str(src), "2", "A02", "3", times=[0, 1, 2], channels=1)

    _timelapse_masks_to_gif(str(src), [0], ["cell"])

    by_name = {os.path.basename(c["path"]): c for c in gif_spy}
    assert set(by_name) == {
        "timelapse_masks_cell_1_A01_1.gif",
        "timelapse_masks_cell_1_A02_1.gif",
        "timelapse_masks_cell_2_A02_3.gif",
    }
    assert by_name["timelapse_masks_cell_1_A01_1.gif"]["masks"].shape[0] == 2
    assert by_name["timelapse_masks_cell_1_A02_1.gif"]["masks"].shape[0] == 1
    assert by_name["timelapse_masks_cell_2_A02_3.gif"]["masks"].shape[0] == 3
    return {k: v["masks"].shape[0] for k, v in sorted(by_name.items())}


def _scn_timelapse_masks_to_gif_skips_unparsable_filenames(tmp_path, monkeypatch):
    gif_spy = _gif_spy(monkeypatch)
    """.npy files that do not carry plate_well_field_time are ignored."""
    from spacr.timelapse import _timelapse_masks_to_gif

    src = tmp_path / "masks"
    src.mkdir()
    _write_npy_series(str(src), "1", "A01", "1", times=[0], channels=1)
    np.save(src / "notes.npy", np.zeros((6, 6, 1), dtype=np.uint16))
    np.save(src / "1_a01_1_0.npy", np.zeros((6, 6, 1), dtype=np.uint16))  # lowercase well
    (src / "ignored.txt").write_text("not an array", encoding="utf-8")

    _timelapse_masks_to_gif(str(src), [0], ["cell"])

    assert len(gif_spy) == 1
    assert gif_spy[0]["filenames"] == ["1_A01_1_0.npy"]
    return gif_spy[0]["filenames"]


def _scn_timelapse_masks_to_gif_empty_folder_still_creates_gif_dir(tmp_path, monkeypatch):
    gif_spy = _gif_spy(monkeypatch)
    """No matching arrays -> the movies/gif folder exists but nothing is rendered."""
    from spacr.timelapse import _timelapse_masks_to_gif

    src = tmp_path / "masks"
    src.mkdir()

    _timelapse_masks_to_gif(str(src), [0], ["cell"])

    assert (tmp_path / "movies" / "gif").is_dir()
    assert os.listdir(tmp_path / "movies" / "gif") == []
    assert gif_spy == []
    return (os.listdir(tmp_path / "movies" / "gif"), len(gif_spy))


def _scn_timelapse_masks_to_gif_reuses_existing_gif_folder(tmp_path, monkeypatch):
    gif_spy = _gif_spy(monkeypatch)
    """makedirs(exist_ok=True): a pre-existing folder and its contents survive."""
    from spacr.timelapse import _timelapse_masks_to_gif

    src = tmp_path / "masks"
    src.mkdir()
    gif_folder = tmp_path / "movies" / "gif"
    gif_folder.mkdir(parents=True)
    (gif_folder / "old.gif").write_bytes(b"GIF89a")
    _write_npy_series(str(src), "1", "A01", "1", times=[0], channels=1)

    _timelapse_masks_to_gif(str(src), [0], ["cell"])

    assert (gif_folder / "old.gif").read_bytes() == b"GIF89a"
    assert len(gif_spy) == 1
    assert os.path.exists(gif_spy[0]["path"])
    return ((gif_folder / "old.gif").read_bytes(), len(gif_spy))


def _scn_movie_labels_do_not_modify_input_pixels(tmp_path, monkeypatch):
    from spacr.timelapse import _npz_to_movie
    fake_writer = _fake_writer(monkeypatch)

    frame = np.full((64, 200, 3), 17, dtype=np.uint8)
    frame.flags.writeable = False
    _npz_to_movie([frame], ['field.npy'], str(tmp_path / 'm.avi'))
    assert np.all(frame == 17)
    assert fake_writer.instances[0].frames[0].max() > 200
    return (int(frame.max()), int(fake_writer.instances[0].frames[0].max()) > 200)


def _scn_float64_movie_has_the_same_scale_as_float32(tmp_path, monkeypatch):
    from spacr.timelapse import _npz_to_movie
    fake_writer = _fake_writer(monkeypatch)

    _npz_to_movie([np.full((16, 16), .5)], ['a'], str(tmp_path / 'm.avi'))
    written = fake_writer.instances[0].frames[0]
    assert written.dtype == np.uint8
    assert written[0, 0].tolist() == [127, 127, 127]
    return written[0, 0].tolist()


_ENCODE_SCENARIOS = [
    pytest.param(_scn_npz_to_movie_float32_is_clipped_and_scaled, [127, 255, 0], id="float32-clipped-and-scaled"),
    pytest.param(_scn_npz_to_movie_uint16_is_scaled_to_8bit, pytest.approx([255, 128], abs=1), id="uint16-scaled-to-8bit"),
    pytest.param(_scn_npz_to_movie_single_channel_axis_is_expanded, [200, 200, 200], id="single-channel-axis-expanded"),
    pytest.param(_scn_npz_to_movie_two_channel_becomes_red_green, [0, 222, 111], id="two-channel-red-green"),
    pytest.param(_scn_npz_to_movie_uint16_two_channel_is_scaled_then_packed, pytest.approx([0, 128, 255], abs=1), id="uint16-two-channel-scaled-then-packed"),
    pytest.param(_scn_npz_to_movie_three_channel_is_rgb_to_bgr_swapped, [30, 20, 10], id="three-channel-rgb-to-bgr"),
    pytest.param(_scn_npz_to_movie_stamps_filename_on_every_frame, [0, 0], id="filename-stamped-on-every-frame"),
    pytest.param(_scn_npz_to_movie_writes_real_containers, (3, 64), id="real-mp4-and-avi-containers"),
    pytest.param(_scn_movie_labels_do_not_modify_input_pixels, (17, True), id="input-pixels-untouched"),
    pytest.param(_scn_float64_movie_has_the_same_scale_as_float32, [127, 127, 127], id="float64-scales-like-float32"),
]


@pytest.mark.parametrize("scenario, expected", _ENCODE_SCENARIOS)
def test_npz_to_movie_encodes_each_frame_kind_as_8bit_bgr_with_the_name_stamped(
        tmp_path, monkeypatch, scenario, expected):
    observed = scenario(tmp_path, monkeypatch)
    assert observed == expected


def _refuses(frames, names, match=None, **kw):
    def scenario(tmp_path, monkeypatch):
        from spacr.timelapse import _npz_to_movie
        fake_writer = _fake_writer(monkeypatch)
        with pytest.raises(ValueError, match=match):
            _npz_to_movie(frames, names, str(tmp_path / 'm.avi'), **kw)
        assert fake_writer.instances == []   # refused before the writer opened
        return len(fake_writer.instances)
    return scenario


_ONE = [np.zeros((16, 16), np.uint8)]
def _scn_movie_writer_failure_is_reported_and_released(tmp_path, monkeypatch):
    from spacr.timelapse import _npz_to_movie
    fake_writer = _fake_writer(monkeypatch)

    monkeypatch.setattr(fake_writer, 'isOpened', lambda self: False, raising=False)
    with pytest.raises(OSError, match='writer'):
        _npz_to_movie([np.zeros((16, 16), np.uint8)], ['a'], str(tmp_path / 'm.avi'))
    assert fake_writer.instances[0].released
    assert not fake_writer.instances[0].frames
    return (fake_writer.instances[0].released, len(fake_writer.instances[0].frames))


def _scn_movie_writer_is_released_when_encoding_raises(tmp_path, monkeypatch):
    from spacr.timelapse import _npz_to_movie
    fake_writer = _fake_writer(monkeypatch)

    def fail(self, frame):
        raise RuntimeError('codec failed')
    monkeypatch.setattr(fake_writer, 'write', fail)
    with pytest.raises(RuntimeError, match='codec failed'):
        _npz_to_movie([np.zeros((16, 16), np.uint8)], ['a'], str(tmp_path / 'm.avi'))
    assert fake_writer.instances[0].released
    return fake_writer.instances[0].released


_REFUSAL_SCENARIOS = [
    pytest.param(_refuses([], []), 0, id="no-frames"),
    pytest.param(_refuses([np.zeros((10, 10), np.uint8)], []), 0, id="name-count-mismatch"),
    pytest.param(_refuses([np.zeros((10, 10), np.uint8), np.zeros((12, 10), np.uint8)], ['a', 'b']), 0, id="mixed-sizes"),
    pytest.param(_refuses([np.zeros((10, 10), np.uint8), np.zeros((10, 10, 2, 3), np.uint8)], ['a', 'b']), 0, id="mixed-ndim"),
    pytest.param(_refuses([np.zeros((0, 10), np.uint8)], ['a']), 0, id="zero-height"),
    pytest.param(_refuses([np.zeros((10, 10, 0), np.uint8)], ['a']), 0, id="zero-channels"),
    pytest.param(_refuses([np.zeros((10, 10), np.int32)], ['a']), 0, id="unsupported-dtype"),
    pytest.param(_refuses([np.full((10, 10), np.nan)], ['a']), 0, id="nan-pixels"),
    pytest.param(_refuses(_ONE, ['a'], 'fps', fps=0), 0, id="fps-zero"),
    pytest.param(_refuses(_ONE, ['a'], 'fps', fps=-1), 0, id="fps-negative"),
    pytest.param(_refuses(_ONE, ['a'], 'fps', fps=np.nan), 0, id="fps-nan"),
    pytest.param(_refuses(_ONE, ['a'], 'fps', fps=np.inf), 0, id="fps-inf"),
    pytest.param(_scn_movie_writer_failure_is_reported_and_released, (True, 0), id="writer-fails-to-open-is-released"),
    pytest.param(_scn_movie_writer_is_released_when_encoding_raises, True, id="encode-error-releases-writer"),
]


@pytest.mark.parametrize("scenario, expected", _REFUSAL_SCENARIOS)
def test_npz_to_movie_refuses_bad_input_before_opening_and_releases_a_failed_writer(
        tmp_path, monkeypatch, scenario, expected):
    observed = scenario(tmp_path, monkeypatch)
    assert observed == expected


_SCMOVIE_SCENARIOS = [
    pytest.param(_scn_scmovie_writes_one_movie_per_object, ["p1_A01_f01_1.mp4", "p1_A01_f01_2.mp4"], id="one-movie-per-object"),
    pytest.param(_scn_scmovie_orders_frames_by_time_and_pads_to_max_size, ((20, 14), 10, [10, 30, 50]), id="time-ordered-and-padded"),
    pytest.param(_scn_scmovie_deduplicates_folder_paths, 2, id="duplicate-folders-once"),
    pytest.param(_scn_scmovie_ignores_non_png_and_unparsable_names, [], id="ignores-non-png-and-unparsable"),
    pytest.param(_scn_scmovie_separates_wells_and_fields, ["p1_A01_f01_1.mp4", "p1_A02_f01_1.mp4", "p1_A02_f02_1.mp4"], id="wells-and-fields-separate"),
]


@pytest.mark.parametrize("scenario, expected", _SCMOVIE_SCENARIOS)
def test_scmovie_writes_one_time_ordered_padded_movie_per_object_group(tmp_path, monkeypatch, scenario, expected):
    observed = scenario(tmp_path, monkeypatch)
    assert observed == expected


_GIF_SCENARIOS = [
    pytest.param(_scn_masks_to_gif_builds_label_colormap_and_delegates, (5, (0, 4), "timelapse_masks_cell_1_A01_1.gif"), id="label-colormap-and-hand-off"),
    pytest.param(_scn_masks_to_gif_single_label_sequence, (1, (0, 0)), id="background-only-sequence"),
    pytest.param(_scn_masks_to_gif_writes_a_real_gif, (1, True), id="real-gif-at-minimum-size"),
    pytest.param(_scn_masks_to_gif_display_helper_reads_and_displays_bytes, 1, id="display-helper-shows-bytes"),
    pytest.param(_scn_timelapse_masks_to_gif_one_gif_per_channel, (["1_A01_1_0.npy", "1_A01_1_1.npy", "1_A01_1_2.npy"], [10, 11, 12], [100, 101, 102]), id="one-gif-per-channel"),
    pytest.param(_scn_timelapse_masks_to_gif_groups_by_plate_well_field, {"timelapse_masks_cell_1_A01_1.gif": 2, "timelapse_masks_cell_1_A02_1.gif": 1, "timelapse_masks_cell_2_A02_3.gif": 3}, id="grouped-by-plate-well-field"),
    pytest.param(_scn_timelapse_masks_to_gif_skips_unparsable_filenames, ["1_A01_1_0.npy"], id="skips-unparsable-names"),
    pytest.param(_scn_timelapse_masks_to_gif_empty_folder_still_creates_gif_dir, ([], 0), id="empty-folder-makes-gif-dir"),
    pytest.param(_scn_timelapse_masks_to_gif_reuses_existing_gif_folder, (b"GIF89a", 1), id="existing-gif-folder-reused"),
]


@pytest.mark.parametrize("scenario, expected", _GIF_SCENARIOS)
def test_mask_gifs_use_one_colour_per_label_and_one_file_per_channel_and_series(
        tmp_path, monkeypatch, scenario, expected):
    observed = scenario(tmp_path, monkeypatch)
    assert observed == expected
