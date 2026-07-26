"""CPU-only coverage for the stack-normalization block of :mod:`spacr.io`.

Covers ``_normalize_stack``, ``_normalize_timelapse``,
``_create_movies_from_npy_per_channel`` and ``delete_empty_subdirectories``.

The interesting branches here are the per-frame ones inside
``_normalize_stack``: a frame that is entirely zero (no non-zero pixels to
take percentiles from) and a frame whose signal-to-noise ratio is below the
configured threshold. Both must be copied through *unrescaled*, which is what
the assertions below pin down.

Everything runs on tiny float32/uint16 arrays so the file stays well under a
second and never touches a GPU, a network or a display.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _no_stray_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _write_stack_npz(path, data, names=None):
    """Write a ``(N, H, W, C)`` array in the npz layout ``_normalize_stack`` reads."""
    if names is None:
        names = [f"img_{i}.npy" for i in range(data.shape[0])]
    np.savez(str(path), data=data, filenames=np.array(names))
    return path


def _ramp(n=64, lo=1.0, hi=1000.0):
    side = int(round(n ** 0.5))
    return np.linspace(lo, hi, n, dtype=np.float32).reshape(side, side)


# ---------------------------------------------------------------------------
# _normalize_stack
# ---------------------------------------------------------------------------

def test_normalize_stack_zero_frame_and_flat_frame_are_copied_raw(tmp_path):
    """All-zero and low-SNR frames bypass rescaling and are copied verbatim.

    Frame 0 has a wide dynamic range (SNR ~47 > 5) so it is percentile
    rescaled into [0, 1]. Frame 1 is entirely zero, so there are no non-zero
    pixels to take percentiles from and the ratio is forced to 0. Frame 2 is
    flat, so upper == lower and the ratio is exactly 1. Neither of the last
    two beats the SNR threshold, so both must survive untouched.
    """
    from spacr.io import _normalize_stack

    src = tmp_path / "stack"
    src.mkdir()
    frames = np.zeros((3, 8, 8, 1), dtype=np.float32)
    frames[0, :, :, 0] = _ramp()
    # frames[1] deliberately left all-zero
    frames[2, :, :, 0] = 500.0
    _write_stack_npz(src / "plate1.npz", frames, names=["a.npy", "b.npy", "c.npy"])

    _normalize_stack(
        str(src),
        backgrounds=[0],
        remove_backgrounds=[False],
        signal_to_noise=[5],
        signal_thresholds=[1.0],
    )

    out = tmp_path / "masks" / "plate1_norm_stack.npz"
    assert out.exists()
    with np.load(str(out)) as d:
        res = d["data"]
        names = d["filenames"]

    assert res.shape == (3, 8, 8, 1)
    assert res.dtype == np.float32
    assert [str(n) for n in names] == ["a.npy", "b.npy", "c.npy"]

    # frame 0 -> rescale branch: squeezed into [0, 1] with both ends saturated
    assert res[0, ..., 0].min() == pytest.approx(0.0)
    assert res[0, ..., 0].max() == pytest.approx(1.0)

    # frame 1 -> empty-frame branch (ratio forced to 0): still all zeros
    assert np.count_nonzero(res[1]) == 0

    # frame 2 -> ratio 1.0 <= 5: raw values kept, NOT normalized to [0, 1]
    assert np.allclose(res[2, ..., 0], 500.0)


def test_normalize_stack_remove_background_zeroes_dim_pixels(tmp_path):
    """``remove_backgrounds=[True]`` clamps sub-background pixels to zero."""
    from spacr.io import _normalize_stack

    src = tmp_path / "stack"
    src.mkdir()
    base = _ramp()
    frames = np.zeros((2, 8, 8, 1), dtype=np.float32)
    frames[0, :, :, 0] = base
    frames[1, :, :, 0] = base
    raw = base.copy()
    _write_stack_npz(src / "bg.npz", frames)

    _normalize_stack(
        str(src),
        backgrounds=[200],
        remove_backgrounds=[True],
        signal_to_noise=[2],
        signal_thresholds=[1.0],
    )

    with np.load(str(tmp_path / "masks" / "bg_norm_stack.npz")) as d:
        res = d["data"]

    dim = raw < 200
    assert dim.any() and (~dim).any()  # the fixture actually exercises both sides
    assert np.all(res[0, ..., 0][dim] == 0)
    assert res[0, ..., 0][~dim].max() == pytest.approx(1.0)


def test_normalize_stack_handles_multiple_channels_and_files(tmp_path):
    """Per-channel settings are indexed independently and every npz is written."""
    from spacr.io import _normalize_stack

    src = tmp_path / "stack"
    src.mkdir()
    for stem in ("p1", "p2"):
        frames = np.zeros((2, 8, 8, 2), dtype=np.float32)
        frames[:, :, :, 0] = _ramp()
        frames[:, :, :, 1] = 300.0  # flat channel -> ratio 1 -> raw copy
        _write_stack_npz(src / f"{stem}.npz", frames)

    _normalize_stack(
        str(src),
        backgrounds=[0, 0],
        remove_backgrounds=[False, False],
        signal_to_noise=[5, 5],
        signal_thresholds=[1.0, 1.0],
    )

    produced = sorted(p.name for p in (tmp_path / "masks").iterdir())
    assert produced == ["p1_norm_stack.npz", "p2_norm_stack.npz"]
    with np.load(str(tmp_path / "masks" / "p1_norm_stack.npz")) as d:
        res = d["data"]
    assert res.shape == (2, 8, 8, 2)
    assert res[..., 0].max() == pytest.approx(1.0)   # channel 0 rescaled
    assert np.allclose(res[..., 1], 300.0)           # channel 1 untouched


def test_normalize_stack_uses_int_defaults_when_args_are_none(tmp_path):
    """The ``None`` defaults resolve to background 100 / SNR 5 / threshold 1000."""
    from spacr.io import _normalize_stack

    src = tmp_path / "stack"
    src.mkdir()
    frames = np.zeros((2, 8, 8, 3), dtype=np.float32)
    for c in range(3):
        frames[:, :, :, c] = _ramp(lo=50.0, hi=5000.0)
    _write_stack_npz(src / "defaults.npz", frames)

    _normalize_stack(str(src))

    with np.load(str(tmp_path / "masks" / "defaults_norm_stack.npz")) as d:
        res = d["data"]
    # default remove_backgrounds is all-False, so nothing is clamped away and
    # the wide-range channels all clear the default SNR of 5 -> rescaled.
    assert res.shape == (2, 8, 8, 3)
    assert res.max() == pytest.approx(1.0)
    assert res.min() == pytest.approx(0.0)


def test_normalize_stack_honours_save_dtype(tmp_path):
    """``save_dtype`` controls the on-disk dtype of the normalized stack."""
    from spacr.io import _normalize_stack

    src = tmp_path / "stack"
    src.mkdir()
    frames = np.zeros((2, 8, 8, 1), dtype=np.float32)
    frames[:, :, :, 0] = _ramp()
    _write_stack_npz(src / "dt.npz", frames)

    _normalize_stack(
        str(src),
        backgrounds=[0],
        remove_backgrounds=[False],
        signal_to_noise=[5],
        signal_thresholds=[1.0],
        save_dtype=np.float64,
    )

    with np.load(str(tmp_path / "masks" / "dt_norm_stack.npz")) as d:
        res = d["data"]
    assert res.dtype == np.float64
    assert res.max() == pytest.approx(1.0)


def test_normalize_stack_first_frame_all_zero(tmp_path):
    """A leading blank frame must be copied through, not crash the run."""
    from spacr.io import _normalize_stack

    src = tmp_path / "stack"
    src.mkdir()
    frames = np.zeros((2, 8, 8, 1), dtype=np.float32)
    # frames[0] is all-zero; only frames[1] carries signal
    frames[1, :, :, 0] = _ramp()
    _write_stack_npz(src / "blankfirst.npz", frames)

    _normalize_stack(
        str(src),
        backgrounds=[0],
        remove_backgrounds=[False],
        signal_to_noise=[5],
        signal_thresholds=[1.0],
    )

    out = tmp_path / "masks" / "blankfirst_norm_stack.npz"
    assert out.exists()
    with np.load(str(out)) as d:
        res = d["data"]
    assert np.count_nonzero(res[0]) == 0
    assert res[1, ..., 0].max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _normalize_timelapse
# ---------------------------------------------------------------------------

def test_normalize_timelapse_rescales_each_frame_to_full_dtype_range(tmp_path):
    """Every frame is independently stretched to the input dtype's full range."""
    from spacr.io import _normalize_timelapse

    src = tmp_path / "stack"
    src.mkdir()
    data = np.zeros((2, 4, 4, 1), dtype=np.uint16)
    data[0, :, :, 0] = np.linspace(100, 1600, 16).reshape(4, 4).astype(np.uint16)
    data[1, :, :, 0] = np.linspace(50, 800, 16).reshape(4, 4).astype(np.uint16)
    _write_stack_npz(src / "tl.npz", data, names=["t0.npy", "t1.npy"])

    _normalize_timelapse(str(src))

    out = tmp_path / "masks" / "tl_norm_timelapse.npz"
    assert out.exists()
    with np.load(str(out)) as d:
        res = d["data"]
        names = d["filenames"]

    assert res.shape == data.shape
    assert res.dtype == np.float32
    assert [str(n) for n in names] == ["t0.npy", "t1.npy"]
    # each frame is normalized on its OWN percentiles, so both saturate
    for i in range(2):
        assert res[i, ..., 0].min() == pytest.approx(0.0)
        assert res[i, ..., 0].max() == pytest.approx(65535.0)
    # frame 1 was dimmer than frame 0 but ends up equally bright -> per-frame
    assert data[1, ..., 0].max() < data[0, ..., 0].max()


def test_normalize_timelapse_handles_two_channels(tmp_path):
    """Channel loop writes into the right slice of the output stack."""
    from spacr.io import _normalize_timelapse

    src = tmp_path / "stack"
    src.mkdir()
    data = np.zeros((2, 4, 4, 2), dtype=np.uint16)
    data[..., 0] = np.linspace(10, 900, 32).reshape(2, 4, 4).astype(np.uint16)
    data[..., 1] = np.linspace(1000, 5000, 32).reshape(2, 4, 4).astype(np.uint16)
    _write_stack_npz(src / "two.npz", data)

    _normalize_timelapse(str(src), lower_percentile=5, save_dtype=np.float32)

    with np.load(str(tmp_path / "masks" / "two_norm_timelapse.npz")) as d:
        res = d["data"]
    assert res.shape == (2, 4, 4, 2)
    for c in range(2):
        assert res[..., c].max() == pytest.approx(65535.0)


# ---------------------------------------------------------------------------
# _create_movies_from_npy_per_channel
# ---------------------------------------------------------------------------

def test_create_movies_one_movie_per_field_and_channel(tmp_path, rng):
    """Each (plate, well, field) x channel gets its own mp4.

    Regression guard for the dedent bug called out in the source comment:
    when the channel loop escaped the per-field loop, only the LAST field
    ever produced a movie.
    """
    from spacr.io import _create_movies_from_npy_per_channel

    src = tmp_path / "stack"
    src.mkdir()
    for field in ("f1", "f2"):
        for t in range(3):
            arr = rng.integers(0, 500, (16, 16, 2)).astype(np.uint16)
            np.save(str(src / f"plate1_A01_{field}_{t}.npy"), arr)
    # does not match  (\w+)_(\w+)_(\w+)_(\d+)\.npy  -> must be skipped entirely
    np.save(str(src / "junk.npy"), np.zeros((16, 16, 2), dtype=np.uint16))

    _create_movies_from_npy_per_channel(str(src), fps=2)

    movies = sorted(p.name for p in (tmp_path / "movies").iterdir())
    assert movies == [
        "plate1_A01_f1_channel_0.mp4",
        "plate1_A01_f1_channel_1.mp4",
        "plate1_A01_f2_channel_0.mp4",
        "plate1_A01_f2_channel_1.mp4",
    ]


def test_create_movies_normalizes_per_channel_to_uint8(tmp_path, monkeypatch):
    """Frames handed to the writer are uint8 (H, W, 1) rescaled on 1/99 pct."""
    from spacr import io as IO

    src = tmp_path / "stack"
    src.mkdir()
    for t in range(2):
        arr = np.zeros((4, 4, 1), dtype=np.uint16)
        arr[..., 0] = np.linspace(0, 1000 * (t + 1), 16).reshape(4, 4)
        np.save(str(src / f"p_B02_g3_{t}.npy"), arr)

    captured = {}

    def fake_movie(arrays, filenames, save_path, fps):
        captured["arrays"] = arrays
        captured["filenames"] = list(filenames)
        captured["save_path"] = save_path
        captured["fps"] = fps

    import spacr.timelapse as TL

    monkeypatch.setattr(TL, "_npz_to_movie", fake_movie)
    IO._create_movies_from_npy_per_channel(str(src), fps=7)

    assert captured["fps"] == 7
    assert captured["save_path"].endswith("p_B02_g3_channel_0.mp4")
    assert captured["filenames"] == ["p_B02_g3_0.npy", "p_B02_g3_1.npy"]
    frames = captured["arrays"]
    assert len(frames) == 2
    for f in frames:
        assert f.dtype == np.uint8
        assert f.shape == (4, 4, 1)
    # global (not per-frame) percentiles: the brighter t=1 frame saturates,
    # the dimmer t=0 frame does not.
    assert frames[1].max() == 255
    assert frames[0].max() < 255


def test_create_movies_with_no_matching_files_writes_nothing(tmp_path):
    """A src with only unmatched names still creates movies/ but no mp4s."""
    from spacr.io import _create_movies_from_npy_per_channel

    src = tmp_path / "stack"
    src.mkdir()
    np.save(str(src / "nomatch.npy"), np.zeros((4, 4, 1), dtype=np.uint16))
    (src / "ignored.txt").write_text("not a npy")

    _create_movies_from_npy_per_channel(str(src))

    assert (tmp_path / "movies").is_dir()
    assert list((tmp_path / "movies").iterdir()) == []


# ---------------------------------------------------------------------------
# delete_empty_subdirectories
# ---------------------------------------------------------------------------

def test_delete_empty_subdirectories_removes_nested_empties_bottom_up(tmp_path):
    """One call clears a whole empty chain but keeps anything holding a file."""
    from spacr.io import delete_empty_subdirectories

    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "keep" / "inner").mkdir(parents=True)
    (tmp_path / "keep" / "inner" / "x.txt").write_text("x")

    delete_empty_subdirectories(str(tmp_path))

    assert not (tmp_path / "a").exists()
    assert (tmp_path / "keep" / "inner" / "x.txt").exists()
    assert tmp_path.exists()  # the root itself is never removed


def test_delete_empty_subdirectories_swallows_oserror(tmp_path, monkeypatch):
    """An rmdir that fails for a non-emptiness reason is skipped, not raised."""
    from spacr import io as IO

    target = tmp_path / "locked"
    target.mkdir()

    real_rmdir = os.rmdir

    def flaky_rmdir(path, *args, **kwargs):
        if os.path.basename(str(path)) == "locked":
            raise PermissionError(13, "Permission denied")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(IO.os, "rmdir", flaky_rmdir)

    IO.delete_empty_subdirectories(str(tmp_path))  # must not raise

    assert target.is_dir()
