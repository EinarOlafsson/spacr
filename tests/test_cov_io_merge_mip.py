"""Coverage for spacr.io channel merge / MIP / concatenate helpers.

Targets ``_merge_channels`` (plot branch), ``_mip_all`` and
``_concatenate_channel`` (timelapse branch, randomize branch and the
ragged-shape padding branch) in spacr/io.py.

Everything here is CPU-only and offline: the fixtures are tiny synthetic
TIFF/npy arrays written into ``tmp_path``.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _no_figures():
    """Never let a stray figure survive a test."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_tif(path: Path, arr: np.ndarray) -> None:
    import tifffile
    tifffile.imwrite(str(path), arr)


def _chan_dirs(root: Path, n_chan: int = 3, names=("f0", "f1")) -> Path:
    """Build ``root/<chan>/<field>.tif`` — the layout _merge_channels wants.

    Each channel image is a constant plane whose value encodes the channel,
    so the merged stack can be verified channel by channel.
    """
    root.mkdir(parents=True, exist_ok=True)
    for c in range(1, n_chan + 1):
        d = root / str(c)
        d.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(names):
            arr = np.full((6, 7), 100 * c + i, dtype=np.uint16)
            _write_tif(d / f"{name}.tif", arr)
    return root


def _write_npy_stack(path: Path, shape, fill) -> np.ndarray:
    arr = np.full(shape, fill, dtype=np.uint16)
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------------------
# _merge_channels
# ---------------------------------------------------------------------------

def test_merge_channels_plot_true_forwards_stack_dir(tmp_path, monkeypatch):
    """plot=True must hand the *stack* directory to plot.plot_arrays (io.py:1027)."""
    import spacr.plot as SP
    from spacr.io import _merge_channels

    src = _chan_dirs(tmp_path / "src", n_chan=3)
    calls = []
    monkeypatch.setattr(SP, "plot_arrays", lambda p, *a, **k: calls.append(p))

    n = _merge_channels(str(src), plot=True)

    assert n == 3, "return value must be the number of single-channel folders"
    assert calls == [str(src / "stack")]
    stacks = sorted((src / "stack").glob("*.npy"))
    assert [p.name for p in stacks] == ["f0.npy", "f1.npy"]
    # channel order follows the sorted channel-folder names 1,2,3
    arr = np.load(stacks[0])
    assert arr.shape == (6, 7, 3)
    assert [int(arr[0, 0, c]) for c in range(3)] == [100, 200, 300]


def test_merge_channels_skips_non_file_entries(tmp_path):
    """A sub-directory inside the first channel folder must not be merged."""
    from spacr.io import _merge_channels

    src = _chan_dirs(tmp_path / "src", n_chan=2)
    (src / "1" / "not_an_image").mkdir()

    n = _merge_channels(str(src), plot=False)

    assert n == 2
    assert sorted(p.name for p in (src / "stack").glob("*")) == ["f0.npy", "f1.npy"]


def test_merge_channels_non_empty_stack_dir_is_left_alone(tmp_path):
    """When stack/ already has content the merge loop is skipped entirely."""
    from spacr.io import _merge_channels

    src = _chan_dirs(tmp_path / "src", n_chan=2)
    stack = src / "stack"
    stack.mkdir()
    sentinel = stack / "already_here.npy"
    np.save(sentinel, np.zeros((2, 2, 1), dtype=np.uint16))

    n = _merge_channels(str(src), plot=False)

    assert n == 2
    assert [p.name for p in stack.glob("*.npy")] == ["already_here.npy"]


# ---------------------------------------------------------------------------
# _mip_all
# ---------------------------------------------------------------------------

def test_mip_all_appends_max_projection(tmp_path):
    from spacr.io import _mip_all

    arr = np.zeros((4, 5, 3), dtype=np.uint16)
    arr[..., 0] = 7
    arr[..., 1] = 11
    arr[..., 2] = 3
    np.save(tmp_path / "a.npy", arr)

    _mip_all(str(tmp_path), include_first_chan=True)

    out = np.load(tmp_path / "a.npy")
    assert out.shape == (4, 5, 4)
    assert np.array_equal(out[..., :3], arr)
    assert np.all(out[..., 3] == 11)


def test_mip_all_exclude_first_channel(tmp_path):
    from spacr.io import _mip_all

    arr = np.zeros((4, 5, 3), dtype=np.uint16)
    arr[..., 0] = 99   # brightest, but must be excluded
    arr[..., 1] = 11
    arr[..., 2] = 3
    np.save(tmp_path / "a.npy", arr)

    _mip_all(str(tmp_path), include_first_chan=False)

    out = np.load(tmp_path / "a.npy")
    assert out.shape == (4, 5, 4)
    assert np.all(out[..., 3] == 11)


def test_mip_all_promotes_2d_array_and_pads_with_zeros(tmp_path):
    """A 2-D array gets an axis then a zero plane concatenated onto it."""
    from spacr.io import _mip_all

    arr = np.full((3, 4), 5, dtype=np.uint16)
    np.save(tmp_path / "flat.npy", arr)
    # a non-npy file in the same folder must be ignored
    (tmp_path / "note.txt").write_text("ignore me")

    _mip_all(str(tmp_path))

    out = np.load(tmp_path / "flat.npy")
    assert out.shape == (3, 4, 2)
    assert np.all(out[..., 0] == 5)
    assert np.all(out[..., 1] == 0)
    assert (tmp_path / "note.txt").read_text() == "ignore me"


# ---------------------------------------------------------------------------
# _concatenate_channel — randomize branch
# ---------------------------------------------------------------------------

def test_concatenate_channel_randomize_shuffles_and_keeps_pairing(tmp_path, monkeypatch):
    """randomize=True must shuffle paths (io.py:1140) *and* keep data/filename
    rows aligned after the shuffle."""
    from spacr.io import _concatenate_channel

    src = tmp_path / "proj" / "stack"
    src.mkdir(parents=True)
    fills = {}
    for i in range(6):
        name = f"plate1_A01_{i}.npy"
        fills[name] = 10 + i
        _write_npy_stack(src / name, (4, 5, 3), 10 + i)

    seen = {}

    def fake_shuffle(seq):
        # deterministic stand-in for random.shuffle: reverse-alphabetical
        seen["called"] = True
        seq.sort(key=os.path.basename, reverse=True)

    monkeypatch.setattr(random, "shuffle", fake_shuffle)

    out = _concatenate_channel(str(src), channels=[0, 1, None],
                               randomize=True, timelapse=False, batch_size=100)

    assert seen.get("called") is True
    assert out == str(tmp_path / "proj" / "channel_stack")
    npzs = sorted(Path(out).glob("*.npz"))
    assert [p.name for p in npzs] == ["stack_0.npz"]

    with np.load(npzs[0]) as z:
        data, filenames = z["data"], list(z["filenames"])
    # None channels are dropped -> 2 channels kept
    assert data.shape == (6, 4, 5, 2)
    assert filenames == sorted(fills, reverse=True)
    for row, name in enumerate(filenames):
        assert int(data[row, 0, 0, 0]) == fills[name]


def test_concatenate_channel_batches_split_output_files(tmp_path):
    """batch_size smaller than the file count writes one npz per batch."""
    from spacr.io import _concatenate_channel

    src = tmp_path / "proj" / "stack"
    src.mkdir(parents=True)
    for i in range(5):
        _write_npy_stack(src / f"plate1_A01_{i}.npy", (3, 3, 2), i)
    (src / "ignored.txt").write_text("not an npy")

    out = _concatenate_channel(str(src), channels=[0, 1], randomize=False,
                               timelapse=False, batch_size=2)

    npzs = sorted(Path(out).glob("*.npz"), key=lambda p: p.name)
    assert [p.name for p in npzs] == ["stack_0.npz", "stack_1.npz", "stack_2.npz"]
    counts = []
    all_names = []
    for p in npzs:
        with np.load(p) as z:
            counts.append(z["data"].shape[0])
            all_names.extend(list(z["filenames"]))
    assert counts == [2, 2, 1]
    assert sorted(all_names) == [f"plate1_A01_{i}.npy" for i in range(5)]


# ---------------------------------------------------------------------------
# _concatenate_channel — ragged shapes get zero-padded (io.py:1160-1168)
# ---------------------------------------------------------------------------

def test_concatenate_channel_pads_arrays_of_mixed_shape(tmp_path):
    from spacr.io import _concatenate_channel

    src = tmp_path / "proj" / "stack"
    src.mkdir(parents=True)
    _write_npy_stack(src / "small.npy", (10, 12, 3), 4)
    _write_npy_stack(src / "tall.npy", (14, 8, 3), 9)

    out = _concatenate_channel(str(src), channels=[0, 2], randomize=False,
                               timelapse=False, batch_size=100)

    npzs = list(Path(out).glob("*.npz"))
    assert [p.name for p in npzs] == ["stack_0.npz"]
    with np.load(npzs[0]) as z:
        data, filenames = z["data"], [str(f) for f in z["filenames"]]

    # padded to the per-axis maximum of the two shapes: (14, 12)
    assert data.shape == (2, 14, 12, 2)
    idx = {name: i for i, name in enumerate(filenames)}
    small = data[idx["small.npy"]]
    tall = data[idx["tall.npy"]]
    assert np.all(small[:10, :12] == 4)
    assert np.all(small[10:, :] == 0)      # padded rows
    assert np.all(tall[:14, :8] == 9)
    assert np.all(tall[:, 8:] == 0)        # padded columns


# ---------------------------------------------------------------------------
# _concatenate_channel — timelapse branch
# ---------------------------------------------------------------------------

def test_concatenate_channel_timelapse_writes_one_npz_per_field(tmp_path):
    """timelapse=True should emit <plate>_<well>_<field>.npz per field, with the
    frames stacked in timepoint order."""
    from spacr.io import _concatenate_channel

    src = tmp_path / "proj" / "stack"
    src.mkdir(parents=True)
    # plate_well_field_time.npy — two timepoints of one field.
    _write_npy_stack(src / "plate1_A01_1_2.npy", (8, 9, 3), 20)
    _write_npy_stack(src / "plate1_A01_1_1.npy", (8, 9, 3), 10)

    out = _concatenate_channel(str(src), channels=[0, 1], randomize=False,
                               timelapse=True, batch_size=10)

    assert out == str(tmp_path / "proj" / "channel_stack")
    npzs = sorted(Path(out).glob("*.npz"))
    assert [p.name for p in npzs] == ["plate1_A01_1.npz"]
    with np.load(npzs[0]) as z:
        data, filenames = z["data"], [str(f) for f in z["filenames"]]
    assert data.shape == (2, 8, 9, 2)
    assert filenames == ["plate1_A01_1_1.npy", "plate1_A01_1_2.npy"]
    assert int(data[0, 0, 0, 0]) == 10
    assert int(data[1, 0, 0, 0]) == 20


def test_concatenate_channel_timelapse_returns_dir_without_raising(tmp_path):
    """The timelapse branch must never propagate an exception to the caller: it
    catches, reports, and still returns the channel_stack location."""
    from spacr.io import _concatenate_channel

    src = tmp_path / "proj" / "stack"
    src.mkdir(parents=True)
    _write_npy_stack(src / "plate1_A01_1_1.npy", (6, 6, 2), 3)

    out = _concatenate_channel(str(src), channels=[0], randomize=False,
                               timelapse=True, batch_size=5)

    assert out == str(tmp_path / "proj" / "channel_stack")
    assert os.path.isdir(out)
