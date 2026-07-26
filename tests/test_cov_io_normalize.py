"""CPU-only coverage for the normalisation block of ``spacr.io``.

Covers ``spacr/io.py`` lines 1181-1486:

* ``_normalize_img_batch``      - per-channel background / signal-to-noise
                                  selection (nucleus, cell, pathogen,
                                  organelle, generic) and the global upper
                                  percentile search + fallback.
* ``concatenate_and_normalize`` - both the timelapse and the batched
                                  non-timelapse concatenation paths,
                                  including shape padding, unreadable
                                  ``.npy`` files and the gated plotting hook.
* ``_get_lists_for_normalization`` - per-object-type list assembly.

Everything here is synthetic, offline and runs in well under a second.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let an Agg figure leak out of a test."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _stack(n=3, h=8, w=8, c=2, seed=0):
    """(N, H, W, C) float32 stack with strictly positive intensities."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, h, w, c)) * 2950.0 + 50.0).astype(np.float32)


def _plane(h=8, w=8, c=2, seed=0):
    """A single (H, W, C) field as written by the stack step."""
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, c)) * 2950.0 + 50.0).astype(np.float32)


def _base_settings(**over):
    """Settings dict with every key ``concatenate_and_normalize`` touches."""
    s = {
        "timelapse": False,
        "randomize": False,
        "batch_size": 10,
        "lower_percentile": 2,
        "nucleus_channel": 0,
        "cell_channel": 1,
        "pathogen_channel": None,
        "organelle_channel": None,
        "nucleus_background": 100,
        "nucleus_Signal_to_noise": 5,
        "remove_background_nucleus": False,
        "cell_background": 100,
        "cell_Signal_to_noise": 5,
        "remove_background_cell": False,
        "background": 100,
        "Signal_to_noise": 5,
        "remove_background": False,
        "plot": False,
        "figuresize": 4,
        "cmap": "inferno",
        "nr": 1,
    }
    s.update(over)
    return s


def _sorted_listdir(real_listdir):
    """Wrap os.listdir so directory iteration order is deterministic."""
    def _listdir(*args, **kwargs):
        return sorted(real_listdir(*args, **kwargs))
    return _listdir


def _write_fields(dirpath, names, shape=(8, 8, 2), seed=0):
    """Write one ``.npy`` per name; returns the list of absolute paths."""
    os.makedirs(dirpath, exist_ok=True)
    paths = []
    for i, name in enumerate(names):
        arr = _plane(*shape, seed=seed + i)
        p = os.path.join(dirpath, name)
        np.save(p, arr)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# _normalize_img_batch - per-object-type parameter selection
# ---------------------------------------------------------------------------

def test_normalize_img_batch_pathogen_channel_uses_pathogen_settings(capsys):
    """The pathogen branch must pick up pathogen_* background / SNR / removal."""
    from spacr.io import _normalize_img_batch

    stack = _stack(n=2, c=2, seed=3)
    settings = {
        "nucleus_channel": None,
        "cell_channel": None,
        "pathogen_channel": 1,
        "organelle_channel": None,
        "pathogen_background": 250,
        "pathogen_Signal_to_noise": 2,
        "remove_background_pathogen": True,
        "lower_percentile": 2,
    }
    out = _normalize_img_batch(stack.copy(), [1], np.float32, settings)

    printed = capsys.readouterr().out
    assert ("Processing channel 1: background=250, signal_threshold=500, "
            "remove_background=True") in printed
    assert out.shape == stack.shape
    assert out.dtype == np.float32
    # Channel 0 was never normalised -> stays all-zero in the output buffer.
    assert np.all(out[:, :, :, 0] == 0)
    assert out[:, :, :, 1].max() == pytest.approx(1.0, abs=1e-5)


def test_normalize_img_batch_organelle_channel_uses_organelle_settings(capsys):
    """organelle_* keys, when present, override the generic defaults."""
    from spacr.io import _normalize_img_batch

    stack = _stack(n=2, c=3, seed=4)
    settings = {
        "nucleus_channel": 0,
        "cell_channel": 1,
        "pathogen_channel": None,
        "organelle_channel": 2,
        "background": 100,
        "Signal_to_noise": 5,
        "remove_background": False,
        "organelle_background": 200,
        "organelle_Signal_to_noise": 3,
        "remove_background_organelle": True,
        "lower_percentile": 2,
    }
    out = _normalize_img_batch(stack.copy(), [2], np.float32, settings)

    printed = capsys.readouterr().out
    assert ("Processing channel 2: background=200, signal_threshold=600, "
            "remove_background=True") in printed
    assert out.shape == stack.shape
    assert np.all(out[:, :, :, :2] == 0)


def test_normalize_img_batch_organelle_channel_falls_back_to_generic(capsys):
    """Without organelle_* keys the organelle branch reuses the generic values."""
    from spacr.io import _normalize_img_batch

    stack = _stack(n=2, c=2, seed=5)
    settings = {
        "nucleus_channel": None,
        "cell_channel": None,
        "pathogen_channel": None,
        "organelle_channel": 0,
        "background": 300,
        "Signal_to_noise": 4,
        "remove_background": False,
        "lower_percentile": 2,
    }
    out = _normalize_img_batch(stack.copy(), [0], np.float32, settings)

    printed = capsys.readouterr().out
    assert ("Processing channel 0: background=300, signal_threshold=1200, "
            "remove_background=False") in printed
    assert out.dtype == np.float32
    assert out[:, :, :, 0].min() == pytest.approx(0.0, abs=1e-6)


def test_normalize_img_batch_remove_background_zeroes_more_pixels():
    """remove_background=True must push sub-background pixels to 0 in the output."""
    from spacr.io import _normalize_img_batch

    stack = _stack(n=2, c=1, seed=6)
    common = {
        "nucleus_channel": None,
        "cell_channel": None,
        "pathogen_channel": None,
        "organelle_channel": None,
        "background": 1500,
        "Signal_to_noise": 1,
        "lower_percentile": 2,
    }
    kept = _normalize_img_batch(stack.copy(), [0], np.float32,
                                dict(common, remove_background=False))
    removed = _normalize_img_batch(stack.copy(), [0], np.float32,
                                   dict(common, remove_background=True))
    n_zero_kept = int(np.count_nonzero(kept[:, :, :, 0] == 0))
    n_zero_removed = int(np.count_nonzero(removed[:, :, :, 0] == 0))
    # Everything below 1500 is forced to 0 before the percentile is taken.
    assert n_zero_removed > n_zero_kept
    assert n_zero_removed >= int(np.count_nonzero(stack[:, :, :, 0] < 1500))


def test_normalize_img_batch_upper_percentile_break_uses_first_hit():
    """The first percentile at/above signal_threshold wins (loop `break`)."""
    from skimage import exposure
    from spacr.io import _normalize_img_batch

    stack = _stack(n=2, h=10, w=10, c=1, seed=8)
    settings = {
        "nucleus_channel": None,
        "cell_channel": None,
        "pathogen_channel": None,
        "organelle_channel": None,
        "background": 1,
        "Signal_to_noise": 1,       # threshold=1 -> the 98th percentile hits
        "remove_background": False,
        "lower_percentile": 2,
    }
    out = _normalize_img_batch(stack.copy(), [0], np.float32, settings)

    chan = stack[:, :, :, 0]
    nz = chan[chan != 0]
    lo = np.percentile(nz, 2)
    hi = np.percentile(nz, 98.0)
    expected = np.stack([
        exposure.rescale_intensity(chan[i], in_range=(lo, hi), out_range=(0, 1))
        for i in range(chan.shape[0])
    ]).astype(np.float32)
    assert np.allclose(out[:, :, :, 0], expected, atol=1e-6)
    # Sanity: the 98th and the 99.5th percentile really do differ here, so the
    # assertion above genuinely discriminates the break from the fallback.
    assert not np.isclose(hi, np.percentile(nz, 99.5))


def test_normalize_img_batch_upper_percentile_fallback_to_99_5():
    """When no percentile reaches signal_threshold, the 99.5th is used."""
    from skimage import exposure
    from spacr.io import _normalize_img_batch

    stack = _stack(n=2, h=10, w=10, c=1, seed=8)
    settings = {
        "nucleus_channel": None,
        "cell_channel": None,
        "pathogen_channel": None,
        "organelle_channel": None,
        "background": 100000,
        "Signal_to_noise": 10,      # threshold=1e6, far above any pixel
        "remove_background": False,
        "lower_percentile": 2,
    }
    out = _normalize_img_batch(stack.copy(), [0], np.float32, settings)

    chan = stack[:, :, :, 0]
    nz = chan[chan != 0]
    lo = np.percentile(nz, 2)
    hi = np.percentile(nz, 99.5)
    expected = np.stack([
        exposure.rescale_intensity(chan[i], in_range=(lo, hi), out_range=(0, 1))
        for i in range(chan.shape[0])
    ]).astype(np.float32)
    assert np.allclose(out[:, :, :, 0], expected, atol=1e-6)


def test_normalize_img_batch_accepts_string_channels_and_save_dtype():
    """Channel indices arriving as strings are coerced; save_dtype is honoured."""
    from spacr.io import _normalize_img_batch

    stack = _stack(n=2, c=2, seed=9)
    settings = _base_settings()
    out = _normalize_img_batch(stack.copy(), ["0", "1"], np.float16, settings)
    assert out.dtype == np.float16
    assert out.shape == stack.shape
    assert float(out.max()) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# concatenate_and_normalize - non-timelapse batching
# ---------------------------------------------------------------------------

def test_concatenate_and_normalize_single_batch(tmp_path, capsys):
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    names = ["plate1_A01_1.npy", "plate1_A01_2.npy",
             "plate1_A02_1.npy", "plate1_A02_2.npy"]
    _write_fields(src, names)

    out_fldr = concatenate_and_normalize(str(src), [0, 1],
                                         save_dtype=np.float32,
                                         settings=_base_settings(batch_size=10))

    assert out_fldr == os.path.join(str(tmp_path), "masks")
    saved = sorted(os.listdir(out_fldr))
    assert saved == ["stack_0_norm.npz"]

    with np.load(os.path.join(out_fldr, "stack_0_norm.npz")) as npz:
        data = npz["data"]
        filenames = [str(f) for f in npz["filenames"]]
    assert data.shape == (4, 8, 8, 2)
    assert data.dtype == np.float32
    assert sorted(filenames) == sorted(names)
    assert 0.0 <= float(data.min()) and float(data.max()) <= 1.0
    assert "All files concatenated and normalized" in capsys.readouterr().out


def test_concatenate_and_normalize_multiple_batches_with_randomize(tmp_path):
    """batch_size < nr_files -> several npz files; the tail batch is flushed."""
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    names = [f"plate1_A0{i}_1.npy" for i in range(1, 6)]   # 5 fields
    _write_fields(src, names)

    random.seed(1234)
    out_fldr = concatenate_and_normalize(
        str(src), [0, 1], settings=_base_settings(batch_size=2, randomize=True))

    saved = sorted(os.listdir(out_fldr))
    assert saved == ["stack_0_norm.npz", "stack_1_norm.npz", "stack_2_norm.npz"]

    seen, rows = [], 0
    for f in saved:
        with np.load(os.path.join(out_fldr, f)) as npz:
            seen.extend(str(x) for x in npz["filenames"])
            rows += npz["data"].shape[0]
    assert rows == 5
    assert sorted(seen) == sorted(names)
    # The final (partial) batch holds the single leftover field.
    with np.load(os.path.join(out_fldr, "stack_2_norm.npz")) as npz:
        assert npz["data"].shape == (1, 8, 8, 2)


def test_concatenate_and_normalize_selects_channel_subset(tmp_path):
    """Only the requested channels survive into the saved array."""
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    _write_fields(src, ["plate1_A01_1.npy", "plate1_A01_2.npy"], shape=(8, 8, 3))

    out_fldr = concatenate_and_normalize(str(src), [0, 2],
                                         settings=_base_settings())
    with np.load(os.path.join(out_fldr, "stack_0_norm.npz")) as npz:
        data = npz["data"]
    assert data.shape == (2, 8, 8, 2)
    assert float(data.max()) == pytest.approx(1.0, abs=1e-5)


def test_concatenate_and_normalize_ignores_non_npy_and_bad_npy(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    """Non-.npy files are skipped; an unreadable .npy is reported and skipped."""
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    _write_fields(src, ["g1_A01_1.npy", "g2_A01_1.npy", "g3_A01_1.npy"])
    (src / "readme.txt").write_text("not an array")
    (src / "b_broken.npy").write_bytes(b"definitely not a numpy file")

    # Deterministic iteration order so the broken file is NOT last (the
    # trailing-batch flush is exercised by its own test below).
    monkeypatch.setattr(os, "listdir", _sorted_listdir(os.listdir))

    out_fldr = concatenate_and_normalize(str(src), [0, 1],
                                         settings=_base_settings(batch_size=10))

    printed = capsys.readouterr().out
    assert "Error loading file" in printed and "b_broken.npy" in printed

    with np.load(os.path.join(out_fldr, "stack_0_norm.npz")) as npz:
        filenames = [str(f) for f in npz["filenames"]]
        assert npz["data"].shape == (3, 8, 8, 2)
    assert sorted(filenames) == ["g1_A01_1.npy", "g2_A01_1.npy", "g3_A01_1.npy"]
    assert "readme.txt" not in filenames


def test_concatenate_and_normalize_bad_last_file_keeps_trailing_batch(
        tmp_path, monkeypatch):
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    _write_fields(src, ["a_A01_1.npy", "b_A01_1.npy"])
    (src / "z_broken.npy").write_bytes(b"definitely not a numpy file")

    monkeypatch.setattr(os, "listdir", _sorted_listdir(os.listdir))

    out_fldr = concatenate_and_normalize(str(src), [0, 1],
                                         settings=_base_settings(batch_size=10))

    saved = os.path.join(out_fldr, "stack_0_norm.npz")
    assert os.path.exists(saved), "the two readable fields were dropped"
    with np.load(saved) as npz:
        assert npz["data"].shape[0] == 2


def test_concatenate_and_normalize_pads_mismatched_shapes(tmp_path, capsys):
    """Fields of different X/Y sizes are zero-padded up to the batch maximum."""
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    src.mkdir()
    np.save(src / "a_small.npy", _plane(8, 8, 2, seed=0))
    np.save(src / "b_wide.npy", _plane(10, 6, 2, seed=1))

    out_fldr = concatenate_and_normalize(str(src), [0, 1],
                                         settings=_base_settings(batch_size=10))

    printed = capsys.readouterr().out
    assert "arrays with multiple shapes found in batch" in printed

    with np.load(os.path.join(out_fldr, "stack_0_norm.npz")) as npz:
        data = npz["data"]
        filenames = [str(f) for f in npz["filenames"]]
    assert data.shape == (2, 10, 8, 2)

    i_small = filenames.index("a_small.npy")
    i_wide = filenames.index("b_wide.npy")
    # Padded regions stay zero after normalisation.
    assert np.all(data[i_small, 8:, :, :] == 0)
    assert np.all(data[i_wide, :, 6:, :] == 0)
    # ...while the real data region is not all zero.
    assert data[i_small, :8, :, :].max() > 0
    assert data[i_wide, :, :6, :].max() > 0


def test_concatenate_and_normalize_plots_only_first_batch(tmp_path, monkeypatch,
                                                          capsys):
    """settings['plot'] triggers plot_arrays exactly once, for batch 0."""
    import spacr.plot
    from spacr.io import concatenate_and_normalize

    calls = []
    monkeypatch.setattr(spacr.plot, "plot_arrays",
                        lambda *a, **kw: calls.append((a, kw)))

    src = tmp_path / "stack"
    _write_fields(src, ["p_A01_1.npy", "p_A01_2.npy", "p_A01_3.npy"])

    out_fldr = concatenate_and_normalize(
        str(src), [0, 1],
        settings=_base_settings(batch_size=1, plot=True, figuresize=6,
                                cmap="viridis", nr=2))

    assert len(os.listdir(out_fldr)) == 3
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == os.path.join(out_fldr, "stack_0_norm.npz")
    assert args[1] == 6 and args[2] == "viridis"
    assert kwargs == {"nr": 2, "normalize": False}
    assert "plotting:" in capsys.readouterr().out


def test_concatenate_and_normalize_no_plot_when_disabled(tmp_path, monkeypatch):
    import spacr.plot
    from spacr.io import concatenate_and_normalize

    calls = []
    monkeypatch.setattr(spacr.plot, "plot_arrays",
                        lambda *a, **kw: calls.append(a))

    src = tmp_path / "stack"
    _write_fields(src, ["p_A01_1.npy", "p_A01_2.npy"])
    concatenate_and_normalize(str(src), [0, 1],
                             settings=_base_settings(plot=False))
    assert calls == []


def test_concatenate_and_normalize_requires_settings(tmp_path):
    """settings is mandatory: omitting it fails fast and says so."""
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    _write_fields(src, ["p_A01_1.npy"])

    with pytest.raises(ValueError) as exc:
        concatenate_and_normalize(str(src), [0, 1], settings=None)
    assert "requires a settings dict" in str(exc.value)
    # Fails before any side effect: no masks/ folder is left behind.
    assert not (tmp_path / "masks").exists()


def test_concatenate_and_normalize_accepts_string_channels(tmp_path):
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    _write_fields(src, ["p_A01_1.npy", "p_A01_2.npy"], shape=(8, 8, 3))

    out_fldr = concatenate_and_normalize(str(src), ["0", "2"],
                                         settings=_base_settings())
    with np.load(os.path.join(out_fldr, "stack_0_norm.npz")) as npz:
        assert npz["data"].shape == (2, 8, 8, 2)


def test_concatenate_and_normalize_drops_none_channels(tmp_path):
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    _write_fields(src, ["p_A01_1.npy", "p_A01_2.npy"])

    out_fldr = concatenate_and_normalize(str(src), [0, None, 1],
                                         settings=_base_settings())
    with np.load(os.path.join(out_fldr, "stack_0_norm.npz")) as npz:
        assert npz["data"].shape == (2, 8, 8, 2)


# ---------------------------------------------------------------------------
# concatenate_and_normalize - timelapse path
# ---------------------------------------------------------------------------

def test_concatenate_and_normalize_timelapse_groups_by_field(tmp_path,
                                                             monkeypatch):
    """Timelapse mode writes one npz per plate_well_field, ordered by timepoint."""
    import spacr.plot
    from spacr.io import concatenate_and_normalize

    calls = []
    monkeypatch.setattr(spacr.plot, "plot_arrays",
                        lambda *a, **kw: calls.append(a))

    src = tmp_path / "stack"
    names = [
        "plate1_A01_1_0002.npy", "plate1_A01_1_0001.npy",
        "plate1_A01_2_0001.npy", "plate1_A01_2_0002.npy",
    ]
    _write_fields(src, names)
    # Ignored by _generate_time_lists: wrong extension / too few parts.
    (src / "junk.txt").write_text("x")
    np.save(src / "plate1_A01.npy", _plane())

    out_fldr = concatenate_and_normalize(
        str(src), [0, 1],
        settings=_base_settings(timelapse=True, plot=True, nr=1))

    saved = sorted(os.listdir(out_fldr))
    assert saved == ["plate1_A01_1_norm_timelapse.npz",
                     "plate1_A01_2_norm_timelapse.npz"]

    with np.load(os.path.join(out_fldr,
                              "plate1_A01_1_norm_timelapse.npz")) as npz:
        data = npz["data"]
        filenames = [str(f) for f in npz["filenames"]]
    assert data.shape == (2, 8, 8, 2)
    assert data.dtype == np.float32
    # sorted by timepoint, not by directory order
    assert filenames == ["plate1_A01_1_0001.npy", "plate1_A01_1_0002.npy"]

    # plot_arrays is called for the first group only.
    assert len(calls) == 1
    assert calls[0][0].endswith("_norm_timelapse.npz")


def test_concatenate_and_normalize_timelapse_handles_bad_metadata(tmp_path,
                                                                  capsys):
    """A group whose frames disagree in shape is reported, not raised."""
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    src.mkdir()
    np.save(src / "plate1_A01_1_0001.npy", _plane(8, 8, 2, seed=0))
    np.save(src / "plate1_A01_1_0002.npy", _plane(12, 12, 2, seed=1))

    out_fldr = concatenate_and_normalize(
        str(src), [0, 1], settings=_base_settings(timelapse=True))

    printed = capsys.readouterr().out
    assert "Error processing files" in printed
    assert "plate_well_field_time.npy" in printed
    assert "Error:" in printed
    assert out_fldr == os.path.join(str(tmp_path), "masks")
    assert os.listdir(out_fldr) == []


def test_concatenate_and_normalize_timelapse_no_valid_files(tmp_path):
    """No parsable timelapse names -> nothing written, folder still returned."""
    from spacr.io import concatenate_and_normalize

    src = tmp_path / "stack"
    src.mkdir()
    (src / "nope.txt").write_text("x")

    out_fldr = concatenate_and_normalize(
        str(src), [0, 1], settings=_base_settings(timelapse=True))
    assert os.path.isdir(out_fldr)
    assert os.listdir(out_fldr) == []


# ---------------------------------------------------------------------------
# _get_lists_for_normalization
# ---------------------------------------------------------------------------

def test_get_lists_for_normalization_pathogen_only():
    """Only the pathogen channel is configured -> single-entry lists."""
    from spacr.io import _get_lists_for_normalization

    settings = {
        "nucleus_channel": None,
        "cell_channel": None,
        "pathogen_channel": 3,
        "pathogen_background": 130,
        "pathogen_Signal_to_noise": 2,
        "remove_background_pathogen": True,
    }
    bg, snr, thr, rb = _get_lists_for_normalization(settings)
    assert bg == [130]
    assert snr == [2]
    assert thr == [260]
    assert rb == [True]


def test_get_lists_for_normalization_order_follows_object_type():
    """Entries come out in nucleus/cell/pathogen order, not channel-index order."""
    from spacr.io import _get_lists_for_normalization

    settings = {
        "nucleus_channel": 5,
        "cell_channel": 1,
        "pathogen_channel": 3,
        "nucleus_background": 111,
        "nucleus_Signal_to_noise": 3,
        "remove_background_nucleus": False,
        "cell_background": 222,
        "cell_Signal_to_noise": 4,
        "remove_background_cell": True,
        "pathogen_background": 333,
        "pathogen_Signal_to_noise": 5,
        "remove_background_pathogen": False,
    }
    bg, snr, thr, rb = _get_lists_for_normalization(settings)
    assert bg == [111, 222, 333]
    assert snr == [3, 4, 5]
    assert thr == [333, 888, 1665]
    assert rb == [False, True, False]


def test_get_lists_for_normalization_all_none():
    from spacr.io import _get_lists_for_normalization

    settings = {"nucleus_channel": None, "cell_channel": None,
                "pathogen_channel": None}
    assert _get_lists_for_normalization(settings) == ([], [], [], [])
