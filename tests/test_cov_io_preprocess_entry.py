"""CPU-only coverage for the spacr.io preprocessing entry points.

Targets ``spacr/io.py`` lines 1717-1974:

* :func:`spacr.io.preprocess_img_data` — extension sniffing (found / not
  found + the existing ``stack``/``channel_stack``/``masks`` short
  circuits), the mask-channel dedup + int coercion loop, test mode (both
  the deletable and the non-deletable pre-existing ``test/`` folder), the
  stack-building ``try``/``except`` (channel-count mismatch, timelapse
  movies, plotting, all-to-MIP) and the ``cellpose_*`` channel remap.
* :func:`spacr.io._get_avg_object_size` — the malformed-mask warning and
  the empty-input branches.

The heavy collaborators (``_rename_and_organize_image_files``,
``plot_arrays``, ``concatenate_and_normalize``, ``_create_movies_from_npy_per_channel``)
are replaced with recording stubs so the branch under test is isolated —
except in ``test_end_to_end_real_run``, which runs the genuine pipeline on
eight 128x128 synthetic Yokogawa TIFFs and inspects the emitted arrays.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_tif(path, arr):
    """Write a real TIFF (tifffile when available, Pillow otherwise)."""
    try:
        import tifffile
        tifffile.imwrite(str(path), arr)
        return
    except Exception:
        from PIL import Image
        Image.fromarray(arr).save(str(path))


def _tiny_img(seed=0, shape=(16, 16)):
    """Small deterministic uint16 image with a bright square (never flat)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(50, 200, size=shape, dtype=np.uint16)
    img[4:10, 4:10] = 40000
    return img


class _Rec:
    """Callable that records every call and returns a fixed value."""

    def __init__(self, ret=None):
        self.calls = []
        self.ret = ret

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.ret

    @property
    def n(self):
        return len(self.calls)


def _fake_rename(n_files=4, n_channels=2, shape=(8, 8)):
    """Stub for ``_rename_and_organize_image_files``.

    Creates ``<src>/stack`` with ``n_files`` real ``.npy`` arrays (so the
    downstream ``os.path.exists`` checks and ``_mip_all`` see something
    genuine) and reports ``n_channels`` channel folders.
    """
    rec = _Rec(ret=n_channels)

    def _f(src, regex, batch_size, metadata_type, img_format, timelapse=False,
           save_original_images=True):
        rec.calls.append(((src, regex, batch_size, metadata_type, img_format),
                          {"timelapse": timelapse,
                           "save_original_images": save_original_images}))
        stack = os.path.join(src, "stack")
        os.makedirs(stack, exist_ok=True)
        for i in range(n_files):
            arr = np.arange(shape[0] * shape[1] * n_channels, dtype=np.uint16)
            arr = arr.reshape(shape[0], shape[1], n_channels) + i
            np.save(os.path.join(stack, f"plate1_A01_00{i}_T0001.npy"), arr)
        return n_channels

    return _f, rec


def _patch_common(monkeypatch, rename=None, keep=()):
    """Patch every heavyweight collaborator; return the recorder dict.

    ``keep`` names recorders that should NOT be installed, so the genuine
    spacr implementation runs instead.
    """
    import spacr.io as IO
    import spacr.plot as PLOT

    recs = {
        "concat": _Rec(ret="masks"),
        "plot": _Rec(),
        "movies": _Rec(),
        "mip": _Rec(),
        "merge": _Rec(ret=2),
    }
    targets = {
        "concat": (IO, "concatenate_and_normalize"),
        "plot": (PLOT, "plot_arrays"),
        "movies": (IO, "_create_movies_from_npy_per_channel"),
        "mip": (IO, "_mip_all"),
        "merge": (IO, "_merge_channels"),
    }
    for key, (mod, attr) in targets.items():
        if key in keep:
            continue
        monkeypatch.setattr(mod, attr, recs[key])
    if rename is not None:
        monkeypatch.setattr(IO, "_rename_and_organize_image_files", rename)
    return recs


def _settings(src, **over):
    s = {
        "src": str(src),
        "metadata_type": "cellvoyager",
        "custom_regex": None,
        "channels": [0, 1],
        "nucleus_channel": 0,
        "cell_channel": 1,
        "pathogen_channel": None,
        "organelle_channel": None,
        "plot": False,
        "batch_size": 1,
        "test_mode": False,
        "timelapse": False,
        "all_to_mip": False,
        "normalize": True,
    }
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# no valid image files -> the "existing folder" short circuits
# ---------------------------------------------------------------------------

def test_existing_masks_folder_short_circuits_and_empty_dirs_are_pruned(tmp_path, capsys, monkeypatch):
    """No images + a masks/ folder -> return untouched settings immediately."""
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    (src / "notes.txt").write_text("not an image")
    for name in ("stack", "channel_stack", "masks"):
        d = src / name
        d.mkdir()
        (d / "keep.bin").write_bytes(b"x")
    (src / "empty_dir").mkdir()

    recs = _patch_common(monkeypatch)
    settings = _settings(src)
    out_settings, out_src = preprocess_img_data(settings)

    # Early return: same dict object, no defaults applied, src unchanged.
    assert out_settings is settings
    assert "lower_percentile" not in out_settings
    assert "save_dtype" not in out_settings
    assert out_src == str(src)
    # Nothing downstream ran.
    assert recs["concat"].n == 0
    assert recs["merge"].n == 0
    # delete_empty_subdirectories ran (folder had < 100 entries).
    assert not (src / "empty_dir").exists()
    assert (src / "masks").is_dir()

    out = capsys.readouterr().out
    assert "Could not find any" in out
    assert "Found existing stack folder." in out
    assert "Found existing channel_stack folder." in out
    assert "Found existing masks folder. Skipping preprocessing" in out


def test_existing_stack_without_masks_is_reused_not_remerged(tmp_path, capsys, monkeypatch):
    """stack/ present, masks/ absent -> skip _merge_channels, normalize stack/."""
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    (src / "notes.txt").write_text("not an image")
    stack = src / "stack"
    stack.mkdir()
    np.save(stack / "fov.npy", np.zeros((4, 4, 2), dtype=np.uint16))
    cs = src / "channel_stack"
    cs.mkdir()
    (cs / "keep.bin").write_bytes(b"x")

    recs = _patch_common(monkeypatch)
    settings = _settings(src)
    out_settings, out_src = preprocess_img_data(settings)

    assert recs["merge"].n == 0, "_merge_channels must not run when stack/ exists"
    assert recs["concat"].n == 1
    kwargs = recs["concat"].calls[0][1]
    assert kwargs["src"] == str(stack)
    assert kwargs["channels"] == [0, 1]
    assert kwargs["save_dtype"] is np.float32
    # Defaults were applied this time (no early return).
    assert out_settings["lower_percentile"] == 2
    assert out_src == str(src)

    out = capsys.readouterr().out
    assert "Found existing stack folder." in out
    assert "Found existing channel_stack folder." not in out.split("Found existing stack folder.")[0]


def test_channel_subfolders_are_merged_into_a_stack(tmp_path, monkeypatch):
    """No recognised image extension in src -> _merge_channels builds stack/."""
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    (src / "notes.txt").write_text("not an image")
    imgs = {}
    for chan in ("0", "1"):
        d = src / chan
        d.mkdir()
        for i, name in enumerate(("fovA.tif", "fovB.tif")):
            arr = _tiny_img(seed=int(chan) * 10 + i)
            _write_tif(d / name, arr)
            imgs[(chan, name)] = arr

    recs = _patch_common(monkeypatch, keep=("merge",))  # real _merge_channels
    settings = _settings(src)
    _, out_src = preprocess_img_data(settings)

    stack = src / "stack"
    assert stack.is_dir()
    npys = sorted(p.name for p in stack.glob("*.npy"))
    assert npys == ["fovA.npy", "fovB.npy"]
    merged = np.load(stack / "fovA.npy")
    assert merged.shape == (16, 16, 2)
    assert np.array_equal(merged[:, :, 0], imgs[("0", "fovA.tif")])
    assert np.array_equal(merged[:, :, 1], imgs[("1", "fovA.tif")])
    assert recs["concat"].calls[0][1]["src"] == str(stack)
    assert out_src == str(src)


# ---------------------------------------------------------------------------
# mask-channel dedup / coercion + the cellpose_* remap
# ---------------------------------------------------------------------------

def test_mask_channels_dedup_and_cellpose_remap(tmp_path, monkeypatch):
    """Duplicate + None channels collapse; cellpose_* maps to stack position."""
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    stack = src / "stack"
    stack.mkdir()
    np.save(stack / "fov.npy", np.zeros((4, 4, 2), dtype=np.uint16))

    recs = _patch_common(monkeypatch)
    settings = _settings(src, nucleus_channel=1, cell_channel=0,
                         pathogen_channel=1, organelle_channel=None)
    out_settings, _ = preprocess_img_data(settings)

    # order of first appearance: nucleus(1) then cell(0); pathogen(1) is a dup.
    assert recs["concat"].calls[0][1]["channels"] == [1, 0]
    assert out_settings["cellpose_nucleus_channel"] == 0
    assert out_settings["cellpose_cell_channel"] == 1
    assert out_settings["cellpose_pathogen_channel"] == 0
    assert "cellpose_organelle_channel" not in out_settings


def test_uncoercible_channel_values_are_dropped(tmp_path, monkeypatch):
    """'abc' (ValueError) and a tuple (TypeError) never reach the stack slice."""
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    stack = src / "stack"
    stack.mkdir()
    np.save(stack / "fov.npy", np.zeros((4, 4, 2), dtype=np.uint16))

    recs = _patch_common(monkeypatch)
    settings = _settings(src, nucleus_channel=(0, 1), cell_channel="abc",
                         pathogen_channel=2, organelle_channel=None)
    out_settings, _ = preprocess_img_data(settings)

    assert recs["concat"].calls[0][1]["channels"] == [2]
    assert out_settings["cellpose_pathogen_channel"] == 0
    assert "cellpose_nucleus_channel" not in out_settings
    assert "cellpose_cell_channel" not in out_settings


@pytest.mark.xfail(strict=True, reason=(
    "BUG: preprocess_img_data coerces channel indices to int when building "
    "mask_channels but looks them up un-coerced in `seen`, so string channel "
    "indices ('0') silently produce no cellpose_* remap"))
def test_string_channel_indices_are_remapped(tmp_path, monkeypatch):
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    stack = src / "stack"
    stack.mkdir()
    np.save(stack / "fov.npy", np.zeros((4, 4, 2), dtype=np.uint16))

    recs = _patch_common(monkeypatch)
    settings = _settings(src, nucleus_channel="1", cell_channel="0")
    out_settings, _ = preprocess_img_data(settings)

    # The coercion in the dedup loop works ...
    assert recs["concat"].calls[0][1]["channels"] == [1, 0]
    # ... but the remap below it silently does nothing.
    assert out_settings.get("cellpose_nucleus_channel") == 0
    assert out_settings.get("cellpose_cell_channel") == 1


# ---------------------------------------------------------------------------
# test mode
# ---------------------------------------------------------------------------

def test_test_mode_reports_undeletable_test_folder_and_uses_subset(
        yokogawa_cellvoyager_dir, capsys, monkeypatch):
    """A non-empty test/ folder cannot be rmdir'd -> error is reported, run continues."""
    from spacr.io import preprocess_img_data

    src = yokogawa_cellvoyager_dir["src"]
    leftover = src / "test"
    leftover.mkdir()
    (leftover / "stale.txt").write_text("blocks rmdir")

    recs = _patch_common(monkeypatch)  # real _rename_and_organize runs
    settings = _settings(src, plot=False, test_mode=True, test_images=1,
                         random_test=True, batch_size=1)
    out_settings, out_src = preprocess_img_data(settings)

    out = capsys.readouterr().out
    assert "Error deleting test directory" in out
    assert "Delete manually before running test mode" in out
    assert "Running spacr in test mode" in out

    # src moved into the test folder, and test mode forces plotting on.
    assert out_src == str(leftover)
    assert out_settings["src"] == str(leftover)
    assert out_settings["plot"] is True

    # exactly one (plate, well, field) set (= 2 channel files) was copied and
    # turned into a single 2-channel stack; the originals stay untouched.
    stack_files = sorted((leftover / "stack").glob("*.npy"))
    assert len(stack_files) == 1
    assert np.load(stack_files[0]).shape == (128, 128, 2)
    assert len(sorted((leftover / "orig").glob("*.tif"))) == 2
    assert len(sorted(src.glob("*.tif"))) == 8

    # plot_arrays was driven from the forced settings['plot'] = True.
    assert recs["plot"].n >= 1
    assert recs["plot"].calls[0][0][0] == str(leftover / "stack")
    assert recs["concat"].calls[0][1]["src"] == str(leftover / "stack")


def test_test_mode_deletes_empty_test_folder_when_cleanup_is_skipped(tmp_path, capsys, monkeypatch):
    """>= 100 entries -> no empty-dir pruning, so test/ survives to be rmdir'd."""
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    for i, (well, field, chan) in enumerate([("A01", "001", "01"), ("A01", "001", "02"),
                                             ("A01", "002", "01"), ("A01", "002", "02")]):
        name = f"plate1_{well}_T0001F{field}L01A01Z01C{chan}.tif"
        _write_tif(src / name, _tiny_img(seed=i))
    for i in range(100):
        (src / f"pad_{i:03d}.txt").write_text("x")
    (src / "test").mkdir()          # empty: must survive the < 100 pruning check
    (src / "junk").mkdir()          # empty: proves pruning was skipped

    recs = _patch_common(monkeypatch)
    settings = _settings(src, test_mode=True, test_images=10, random_test=True,
                         batch_size=1)
    out_settings, out_src = preprocess_img_data(settings)

    out = capsys.readouterr().out
    assert "Deleted test directory" in out
    assert "Found 4 tif files" in out
    assert (src / "junk").is_dir(), "empty-dir pruning should have been skipped"

    assert out_src == str(src / "test")
    assert out_settings["plot"] is True
    # both fields were copied into test/ and stacked
    assert len(sorted((src / "test" / "stack").glob("*.npy"))) == 2
    assert recs["concat"].n == 1


# ---------------------------------------------------------------------------
# the stack-building try block
# ---------------------------------------------------------------------------

def test_channel_count_mismatch_rewrites_settings_channels(tmp_path, capsys, monkeypatch):
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    _write_tif(src / "plate1_A01_T0001F001L01A01Z01C01.tif", _tiny_img())

    rename, rename_rec = _fake_rename(n_files=4, n_channels=3)
    recs = _patch_common(monkeypatch, rename=rename)
    settings = _settings(src, channels=[0, 1])
    out_settings, _ = preprocess_img_data(settings)

    assert rename_rec.n == 1
    assert out_settings["channels"] == [0, 1, 2]
    out = capsys.readouterr().out
    assert "Number of channels does not match number of channel folders" in out
    assert "Changing channels from [0, 1] to [0, 1, 2]" in out
    assert recs["concat"].n == 1


def test_timelapse_triggers_movie_generation(tmp_path, monkeypatch):
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    _write_tif(src / "plate1_A01_T0001F001L01A01Z01C01.tif", _tiny_img())

    rename, rename_rec = _fake_rename(n_files=2, n_channels=2)
    recs = _patch_common(monkeypatch, rename=rename)
    settings = _settings(src, timelapse=True, fps=7)
    preprocess_img_data(settings)

    assert rename_rec.calls[0][1]["timelapse"] is True
    assert recs["movies"].n == 1
    args, kwargs = recs["movies"].calls[0]
    assert args[0] == str(src / "stack")
    assert kwargs == {"fps": 7}


def test_all_to_mip_appends_projection_and_replots(tmp_path, capsys, monkeypatch):
    """all_to_mip + plot -> real _mip_all adds a MIP plane, plot runs twice."""
    from spacr.io import preprocess_img_data
    import spacr.io as IO

    src = tmp_path / "plate1"
    src.mkdir()
    _write_tif(src / "plate1_A01_T0001F001L01A01Z01C01.tif", _tiny_img())

    rename, _ = _fake_rename(n_files=2, n_channels=2)
    recs = _patch_common(monkeypatch, rename=rename, keep=("mip",))  # real MIP
    assert IO._mip_all.__module__ == "spacr.io"

    settings = _settings(src, all_to_mip=True, plot=True, figuresize=4,
                         cmap="gray", nr=1)
    preprocess_img_data(settings)

    stack = src / "stack"
    arrs = sorted(stack.glob("*.npy"))
    assert len(arrs) == 2
    for p in arrs:
        a = np.load(p)
        assert a.shape == (8, 8, 3), "MIP plane was not appended"
        assert np.array_equal(a[:, :, 2], np.max(a[:, :, :2], axis=2))

    # plotted once before and once after the MIP
    assert recs["plot"].n == 2
    for call in recs["plot"].calls:
        assert call[0][0] == str(stack)
        assert call[0][1] == 4 and call[0][2] == "gray"
    assert "plotting 1 images" in capsys.readouterr().out


@pytest.mark.xfail(strict=True, reason=(
    "BUG: preprocess_img_data computes `all_imgs = len(stack_path)` — the "
    "LENGTH OF THE PATH STRING — instead of the number of stacked images, so "
    "the batch-size sanity check fires (and aborts the stack step) purely "
    "because of how long the source path happens to be"))
def test_batch_size_check_uses_image_count_not_path_length(tmp_path, capsys, monkeypatch):
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    _write_tif(src / "plate1_A01_T0001F001L01A01Z01C01.tif", _tiny_img())

    # 4 images: no batch of size 1 is possible for any batch_size > 1. But the
    # path-length arithmetic makes `len(stack_path) % batch_size == 1` here.
    stack_path = os.path.join(str(src), "stack")
    bad_batch = len(stack_path) - 1
    assert len(stack_path) % bad_batch == 1  # guard: the trap is armed

    rename, _ = _fake_rename(n_files=4, n_channels=2)
    recs = _patch_common(monkeypatch, rename=rename)
    settings = _settings(src, batch_size=bad_batch, plot=True)
    preprocess_img_data(settings)

    out = capsys.readouterr().out
    assert "Last batch of size 1 detected" not in out
    assert recs["plot"].n == 1, "the plotting step was skipped by a bogus error"


# ---------------------------------------------------------------------------
# real end-to-end run (no stubs)
# ---------------------------------------------------------------------------

def test_end_to_end_real_run(yokogawa_cellvoyager_dir):
    """Raw CellVoyager TIFFs -> stack/*.npy -> masks/*.npz, all for real."""
    from spacr.io import preprocess_img_data

    src = yokogawa_cellvoyager_dir["src"]
    settings = _settings(
        src,
        plot=False,
        batch_size=1,                 # keeps len(stack_path) % batch_size == 0
        randomize=False,
        lower_percentile=2,
        nucleus_channel=0, cell_channel=1,
        nucleus_background=100, nucleus_Signal_to_noise=5,
        remove_background_nucleus=False,
        cell_background=100, cell_Signal_to_noise=5,
        remove_background_cell=False,
    )
    out_settings, out_src = preprocess_img_data(settings)

    assert out_src == str(src)
    # 2 wells x 2 fields -> 4 FOV stacks, each with the 2 acquired channels.
    stacks = sorted((src / "stack").glob("*.npy"))
    assert len(stacks) == 4
    assert np.load(stacks[0]).shape == (128, 128, 2)
    # originals preserved under orig/
    assert len(sorted((src / "orig").glob("*.tif"))) == 8

    # normalized output, one npz per batch of 1
    npzs = sorted((src / "masks").glob("*_norm.npz"))
    assert len(npzs) == 4
    with np.load(npzs[0]) as z:
        data = z["data"]
        names = z["filenames"]
    assert data.shape == (1, 128, 128, 2)
    assert data.dtype == np.float32
    assert data.min() >= 0.0 and data.max() <= 1.0
    assert data.max() > 0.0
    assert len(names) == 1 and str(names[0]).endswith(".npy")

    assert out_settings["cellpose_nucleus_channel"] == 0
    assert out_settings["cellpose_cell_channel"] == 1
    assert out_settings["save_dtype"] == "uint16"   # defaults were applied


# ---------------------------------------------------------------------------
# _get_avg_object_size edge branches
# ---------------------------------------------------------------------------

def test_get_avg_object_size_warns_on_invalid_dimensionality(capsys):
    from spacr.io import _get_avg_object_size

    bad = np.array([1, 2, 3], dtype=np.int32)          # ndim 1, non-empty
    avg_n, avg_size = _get_avg_object_size([bad])

    assert avg_n == 0
    assert avg_size == 0
    out = capsys.readouterr().out
    assert "Mask 0 has invalid dimension: 1" in out
    assert "is empty" not in out


def test_get_avg_object_size_mixes_empty_and_invalid_masks(capsys):
    from spacr.io import _get_avg_object_size

    good = np.zeros((10, 10), dtype=np.int32)
    good[2:5, 2:5] = 1                                  # one 9-px object
    empty = np.zeros((10, 10), dtype=np.int32)
    invalid = np.ones((2, 2, 2, 2), dtype=np.int32)     # ndim 4, non-empty

    avg_n, avg_size = _get_avg_object_size([good, empty, invalid])

    assert avg_n == pytest.approx(1 / 3)
    assert avg_size == pytest.approx(9.0)
    out = capsys.readouterr().out
    assert "Mask 1 is empty." in out
    assert "Mask 2 has invalid dimension: 4" in out


def test_get_avg_object_size_empty_input_returns_zeros():
    from spacr.io import _get_avg_object_size

    assert _get_avg_object_size([]) == (0, 0)
