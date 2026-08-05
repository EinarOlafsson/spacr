"""measure_crop's settings-validation and early-return branches, plus the
error paths of generate_cellpose_train_set.

All CPU-only: the validations fire before any segmentation work.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from spacr.settings import get_measure_crop_settings


def _merged(tmp_path, rng, n_chan=4):
    """A minimal merged/ folder with one stack (image channels + 3 masks)."""
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    (tmp_path / "measurements").mkdir(parents=True, exist_ok=True)
    H = W = 32
    layers = [rng.integers(50, 400, (H, W)).astype(np.uint16) for _ in range(n_chan)]
    cell = np.zeros((H, W), np.uint16); cell[4:20, 4:20] = 1
    nuc = np.zeros((H, W), np.uint16); nuc[8:14, 8:14] = 1
    pat = np.zeros((H, W), np.uint16); pat[10:12, 10:12] = 1
    data = np.stack(layers + [cell, nuc, pat], axis=-1).astype(np.uint16)
    np.save(merged / "plate1_A01_F001.npy", data)
    return merged


def _settings(merged, **over):
    s = get_measure_crop_settings(settings={})
    s.update({
        "src": str(merged), "channels": [0, 1, 2, 3],
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "png_dims": [0, 1, 2], "png_size": [32, 32],
        "save_measurements": True, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": [1, 99], "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        "cytoplasm": True,
    })
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# early-return validations
# ---------------------------------------------------------------------------

def test_non_integer_mask_dims_aborts(tmp_path, rng, capsys):
    """Non-int mask/size settings trip the int_setting_keys guard."""
    from spacr.measure import measure_crop
    merged = _merged(tmp_path, rng)
    measure_crop(_settings(merged, cell_mask_dim="four"))
    assert "must all be integers" in capsys.readouterr().out


def test_channels_not_a_list_aborts(tmp_path, rng, capsys):
    from spacr.measure import measure_crop
    merged = _merged(tmp_path, rng)
    measure_crop(_settings(merged, channels="0,1,2"))
    assert "channels should be a list" in capsys.readouterr().out


def test_crop_mode_not_a_list_is_coerced(tmp_path, rng, capsys):
    """A bare string crop_mode is wrapped into a list with a warning."""
    from spacr.measure import measure_crop
    merged = _merged(tmp_path, rng)
    measure_crop(_settings(merged, crop_mode="cell"))
    out = capsys.readouterr().out
    assert "crop_mode should be a list" in out or "Converted crop_mode" in out


def test_timelapse_nucleus_announces_relabel(tmp_path, rng, capsys):
    """timelapse_objects='nucleus' logs that cells get nucleus labels."""
    from spacr.measure import measure_crop
    merged = _merged(tmp_path, rng)
    measure_crop(_settings(merged, timelapse_objects="nucleus"))
    assert "cells will be relabeled" in capsys.readouterr().out


def test_timelapse_disables_save_png(tmp_path, rng):
    """timelapse=True forces save_png off before anything else runs."""
    from spacr.measure import measure_crop
    merged = _merged(tmp_path, rng)
    s = _settings(merged, timelapse=True, save_png=True,
                  timelapse_objects="cell")
    measure_crop(s)
    assert s["save_png"] is False


def test_src_must_be_str_or_list(tmp_path, rng):
    """A non-str/list source is rejected instead of failing silently."""
    from spacr.measure import measure_crop
    s = _settings(_merged(tmp_path, rng))
    s["src"] = 12345
    with pytest.raises(ValueError, match="src must be a string or a list"):
        measure_crop(s)


# ---------------------------------------------------------------------------
# generate_cellpose_train_set error paths
# ---------------------------------------------------------------------------

def test_train_set_skips_unreadable_mask(tmp_path, rng, capsys):
    """A mask cv2 can't decode is reported and skipped, not fatal."""
    from spacr.measure import generate_cellpose_train_set
    folder = tmp_path / "expA"
    (folder / "masks").mkdir(parents=True)
    (folder / "masks" / "broken.tif").write_bytes(b"not a tiff")
    (folder / "broken.tif").write_bytes(b"not a tiff")
    generate_cellpose_train_set([str(folder)], str(tmp_path / "out"),
                                min_objects=1)
    assert "Error reading" in capsys.readouterr().out


def test_train_set_reports_copy_failure(tmp_path, rng, capsys):
    """A mask that passes min_objects but whose image is missing is reported."""
    from spacr.measure import generate_cellpose_train_set
    folder = tmp_path / "expB"
    (folder / "masks").mkdir(parents=True)
    m = np.zeros((32, 32), np.uint16)
    for i in range(1, 7):
        m[i * 4:i * 4 + 2, 2:4] = i
    tifffile.imwrite(folder / "masks" / "lonely.tif", m)
    # NOTE: no matching image alongside -> shutil.copy of the image fails
    generate_cellpose_train_set([str(folder)], str(tmp_path / "out2"),
                                min_objects=5)
    assert "Error copying" in capsys.readouterr().out
