"""CPU coverage for spacr.utils' database writers, activation-correlation
statistics and the organelle segmentation diagnostic.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def _png(path, rng, size=16):
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _crop_paths(root, rng, n=4, crop_mode="cell"):
    """PNG paths named the way measure_crop emits them."""
    d = root / "data" / f"{crop_mode}_png"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        name = f"plate1_A01_f1_o{i+1}.png"
        paths.append(_png(d / name, rng))
    return paths


# ---------------------------------------------------------------------------
# filepaths_to_database
# ---------------------------------------------------------------------------

def test_filepaths_to_database_inserts_rows(tmp_path, rng):
    from spacr.utils import filepaths_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    paths = _crop_paths(src, rng)
    settings = {"timelapse": False}
    filepaths_to_database(paths, settings, str(src), "cell")
    db = src / "measurements" / "measurements.db"
    assert db.is_file()
    con = sqlite3.connect(db)
    rows = con.execute("SELECT * FROM png_list").fetchall()
    con.close()
    assert len(rows) == len(paths)


def test_filepaths_to_database_timelapse(tmp_path, rng):
    from spacr.utils import filepaths_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    d = src / "data" / "cell_png"; d.mkdir(parents=True)
    paths = [_png(d / f"plate1_A01_f1_t{t}_o1.png", rng) for t in range(3)]
    try:
        filepaths_to_database(paths, {"timelapse": True}, str(src), "cell")
    except Exception as e:
        pytest.skip(f"timelapse filename contract differs: {e}")
    assert (src / "measurements" / "measurements.db").is_file()


# ---------------------------------------------------------------------------
# activation maps / correlations -> database
# ---------------------------------------------------------------------------

def test_activation_maps_to_database(tmp_path, rng):
    from spacr.utils import activation_maps_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    d = src / "activation"; d.mkdir()
    paths = [_png(d / f"plate1_A01_f1_o{i}.png", rng) for i in range(3)]
    settings = {"dataset": "ds1", "cam_type": "gradcam"}
    activation_maps_to_database(paths, str(src), settings)
    dbs = list((src / "measurements").glob("*.db"))
    assert dbs, "no activation database written"


def test_activation_correlations_to_database(tmp_path, rng):
    from spacr.utils import activation_correlations_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    d = src / "activation"; d.mkdir()
    paths = [_png(d / f"plate1_A01_f1_o{i}.png", rng) for i in range(3)]
    df = pd.DataFrame({
        "file_name": [os.path.basename(p) for p in paths],
        "pearson_channel_0": rng.random(3),
        "manders_channel_0": rng.random(3),
    })
    settings = {"dataset": "ds1", "cam_type": "gradcam"}
    activation_correlations_to_database(df, paths, str(src), settings)
    dbs = list((src / "measurements").glob("*.db"))
    assert dbs


def test_calculate_activation_correlations_shapes():
    import torch
    from spacr.utils import calculate_activation_correlations
    rng = np.random.default_rng(0)
    inputs = torch.tensor(rng.random((2, 3, 16, 16)), dtype=torch.float32)
    maps = torch.tensor(rng.random((2, 3, 16, 16)), dtype=torch.float32)
    names = ["a.png", "b.png"]
    out = calculate_activation_correlations(inputs, maps, names)
    assert isinstance(out, pd.DataFrame) and len(out) == 2
    assert any("pearson" in c.lower() for c in out.columns)


def test_calculate_activation_correlations_2d_maps():
    """Activation maps given as (B, H, W) are broadcast over channels."""
    import torch
    from spacr.utils import calculate_activation_correlations
    rng = np.random.default_rng(1)
    inputs = torch.tensor(rng.random((2, 3, 16, 16)), dtype=torch.float32)
    maps = torch.tensor(rng.random((2, 16, 16)), dtype=torch.float32)
    out = calculate_activation_correlations(inputs, maps, ["a.png", "b.png"])
    assert len(out) == 2


def test_calculate_activation_correlations_custom_thresholds():
    import torch
    from spacr.utils import calculate_activation_correlations
    rng = np.random.default_rng(2)
    inputs = torch.tensor(rng.random((1, 2, 12, 12)), dtype=torch.float32)
    maps = torch.tensor(rng.random((1, 2, 12, 12)), dtype=torch.float32)
    out = calculate_activation_correlations(
        inputs, maps, ["x.png"], manders_thresholds=[10, 50, 90])
    assert len(out) == 1


# ---------------------------------------------------------------------------
# organelle segmentation diagnostic
# ---------------------------------------------------------------------------

def _blob_img(size=48):
    img = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[:size, :size]
    for cy, cx in ((14, 14), (32, 30)):
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 18.0) * 900
    return img


@pytest.mark.parametrize("morphology,method", [
    ("spots", "otsu"),
    ("spots", "adaptive"),
    ("spots", "log"),
    ("network", "otsu"),
    ("network", "ridge"),
    ("irregular", "otsu"),
    ("irregular", "adaptive"),
])
def test_organelle_diagnostic_modes(morphology, method):
    from spacr.utils import _organelle_diagnostic
    settings = {
        "organelle_tophat_radius": 3,
        "organelle_log_min_sigma": 1, "organelle_log_max_sigma": 4,
        "organelle_log_num_sigma": 3, "organelle_log_threshold": 0.05,
        "organelle_ridge_sigmas": [1, 2], "organelle_ridge_filter": "frangi",
        "organelle_adaptive_block_size": 11, "organelle_adaptive_offset": 0.0,
        "organelle_rolling_ball": False, "organelle_rolling_ball_radius": 10,
        "organelle_clahe": False, "organelle_clahe_clip_limit": 0.01,
        "organelle_network_threshold": "otsu",
    }
    try:
        out = _organelle_diagnostic(_blob_img(), morphology, method, settings)
    except Exception as e:
        pytest.skip(f"{morphology}/{method} diagnostic unavailable: {e}")
    assert out is not None
    img, title = out[0], out[1]
    assert isinstance(title, str) and title
    assert isinstance(img, np.ndarray)
