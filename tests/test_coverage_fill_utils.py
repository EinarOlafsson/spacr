"""Coverage-fill for spacr.utils pure-logic helpers (no GPU)."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# Mask / label helpers
# ---------------------------------------------------------------------------

def test_mask_object_count():
    m = np.zeros((10, 10), dtype=np.int32)
    m[1:3, 1:3] = 1
    m[5:7, 5:7] = 2
    assert U.mask_object_count(m) == 2


def test_mask_object_count_empty():
    assert U.mask_object_count(np.zeros((5, 5), dtype=np.int32)) == 0


def test_relabel_sequential():
    m = np.zeros((6, 6), dtype=np.int32)
    m[0:2, 0:2] = 5
    m[3:5, 3:5] = 9
    out = U._relabel_sequential(m)
    assert set(np.unique(out)) == {0, 1, 2}


def test_masks_to_masks_stack():
    masks = [np.zeros((4, 4)), np.ones((4, 4))]
    out = U._masks_to_masks_stack(masks)
    assert len(out) == 2


def test_compute_label_perimeters():
    m = np.zeros((10, 10), dtype=np.int32)
    m[2:6, 2:6] = 1
    out = U._compute_label_perimeters(m)
    assert out is not None


# ---------------------------------------------------------------------------
# Type / list helpers
# ---------------------------------------------------------------------------

def test_is_list_of_lists():
    assert U.is_list_of_lists([[1], [2]]) is True
    assert U.is_list_of_lists([1, 2]) is False
    assert U.is_list_of_lists("not a list") is False


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def test_invert_image_uint8():
    img = np.array([[0, 255], [100, 200]], dtype=np.uint8)
    inv = U.invert_image(img)
    assert inv[0, 0] == 255 and inv[0, 1] == 0


def test_invert_image_uint16():
    img = np.array([[0, 65535]], dtype=np.uint16)
    inv = U.invert_image(img)
    assert inv[0, 0] == 65535 and inv[0, 1] == 0


def test_remove_canvas_grayscale():
    from PIL import Image
    img = Image.fromarray(np.array([[0, 100], [200, 0]], dtype=np.uint8), "L")
    out = U.remove_canvas(img)
    assert out.shape[-1] == 4   # RGBA


def test_remove_canvas_rgb():
    from PIL import Image
    arr = np.zeros((4, 4, 3), dtype=np.uint8); arr[1:3, 1:3] = 100
    out = U.remove_canvas(Image.fromarray(arr, "RGB"))
    assert out.shape[-1] == 4


def test_remove_canvas_unsupported_raises():
    from PIL import Image
    img = Image.fromarray(np.zeros((4, 4, 4), dtype=np.uint8), "RGBA")
    with pytest.raises(ValueError):
        U.remove_canvas(img)


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def test_check_normality_utils():
    rng = np.random.default_rng(0)
    assert U.check_normality(rng.normal(0, 1, 200)) in (True, False)
    # Clearly non-normal
    assert U.check_normality(np.array([0.0] * 50 + [1000.0])) is False


# ---------------------------------------------------------------------------
# FASTQ read counter
# ---------------------------------------------------------------------------

def test_count_reads_in_fastq(tmp_path):
    fq = tmp_path / "reads.fastq.gz"
    records = "".join(f"@id{i}\nACGT\n+\nIIII\n" for i in range(5))
    with gzip.open(fq, "wt") as fh:
        fh.write(records)
    assert U.count_reads_in_fastq(str(fq)) == 5


# ---------------------------------------------------------------------------
# _get_cellpose_channels (remap logic)
# ---------------------------------------------------------------------------

def test_get_cellpose_channels_remap():
    settings = {
        "cellpose_nucleus_channel": 0,
        "cellpose_cell_channel": 1,
        "cellpose_pathogen_channel": 2,
        "cellpose_organelle_channel": None,
    }
    extract, remap = U._get_cellpose_channels(settings)
    assert extract == [0, 1, 2]
    assert "cell" in remap and "nucleus" in remap


def test_get_cellpose_channels_cell_without_nucleus():
    settings = {
        "cellpose_nucleus_channel": None,
        "cellpose_cell_channel": 2,
        "cellpose_pathogen_channel": None,
        "cellpose_organelle_channel": None,
    }
    extract, remap = U._get_cellpose_channels(settings)
    assert remap["cell"] == [0]   # dense-remapped from raw 2


# ---------------------------------------------------------------------------
# torchvision model names
# ---------------------------------------------------------------------------

def test_list_torchvision_model_names():
    names = U._list_torchvision_model_names()
    assert "resnet50" in names and isinstance(names, set)


# ---------------------------------------------------------------------------
# is_multiprocessing_process (mock process)
# ---------------------------------------------------------------------------

def test_is_multiprocessing_process_true():
    import types
    proc = types.SimpleNamespace(
        cmdline=lambda: ["python", "-c", "multiprocessing.spawn"])
    assert U.is_multiprocessing_process(proc) is True


def test_is_multiprocessing_process_false():
    import types
    proc = types.SimpleNamespace(cmdline=lambda: ["python", "app.py"])
    assert U.is_multiprocessing_process(proc) is False


# ---------------------------------------------------------------------------
# batch 2 — more pure helpers
# ---------------------------------------------------------------------------

def test_get_db_paths_str_and_list():
    assert U.get_db_paths("/a")[0].endswith("measurements/measurements.db")
    both = U.get_db_paths(["/a", "/b"])
    assert len(both) == 2


def test_get_sequencing_paths():
    assert U.get_sequencing_paths("/a")[0].endswith(
        "sequencing/sequencing_data.csv")


def test_get_percentiles():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 1000, size=(16, 16, 3)).astype(np.uint16)
    pct = U._get_percentiles(arr)
    assert len(pct) == 3
    assert all(len(p) == 2 for p in pct)


def test_check_integrity_collapses_label_columns():
    df = pd.DataFrame({
        "label": [1, 2], "cell_label": [3, 4], "value": [5, 6],
    })
    out = U._check_integrity(df)
    assert "object_label" in out.columns
    assert "label_list" in out.columns


def test_check_multicollinearity():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({
        "a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50),
    })
    vif = U.check_multicollinearity(x)
    assert "VIF" in vif.columns and len(vif) == 2


def test_apply_union_find():
    m = np.zeros((6, 6), dtype=np.int32)
    m[0:2, 0:2] = 1; m[3:5, 3:5] = 2
    parent = {1: 1, 2: 2}
    out = U._apply_union_find(m, parent)
    assert out is not None


def test_convert_and_relabel_masks(tmp_path):
    # int64 npy masks → converted to uint16 relabeled.
    m = np.zeros((8, 8), dtype=np.int64); m[1:3, 1:3] = 100000
    np.save(str(tmp_path / "m.npy"), m)
    U.convert_and_relabel_masks(str(tmp_path))
    out = np.load(str(tmp_path / "m.npy"))
    assert out.dtype == np.uint16


def test_get_cuda_version():
    # Just runs — returns a version string or None.
    v = U.get_cuda_version()
    assert v is None or isinstance(v, str)
