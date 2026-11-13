"""Coverage-fill batch 2 for spacr.io pure helpers (no GPU)."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
import tifffile

import matplotlib
matplotlib.use("Agg")

from spacr import io as IO


# ---------------------------------------------------------------------------
# directory helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# object-count DB
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _results_to_csv
# ---------------------------------------------------------------------------

def test_results_to_csv(tmp_path):
    cells = pd.DataFrame({"a": [1, 2]})
    wells = pd.DataFrame({"b": [3, 4]})
    c, w = IO._results_to_csv(str(tmp_path), cells, wells)
    assert (tmp_path / "results" / "cells.csv").exists()
    assert (tmp_path / "results" / "wells.csv").exists()


# ---------------------------------------------------------------------------
# _copy_misclassified
# ---------------------------------------------------------------------------

def test_copy_misclassified(tmp_path):
    # build image files under a pc and an nc path
    pc_dir = tmp_path / "cls" / "pc"; pc_dir.mkdir(parents=True)
    nc_dir = tmp_path / "cls" / "nc"; nc_dir.mkdir(parents=True)
    pc_img = pc_dir / "a.png"; pc_img.write_bytes(b"x")
    nc_img = nc_dir / "b.png"; nc_img.write_bytes(b"y")
    df = pd.DataFrame({
        "filename": [str(pc_img), str(nc_img)],
        "true_label": [1, 0],
        "predicted_label": [0, 1],   # both misclassified
    })
    IO._copy_misclassified(df)
    assert list((tmp_path / "cls" / "misclassified").rglob("*.png"))


# ---------------------------------------------------------------------------
# apply_augmentation / process_instruction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["rotate90", "rotate180", "rotate270",
                                    "flip_h", "flip_v", "none"])
def test_apply_augmentation(method):
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = IO.apply_augmentation(img, method)
    assert out.shape == (4, 4)


def test_process_instruction(tmp_path):
    img = (np.random.default_rng(0).random((8, 8)) * 255).astype(np.uint8)
    msk = np.zeros((8, 8), dtype=np.uint16); msk[1:4, 1:4] = 1
    src_img = tmp_path / "img.tif"; src_msk = tmp_path / "msk.tif"
    tifffile.imwrite(str(src_img), img); tifffile.imwrite(str(src_msk), msk)
    entry = {
        "src_img": str(src_img), "src_msk": str(src_msk),
        "dst_img": str(tmp_path / "out_img.tif"),
        "dst_msk": str(tmp_path / "out_msk.tif"),
        "augment": "rotate90",
    }
    assert IO.process_instruction(entry) == 1
    assert (tmp_path / "out_img.tif").exists()


# ---------------------------------------------------------------------------
# generate_cellpose_train_test
# ---------------------------------------------------------------------------

def test_generate_cellpose_train_test(tmp_path):
    src = tmp_path / "data"
    (src / "masks").mkdir(parents=True)
    for i in range(10):
        img = (np.random.default_rng(i).random((16, 16)) * 255).astype(np.uint8)
        msk = np.zeros((16, 16), dtype=np.uint16); msk[2:5, 2:5] = 1
        tifffile.imwrite(str(src / f"img{i}.tif"), img)
        tifffile.imwrite(str(src / "masks" / f"img{i}.tif"), msk)
    IO.generate_cellpose_train_test(str(src), test_split=0.2)
    # `assert ... or True` under a swallowed skip could not fail either way,
    # and it looked in the wrong place: the split folders are siblings of
    # `src`, not children (dirname(src)).
    train, test = tmp_path / "train", tmp_path / "test"
    assert sorted(p.name for p in train.glob("*.tif")) != []
    assert len(list(train.glob("*.tif"))) == 8
    assert len(list(test.glob("*.tif"))) == 2
    # every image is copied together with its mask
    assert len(list((train / "masks").glob("*.tif"))) == 8
    assert len(list((test / "masks").glob("*.tif"))) == 2
    # and the two sets are disjoint
    assert not ({p.name for p in train.glob("*.tif")}
                & {p.name for p in test.glob("*.tif")})


# ---------------------------------------------------------------------------
# _check_masks
# ---------------------------------------------------------------------------

def test_check_masks(tmp_path):
    # b already exists as a whole mask → filtered out. A zero-byte b.npy
    # would be regenerated: masks are validated before reuse (2026-09-19).
    np.save(tmp_path / "b.npy", np.zeros((4, 4), np.uint16))
    batch = [np.zeros((4, 4)), np.ones((4, 4)), np.full((4, 4), 2)]
    names = ["a.npy", "b.npy", "c.npy"]
    fb, fn = IO._check_masks(batch, names, str(tmp_path))
    assert fn == ["a.npy", "c.npy"] and len(fb) == 2


# ---------------------------------------------------------------------------
# model-stats CSV + plotting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# normalization helpers
# ---------------------------------------------------------------------------


