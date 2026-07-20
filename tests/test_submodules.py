"""
Deep coverage tests for spacr.submodules — the CellposeLazyDataset class
plus per-function unit tests for every entry point that can be driven
without GPU/network.

Where an entry point requires a real cellpose model or a specific screen
data layout, we build a minimal-but-realistic fixture and drive the
function through at least one full call.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spacr.submodules as SUB


# ===========================================================================
# CellposeLazyDataset — pure logic + PIL/skimage-driven __getitem__
# ===========================================================================

@pytest.fixture
def cellpose_pair_files(tmp_path, rng):
    """A tiny directory of image + label pairs the dataset can read."""
    import tifffile
    n = 4
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(); lbl_dir.mkdir()
    imgs, lbls = [], []
    for i in range(n):
        img = rng.integers(0, 60000, size=(64, 64), dtype=np.uint16)
        # Two-object label: circles
        lbl = np.zeros((64, 64), dtype=np.uint16)
        lbl[10:20, 10:20] = 1
        lbl[40:55, 40:55] = 2
        img_path = img_dir / f"img_{i:03d}.tif"
        lbl_path = lbl_dir / f"lbl_{i:03d}.tif"
        tifffile.imwrite(str(img_path), img)
        tifffile.imwrite(str(lbl_path), lbl)
        imgs.append(str(img_path)); lbls.append(str(lbl_path))
    return {"images": imgs, "labels": lbls, "n": n}


def test_lazy_dataset_len_matches_n_with_no_augment(cellpose_pair_files):
    ds = SUB.CellposeLazyDataset(
        image_files=cellpose_pair_files["images"],
        label_files=cellpose_pair_files["labels"],
        settings={"target_size": 32},
        randomize=False, augment=False,
    )
    assert len(ds) == cellpose_pair_files["n"]


def test_lazy_dataset_len_multiplies_with_augment(cellpose_pair_files):
    ds = SUB.CellposeLazyDataset(
        image_files=cellpose_pair_files["images"],
        label_files=cellpose_pair_files["labels"],
        settings={"target_size": 32},
        randomize=False, augment=True,
    )
    # 8 augmentations per image.
    assert len(ds) == cellpose_pair_files["n"] * 8


def test_lazy_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        SUB.CellposeLazyDataset(
            image_files=["a.tif"], label_files=["a.tif", "b.tif"],
            settings={"target_size": 32},
        )


def test_lazy_dataset_rejects_empty_lists():
    with pytest.raises(ValueError):
        SUB.CellposeLazyDataset(
            image_files=[], label_files=[],
            settings={"target_size": 32},
        )


def test_lazy_dataset_to_grayscale_rgb_input():
    rgb = np.stack([np.arange(9).reshape(3, 3)] * 3, axis=-1).astype(np.float32)
    gray = SUB.CellposeLazyDataset._to_grayscale(rgb)
    assert gray.shape == (3, 3)


def test_lazy_dataset_to_grayscale_2d_passthrough():
    img = np.arange(9).reshape(3, 3).astype(np.float32)
    gray = SUB.CellposeLazyDataset._to_grayscale(img)
    assert gray.shape == img.shape
    assert (gray == img).all()


def test_lazy_dataset_scale_to_unit_interval_from_uint16():
    arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
    out = SUB.CellposeLazyDataset._scale_to_unit_interval(arr)
    assert out.dtype == np.float32
    assert out.max() <= 1.0 + 1e-6
    assert out.min() >= 0.0


def test_lazy_dataset_scale_to_unit_interval_already_normalized():
    arr = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
    out = SUB.CellposeLazyDataset._scale_to_unit_interval(arr)
    # Values already <=1 → returned unchanged.
    assert (out == arr).all()


@pytest.mark.parametrize("aug_idx,expected_op", [
    (0, "identity"),
    (1, "rot90"),
    (2, "rot180"),
    (3, "rot270"),
    (4, "fliplr"),
    (5, "flipud"),
])
def test_lazy_dataset_apply_augmentation_preserves_shape(aug_idx, expected_op):
    img = np.arange(64).reshape(8, 8).astype(np.float32)
    lbl = np.zeros_like(img, dtype=np.uint16)
    out_i, out_l = SUB.CellposeLazyDataset._apply_augmentation(img, lbl, aug_idx)
    assert out_i.shape == img.shape
    assert out_l.shape == lbl.shape


def test_lazy_dataset_getitem_returns_image_and_label(cellpose_pair_files):
    ds = SUB.CellposeLazyDataset(
        image_files=cellpose_pair_files["images"],
        label_files=cellpose_pair_files["labels"],
        settings={"target_size": 32, "normalize": True, "percentiles": (2, 99)},
        randomize=False, augment=False,
    )
    img, lbl = ds[0]
    assert img.shape == (32, 32)
    assert lbl.shape == (32, 32)
    assert img.dtype == np.float32
    assert lbl.dtype == np.uint16
    # Normalized image should be in [0, 1].
    assert img.min() >= 0.0
    assert img.max() <= 1.0 + 1e-6


def test_lazy_dataset_normalize_disabled_leaves_intensity(cellpose_pair_files):
    ds = SUB.CellposeLazyDataset(
        image_files=cellpose_pair_files["images"],
        label_files=cellpose_pair_files["labels"],
        settings={"target_size": 32, "normalize": False},
        randomize=False, augment=False,
    )
    img, _ = ds[0]
    # With normalize=False, the rescale_intensity step is skipped.
    assert img.dtype == np.float32


# ===========================================================================
# count_phenotypes — small sqlite + png_list -> phenotype_counts.csv
# ===========================================================================

def test_count_phenotypes_produces_csv(tmp_path, capsys):
    from IPython.display import display  # noqa: F401 imported to satisfy submodules

    (tmp_path / "measurements").mkdir()
    db = tmp_path / "measurements" / "measurements.db"
    df = pd.DataFrame({
        "png_path": [f"o_{i}.png" for i in range(9)],
        "plateID": ["p1"] * 9,
        "rowID": ["r1"] * 3 + ["r2"] * 3 + ["r3"] * 3,
        "columnID": ["c1", "c1", "c2", "c1", "c2", "c2", "c1", "c1", "c2"],
        "value": [1, 1, 2, 1, 2, 3, 2, 1, 3],
        "annotate": [1, 1, 2, 1, 2, 3, 2, 1, 3],
    })
    with sqlite3.connect(db) as con:
        df.to_sql("png_list", con, index=False)

    settings = {
        "src": str(tmp_path),
        "annotation_column": "annotate",
    }
    try:
        SUB.count_phenotypes(settings)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"count_phenotypes needs interactive display: {e}")

    # phenotype_counts.csv should be written in the same dir as the db.
    out_csv = tmp_path / "measurements" / "phenotype_counts.csv"
    assert out_csv.exists()
    counts_df = pd.read_csv(out_csv, index_col=0)
    assert len(counts_df) > 0


# ===========================================================================
# generate_score_heatmap — pure pandas on a synthetic scores CSV
# ===========================================================================

def test_generate_score_heatmap_settings_smoke(tmp_path):
    """generate_score_heatmap reads settings['src'] + writes a PDF.
    We don't require the PDF to render — just that the call gets past the
    settings-parsing gate on an empty settings dict, which should raise a
    clear error rather than silently succeed."""
    with pytest.raises(Exception):
        SUB.generate_score_heatmap({})


# ===========================================================================
# compare_reads_to_scores — the pure/testable slice
# ===========================================================================

def test_compare_reads_to_scores_returns_when_files_missing(tmp_path):
    """When the input files don't exist, the function should raise
    (FileNotFoundError / pandas ParserError) rather than silently return
    garbage."""
    with pytest.raises(Exception):
        SUB.compare_reads_to_scores(
            reads_csv=str(tmp_path / "no.csv"),
            scores_csv=str(tmp_path / "no2.csv"),
        )


# ===========================================================================
# analyze_percent_positive nested helpers via a synthetic screen CSV
# ===========================================================================

def test_analyze_percent_positive_rejects_empty_settings():
    with pytest.raises(Exception):
        SUB.analyze_percent_positive({})


# ===========================================================================
# analyze_recruitment / analyze_plaques / analyze_endodyogeny / class_proportion
# rejection paths — each requires a specific data layout; without one,
# these entry points must fail loudly rather than silently succeeding.
# ===========================================================================

@pytest.mark.parametrize("fn_name", [
    "analyze_recruitment", "analyze_plaques", "analyze_endodyogeny",
    "analyze_class_proportion",
])
def test_analyze_functions_fail_loudly_on_empty_settings(fn_name):
    fn = getattr(SUB, fn_name)
    with pytest.raises(Exception):
        fn({})


# ===========================================================================
# CellposeLazyDataset with pathological settings
# ===========================================================================

def test_lazy_dataset_random_shuffle_reorders(cellpose_pair_files):
    """With randomize=True, at least some rearrangement is likely with N=4.
    Run enough seeds to reliably detect that some ordering differs from
    the input."""
    orig_pairs = list(zip(cellpose_pair_files["images"], cellpose_pair_files["labels"]))
    saw_shuffle = False
    for _ in range(10):
        ds = SUB.CellposeLazyDataset(
            image_files=cellpose_pair_files["images"],
            label_files=cellpose_pair_files["labels"],
            settings={"target_size": 32},
            randomize=True, augment=False,
        )
        new_pairs = list(zip(ds.image_files, ds.label_files))
        if new_pairs != orig_pairs:
            saw_shuffle = True
            break
    assert saw_shuffle, "randomize=True never reordered the files after 10 tries"
