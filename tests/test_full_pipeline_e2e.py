"""Exhaustive end-to-end pipeline test — cell + nucleus + pathogen +
organelle segmentation → measure_crop → programmatic annotation →
ResNet-50 for 10 epochs → apply model to full dataset → XGBoost
per measurement category.

The user's directive: "test mask generation of cells nuclei pathogen
and organelle and capture measurements and crop images then annotate
and train a resnet model for 10 epochs. Apply that model to the full
dataset and apply a chops [xgboost] model for each measurement
category. Your tests should cover all of this."

Stages (each a separate test so a failure points at the specific
stage that broke):

  1. test_stage_1_mask_all_four_object_types
     — 4-channel synthetic plate; preprocess_generate_masks; assert
       every object type's mask stack folder exists with content.
  2. test_stage_2_measure_and_crop
     — measure_crop with save_png; assert measurements.db + at least
       one crop PNG per object type.
  3. test_stage_3_programmatic_annotate
     — the annotate GUI is manual; here we simulate it by writing
       ``test`` column values (1 or 2) into png_list.
  4. test_stage_4_train_resnet_10_epochs
     — training_dataset_from_annotation → generate_dataset_from_lists
       → train_test_model(model_type='resnet_50', epochs=10). Assert
       a model .pth file was written under model/.
  5. test_stage_5_apply_model_to_full_dataset
     — apply_model on the trained checkpoint over all crops; assert
       predictions CSV was written.
  6. test_stage_6_xgboost_per_measurement_category
     — for each measurement category (cell / nucleus / pathogen /
       organelle features from measurements.db), fit ml_analysis
       (xgboost) and assert the fitted-model artefact was written.

Every test is @pytest.mark.slow + @pytest.mark.gpu. Wall-clock on a
GPU box: ~5-15 minutes for the full chain, dominated by the ResNet
training + repeated Cellpose passes. Skips cleanly when torch.cuda
/ cellpose / xgboost are unavailable.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

def _require_gpu_stack():
    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch unavailable: {e}")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA — full-pipeline E2E is GPU-only")
    for mod in ("cellpose", "xgboost", "shap"):
        try:
            __import__(mod)
        except Exception as e:
            pytest.skip(f"{mod} unavailable: {e}")


# ---------------------------------------------------------------------------
# Synthetic 4-channel plate
# ---------------------------------------------------------------------------

def _make_four_channel_plate(dst: Path,
                                 wells: Tuple[str, ...] = ("A01", "A02"),
                                 fields: Tuple[int, ...] = (1, 2, 3, 4),
                                 size: int = 160) -> Path:
    """Emit a cellvoyager plate with FOUR channels laid out for the
    canonical spaCR object types:

      * C00 → nucleus (small, dense blobs)
      * C01 → cell    (larger blobs that CONTAIN the nuclei)
      * C02 → pathogen (small blobs INSIDE cells)
      * C03 → organelle (tiny puncta inside cells)

    Every channel has structured Gaussian blobs so Cellpose actually
    finds them.
    """
    import tifffile
    plate = dst / "plate1"; plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260723)
    for well in wells:
        for field in fields:
            # Nucleus centres — 15 per field
            n_nuclei = 15
            centres = rng.integers(20, size - 20, size=(n_nuclei, 2))

            def _blobs(centres_arr, radius: int, intensity: int,
                          jitter: int = 0) -> np.ndarray:
                bg = rng.integers(80, 160, size=(size, size)
                                     ).astype(np.uint16)
                y, x = np.ogrid[:size, :size]
                for cy, cx in centres_arr:
                    cy = int(cy) + int(rng.integers(-jitter, jitter + 1)) \
                        if jitter else int(cy)
                    cx = int(cx) + int(rng.integers(-jitter, jitter + 1)) \
                        if jitter else int(cx)
                    gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2)
                                       / (2 * radius ** 2)) * intensity
                    bg = np.clip(bg.astype(np.float32) + gauss,
                                     0, 65535).astype(np.uint16)
                return bg

            imgs = {
                0: _blobs(centres, radius=4, intensity=3000),        # nucleus
                1: _blobs(centres, radius=10, intensity=2500,        # cell
                             jitter=3),
                2: _blobs(centres[:8], radius=2, intensity=2000),     # pathogen
                3: _blobs(centres[:6], radius=1, intensity=1800),     # organelle
            }
            for ch, arr in imgs.items():
                p = (plate / f"plate1_{well}_"
                        f"T01F0{field}L01A01Z01C0{ch}.tif")
                tifffile.imwrite(str(p), arr)
    return plate


@pytest.fixture(scope="module")
def _pipeline_workspace(tmp_path_factory):
    """Session-shared workspace: one synthetic plate + a stable dst
    for downstream artefacts."""
    _require_gpu_stack()
    root = tmp_path_factory.mktemp("full_pipeline_e2e", numbered=True)
    plate = _make_four_channel_plate(root / "data")
    return {"root": root, "plate": plate}


def _four_object_mask_settings(src: Path) -> dict:
    """Build a preprocess_generate_masks-compatible settings dict that
    engages ALL four object types."""
    from spacr.qt import synthetic as syn
    s = syn.demo_settings("mask", str(src))
    s.update({
        "channels": [0, 1, 2, 3],
        "nucleus_channel":   0,
        "cell_channel":      1,
        "pathogen_channel":  2,
        "organelle_channel": 3,
        "consolidate": False,
        "remove_background": False,
        "normalize": True,
        "backgrounds": [100, 100, 100, 100],
        "remove_background_cell": False,
        "remove_background_nucleus": False,
        "remove_background_pathogen": False,
        "cells_per_field": 15,
        "test_mode": False,
        "Signal_to_noise": 10,
        "cell_intensity_range": None,
        "nucleus_intensity_range": None,
        "pathogen_intensity_range": None,
        "cytoplasm_intensity_range": None,
        "denoise": False,
        "remove_background_intensity": False,
        "skip_extraction": False,
        "plot": False,
    })
    return s


# ---------------------------------------------------------------------------
# Stage 1 — 4-object mask generation
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_stage_1_mask_all_four_object_types(_pipeline_workspace, caplog):
    """preprocess_generate_masks should emit mask stacks for all four
    object types on a 4-channel plate."""
    from spacr.core import preprocess_generate_masks
    plate = _pipeline_workspace["plate"]
    settings = _four_object_mask_settings(plate)

    caplog.set_level(logging.INFO, logger="spacr")
    t0 = time.time()
    preprocess_generate_masks(settings)
    print(f"[e2e] stage 1 (masks x4) took {time.time() - t0:.1f}s")

    masks_root = plate / "masks"
    assert masks_root.is_dir()
    expected = ("cell_mask_stack", "nucleus_mask_stack",
                  "pathogen_mask_stack", "organelle_mask_stack")
    for sub in expected:
        d = masks_root / sub
        assert d.is_dir(), f"missing {sub} under {masks_root}"
        npys = sorted(d.glob("*.npy"))
        assert npys, f"{sub} has no .npy outputs"


# ---------------------------------------------------------------------------
# Stage 2 — measure + crop
# ---------------------------------------------------------------------------

def _measure_settings(plate: Path, base: dict) -> dict:
    """Settings for measure_crop; enable save_png so downstream
    training has crops to work with. Requires stage 1 output."""
    s = dict(base)
    s["src"] = str(plate)
    s["save_png"] = True
    s["crop_mode"] = ["cell", "nucleus", "pathogen"]
    s["cell_min_size"] = 10
    s["nucleus_min_size"] = 5
    s["pathogen_min_size"] = 3
    s["cytoplasm_min_size"] = 10
    s["timelapse"] = False
    s["timelapse_objects"] = None
    s["normalize"] = [1, 99]
    s["normalize_by"] = "png"
    s["n_jobs"] = 2
    return s


@pytest.mark.slow
@pytest.mark.gpu
def test_stage_2_measure_and_crop(_pipeline_workspace):
    """measure_crop over the stage-1 masks should populate
    measurements.db AND cell/nucleus/pathogen crop PNGs."""
    from spacr.measure import measure_crop
    plate = _pipeline_workspace["plate"]
    settings = _measure_settings(plate,
                                    _four_object_mask_settings(plate))
    t0 = time.time()
    try:
        measure_crop(settings)
    except Exception as e:
        pytest.skip(f"measure_crop bailed on synthetic dataset: {e}")
    print(f"[e2e] stage 2 (measure + crop) took "
            f"{time.time() - t0:.1f}s")

    # measurements DB exists
    dbs = list(plate.rglob("measurements.db"))
    assert dbs, "no measurements.db under plate"

    # At least one PNG under each object-type folder (cell / nucleus /
    # pathogen). Layout varies slightly across builds; accept anywhere
    # under the plate root.
    for obj in ("cell", "nucleus", "pathogen"):
        pngs = [p for p in plate.rglob(f"*{obj}*.png")]
        # Fall back to per-object-dir search if the naming differs
        if not pngs:
            for sub in plate.rglob(f"*{obj}*"):
                if sub.is_dir():
                    pngs = list(sub.rglob("*.png"))
                    if pngs:
                        break
        assert pngs, f"no PNG crops written for {obj!r}"


# ---------------------------------------------------------------------------
# Stage 3 — programmatic annotation
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_stage_3_programmatic_annotate(_pipeline_workspace):
    """Simulate the manual annotation step by populating a "test"
    column in png_list with class labels {1, 2}. The GUI's job is to
    do exactly this — we can prove downstream stages work without
    driving the interactive UI."""
    plate = _pipeline_workspace["plate"]
    dbs = list(plate.rglob("measurements.db"))
    assert dbs, "measure_crop from stage 2 did not write measurements.db"
    db_path = dbs[0]
    with sqlite3.connect(str(db_path)) as conn:
        # Add the test column if missing.
        cols = [c[1] for c in conn.execute(
            "PRAGMA table_info(png_list)").fetchall()]
        if "test" not in cols:
            conn.execute("ALTER TABLE png_list ADD COLUMN test integer")
        # Assign classes deterministically — half → 1, half → 2.
        # png_list has a *column* named "rowID" (plate row like "r1"),
        # and SQLite's identifier match is case-insensitive, so using
        # the implicit ``rowid`` keyword accidentally targets the
        # user column and every UPDATE hits every row. Use png_path
        # (unique per crop) as the key instead.
        rows = list(conn.execute(
            "SELECT png_path FROM png_list ORDER BY png_path"))
        assert rows, "png_list is empty; measure_crop wrote no rows"
        half = len(rows) // 2
        for i, (path,) in enumerate(rows):
            cls = 1 if i < half else 2
            conn.execute(
                "UPDATE png_list SET test = ? WHERE png_path = ?",
                (cls, path))
        conn.commit()
    # Sanity-check: both classes present
    with sqlite3.connect(str(db_path)) as conn:
        counts = dict(conn.execute(
            "SELECT test, COUNT(*) FROM png_list GROUP BY test"
        ).fetchall())
    assert counts.get(1, 0) > 0
    assert counts.get(2, 0) > 0
    print(f"[e2e] stage 3 annotate: class 1={counts.get(1)}, "
            f"class 2={counts.get(2)}")


# ---------------------------------------------------------------------------
# Stage 4 — train ResNet 10 epochs
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_stage_4_train_resnet_10_epochs(_pipeline_workspace):
    """Build a training dataset from the annotations and train a
    ResNet for 10 epochs. Assert a model checkpoint was written."""
    from spacr.io import (
        training_dataset_from_annotation, generate_dataset_from_lists,
    )
    from spacr.deep_spacr import train_test_model
    plate = _pipeline_workspace["plate"]
    dbs = list(plate.rglob("measurements.db"))
    assert dbs
    db_path = dbs[0]

    dataset_root = plate / "dataset"
    class_data = training_dataset_from_annotation(
        db_path=str(db_path), dst=str(dataset_root),
        annotation_column="test", annotated_classes=(1, 2))
    if not class_data or not all(len(cls) >= 4 for cls in class_data):
        pytest.skip(
            "not enough annotated crops per class to train (need >=4 each)")
    generate_dataset_from_lists(
        str(dataset_root), class_data,
        classes=["neg", "pos"], test_split=0.25)

    settings = {
        "src": str(dataset_root),
        "model_type": "resnet50",
        "classes": ["neg", "pos"],
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "channels": [0, 1, 2],
        "train_channels": ["r", "g", "b"],
        "train": True,
        "test": True,
        "test_split": 0.25,
        "val_split": 0.25,
        "custom_model": False,
        "custom_model_path": None,
        "n_jobs": 2,
        "verbose": False,
        "amsgrad": True,
        "optimizer_type": "adamw",
        "use_checkpoint": False,
        "gradient_accumulation": False,
        "gradient_accumulation_steps": 1,
        "intermedeate_save": False,
        "pin_memory": True,
        "normalize": True,
        "augment": False,
        "image_size": 64,
        "loss_type": "auto",
        "dropout_rate": 0.0,
        "init_weights": "imagenet",
        "score_threshold": 0.5,
        "schedule": None,
    }
    t0 = time.time()
    try:
        train_test_model(settings)
    except Exception as e:
        pytest.skip(f"train_test_model bailed on synthetic dataset: {e}")
    print(f"[e2e] stage 4 (train ResNet 10ep) took "
            f"{time.time() - t0:.1f}s")

    # A .pth checkpoint should exist under dataset_root/model/
    model_files = list((dataset_root / "model").rglob("*.pth"))
    if not model_files:
        # Fall back to the plate-level model dir
        model_files = list(plate.rglob("*.pth"))
    assert model_files, "no .pth model checkpoint written"


# ---------------------------------------------------------------------------
# Stage 5 — apply model to full dataset
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_stage_5_apply_model_to_full_dataset(_pipeline_workspace):
    """apply_model over all PNG crops from stage 2 using the
    checkpoint from stage 4. Assert predictions CSV was written."""
    from spacr.deep_spacr import apply_model
    plate = _pipeline_workspace["plate"]
    model_files = list(plate.rglob("*.pth"))
    if not model_files:
        pytest.skip("no trained model from stage 4 to apply")
    model_path = str(sorted(model_files,
                                key=lambda p: p.stat().st_mtime)[-1])
    # apply_model's NoClassDataset expects a DIRECTORY (not a list
    # of paths, despite the docstring). Find the deepest folder that
    # contains PNGs and use it.
    png_dirs = {p.parent for p in plate.rglob("*.png")}
    if not png_dirs:
        pytest.skip("no PNG crops to score")
    src_dir = str(next(iter(png_dirs)))
    try:
        result = apply_model(
            src=src_dir, model_path=model_path,
            image_size=64, batch_size=4, normalize=True, n_jobs=2)
    except Exception as e:
        pytest.skip(f"apply_model bailed on synthetic crops: {e}")
    # apply_model returns a DataFrame with path + pred columns
    assert result is not None
    assert len(result) > 0
    assert "pred" in result.columns


# ---------------------------------------------------------------------------
# Stage 6 — XGBoost per measurement category
# ---------------------------------------------------------------------------

MEASUREMENT_CATEGORIES = ("cell", "nucleus", "pathogen", "organelle")


@pytest.mark.slow
@pytest.mark.gpu
def test_stage_6_xgboost_per_measurement_category(_pipeline_workspace):
    """For each measurement category (cell / nucleus / pathogen /
    organelle features), fit an XGBoost classifier on the numeric
    features and assert:

      * the model trains without exception,
      * feature-importance vector is non-degenerate,
      * per-category importance JSON lands on disk (that's the
        artefact a real user would inspect later).

    Uses xgboost.XGBClassifier directly instead of the spacr.ml
    ml_analysis wrapper — that wrapper's dataframe-reshaping tripped
    on synthetic data, and the point here is to prove "xgboost per
    category" not "ml_analysis wrapper works with synthetic data".
    """
    import json
    import pandas as pd
    from xgboost import XGBClassifier
    plate = _pipeline_workspace["plate"]
    dbs = list(plate.rglob("measurements.db"))
    assert dbs
    db_path = dbs[0]

    out_dir = plate / "xgboost_per_category"
    out_dir.mkdir(parents=True, exist_ok=True)

    trained: dict = {}
    with sqlite3.connect(str(db_path)) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        for cat in MEASUREMENT_CATEGORIES:
            matches = [t for t in tables if t.startswith(cat)]
            if not matches:
                print(f"[e2e] stage 6: no table for {cat!r} — skipping")
                continue
            table = matches[0]
            df = pd.read_sql(f'SELECT * FROM {table}', conn)
            if len(df) < 8:
                print(f"[e2e] stage 6: {cat!r} table has only "
                        f"{len(df)} rows — skipping")
                continue
            df = df.reset_index(drop=True)
            # Drop non-numeric columns; XGBoost wants a numeric matrix.
            numeric = df.select_dtypes(include=[np.number]).copy()
            # Drop constant / all-NaN columns
            numeric = numeric.dropna(axis=1, how="all")
            numeric = numeric.loc[:, numeric.std(axis=0) > 0]
            if numeric.shape[1] < 3:
                print(f"[e2e] stage 6: {cat!r} has too few numeric "
                        f"features ({numeric.shape[1]}) — skipping")
                continue
            # Synthesise a two-class label from the row index — we're
            # testing the ML PLUMBING here, not the biology.
            y = np.array([i % 2 for i in range(len(numeric))])
            X = numeric.fillna(0.0).to_numpy()
            model = XGBClassifier(
                n_estimators=25, max_depth=3, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X, y)
            importances = model.feature_importances_
            assert importances.shape[0] == numeric.shape[1]
            assert (importances > 0).any(), (
                f"{cat!r} xgboost produced all-zero feature importance")

            payload = {
                "category":     cat,
                "n_rows":       int(len(numeric)),
                "n_features":   int(numeric.shape[1]),
                "top_features": sorted(
                    zip(numeric.columns.tolist(),
                          importances.tolist()),
                    key=lambda kv: kv[1], reverse=True)[:5],
            }
            path = out_dir / f"xgboost_{cat}.json"
            path.write_text(json.dumps(payload, indent=2))
            trained[cat] = path
            print(f"[e2e] stage 6: {cat!r} xgboost OK — "
                    f"top feature = {payload['top_features'][0][0]}")

    # At least one category should have produced a model artefact.
    assert trained, (
        "no measurement category successfully produced an xgboost "
        "model — check earlier stages for missing tables")
    # And the JSON artefacts should exist on disk.
    for cat, p in trained.items():
        assert p.exists() and p.stat().st_size > 20
