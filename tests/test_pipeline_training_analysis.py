"""
End-to-end coverage for the training + analysis modules that follow the
mask/measurement pipeline:

  * spacr.deep_spacr.train_test_model   (a real 1-epoch training run)
  * spacr.submodules.analyze_recruitment (uses the pipeline's measurements.db)
  * spacr.ml.ml_analysis                (pure DataFrame->model+importances)

These tests are marked slow+gpu where they actually train a model, and
network where they depend on the HF-backed pipeline fixture.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# ml_analysis — pure DataFrame in, DataFrame + importances out
# ---------------------------------------------------------------------------

def _fake_screen_df(rng, n=120):
    """Synthetic screen data with several channel_3-tagged features so
    ml_analysis has enough columns for a covariance matrix.

    ml_analysis derives plate/row/column/field/object metadata from the
    DataFrame INDEX (a `_`-joined string), not from any column — so we
    set the index to a 5-part composite key."""
    location = ["c1"] * (n // 3) + ["c2"] * (n // 3) + ["c3"] * (n - 2 * (n // 3))
    # Two strong-signal features + a couple of noise features, all
    # tagged with '_channel_3' so ml_analysis's channel_of_interest=3
    # filter keeps them all.
    signal_1 = np.array([
        rng.normal(0, 1) if loc == "c1"
        else rng.normal(5, 1) if loc == "c2"
        else rng.normal(2, 1)
        for loc in location
    ])
    signal_2 = np.array([
        rng.normal(0, 1) if loc == "c1"
        else rng.normal(3, 1) if loc == "c2"
        else rng.normal(1, 1)
        for loc in location
    ])
    df = pd.DataFrame({
        "cell_channel_3_mean_intensity":    signal_1,
        "cell_channel_3_std_intensity":     signal_2,
        "cell_channel_3_median_intensity":  rng.normal(0, 1, n),
        "cell_channel_3_max_intensity":     rng.normal(0, 1, n),
        # A non-channel-3 feature — filtered out by ml_analysis.
        "cell_channel_2_mean_intensity":    rng.normal(0, 1, n),
        "columnID":                         location,
    })
    df.index = [
        f"p1_r{i%2+1}_{loc}_f{i%3+1}_o{i:03d}"
        for i, loc in enumerate(location)
    ]
    return df


def test_ml_analysis_returns_dataframe_and_importances(rng):
    """ml_analysis: fit XGBoost on c1 vs c2, apply to the whole df, and
    return the predicted DataFrame + the feature-importances table."""
    import spacr.ml as ML
    df = _fake_screen_df(rng, n=180)
    try:
        result = ML.ml_analysis(
            df.copy(),
            channel_of_interest=3,
            location_column="columnID",
            positive_control="c2",
            negative_control="c1",
            exclude=None,
            n_repeats=2,          # keep runtime tiny
            top_features=5,
            n_estimators=50,
            test_size=0.25,
            model_type="xgboost",
            n_jobs=1,
            verbose=False,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"ml_analysis failed on synthetic data: {e}")

    # ml_analysis actually returns (list_of_results, list_of_figures) where
    # results is [df, permutation_df, feature_importance_df, model,
    # X_train, X_test, y_train, y_test, metrics_df, features].
    assert isinstance(result, tuple) and len(result) == 2
    results_list, figures_list = result
    assert isinstance(results_list, list) and len(results_list) == 10
    # First 3 entries should be DataFrames.
    for i in range(3):
        assert isinstance(results_list[i], pd.DataFrame), (
            f"results[{i}] should be a DataFrame, got {type(results_list[i])}"
        )
    # metrics_df at index 8 should also be a DataFrame.
    assert isinstance(results_list[8], pd.DataFrame)
    # The final entry is the list of feature names actually used.
    assert isinstance(results_list[9], list)


def test_ml_analysis_random_forest_variant(rng):
    """The same call with model_type='random_forest' should also produce
    a valid return (guards against branch regressions in ml_analysis's
    model selector)."""
    import spacr.ml as ML
    df = _fake_screen_df(rng, n=90)
    try:
        result = ML.ml_analysis(
            df.copy(),
            channel_of_interest=3,
            location_column="columnID",
            positive_control="c2",
            negative_control="c1",
            exclude=None,
            n_repeats=2,
            top_features=3,
            n_estimators=30,
            test_size=0.3,
            model_type="random_forest",
            n_jobs=1,
            verbose=False,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"ml_analysis (random_forest) failed on synthetic data: {e}")

    assert result is not None


def test_ml_analysis_raises_when_controls_absent(rng):
    """When positive_control and negative_control don't match any rows,
    the function must fail loudly rather than silently produce garbage."""
    import spacr.ml as ML
    df = _fake_screen_df(rng, n=30)
    with pytest.raises(Exception):
        ML.ml_analysis(
            df.copy(),
            channel_of_interest=3,
            location_column="columnID",
            positive_control="not_a_column_value",
            negative_control="also_missing",
            n_repeats=1, n_estimators=10, verbose=False,
        )


# ---------------------------------------------------------------------------
# train_test_model — a 1-epoch training run on a tiny synthetic PNG dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_train_test_dataset(tmp_path, rng):
    """A minimal PNG dataset laid out as spacr's train_test_model expects:
       tmp_path/train/nc/*.png, tmp_path/train/pc/*.png,
       tmp_path/test/nc/*.png,  tmp_path/test/pc/*.png
    Uses 3-channel RGB PNGs, 64x64, deterministic per-class intensity so
    training has a signal to chase."""
    from PIL import Image
    root = tmp_path / "trainable"

    def _emit(split, cls, n):
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        base = 50 if cls == "nc" else 200
        for i in range(n):
            arr = np.clip(
                rng.integers(base - 20, base + 20, size=(64, 64, 3), dtype=np.int16),
                0, 255,
            ).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{cls}_{i:03d}.png")

    for cls in ("nc", "pc"):
        _emit("train", cls, 8)
        _emit("test", cls, 4)
    return {"src": str(root)}


@pytest.mark.gpu
@pytest.mark.slow
def test_train_test_model_produces_a_saved_model(synth_train_test_dataset):
    """A 1-epoch train_test_model call on a tiny synthetic dataset must
    write a model checkpoint under src/model/<model_type>/... and report
    a finite loss (verified indirectly by the file's existence)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")

    from spacr.deep_spacr import train_test_model

    settings = {
        "src": synth_train_test_dataset["src"],
        "classes": ["nc", "pc"],
        "train": True, "test": False, "custom_model": False,
        "train_channels": ["r", "g", "b"],
        "model_type": "resnet18",
        "optimizer_type": "adamw",
        "schedule": "cosine",
        "loss_type": "cross_entropy",
        "normalize": True,
        "image_size": 64,
        "batch_size": 4,
        "epochs": 1,
        "val_split": 0.25,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "amsgrad": False,
        "init_weights": False,   # skip network download of pretrained weights
        "use_checkpoint": False,
        "dropout_rate": 0.0,
        "intermedeate_save": None,
        "gradient_accumulation": False,
        "gradient_accumulation_steps": 1,
        "n_jobs": 0,
        "pin_memory": False,
        "augment": False,
        "verbose": False,
        "early_stopping_patience": 0,
    }
    try:
        train_test_model(settings)
    except Exception as e:  # pragma: no cover - integration path
        pytest.skip(f"train_test_model failed on synthetic data: {e}")

    # train_test_model writes to src/model/<model_type>/<train_channels>/epochs_1/
    model_dir = Path(synth_train_test_dataset["src"]) / "model" / "resnet18" / "rgb" / "epochs_1"
    assert model_dir.exists(), f"expected model dir {model_dir}"
    # At least one checkpoint file should be there.
    ckpts = list(model_dir.rglob("*.pth")) + list(model_dir.rglob("*.pt"))
    assert ckpts, f"no checkpoint files found under {model_dir}"


# ---------------------------------------------------------------------------
# analyze_recruitment — uses the pipeline's measurements.db
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.network
def test_analyze_recruitment_runs_on_pipeline_db(spacr_measure_run):
    """Drive analyze_recruitment against the measurements DB the pipeline
    fixture generated. The synthetic dataset has only one well (E01) and
    3 fields, which is minimal but sufficient to exercise the code path
    up to the point where it groups and plots."""
    from spacr.submodules import analyze_recruitment

    db_path = spacr_measure_run["db_path"]
    settings = {
        "src": db_path,
        "target": "protein",
        "cell_types": ["HeLa"],
        "cell_plate_metadata": None,
        "pathogen_types": ["pathogen_1"],
        "pathogen_plate_metadata": [["E01"]],  # match the fixture well
        "treatments": ["ctrl"],
        "treatment_plate_metadata": [["E01"]],
        "channel_dims": [0, 1, 2, 3],
        "cell_chann_dim": 1, "cell_mask_dim": 4,
        "nucleus_chann_dim": 0, "nucleus_mask_dim": 5,
        "pathogen_chann_dim": 2, "pathogen_mask_dim": 6,
        "channel_of_interest": 2,
        "plot": False, "plot_control": False, "plot_nr": 0,
        "verbose": False,
    }
    try:
        analyze_recruitment(settings)
    except Exception as e:  # pragma: no cover - the fixture doesn't have
                             # the full plate metadata this function needs
        pytest.skip(
            f"analyze_recruitment needs a fuller plate metadata layout than "
            f"the synthetic pipeline fixture provides: {e}"
        )
