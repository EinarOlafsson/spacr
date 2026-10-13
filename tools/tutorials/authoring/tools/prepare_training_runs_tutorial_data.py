#!/usr/bin/env python3
"""Build disclosed synthetic spaCR training logs for lesson 28.

The transferred microscopy tutorial dataset contains no saved classifier
training curves.  These files follow the exact directory and CSV layout that
``spacr.deep_spacr`` writes, allowing the real Training Runs parser, comparer,
settings diff, and plotting code to be demonstrated without implying that a
model was trained on the transferred experiment.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "synthetic" / "training_runs"


def _write_curve(folder: Path, split: str, accuracy, loss) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("train.csv" if split == "train" else "validation.csv")
    rows = []
    for epoch, (acc, err) in enumerate(zip(accuracy, loss), start=1):
        rows.append({
            "accuracy": float(acc),
            "neg_accuracy": float(max(0.0, acc - 0.025)),
            "pos_accuracy": float(min(1.0, acc + 0.025)),
            "prauc": float(max(0.0, acc - 0.045)),
            "optimal_threshold": float(0.48 + 0.015 * np.sin(epoch / 3)),
            "loss": float(err),
            "epoch": epoch,
            "Accuracy": float(acc),
        })
    frame = pd.DataFrame(rows)
    # spaCR appends one one-row frame per epoch, including a repeated index 0.
    for index in range(len(frame)):
        frame.iloc[[index]].reset_index(drop=True).to_csv(
            path, mode="w" if index == 0 else "a",
            header=index == 0, index=True,
        )


def _write_settings(project: Path, model: str, epochs: int, settings: dict) -> None:
    folder = project / "settings"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(settings.items(), columns=["Key", "Value"]).to_csv(
        folder / f"train_test_{model}_{epochs}.csv", index=False)


def _single_run(name: str, *, model: str, epochs: int, train_acc, val_acc,
                train_loss, val_loss, settings: dict) -> Path:
    project = OUTPUT / name
    run = project / "model" / model / "0_1" / f"epochs_{epochs}"
    _write_curve(run, "train", train_acc, train_loss)
    _write_curve(run, "val", val_acc, val_loss)
    _write_settings(project, model, epochs, settings)
    return run


def _cv_run() -> Path:
    project = OUTPUT / "cross_validated"
    model = "resnet50"
    epochs = 18
    run = project / "model" / model / "0_1" / f"epochs_{epochs}"
    x = np.linspace(0.0, 1.0, epochs)
    for fold, offset in ((1, -0.018), (2, 0.006), (3, 0.022)):
        train = 0.54 + 0.39 * (1 - np.exp(-3.0 * x)) + offset / 3
        val = 0.52 + 0.34 * (1 - np.exp(-2.7 * x)) + offset
        val[-3:] -= np.array([0.0, 0.006, 0.012])
        train_loss = 1.15 * np.exp(-2.7 * x) + 0.18
        val_loss = 1.20 * np.exp(-2.25 * x) + 0.27 - offset / 2
        _write_curve(run / f"fold_{fold}", "train", train, train_loss)
        _write_curve(run / f"fold_{fold}", "val", val, val_loss)
    _write_settings(project, model, epochs, {
        "src": "/synthetic/tutorial/cross_validated",
        "model_type": model,
        "epochs": epochs,
        "batch_size": 32,
        "learning_rate": 0.0002,
        "loss_type": "focal_loss",
        "augment": True,
        "classes": "['nc', 'pc']",
        "cross_validation_folds": 3,
        "n_jobs": 12,
        "device": "cuda",
    })
    return run


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    x20 = np.linspace(0.0, 1.0, 20)
    baseline_train = 0.53 + 0.43 * (1 - np.exp(-3.1 * x20))
    baseline_val = 0.51 + 0.31 * (1 - np.exp(-3.0 * x20))
    baseline_val[14:] -= np.linspace(0.0, 0.055, 6)
    baseline = _single_run(
        "baseline", model="maxvit_t", epochs=20,
        train_acc=baseline_train, val_acc=baseline_val,
        train_loss=1.25 * np.exp(-2.8 * x20) + 0.14,
        val_loss=np.r_[1.30 * np.exp(-2.35 * x20[:14]) + 0.27,
                       np.linspace(0.32, 0.43, 6)],
        settings={
            "src": "/synthetic/tutorial/baseline", "model_type": "maxvit_t",
            "epochs": 20, "batch_size": 64, "learning_rate": 0.0001,
            "loss_type": "focal_loss", "augment": True,
            "classes": "['nc', 'pc']", "cross_validation_folds": 0,
            "n_jobs": 30, "device": "cuda",
        },
    )

    x25 = np.linspace(0.0, 1.0, 25)
    tuned_train = 0.54 + 0.41 * (1 - np.exp(-3.25 * x25))
    tuned_val = 0.52 + 0.36 * (1 - np.exp(-3.05 * x25))
    tuned_val[21:] -= np.array([0.0, 0.004, 0.010, 0.018])
    tuned = _single_run(
        "tuned", model="maxvit_t", epochs=25,
        train_acc=tuned_train, val_acc=tuned_val,
        train_loss=1.18 * np.exp(-3.0 * x25) + 0.15,
        val_loss=np.r_[1.24 * np.exp(-2.65 * x25[:21]) + 0.235,
                       np.array([0.31, 0.315, 0.328, 0.35])],
        settings={
            "src": "/synthetic/tutorial/tuned", "model_type": "maxvit_t",
            "epochs": 25, "batch_size": 32, "learning_rate": 0.0003,
            "loss_type": "focal_loss", "augment": True,
            "classes": "['nc', 'pc']", "cross_validation_folds": 0,
            "n_jobs": 8, "device": "cuda",
        },
    )

    cross_validated = _cv_run()
    broken = OUTPUT / "incomplete" / "model" / "maxvit_t" / "0_1" / "epochs_8"
    broken.mkdir(parents=True)
    (broken / "maxvit_t_epoch_8_channels_0_1.pth").write_bytes(
        b"synthetic tutorial checkpoint placeholder")

    manifest = {
        "schema": 1,
        "disclosure": "Synthetic progress logs; no model was trained.",
        "purpose": "Exercise the real spaCR Training Runs parser and UI.",
        "runs": {
            "baseline": str(baseline),
            "tuned": str(tuned),
            "cross_validated": str(cross_validated),
            "incomplete": str(broken),
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
