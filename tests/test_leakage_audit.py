"""Strict train/test, validation and whole-CV leakage guarantees."""
from __future__ import annotations

import json

import numpy as np
import pytest


def _crop(root, split, cls, name, payload=b"crop"):
    path = root / split / cls / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_train_test_audit_finds_renamed_byte_identical_copy(tmp_path):
    from spacr.classifier_evaluation import audit_dataset_splits

    _crop(tmp_path, "train", "neg", "plate1_A01_f1_o1.png", b"same pixels")
    _crop(tmp_path, "test", "neg", "plate9_Z99_f9_o9.png", b"same pixels")
    report = audit_dataset_splits(
        tmp_path, group_by="none", hash_content=True,
    )
    assert not report.passed
    assert report.overlap_counts["content_sha256"] == 1
    assert "content_sha256" in report.critical_levels


def test_train_test_audit_finds_well_and_augmentation_lineage(tmp_path):
    from spacr.classifier_evaluation import audit_dataset_splits

    _crop(tmp_path, "train", "neg", "plate1_A01_f1_o1.png", b"one")
    _crop(
        tmp_path, "test", "neg", "plate1_A01_f1_o1_rot90.png", b"rotated",
    )
    report = audit_dataset_splits(tmp_path, group_by="well")
    assert {"augmentation_family", "object", "well"} <= set(
        report.critical_levels
    )


def test_unknown_protected_identity_fails_strict_audit(tmp_path):
    from spacr.classifier_evaluation import audit_dataset_splits

    _crop(tmp_path, "train", "neg", "unparseable.png", b"one")
    _crop(tmp_path, "test", "neg", "also-unparseable.png", b"two")
    report = audit_dataset_splits(
        tmp_path, group_by="well", require_identity=True,
    )
    assert not report.passed
    assert "unverifiable_well" in report.critical_levels
    assert report.unverifiable_counts["well"] == 2


def test_whole_cv_audit_detects_family_in_two_held_out_folds():
    from spacr.classifier_evaluation import audit_cv_folds

    paths = [
        "plate1_A01_f1_o1.png",
        "plate1_A01_f1_o1_flip_h.png",
        "plate1_A02_f1_o2.png",
        "plate1_A03_f1_o3.png",
    ]
    folds = [
        (np.asarray([1, 2, 3]), np.asarray([0])),
        (np.asarray([0, 2, 3]), np.asarray([1])),
        (np.asarray([0, 1]), np.asarray([2, 3])),
    ]
    audit = audit_cv_folds(
        paths, folds, labels=[0, 0, 1, 1], group_by="well",
    )
    assert not audit.passed
    assert audit.overlap_counts["augmentation_family"] == 1
    assert "augmentation_family" in audit.critical_levels
    assert "well" in audit.critical_levels


def test_whole_cv_audit_proves_complete_group_partition():
    from spacr.classifier_evaluation import audit_cv_folds

    paths = [
        "plate1_A01_f1_o1.png",
        "plate1_A01_f1_o2.png",
        "plate1_A02_f1_o3.png",
        "plate1_A02_f1_o4.png",
    ]
    folds = [
        (np.asarray([2, 3]), np.asarray([0, 1])),
        (np.asarray([0, 1]), np.asarray([2, 3])),
    ]
    audit = audit_cv_folds(
        paths, folds, labels=[0, 1, 0, 1], group_by="well",
    )
    assert audit.passed
    assert audit.validation_membership_missing == []
    assert audit.validation_membership_duplicate == []


def test_whole_cv_audit_rejects_missing_validation_membership():
    from spacr.classifier_evaluation import audit_cv_folds

    audit = audit_cv_folds(
        ["plate1_A01_f1_o1.png", "plate1_A02_f1_o2.png"],
        [(np.asarray([1]), np.asarray([0]))],
        labels=[0, 1],
        group_by="well",
    )
    assert not audit.passed
    assert audit.validation_membership_missing == [1]
    assert "validation_membership_missing" in audit.critical_levels


def test_group_holdout_keeps_wells_intact_and_near_requested_size():
    from spacr.io import make_validation_holdout

    labels = np.tile([0, 1], 12)
    groups = np.repeat([f"plate1_A{i:02d}" for i in range(1, 7)], 4)
    train, validation = make_validation_holdout(
        labels, 0.25, groups, seed=5,
    )
    assert set(groups[train]).isdisjoint(groups[validation])
    assert 4 <= len(validation) <= 8
    assert set(labels[validation]) == {0, 1}


def test_ordinary_loader_holdout_keeps_wells_on_one_side(tmp_path):
    from PIL import Image
    from spacr.classifier_evaluation import sample_identity
    from spacr.io import dataset_filenames, generate_loaders

    for class_index, cls in enumerate(("neg", "pos")):
        folder = tmp_path / "train" / cls
        folder.mkdir(parents=True)
        for well_index in range(1, 5):
            for object_index in range(2):
                pixels = np.full(
                    (8, 8, 3), class_index * 80 + well_index, dtype=np.uint8,
                )
                Image.fromarray(pixels).save(
                    folder / (
                        f"plate1_A{well_index:02d}_f1_"
                        f"o{class_index}{object_index}.png"
                    )
                )
    train, validation, _ = generate_loaders(
        str(tmp_path),
        mode="train",
        classes=["neg", "pos"],
        validation_split=0.25,
        group_by="well",
        n_jobs=0,
        batch_size=4,
        image_size=8,
    )
    train_wells = {
        sample_identity(path)["well"]
        for path in dataset_filenames(train.dataset)
    }
    validation_wells = {
        sample_identity(path)["well"]
        for path in dataset_filenames(validation.dataset)
    }
    assert train_wells.isdisjoint(validation_wells)


def test_cli_exit_code_and_json_report(tmp_path, capsys):
    from spacr.cli_leakage import main

    _crop(tmp_path, "train", "neg", "plate1_A01_f1_o1.png", b"same")
    _crop(tmp_path, "test", "neg", "plate2_A01_f1_o2.png", b"same")
    output = tmp_path / "audit.json"
    assert main([
        str(tmp_path), "--group-by", "none", "--output", str(output),
    ]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert json.loads(output.read_text())["passed"] is False


def test_classifier_stops_before_training_on_leaked_dataset(tmp_path):
    from spacr.classifier_evaluation import LeakageError
    from spacr.deep_spacr import train_test_model

    for split in ("train", "test"):
        for cls in ("neg", "pos"):
            _crop(
                tmp_path, split, cls,
                f"plate1_A01_f1_o{1 if cls == 'neg' else 2}.png",
                cls.encode(),
            )
    with pytest.raises(LeakageError, match="leakage"):
        train_test_model({
            "src": str(tmp_path),
            "classes": ["neg", "pos"],
            "train": True,
            "test": False,
            "model_type": "resnet18",
            "epochs": 1,
            "n_jobs": 0,
        })
