"""Classifier evaluation workbench backend tests."""

from __future__ import annotations

import json

import numpy as np
import pytest


def _predictions(seed=4):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], 20)
    signal = np.where(labels == 1, 1.4, -1.4) + rng.normal(0, 0.8, len(labels))
    pos = 1.0 / (1.0 + np.exp(-signal))
    probabilities = np.column_stack([1.0 - pos, pos])
    folds = np.tile(np.arange(4), 10)
    paths = [
        f"plate{1 + (i % 2)}_A{1 + (i % 4):02d}_f{1 + (i % 3)}_o{i}.png"
        for i in range(len(labels))
    ]
    return labels, probabilities, folds, paths


def test_sample_identity_collapses_exported_augmentation_suffixes():
    from spacr.classifier_evaluation import augmentation_family, sample_identity

    original = "plate1_A01_f2_o7.png"
    augmented = "plate1_A01_f2_o7_rot90.png"
    assert augmentation_family(original) == augmentation_family(augmented)
    identity = sample_identity(augmented)
    assert identity["plate"] == "plate1"
    assert identity["well"] == "plate1_A01"
    assert identity["field"] == "plate1_A01_f2"


def test_sample_identity_parses_canonical_prcfo_row_and_column_tokens():
    from spacr.classifier_evaluation import sample_identity

    identity = sample_identity("plate1_r1_c1_f2_o7.png")
    assert identity["plate"] == "plate1"
    assert identity["well"] == "plate1_r1_c1"
    assert identity["field"] == "plate1_r1_c1_f2"


def test_leakage_audit_finds_object_and_requested_group_overlap():
    from spacr.classifier_evaluation import audit_split_leakage

    train = [
        "/train/plate1_A01_f1_o1.png",
        "/train/plate1_A02_f1_o2.png",
    ]
    validation = [
        "/val/plate1_A01_f1_o1_rot90.png",
        "/val/plate2_A01_f1_o3.png",
    ]
    report = audit_split_leakage(train, validation, group_by="well")

    assert report.passed is False
    assert report.overlap_counts["augmentation_family"] == 1
    assert report.overlap_counts["well"] == 1
    assert "augmentation_family" in report.critical_levels
    assert "well" in report.critical_levels


def test_plate_overlap_is_not_a_failure_for_well_grouping():
    from spacr.classifier_evaluation import audit_split_leakage

    report = audit_split_leakage(
        ["plate1_A01_f1_o1.png"],
        ["plate1_A02_f1_o2.png"],
        group_by="well",
    )
    assert report.passed
    assert report.overlap_counts["plate"] == 1
    assert report.critical_levels == []


def test_leakage_audit_can_stop_a_run():
    from spacr.classifier_evaluation import LeakageError, audit_split_leakage

    with pytest.raises(LeakageError, match="Train/validation leakage"):
        audit_split_leakage(
            ["plate1_A01_f1_o1.png"],
            ["plate1_A01_f1_o1.png"],
            group_by="well",
            raise_on_leakage=True,
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.2, 0.8],
        [[0.2], [0.8]],
        [[0.8, 0.2], [0.2, 0.8]],
    ],
)
def test_probability_normalization_accepts_binary_and_matrix_shapes(
        probabilities):
    from spacr.classifier_evaluation import normalize_probabilities

    result = normalize_probabilities(probabilities)
    assert result.shape == (2, 2)
    assert np.allclose(result.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([[0.2, np.nan]], "NaN"),
        ([[1.2, -0.2]], "between 0 and 1"),
        ([[0.0, 0.0]], "sums to zero"),
    ],
)
def test_invalid_probabilities_fail_loudly(probabilities, message):
    from spacr.classifier_evaluation import normalize_probabilities

    with pytest.raises(ValueError, match=message):
        normalize_probabilities(probabilities)


def test_cross_fitted_temperature_never_uses_the_target_fold(monkeypatch):
    import spacr.classifier_evaluation as CE

    y, probabilities, folds, _paths = _predictions()
    calls = []
    original = CE.fit_temperature

    def spy(labels, probs):
        calls.append((np.asarray(labels).copy(), len(probs)))
        return original(labels, probs)

    monkeypatch.setattr(CE, "fit_temperature", spy)
    calibrated, temperatures = CE.cross_calibrate_probabilities(
        y, probabilities, folds, method="temperature",
    )

    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert set(temperatures) == {"0", "1", "2", "3"}
    assert [size for _labels, size in calls] == [30, 30, 30, 30]


def test_sparse_cross_fitted_calibration_warns_and_retains_raw(capsys):
    import spacr.classifier_evaluation as CE

    probabilities = np.asarray([
        [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9],
    ])
    warnings = []
    calibrated, temperatures = CE.cross_calibrate_probabilities(
        [0, 0, 1, 1],
        probabilities,
        [0, 0, 1, 1],
        warnings_out=warnings,
    )
    assert np.allclose(calibrated, probabilities)
    assert temperatures == {"0": 1.0, "1": 1.0}
    assert len(warnings) == 2
    assert "raw probabilities were retained" in capsys.readouterr().out


def test_prediction_evaluation_builds_all_workbench_tables():
    from spacr.classifier_evaluation import evaluate_predictions

    y, probabilities, folds, paths = _predictions()
    result = evaluate_predictions(
        y,
        probabilities,
        paths,
        classes=["negative", "positive"],
        fold_ids=folds,
        calibration_method="temperature",
        calibration_bins=5,
    )

    assert result["summary"]["n"] == len(y)
    assert result["summary"]["n_classes"] == 2
    assert 0 <= result["summary"]["expected_calibration_error"] <= 1
    assert result["confusion_counts"].to_numpy().sum() == len(y)
    assert list(result["confusion_counts"].index) == ["negative", "positive"]
    assert set(result["per_plate"]["plate"]) == {"plate1", "plate2"}
    assert result["per_plate"]["n"].sum() == len(y)
    assert set(result["calibration"]["class_name"]) == {
        "negative", "positive",
    }
    frame = result["predictions"]
    assert {
        "fold", "sample", "plate", "well", "field", "true_class",
        "predicted_class", "confidence", "raw_prob_negative",
        "prob_positive",
    }.issubset(frame.columns)


def test_multiclass_evaluation_preserves_arbitrary_class_count():
    from spacr.classifier_evaluation import evaluate_predictions

    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
    ])
    paths = [f"p1_A01_f1_o{i}.png" for i in range(6)]
    result = evaluate_predictions(
        labels, probabilities, paths, classes=["a", "b", "c"],
    )
    assert result["confusion_counts"].shape == (3, 3)
    assert result["summary"]["classes"] == ["a", "b", "c"]
    assert {"prob_a", "prob_b", "prob_c"}.issubset(
        result["predictions"].columns,
    )


def test_probability_columns_remain_unique_for_colliding_class_names():
    from spacr.classifier_evaluation import evaluate_predictions

    result = evaluate_predictions(
        [0, 1, 2],
        np.eye(3),
        ["p_A01_1_1.png", "p_A02_1_2.png", "p_A03_1_3.png"],
        classes=["A B", "A/B", ""],
    )
    assert result["summary"]["probability_column_names"] == [
        "A_B", "A_B_2", "class_2",
    ]
    probability_columns = [
        column
        for column in result["predictions"]
        if column.startswith("prob_")
    ]
    assert len(probability_columns) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"calibration_bins": 1}, "at least 2"),
        ({"classes": ["same", "same"]}, "unique"),
    ],
)
def test_invalid_evaluation_schema_fails_loudly(kwargs, message):
    from spacr.classifier_evaluation import evaluate_predictions

    with pytest.raises(ValueError, match=message):
        evaluate_predictions(
            [0, 1],
            [[0.9, 0.1], [0.1, 0.9]],
            ["p_A01_1_1.png", "p_A02_1_2.png"],
            **kwargs,
        )


def test_nested_group_folds_keep_groups_out_of_every_boundary():
    from spacr.classifier_evaluation import nested_group_folds

    groups = np.repeat([f"well_{i}" for i in range(12)], 4)
    labels = np.tile([0, 0, 1, 1], 12)
    nested = nested_group_folds(
        labels,
        outer_splits=3,
        inner_splits=2,
        groups=groups,
        seed=9,
    )

    seen_outer = []
    for fold in nested:
        outer_train = fold["train"]
        outer_validation = fold["validation"]
        assert set(groups[outer_train]).isdisjoint(groups[outer_validation])
        seen_outer.extend(outer_validation.tolist())
        for inner_train, inner_validation in fold["inner"]:
            assert set(groups[inner_train]).isdisjoint(
                groups[inner_validation],
            )
            assert set(inner_train).issubset(set(outer_train))
            assert set(inner_validation).issubset(set(outer_train))
    assert sorted(seen_outer) == list(range(len(labels)))


@pytest.mark.parametrize(("outer", "inner"), [(1, 2), (2, 1), (0, 0)])
def test_nested_group_folds_rejects_non_cv_fold_counts(outer, inner):
    from spacr.classifier_evaluation import nested_group_folds

    with pytest.raises(ValueError, match="at least 2"):
        nested_group_folds(
            [0, 1, 0, 1],
            outer_splits=outer,
            inner_splits=inner,
        )


def test_bundle_round_trip_and_figures(tmp_path):
    from spacr.classifier_evaluation import (
        EVALUATION_FILES,
        LeakageReport,
        evaluate_predictions,
        load_evaluation_bundle,
        write_evaluation_bundle,
    )

    y, probabilities, folds, paths = _predictions()
    evaluation = evaluate_predictions(
        y, probabilities, paths, classes=["negative", "positive"],
        fold_ids=folds,
    )
    report = LeakageReport(
        group_by="well",
        train_samples=30,
        validation_samples=10,
        overlap_counts={"well": 0},
        examples={"well": []},
    )
    manifest = write_evaluation_bundle(
        tmp_path / "evaluation",
        evaluation,
        leakage_reports=[report],
    )
    loaded = load_evaluation_bundle(manifest)

    assert loaded["summary"]["n"] == len(y)
    assert len(loaded["predictions"]) == len(y)
    assert loaded["confusion_counts"].to_numpy().sum() == len(y)
    assert loaded["leakage"]["passed"] is True
    assert (manifest.parent / EVALUATION_FILES["confusion_figure"]).is_file()
    assert (manifest.parent / EVALUATION_FILES["calibration_figure"]).is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1


def test_find_bundles_is_newest_first(tmp_path):
    from spacr.classifier_evaluation import (
        EVALUATION_FILES,
        find_evaluation_bundles,
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / EVALUATION_FILES["manifest"]).write_text("{}", encoding="utf-8")
    (second / EVALUATION_FILES["manifest"]).write_text("{}", encoding="utf-8")
    (first / EVALUATION_FILES["manifest"]).touch()
    (second / EVALUATION_FILES["manifest"]).touch()
    # Explicit nanosecond mtimes make ordering deterministic on coarse FSes.
    import os
    os.utime(first / EVALUATION_FILES["manifest"], ns=(1, 1))
    os.utime(second / EVALUATION_FILES["manifest"], ns=(2, 2))

    found = find_evaluation_bundles(tmp_path)
    assert found == [
        second / EVALUATION_FILES["manifest"],
        first / EVALUATION_FILES["manifest"],
    ]


def test_classifier_settings_have_documented_defaults():
    from spacr import settings as S

    defaults = S.deep_spacr_defaults({})
    expected = {
        "classifier_evaluation": True,
        "nested_cv_inner_folds": 0,
        "evaluation_calibration": "temperature",
        "evaluation_bins": 10,
        "evaluation_fail_on_leakage": True,
    }
    assert {key: defaults[key] for key in expected} == expected
    for key in expected:
        assert "API:" in S.tooltips[key]
