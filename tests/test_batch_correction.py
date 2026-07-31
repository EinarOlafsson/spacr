"""Focused tests for shared plate/batch-effect correction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _two_batch_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Return two equally shaped batches with a pure location/scale effect."""
    first = np.arange(1.0, 7.0)
    features = pd.DataFrame({
        "feature_a": np.concatenate([first, 10.0 + 2.0 * first]),
        "feature_b": np.concatenate([first * 3.0, 20.0 + first * 6.0]),
    })
    batch = pd.Series(["plate1"] * 6 + ["plate2"] * 6)
    return features, batch


def test_center_removes_batch_location_shift_and_preserves_global_mean():
    from spacr.batch_correction import correct_batch_effects

    features, batch = _two_batch_frame()
    corrected, report = correct_batch_effects(
        features, batch, method="center", min_samples=3,
    )

    centers = corrected.groupby(batch).mean()
    assert np.allclose(centers.iloc[0], centers.iloc[1])
    assert np.allclose(corrected.mean(), features.mean())
    assert report.centroid_spread_after == pytest.approx(0.0, abs=1e-12)
    assert report.centroid_spread_before > report.centroid_spread_after


def test_zscore_aligns_each_batch_location_and_scale():
    from spacr.batch_correction import correct_batch_effects

    features, batch = _two_batch_frame()
    corrected, _report = correct_batch_effects(
        features, batch, method="zscore", min_samples=3,
    )

    grouped = corrected.groupby(batch)
    assert np.allclose(grouped.mean().iloc[0], grouped.mean().iloc[1])
    assert np.allclose(
        grouped.std(ddof=0).iloc[0],
        grouped.std(ddof=0).iloc[1],
    )


def test_robust_zscore_aligns_medians_without_hiding_an_outlier():
    from spacr.batch_correction import correct_batch_effects

    features = pd.DataFrame({
        "signal": [0, 1, 2, 3, 1000, 10, 11, 12, 13, 1010],
    })
    batch = pd.Series(["a"] * 5 + ["b"] * 5)

    corrected, _report = correct_batch_effects(
        features, batch, method="robust_zscore", min_samples=3,
    )

    medians = corrected.groupby(batch).median()["signal"]
    assert medians["a"] == pytest.approx(medians["b"])
    assert corrected.loc[4, "signal"] > corrected.loc[3, "signal"] + 100
    assert corrected.loc[9, "signal"] > corrected.loc[8, "signal"] + 100


def test_control_center_aligns_controls_and_preserves_treatment_delta():
    from spacr.batch_correction import correct_batch_effects

    batch = pd.Series(["plate1"] * 8 + ["plate2"] * 8)
    condition = pd.Series((["neg"] * 4 + ["drug"] * 4) * 2)
    base = np.tile([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0], 2)
    features = pd.DataFrame({
        "signal": base + np.repeat([0.0, 10.0], 8),
    })

    corrected, report = correct_batch_effects(
        features,
        batch,
        method="control_center",
        control=condition,
        control_values="neg",
        min_samples=3,
    )

    controls = corrected.loc[condition == "neg"].groupby(batch).median()
    assert controls.loc["plate1", "signal"] == pytest.approx(
        controls.loc["plate2", "signal"],
    )
    for plate in ("plate1", "plate2"):
        rows = batch == plate
        delta = (
            corrected.loc[rows & (condition == "drug"), "signal"].median()
            - corrected.loc[rows & (condition == "neg"), "signal"].median()
        )
        assert delta == pytest.approx(5.0)
    assert report.controls == 8


def test_missing_control_policy_is_explicit():
    from spacr.batch_correction import correct_batch_effects

    features = pd.DataFrame({"signal": np.arange(12.0)})
    batch = pd.Series(["a"] * 6 + ["b"] * 6)
    control = pd.Series(["neg"] * 3 + ["drug"] * 9)

    with pytest.raises(ValueError, match="No usable reference controls"):
        correct_batch_effects(
            features, batch, method="control_center", control=control,
            control_values="neg", min_samples=3,
        )

    corrected, report = correct_batch_effects(
        features, batch, method="control_center", control=control,
        control_values="neg", min_samples=3, missing_control="skip",
    )
    assert np.allclose(corrected.loc[batch == "b"], features.loc[batch == "b"])
    assert any("unchanged" in warning for warning in report.warnings)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "mystery"}, "Unknown batch_correction"),
        ({"method": "center", "missing_control": "guess"}, "must be 'error'"),
    ],
)
def test_invalid_settings_fail_loudly(kwargs, message):
    from spacr.batch_correction import correct_batch_effects

    features, batch = _two_batch_frame()
    with pytest.raises(ValueError, match=message):
        correct_batch_effects(features, batch, **kwargs)


def test_non_numeric_values_and_missing_batch_labels_fail_loudly():
    from spacr.batch_correction import correct_batch_effects

    with pytest.raises(ValueError, match="non-numeric"):
        correct_batch_effects(
            pd.DataFrame({"signal": ["1", "bad", "3"]}),
            pd.Series(["a", "a", "a"]),
            method="none",
        )
    with pytest.raises(ValueError, match="missing for 1"):
        correct_batch_effects(
            pd.DataFrame({"signal": [1.0, 2.0, 3.0]}),
            pd.Series(["a", None, "b"]),
            method="center",
        )


def test_single_batch_is_a_reported_noop():
    from spacr.batch_correction import correct_batch_effects

    features = pd.DataFrame({"signal": [1.0, 2.0, 3.0]})
    corrected, report = correct_batch_effects(
        features, pd.Series(["a"] * 3), method="center",
    )
    pd.testing.assert_frame_equal(corrected, features)
    assert report.warnings == ["Only 1 batch was present; correction was a no-op."]


def test_metadata_adapter_validates_columns_and_never_adds_metadata():
    from spacr.batch_correction import correct_from_metadata

    features, batch = _two_batch_frame()
    metadata = pd.DataFrame({
        "plateID": batch,
        "columnID": ["c1", "c2", "c3", "c1", "c2", "c3"] * 2,
    })
    corrected, _report = correct_from_metadata(
        features,
        metadata,
        batch_correction="center",
    )
    assert list(corrected.columns) == list(features.columns)
    with pytest.raises(ValueError, match="batch_column='runID'"):
        correct_from_metadata(
            features, metadata, batch_correction="center",
            batch_column="runID",
        )


def test_settings_adapter_uses_context_fallbacks_for_blank_controls():
    from spacr.batch_correction import correction_kwargs

    result = correction_kwargs(
        {
            "batch_correction": "control_center",
            "batch_control_column": "",
            "batch_control_values": None,
        },
        default_control_column="condition",
        default_control_values="negative",
    )
    assert result["batch_control_column"] == "condition"
    assert result["batch_control_values"] == "negative"


def test_report_writer_is_atomic_and_json_serializable(tmp_path):
    from spacr.batch_correction import correct_batch_effects, write_report

    features, batch = _two_batch_frame()
    _corrected, report = correct_batch_effects(
        features, batch, method="center",
    )
    destination = write_report(report, tmp_path / "correction.json")

    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data["method"] == "center"
    assert data["rows"] == len(features)
    assert not (tmp_path / ".correction.json.tmp").exists()


def test_preprocess_data_applies_correction_before_final_scaling():
    from spacr.utils import preprocess_data

    frame = pd.DataFrame({
        "cell_channel_0_mean_intensity": np.concatenate([
            np.arange(10.0), np.arange(10.0) + 100.0,
        ]),
        "plateID": ["plate1"] * 10 + ["plate2"] * 10,
        "columnID": ["c1"] * 20,
    })

    raw = preprocess_data(
        frame, filter_by="channel_0", remove_highly_correlated=False,
        log_data=False, exclude=None,
    )
    corrected = preprocess_data(
        frame, filter_by="channel_0", remove_highly_correlated=False,
        log_data=False, exclude=None, batch_correction="center",
        batch_column="plateID",
    )

    assert abs(raw[:10].mean() - raw[10:].mean()) > 1.0
    assert corrected[:10].mean() == pytest.approx(
        corrected[10:].mean(), abs=1e-12,
    )


def test_defaults_and_gui_categories_expose_batch_correction():
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import (
        _APP_COMBO_OPTIONS,
        SettingsWidgets,
        categories_for_app,
        resolve_default_settings,
    )
    from spacr.settings import categories

    for app_key in ("umap", "ml_analyze", "regression"):
        defaults = resolve_default_settings(app_key)
        assert defaults["batch_correction"] == "none"
        assert _APP_COMBO_OPTIONS[app_key]["batch_missing_control"] == [
            "error", "skip",
        ]
        grouped = categories_for_app(app_key, categories)
        assert grouped["Plate & Batch Correction"] == [
            "batch_correction",
            "batch_column",
            "batch_control_column",
            "batch_control_values",
            "batch_min_samples",
            "batch_missing_control",
        ]
        for key in grouped["Plate & Batch Correction"]:
            assert "/batch_correction/index.html" in (
                SettingsWidgets(app_key).plain_tooltip_for(key)
            )


def test_every_batch_setting_has_an_informative_api_tooltip():
    from spacr.settings import tooltips

    keys = (
        "batch_correction",
        "batch_column",
        "batch_control_column",
        "batch_control_values",
        "batch_min_samples",
        "batch_missing_control",
    )
    for key in keys:
        text = tooltips[key]
        assert text.startswith("(")
        assert len(text.split()) >= 20
        assert "spacr.batch_correction.correct_batch_effects" in text
