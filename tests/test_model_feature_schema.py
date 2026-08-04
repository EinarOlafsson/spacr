"""Schema-driven separation of model features from provenance."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from spacr import schema


def _joined_frame():
    return pd.DataFrame({
        # Numeric identity/provenance must not leak into a model.
        "object_label": [1, 2],
        "cell_id": [1.0, 2.0],
        "measurement_ndim": [3, 3],
        "n_z": [5, 5],
        "voxel_size_z_um": [1.5, 1.5],
        "voxel_size_xy_um": [0.25, 0.25],
        "object_label_nucleus": [1, 2],
        "measurement_ndim_pathogen": [3, 3],
        # Measurement namespaces are model features.
        "cell_area": [100.0, 120.0],
        "cell_channel_0_mean_intensity": [4.0, 8.0],
        "nucleus_area_nucleus": [20.0, 24.0],
        # Derived/user data requires an explicit contract.
        "recruitment": [0.2, 0.8],
        "condition_code": [0, 1],
        "condition": ["control", "treated"],
    })


def test_provenance_is_excluded_even_when_numeric():
    frame = _joined_frame()

    features = schema.model_feature_columns(frame)

    assert features == [
        "cell_area",
        "cell_channel_0_mean_intensity",
        "nucleus_area_nucleus",
        "recruitment",
    ]
    assert not {
        "object_label",
        "cell_id",
        "measurement_ndim",
        "n_z",
        "voxel_size_z_um",
        "voxel_size_xy_um",
        "object_label_nucleus",
        "measurement_ndim_pathogen",
    } & set(features)


def test_known_and_explicit_derived_features_are_admitted_not_annotations():
    frame = _joined_frame()
    frame["custom_ratio"] = [1.5, 2.5]

    features = schema.model_feature_columns(
        frame,
        extra_features=["custom_ratio"],
    )

    assert "recruitment" in features
    assert "custom_ratio" in features
    assert "condition_code" not in features
    assert "condition" not in features


def test_generic_frames_require_an_explicit_unknown_policy():
    frame = pd.DataFrame({
        "signal": [1.0, 2.0],
        "noise": [3.0, 4.0],
        "object_label": [1, 2],
        "label": ["a", "b"],
    })

    assert schema.model_feature_columns(frame) == []
    assert schema.model_feature_columns(
        frame, allow_unknown=True) == ["signal", "noise"]


def test_declared_object_feature_must_be_numeric():
    frame = pd.DataFrame({"cell_area": ["large", "small"]})

    with pytest.raises(
            schema.ModelFeatureSchemaError,
            match=r"cell_area \(object\)"):
        schema.model_feature_columns(frame)


def test_every_unusable_feature_is_named_in_one_error():
    """One run, one error, every offending column and its dtype in it.

    Naming one column and stopping made the user pay for a whole read/merge
    per column they had to discover.
    """
    frame = pd.DataFrame({
        "cell_area": ["large", "small"],
        "cell_channel_0_mode_intensity": [None, None],
        "nucleus_perimeter": ["n/a", "n/a"],
        "cell_channel_1_mean_intensity": [1.0, 2.0],
    })

    with pytest.raises(schema.ModelFeatureSchemaError) as excinfo:
        schema.model_feature_columns(frame)

    message = str(excinfo.value)
    assert "3 declared model features" in message
    for name in ("cell_area", "cell_channel_0_mode_intensity",
                 "nucleus_perimeter"):
        assert f"{name} (object)" in message
    # The one usable column is not blamed.
    assert "cell_channel_1_mean_intensity" not in message
    # The user is told which control fixes it, and that it takes several.
    assert "Exclude" in message
    assert "any number of columns" in message
    # The all-missing column is diagnosed differently from the text one.
    assert "every value is missing" in message
    assert "'n/a'" in message


def test_a_measurement_that_is_null_everywhere_is_a_number_not_an_error():
    """The reported crash, at its smallest.

    ``pandas.read_sql`` types an all-NULL result column ``object`` whatever
    SQLite declared, and spaCR writes NULL for an honest NaN -- so an
    entirely-NaN measurement came back as a column of ``None`` and was
    refused as "non-numeric" on data that has nothing wrong with it.
    """
    frame = pd.DataFrame({
        "cell_channel_0_mode_intensity": [None, None, None],
        "cell_area": [1.0, 2.0, 3.0],
    })

    converted = schema.coerce_model_feature_types(frame)

    assert converted["cell_channel_0_mode_intensity"].dtype == "float64"
    assert converted["cell_channel_0_mode_intensity"].isna().all()
    assert schema.model_feature_columns(converted) == [
        "cell_channel_0_mode_intensity", "cell_area"]


def test_repairing_an_all_null_measurement_warns_about_nothing():
    """Nothing was lost, so nothing is announced. Only text coercion is loud."""
    frame = pd.DataFrame({"cell_channel_0_mode_intensity": [None, None]})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        schema.coerce_model_feature_types(frame)


def test_numeric_text_measurements_are_losslessly_coerced():
    frame = pd.DataFrame({
        "cell_channel_0_mode_intensity": ["12.5", " 7 ", None, ""],
        "object_label": [1, 2, 3, 4],
    })

    with pytest.warns(UserWarning, match="stored as text"):
        converted = schema.coerce_model_feature_types(frame)

    assert converted is not frame
    assert converted["cell_channel_0_mode_intensity"].tolist()[:2] == [
        12.5, 7.0]
    assert converted["cell_channel_0_mode_intensity"].isna().tolist() == [
        False, False, True, True]
    assert schema.model_feature_columns(converted) == [
        "cell_channel_0_mode_intensity"]
    # The database frame remains suitable for export/auditing.
    assert frame["cell_channel_0_mode_intensity"].iloc[0] == "12.5"


def test_numeric_coercion_reports_genuinely_invalid_measurements():
    frame = pd.DataFrame({
        "cell_channel_0_mode_intensity": ["12.5", "saturated"],
    })

    with pytest.raises(
            schema.ModelFeatureSchemaError,
            match=r"cell_channel_0_mode_intensity.*saturated"):
        schema.coerce_model_feature_types(frame)


def test_coercion_reports_every_unrecoverable_column_at_once():
    """'12.0' is recoverable and is recovered; 'n/a' is not and never becomes
    NaN behind the user's back. Both verdicts, all columns, one error."""
    frame = pd.DataFrame({
        "cell_channel_0_mode_intensity": ["12.0", "n/a"],
        "nucleus_area": ["3", "not measured"],
        "cell_perimeter": ["1.5", "2.5"],
    })

    with pytest.raises(schema.ModelFeatureSchemaError) as excinfo:
        schema.coerce_model_feature_types(frame)

    message = str(excinfo.value)
    assert "2 declared model features" in message
    assert "cell_channel_0_mode_intensity (object)" in message
    assert "nucleus_area (object)" in message
    assert "cell_perimeter" not in message
    assert "'n/a'" in message and "'not measured'" in message


def test_excluded_invalid_measurement_does_not_block_model_boundary():
    frame = pd.DataFrame({
        "cell_area": [10.0, 12.0],
        "cell_channel_0_mode_intensity": ["bad", "bad"],
    })

    converted = schema.coerce_model_feature_types(
        frame, exclude=["cell_channel_0_mode_intensity"])

    assert converted is frame
    assert schema.model_feature_columns(
        converted, exclude=["cell_channel_0_mode_intensity"]) == ["cell_area"]


def test_feature_frame_preserves_index_and_column_order():
    frame = _joined_frame()
    frame.index = [10, 20]

    selected = schema.model_feature_frame(frame)

    assert selected.index.tolist() == [10, 20]
    assert selected.columns.tolist() == schema.model_feature_columns(frame)
