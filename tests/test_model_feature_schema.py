"""Schema-driven separation of model features from provenance."""

from __future__ import annotations

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
            match=r"cell_area.*must be numeric"):
        schema.model_feature_columns(frame)


def test_feature_frame_preserves_index_and_column_order():
    frame = _joined_frame()
    frame.index = [10, 20]

    selected = schema.model_feature_frame(frame)

    assert selected.index.tolist() == [10, 20]
    assert selected.columns.tolist() == schema.model_feature_columns(frame)
