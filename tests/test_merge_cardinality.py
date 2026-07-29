"""Explicit cardinality contracts for object-table merges."""

import sqlite3

import pandas as pd
import pytest

from spacr.io import MergeCardinalityError, _merge_with_cardinality
from spacr.utils import (_merge_and_save_to_database,
                         _update_database_with_merged_info, merge_dataframes)


def test_many_to_one_allows_repeated_object_metadata_keys():
    """Many objects can share a well, but the well-count table stays unique."""
    metadata = pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c1", "plate1_r1_c2"],
        "object_label": [1, 2, 1],
    })
    counts = pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c2"],
        "cells_per_well": [2, 1],
    })

    merged = _merge_with_cardinality(
        metadata,
        counts,
        on="prc",
        validate="many_to_one",
        left_name="object metadata",
        right_name="well counts",
    )

    assert list(merged["cells_per_well"]) == [2, 2, 1]


def test_many_to_one_names_duplicated_right_keys():
    metadata = pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c1"],
        "object_label": [1, 2],
    })
    counts = pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c1"],
        "cells_per_well": [2, 99],
    })

    with pytest.raises(MergeCardinalityError) as excinfo:
        _merge_with_cardinality(
            metadata,
            counts,
            on="prc",
            validate="many_to_one",
            left_name="object metadata",
            right_name="well counts",
        )

    message = str(excinfo.value)
    assert "validate='many_to_one'" in message
    assert "well counts has duplicated ['prc']" in message
    assert "plate1_r1_c1" in message


def test_one_to_one_names_duplicated_indexes_on_both_sides():
    left = pd.DataFrame({"area": [10, 11]}, index=["key1", "key1"])
    right = pd.DataFrame({"intensity": [1, 2]}, index=["key1", "key1"])

    with pytest.raises(MergeCardinalityError) as excinfo:
        _merge_with_cardinality(
            left,
            right,
            left_index=True,
            right_index=True,
            validate="one_to_one",
            left_name="grouped cell data",
            right_name="grouped nucleus data",
        )

    message = str(excinfo.value)
    assert "grouped cell data has duplicated index" in message
    assert "grouped nucleus data has duplicated index" in message
    assert "key1" in message


def test_measurement_writer_rejects_duplicate_object_labels(tmp_path):
    """Morphology/intensity fan-out is rejected before rows reach SQLite."""
    morphology = pd.DataFrame({
        "label": [1, 1],
        "cell_area": [10.0, 11.0],
    })
    intensity = pd.DataFrame({
        "label": [1],
        "cell_channel_0_mean_intensity": [3.0],
    })

    with pytest.raises(
            pd.errors.MergeError,
            match="not a one-to-one merge"):
        _merge_and_save_to_database(
            morphology,
            intensity,
            "cell",
            str(tmp_path),
            "plate1_A01_1",
            "experiment",
        )

    assert not (tmp_path / "measurements" / "measurements.db").exists()


def test_database_metadata_update_rejects_duplicate_source_keys(tmp_path):
    """Updating a table must not multiply its existing rows."""
    db_path = tmp_path / "measurements.db"
    existing = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
        "png_path": ["one.png", "two.png"],
    })
    with sqlite3.connect(db_path) as connection:
        existing.to_sql("png_list", connection, index=False)

    annotations = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o1"],
        "condition": ["control", "treated"],
    })
    with pytest.raises(
            pd.errors.MergeError,
            match="not a many-to-one merge"):
        _update_database_with_merged_info(
            str(db_path),
            annotations,
            columns=["condition", "prcfo"],
        )

    with sqlite3.connect(db_path) as connection:
        unchanged = pd.read_sql_query("SELECT * FROM png_list", connection)
    pd.testing.assert_frame_equal(unchanged, existing)


def test_image_feature_merge_rejects_duplicate_feature_keys():
    """One feature vector, at most, may be attached to each crop key."""
    image_paths = pd.DataFrame(
        {"png_path": ["one.png"]},
        index=["plate1_r1_c1_f1_o1"],
    )
    features = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o1"],
        "area": [10.0, 11.0],
    })

    with pytest.raises(
            pd.errors.MergeError,
            match="not a many-to-one merge"):
        merge_dataframes(features, image_paths, verbose=False)
