"""AnnData export — everything that happens before ``anndata`` is needed.

``tests/test_anndata_export.py`` covers the written ``.h5ad`` and skips
wholesale where the optional extra is absent. Almost none of this module
needs it: the read, the ``X``/``obs`` boundary, the ``var`` dictionary, the
NaN policies, the embedding alignment and the settings seam are pandas and
numpy, and they are what decides whether the exported matrix means what it
says. They are tested here so they are covered on an installation without
the extra -- which is the one where a regression in them would otherwise go
unseen until somebody installed it.

The claims worth stating out loud, because each has cost this codebase
something:

* an annotation or a model score must NOT land in ``X``, where scanpy would
  scale and cluster it as a measurement;
* ``obs_names`` is spaCR's own object key, and two rows claiming one key is
  an error rather than a silent deduplication;
* every count in the NaN report is measured on the matrix the policy was
  GIVEN, so ``frac_missing`` can never exceed 1.0 -- it once reported
  114.3%;
* an embedding carrying the key columns is aligned BY KEY; a positional
  take attaches the wrong point to every object after the first gap.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import schema
from spacr.anndata_export import (
    CONDITION_FALLBACK, DEFAULT_CONDITION_MAP, DEFAULT_TABLES, NAN_DROP_FEATURES,
    NAN_DROP_OBJECTS, NAN_KEEP, NAN_MEAN, NAN_POLICIES, NAN_ZERO,
    DuplicateObjectKeys, ExportResult, _align_embedding, _apply_nan_policy,
    _available_tables, _build_obs, _build_var, _filter_record, _h5ad_safe,
    _label_columns, _measurement_units, _obsm_name, _project_root,
    _read_frame, _redundant_identity_columns, _relationships, _run_id_from_db,
    _source_table, _warn_about_missing, _worst_features,
    anndata_export_settings, default_out_path, feature_columns,
    resolve_db_path, run_anndata_export,
)
from spacr.selection import CategoryFilter, DataFilter, RangeFilter, Selection


# ---------------------------------------------------------------------------
# Fixtures: real sqlite, written the way a measurements database is
# ---------------------------------------------------------------------------

def _write_db(path, tables):
    with sqlite3.connect(str(path)) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return str(path)


def _cells(n=4, plate="plate1"):
    return pd.DataFrame({
        "plateID": [plate] * n,
        "rowID": [f"r{i + 1}" for i in range(n)],
        "columnID": ["c1", "c2"] * (n // 2) if n % 2 == 0 else ["c1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": range(1, n + 1),
        "cell_area": np.linspace(100.0, 400.0, n),
        "cell_channel_1_mean_intensity": np.linspace(1.0, 4.0, n),
        "measurement_units": ["um"] * n,
    })


@pytest.fixture
def cells():
    return _cells()


# ---------------------------------------------------------------------------
# ExportResult
# ---------------------------------------------------------------------------

def _result(**kwargs):
    base = dict(path="", n_obs=10, n_vars=5, n_obs_before_filter=10)
    base.update(kwargs)
    return ExportResult(**base)


def test_the_counted_shape_is_recorded_when_the_module_built_the_record():
    result = _result(n_obs=6, n_vars=3, n_obs_counted=10, n_vars_counted=5)
    assert result.counted_shape == (10, 5)


def test_an_older_record_reconstructs_the_shape_from_the_drops():
    """``drop_objects`` removes rows and ``drop_features`` columns; adding
    them back recovers the matrix the policy was given."""
    result = _result(n_obs=6, n_vars=3, dropped_objects=4,
                     dropped_features=("a", "b"))
    assert result.counted_shape == (10, 5)


def test_the_missing_fraction_divides_by_the_matrix_it_was_counted_in():
    """It once read 114.3%: the pre-policy count over the post-policy shape."""
    result = _result(n_obs=6, n_vars=3, dropped_objects=4, n_missing=24)
    assert result.counted_shape == (10, 3)
    assert result.frac_missing == pytest.approx(24 / 30)
    assert result.frac_missing <= 1.0


def test_an_empty_matrix_is_not_a_division_by_zero():
    assert _result(n_obs=0, n_vars=0, n_obs_before_filter=0,
                   n_missing=0).frac_missing == 0.0


def test_a_plain_description_is_the_shape_and_where_it_went():
    assert _result(path="/data/out.h5ad").describe() == (
        "10 objects x 5 features -> /data/out.h5ad")
    assert _result().describe() == "10 objects x 5 features (in memory)"


def test_the_description_charges_each_loss_to_the_stage_that_caused_it():
    """Charging the policy's row drops to the filter read one loss as two."""
    text = _result(
        path="/data/out.h5ad", n_obs=6, n_vars=3, n_obs_before_filter=20,
        n_obs_counted=10, n_vars_counted=5, nan_policy=NAN_DROP_OBJECTS,
        n_missing=24, n_infinite=2, dropped_objects=4,
        dropped_features=("a", "b"), obsm_keys=("X_umap",),
        artifact_id="art-1").describe()

    assert "filtered from 20 objects (10 removed)" in text
    assert "24 missing values (48.0% of the 10 x 5 matrix" in text
    assert "2 non-finite values treated as missing" in text
    assert "dropped objects (nan_policy 'drop_objects'): 4" in text
    assert "obsm: X_umap" in text
    assert "artifact art-1" in text


def test_a_long_list_of_dropped_features_is_summarised():
    text = _result(nan_policy=NAN_DROP_FEATURES,
                   dropped_features=tuple(f"f{i}" for i in range(8))).describe()
    assert "f0, f1, f2, f3, f4 +3" in text


def test_the_over_clause_is_left_out_when_nothing_was_dropped():
    text = _result(n_obs=10, n_vars=5, n_obs_counted=10, n_vars_counted=5,
                   n_missing=5).describe()
    assert "5 missing values (10.0%), policy 'keep'" in text
    assert "the policy was given" not in text


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_a_file_that_is_not_there_holds_no_tables(tmp_path):
    assert _available_tables(tmp_path / "never_written.db") == ()


def test_the_tables_are_reported_in_name_order(tmp_path, cells):
    path = _write_db(tmp_path / "m.db", {"cell": cells, "nucleus": cells})
    assert _available_tables(path) == ("cell", "nucleus")


def test_an_empty_database_is_named_and_refused(tmp_path):
    sqlite3.connect(str(tmp_path / "empty.db")).close()
    with pytest.raises(ValueError, match="holds no tables"):
        _read_frame(str(tmp_path / "empty.db"), DEFAULT_TABLES, None)


def test_a_single_table_that_is_not_there_lists_the_ones_that_are(tmp_path,
                                                                  cells):
    path = _write_db(tmp_path / "m.db", {"cell": cells})
    with pytest.raises(ValueError) as caught:
        _read_frame(path, DEFAULT_TABLES, "nucleus")
    assert "'nucleus' is not in" in str(caught.value)
    assert "cell" in str(caught.value)


def test_a_single_table_export_is_stamped_with_its_object_type(tmp_path,
                                                               cells):
    """Two per-table exports index the same object two ways without it: each
    mask is labelled from 1 independently."""
    path = _write_db(tmp_path / "m.db", {"cell": cells, "nucleus": cells})

    frame, read = _read_frame(path, DEFAULT_TABLES, "nucleus")

    assert read == ("nucleus",)
    assert set(frame[schema.OBJECT_TYPE_COLUMN if hasattr(schema, "OBJECT_TYPE_COLUMN")
                     else "object_type"]) == {"nucleus"}
    assert len(frame) == len(cells)


def test_a_database_with_none_of_the_wanted_tables_is_refused(tmp_path,
                                                              cells):
    path = _write_db(tmp_path / "m.db", {"something_else": cells})
    with pytest.raises(ValueError, match="none of"):
        _read_frame(path, ("cell", "nucleus"), None)


def test_a_join_with_no_cell_table_says_to_export_one_table_on_its_own(
        tmp_path, cells):
    """The joined export is anchored on 'cell'; without it there is no anchor."""
    path = _write_db(tmp_path / "m.db", {"nucleus": cells})
    with pytest.raises(ValueError) as caught:
        _read_frame(path, ("cell", "nucleus"), None)
    assert "anchored on the 'cell' table" in str(caught.value)
    assert "single_table=" in str(caught.value)


def test_a_join_that_returns_nothing_names_the_doctor_command(tmp_path,
                                                              cells,
                                                              monkeypatch):
    from spacr import io

    path = _write_db(tmp_path / "m.db", {"cell": cells})
    monkeypatch.setattr(io, "_read_and_join_tables",
                        lambda *a, **k: None)

    with pytest.raises(ValueError) as caught:
        _read_frame(path, ("cell",), None)
    assert "spacr doctor --db" in str(caught.value)


def test_a_joined_export_is_anchored_on_the_cell_table(tmp_path, cells):
    path = _write_db(tmp_path / "m.db", {"cell": cells})
    frame, read = _read_frame(path, ("cell", "nucleus"), None)
    assert read == ("cell",)
    assert len(frame) == len(cells)


# ---------------------------------------------------------------------------
# measurement_units
# ---------------------------------------------------------------------------

def test_a_legacy_frame_with_no_units_column_states_none(cells):
    assert _measurement_units(cells.drop(columns=["measurement_units"])) is None


def test_one_calibration_is_reported(cells):
    assert _measurement_units(cells) == "um"


def test_two_calibrations_are_reported_as_none_not_as_a_majority_vote(cells):
    """``parse_column`` then states the condition instead of asserting a unit
    that is wrong for some of the rows."""
    mixed = cells.copy()
    mixed.loc[mixed.index[:2], "measurement_units"] = "px"
    assert _measurement_units(mixed) is None


# ---------------------------------------------------------------------------
# The X / obs boundary
# ---------------------------------------------------------------------------

def test_a_model_score_is_not_a_feature(cells):
    """Offering a model column as an annotator gave a real database
    kappa = -0.004; putting one in X would have scanpy cluster on it."""
    frame = cells.copy()
    frame["pred"] = [0.1, 0.9, 0.2, 0.8]
    frame["cell_png_prediction"] = [1, 0, 1, 0]

    _annotations, predictions = _label_columns(frame, None)

    assert "pred" in predictions or "cell_png_prediction" in predictions
    for column in predictions:
        assert column not in feature_columns(frame)


def test_a_human_annotation_is_recognised_from_the_database(tmp_path, cells):
    png = pd.DataFrame({
        "plateID": ["plate1"] * 4, "rowID": [f"r{i + 1}" for i in range(4)],
        "columnID": ["c1"] * 4, "fieldID": ["f1"] * 4,
        "cell_id": [f"plate1_r{i + 1}_c1_f1_{i + 1}" for i in range(4)],
        "png_path": [f"/crops/{i}.png" for i in range(4)],
        "test": [1, 0, 1, 0],
    })
    path = _write_db(tmp_path / "m.db", {"cell": cells, "png_list": png})
    frame = cells.copy()
    frame["test"] = [1, 0, 1, 0]

    annotations, _predictions = _label_columns(frame, path)

    assert "test" in annotations
    assert "test" not in feature_columns(frame, db_path=path)


def test_a_database_that_cannot_be_read_costs_the_hint_not_the_export(cells,
                                                                      tmp_path):
    """Losing the whole export over a missing optional table would be worse."""
    annotations, _predictions = _label_columns(
        cells, str(tmp_path / "never_written.db"))
    assert annotations == ()
    assert "cell_area" in feature_columns(cells,
                                          db_path=str(tmp_path / "gone.db"))


def test_identity_and_provenance_columns_are_never_features(cells):
    features = feature_columns(cells)
    assert "cell_area" in features
    assert "cell_channel_1_mean_intensity" in features
    for column in ("plateID", "rowID", "columnID", "fieldID", "object_label",
                   "measurement_units"):
        assert column not in features


def test_text_columns_and_the_cluster_label_are_never_features(cells):
    frame = cells.copy()
    frame["cluster"] = [0, 1, 0, 1]
    frame["note"] = ["a", "b", "c", "d"]
    features = feature_columns(frame)
    assert "cluster" not in features
    assert "note" not in features


def test_the_caller_can_exclude_a_column_by_name(cells):
    assert "cell_area" not in feature_columns(cells, exclude=["cell_area"])


# ---------------------------------------------------------------------------
# var
# ---------------------------------------------------------------------------

def test_a_column_is_attributed_to_the_table_it_came_out_of():
    from spacr.feature_dict import describe_columns

    tables = ("cell", "nucleus")
    entry = describe_columns(["nucleus_area"], None)[0]
    assert _source_table("nucleus_area", entry, tables) == "nucleus"


def test_a_join_suffix_attributes_a_column_when_the_dictionary_cannot():
    class _Entry:
        object_type = None
        object_type_2 = None

    assert _source_table("mystery_nucleus", _Entry(), ("cell", "nucleus")) == (
        "nucleus")


def test_a_count_column_is_attributed_to_the_child_it_counts():
    class _Entry:
        object_type = None
        object_type_2 = None

    assert _source_table("count_pathogen", _Entry(),
                         ("cell", "pathogen")) == "pathogen"


def test_an_unattributable_column_falls_back_to_the_anchor():
    class _Entry:
        object_type = None
        object_type_2 = None

    assert _source_table("mystery", _Entry(), ("cell",)) == "cell"
    assert _source_table("mystery", _Entry(), ()) == ""


def test_var_documents_every_feature_and_counts_its_missing_values(cells):
    frame = cells.copy()
    frame.loc[frame.index[:2], "cell_area"] = np.nan
    frame["cell_channel_1_mean_intensity"] = [1.0, np.inf, 3.0, 4.0]
    features = ["cell_area", "cell_channel_1_mean_intensity"]

    var = _build_var(features, frame, ("cell", "nucleus"), "cell", "um")

    assert list(var.index) == features
    assert var.loc["cell_area", "n_missing"] == 2
    assert var.loc["cell_area", "frac_missing"] == pytest.approx(0.5)
    assert var.loc["cell_channel_1_mean_intensity", "n_infinite"] == 1
    assert var.loc["cell_area", "measurement_units"] == "um"
    assert str(var["family"].dtype) == "category"
    assert str(var["source_table"].dtype) == "category"


def test_an_aggregated_child_column_is_flagged_as_one(cells):
    frame = cells.copy()
    frame["nucleus_area"] = [10.0, 20.0, 30.0, 40.0]

    var = _build_var(["nucleus_area"], frame,
                     ("cell", "nucleus"), "cell", None)

    assert bool(var.loc["nucleus_area", "is_aggregated"]) is True


def test_a_child_anchored_export_aggregates_nothing(cells):
    frame = cells.copy()
    frame["nucleus_area"] = [10.0, 20.0, 30.0, 40.0]

    var = _build_var(["nucleus_area"], frame,
                     ("nucleus",), "nucleus", None)

    assert bool(var.loc["nucleus_area", "is_aggregated"]) is False


def test_var_on_an_empty_frame_reports_no_missing_fraction(cells):
    empty = cells.iloc[0:0]
    var = _build_var(["cell_area"], empty, ("cell",), "cell", None)
    assert var.loc["cell_area", "frac_missing"] == 0.0


# ---------------------------------------------------------------------------
# The join's suffixed identity copies
# ---------------------------------------------------------------------------

def test_a_suffixed_identity_copy_is_dropped_when_the_real_one_is_there():
    """``object_label_nucleus`` is the MEAN of the child labels -- a number
    with no referent at all."""
    columns = ["plateID", "plateID_nucleus", "object_label",
               "object_label_nucleus", "nucleus_area_nucleus"]

    drop = _redundant_identity_columns(columns, ("nucleus",))

    assert drop == ["plateID_nucleus", "object_label_nucleus"]
    # A real measurement keeps its suffix.
    assert "nucleus_area_nucleus" not in drop


def test_a_suffixed_copy_with_no_unsuffixed_column_is_kept():
    assert _redundant_identity_columns(["plateID_nucleus"], ("nucleus",)) == []


def test_nothing_is_dropped_when_nothing_was_joined():
    assert _redundant_identity_columns(["plateID", "plateID_nucleus"], ()) == []


# ---------------------------------------------------------------------------
# obs
# ---------------------------------------------------------------------------

def test_obs_is_indexed_by_spacrs_own_object_key(cells):
    from spacr.selection import object_keys

    obs = _build_obs(cells, ["cell_area"], (), (), timelapse=False,
                     condition_map=None, condition_column="columnID")

    assert list(obs.index) == list(object_keys(cells))
    assert obs.index.name == "object_key"
    assert "cell_area" not in obs.columns
    assert "plateID" in obs.columns


def test_two_rows_claiming_one_key_is_an_error_not_a_deduplication(cells):
    """Deduplicating changes the numbers, and is the caller's decision."""
    doubled = pd.concat([cells, cells.iloc[[0]]], ignore_index=True)

    with pytest.raises(DuplicateObjectKeys) as caught:
        _build_obs(doubled, ["cell_area"], (), (), timelapse=False,
                   condition_map=None, condition_column="columnID")

    assert "1 of 5 rows repeat an object key" in str(caught.value)
    assert "pass timelapse=True" in str(caught.value)


def test_a_timelapse_frame_says_nothing_about_timelapse_in_its_error(cells):
    doubled = pd.concat([cells, cells.iloc[[0]]], ignore_index=True)
    doubled[schema.TIME_KEY] = 0

    with pytest.raises(DuplicateObjectKeys) as caught:
        _build_obs(doubled, ["cell_area"], (), (), timelapse=True,
                   condition_map=None, condition_column="columnID")

    assert "pass timelapse=True" not in str(caught.value)


def test_the_condition_map_labels_the_columns_it_names(cells):
    obs = _build_obs(cells, ["cell_area"], (), (), timelapse=False,
                     condition_map=DEFAULT_CONDITION_MAP,
                     condition_column="columnID")

    assert list(obs["condition"].astype(str)) == ["neg", "pos", "neg", "pos"]


def test_a_column_the_map_does_not_name_falls_back(cells):
    frame = cells.copy()
    frame["columnID"] = ["c9"] * len(frame)

    obs = _build_obs(frame, ["cell_area"], (), (), timelapse=False,
                     condition_map=DEFAULT_CONDITION_MAP,
                     condition_column="columnID")

    assert set(obs["condition"].astype(str)) == {CONDITION_FALLBACK}


def test_a_condition_column_that_is_not_there_adds_no_condition(cells):
    obs = _build_obs(cells, ["cell_area"], (), (), timelapse=False,
                     condition_map=DEFAULT_CONDITION_MAP,
                     condition_column="not_a_column")
    assert "condition" not in obs.columns


def test_low_cardinality_text_is_stored_categorical(cells):
    """On a million-object export it is a 40 MB obs against a 4 MB one."""
    frame = cells.copy()
    frame["test"] = [1, 0, 1, 0]

    obs = _build_obs(frame, ["cell_area"], ("test",), (), timelapse=False,
                     condition_map=None, condition_column="columnID")

    assert str(obs["plateID"].dtype) == "category"
    assert str(obs["test"].dtype) == "category"


def test_a_high_cardinality_column_stays_as_it_is():
    frame = pd.DataFrame({
        "plateID": [f"plate{i}" for i in range(3000)],
        "rowID": ["r1"] * 3000, "columnID": ["c1"] * 3000,
        "fieldID": ["f1"] * 3000, "object_label": range(3000),
        "cell_area": np.arange(3000, dtype=float),
    })
    obs = _build_obs(frame, ["cell_area"], (), (), timelapse=False,
                     condition_map=None, condition_column="columnID")
    assert str(obs["plateID"].dtype) != "category"


def test_the_caller_can_drop_columns_from_obs(cells):
    obs = _build_obs(cells, ["cell_area"], (), (), timelapse=False,
                     condition_map=None, condition_column="columnID",
                     drop_columns=["measurement_units", "not_a_column"])
    assert "measurement_units" not in obs.columns


# ---------------------------------------------------------------------------
# The NaN policies
# ---------------------------------------------------------------------------

def _matrix():
    """4 x 3, with NaN in two columns and two rows."""
    values = np.arange(12, dtype=float).reshape(4, 3)
    values[0, 0] = np.nan
    values[1, 0] = np.nan
    values[2, 2] = np.nan
    return values


FEATURES = ["a", "b", "c"]


def test_a_policy_that_is_not_one_lists_the_ones_there_are():
    with pytest.raises(ValueError) as caught:
        _apply_nan_policy(_matrix(), list(FEATURES), "guess")
    assert "guess" in str(caught.value)
    assert "keep" in str(caught.value)


def test_keeping_the_nan_reports_them_and_changes_nothing():
    matrix, features, keep_rows, mask, report = _apply_nan_policy(
        _matrix(), list(FEATURES), NAN_KEEP)

    assert np.isnan(matrix).sum() == 3
    assert features == FEATURES
    assert keep_rows.all()
    assert mask is None
    assert report["n_missing"] == 3
    assert report["n_objects_counted"] == 4
    assert report["n_features_counted"] == 3
    assert report["n_features_with_missing"] == 2
    assert report["n_objects_with_missing"] == 3
    assert report["worst_features"] == [["a", 2], ["c", 1]]


def test_a_matrix_with_no_nan_short_circuits_every_policy():
    clean = np.arange(12, dtype=float).reshape(4, 3)
    for policy in NAN_POLICIES:
        matrix, features, keep_rows, mask, report = _apply_nan_policy(
            clean.copy(), list(FEATURES), policy)
        assert report["n_missing"] == 0
        assert features == FEATURES
        assert keep_rows.all()
        assert mask is None
        assert report["worst_features"] == []


def test_dropping_features_removes_the_columns_and_names_them():
    matrix, features, keep_rows, mask, report = _apply_nan_policy(
        _matrix(), list(FEATURES), NAN_DROP_FEATURES)

    assert features == ["b"]
    assert matrix.shape == (4, 1)
    assert report["dropped_features"] == ["a", "c"]
    assert keep_rows.all()
    assert mask is None


def test_dropping_objects_removes_the_rows_and_counts_them():
    matrix, features, keep_rows, mask, report = _apply_nan_policy(
        _matrix(), list(FEATURES), NAN_DROP_OBJECTS)

    assert matrix.shape == (1, 3)
    assert list(keep_rows) == [False, False, False, True]
    assert report["dropped_objects"] == 3
    assert features == FEATURES


def test_zero_filling_records_the_mask_that_keeps_it_distinguishable():
    matrix, _features, _keep, mask, report = _apply_nan_policy(
        _matrix(), list(FEATURES), NAN_ZERO)

    assert not np.isnan(matrix).any()
    assert matrix[0, 0] == 0.0
    assert report["imputed"] is True
    assert mask is not None and mask.sum() == 3


def test_mean_filling_uses_each_features_own_mean():
    matrix, _features, _keep, mask, report = _apply_nan_policy(
        _matrix(), list(FEATURES), NAN_MEAN)

    # Column 'a' holds 6.0 and 9.0; the two NaN become their mean.
    assert matrix[0, 0] == pytest.approx(7.5)
    assert matrix[1, 0] == pytest.approx(7.5)
    assert report["imputed"] is True
    assert mask.sum() == 3


def test_an_all_nan_feature_means_zero_rather_than_nan():
    values = np.full((3, 2), np.nan)
    values[:, 1] = [1.0, 2.0, 3.0]

    matrix, _features, _keep, _mask, _report = _apply_nan_policy(
        values, ["all_nan", "b"], NAN_MEAN)

    assert list(matrix[:, 0]) == [0.0, 0.0, 0.0]


def test_the_worst_features_are_the_ones_with_the_most_missing():
    missing = np.array([[True, False, True], [True, False, False],
                        [True, False, True]])
    assert _worst_features(missing, ["a", "b", "c"]) == [["a", 3], ["c", 2]]
    assert _worst_features(missing, ["a", "b", "c"], limit=1) == [["a", 3]]


def test_a_matrix_with_nothing_missing_has_no_worst_features():
    assert _worst_features(np.zeros((2, 2), dtype=bool), ["a", "b"]) == []
    assert _worst_features(np.ones((2, 0), dtype=bool), []) == []


def test_the_warning_names_the_columns_and_the_scanpy_calls(capsys):
    _warn_about_missing({"n_missing": 12, "n_features_with_missing": 2,
                         "worst_features": [["a", 8], ["c", 4]]}, NAN_KEEP)

    printed = capsys.readouterr().out
    assert "12 missing values kept in X across 2 features" in printed
    assert "sc.pp.pca" in printed
    assert "a (8), c (4)" in printed


def test_no_warning_when_the_policy_dealt_with_them(capsys):
    _warn_about_missing({"n_missing": 12}, NAN_MEAN)
    _warn_about_missing({"n_missing": 0}, NAN_KEEP)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def test_an_embedding_name_is_normalised_to_the_scanpy_convention():
    assert _obsm_name("umap") == "X_umap"
    assert _obsm_name("X_pca") == "X_pca"


def test_a_keyed_embedding_is_aligned_by_key_not_by_position(cells):
    """A positional take attaches the wrong point after the first gap."""
    from spacr.selection import object_keys

    embedding = cells[list(_key_columns())].copy()
    embedding["umap1"] = [10.0, 20.0, 30.0, 40.0]
    embedding["umap2"] = [1.0, 2.0, 3.0, 4.0]
    # The export keeps rows 0 and 3 only.
    kept = object_keys(cells.iloc[[0, 3]])

    values = _align_embedding(embedding, pd.Index(kept), timelapse=False,
                              name="umap")

    assert values.shape == (2, 2)
    assert list(values[:, 0]) == [10.0, 40.0]


def _key_columns():
    from spacr.selection import OBJECT_KEY_COLUMNS
    return OBJECT_KEY_COLUMNS


def test_a_keyed_embedding_with_no_coordinates_is_refused(cells):
    embedding = cells[list(_key_columns())].copy()
    with pytest.raises(ValueError, match="no numeric coordinate columns"):
        _align_embedding(embedding, pd.Index(["a"]), timelapse=False,
                         name="umap")


def test_an_embedding_computed_on_another_population_is_named_and_refused(
        cells):
    from spacr.selection import object_keys

    embedding = cells.iloc[:2][list(_key_columns())].copy()
    embedding["umap1"] = [1.0, 2.0]

    with pytest.raises(ValueError) as caught:
        _align_embedding(embedding, pd.Index(object_keys(cells)),
                         timelapse=False, name="umap")

    assert "has no coordinates for 2 of the 4 exported objects" in str(
        caught.value)
    assert "recompute it on the filtered frame" in str(caught.value)


def test_a_bare_array_is_taken_positionally(cells):
    values = _align_embedding(np.arange(8, dtype=float).reshape(4, 2),
                              pd.Index(range(4)), timelapse=False, name="pca")
    assert values.shape == (4, 2)
    assert values.dtype == np.float32


def test_a_one_dimensional_embedding_becomes_one_column():
    values = _align_embedding([1.0, 2.0, 3.0], pd.Index(range(3)),
                              timelapse=False, name="pseudotime")
    assert values.shape == (3, 1)


def test_an_array_of_the_wrong_length_says_how_to_align_it_properly():
    with pytest.raises(ValueError) as caught:
        _align_embedding(np.zeros((2, 2)), pd.Index(range(5)),
                         timelapse=False, name="umap")
    assert "has 2 rows but the export has 5 objects" in str(caught.value)
    assert "aligned by object key" in str(caught.value)


# ---------------------------------------------------------------------------
# uns
# ---------------------------------------------------------------------------

def test_everything_h5ad_cannot_store_becomes_something_it_can():
    """Losing a finished export to a provenance field would be absurd."""
    assert _h5ad_safe(None) == ""
    assert _h5ad_safe("text") == "text"
    assert _h5ad_safe(True) is True
    assert _h5ad_safe(3) == 3
    assert _h5ad_safe(np.int64(3)) == 3
    assert _h5ad_safe({"a": None, "b": (1, 2)}) == {"a": "", "b": [1, 2]}
    assert _h5ad_safe([None, "x"]) == ["", "x"]
    assert sorted(_h5ad_safe({1, 2})) == [1, 2]
    array = np.arange(3)
    assert _h5ad_safe(array) is array
    assert _h5ad_safe(object()).startswith("<object object")


def test_no_filter_is_described_as_no_filter():
    record = _filter_record(None, None)
    assert record == {"description": "no filter", "clauses": [],
                      "selection_source": "", "selection_size": 0}


def test_a_range_clause_is_recorded_with_both_bounds():
    record = _filter_record(
        DataFilter().add(RangeFilter("cell_area", low=200.0)), None)

    clause = record["clauses"][0]
    assert clause["column"] == "cell_area"
    assert clause["type"] == "RangeFilter"
    assert clause["low"] == 200.0
    assert clause["high"] == ""


def test_a_category_clause_is_recorded_with_its_values():
    record = _filter_record(
        DataFilter().add(CategoryFilter("columnID", ["c1", "c2"])), None)

    clause = record["clauses"][0]
    assert clause["type"] == "CategoryFilter"
    assert clause["values"] == ["c1", "c2"]


def test_an_active_selection_is_recorded_with_its_source(cells):
    selection = Selection.from_frame(cells, source="umap lasso")
    record = _filter_record(None, selection)
    assert record["selection_source"] == "umap lasso"
    assert record["selection_size"] == len(cells)


def test_an_inactive_selection_records_nothing():
    record = _filter_record(None, Selection())
    assert record["selection_source"] == ""


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

def test_a_joined_export_records_the_children_it_aggregated():
    record = _relationships(("cell", "nucleus", "pathogen"), "cell",
                            joined=True)

    assert record["anchor"] == "cell"
    assert set(record["children"]) == {"nucleus", "pathogen"}
    assert record["children"]["nucleus"]["count_column"] == "count_nucleus"
    assert record["children"]["nucleus"]["aggregated"] is True
    assert "never obsp" in record["storage"]


def test_a_child_anchored_export_records_the_parent_key_instead():
    record = _relationships(("nucleus",), "nucleus", joined=False)

    assert record["children"] == {}
    assert record["parent"]["table"] == "cell"
    assert record["parent"]["aggregated"] is False


def test_an_anchor_with_no_parent_records_no_parent():
    record = _relationships(("cell",), "cell", joined=False)
    assert "parent" not in record


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_a_database_with_no_history_has_no_run_id(tmp_path, cells):
    path = _write_db(tmp_path / "m.db", {"cell": cells})
    assert _run_id_from_db(path) == ""


def test_the_most_recent_recorded_run_is_the_one_reported(tmp_path, cells,
                                                          monkeypatch):
    from spacr import io

    path = _write_db(tmp_path / "m.db", {"cell": cells})
    monkeypatch.setattr(io, "read_settings_history",
                        lambda db: [{"run_id": "run-1"}, {"run_id": ""},
                                    {"run_id": "run-2"}])
    assert _run_id_from_db(path) == "run-2"


def test_a_history_that_cannot_be_read_is_no_run_id(tmp_path, monkeypatch):
    from spacr import io

    monkeypatch.setattr(io, "read_settings_history",
                        lambda db: (_ for _ in ()).throw(
                            RuntimeError("no settings_history table")))
    assert _run_id_from_db(str(tmp_path / "m.db")) == ""


def test_a_history_with_no_run_ids_at_all_is_no_run_id(tmp_path, monkeypatch):
    from spacr import io

    monkeypatch.setattr(io, "read_settings_history", lambda db: [{"a": 1}])
    assert _run_id_from_db(str(tmp_path / "m.db")) == ""


# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

def test_a_project_root_is_two_directories_above_the_database(tmp_path):
    db = tmp_path / "project" / "measurements" / "measurements.db"
    assert _project_root(str(db), None) == str(tmp_path / "project")


def test_a_database_somewhere_else_is_its_own_root(tmp_path):
    db = tmp_path / "elsewhere" / "m.db"
    assert _project_root(str(db), None) == str(tmp_path / "elsewhere")


def test_an_explicit_project_wins(tmp_path):
    db = tmp_path / "project" / "measurements" / "measurements.db"
    assert _project_root(str(db), tmp_path / "other") == str(tmp_path / "other")


@pytest.mark.parametrize("suffix", [".db", ".sqlite", ".sqlite3", ".DB"])
def test_a_path_that_ends_in_a_database_is_taken_as_one(tmp_path, suffix):
    path = tmp_path / f"m{suffix}"
    assert resolve_db_path(path) == os.path.abspath(path)


def test_a_project_root_resolves_to_its_measurements_database(tmp_path):
    assert resolve_db_path(tmp_path) == str(
        tmp_path / "measurements" / "measurements.db")


def test_the_default_output_is_named_after_the_project(tmp_path):
    project = tmp_path / "exp1"
    assert default_out_path(project) == str(
        project / "results" / "exp1.h5ad")


def test_a_single_table_export_gets_its_own_file_name(tmp_path):
    project = tmp_path / "exp1"
    assert default_out_path(project, "nucleus") == str(
        project / "results" / "exp1_nucleus.h5ad")


def test_the_settings_fill_themselves_in_without_touching_the_callers_dict():
    given = {"src": "/data/exp1"}
    resolved = anndata_export_settings(given)

    assert given == {"src": "/data/exp1"}
    assert resolved["anndata_nan_policy"] == NAN_KEEP
    assert resolved["anndata_tables"] == list(DEFAULT_TABLES)
    assert resolved["anndata_register_artifact"] is True
    assert anndata_export_settings()["src"] == ""


def test_a_run_with_no_src_says_what_src_means():
    with pytest.raises(ValueError) as caught:
        run_anndata_export({})
    assert "anndata_export needs src" in str(caught.value)
    assert "measurements/measurements.db" in str(caught.value)


def test_a_comma_separated_table_list_is_taken_apart(tmp_path, monkeypatch):
    """A settings.csv round trip spells a list as one cell."""
    seen = {}

    import spacr.anndata_export as ax

    monkeypatch.setattr(ax, "export_anndata",
                        lambda db, out, **kw: seen.update(db=db, out=out, **kw))

    run_anndata_export({"src": str(tmp_path),
                        "anndata_tables": "cell, nucleus",
                        "anndata_row_limit": 0,
                        "anndata_compression": ""})

    assert seen["tables"] == ("cell", "nucleus")
    assert seen["compression"] is None
    assert seen["row_limit"] is None
    assert seen["single_table"] is None
    assert seen["out"] == default_out_path(str(tmp_path))


def test_an_explicit_output_and_single_table_are_passed_through(tmp_path,
                                                                monkeypatch):
    seen = {}

    import spacr.anndata_export as ax

    monkeypatch.setattr(ax, "export_anndata",
                        lambda db, out, **kw: seen.update(db=db, out=out, **kw))

    run_anndata_export({"src": str(tmp_path),
                        "anndata_out": str(tmp_path / "chosen.h5ad"),
                        "anndata_single_table": "nucleus",
                        "anndata_row_limit": 50,
                        "anndata_compute_umap": True})

    assert seen["out"] == str(tmp_path / "chosen.h5ad")
    assert seen["single_table"] == "nucleus"
    assert seen["row_limit"] == 50
    assert seen["compute_umap"] is True


# ---------------------------------------------------------------------------
# Registration seams
# ---------------------------------------------------------------------------

def test_the_settings_are_registered_once():
    from spacr.anndata_export import register_anndata_settings
    from spacr.settings import has_registered_defaults
    from spacr.anndata_export import APP_KEY

    register_anndata_settings()
    assert has_registered_defaults(APP_KEY) is True
    assert register_anndata_settings() is False


def test_the_app_is_registered_once():
    from spacr.anndata_export import APP_KEY, register_anndata_app

    register_anndata_app()
    from spacr.qt.app import APPS

    assert [row[0] for row in APPS].count(APP_KEY) == 1
    assert register_anndata_app() is False


# ---------------------------------------------------------------------------
# The optional extra itself
# ---------------------------------------------------------------------------

def test_the_missing_extra_is_an_import_error_with_the_install_line():
    """An ImportError subclass, so a caller guarding with ``except
    ImportError`` keeps working -- and the message, not a traceback out of
    anndata's own import machinery, is what reaches the user."""
    from spacr.anndata_export import AnnDataExtraMissing, require_anndata

    try:
        import anndata                                        # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("the anndata extra is installed here")

    with pytest.raises(AnnDataExtraMissing) as caught:
        require_anndata()

    assert isinstance(caught.value, ImportError)
    assert "pip install" in str(caught.value)


# ---------------------------------------------------------------------------
# png_list, attached
# ---------------------------------------------------------------------------

def _png(n=4, ids=None, extra=None):
    frame = pd.DataFrame({
        "plateID": ["plate1"] * n,
        "rowID": [f"r{i + 1}" for i in range(n)],
        "columnID": (["c1", "c2"] * (n // 2 + 1))[:n],
        "fieldID": ["f1"] * n,
        "cell_id": ids if ids is not None else [f"o{i + 1}" for i in range(n)],
        "png_path": [f"/crops/{i}.png" for i in range(n)],
    })
    for name, values in (extra or {}).items():
        frame[name] = values
    return frame


def test_the_annotation_and_score_columns_come_back_out_of_png_list(tmp_path,
                                                                    cells):
    """Without them ``obs`` has no label to train on, group by or colour a
    UMAP with."""
    from spacr.anndata_export import _attach_png_labels
    from spacr.selection import with_object_type

    png = _png(extra={"test": [1, 0, 1, 0], "score": [0.1, 0.9, 0.2, 0.8]})
    path = _write_db(tmp_path / "m.db", {"cell": cells, "png_list": png})
    frame = with_object_type(cells, "cell")

    out, added = _attach_png_labels(frame, path, "cell", timelapse=False)

    assert sorted(added) == ["score", "test"]
    assert list(out["test"]) == [1, 0, 1, 0]
    assert out["score"].iloc[1] == pytest.approx(0.9)


def test_a_png_list_of_only_metadata_columns_adds_nothing(tmp_path, cells):
    from spacr.anndata_export import _attach_png_labels
    from spacr.selection import with_object_type

    path = _write_db(tmp_path / "m.db", {"cell": cells, "png_list": _png()})

    out, added = _attach_png_labels(with_object_type(cells, "cell"), path,
                                    "cell", timelapse=False)

    assert added == []


def test_crops_with_no_readable_object_id_attach_nothing(tmp_path, cells):
    """``omulti``, ``onone``, ``error`` and NULL are ordinary outcomes."""
    from spacr.anndata_export import _attach_png_labels
    from spacr.selection import with_object_type

    png = _png(ids=["omulti", "onone", "error", None],
               extra={"test": [1, 0, 1, 0]})
    path = _write_db(tmp_path / "m.db", {"cell": cells, "png_list": png})

    out, added = _attach_png_labels(with_object_type(cells, "cell"), path,
                                    "cell", timelapse=False)

    assert added == []
    assert "test" not in out.columns


def test_two_crops_of_one_object_that_agree_attach_once(tmp_path, cells):
    from spacr.anndata_export import _attach_png_labels
    from spacr.selection import with_object_type

    png = _png(extra={"test": [1, 0, 1, 0]})
    duplicate = pd.concat([png, png.iloc[[0]]], ignore_index=True)
    path = _write_db(tmp_path / "m.db",
                     {"cell": cells, "png_list": duplicate})

    out, added = _attach_png_labels(with_object_type(cells, "cell"), path,
                                    "cell", timelapse=False)

    assert added == ["test"]
    assert list(out["test"]) == [1, 0, 1, 0]


def test_two_crops_of_one_object_that_disagree_attach_nothing(tmp_path,
                                                              cells):
    """Attaching a plausible wrong label is worse than attaching none."""
    from spacr.anndata_export import _attach_png_labels
    from spacr.selection import with_object_type

    png = _png(extra={"test": [1, 0, 1, 0]})
    conflicting = pd.concat([png, png.iloc[[0]]], ignore_index=True)
    conflicting.loc[conflicting.index[-1], "test"] = 0
    path = _write_db(tmp_path / "m.db",
                     {"cell": cells, "png_list": conflicting})

    out, added = _attach_png_labels(with_object_type(cells, "cell"), path,
                                    "cell", timelapse=False)

    assert added == []
    assert "test" not in out.columns


def test_a_timelapse_png_list_that_cannot_be_keyed_attaches_nothing(tmp_path,
                                                                    cells):
    """Attaching the wrong frame's label loses the experiment."""
    from spacr.anndata_export import _attach_png_labels
    from spacr.selection import with_object_type

    png = _png(extra={"test": [1, 0, 1, 0]})
    path = _write_db(tmp_path / "m.db", {"cell": cells, "png_list": png})

    out, added = _attach_png_labels(with_object_type(cells, "cell"), path,
                                    "cell", timelapse=True)

    assert added == []


# ---------------------------------------------------------------------------
# The remaining seams
# ---------------------------------------------------------------------------

def test_a_count_column_for_a_table_that_was_not_joined_still_names_its_child():
    class _Entry:
        object_type = None
        object_type_2 = None

    assert _source_table("count_nucleus", _Entry(), ("cell",)) == "nucleus"


def test_a_timelapse_export_stores_the_timepoint_categorically(cells):
    frame = cells.copy()
    frame[schema.TIME_KEY] = [0, 0, 1, 1]

    obs = _build_obs(frame, ["cell_area"], (), (), timelapse=True,
                     condition_map=None, condition_column="columnID")

    assert str(obs[schema.TIME_KEY].dtype) == "category"
    assert len(obs) == len(frame)


def test_the_umap_is_computed_through_the_reducer_the_umap_app_uses():
    """An exported ``X_umap`` and the app's plot are the same embedding."""
    from spacr.anndata_export import _compute_umap

    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 6))
    matrix[0, 0] = np.nan

    embedding = _compute_umap(matrix, [f"f{i}" for i in range(6)],
                              {"n_neighbors": 5, "min_samples": 2})

    assert embedding.shape == (40, 2)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()


def test_a_set_export_writes_nothing_when_no_object_table_is_present(tmp_path):
    from spacr.anndata_export import export_anndata_set

    path = _write_db(tmp_path / "m.db",
                     {"settings_history": pd.DataFrame({"run_id": ["r1"]})})

    assert export_anndata_set(path, tmp_path / "out") == {}
    assert os.path.isdir(tmp_path / "out")


def test_re_registering_the_app_replaces_the_row_rather_than_doubling_it():
    from spacr.anndata_export import APP_KEY, register_anndata_app
    from spacr.qt.app import APPS

    saved = list(APPS)
    try:
        assert register_anndata_app(replace=True) is True
        assert [row[0] for row in APPS].count(APP_KEY) == 1
    finally:
        # APPS is a process-global list a dozen other test modules read.
        APPS[:] = saved


def test_a_png_list_whose_keys_cannot_be_built_attaches_nothing(tmp_path,
                                                                cells):
    """Both sides of the reindex have to be keyed the same way: a key that
    cannot be built at all means no label can be attached honestly."""
    import spacr.anndata_export as ax
    from spacr.selection import FilterError, object_keys as real_keys, \
        with_object_type

    png = _png(extra={"test": [1, 0, 1, 0]})
    path = _write_db(tmp_path / "m.db", {"cell": cells, "png_list": png})

    def refuse(frame, *, timelapse=False, object_type=None):
        if object_type is not None:
            raise FilterError("this png_list cannot be keyed by object type")
        return real_keys(frame, timelapse=timelapse, object_type=object_type)

    import pytest as _pytest
    monkeypatch = _pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(ax, "object_keys", refuse)
        out, added = ax._attach_png_labels(
            with_object_type(cells, "cell"), path, "cell", timelapse=False)
    finally:
        monkeypatch.undo()

    assert added == []
    assert "test" not in out.columns


def test_a_png_list_that_does_not_carry_the_anchors_id_attaches_nothing(
        tmp_path, cells):
    """``png_list`` from a nucleus-only crop run has no ``cell_id`` to join on."""
    from spacr.anndata_export import _attach_png_labels
    from spacr.selection import with_object_type

    png = _png(extra={"test": [1, 0, 1, 0]}).drop(columns=["cell_id"])
    path = _write_db(tmp_path / "m.db", {"cell": cells, "png_list": png})

    out, added = _attach_png_labels(with_object_type(cells, "cell"), path,
                                    "cell", timelapse=False)

    assert added == []
    assert "test" not in out.columns
