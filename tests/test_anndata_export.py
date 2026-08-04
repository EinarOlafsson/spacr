"""``spacr.anndata_export`` against a database spaCR's own writers built.

Every database here comes out of :func:`spacr.utils._merge_and_save_to_database`,
:func:`spacr.utils.filepaths_to_database` and :func:`spacr.io._save_settings_to_db`
-- the same calls ``measure_crop`` makes -- for the reason
``tests/test_db_contract.py`` states: a hand-built table has whatever columns
the test author remembered, and the columns they forget are exactly the ones
the reader trips on. The join this export is built on
(:func:`spacr.io._read_and_join_tables`) suffixes colliding columns, averages
children onto their parent and attaches ``png_list`` -- none of which a
``to_sql`` of a tidy frame reproduces.

The assertions are against the *source*, never against the exporter's own
output: ``X`` is compared cell by cell with the values read back out of
SQLite, ``obs_names`` with :func:`spacr.selection.object_keys`, ``var`` with
:func:`spacr.feature_dict.parse_column`, and the filtered export with the
frame the same :class:`spacr.selection.DataFilter` produces in pandas.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import importlib
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import schema
from spacr.selection import (OBJECT_KEY_COLUMNS, CategoryFilter, DataFilter,
                             RangeFilter, Selection, object_keys)

anndata = pytest.importorskip(
    "anndata", reason="the AnnData export needs `pip install spacr[anndata]`")

from spacr import anndata_export as ax  # noqa: E402  (after importorskip)


# ---------------------------------------------------------------------------
# a project built by spaCR's writers
# ---------------------------------------------------------------------------

#: Two wells, two fields each. c1 / c2 so the condition map has both controls.
FIELDS = ("plate1_A01_1", "plate1_A01_2", "plate1_A02_1", "plate1_A02_2")

#: Cells per field.
N_OBJECTS = 5

OBJECT_TABLES = ("cell", "cytoplasm", "nucleus", "pathogen")


def measure_field(root, stem, *, n_objects=N_OBJECTS, sparse_pathogen=True):
    """Write one field through the real object-table writer.

    ``sparse_pathogen`` leaves the last cell of every field without a
    pathogen, which is how a real plate looks and is the *only* honest way
    to get NaN into the joined matrix: the join is a left join onto the cell
    table, so a cell with no pathogen row gets NaN in every ``pathogen_*``
    column. Fabricating a NaN by writing one would test pandas, not spaCR.
    """
    from spacr.utils import _merge_and_save_to_database

    labels = list(range(1, n_objects + 1))
    for table in OBJECT_TABLES:
        rows = labels
        if table == "pathogen" and sparse_pathogen:
            rows = labels[:-1]
        n = len(rows)
        morphology = pd.DataFrame({
            "label": rows,
            f"{table}_area": [100.0 + i for i in range(n)],
            f"{table}_perimeter": [40.0 + 2 * i for i in range(n)],
        })
        intensity = pd.DataFrame({
            "label": rows,
            f"{table}_channel_0_mean_intensity": [5.0 + i for i in range(n)],
            f"{table}_channel_1_mean_intensity": [50.0 - i for i in range(n)],
        })
        if table in schema.CHILD_OBJECT_TABLES:
            morphology["cell_id"] = np.asarray(rows, dtype=float)
        _merge_and_save_to_database(morphology, intensity, table, root, stem,
                                    "exp", False)


def write_crops(root, stem, *, n=N_OBJECTS, crop_mode="cell"):
    """Index crop file names through the real ``png_list`` writer."""
    from spacr.utils import filepaths_to_database

    folder = os.path.join(root, "data", f"{crop_mode}_png")
    os.makedirs(folder, exist_ok=True)
    paths = [os.path.join(folder, f"{stem}_{i + 1}.png") for i in range(n)]
    filepaths_to_database(paths, {"timelapse": False}, root, crop_mode)
    return paths


def db_of(root):
    return os.path.join(root, "measurements", "measurements.db")


def add_label_columns(db):
    """Add one human annotation column and one model column to ``png_list``.

    Exactly the way the Annotate app and the classifier do it -- an
    ``ALTER TABLE ... ADD COLUMN`` on ``png_list`` -- so
    :func:`spacr.agreement.annotation_columns` and ``_is_model_column`` see
    what they see in the field.
    """
    connection = sqlite3.connect(db)
    try:
        connection.execute("ALTER TABLE png_list ADD COLUMN infected INTEGER")
        connection.execute("ALTER TABLE png_list ADD COLUMN cv_predictions INTEGER")
        connection.execute("ALTER TABLE png_list ADD COLUMN pred REAL")
        # Keyed on png_path, NOT on rowid. png_list declares a `rowID`
        # column (the plate row, 'r1'), SQLite identifiers are
        # case-insensitive, and a declared column shadows the implicit
        # rowid -- so `WHERE rowid = ?` compares against the plate row and
        # every UPDATE rewrites the whole table. That is bug 1 of
        # tests/test_db_contract.py, and it bit this fixture first.
        paths = [r[0] for r in connection.execute(
            "SELECT png_path FROM png_list ORDER BY png_path")]
        assert len(set(paths)) == len(paths), "png_path is png_list's row key"
        for i, path in enumerate(paths):
            connection.execute(
                "UPDATE png_list SET infected=?, cv_predictions=?, pred=? "
                "WHERE png_path=?",
                (i % 2, (i + 1) % 2, 0.1 * (i % 10), path))
        connection.commit()
    finally:
        connection.close()


def build_project(root, *, fields=FIELDS, labels=True):
    """A project database with every writer that touches ``measurements.db``."""
    os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    for stem in fields:
        measure_field(root, stem)
        write_crops(root, stem)
    db = db_of(root)

    from spacr.io import _save_settings_to_db

    _save_settings_to_db({"src": os.path.join(root, "data"),
                          "stage": "measure", "experiment": "exp"})
    if labels:
        add_label_columns(db)
    return db


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """One real project, shared by the read-only assertions."""
    root = str(tmp_path_factory.mktemp("anndata_project"))
    return root, build_project(root)


@pytest.fixture()
def fresh_project(tmp_path):
    """A private project for tests that write files or a registry."""
    root = str(tmp_path / "project")
    return root, build_project(root)


@pytest.fixture(scope="module")
def joined_frame(project):
    """The frame the exporter starts from, read the same way it reads it.

    Stamped with ``cell`` because that is what the joined export is: one row
    per cell, with the children arriving as columns. Without the stamp the
    keys here would be the untyped ones spaCR wrote before the object type
    went into the key, and the comparisons below would be testing a different
    identity from the one the exporter now writes.
    """
    from spacr.io import _read_and_join_tables
    from spacr.selection import with_object_type

    _root, db = project
    return with_object_type(
        _read_and_join_tables(db, list(ax.DEFAULT_TABLES)), "cell")


@pytest.fixture(scope="module")
def exported(project):
    """One default export of the shared project."""
    _root, db = project
    adata, result = ax.build_anndata(db, verbose=False)
    return adata, result


# ---------------------------------------------------------------------------
# the source really is what we think it is
# ---------------------------------------------------------------------------

def test_the_writers_built_a_database_with_every_owned_table(project):
    _root, db = project
    connection = sqlite3.connect(db)
    try:
        present = {r[0] for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        connection.close()
    assert {"cell", "cytoplasm", "nucleus", "pathogen", "png_list"} <= present
    assert present & set(schema.OWNED_TABLES)


def test_the_join_really_produced_missing_pathogen_values(joined_frame):
    """The NaN this file tests is a real left-join miss, not a written NaN."""
    pathogen = [c for c in joined_frame.columns
                if c.startswith("pathogen_") and c.endswith("_area")]
    assert pathogen, "the join lost the pathogen features entirely"
    assert joined_frame[pathogen[0]].isna().sum() == len(FIELDS), (
        "one cell per field was written without a pathogen row; the left "
        "join should leave exactly that many NaN")


# ---------------------------------------------------------------------------
# X
# ---------------------------------------------------------------------------

def test_X_shape_matches_the_source(exported, joined_frame):
    adata, result = exported
    assert adata.n_obs == len(joined_frame) == len(FIELDS) * N_OBJECTS
    assert adata.n_vars == len(
        ax.feature_columns(joined_frame))
    assert (result.n_obs, result.n_vars) == adata.shape


def test_X_values_match_the_source_cell_for_cell(exported, joined_frame):
    adata, _ = exported
    for name in adata.var_names:
        expected = pd.to_numeric(joined_frame[name], errors="coerce")
        got = np.asarray(adata[:, name].X).ravel()
        np.testing.assert_allclose(
            got, expected.to_numpy(dtype=np.float32), rtol=0, atol=0,
            err_msg=f"X column {name!r} does not match the database")


def test_X_is_float32_by_default_and_float64_on_request(project):
    _root, db = project
    adata, _ = ax.build_anndata(db, verbose=False)
    assert adata.X.dtype == np.float32
    wide, _ = ax.build_anndata(db, dtype="float64", verbose=False)
    assert wide.X.dtype == np.float64


def test_X_holds_only_measurements(exported):
    """No identity, no annotation, no model output, no cluster label."""
    adata, _ = exported
    for name in adata.var_names:
        assert not schema.is_provenance_column(name), name
    assert "infected" not in set(adata.var_names)
    assert "cv_predictions" not in set(adata.var_names)
    assert "pred" not in set(adata.var_names)
    assert "count_nucleus" not in set(adata.var_names)


# ---------------------------------------------------------------------------
# obs
# ---------------------------------------------------------------------------

def test_obs_names_are_the_schema_object_key(exported, joined_frame):
    adata, _ = exported
    expected = object_keys(joined_frame)
    assert list(adata.obs_names) == list(expected)
    assert adata.obs_names.is_unique
    # And the key really is the schema's, not a lookalike.
    row = joined_frame.iloc[0]
    assert adata.obs_names[0] == schema.KEY_SEPARATOR.join(
        str(row[c]) for c in OBJECT_KEY_COLUMNS[:-1]
    ) + schema.KEY_SEPARATOR + schema.object_id(
        row[schema.OBJECT_LABEL_KEY], object_type="cell")
    # The type is IN the key, so a cell 1 and a nucleus 1 in one field are
    # two observations rather than one collision AnnData would refuse.
    assert adata.obs_names[0].endswith("_cell1")


def test_obs_carries_every_object_key_column(exported):
    adata, _ = exported
    for column in OBJECT_KEY_COLUMNS:
        assert column in adata.obs.columns, column


def test_obs_carries_the_metadata_a_downstream_user_asks_for(exported):
    adata, _ = exported
    for column in ("prcf", "file_name", "png_path", "count_nucleus",
                   "count_pathogen", "n_missing_features"):
        assert column in adata.obs.columns, column


def test_obs_carries_the_annotation_and_the_prediction_separately(exported):
    """spaCR already knows which column a human wrote and which a model did."""
    adata, _ = exported
    provenance = adata.uns["spacr"]
    assert "infected" in adata.obs.columns
    assert "infected" in list(provenance["annotation_columns"])
    assert set(provenance["prediction_columns"]) >= {"cv_predictions", "pred"}
    assert "cv_predictions" in adata.obs.columns
    assert "infected" not in list(provenance["prediction_columns"])


def test_the_table_join_drops_the_labels_and_the_export_puts_them_back(
        project, joined_frame, exported):
    """The gap this export had to close, pinned in both directions.

    ``spacr.io._read_and_join_tables`` takes six named columns off
    ``png_list`` and drops the rest -- which is every annotation column the
    Annotate app added and every score the classifier wrote. For an AnnData
    export those are among the most valuable columns in the database, so
    they are re-attached by object key.
    """
    for label in ("infected", "cv_predictions", "pred"):
        assert label not in joined_frame.columns, (
            "the join is expected to drop it; if it stopped dropping it, "
            "_attach_png_labels is now redundant")
    adata, result = exported
    for label in ("infected", "cv_predictions", "pred"):
        assert label in adata.obs.columns, label
    assert any("png_list" in note for note in result.warnings)


def test_the_attached_labels_match_png_list_row_for_row(project):
    _root, db = project
    adata, _ = ax.build_anndata(db, verbose=False)

    connection = sqlite3.connect(db)
    try:
        png = pd.read_sql(
            "SELECT plateID, rowID, columnID, fieldID, cell_id, infected "
            "FROM png_list", connection)
    finally:
        connection.close()
    from spacr.utils import object_label_from_png_id

    png[schema.OBJECT_LABEL_KEY] = object_label_from_png_id(
        png["cell_id"]).astype("int64")
    expected = png.set_index(
        object_keys(png, object_type="cell"))["infected"]
    got = adata.obs["infected"].astype(float)
    for key in adata.obs_names:
        assert float(got.loc[key]) == float(expected.loc[key]), key


def test_attaching_the_labels_can_be_turned_off(project):
    _root, db = project
    adata, _ = ax.build_anndata(db, attach_labels=False, verbose=False)
    assert "infected" not in adata.obs.columns
    assert "cv_predictions" not in adata.obs.columns
    assert "png_path" in adata.obs.columns, "the join's own columns remain"


def test_a_filter_can_name_an_attached_annotation_column(project):
    """"Export the cells I annotated as infected" -- the point of attaching."""
    _root, db = project
    adata, _ = ax.build_anndata(
        db, data_filter=DataFilter().add(CategoryFilter("infected", (1,))),
        verbose=False)
    assert 0 < adata.n_obs < len(FIELDS) * N_OBJECTS
    assert set(adata.obs["infected"].astype(float)) == {1.0}


def test_obs_condition_matches_spacr_map_condition(project):
    """The condition preset is the one ``spacr.utils.map_condition`` applies."""
    from spacr.utils import map_condition

    _root, db = project
    adata, _ = ax.build_anndata(
        db, condition_map=ax.DEFAULT_CONDITION_MAP, verbose=False)
    expected = [map_condition(value) for value in adata.obs[schema.COLUMN_KEY]]
    assert list(adata.obs["condition"]) == expected
    assert set(adata.obs["condition"]) == {"neg", "pos"}


def test_the_condition_preset_cannot_drift_from_map_condition():
    """A four-entry dict duplicated to avoid a torch import, pinned."""
    from spacr.utils import map_condition

    for value, label in ax.DEFAULT_CONDITION_MAP.items():
        assert map_condition(value) == label
    assert map_condition("c99") == ax.CONDITION_FALLBACK


def test_obs_drops_the_joins_redundant_identity_copies(exported):
    adata, _ = exported
    for noise in ("plateID_nucleus", "object_label_pathogen",
                  "rowID_cytoplasm", "file_name_nucleus"):
        assert noise not in adata.obs.columns, noise
    assert "plateID" in adata.obs.columns


def test_the_redundant_copies_can_be_kept(project):
    _root, db = project
    adata, _ = ax.build_anndata(
        db, drop_redundant_identity=False, verbose=False)
    assert "plateID_nucleus" in adata.obs.columns


def test_duplicate_object_keys_are_refused_rather_than_repaired(tmp_path):
    """spaCR's writers append, so one object can end up with two rows.

    The joined export never gets this far -- ``_read_and_join_tables``
    validates the merge cardinality first and raises
    ``MergeCardinalityError``, which is the better error and is spaCR's own.
    The single-table path reads the table directly, so this is where the
    duplicate reaches AnnData's unique-``obs_names`` requirement.
    """
    root = str(tmp_path / "dup")
    build_project(root, fields=("plate1_A01_1",), labels=False)
    # Measure the same field a second time, exactly as a re-run would.
    measure_field(root, "plate1_A01_1")

    with pytest.raises(ax.DuplicateObjectKeys) as excinfo:
        ax.build_anndata(db_of(root), single_table="cell", verbose=False)
    assert "unique obs_names" in str(excinfo.value)
    assert "plate1_r1_c1_f1_cell1" in str(excinfo.value)
    assert "will not guess" in str(excinfo.value)

    from spacr.io import MergeCardinalityError

    with pytest.raises(MergeCardinalityError):
        ax.build_anndata(db_of(root), verbose=False)


# ---------------------------------------------------------------------------
# var
# ---------------------------------------------------------------------------

def test_var_matches_the_feature_dictionary(exported):
    from spacr.feature_dict import parse_column

    adata, _ = exported
    units = adata.uns["spacr"]["measurement_units"] or None
    for name in adata.var_names:
        entry = parse_column(name, units)
        row = adata.var.loc[name]
        assert row["family"] == entry.family, name
        assert row["object_type"] == (entry.object_type or ""), name
        assert row["feature_key"] == (entry.key or ""), name
        assert row["description"] == (entry.description or ""), name
        assert row["unit"] == (entry.unit or ""), name
        assert row["computed_by"] == entry.computed_by, name
        assert row["channel"] == (-1 if entry.channel is None
                                  else entry.channel), name
        assert row["channel_scope"] == entry.channel_scope, name


def test_var_names_the_channel_and_the_object_of_every_feature(exported):
    adata, _ = exported
    intensity = adata.var[adata.var["family"] == "intensity"]
    assert len(intensity), "the fixture writes intensity features"
    assert set(intensity["channel"]) == {0, 1}
    assert set(adata.var["object_type"]) >= {"cell", "nucleus", "pathogen"}


def test_var_flags_the_aggregated_child_features(exported):
    """A nucleus feature on a cell row is a mean, and says so."""
    adata, _ = exported
    aggregated = adata.var[adata.var["is_aggregated"]]
    assert set(aggregated["source_table"]) == {"nucleus", "pathogen"}
    cell = adata.var[adata.var["source_table"] == "cell"]
    assert not cell["is_aggregated"].any()
    cytoplasm = adata.var[adata.var["source_table"] == "cytoplasm"]
    assert len(cytoplasm) and not cytoplasm["is_aggregated"].any(), (
        "cytoplasm is one-to-one with the cell; it is joined, not aggregated")


def test_var_counts_the_missing_values_per_feature(exported, joined_frame):
    adata, _ = exported
    for name in adata.var_names:
        expected = int(pd.to_numeric(
            joined_frame[name], errors="coerce").isna().sum())
        assert int(adata.var.loc[name, "n_missing"]) == expected, name
    pathogen = [n for n in adata.var_names if n.startswith("pathogen_")]
    assert all(adata.var.loc[n, "n_missing"] == len(FIELDS) for n in pathogen)


# ---------------------------------------------------------------------------
# NaN
# ---------------------------------------------------------------------------

def test_the_default_keeps_nan_and_reports_it(exported):
    adata, result = exported
    assert result.nan_policy == ax.NAN_KEEP
    assert result.n_missing > 0
    assert np.isnan(np.asarray(adata.X)).sum() == result.n_missing
    report = adata.uns["spacr"]["nan"]
    assert report["policy"] == ax.NAN_KEEP
    assert report["n_missing"] == result.n_missing
    assert report["n_objects_with_missing"] == len(FIELDS)
    assert not report["imputed"]
    worst = [name for name, _count in report["worst_features"]]
    assert all(name.startswith("pathogen_") for name in worst)
    assert "missing" not in adata.layers


def test_obs_counts_missing_features_per_object(exported):
    adata, _ = exported
    per_object = np.isnan(np.asarray(adata.X)).sum(axis=1)
    np.testing.assert_array_equal(
        adata.obs["n_missing_features"].to_numpy(), per_object)


def test_drop_features_removes_exactly_the_incomplete_columns(project):
    _root, db = project
    adata, result = ax.build_anndata(
        db, nan_policy=ax.NAN_DROP_FEATURES, verbose=False)
    assert not np.isnan(np.asarray(adata.X)).any()
    assert result.dropped_features
    assert all(name.startswith("pathogen_")
               for name in result.dropped_features)
    assert set(result.dropped_features).isdisjoint(set(adata.var_names))
    assert len(adata.var) == adata.n_vars


def test_drop_objects_removes_exactly_the_incomplete_rows(project):
    _root, db = project
    adata, result = ax.build_anndata(
        db, nan_policy=ax.NAN_DROP_OBJECTS, verbose=False)
    assert not np.isnan(np.asarray(adata.X)).any()
    assert result.dropped_objects == len(FIELDS)
    assert adata.n_obs == len(FIELDS) * N_OBJECTS - len(FIELDS)
    assert len(adata.obs) == adata.n_obs
    # The rows that went are the ones with no pathogen: the last of each field.
    assert not any(key.endswith(f"_cell{N_OBJECTS}")
                   for key in adata.obs_names)


@pytest.mark.parametrize("policy", [ax.NAN_ZERO, ax.NAN_MEAN])
def test_imputing_policies_record_what_they_invented(project, policy):
    _root, db = project
    adata, result = ax.build_anndata(db, nan_policy=policy, verbose=False)
    assert not np.isnan(np.asarray(adata.X)).any()
    assert adata.uns["spacr"]["nan"]["imputed"] is True
    assert "missing" in adata.layers, (
        "an imputed matrix that cannot be told from a measured one is a trap")
    mask = np.asarray(adata.layers["missing"], dtype=bool)
    assert mask.sum() == result.n_missing
    filled = np.asarray(adata.X)[mask]
    if policy == ax.NAN_ZERO:
        assert np.all(filled == 0.0)
    else:
        assert np.all(np.isfinite(filled))
        assert not np.all(filled == 0.0)


def test_an_unknown_nan_policy_is_refused_by_name(project):
    _root, db = project
    with pytest.raises(ValueError) as excinfo:
        ax.build_anndata(db, nan_policy="interpolate", verbose=False)
    assert "interpolate" in str(excinfo.value)
    assert "drop_features" in str(excinfo.value)


def test_infinities_are_treated_as_missing_and_counted(fresh_project):
    """+/-inf survives dropna and then destroys every scaled statistic."""
    root, db = fresh_project
    connection = sqlite3.connect(db)
    try:
        connection.execute(
            "UPDATE cell SET cell_area = 1e999 WHERE object_label = 1")
        connection.commit()
    finally:
        connection.close()

    adata, result = ax.build_anndata(db, verbose=False)
    assert result.n_infinite > 0
    column = np.asarray(adata[:, "cell_area"].X).ravel()
    assert np.isnan(column).sum() >= result.n_infinite
    assert not np.isinf(column).any()
    assert any("non-finite" in note for note in result.warnings)


# ---------------------------------------------------------------------------
# obsm
# ---------------------------------------------------------------------------

def _fake_embedding(frame):
    """A 2-D embedding carrying the object key columns, one row per object."""
    values = frame[list(OBJECT_KEY_COLUMNS)].copy()
    values["umap_1"] = np.arange(len(frame), dtype=float)
    values["umap_2"] = np.arange(len(frame), dtype=float) * -2.0
    return values


def test_obsm_X_umap_matches_the_source_embedding(project, joined_frame):
    _root, db = project
    embedding = _fake_embedding(joined_frame)
    adata, result = ax.build_anndata(
        db, embeddings={"umap": embedding}, verbose=False)
    assert "X_umap" in adata.obsm
    assert result.obsm_keys == ("X_umap",)
    np.testing.assert_allclose(
        adata.obsm["X_umap"],
        embedding[["umap_1", "umap_2"]].to_numpy(dtype=np.float32))


def test_a_keyed_embedding_is_aligned_by_key_not_by_position(project,
                                                             joined_frame):
    """The case a positional take gets silently wrong: a filtered export."""
    _root, db = project
    embedding = _fake_embedding(joined_frame).sample(
        frac=1.0, random_state=0)          # shuffled on purpose
    data_filter = DataFilter().add(
        CategoryFilter(schema.COLUMN_KEY, ("c2",)))
    adata, _ = ax.build_anndata(
        db, embeddings={"umap": embedding}, data_filter=data_filter,
        verbose=False)

    # The embedding carries only the key columns, so it keys its rows
    # UNTYPED -- which is what an embedding computed by any earlier release,
    # or by a caller that never states a type, looks like. It still names the
    # same objects, and the exporter resolves it by dropping the type rather
    # than reporting a population mismatch.
    from spacr.selection import untyped_object_key

    lookup = embedding.set_index(object_keys(embedding))
    expected = lookup.loc[[untyped_object_key(k) for k in adata.obs_names],
                          ["umap_1", "umap_2"]]
    np.testing.assert_allclose(
        adata.obsm["X_umap"], expected.to_numpy(dtype=np.float32))


def test_a_positional_embedding_of_the_right_length_is_accepted(project):
    _root, db = project
    n = len(FIELDS) * N_OBJECTS
    array = np.arange(n * 2, dtype=float).reshape(n, 2)
    adata, _ = ax.build_anndata(db, embeddings={"X_pca": array}, verbose=False)
    np.testing.assert_allclose(adata.obsm["X_pca"],
                               array.astype(np.float32))


def test_a_misaligned_embedding_is_refused_with_the_fix_named(project):
    _root, db = project
    with pytest.raises(ValueError) as excinfo:
        ax.build_anndata(db, embeddings={"umap": np.zeros((3, 2))},
                         verbose=False)
    message = str(excinfo.value)
    assert "3 rows" in message
    assert "object key" in message


def test_a_keyed_embedding_missing_filtered_objects_is_refused(project,
                                                               joined_frame):
    _root, db = project
    partial = _fake_embedding(joined_frame).iloc[:4]
    with pytest.raises(ValueError) as excinfo:
        ax.build_anndata(db, embeddings={"umap": partial}, verbose=False)
    assert "no coordinates" in str(excinfo.value)


def test_embedding_names_follow_the_scanpy_convention(project, joined_frame):
    _root, db = project
    embedding = _fake_embedding(joined_frame)
    adata, _ = ax.build_anndata(
        db, embeddings={"umap": embedding, "X_pca": embedding}, verbose=False)
    assert set(adata.obsm) == {"X_umap", "X_pca"}


@pytest.mark.slow
def test_compute_umap_uses_spacrs_own_reducer(project):
    pytest.importorskip("umap")
    _root, db = project
    adata, result = ax.build_anndata(
        db, compute_umap=True,
        umap_settings={"n_neighbors": 4, "min_samples": 2}, verbose=False)
    assert "X_umap" in adata.obsm
    assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)
    assert np.isfinite(adata.obsm["X_umap"]).all(), (
        "X still holds NaN; the reducer must be given an imputed copy")
    assert "X_umap" in result.obsm_keys
    assert adata.uns["spacr"]["umap"]["computed_by"].endswith(
        "reduction_and_clustering")
    # The imputation was for the reducer only.
    assert np.isnan(np.asarray(adata.X)).any()


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------

def test_a_filtered_export_holds_exactly_the_filtered_objects(project,
                                                              joined_frame):
    _root, db = project
    data_filter = DataFilter().add(
        CategoryFilter(schema.COLUMN_KEY, ("c1",))).add(
        RangeFilter("cell_area", low=101.0))

    adata, result = ax.build_anndata(
        db, data_filter=data_filter, verbose=False)
    expected = data_filter.apply(joined_frame)

    assert list(adata.obs_names) == list(object_keys(expected))
    assert adata.n_obs == len(expected) < len(joined_frame)
    assert result.n_obs_before_filter == len(joined_frame)
    np.testing.assert_allclose(
        np.asarray(adata[:, "cell_area"].X).ravel(),
        expected["cell_area"].to_numpy(dtype=np.float32))
    assert adata.uns["spacr"]["filter"]["description"] == data_filter.describe()
    columns = {c["column"] for c in adata.uns["spacr"]["filter"]["clauses"]}
    assert columns == {schema.COLUMN_KEY, "cell_area"}


def test_a_selection_narrows_to_exactly_its_keys(project, joined_frame):
    _root, db = project
    from spacr.selection import untyped_object_keys

    # Deliberately UNTYPED -- the keys a selection saved before the object
    # type went into the key. They still name the same objects, so an old
    # selection still exports the population it always meant.
    wanted = list(untyped_object_keys(joined_frame))[::4]
    selection = Selection.from_keys(wanted, source="umap")

    adata, _ = ax.build_anndata(db, selection=selection, verbose=False)
    assert set(adata.obs_names) == set(list(object_keys(joined_frame))[::4])
    assert adata.n_obs == len(wanted)
    assert adata.uns["spacr"]["filter"]["selection_source"] == "umap"
    assert adata.uns["spacr"]["filter"]["selection_size"] == len(wanted)


def test_a_filter_and_a_selection_compose(project, joined_frame):
    _root, db = project
    data_filter = DataFilter().add(
        CategoryFilter(schema.COLUMN_KEY, ("c1",)))
    inside = data_filter.apply(joined_frame)
    keys = list(object_keys(inside))[:3] + list(object_keys(joined_frame))[-2:]

    adata, _ = ax.build_anndata(
        db, data_filter=data_filter,
        selection=Selection.from_keys(keys), verbose=False)
    assert list(adata.obs_names) == list(object_keys(inside))[:3]


def test_an_empty_filter_result_is_an_answer_not_an_error(project):
    _root, db = project
    adata, result = ax.build_anndata(
        db, data_filter=DataFilter().add(
            CategoryFilter(schema.PLATE_KEY, ("no_such_plate",))),
        verbose=False)
    assert adata.n_obs == 0
    assert adata.n_vars > 0, "var still describes the schema"
    assert result.n_obs_before_filter == len(FIELDS) * N_OBJECTS


def test_row_limit_caps_the_export_and_says_it_is_a_cap(project):
    _root, db = project
    adata, result = ax.build_anndata(db, row_limit=7, verbose=False)
    assert adata.n_obs == 7
    assert result.n_obs_before_filter == len(FIELDS) * N_OBJECTS
    assert any("row_limit" in note for note in result.warnings)


def test_a_filter_naming_an_absent_column_raises_rather_than_widening(project):
    from spacr.selection import FilterError

    _root, db = project
    with pytest.raises(FilterError):
        ax.build_anndata(
            db, data_filter=DataFilter().add(RangeFilter("no_column", low=1)),
            verbose=False)


# ---------------------------------------------------------------------------
# uns provenance
# ---------------------------------------------------------------------------

def test_uns_records_the_provenance_of_the_export(project):
    from spacr.artifacts import settings_hash
    from spacr.version import get_version

    root, db = project
    settings = {"src": root, "experiment": "exp", "verbose": True}
    adata, _ = ax.build_anndata(db, settings=settings, run_id="run-xyz",
                                verbose=False)
    provenance = adata.uns["spacr"]
    assert provenance["spacr_version"] == get_version()
    assert provenance["settings_hash"] == settings_hash(settings)
    assert provenance["run_id"] == "run-xyz"
    assert provenance["source_database"] == os.path.abspath(db)
    assert set(provenance["source_tables"]) <= set(ax.DEFAULT_TABLES)
    assert provenance["object_key_columns"] == list(OBJECT_KEY_COLUMNS)
    assert provenance["anchor_object"] == "cell"
    assert provenance["n_objects"] == adata.n_obs
    assert adata.uns["spacr_settings"]["experiment"] == "exp"
    # `verbose` is cosmetic, so the material settings drop it -- which is
    # what makes the hash mean "could this change the numbers?".
    assert "verbose" not in adata.uns["spacr_settings"]


def test_uns_falls_back_to_the_run_id_the_database_recorded(project):
    _root, db = project
    adata, _ = ax.build_anndata(db, verbose=False)
    from spacr.io import read_settings_history

    history = read_settings_history(db)
    recorded = [h["run_id"] for h in history if h["run_id"]]
    assert adata.uns["spacr"]["run_id"] == (recorded[-1] if recorded else "")


def test_uns_reports_how_much_of_the_matrix_the_dictionary_explains(exported):
    adata, _ = exported
    coverage = adata.uns["spacr"]["feature_dictionary"]
    assert coverage["total"] == adata.n_vars
    assert coverage["explained"] == adata.n_vars, (
        "every feature the fixture writes is in the spaCR data dictionary")


# ---------------------------------------------------------------------------
# relationships
# ---------------------------------------------------------------------------

def test_the_parent_links_are_never_in_obsp(exported):
    adata, _ = exported
    assert not len(adata.obsp), (
        "obsp is an n_obs x n_obs relation among THIS AnnData's observations; "
        "the nuclei and pathogens are not observations of a cell-anchored "
        "export at all")


def test_the_joined_export_records_the_aggregation_it_performed(exported):
    adata, _ = exported
    relationships = adata.uns["spacr"]["relationships"]
    assert relationships["anchor"] == "cell"
    assert set(relationships["children"]) == {"nucleus", "pathogen"}
    for table, entry in relationships["children"].items():
        assert entry["aggregated"] is True
        assert entry["count_column"] == f"count_{table}"
        assert adata.obs[entry["count_column"]].notna().any()


def test_a_child_table_export_carries_the_parent_key_in_obs(project):
    """The link a downstream user can actually group by."""
    _root, db = project
    adata, _ = ax.build_anndata(db, single_table="nucleus", verbose=False)

    assert adata.n_obs == len(FIELDS) * N_OBJECTS
    assert "cell_id" in adata.obs.columns
    assert adata.obs["cell_id"].notna().all()
    relationships = adata.uns["spacr"]["relationships"]
    assert relationships["anchor"] == "nucleus"
    assert relationships["parent"]["table"] == "cell"
    assert relationships["parent"]["obs_column"] == (
        schema.OBJECT_TABLE_SCHEMAS["nucleus"].parent_column)
    assert relationships["parent"]["aggregated"] is False
    assert not len(adata.obsp)
    # And the features really are that one table's.
    assert set(adata.var["object_type"]) == {"nucleus"}
    assert not adata.var["is_aggregated"].any()


def test_a_child_export_keeps_the_granularity_the_join_destroys(project):
    """A nucleus export is per nucleus; the joined one is per cell."""
    _root, db = project
    child, _ = ax.build_anndata(db, single_table="pathogen", verbose=False)
    joined, _ = ax.build_anndata(db, verbose=False)
    assert child.n_obs == len(FIELDS) * (N_OBJECTS - 1)
    assert joined.n_obs == len(FIELDS) * N_OBJECTS
    assert not np.isnan(np.asarray(child.X)).any(), (
        "a pathogen table holds only pathogens that exist")


def test_an_unknown_single_table_is_refused_with_what_is_there(project):
    _root, db = project
    with pytest.raises(ValueError) as excinfo:
        ax.build_anndata(db, single_table="mitochondrion", verbose=False)
    assert "mitochondrion" in str(excinfo.value)
    assert "cell" in str(excinfo.value)


# ---------------------------------------------------------------------------
# writing, reading back, and the artifact registry
# ---------------------------------------------------------------------------

def test_the_written_file_reads_back_identically(fresh_project):
    root, db = fresh_project
    out = os.path.join(root, "results", "exp.h5ad")

    from spacr.io import _read_and_join_tables

    frame = _read_and_join_tables(db, list(ax.DEFAULT_TABLES))
    embedding_frame = _fake_embedding(frame)

    result = ax.export_anndata(
        db, out, embeddings={"umap": embedding_frame},
        condition_map=ax.DEFAULT_CONDITION_MAP, verbose=False)
    assert os.path.isfile(out)
    assert result.path == os.path.abspath(out)

    back = anndata.read_h5ad(out)
    assert back.shape == (result.n_obs, result.n_vars)
    assert list(back.obs_names) == list(object_keys(frame, object_type="cell"))
    assert list(back.var_names) == ax.feature_columns(frame, db_path=db)
    np.testing.assert_allclose(
        back.obsm["X_umap"],
        embedding_frame[["umap_1", "umap_2"]].to_numpy(dtype=np.float32))
    for name in back.var_names:
        np.testing.assert_allclose(
            np.asarray(back[:, name].X).ravel(),
            pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float32))
    assert back.uns["spacr"]["source_database"] == os.path.abspath(db)
    assert back.uns["spacr"]["relationships"]["anchor"] == "cell"
    assert set(back.obs["condition"]) == {"neg", "pos"}


def test_an_uncompressed_write_round_trips_too(fresh_project):
    root, db = fresh_project
    out = os.path.join(root, "results", "plain.h5ad")
    ax.export_anndata(db, out, compression=None, register=False, verbose=False)
    assert anndata.read_h5ad(out).n_obs == len(FIELDS) * N_OBJECTS


def test_the_export_registers_as_an_artifact_downstream_of_the_database(
        fresh_project):
    from spacr import artifacts, ports

    root, db = fresh_project
    upstream = artifacts.register(
        project=root, module="measure", kind=ports.MEASUREMENTS_DB,
        role="measurements", path=db, settings={"experiment": "exp"})

    out = os.path.join(root, "results", "exp.h5ad")
    result = ax.export_anndata(db, out, settings={"experiment": "exp"},
                               verbose=False)

    assert result.artifact_id
    records = artifacts.by_kind(ax.ANNDATA_KIND, project=root)
    assert [r.artifact_id for r in records] == [result.artifact_id]
    record = records[0]
    assert record.module == ax.APP_KEY
    assert record.path == os.path.abspath(out)
    assert upstream.artifact_id in record.inputs
    assert record.extra["n_obs"] == result.n_obs
    assert record.extra["source_database"] == os.path.abspath(db)

    downstream = artifacts.downstream_of(upstream, project=root)
    assert result.artifact_id in {a.artifact_id for a in downstream}


def test_registration_can_be_turned_off(fresh_project):
    from spacr import artifacts

    root, db = fresh_project
    out = os.path.join(root, "results", "exp.h5ad")
    result = ax.export_anndata(db, out, register=False, verbose=False)
    assert result.artifact_id == ""
    assert not artifacts.by_kind(ax.ANNDATA_KIND, project=root)


def test_export_anndata_set_writes_one_file_per_object_type(fresh_project):
    root, db = fresh_project
    out_dir = os.path.join(root, "results", "set")
    results = ax.export_anndata_set(db, out_dir, register=False, verbose=False)

    assert set(results) == {"cell", "nucleus", "pathogen", "cytoplasm"}
    for table, result in results.items():
        assert os.path.isfile(result.path)
        adata = anndata.read_h5ad(result.path)
        assert adata.uns["spacr"]["anchor_object"] == table
        assert set(adata.var["object_type"]) == {table}

    child = anndata.read_h5ad(results["nucleus"].path)
    parent = child.uns["spacr"]["relationships"]["parent"]
    assert parent["file"] == "cell.h5ad"
    assert parent["obs_column"] == "cell_id"
    assert os.path.isfile(os.path.join(out_dir, parent["file"]))


# ---------------------------------------------------------------------------
# the missing optional dependency
# ---------------------------------------------------------------------------

class _BlockAnnData:
    """A meta-path finder that makes ``import anndata`` fail."""

    def find_module(self, fullname, path=None):        # pragma: no cover
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "anndata" or fullname.startswith("anndata."):
            raise ModuleNotFoundError(f"No module named {fullname!r}",
                                      name=fullname)
        return None


@pytest.fixture()
def without_anndata(monkeypatch):
    """Make :mod:`anndata` unimportable for the duration of one test."""
    finder = _BlockAnnData()
    monkeypatch.setattr(sys, "meta_path", [finder] + list(sys.meta_path))
    for name in [n for n in sys.modules if n == "anndata"
                 or n.startswith("anndata.")]:
        monkeypatch.delitem(sys.modules, name)
    yield


def test_the_module_imports_without_the_extra(without_anndata):
    """`import spacr.anndata_export` must not need the extra."""
    module = importlib.reload(ax)
    assert module.ANNDATA_EXTRA == "anndata"
    assert callable(module.export_anndata)


def test_require_anndata_gives_the_install_line_not_a_traceback(
        without_anndata):
    with pytest.raises(ax.AnnDataExtraMissing) as excinfo:
        ax.require_anndata()
    message = str(excinfo.value)
    assert 'python -m pip install "spacr[anndata]"' in message
    assert "anndata" in message
    assert isinstance(excinfo.value, ImportError)


def test_the_export_entry_points_fail_the_same_friendly_way(project,
                                                            without_anndata,
                                                            tmp_path):
    _root, db = project
    with pytest.raises(ax.AnnDataExtraMissing) as excinfo:
        ax.build_anndata(db, verbose=False)
    assert 'pip install "spacr[anndata]"' in str(excinfo.value)

    with pytest.raises(ax.AnnDataExtraMissing):
        ax.export_anndata(db, str(tmp_path / "x.h5ad"), verbose=False)
    assert not (tmp_path / "x.h5ad").exists(), (
        "the missing-extra check must happen before anything is written")


def test_the_missing_message_names_the_module_that_was_missing(
        without_anndata):
    with pytest.raises(ax.AnnDataExtraMissing) as excinfo:
        ax.require_anndata()
    assert "missing module: anndata" in str(excinfo.value)


def test_anndata_is_imported_lazily_and_never_at_module_scope():
    """The packaging contract, asserted here too so it fails loudly."""
    import ast
    import pathlib

    source = pathlib.Path(ax.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module.split(".")[0]]
        else:
            continue
        assert "anndata" not in names, (
            "anndata is an extra; a module-scope import makes "
            "`import spacr.anndata_export` an ImportError for everyone who "
            "did not type it")


def test_setup_py_declares_the_extra_the_message_names():
    import ast
    import pathlib

    setup = pathlib.Path(ax.__file__).parents[2] / "setup.py"
    tree = ast.parse(setup.read_text(encoding="utf-8"))
    extras = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup":
            for keyword in node.keywords:
                if keyword.arg == "extras_require":
                    extras = ast.literal_eval(keyword.value)
    assert extras is not None
    assert ax.ANNDATA_EXTRA in extras
    assert any(spec.startswith("anndata") for spec in extras[ax.ANNDATA_EXTRA])


# ---------------------------------------------------------------------------
# registration seams
# ---------------------------------------------------------------------------

def test_the_settings_registered_through_the_seam():
    """Reached through `defaults_for`, which is what a settings panel calls."""
    from spacr import settings as settings_module

    assert settings_module.has_registered_defaults(ax.APP_KEY)
    resolved = settings_module.defaults_for(ax.APP_KEY)
    assert resolved["anndata_nan_policy"] == ax.NAN_KEEP
    assert resolved["anndata_dtype"] == "float32"
    for key in ax._TYPES:
        assert settings_module.expected_types.get(key) is ax._TYPES[key], key
        assert key in settings_module.tooltips, key
    assert settings_module.descriptions[ax.APP_KEY] == ax._DESCRIPTION


def test_the_defaults_factory_does_not_mutate_its_argument():
    original = {"src": "/data"}
    resolved = ax.anndata_export_settings(original)
    assert original == {"src": "/data"}
    assert resolved["src"] == "/data"
    assert resolved["anndata_compression"] == "gzip"


def test_registering_the_settings_twice_is_a_no_op():
    assert ax.register_anndata_settings() is False


# ---------------------------------------------------------------------------
# the headless entry point: fn(settings), which is what everything dispatches
# ---------------------------------------------------------------------------

def test_a_project_root_resolves_to_the_database_every_writer_leaves(tmp_path):
    """`src` is a project root everywhere else in spaCR, so it is here too."""
    root = tmp_path / "plate1"
    assert ax.resolve_db_path(str(root)) == str(
        root / "measurements" / "measurements.db")


@pytest.mark.parametrize("name", ["measurements.db", "other.sqlite",
                                  "x.SQLITE3"])
def test_a_path_that_is_already_a_database_is_taken_as_one(tmp_path, name):
    """Passing the .db directly is the notebook spelling; do not append."""
    db = tmp_path / name
    assert ax.resolve_db_path(str(db)) == str(db)


def test_the_default_output_is_named_after_the_project(tmp_path):
    """A folder of `export.h5ad` files is a folder nobody can tell apart."""
    root = tmp_path / "screen_A"
    out = ax.default_out_path(str(root))
    assert out == str(root / "results" / "screen_A.h5ad")
    # A one-table export is a DIFFERENT file, at a different granularity,
    # so it must not overwrite the joined one.
    assert ax.default_out_path(str(root), "nucleus") == str(
        root / "results" / "screen_A_nucleus.h5ad")


def test_run_anndata_export_writes_the_file_the_settings_describe(
        fresh_project):
    """The `fn(settings)` shape spacr-run and the Run button both call."""
    root, _db = fresh_project
    result = ax.run_anndata_export({"src": root, "anndata_compression": ""})

    assert result.path == os.path.join(root, "results",
                                       os.path.basename(root) + ".h5ad")
    assert os.path.isfile(result.path)
    assert result.n_obs > 0 and result.n_vars > 0

    anndata = ax.require_anndata()
    written = anndata.read_h5ad(result.path)
    assert written.shape == (result.n_obs, result.n_vars)


def test_run_anndata_export_honours_the_keys_the_panel_shows(fresh_project,
                                                             tmp_path):
    """Every setting the form draws has to reach the export, or the form is
    a set of switches that do nothing -- which is the bug the whole
    registration seam exists to stop."""
    root, _db = fresh_project
    out = tmp_path / "chosen.h5ad"
    result = ax.run_anndata_export({
        "src": root,
        "anndata_out": str(out),
        "anndata_single_table": "nucleus",
        "anndata_row_limit": 3,
        "anndata_dtype": "float64",
        "anndata_nan_policy": ax.NAN_ZERO,
        "anndata_register_artifact": False,
        "anndata_compression": "",
    })

    assert result.path == str(out) and os.path.isfile(out)
    assert result.n_obs == 3, "the row limit was ignored"
    assert result.nan_policy == ax.NAN_ZERO
    assert result.artifact_id == "", "registration was asked to be off"

    anndata = ax.require_anndata()
    written = anndata.read_h5ad(out)
    assert written.X.dtype == np.float64
    # single_table means one row per nucleus, not the join averaged onto
    # its parent cell -- the whole reason the setting exists.
    assert written.uns["spacr"]["anchor_object"] == "nucleus"


def test_a_comma_separated_table_list_survives_a_settings_csv_round_trip(
        fresh_project, tmp_path):
    """`--set anndata_tables=cell,nucleus` must not export a table called
    "cell,nucleus"; a settings.csv spells a list as one cell."""
    root, _db = fresh_project
    out = tmp_path / "two.h5ad"
    result = ax.run_anndata_export({
        "src": root, "anndata_out": str(out),
        "anndata_tables": "cell, nucleus",
        "anndata_compression": "",
    })
    assert result.n_obs > 0
    assert os.path.isfile(out)


def test_run_anndata_export_without_src_says_so_rather_than_guessing():
    with pytest.raises(ValueError) as excinfo:
        ax.run_anndata_export({})
    assert "src" in str(excinfo.value)


def test_the_registries_all_name_run_anndata_export():
    """cli, validate and the Qt bridge must dispatch to the same callable."""
    from spacr import cli
    from spacr.validate import APP_FUNCTIONS

    assert cli.MODULES[ax.APP_KEY].func_name == "run_anndata_export"
    assert cli.MODULES[ax.APP_KEY].module_name == "spacr.anndata_export"
    assert APP_FUNCTIONS[ax.APP_KEY].endswith("run_anndata_export")
    assert ax.APP_KEY not in cli.INTERACTIVE_ONLY, (
        "an app with a settings form and a real headless entry point must "
        "not also claim it cannot run headless")
    assert cli.resolve_module("h5ad").key == ax.APP_KEY


def test_the_qt_app_row_registers_when_there_is_a_gui_to_register_with():
    """No Qt app module in this process means no row, and no import of one.

    The teardown re-registers rather than leaving the registry short. It
    used to end on ``unregister_app`` on the grounds that this test was
    what had put the row there; ``spacr.qt.app`` now names
    ``register_anndata_app`` in ``_SELF_REGISTERING_APPS``, so the row is
    part of the shipped registry and deleting it here would fail the
    whole-registry inventories in a different file entirely.
    """
    if "spacr.qt.app" not in sys.modules:
        assert ax.register_anndata_app() is False
        assert "spacr.qt.app" not in sys.modules, (
            "registering the app row must not import the Qt app module into "
            "a headless export")
        return
    app = sys.modules["spacr.qt.app"]
    ax.register_anndata_app(replace=True)
    assert any(row[0] == ax.APP_KEY for row in app.APPS)
    # `spacr.qt.maturity` reassessed every alpha module against the
    # evidence in the repository and this one no longer qualifies; the
    # reason is recorded beside the decision. Applied here because the
    # promotions land in `register_self_registering_modules`, which every
    # launch calls but a bare test process may not have. `apply` alone,
    # not the whole registration pass: it touches only APP_STAGE, so it
    # cannot re-register a module a test has deliberately removed.
    from spacr.qt import maturity
    maturity.apply()
    assert app.APP_STAGE[ax.APP_KEY] == app.STAGE_BETA
    # `replace=True` re-registered it in place, and row order is what the
    # sidebar walks: a row filed at the end rather than beside its own
    # section draws that section's heading a second time. So check the
    # blocks are still contiguous, not just that the row came back.
    row = next(r for r in app.APPS if r[0] == ax.APP_KEY)
    assert row[3] == app.SECTION_EXPLORE
    blocks = [s for i, s in enumerate(r[3] for r in app.APPS)]
    runs = [s for i, s in enumerate(blocks) if i == 0 or blocks[i - 1] != s]
    assert len(runs) == len(set(runs)), f"a section is split in two: {runs}"


# ---------------------------------------------------------------------------
# the result record
# ---------------------------------------------------------------------------

def test_the_result_describes_what_happened():
    result = ax.ExportResult(path="/tmp/x.h5ad", n_obs=10, n_vars=4,
                             n_obs_before_filter=100, n_missing=8,
                             nan_policy=ax.NAN_DROP_FEATURES,
                             dropped_features=("a", "b"), dropped_objects=3,
                             obsm_keys=("X_umap",), artifact_id="abc")
    text = result.describe()
    assert "10 objects x 4 features" in text
    assert "filtered from 100" in text
    assert "drop_features" in text
    assert "X_umap" in text
    assert "abc" in text
    assert result.frac_missing == pytest.approx(8 / 40)


def test_frac_missing_of_an_empty_export_is_zero_not_a_zero_division():
    assert ax.ExportResult(path="", n_obs=0, n_vars=0,
                           n_obs_before_filter=0).frac_missing == 0.0


# ---------------------------------------------------------------------------
# refusals that name the fix
# ---------------------------------------------------------------------------

def test_a_database_that_is_not_a_measurements_db_is_refused(tmp_path):
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()
    with pytest.raises(ValueError) as excinfo:
        ax.build_anndata(str(empty), verbose=False)
    assert "measurements.db" in str(excinfo.value)


def test_a_database_without_a_cell_table_names_the_alternative(tmp_path):
    root = str(tmp_path / "nocell")
    build_project(root, fields=("plate1_A01_1",), labels=False)
    db = db_of(root)
    connection = sqlite3.connect(db)
    try:
        connection.execute("DROP TABLE cell")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError) as excinfo:
        ax.build_anndata(db, verbose=False)
    assert "single_table" in str(excinfo.value)
