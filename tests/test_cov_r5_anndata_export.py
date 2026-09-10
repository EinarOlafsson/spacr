"""The AnnData build and write, driven through a stand-in ``anndata``.

:mod:`spacr.anndata_export` is guarded end to end by
:func:`spacr.anndata_export.require_anndata`, so on a machine without the
optional extra installed *everything past that import* -- the whole of
:func:`build_anndata`, the write, the artifact registration and the
per-object-table set -- is unreachable. It is unreachable for the test
machine, not for the code: the module never touches ``anndata`` except
through the one name ``require_anndata`` returns, which is exactly the seam
this file uses. A recording stand-in is installed in ``sys.modules`` and the
real functions run against it.

The stand-in is not a silent yes-man. It enforces the three rules the real
container enforces and that this module's own code exists to satisfy:

* ``obs`` / ``var`` must be as long as ``X`` is wide and tall -- the
  alignment ``keep_rows`` and the dropped-feature reindex are there to keep;
* ``obs_names`` must be unique -- what :class:`DuplicateObjectKeys` guards;
* everything in ``uns`` must be HDF5-storable -- what :func:`_h5ad_safe`
  converts for, so a settings value that is not is caught by the write
  rather than by a reader months later.

Every assertion below is against the *source* -- the values in SQLite, the
keys :func:`spacr.selection.object_keys` produces, the frame the same
:class:`DataFilter` selects in pandas -- never against the exporter echoing
itself back.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import os
import pickle
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import anndata_export as ax
from spacr.selection import (OBJECT_KEY_COLUMNS, CategoryFilter, DataFilter,
                             RangeFilter, Selection, object_keys)


# ---------------------------------------------------------------------------
# the stand-in
# ---------------------------------------------------------------------------

def _assert_h5ad_storable(value, where):
    """Raise the way ``h5py`` does on something ``uns`` cannot hold.

    ``None``, a tuple, a set and an arbitrary object all reach the HDF5
    writer as an unsupported type; :func:`spacr.anndata_export._h5ad_safe`
    exists to turn them into something that is not, and a stand-in that
    accepted them would have retired that function silently.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: key {key!r} is not a string")
            _assert_h5ad_storable(item, f"{where}[{key!r}]")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_h5ad_storable(item, f"{where}[{index}]")
    elif not isinstance(value, (str, bool, int, float, np.integer,
                                np.floating, np.ndarray)):
        raise TypeError(
            f"{where}: {type(value).__name__} cannot be written to h5ad")


class AnnDataDouble:
    """Enough of ``anndata.AnnData`` to hold a spaCR export, and no more."""

    def __init__(self, X=None, obs=None, var=None, **_ignored):
        matrix = np.asarray(X)
        if matrix.ndim != 2:
            raise ValueError(f"X must be 2-D, got {matrix.ndim}-D")
        if obs is not None and len(obs) != matrix.shape[0]:
            raise ValueError(
                f"obs has {len(obs)} rows but X has {matrix.shape[0]}")
        if var is not None and len(var) != matrix.shape[1]:
            raise ValueError(
                f"var has {len(var)} rows but X has {matrix.shape[1]} columns")
        self.X = matrix
        self.obs = (pd.DataFrame(index=pd.RangeIndex(matrix.shape[0]))
                    if obs is None else obs)
        self.var = (pd.DataFrame(index=pd.RangeIndex(matrix.shape[1]))
                    if var is None else var)
        if not pd.Index(self.obs.index).is_unique:
            raise ValueError("obs_names must be unique")
        self.obsm = {}
        self.layers = {}
        self.uns = {}

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def var_names(self):
        return self.var.index

    @property
    def n_obs(self):
        return int(self.X.shape[0])

    @property
    def n_vars(self):
        return int(self.X.shape[1])

    @property
    def shape(self):
        return (self.n_obs, self.n_vars)

    def __len__(self):
        return self.n_obs

    def write_h5ad(self, path, compression=None):
        if compression not in (None, "gzip", "lzf"):
            raise ValueError(f"unknown compression {compression!r}")
        _assert_h5ad_storable(self.uns, "uns")
        with open(path, "wb") as handle:
            pickle.dump({"X": self.X, "obs": self.obs, "var": self.var,
                         "obsm": self.obsm, "layers": self.layers,
                         "uns": self.uns, "compression": compression}, handle)


def _read_double(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    adata = AnnDataDouble(X=payload["X"], obs=payload["obs"],
                          var=payload["var"])
    adata.obsm.update(payload["obsm"])
    adata.layers.update(payload["layers"])
    adata.uns.update(payload["uns"])
    adata.compression = payload["compression"]
    return adata


@pytest.fixture
def anndata_double(monkeypatch):
    """Install the stand-in as ``anndata`` for the duration of one test."""
    module = types.ModuleType("anndata")
    module.AnnData = AnnDataDouble
    module.read_h5ad = _read_double
    module.__version__ = "0+spacr.test.double"
    monkeypatch.setitem(sys.modules, "anndata", module)
    return module


# ---------------------------------------------------------------------------
# databases
# ---------------------------------------------------------------------------

def _write_db(path, tables):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with sqlite3.connect(str(path)) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return str(path)


def _keys(n, plate="plate1"):
    return {
        "plateID": [plate] * n,
        "rowID": [f"r{i + 1}" for i in range(n)],
        "columnID": [("c1", "c2")[i % 2] for i in range(n)],
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
    }


def _prcf(keys):
    return [f"{p}_{r}_{c}_{f}" for p, r, c, f in
            zip(keys["plateID"], keys["rowID"], keys["columnID"],
                keys["fieldID"])]


def _cells(n=4, units="um"):
    """A cell table with the five key columns and two measurements."""
    keys = _keys(n)
    return pd.DataFrame({
        **keys,
        "prcf": _prcf(keys),
        "cell_area": np.linspace(100.0, 400.0, n),
        "cell_channel_1_mean_intensity": np.linspace(1.0, 4.0, n),
        "measurement_units": [units] * n,
    })


def _nuclei(n=4, units="um"):
    """A child table, one nucleus per cell, linked by ``cell_id``."""
    keys = _keys(n)
    return pd.DataFrame({
        **keys,
        "prcf": _prcf(keys),
        "cell_id": list(range(1, n + 1)),
        "nucleus_area": np.linspace(10.0, 40.0, n),
        "measurement_units": [units] * n,
    })


def _png(n=4, extra=True):
    """``png_list`` as the crop writer leaves it, optionally annotated."""
    keys = _keys(n)
    frame = pd.DataFrame({
        **keys,
        "prcf": _prcf(keys),
        "cell_id": [f"o{i + 1}" for i in range(n)],
        "png_path": [f"/crops/{i + 1}.png" for i in range(n)],
    })
    if extra:
        frame["test"] = [i % 2 for i in range(n)]
    return frame


def _feature_frame(db, table="cell"):
    """The table straight out of SQLite, for comparing ``X`` against."""
    connection = sqlite3.connect(db)
    try:
        return pd.read_sql(f'SELECT * FROM "{table}"', connection)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# require_anndata's success path
# ---------------------------------------------------------------------------

def test_the_module_is_reached_only_through_require_anndata(anndata_double,
                                                            tmp_path):
    """``require_anndata`` returns the imported module, and that return value
    is what builds the container -- so replacing it replaces every AnnData
    the module makes, which is what the rest of this file depends on."""
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells()})

    returned = ax.require_anndata()
    adata, _result = ax.build_anndata(db, single_table="cell", verbose=False)

    assert returned is anndata_double
    assert returned is sys.modules["anndata"]
    assert isinstance(adata, anndata_double.AnnData)


# ---------------------------------------------------------------------------
# the plain build
# ---------------------------------------------------------------------------

def test_a_single_table_export_is_the_table_split_into_x_obs_and_var(
        anndata_double, tmp_path):
    """The whole default path: identity to ``obs_names``, measurements to
    ``X``, everything else to ``obs``, one ``var`` row per feature, and a
    provenance record naming the table it came from."""
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells()})
    source = _feature_frame(db)

    adata, result = ax.build_anndata(db, single_table="cell", verbose=False)

    features = ["cell_area", "cell_channel_1_mean_intensity"]
    assert list(adata.var_names) == features
    np.testing.assert_allclose(
        np.asarray(adata.X),
        source[features].to_numpy(dtype=np.float32))
    assert adata.X.dtype == np.float32
    assert list(adata.obs_names) == list(
        object_keys(ax._read_frame(db, (), "cell")[0]))
    # The measurements left obs; the identity did not.
    assert "cell_area" not in adata.obs.columns
    assert list(adata.obs["plateID"].astype(str)) == list(source["plateID"])
    assert adata.uns["spacr"]["anchor_object"] == "cell"
    assert adata.uns["spacr"]["joined"] is False
    assert adata.uns["spacr"]["source_tables"] == ["cell"]
    assert (result.n_obs, result.n_vars) == (len(source), len(features))
    assert result.n_obs_before_filter == len(source)
    assert result.n_missing == 0 and result.n_infinite == 0
    assert result.nan_policy == ax.NAN_KEEP
    assert result.path == "" and result.warnings == ()


def test_a_table_with_no_measurements_in_it_is_refused_by_name(
        anndata_double, tmp_path):
    """Every numeric column being identity is a database whose object tables
    were never written, and the export says which database and what to run
    rather than handing back an AnnData with a zero-width X."""
    keys = _keys(3)
    empty = pd.DataFrame({**keys, "prcf": _prcf(keys)})
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": empty})

    with pytest.raises(ValueError) as caught:
        ax.build_anndata(db, single_table="cell", verbose=False)

    message = str(caught.value)
    assert "no feature columns found" in message
    assert db in message
    assert "spacr doctor --db" in message


def test_verbose_prints_the_summary_and_every_note_it_collected(
        anndata_double, tmp_path, capsys):
    """``row_limit`` is a cap, not a filter, so the export that comes back is
    shorter than the one asked for -- and the reason is printed, not left to
    be inferred from a row count."""
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells(6)})

    adata, result = ax.build_anndata(db, single_table="cell", row_limit=2,
                                     verbose=True)

    printed = capsys.readouterr().out
    assert result.n_obs == 2 and result.n_obs_before_filter == 6
    assert adata.n_obs == 2
    assert list(adata.obs["object_label"]) == [1, 2], (
        "a cap keeps the first rows in table order")
    assert len(result.warnings) == 1
    assert "row_limit=2 truncated the export from 6 objects" in printed
    assert printed.count(result.warnings[0]) == 1
    assert "  note: " in printed
    assert result.describe() in printed


def test_a_filter_selects_exactly_the_objects_pandas_would(anndata_double,
                                                           tmp_path):
    """The filter is :class:`spacr.selection.DataFilter`, applied to the same
    frame -- so the exported objects are the frame's own mask, not a second
    filter language that happens to agree today."""
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells(6)})
    source = _feature_frame(db)
    data_filter = DataFilter().add(RangeFilter("cell_area", low=200.0))

    adata, result = ax.build_anndata(db, single_table="cell",
                                     data_filter=data_filter, verbose=False)

    expected = source.loc[data_filter.mask(source)]
    assert len(expected) < len(source), "the filter has to remove something"
    assert list(adata.obs["object_label"]) == list(expected["object_label"])
    assert result.n_obs == len(expected)
    assert result.n_obs_before_filter == len(source)
    assert adata.uns["spacr"]["filter"]["description"] == \
        data_filter.describe()
    assert adata.uns["spacr"]["filter"]["clauses"][0]["column"] == "cell_area"


def test_a_selection_narrows_the_export_to_the_keys_it_holds(anndata_double,
                                                             tmp_path):
    """A lasso in a linked view publishes a :class:`Selection` of object
    keys; "export what I am looking at" is that object handed straight in,
    and the keys that come back are the ones it held."""
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells(6)})
    source = _feature_frame(db)
    all_keys = list(object_keys(source, object_type="cell"))
    chosen = [all_keys[1], all_keys[4]]
    selection = Selection(keys=chosen, source="umap-lasso")

    adata, result = ax.build_anndata(db, single_table="cell",
                                     selection=selection, verbose=False)

    assert list(adata.obs_names) == chosen
    assert result.n_obs == 2 and result.n_obs_before_filter == 6
    assert adata.uns["spacr"]["filter"]["selection_source"] == "umap-lasso"
    assert adata.uns["spacr"]["filter"]["selection_size"] == 2


# ---------------------------------------------------------------------------
# png_list labels
# ---------------------------------------------------------------------------

def test_an_annotation_from_png_list_arrives_in_time_to_be_filtered_on(
        anndata_double, tmp_path):
    """The join drops everything ``png_list`` carries beyond six named
    columns, which is precisely the annotation the user wants to export by.
    Attaching it before the mask runs is what makes "export the cells I
    called infected" one call."""
    db = _write_db(tmp_path / "measurements" / "m.db",
                   {"cell": _cells(), "nucleus": _nuclei(),
                    "png_list": _png()})

    adata, result = ax.build_anndata(
        db, data_filter=DataFilter().add(CategoryFilter("test", ["1"])),
        verbose=False)

    assert "test" in adata.obs.columns, "the label came out of png_list"
    assert list(adata.obs["test"].astype(int)) == [1, 1]
    assert list(adata.obs["object_label"]) == [2, 4]
    assert result.n_obs_before_filter == 4
    assert any("attached 1 label column(s) from png_list" in note
               for note in result.warnings)
    assert "test" in adata.uns["spacr"]["annotation_columns"]
    assert "test" not in list(adata.var_names), (
        "an annotation is not a measurement and must not land in X")


def test_a_png_list_carrying_only_metadata_attaches_nothing(anndata_double,
                                                            tmp_path):
    """The columns are recognised, not guessed: what ``png_list`` gets from
    the crop writer is metadata and is left where it is, so a crop table with
    nothing added to it produces no note and no new obs column."""
    annotated = _write_db(tmp_path / "a" / "m.db",
                          {"cell": _cells(), "png_list": _png(extra=True)})
    bare = _write_db(tmp_path / "b" / "m.db",
                     {"cell": _cells(), "png_list": _png(extra=False)})

    with_label, with_result = ax.build_anndata(annotated, single_table="cell",
                                               verbose=False)
    without, without_result = ax.build_anndata(bare, single_table="cell",
                                               verbose=False)

    assert "test" in with_label.obs.columns
    assert any("png_list" in note for note in with_result.warnings)
    assert "test" not in without.obs.columns
    assert without_result.warnings == ()
    assert list(without.var_names) == list(with_label.var_names), (
        "the two databases differ only in the annotation column")


# ---------------------------------------------------------------------------
# the joined export
# ---------------------------------------------------------------------------

def test_the_join_suffixed_identity_copies_are_dropped_and_said_so(
        anndata_double, tmp_path):
    """A child table measured under a different calibration keeps its own
    ``measurement_units`` through the join, and the export drops the
    suffixed copy from ``obs`` -- naming it, and naming the option that
    keeps it."""
    db = _write_db(tmp_path / "measurements" / "m.db",
                   {"cell": _cells(units="um"),
                    "nucleus": _nuclei(units="px")})

    dropped, dropped_result = ax.build_anndata(db, verbose=False)
    kept, kept_result = ax.build_anndata(db, drop_redundant_identity=False,
                                         verbose=False)

    assert "measurement_units_nucleus" not in dropped.obs.columns
    assert "measurement_units_nucleus" in kept.obs.columns
    assert "plateID_nucleus" in kept.obs.columns
    assert "measurement_units" in dropped.obs.columns, (
        "the anchor's own copy stays")
    note = "".join(dropped_result.warnings)
    assert "join-suffixed copies of identity columns" in note
    assert "measurement_units_nucleus" in note
    assert "drop_redundant_identity=False" in note
    assert kept_result.warnings == ()
    assert list(dropped.var_names) == list(kept.var_names), (
        "identity is dropped from obs; the feature matrix is untouched")
    assert dropped.uns["spacr"]["joined"] is True
    assert dropped.uns["spacr"]["relationships"]["children"]["nucleus"][
        "count_column"] == "count_nucleus"


def test_a_join_with_nothing_redundant_in_it_adds_no_note(anndata_double,
                                                          tmp_path):
    """Only a suffix the join actually produced is a candidate: a database
    whose only object table is ``cell`` joins to nothing, so the drop runs
    and finds nothing to drop."""
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells()})

    adata, result = ax.build_anndata(db, verbose=False)

    assert adata.uns["spacr"]["source_tables"] == ["cell"]
    assert adata.uns["spacr"]["joined"] is True
    assert result.warnings == ()
    assert adata.n_obs == 4


# ---------------------------------------------------------------------------
# non-finite values and the nan policies
# ---------------------------------------------------------------------------

def test_an_infinity_becomes_nan_before_the_policy_sees_it(anndata_double,
                                                           tmp_path):
    """An inf survives ``dropna`` and then destroys any scaling or distance
    computed from it, so it is converted -- and counted, because a silent
    conversion of a measured value is what provenance exists to prevent."""
    cells = _cells()
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": cells})
    with sqlite3.connect(db) as connection:
        connection.execute(
            "UPDATE cell SET cell_area = 1e400 WHERE object_label = 1")

    adata, result = ax.build_anndata(db, single_table="cell", verbose=False)

    matrix = np.asarray(adata.X, dtype=float)
    assert np.isinf(_feature_frame(db)["cell_area"]).sum() == 1, (
        "the fixture really does hold an infinity")
    assert result.n_infinite == 1
    assert not np.isinf(matrix).any()
    assert np.isnan(matrix[0, 0])
    assert "non-finite (+/-inf) values were converted to NaN" in \
        "".join(result.warnings)
    assert int(adata.uns["spacr"]["nan"]["n_missing"]) == 1


def _cells_with_a_hole(n=4):
    """A cell table with one NaN, in one feature of one object."""
    cells = _cells(n)
    cells.loc[0, "cell_area"] = np.nan
    return cells


def test_drop_objects_removes_the_row_and_keeps_obs_aligned_to_x(
        anndata_double, tmp_path):
    """``obs`` is sliced by the same mask as ``X``; the stand-in refuses a
    container whose ``obs`` is longer than its matrix, so a mask applied to
    one and not the other would not survive this call."""
    db = _write_db(tmp_path / "measurements" / "m.db",
                   {"cell": _cells_with_a_hole()})

    adata, result = ax.build_anndata(db, single_table="cell",
                                     nan_policy=ax.NAN_DROP_OBJECTS,
                                     verbose=False)

    assert result.dropped_objects == 1
    assert (adata.n_obs, adata.n_vars) == (3, 2)
    assert len(adata.obs) == 3
    assert list(adata.obs["object_label"]) == [2, 3, 4]
    assert not np.isnan(np.asarray(adata.X, dtype=float)).any()
    assert result.n_missing == 1, "the count is of the matrix as received"
    assert result.n_obs_counted == 4


def test_drop_features_removes_the_column_and_reindexes_var(anndata_double,
                                                            tmp_path):
    """``var`` was built before the policy ran, so a dropped feature has to
    be taken out of it too -- and the stand-in checks that ``var`` is exactly
    as long as ``X`` is wide."""
    db = _write_db(tmp_path / "measurements" / "m.db",
                   {"cell": _cells_with_a_hole()})

    adata, result = ax.build_anndata(db, single_table="cell",
                                     nan_policy=ax.NAN_DROP_FEATURES,
                                     verbose=False)

    assert result.dropped_features == ("cell_area",)
    assert list(adata.var_names) == ["cell_channel_1_mean_intensity"]
    assert (adata.n_obs, adata.n_vars) == (4, 1)
    assert len(adata.var) == 1
    assert result.n_vars_counted == 2, "counted over the matrix as received"


def test_var_keeps_both_the_pre_policy_and_the_written_missing_counts(
        anndata_double, tmp_path):
    """For an imputing policy the raw counts are the only remaining record
    that a value was invented, and ``n_missing`` describes what was actually
    written -- which is what a reader inspecting ``var`` is asking about."""
    db = _write_db(tmp_path / "measurements" / "m.db",
                   {"cell": _cells_with_a_hole()})

    adata, _result = ax.build_anndata(db, single_table="cell",
                                      nan_policy=ax.NAN_MEAN, verbose=False)

    assert int(adata.var.loc["cell_area", "n_missing_raw"]) == 1
    assert float(adata.var.loc["cell_area", "frac_missing_raw"]) == 0.25
    assert int(adata.var.loc["cell_area", "n_missing"]) == 0
    assert float(adata.var.loc["cell_area", "frac_missing"]) == 0.0
    assert list(adata.obs["n_missing_features"]) == [1, 0, 0, 0], (
        "per object, the mask counts what was imputed for it")


def test_an_imputing_policy_writes_the_mask_that_keeps_it_honest(
        anndata_double, tmp_path):
    """``layers['missing']`` defaults on exactly when values were invented,
    and marks the cells that were: without it an imputed matrix is
    indistinguishable from a measured one."""
    db = _write_db(tmp_path / "measurements" / "m.db",
                   {"cell": _cells_with_a_hole()})
    source = _feature_frame(db)

    imputed, _result = ax.build_anndata(db, single_table="cell",
                                        nan_policy=ax.NAN_MEAN, verbose=False)
    unmasked, _unmasked_result = ax.build_anndata(
        db, single_table="cell", nan_policy=ax.NAN_MEAN, missing_layer=False,
        verbose=False)

    mask = np.asarray(imputed.layers["missing"])
    assert mask.shape == (4, 2)
    assert bool(mask[0, 0]) and int(mask.sum()) == 1
    assert "missing" not in unmasked.layers, (
        "the same database with the layer refused writes no layer")
    filled = float(np.asarray(imputed.X, dtype=float)[0, 0])
    assert filled == pytest.approx(
        float(source["cell_area"].mean()), rel=1e-6), (
        "the hole was filled with the feature's own mean")


def test_the_missing_layer_can_be_asked_for_under_the_keeping_policy(
        anndata_double, tmp_path):
    """``keep`` invents nothing, so there is no imputation mask -- but a
    caller who wants the NaN pattern as an array can ask, and gets it
    computed from the matrix that was written."""
    db = _write_db(tmp_path / "measurements" / "m.db",
                   {"cell": _cells_with_a_hole()})

    asked, _asked_result = ax.build_anndata(db, single_table="cell",
                                            missing_layer=True, verbose=False)
    default, _default_result = ax.build_anndata(db, single_table="cell",
                                                verbose=False)

    mask = np.asarray(asked.layers["missing"])
    assert mask.dtype == bool
    np.testing.assert_array_equal(
        mask, np.isnan(np.asarray(asked.X, dtype=float)))
    assert int(mask.sum()) == 1
    assert "missing" not in default.layers, (
        "the same export without the flag carries no layer")


# ---------------------------------------------------------------------------
# obsm
# ---------------------------------------------------------------------------

def test_an_embedding_handed_in_is_aligned_by_object_key(anndata_double,
                                                         tmp_path):
    """A keyed embedding is reindexed onto the exported objects, so the row
    order of the embedding never has to match the row order of the export --
    and the name picks up the scanpy ``X_`` convention."""
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells()})
    source = _feature_frame(db)
    coordinates = source[list(OBJECT_KEY_COLUMNS)].copy()
    coordinates["umap_1"] = [0.0, 1.0, 2.0, 3.0]
    coordinates["umap_2"] = [0.0, -1.0, -2.0, -3.0]
    # Row order deliberately reversed: an embedding aligned positionally
    # would attach every object's coordinates to the wrong object.
    shuffled = coordinates.iloc[::-1].reset_index(drop=True)

    adata, result = ax.build_anndata(
        db, single_table="cell", embeddings={"pca": shuffled}, verbose=False)

    embedding = np.asarray(adata.obsm["X_pca"], dtype=float)
    np.testing.assert_allclose(
        embedding, coordinates[["umap_1", "umap_2"]].to_numpy(float))
    assert result.obsm_keys == ("X_pca",)
    assert "X_pca" in adata.obsm and "pca" not in adata.obsm, (
        "the scanpy X_ convention is applied to the name given")


def test_compute_umap_reduces_when_there_are_enough_objects_and_says_so(
        anndata_double, tmp_path):
    """Driven through PCA so the embedding is deterministic. Two objects is
    below what any reducer can use, and the export says that in a note
    instead of writing an ``X_umap`` that is not one."""
    from spacr.utils import reduction_and_clustering

    enough = _write_db(tmp_path / "big" / "m.db", {"cell": _cells(6)})
    too_few = _write_db(tmp_path / "small" / "m.db", {"cell": _cells(2)})
    settings = {"reduction_method": "pca", "min_samples": 2}

    reduced, reduced_result = ax.build_anndata(
        enough, single_table="cell", compute_umap=True,
        umap_settings=settings, verbose=False)
    refused, refused_result = ax.build_anndata(
        too_few, single_table="cell", compute_umap=True,
        umap_settings=settings, verbose=False)

    expected, _labels, _reducer = reduction_and_clustering(
        np.asarray(reduced.X, dtype=float), n_neighbors=15, min_dist=0.1,
        metric="euclidean", eps=0.5, min_samples=2, clustering="dbscan",
        reduction_method="pca", verbose=False, n_jobs=1)
    np.testing.assert_allclose(np.asarray(reduced.obsm["X_umap"]),
                               np.asarray(expected, dtype=np.float32),
                               rtol=1e-5, atol=1e-5)
    assert reduced_result.obsm_keys == ("X_umap",)
    assert reduced.uns["spacr"]["umap"]["computed_by"] == \
        "spacr.utils.reduction_and_clustering"
    assert "X_umap" not in refused.obsm
    assert "umap" not in refused.uns["spacr"], (
        "no embedding was computed, so there is nothing to describe")
    assert any("UMAP needs at least 3" in note
               for note in refused_result.warnings)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def test_the_written_file_holds_the_matrix_and_the_provenance(anndata_double,
                                                              tmp_path):
    """``export_anndata`` creates the parent directory, writes what
    ``build_anndata`` built, and fills ``path`` in on the result it returns.
    The read-back is the check that nothing was lost on the way out."""
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells()})
    out = tmp_path / "results" / "deep" / "export.h5ad"

    result = ax.export_anndata(db, str(out), single_table="cell",
                               register=False, verbose=False)

    assert os.path.isfile(out)
    written = _read_double(str(out))
    assert result.path == str(out)
    assert result.artifact_id == ""
    assert (written.n_obs, written.n_vars) == (result.n_obs, result.n_vars)
    np.testing.assert_allclose(
        np.asarray(written.X),
        _feature_frame(db)[["cell_area", "cell_channel_1_mean_intensity"]]
        .to_numpy(dtype=np.float32))
    assert written.uns["spacr"]["source_database"] == db
    assert written.compression == "gzip"


def test_a_settings_value_hdf5_cannot_store_does_not_cost_the_export(
        anndata_double, tmp_path):
    """``uns`` goes through ``_h5ad_safe`` before the write, and the
    stand-in's writer raises on anything HDF5 would -- so an unstorable
    settings value reaching ``uns`` would fail this write, not merely be
    tidied afterwards."""
    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells()})
    out = tmp_path / "results" / "export.h5ad"

    result = ax.export_anndata(
        db, str(out), single_table="cell", register=False, verbose=False,
        settings={"nucleus_channel": None, "channels": (1, 2),
                  "model": object()})

    written = _read_double(str(out))
    stored = written.uns["spacr_settings"]
    assert result.n_obs == 4
    assert stored["nucleus_channel"] == ""
    assert stored["channels"] == [1, 2]
    assert isinstance(stored["model"], str) and "object object" in \
        stored["model"]


def test_the_export_is_registered_with_the_measurements_db_as_its_input(
        anndata_double, tmp_path, monkeypatch):
    """A re-run of Measure has to mark this export stale, which it can only
    do if the export records the database artifact it was built from."""
    monkeypatch.delenv("SPACR_ARTIFACTS_DB", raising=False)
    from spacr import artifacts, ports

    root = tmp_path / "project"
    db = _write_db(root / "measurements" / "measurements.db",
                   {"cell": _cells()})
    upstream = artifacts.register(project=str(root), module="measure",
                                  kind=ports.MEASUREMENTS_DB, role="db",
                                  path=db)

    result = ax.export_anndata(db, str(root / "results" / "e.h5ad"),
                               single_table="cell", verbose=False)

    record = artifacts.latest(ax.ANNDATA_KIND, project=str(root))
    assert result.artifact_id == record.artifact_id != ""
    assert upstream.artifact_id in tuple(record.inputs)
    assert record.extra["n_obs"] == 4
    assert record.extra["nan_policy"] == ax.NAN_KEEP
    assert record.extra["source_database"] == db


def test_registration_is_a_choice_and_an_unregistered_project_has_no_record(
        anndata_double, tmp_path, monkeypatch):
    """Both halves in one project: ``register=False`` leaves the registry
    untouched, and the very next export into the same root fills it -- with
    no input, because nothing registered the database it read."""
    monkeypatch.delenv("SPACR_ARTIFACTS_DB", raising=False)
    from spacr import artifacts

    root = tmp_path / "project"
    db = _write_db(root / "measurements" / "measurements.db",
                   {"cell": _cells()})

    quiet = ax.export_anndata(db, str(root / "results" / "quiet.h5ad"),
                              single_table="cell", register=False,
                              verbose=False)
    assert quiet.artifact_id == ""
    assert artifacts.by_kind(ax.ANNDATA_KIND, project=str(root)) == []

    loud = ax.export_anndata(db, str(root / "results" / "loud.h5ad"),
                             single_table="cell", verbose=False)

    records = artifacts.by_kind(ax.ANNDATA_KIND, project=str(root))
    assert [record.artifact_id for record in records] == [loud.artifact_id]
    assert tuple(records[0].inputs) == ()
    assert os.path.isfile(root / "results" / "quiet.h5ad"), (
        "the unregistered export is still a finished file")


def test_a_registry_that_cannot_be_opened_does_not_lose_the_export(
        anndata_double, tmp_path, monkeypatch):
    """A project root that is a regular file is what a read-only project
    looks like from here: there is nowhere to put ``artifacts.db``. The file
    is the product and the registry is a convenience, so the export survives
    with a warning and an empty artifact id."""
    monkeypatch.delenv("SPACR_ARTIFACTS_DB", raising=False)

    db = _write_db(tmp_path / "measurements" / "m.db", {"cell": _cells()})
    blocked = tmp_path / "root_is_a_file"
    blocked.write_text("not a directory")
    out = tmp_path / "written" / "export.h5ad"

    with pytest.warns(RuntimeWarning) as caught:
        result = ax.export_anndata(db, str(out), single_table="cell",
                                   project=str(blocked), verbose=False)

    warned = "\n".join(str(record.message) for record in caught)
    assert "could not be registered with spacr.artifacts" in warned
    assert str(out) in warned
    assert result.artifact_id == ""
    assert result.path == str(out)
    written = _read_double(str(out))
    assert written.n_obs == 4
    assert written.uns["spacr"]["source_database"] == db, (
        "the provenance the registry would have held is in the file")


# ---------------------------------------------------------------------------
# one file per object table
# ---------------------------------------------------------------------------

def test_a_set_writes_one_file_per_object_table_and_links_the_children(
        anndata_double, tmp_path):
    """A nucleus is not a cell. Each file holds one object type at its own
    granularity, and the child records the sibling file its parents are in
    plus the ``obs`` column that joins to it -- the cell file, having no
    parent, records none."""
    db = _write_db(tmp_path / "measurements" / "m.db",
                   {"cell": _cells(), "nucleus": _nuclei()})
    out = tmp_path / "set"

    results = ax.export_anndata_set(db, str(out), register=False,
                                    prefix="exp1_", verbose=False)

    assert sorted(results) == ["cell", "nucleus"]
    assert os.path.isfile(out / "exp1_cell.h5ad")
    child = _read_double(str(out / "exp1_nucleus.h5ad"))
    parent = child.uns["spacr"]["relationships"]["parent"]
    assert parent["file"] == "exp1_cell.h5ad"
    assert parent["obs_column"] == "cell_id"
    assert parent["table"] == "cell"
    assert list(child.var_names) == ["nucleus_area"]
    assert results["nucleus"].n_obs == 4
    cell_file = _read_double(str(out / "exp1_cell.h5ad"))
    assert "parent" not in cell_file.uns["spacr"]["relationships"], (
        "the anchor has no parent file to point at")


def test_a_child_file_that_cannot_be_reopened_warns_instead_of_taking_the_set(
        anndata_double, tmp_path):
    """The stamp is a small re-open, and a file that will not open must cost
    the link, not the whole set -- nor the bytes already on disk."""
    child = tmp_path / "nucleus.h5ad"
    child.write_bytes(b"this is not a container")

    with pytest.warns(RuntimeWarning) as caught:
        ax._stamp_parent_file(str(child), str(tmp_path / "cell.h5ad"),
                              "cell_id")

    warned = "\n".join(str(record.message) for record in caught)
    assert "could not record the parent file in" in warned
    assert str(child) in warned
    assert child.read_bytes() == b"this is not a container", (
        "a failed stamp must not damage the file it could not read")
