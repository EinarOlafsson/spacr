"""Tidy Parquet tables and the R loader, written from a real measurements.db.

The database comes out of spaCR's own writers
(:func:`spacr.utils._merge_and_save_to_database`,
:func:`spacr.utils.filepaths_to_database`, :func:`spacr.io._save_settings_to_db`),
as in ``tests/test_anndata_export.py``, so the join, the suffixes and the
``png_list`` labels are the ones a real run produces.

Every table is written through :mod:`spacr.tabular`, read back with pandas
and with pyarrow, and compared by dtype and by value -- against the frame
that was written and, for the features, against the rows in SQLite. The R
check runs only where ``Rscript`` with arrow or nanoparquet and
SingleCellExperiment is installed; anywhere else it is skipped and says so.

CPU-only, offline, deterministic. ``anndata`` is not needed.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import schema, tabular
from spacr import anndata_export as ax

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

FIELDS = ("plate1_A01_1", "plate1_A01_2", "plate1_A02_1", "plate1_A02_2")
N_OBJECTS = 5
OBJECT_TABLES = ("cell", "cytoplasm", "nucleus", "pathogen")
KEY_COLUMNS = (schema.PLATE_KEY, schema.ROW_KEY, schema.COLUMN_KEY,
               schema.FIELD_KEY, schema.PRC_KEY, schema.WELL_KEY)


def measure_field(root, stem):
    """Write one field's object tables through the real writer.

    The last cell of each field has no pathogen, which is how the joined
    matrix gets its NaN.
    """
    from spacr.utils import _merge_and_save_to_database

    labels = list(range(1, N_OBJECTS + 1))
    for table in OBJECT_TABLES:
        rows = labels[:-1] if table == "pathogen" else labels
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


def write_crops(root, stem):
    """Index crop file names through the real ``png_list`` writer."""
    from spacr.utils import filepaths_to_database

    folder = os.path.join(root, "data", "cell_png")
    os.makedirs(folder, exist_ok=True)
    paths = [os.path.join(folder, f"{stem}_{i + 1}.png")
             for i in range(N_OBJECTS)]
    filepaths_to_database(paths, {"timelapse": False}, root, "cell")


def add_annotation(db):
    """One human annotation column on ``png_list``, as Annotate adds it."""
    connection = sqlite3.connect(db)
    try:
        connection.execute("ALTER TABLE png_list ADD COLUMN infected INTEGER")
        paths = [row[0] for row in connection.execute(
            "SELECT png_path FROM png_list ORDER BY png_path")]
        for i, path in enumerate(paths):
            connection.execute(
                "UPDATE png_list SET infected=? WHERE png_path=?",
                (i % 2, path))
        connection.commit()
    finally:
        connection.close()


def build_project(root):
    """A project database written by every writer that touches it."""
    os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    for stem in FIELDS:
        measure_field(root, stem)
        write_crops(root, stem)
    from spacr.io import _save_settings_to_db

    _save_settings_to_db({"src": os.path.join(root, "data"),
                          "stage": "measure", "experiment": "exp"})
    db = os.path.join(root, "measurements", "measurements.db")
    add_annotation(db)
    return db


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("tables_project"))
    return root, build_project(root)


@pytest.fixture(scope="module")
def exported(project, tmp_path_factory):
    """One export with the R loader and a keyed embedding."""
    _root, db = project
    parts = ax._assemble_tables(db)
    keys = parts["obs"].index
    frame = parts["frame"]
    embedding = frame[[schema.PLATE_KEY, schema.ROW_KEY, schema.COLUMN_KEY,
                       schema.FIELD_KEY, schema.OBJECT_LABEL_KEY]].copy()
    embedding["x"] = np.arange(len(frame), dtype=float)
    embedding["y"] = -np.arange(len(frame), dtype=float)
    embedding = embedding.iloc[::-1].reset_index(drop=True)
    out = str(tmp_path_factory.mktemp("tables_out"))
    result = ax._export_tables(db, out, r_loader=True, rds=False,
                               embeddings={"umap": embedding},
                               register=False, verbose=False)
    return out, result, parts, keys


def read_both(path, **kwargs):
    """Read a Parquet file through the funnel and through pyarrow."""
    through_funnel = tabular.read_table(path, report=None, **kwargs)
    through_arrow = pq.read_table(path).to_pandas()
    return through_funnel, through_arrow


def _assert_same_table(read, written):
    """Equal frames, where a text column may come back as pandas' str dtype.

    Under pandas 3 Parquet text is read as the ``str`` dtype while the frame
    that was written may still hold it as ``object``. Such a column must hold
    only strings or missing values and be equal value for value; every other
    column, and every dtype that is not text, is compared exactly.
    """
    read = read.copy()
    for column in written.columns:
        if (written[column].dtype == object and column in read
                and pd.api.types.is_string_dtype(read[column].dtype)
                and read[column].dtype != object):
            assert all(isinstance(v, str) or pd.isna(v)
                       for v in written[column]), column
            assert (read[column].isna() == written[column].isna()).all(), column
            read[column] = read[column].astype(object).where(
                read[column].notna(), written[column])
    pd.testing.assert_frame_equal(read, written)


def test_the_export_writes_every_table_and_the_loader(exported):
    out, result, _parts, _keys = exported
    names = sorted(os.path.basename(path) for path in result.files)
    assert names == sorted([
        "objects.parquet", "features.parquet", "wells.parquet",
        "embeddings.parquet", "provenance.parquet", "load_spacr_export.R"])
    assert result.n_objects == len(FIELDS) * N_OBJECTS
    assert result.n_wells == 2
    assert result.r_loader == os.path.join(out, "load_spacr_export.R")
    assert "load_spacr_export()" in result.describe()


def test_objects_round_trip_with_their_dtypes_and_values(exported):
    out, _result, parts, _keys = exported
    written = ax._objects_table(parts, "float64")
    funnel, arrow = read_both(os.path.join(out, "objects.parquet"))
    for frame in (funnel, arrow):
        _assert_same_table(frame, written)
    for column in KEY_COLUMNS + (schema.OBJECT_TYPE_KEY,):
        assert isinstance(funnel[column].dtype, pd.CategoricalDtype), column
    assert funnel["infected"].dtype == np.int64
    assert funnel[schema.OBJECT_LABEL_KEY].dtype == np.int64
    assert funnel["cell_area"].dtype == np.float64
    assert funnel["object_key"].is_unique
    assert set(funnel[schema.WELL_KEY].astype(str)) == {"A01", "A02"}
    assert set(funnel[schema.PRC_KEY].astype(str)) == {
        "plate1_r1_c1", "plate1_r1_c2"}


def test_the_feature_values_are_the_databases_own(project, exported):
    _root, db = project
    out, _result, _parts, _keys = exported
    objects = tabular.read_table(os.path.join(out, "objects.parquet"),
                                 report=None)
    source = tabular.read_database(db, "cell", report=None)[0]
    merged = objects.merge(
        source, on=[schema.PRCF_KEY, schema.OBJECT_LABEL_KEY],
        suffixes=("", "_db"))
    assert len(merged) == len(objects)
    np.testing.assert_array_equal(merged["cell_area"].to_numpy(),
                                  merged["cell_area_db"].to_numpy())
    np.testing.assert_array_equal(
        merged["cell_channel_1_mean_intensity"].to_numpy(),
        merged["cell_channel_1_mean_intensity_db"].to_numpy())
    assert objects["pathogen_area"].isna().sum() == len(FIELDS)


def test_the_arrow_schema_keeps_dictionaries_and_numbers(exported):
    out, _result, _parts, _keys = exported
    table_schema = pq.read_schema(os.path.join(out, "objects.parquet"))
    assert pa.types.is_dictionary(table_schema.field(schema.PLATE_KEY).type)
    assert pa.types.is_dictionary(table_schema.field(schema.WELL_KEY).type)
    assert table_schema.field("cell_area").type == pa.float64()
    assert table_schema.field(schema.OBJECT_LABEL_KEY).type == pa.int64()
    # pandas 3's default string dtype is written as large_string; either is
    # a plain (non-dictionary) string column.
    key_type = table_schema.field("object_key").type
    assert pa.types.is_string(key_type) or pa.types.is_large_string(key_type)


def test_every_file_carries_its_description_and_the_provenance(exported):
    out, _result, parts, _keys = exported
    for name in ("objects", "features", "wells", "embeddings", "provenance"):
        described = tabular._parquet_metadata(
            os.path.join(out, f"{name}.parquet"))
        assert described["format"] == "spacr-tables"
        assert described["table"] == name
        assert described["object_key"] == "object_key"
        assert described["provenance"]["source_database"] == parts["db_path"]
        assert described["provenance"]["n_objects"] == len(parts["obs"])
    objects = tabular._parquet_metadata(os.path.join(out, "objects.parquet"))
    assert objects["feature_columns"] == parts["features"]
    assert objects["annotation_columns"] == ["infected"]
    assert tabular._parquet_metadata(
        os.path.join(out, "wells.parquet"))["statistic"] == "mean"


def test_the_features_table_describes_every_feature_column(exported):
    out, _result, parts, _keys = exported
    features = tabular.read_table(os.path.join(out, "features.parquet"),
                                  canonicalise=False, report=None)
    assert list(features["feature"]) == parts["features"]
    assert "channel" in features.columns
    assert features["n_missing"].dtype == np.int64
    assert isinstance(features["family"].dtype, pd.CategoricalDtype)
    row = features.set_index("feature").loc["cell_channel_0_mean_intensity"]
    assert row["channel"] == 0
    assert row["family"] == "intensity"


def test_the_well_table_is_the_mean_over_each_wells_objects(exported):
    out, _result, parts, _keys = exported
    wells = tabular.read_table(os.path.join(out, "wells.parquet"),
                               report=None)
    objects = tabular.read_table(os.path.join(out, "objects.parquet"),
                                 report=None)
    expected = objects.groupby(schema.PRC_KEY, observed=True)[
        parts["features"]].mean()
    got = wells.set_index(wells[schema.PRC_KEY].astype(str))
    for prc, row in expected.iterrows():
        np.testing.assert_allclose(
            got.loc[str(prc), parts["features"]].to_numpy(dtype=float),
            row.to_numpy(dtype=float), equal_nan=True)
    assert list(wells["n_objects"]) == [2 * N_OBJECTS, 2 * N_OBJECTS]
    assert wells["n_objects"].dtype == np.int64
    assert isinstance(wells[schema.WELL_KEY].dtype, pd.CategoricalDtype)


def test_a_median_well_table_is_the_median(project, tmp_path):
    _root, db = project
    ax._export_tables(db, tmp_path, well_statistic="median",
                      register=False, verbose=False)
    wells = tabular.read_table(tmp_path / "wells.parquet", report=None)
    objects = tabular.read_table(tmp_path / "objects.parquet", report=None)
    medians = objects.groupby(schema.PRC_KEY, observed=True)[
        "cell_area"].median()
    assert list(wells["cell_area"]) == list(medians)
    with pytest.raises(ValueError, match="well_statistic"):
        ax._export_tables(db, tmp_path, well_statistic="mode",
                          register=False, verbose=False)


def test_a_keyed_embedding_is_aligned_by_object_key(exported):
    out, _result, parts, keys = exported
    embeddings = tabular.read_table(
        os.path.join(out, "embeddings.parquet"), report=None)
    assert list(embeddings.columns) == ["object_key", "X_umap_1", "X_umap_2"]
    assert list(embeddings["object_key"]) == [str(k) for k in keys]
    np.testing.assert_array_equal(embeddings["X_umap_1"].to_numpy(),
                                  np.arange(len(keys), dtype=float))
    described = tabular._parquet_metadata(
        os.path.join(out, "embeddings.parquet"))
    assert described["embeddings"] == {"X_umap": ["X_umap_1", "X_umap_2"]}


def test_the_provenance_table_holds_json_values(exported):
    out, _result, parts, _keys = exported
    provenance = tabular.read_table(
        os.path.join(out, "provenance.parquet"), canonicalise=False,
        report=None)
    assert set(provenance["section"].astype(str)) <= {"export", "settings"}
    export = {row.key: json.loads(row.value) for row in
              provenance.itertuples() if row.section == "export"}
    assert export["source_database"] == parts["db_path"]
    assert export["n_features"] == len(parts["features"])
    assert export["tables"]["files"]["objects"] == "objects.parquet"
    assert export["annotation_columns"] == ["infected"]


def test_the_r_loader_names_the_files_and_the_packages(exported):
    out, _result, _parts, _keys = exported
    with open(os.path.join(out, "load_spacr_export.R"), encoding="utf-8") as handle:
        script = handle.read()
    for name in ("objects.parquet", "features.parquet", "wells.parquet",
                 "embeddings.parquet", "provenance.parquet"):
        assert f'"{name}"' in script
    assert "SingleCellExperiment::SingleCellExperiment(" in script
    assert "arrow::read_parquet" in script
    assert "nanoparquet::read_parquet" in script
    assert 'BiocManager::install(\\"SingleCellExperiment\\")' in script
    assert script.count("{") == script.count("}")


def test_rds_data_frames_round_trip_when_pyreadr_is_installed(project,
                                                               tmp_path):
    pyreadr = pytest.importorskip(
        "pyreadr", reason="the .rds data frames need `pip install pyreadr`")
    _root, db = project
    result = ax._export_tables(db, tmp_path, r_loader=True, register=False,
                               verbose=False)
    assert {os.path.basename(p) for p in result.rds_files} == {
        "objects.rds", "features.rds", "wells.rds", "provenance.rds"}
    objects = tabular.read_table(tmp_path / "objects.parquet", report=None)
    from_r = pyreadr.read_r(str(tmp_path / "objects.rds"))[None]
    np.testing.assert_array_equal(from_r["cell_area"].to_numpy(),
                                  objects["cell_area"].to_numpy())
    assert list(from_r["object_key"]) == list(objects["object_key"])
    assert list(from_r[schema.PLATE_KEY]) == list(
        objects[schema.PLATE_KEY].astype(str))


def test_a_missing_pyreadr_is_a_note_unless_rds_was_required(project,
                                                            tmp_path,
                                                            monkeypatch):
    _root, db = project
    monkeypatch.setitem(sys.modules, "pyreadr", None)
    result = ax._export_tables(db, tmp_path, r_loader=True, register=False,
                               verbose=False)
    assert result.rds_files == ()
    assert any("needs pyreadr" in note for note in result.warnings)
    with pytest.raises(ImportError, match="python -m pip install pyreadr"):
        ax._export_tables(db, tmp_path, rds=True, register=False,
                          verbose=False)


def test_a_missing_pyarrow_says_how_to_install_it(project, tmp_path,
                                                  monkeypatch):
    _root, db = project
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match="python -m pip install pyarrow"):
        ax._export_tables(db, tmp_path, register=False, verbose=False)
    assert not os.listdir(tmp_path)


def test_the_run_entry_writes_the_tables_without_anndata(project,
                                                         monkeypatch):
    root, _db = project
    monkeypatch.setitem(sys.modules, "anndata", None)
    result = ax.run_anndata_export({"src": root, "anndata_format": "r",
                                    "anndata_register_artifact": False})
    expected = os.path.join(root, "results",
                            f"{os.path.basename(root)}_tables")
    assert result.directory == expected
    assert os.path.isfile(os.path.join(expected, "objects.parquet"))
    assert os.path.isfile(os.path.join(expected, "load_spacr_export.R"))
    with pytest.raises(ax.AnnDataExtraMissing):
        ax.run_anndata_export({"src": root, "anndata_format": "all"})
    with pytest.raises(ValueError, match="anndata_format"):
        ax.run_anndata_export({"src": root, "anndata_format": "xlsx"})


def test_the_tables_register_as_an_artifact(project, tmp_path):
    root, db = project
    result = ax._export_tables(db, tmp_path / "tables", project=str(tmp_path),
                               verbose=False)
    assert result.artifact_id
    from spacr import artifacts

    records = artifacts.by_kind("parquet_tables", project=str(tmp_path))
    assert [r.artifact_id for r in records] == [result.artifact_id]


def test_the_new_settings_have_defaults_types_and_tooltips():
    defaults = ax.anndata_export_settings()
    assert defaults["anndata_format"] == "h5ad"
    assert defaults["anndata_tidy_dir"] == ""
    assert ax._TYPES["anndata_format"] is str
    for key in ("anndata_format", "anndata_tidy_dir"):
        tip = ax._TOOLTIPS[key]
        assert len(tip) <= 600 and "Default " in tip and tip.endswith(".")


def _r_with_packages():
    """``Rscript`` when it and the loader's R packages are installed."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    probe = subprocess.run(
        [rscript, "-e", "quit(status = as.integer(!("
         "requireNamespace('SingleCellExperiment', quietly = TRUE) && "
         "(requireNamespace('arrow', quietly = TRUE) || "
         "requireNamespace('nanoparquet', quietly = TRUE)))))"],
        capture_output=True, timeout=120)
    return rscript if probe.returncode == 0 else None


@pytest.mark.parametrize("nested_caller", [False, True])
def test_the_sourced_r_loader_resolves_its_own_directory(exported, tmp_path,
                                                        nested_caller):
    """Resolve the export beside its loader, regardless of caller placement.

    :param exported: the real keyed table export fixture.
    :param tmp_path: independent script and working directories.
    :param nested_caller: source through an additional caller when true.
    :returns: ``None``; checks the native R-resolved export directory.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed on this machine")
    out, _result, _parts, _keys = exported
    working = tmp_path / "working"
    caller = tmp_path / "caller"
    working.mkdir()
    caller.mkdir()
    source = f'source({json.dumps(os.path.join(out, "load_spacr_export.R"))})\n'
    if nested_caller:
        nested = tmp_path / "nested"
        nested.mkdir()
        nested_script = nested / "source_loader.R"
        nested_script.write_text(source, encoding="utf-8")
        source = f'source({json.dumps(str(nested_script))})\n'
    check = caller / "check.R"
    check.write_text(
        source + 'stopifnot(identical(.spacr_export_dir, '
        f'normalizePath({json.dumps(out)})))\n', encoding="utf-8")
    completed = subprocess.run([rscript, "--vanilla", str(check)],
                               cwd=working, capture_output=True,
                               text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr


def test_the_r_loader_builds_a_single_cell_experiment(exported, tmp_path):
    rscript = _r_with_packages()
    if rscript is None:
        pytest.skip("Rscript with arrow or nanoparquet and "
                    "SingleCellExperiment is not installed on this machine")
    out, _result, parts, _keys = exported
    check = tmp_path / "check.R"
    check.write_text(
        f'source("{os.path.join(out, "load_spacr_export.R")}")\n'
        'sce <- load_spacr_export()\n'
        'values <- SummarizedExperiment::assay(sce, "measurements")\n'
        'cat(nrow(sce), ncol(sce), sum(values["cell_area", ]),\n'
        '    paste(SingleCellExperiment::reducedDimNames(sce)), "\\n")\n',
        encoding="utf-8")
    completed = subprocess.run([rscript, "--vanilla", str(check)],
                               cwd=tmp_path, capture_output=True,
                               text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    n_features, n_objects, total, name = completed.stdout.split()
    assert int(n_features) == len(parts["features"])
    assert int(n_objects) == len(parts["obs"])
    assert float(total) == pytest.approx(
        float(np.nansum(parts["frame"]["cell_area"])))
    assert name == "X_umap"
