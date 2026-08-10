"""``spacr.anndata_export`` on the databases ``tests/test_anndata_export.py``
does not build: a 3-D run, a timelapse, a plate whose crops were indexed
twice, a project measured with no crops at all, and a path that is not a
database.

Same rule as that file -- every database here is written by
:func:`spacr.utils._merge_and_save_to_database` and
:func:`spacr.utils.filepaths_to_database`, the calls ``measure_crop`` makes,
because a hand-built table has whatever columns the test author remembered.
The variations are made the way the field makes them: a measurement stamp
from :func:`spacr.measure.resolve_measurement_spacing`, a timelapse file
name, a second ``png_list`` row for one object, an ``UPDATE`` on one column.

What these defend, in one line each:

* a 3-D run's ``cell_area`` is documented as the **volume** it is -- the
  central promise of the module docstring's ``var`` paragraph, and the
  reason ``var`` is resolved against the row's own ``measurement_units``;
* a timelapse keys and labels **every frame separately**, and refuses to
  collapse them when it is not told it is one;
* two crops of one object that **disagree** about a label attach nothing,
  rather than a plausible wrong label;
* a provenance value HDF5 cannot store does not cost the user the export;
* an unusable path is named, and is not turned into a database by being
  opened.

One strict ``xfail`` records a real defect found while writing this file:
``ExportResult.frac_missing`` divides a pre-policy numerator by a
post-policy shape and prints 127.3% missing.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import schema

anndata = pytest.importorskip(
    "anndata", reason="the AnnData export needs `pip install spacr[anndata]`")

from spacr import anndata_export as ax  # noqa: E402  (after importorskip)


# ---------------------------------------------------------------------------
# projects spaCR's own writers built
# ---------------------------------------------------------------------------

#: Two wells, one field each. Small on purpose: every assertion below is
#: about a shape or a name, never about a distribution.
FIELDS = ("plate1_A01_1", "plate1_A02_1")

#: Two frames of one field, spelled the way ``measure`` names a timelapse
#: stack: ``<plate>_<well>_<field>_<time>``.
TIMELAPSE_FIELDS = ("plate1_A01_1_1", "plate1_A01_1_2")

OBJECT_TABLES = ("cell", "nucleus", "pathogen")

#: The stamp :func:`spacr.measure.resolve_measurement_spacing` returns for a
#: 3-D run given both voxel sizes -- the case where every geometric column is
#: in micrometres and ``<object>_area`` is a volume.
UM_STAMP = {"measurement_ndim": 3, "measurement_units": "um", "n_z": 7,
            "voxel_size_z_um": 0.5, "voxel_size_xy_um": 0.1}

#: What that same function returns for a 2-D run: pixels, unconditionally,
#: even when a voxel size is configured.
PX_STAMP = {"measurement_ndim": 2, "measurement_units": "px", "n_z": 1,
            "voxel_size_z_um": None, "voxel_size_xy_um": None}


def build_project(root, *, stems=FIELDS, n_cells=3, timelapse=False,
                  stamp=None, crops=True, pathogen_cells=None,
                  channels=1, pathogen_channels=None):
    """Write one project through the real measurement writers.

    ``pathogen_cells`` leaves the remaining cells of every field without a
    pathogen row, which is how a real plate looks and is the only honest way
    to get NaN into the joined matrix: the join is a left join onto the cell
    table, so a cell with no pathogen gets NaN in every ``pathogen_*``
    column.

    :returns: the ``measurements.db`` path.
    """
    from spacr.utils import _merge_and_save_to_database, filepaths_to_database

    os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    keep = n_cells - 1 if pathogen_cells is None else pathogen_cells

    for stem in stems:
        for table in OBJECT_TABLES:
            labels = list(range(1, n_cells + 1))
            if table == "pathogen":
                labels = labels[:keep]
            n = len(labels)
            morphology = pd.DataFrame({
                "label": labels,
                f"{table}_area": [100.0 + i for i in range(n)],
                f"{table}_perimeter": [40.0 + 2 * i for i in range(n)],
            })
            n_channels = (channels if table != "pathogen"
                          else (channels if pathogen_channels is None
                                else pathogen_channels))
            intensity = {"label": labels}
            for channel in range(n_channels):
                intensity[f"{table}_channel_{channel}_mean_intensity"] = [
                    5.0 + i for i in range(n)]
            if table in schema.CHILD_OBJECT_TABLES:
                morphology["cell_id"] = np.asarray(labels, dtype=float)
            _merge_and_save_to_database(
                morphology, pd.DataFrame(intensity), table, root, stem, "exp",
                timelapse, stamp=stamp)
        if crops:
            folder = os.path.join(root, "data", "cell_png")
            os.makedirs(folder, exist_ok=True)
            paths = [os.path.join(folder, f"{stem}_{i + 1}.png")
                     for i in range(n_cells)]
            filepaths_to_database(paths, {"timelapse": timelapse}, root, "cell")
    return os.path.join(root, "measurements", "measurements.db")


def annotate(db, values):
    """Add one human annotation column to ``png_list``, the Annotate way.

    Keyed on ``png_path``: ``png_list`` declares a ``rowID`` column, SQLite
    identifiers are case-insensitive and a declared column shadows the
    implicit rowid, so ``WHERE rowid = ?`` rewrites the whole table. That is
    bug 1 of ``tests/test_db_contract.py``.
    """
    connection = sqlite3.connect(db)
    try:
        connection.execute("ALTER TABLE png_list ADD COLUMN infected INTEGER")
        paths = [row[0] for row in connection.execute(
            "SELECT png_path FROM png_list ORDER BY png_path")]
        assert len(paths) == len(values), "annotate() was given the wrong count"
        for path, value in zip(paths, values):
            connection.execute(
                "UPDATE png_list SET infected=? WHERE png_path=?",
                (value, path))
        connection.commit()
    finally:
        connection.close()


def duplicate_crop_row(db, object_id, **overrides):
    """Index one object's crop a second time, as a re-run of Measure does.

    :param overrides: column values to change on the copy; anything not named
        keeps the original row's value, so the two rows agree except where
        the caller says they do not.
    """
    connection = sqlite3.connect(db)
    try:
        columns = [row[1] for row in
                   connection.execute("PRAGMA table_info(png_list)")]
        original = list(next(iter(connection.execute(
            "SELECT * FROM png_list WHERE cell_id = ?", (object_id,)))))
        original[columns.index("png_path")] = (
            original[columns.index("png_path")].replace(".png", "_copy.png"))
        for name, value in overrides.items():
            original[columns.index(name)] = value
        connection.execute(
            f"INSERT INTO png_list VALUES ({','.join('?' * len(columns))})",
            original)
        connection.commit()
    finally:
        connection.close()


@pytest.fixture(scope="module")
def px_project(tmp_path_factory):
    """One 2-D, pixel-stamped project, shared by the read-only assertions."""
    root = str(tmp_path_factory.mktemp("anndata_px"))
    return root, build_project(root, stamp=PX_STAMP)


# ---------------------------------------------------------------------------
# the units var is resolved against
# ---------------------------------------------------------------------------

def test_a_3d_run_documents_cell_area_as_the_volume_it_is(tmp_path, px_project):
    """``var`` reads the row's stamp instead of trusting the column name.

    ``cell_area`` is a name spaCR keeps in 3-D -- measure.py records the unit
    on the row rather than renaming the column -- so a reader who takes the
    name at face value reports a volume as an area. This is the promise the
    module docstring makes for ``var``, and it is only worth anything if the
    same column really does come out described two different ways.
    """
    from spacr import feature_dict

    db = build_project(str(tmp_path / "um"), stamp=UM_STAMP)
    volume, _ = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False)

    assert volume.uns["spacr"]["measurement_units"] == "um"
    assert volume.var.loc["cell_area", "measurement_units"] == "um"
    unit_um = volume.var.loc["cell_area", "unit"]
    assert unit_um.startswith("um^3"), unit_um
    assert "VOLUME" in unit_um, "a 3-D area column has to say it is a volume"
    # Not a second opinion invented here: the dictionary's own answer for
    # this column measured in these units.
    assert unit_um == feature_dict.parse_column("cell_area", "um").unit

    _root, px_db = px_project
    area, _ = ax.build_anndata(px_db, tables=OBJECT_TABLES, verbose=False)
    unit_px = area.var.loc["cell_area", "unit"]
    assert area.uns["spacr"]["measurement_units"] == "px"
    assert unit_px == "px^2 (pixel count)", unit_px
    assert unit_px != unit_um


def test_a_database_measured_twice_under_different_calibration_says_so(
        tmp_path):
    """Mixed units get the condition, not a majority vote.

    Two calibrations in one table make the geometric columns incomparable
    with each other. Reporting the commoner one would be a unit that is
    wrong for some of the rows and unmarked on all of them, so ``var``
    states every possibility with the condition it holds under, and
    ``uns`` records no units at all.
    """
    db = build_project(str(tmp_path / "mixed"), stamp=UM_STAMP)
    connection = sqlite3.connect(db)
    try:
        # A merged database: one plate measured in um, one re-measured after
        # the voxel size was corrected away.
        connection.execute(
            "UPDATE cell SET measurement_units='px' WHERE object_label=1")
        connection.commit()
    finally:
        connection.close()

    adata, _ = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False)

    assert adata.uns["spacr"]["measurement_units"] == ""
    unit = adata.var.loc["cell_area", "unit"]
    assert unit.startswith("depends on the row's measurement_units column")
    for units in ("px", "px_xy", "um"):
        assert f"measurement_units='{units}'" in unit, units
    assert adata.var.loc["cell_area", "measurement_units"] == ""


# ---------------------------------------------------------------------------
# a path that is not a database
# ---------------------------------------------------------------------------

def test_pointing_at_a_missing_database_says_so_and_creates_nothing(tmp_path):
    """``sqlite3.connect`` on an absent path CREATES it. That is the trap.

    A typo in ``src`` would otherwise leave an empty ``measurements.db``
    behind, and the next run -- or ``spacr doctor`` -- would find a database
    that exists, holds nothing, and looks like a failed measurement rather
    than a wrong path.
    """
    missing = str(tmp_path / "typo" / "measurements.db")
    with pytest.raises(ValueError, match="holds no tables"):
        ax.build_anndata(missing, verbose=False)
    assert not os.path.exists(missing), (
        "the export opened the path and left a database behind")
    assert not os.path.exists(os.path.dirname(missing))


def test_a_table_list_naming_nothing_in_the_database_lists_what_is_there(
        px_project):
    """The message names the tables that ARE present.

    ``tables=`` is a settings key a user edits by hand, so the answer to a
    misspelling has to be the correct spelling and not "nothing to export".
    """
    _root, db = px_project
    with pytest.raises(ValueError) as excinfo:
        ax.build_anndata(db, tables=("organelle_summary",), verbose=False)
    message = str(excinfo.value)
    assert "none of ['organelle_summary']" in message
    for present in OBJECT_TABLES:
        assert present in message


def test_a_table_with_no_measurements_in_it_is_refused_by_name(tmp_path):
    """An object table written without features is a diagnosis, not an empty X.

    AnnData would happily hold an ``n x 0`` matrix, and every question asked
    of it afterwards would return an empty answer with no explanation.
    """
    db = build_project(str(tmp_path / "bare"))
    connection = sqlite3.connect(db)
    try:
        # An organelle table whose measurement pass never ran: the identity
        # columns are there, the features are not.
        connection.execute(
            "CREATE TABLE organelle AS SELECT plateID, rowID, columnID, "
            "fieldID, object_label FROM cell")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError) as excinfo:
        ax.build_anndata(db, single_table="organelle", verbose=False)
    message = str(excinfo.value)
    assert "no feature columns found" in message
    assert db in message, "the message has to name the database it read"


def test_a_project_measured_without_crops_still_exports(tmp_path):
    """No ``png_list`` at all is a project, not a failure.

    ``crop_mode`` is optional in Measure, and the annotation guess in
    :mod:`spacr.agreement` reads ``png_list`` to make it. Losing the whole
    export over an absent optional table would cost the user every feature
    in the database to save them one hint.
    """
    db = build_project(str(tmp_path / "nocrops"), crops=False)
    assert "png_list" not in ax._available_tables(db)

    adata, result = ax.build_anndata(
        db, tables=ax.DEFAULT_TABLES, verbose=False)

    assert result.n_obs == len(FIELDS) * 3
    assert list(adata.uns["spacr"]["source_tables"]) == list(OBJECT_TABLES)
    assert list(adata.uns["spacr"]["annotation_columns"]) == []
    assert "cell_area" in set(adata.var_names)
    assert "png_path" not in set(adata.obs.columns)


# ---------------------------------------------------------------------------
# the labels png_list carries
# ---------------------------------------------------------------------------

def test_two_crops_of_one_object_that_disagree_attach_no_label_at_all(
        tmp_path):
    """A plausible wrong label is worse than a missing one.

    One object with two ``png_list`` rows annotated differently -- a plate
    annotated, re-cropped and annotated again -- has no single answer. Taking
    the first row would put a label in ``obs`` that the user can neither see
    is contested nor trace back, and it would be trained on.
    """
    db = build_project(str(tmp_path / "conflict"))
    annotate(db, [1, 0, 1, 1, 0, 1])
    duplicate_crop_row(db, "o1", infected=0)

    # Without png_list in the join, since the join itself refuses a repeated
    # object key (MergeCardinalityError) before this code is reached.
    adata, result = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False)

    assert "infected" not in set(adata.obs.columns)
    assert "infected" not in set(adata.var_names), "a label is never a feature"
    assert list(adata.uns["spacr"]["annotation_columns"]) == []
    assert not any("attached" in note for note in result.warnings)


def test_two_crops_of_one_object_that_agree_still_attach_the_label(tmp_path):
    """Re-cropping a plate must not silently cost it its annotations.

    The same duplication as above, with both rows saying the same thing.
    There is one answer, so the export gives it -- and gives the *other*
    objects' answers too, which is what a blanket "duplicates, give up"
    would have thrown away.
    """
    db = build_project(str(tmp_path / "agree"))
    annotate(db, [1, 0, 1, 1, 0, 1])
    duplicate_crop_row(db, "o1")

    adata, result = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False)

    assert "infected" in set(adata.obs.columns)
    assert [int(v) for v in adata.obs["infected"]] == [1, 0, 1, 1, 0, 1]
    assert list(adata.obs_names) == [
        "plate1_r1_c1_f1_cell1", "plate1_r1_c1_f1_cell2",
        "plate1_r1_c1_f1_cell3", "plate1_r1_c2_f1_cell1",
        "plate1_r1_c2_f1_cell2", "plate1_r1_c2_f1_cell3"]
    assert any("attached 1 label column" in note for note in result.warnings)


# ---------------------------------------------------------------------------
# timelapse
# ---------------------------------------------------------------------------

def test_a_timelapse_keys_and_labels_every_frame_separately(tmp_path):
    """One object at two frames is two observations, with two labels.

    Everything in a timelapse export hangs off this: the key carries the
    timepoint, so ``obs`` has a row per frame, and the annotation attached
    from ``png_list`` is the one made on *that* frame. Attaching frame 2's
    label to frame 1 would not be an error anywhere downstream -- it would
    be an experiment that reports infection before it happened.
    """
    db = build_project(str(tmp_path / "movie"), stems=TIMELAPSE_FIELDS,
                       timelapse=True)
    annotate(db, [0, 0, 0, 1, 1, 1])

    adata, result = ax.build_anndata(
        db, tables=ax.DEFAULT_TABLES, timelapse=True, verbose=False)

    assert list(adata.obs_names) == [
        "plate1_r1_c1_f1_t1_cell1", "plate1_r1_c1_f1_t1_cell2",
        "plate1_r1_c1_f1_t1_cell3", "plate1_r1_c1_f1_t2_cell1",
        "plate1_r1_c1_f1_t2_cell2", "plate1_r1_c1_f1_t2_cell3"]
    assert [int(v) for v in adata.obs["infected"]] == [0, 0, 0, 1, 1, 1]
    assert adata.uns["spacr"]["timelapse"] is True
    # The timepoint is a group a user plots by, not a number to average.
    assert str(adata.obs[schema.TIME_KEY].dtype) == "category"
    assert list(adata.obs[schema.TIME_KEY]) == ["t1"] * 3 + ["t2"] * 3
    assert result.n_obs == 6


def test_a_timelapse_read_as_a_still_refuses_to_collapse_its_frames(tmp_path):
    """Without ``timelapse=True`` the same object at T frames is one key.

    AnnData requires unique ``obs_names``, so this cannot pass silently --
    but it could pass *wrongly*, by keeping one frame or averaging them. It
    is refused instead, and the refusal names the argument that fixes it,
    because "deduplicate this" is the caller's decision and not the
    exporter's.
    """
    db = build_project(str(tmp_path / "movie"), stems=TIMELAPSE_FIELDS,
                       timelapse=True)

    with pytest.raises(ax.DuplicateObjectKeys) as excinfo:
        ax.build_anndata(db, tables=ax.DEFAULT_TABLES, verbose=False)
    message = str(excinfo.value)
    assert "3 of 6 rows repeat an object key" in message
    assert "timelapse=True" in message
    assert "plate1_r1_c1_f1_cell1" in message


# ---------------------------------------------------------------------------
# provenance that has to survive HDF5
# ---------------------------------------------------------------------------

def test_a_settings_value_hdf5_cannot_store_does_not_cost_the_export(
        tmp_path):
    """``uns`` is written through HDF5, which refuses None and objects.

    A settings dict carries whatever the caller put in it -- a None
    threshold, a model instance, a tuple of channels. ``h5py`` raises on the
    first two, and the export is finished by then: the matrix is built and
    the user is waiting. Losing all of it to a provenance field would be
    absurd, so the unstorable values are coerced and the file is written.
    """
    class _Model:
        def __repr__(self):
            return "<FakeModel n=3>"

    db = build_project(str(tmp_path / "prov"))
    out = str(tmp_path / "results" / "prov.h5ad")
    settings = {"nucleus_threshold": None, "model": _Model(),
                "channels": (0, 1), "normalize": True,
                "channel_weights": np.array([0.5, 1.5])}

    result = ax.export_anndata(db, out, tables=OBJECT_TABLES,
                               settings=settings, register=False,
                               verbose=False)

    assert os.path.isfile(result.path)
    stored = anndata.read_h5ad(out).uns["spacr_settings"]
    assert stored["nucleus_threshold"] == "", "None became an empty string"
    assert stored["model"] == "<FakeModel n=3>", "the object became its repr"
    assert list(stored["channels"]) == [0, 1]
    assert bool(stored["normalize"]) is True


def test_a_database_with_no_settings_history_exports_with_an_empty_run_id(
        px_project):
    """No run to attribute the data to is an empty string, not a failure.

    ``settings_history`` is written by ``_save_settings_to_db``, which a
    database assembled by hand (or by an older spaCR) may never have had.
    The export still has every other piece of provenance, so it says the run
    id is unknown rather than refusing to describe the file at all.
    """
    _root, db = px_project
    assert "settings_history" not in ax._available_tables(db)

    adata, _ = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False)

    assert adata.uns["spacr"]["run_id"] == ""
    assert adata.uns["spacr"]["source_database"] == db
    assert adata.uns["spacr"]["spacr_version"]


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def test_a_one_dimensional_embedding_becomes_one_obsm_column(px_project):
    """``obsm`` is 2-D; a per-object score is a column of it, not a failure.

    A pseudotime, a classifier score or any single derived coordinate
    arrives as a flat array. Refusing it would push the caller into
    reshaping by hand, which is where an off-by-one transposition lands.
    """
    _root, db = px_project
    scores = np.arange(6, dtype=float) * 0.25

    adata, result = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False,
                                     embeddings={"pseudotime": scores})

    assert list(adata.obsm.keys()) == ["X_pseudotime"]
    assert adata.obsm["X_pseudotime"].shape == (6, 1)
    np.testing.assert_allclose(
        adata.obsm["X_pseudotime"].ravel(), scores.astype(np.float32))
    assert result.obsm_keys == ("X_pseudotime",)


def test_a_keyed_embedding_with_no_coordinates_is_refused(px_project):
    """Key columns and nothing else is a mistake with a silent success mode.

    A frame carrying the five key columns is aligned *by key*, so a caller
    who passed the key frame instead of the embedding would otherwise get an
    ``(n, 0)`` obsm entry -- perfectly valid, perfectly useless, and only
    noticed when the plot comes out empty.
    """
    _root, db = px_project
    connection = sqlite3.connect(db)
    try:
        keys = pd.read_sql(
            "SELECT plateID, rowID, columnID, fieldID, object_label "
            "FROM cell", connection)
    finally:
        connection.close()

    with pytest.raises(ValueError) as excinfo:
        ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False,
                         embeddings={"umap": keys})
    message = str(excinfo.value)
    assert "embedding 'umap'" in message
    assert "no numeric coordinate columns" in message


# ---------------------------------------------------------------------------
# missing values
# ---------------------------------------------------------------------------

def test_asking_for_the_missing_mask_under_keep_marks_exactly_the_nan(
        px_project):
    """``missing_layer=True`` is honoured when nothing was imputed.

    The mask defaults on for the imputing policies, but a caller keeping the
    NaN may want it too: it is the only per-cell record that survives
    ``sc.pp.scale``, which overwrites X and takes the NaN pattern with it.
    An all-False mask written under ``keep`` would be worse than none.
    """
    _root, db = px_project
    adata, _ = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False,
                                missing_layer=True)

    mask = np.asarray(adata.layers["missing"])
    matrix = np.asarray(adata.X, dtype=float)
    assert mask.dtype == np.dtype(bool)
    assert mask.shape == adata.shape
    assert bool((mask == np.isnan(matrix)).all())
    # The last cell of each field has no pathogen row, so exactly the
    # pathogen columns of those two objects are missing.
    pathogen = [i for i, name in enumerate(adata.var_names)
                if name.startswith("pathogen_")]
    assert int(mask.sum()) == 2 * len(pathogen)
    assert bool(mask[:, pathogen][[2, 5]].all())


def test_the_summary_line_names_the_infinities_it_converted(tmp_path):
    """+/-inf is treated as missing, and ``describe()`` says it happened.

    An inf comes out of a ratio feature with a zero denominator. It survives
    ``dropna`` and then destroys any scaling, PCA or distance computed from
    it, so it is converted -- and a silent conversion of a measured value is
    exactly the kind of thing a provenance record exists to prevent.
    """
    db = build_project(str(tmp_path / "inf"))
    connection = sqlite3.connect(db)
    try:
        connection.execute(
            "UPDATE cell SET cell_area = 1e400 WHERE object_label = 1")
        connection.commit()
    finally:
        connection.close()

    adata, result = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False)

    assert result.n_infinite == 2
    assert "2 non-finite values treated as missing" in result.describe()
    assert not np.isinf(np.asarray(adata.X, dtype=float)).any()
    assert int(adata.var.loc["cell_area", "n_infinite"]) == 2
    assert int(adata.uns["spacr"]["nan"]["n_missing"]) >= 2


#: ``ExportResult.frac_missing`` documents itself as "NaN as a fraction of
#: the pre-policy matrix", and ``n_missing`` really is the pre-policy count
#: -- but ``n_obs``/``n_vars`` are the shape AFTER the policy dropped rows or
#: columns. Under ``drop_objects`` on the database below that is 28 missing
#: values over a 2 x 11 matrix, and ``describe()`` prints "(127.3%)".
FRAC_MISSING_DENOMINATOR_BUG = (
    "ExportResult.frac_missing divides the pre-policy n_missing by the "
    "post-policy n_obs * n_vars, so a dropping policy reports a fraction "
    "above 100%")


@pytest.mark.xfail(strict=True, reason=FRAC_MISSING_DENOMINATOR_BUG)
def test_frac_missing_is_a_fraction_of_the_matrix_it_counted(tmp_path):
    """A fraction of missing values cannot exceed 1.0.

    Whatever the two numbers are measured over, the printed percentage is
    read as "how much of my data was missing", and 127.3% is not an answer
    to that question. The same export with the default policy reports 42.4%,
    which is the true figure for the matrix the values were counted in.
    """
    db = build_project(str(tmp_path / "sparse"), pathogen_cells=1,
                       pathogen_channels=6)
    _adata, dropped = ax.build_anndata(
        db, tables=OBJECT_TABLES, nan_policy=ax.NAN_DROP_OBJECTS,
        verbose=False)
    _kept, kept = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False)

    # These three are SETUP, not the claim. They were wrong -- (2, 11), 28,
    # 28/(6*11) -- so the test died here and never reached the assertion
    # below, which is the one that encodes the defect. A strict xfail that
    # fails for the wrong reason is green and pins nothing.
    assert (dropped.n_obs, dropped.n_vars) == (2, 14)
    assert dropped.n_missing == kept.n_missing == 32
    assert kept.frac_missing == pytest.approx(32 / (6 * 14))
    # THE CLAIM: a fraction of missing values cannot exceed 1.0. Measured at
    # 1.1428 because the pre-policy numerator is divided by the post-policy
    # shape.
    assert dropped.frac_missing <= 1.0


# ---------------------------------------------------------------------------
# the UMAP guard
# ---------------------------------------------------------------------------

def test_compute_umap_below_three_objects_says_so_instead_of_reducing(
        px_project):
    """UMAP needs at least 3 points; two is a note, not a traceback.

    ``compute_umap`` is a settings checkbox, and ``row_limit`` is the "give
    me something I can open" knob. The two together are an ordinary thing to
    ask for, and the reducer would raise from inside umap-learn on a matrix
    it cannot embed. The export is finished either way -- what is missing is
    one obsm key, and the result says which and why.
    """
    _root, db = px_project
    reducer_was_imported = "umap" in sys.modules
    adata, result = ax.build_anndata(db, tables=OBJECT_TABLES, verbose=False,
                                     row_limit=2, compute_umap=True)

    assert adata.n_obs == 2
    assert list(adata.obsm.keys()) == []
    assert result.obsm_keys == ()
    assert any("UMAP needs at least 3" in note for note in result.warnings)
    # `uns['spacr']['umap']` is written only when the reducer actually ran.
    assert "umap" not in adata.uns["spacr"]
    assert ("umap" in sys.modules) is reducer_was_imported, (
        "the reducer must not be imported when it cannot be run")


# ---------------------------------------------------------------------------
# where the artifact is registered
# ---------------------------------------------------------------------------

def test_an_explicit_project_root_is_where_the_export_is_registered(tmp_path):
    """``project=`` overrides the layout guess, and nothing else registers it.

    The root is normally read off ``<project>/measurements/measurements.db``,
    which is right until someone exports a database they copied somewhere.
    ``spacr.artifacts.is_stale`` answers per project, so a record filed under
    the wrong root is a staleness question nobody can ask.
    """
    from spacr import artifacts

    db = build_project(str(tmp_path / "source"))
    elsewhere = str(tmp_path / "analysis")
    os.makedirs(elsewhere, exist_ok=True)

    result = ax.export_anndata(db, os.path.join(elsewhere, "e.h5ad"),
                               tables=OBJECT_TABLES, project=elsewhere,
                               verbose=False)

    assert result.artifact_id
    filed = artifacts.by_kind(ax.ANNDATA_KIND, project=elsewhere)
    assert [record.artifact_id for record in filed] == [result.artifact_id]
    assert filed[0].extra["n_obs"] == result.n_obs
    assert filed[0].extra["source_database"] == db
    assert artifacts.by_kind(ax.ANNDATA_KIND,
                             project=str(tmp_path / "source")) == []


def test_a_database_outside_a_measurements_folder_registers_beside_itself(
        tmp_path):
    """A loose ``.db`` has no project above it, so it is its own root.

    ``<project>/measurements/measurements.db`` is what every spaCR writer
    leaves, and taking "two directories up" unconditionally would file the
    artifact one level above a database somebody copied to a scratch folder
    -- a directory that may hold three unrelated experiments.
    """
    from spacr import artifacts

    db = build_project(str(tmp_path / "source"))
    loose_dir = str(tmp_path / "scratch")
    os.makedirs(loose_dir, exist_ok=True)
    loose = os.path.join(loose_dir, "copy.db")
    shutil.copyfile(db, loose)

    result = ax.export_anndata(loose, os.path.join(loose_dir, "copy.h5ad"),
                               tables=OBJECT_TABLES, verbose=False)

    assert result.artifact_id
    filed = artifacts.by_kind(ax.ANNDATA_KIND, project=loose_dir)
    assert [record.artifact_id for record in filed] == [result.artifact_id]
    assert artifacts.by_kind(ax.ANNDATA_KIND, project=str(tmp_path)) == []


# ---------------------------------------------------------------------------
# the Qt sidebar row
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_the_registered_app_row_runs_the_headless_entry_point():
    """The Run button and ``spacr-run anndata_export`` must be one function.

    The row registers no ``factory``, so the generic settings screen IS the
    export dialog and ``entry=`` is the whole of its behaviour. A typo there
    resolves to None and the app is "Not runnable" forever, which no import
    and no test of the module itself would notice.
    """
    pytest.importorskip("PySide6")
    import spacr.qt.app as app

    assert any(row[0] == ax.APP_KEY for row in app.APPS), (
        "spacr.qt.app names register_anndata_app in _SELF_REGISTERING_APPS, "
        "so importing it must have put the row there")
    # Already registered: a second call is a no-op, not a duplicate row.
    assert ax.register_anndata_app() is False
    assert ax.register_anndata_app(replace=True) is True
    assert sum(row[0] == ax.APP_KEY for row in app.APPS) == 1

    assert app.registered_entry(ax.APP_KEY) is ax.run_anndata_export
    meta = app.APP_META[ax.APP_KEY]
    assert meta["defaults_module"] == "spacr.anndata_export", (
        "without it the settings form has no keys to draw")
    assert meta["api_module"] == "anndata_export"
    row = next(row for row in app.APPS if row[0] == ax.APP_KEY)
    assert row[3] == app.SECTION_EXPLORE

    from spacr.qt import i18n
    assert len(ax.APP_TRANSLATIONS) == len(i18n.LANGUAGES) - 1, (
        "one translation per non-English UI language, in LANGUAGES order")
