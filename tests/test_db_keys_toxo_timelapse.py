"""Database-key contract for ``spacr.toxo`` and ``spacr.timelapse``.

Every database in this module is built by the REAL spaCR writers --
:func:`spacr.utils._merge_and_save_to_database` and
:func:`spacr.utils.filepaths_to_database` -- because that is how the bugs
covered here survived: the previous tests hand-wrote a sqlite file using the
same wrong column names the readers used, so reader and fixture agreed with
each other and disagreed with every real ``measurements.db``.

The canonical vocabulary the writers emit::

    plateID  rowID ('r<N>')  columnID ('c<N>')  fieldID ('f<N>')
    timeID ('t<N>', object tables, timelapse runs only)
    object_label (INTEGER)
    cell_id (INTEGER in nucleus/pathogen/organelle; TEXT 'o<N>' in png_list)
    prcf  = plate_row_column_field[_time]
    prcfo = prcf_o<N>

``png_list`` spells the timepoint column ``time_id`` while the object tables
spell it ``timeID``; both are accepted on read.

What is pinned here:

* ``preprocess_pathogen_data`` and ``analyze_calcium_oscillations`` used to key
  on ``column_name``, ``timeid`` and ``pathogen_cell_id`` -- three names no
  writer has ever produced -- so both raised ``KeyError`` against a real
  database. The row counts below matter more than the absence of an exception:
  a left merge on a wrong-but-present key would match nothing and leave a
  silently empty ``parasite_count``/``cytoplasm_area``, which is exactly the
  failure mode the fix is defending against.
* A non-timelapse database has no timepoint column at all, so the time key is
  resolved per frame rather than assumed.
* ``spacr.toxo.generate_score_heatmap`` was a stale copy of
  ``spacr.submodules.generate_score_heatmap`` left behind by the
  ``column_name`` -> ``columnID`` rename; it now delegates, so the two must
  agree exactly.
* ``settings['db_table_name']`` is free text from the GUI and its table is
  written with ``if_exists='replace'``. Typing ``cell`` used to destroy the
  measurement table.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_TIME = 6
WELLS = (("A01", "r1"), ("B01", "r2"))
N_CELLS = 3
N_PARASITES = 2          # both inside host cell 1
COLUMN, FIELD, PLATE = "c1", "f1", "plate1"
MEASUREMENT = "cell_channel_1_mean_intensity"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# A measurements.db built by the real writers
# ---------------------------------------------------------------------------

def _intensities(t):
    """Photobleaching envelope, with one calcium spike in cell 1 at t == 4."""
    base = 1000.0 * np.exp(-0.05 * t)
    return [base * (1.4 if t == 4 else 1.0), base * 0.9, base * 1.1]


def _build_measurements_db(root, timelapse=True):
    """Write ``<root>/measurements/measurements.db`` with the real writers.

    :param root: experiment folder (created if needed).
    :param timelapse: when True the file names carry a time index, so the
        writer stamps a ``timeID`` column; when False it writes none at all.
    :returns: path to the database.
    """
    from spacr.utils import _merge_and_save_to_database, filepaths_to_database

    root = str(root)
    os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
    times = list(range(1, N_TIME + 1)) if timelapse else [None]

    for t in times:
        for well, _row in WELLS:
            stem = f"{PLATE}_{well}_1" + (f"_{t}" if timelapse else "")
            tt = 1 if t is None else t

            cell_morph = pd.DataFrame({
                "label": list(range(1, N_CELLS + 1)),
                # constant per track, so the size filter keeps every cell
                "cell_area": [500.0, 480.0, 520.0],
            })
            cell_int = pd.DataFrame({
                "label": list(range(1, N_CELLS + 1)),
                MEASUREMENT: _intensities(tt),
            })
            _merge_and_save_to_database(cell_morph, cell_int, "cell", root,
                                        stem, "exp", timelapse=timelapse)

            pathogen_morph = pd.DataFrame({
                "label": list(range(1, N_PARASITES + 1)),
                "cell_id": [1] * N_PARASITES,      # both parasites in cell 1
                "pathogen_area": [30.0, 31.0],
            })
            pathogen_int = pd.DataFrame({
                "label": list(range(1, N_PARASITES + 1)),
                "pathogen_channel_1_mean_intensity": [5.0, 6.0],
            })
            _merge_and_save_to_database(pathogen_morph, pathogen_int,
                                        "pathogen", root, stem, "exp",
                                        timelapse=timelapse)

            cyto_morph = pd.DataFrame({
                "label": list(range(1, N_CELLS + 1)),
                "cytoplasm_area": [200.0, 210.0, 220.0],
            })
            cyto_int = pd.DataFrame({
                "label": list(range(1, N_CELLS + 1)),
                "cytoplasm_channel_1_mean_intensity": [1.0, 2.0, 3.0],
            })
            _merge_and_save_to_database(cyto_morph, cyto_int, "cytoplasm",
                                        root, stem, "exp", timelapse=timelapse)

    png_paths = [
        os.path.join(root, "cell_png",
                     f"{PLATE}_{well}_1" + (f"_{t}" if timelapse else "")
                     + f"_{obj}.png")
        for t in times for well, _row in WELLS
        for obj in range(1, N_CELLS + 1)
    ]
    filepaths_to_database(png_paths, {"timelapse": timelapse}, root, "cell")

    return os.path.join(root, "measurements", "measurements.db")


def _read(db_path, table):
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM {table}", con)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# The writers really do emit the canonical vocabulary
# ---------------------------------------------------------------------------

def test_real_writers_emit_the_canonical_key_vocabulary(tmp_path):
    """Anchors every expectation below on what the writers actually produce."""
    db = _build_measurements_db(tmp_path / "tl", timelapse=True)

    cell = _read(db, "cell")
    assert {"plateID", "rowID", "columnID", "fieldID", "timeID",
            "prcf", "object_label"}.issubset(cell.columns)
    assert "column_name" not in cell.columns
    assert "timeid" not in cell.columns
    assert set(cell["rowID"]) == {"r1", "r2"}
    assert set(cell["columnID"]) == {COLUMN}
    assert set(cell["timeID"]) == {f"t{i}" for i in range(1, N_TIME + 1)}
    assert set(cell["prcf"]) == {f"{PLATE}_{row}_{COLUMN}_{FIELD}_t{i}"
                                 for _w, row in WELLS
                                 for i in range(1, N_TIME + 1)}

    pathogen = _read(db, "pathogen")
    # the host-cell link is 'cell_id', never 'pathogen_cell_id'
    assert "cell_id" in pathogen.columns
    assert "pathogen_cell_id" not in pathogen.columns

    png = _read(db, "png_list")
    # png_list historically spelled the timepoint 'time_id' while every object
    # table spelled it 'timeID'. Readers accept either, so assert through the
    # resolver rather than pinning whichever spelling the writer emits today.
    from spacr.timelapse import _resolve_time_key
    assert _resolve_time_key(png) in ("timeID", "time_id")
    assert set(png["cell_id"]) == {"o1", "o2", "o3"}
    assert png["prcfo"].iloc[0].startswith(f"{PLATE}_r1_{COLUMN}_{FIELD}_t")


def test_non_timelapse_database_has_no_timepoint_column(tmp_path):
    db = _build_measurements_db(tmp_path / "plain", timelapse=False)
    cell = _read(db, "cell")
    assert not {"timeID", "time_id", "timeid"} & set(cell.columns)
    assert set(cell["prcf"]) == {f"{PLATE}_{row}_{COLUMN}_{FIELD}"
                                 for _w, row in WELLS}


# ---------------------------------------------------------------------------
# preprocess_pathogen_data
# ---------------------------------------------------------------------------

def test_preprocess_pathogen_data_on_a_real_timelapse_table(tmp_path):
    """One row per (well, timepoint, host cell), with the parasite count."""
    from spacr.timelapse import preprocess_pathogen_data

    db = _build_measurements_db(tmp_path / "tl", timelapse=True)
    pathogen = _read(db, "pathogen")
    assert len(pathogen) == N_TIME * len(WELLS) * N_PARASITES

    out = preprocess_pathogen_data(pathogen)

    # both parasites collapse into their single host cell, per timepoint
    assert len(out) == N_TIME * len(WELLS)
    assert set(out["parasite_count"]) == {N_PARASITES}
    # cell_id became object_label so it merges onto the cell table directly
    assert "cell_id" not in out.columns
    assert set(out["object_label"]) == {1}
    assert set(out["timeID"]) == {f"t{i}" for i in range(1, N_TIME + 1)}
    # the parasite's own object_label is dropped, its area is averaged
    assert out["pathogen_area"].iloc[0] == pytest.approx(30.5)


def test_preprocess_pathogen_data_on_a_real_non_timelapse_table(tmp_path):
    """No timeID column at all: the group keys must adapt, not assume."""
    from spacr.timelapse import preprocess_pathogen_data

    db = _build_measurements_db(tmp_path / "plain", timelapse=False)
    pathogen = _read(db, "pathogen")
    assert "timeID" not in pathogen.columns

    out = preprocess_pathogen_data(pathogen)

    assert len(out) == len(WELLS)
    assert set(out["parasite_count"]) == {N_PARASITES}
    assert set(out["object_label"]) == {1}
    assert set(out["rowID"]) == {"r1", "r2"}


def test_preprocess_pathogen_data_names_the_missing_identifier(tmp_path):
    """A frame keyed the old way gets told which columns are missing."""
    from spacr.timelapse import preprocess_pathogen_data

    legacy = pd.DataFrame({
        "plateID": [PLATE] * 2, "rowID": ["r1"] * 2,
        "column_name": [COLUMN] * 2, "fieldID": [FIELD] * 2,
        "pathogen_cell_id": [1, 1], "object_label": [1, 2],
        "pathogen_area": [30.0, 31.0],
    })
    with pytest.raises(KeyError) as excinfo:
        preprocess_pathogen_data(legacy)
    message = str(excinfo.value)
    assert "columnID" in message and "cell_id" in message


def test_object_group_keys_accepts_the_png_list_time_spelling():
    """png_list spells the timepoint 'time_id'; readers accept both."""
    from spacr.timelapse import _object_group_keys, _resolve_time_key

    frame = pd.DataFrame({
        "plateID": [PLATE], "rowID": ["r1"], "columnID": [COLUMN],
        "fieldID": [FIELD], "time_id": ["t1"], "cell_id": ["o1"],
    })
    assert _resolve_time_key(frame) == "time_id"
    assert _object_group_keys(frame, "cell_id") == [
        "plateID", "rowID", "columnID", "fieldID", "time_id", "cell_id"]


def test_resolve_time_key_prefers_the_object_table_spelling():
    from spacr.timelapse import _resolve_time_key

    assert _resolve_time_key(
        pd.DataFrame(columns=["timeID", "time_id", "timeid"])) == "timeID"
    assert _resolve_time_key(pd.DataFrame(columns=["timeid"])) == "timeid"
    assert _resolve_time_key(pd.DataFrame(columns=["prcf"])) is None


# ---------------------------------------------------------------------------
# analyze_calcium_oscillations
# ---------------------------------------------------------------------------

def test_analyze_calcium_oscillations_merges_on_a_real_timelapse_db(tmp_path,
                                                                    capsys):
    """The pathogen and cytoplasm merges must MATCH, not merely not raise.

    A left merge on a wrong key returns the same number of rows with every
    merged-in column null, so the row count alone proves nothing -- the
    non-null counts below are what distinguish a working merge from the
    silent no-op the old key names produced.
    """
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_measurements_db(tmp_path / "tl", timelapse=True)
    n_rows = N_TIME * len(WELLS) * N_CELLS
    assert len(_read(db, "cell")) == n_rows

    out = analyze_calcium_oscillations(db, pathogen="pathogen",
                                       cytoplasm="cytoplasm")
    assert out is not None
    result_df, peak_details_df, fig = out

    # the merges neither dropped nor duplicated a single cell row
    printed = capsys.readouterr().out
    assert f"After pathogen merge: {n_rows} objects" in printed
    assert f"After cytoplasm merge: {n_rows} objects" in printed

    # every cell survives the size + transience filters
    assert len(result_df) == n_rows
    assert result_df["plate_row_column_field_object"].nunique() == \
        len(WELLS) * N_CELLS

    # --- the pathogen merge actually matched ------------------------------
    infected = result_df[result_df["parasite_count"] > 0]
    assert len(infected) == N_TIME * len(WELLS)      # cell 1 of each well
    assert set(infected["parasite_count"]) == {N_PARASITES}
    assert set(infected["object_label"]) == {1}

    # --- the cytoplasm merge actually matched -----------------------------
    assert result_df["cytoplasm_area"].notna().all()
    assert set(result_df["cytoplasm_area"]) == {200.0, 210.0, 220.0}

    # --- identifiers exploded from prcf -----------------------------------
    assert set(result_df["columnID"]) == {COLUMN}
    assert set(result_df["rowID"]) == {"r1", "r2"}
    assert sorted(result_df["time"].unique()) == list(range(1, N_TIME + 1))

    # the injected spike at t == 4 is found in cell 1 of both wells
    spikes = peak_details_df[peak_details_df["time"] == 4]
    assert set(spikes["ID"]) == {f"{PLATE}_{row}_{COLUMN}_{FIELD}_o1"
                                 for _w, row in WELLS}
    assert (spikes["infected"] == N_PARASITES).all()

    assert len(fig.axes) == 1


def test_analyze_calcium_oscillations_track_key_carries_the_o_prefix(tmp_path):
    """The per-track key spells its object index the way prcfo does."""
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_measurements_db(tmp_path / "tl", timelapse=True)
    result_df, _peaks, _fig = analyze_calcium_oscillations(db)

    assert set(result_df["plate_row_column_field_object"]) == {
        f"{PLATE}_{row}_{COLUMN}_{FIELD}_o{obj}"
        for _w, row in WELLS for obj in range(1, N_CELLS + 1)}


def test_analyze_calcium_oscillations_without_optional_tables(tmp_path):
    """pathogen=None leaves every cell uninfected rather than raising."""
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_measurements_db(tmp_path / "tl", timelapse=True)
    result_df, peak_details_df, _fig = analyze_calcium_oscillations(db)

    assert len(result_df) == N_TIME * len(WELLS) * N_CELLS
    assert (result_df["parasite_count"] == 0).all()
    assert (peak_details_df["infected"] == 0).all()


def test_analyze_calcium_oscillations_on_a_non_timelapse_db(tmp_path, capsys):
    """No time axis means nothing to analyse -- said out loud, not a KeyError."""
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_measurements_db(tmp_path / "plain", timelapse=False)

    assert analyze_calcium_oscillations(db, pathogen="pathogen",
                                        cytoplasm="cytoplasm") is None

    printed = capsys.readouterr().out
    # the merges still ran and still matched, on the shorter key
    assert f"After pathogen merge: {len(WELLS) * N_CELLS} objects" in printed
    assert "No time axis" in printed
    assert "timeID" in printed


def _drop_column(db_path, table, column):
    """Rewrite ``table`` without ``column``.

    Read-modify-rewrite rather than ``ALTER TABLE ... DROP COLUMN``, which
    needs SQLite 3.35+ and this suite should not depend on the bundled
    library's vintage.
    """
    frame = _read(db_path, table).drop(columns=[column])
    con = sqlite3.connect(db_path)
    try:
        frame.to_sql(table, con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()


def test_analyze_calcium_oscillations_reads_time_from_prcf(tmp_path):
    """A database whose cell table lost timeID still has t<N> in prcf."""
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_measurements_db(tmp_path / "tl", timelapse=True)
    _drop_column(db, "cell", "timeID")
    assert "timeID" not in _read(db, "cell").columns

    result_df, _peaks, _fig = analyze_calcium_oscillations(db)
    assert sorted(result_df["time"].unique()) == list(range(1, N_TIME + 1))


def test_analyze_calcium_oscillations_pathogen_without_cell_id(tmp_path):
    """A pathogen table measured with no cell mask cannot be attributed."""
    from spacr.timelapse import analyze_calcium_oscillations

    db = _build_measurements_db(tmp_path / "tl", timelapse=True)
    _drop_column(db, "pathogen", "cell_id")

    with pytest.raises(KeyError) as excinfo:
        analyze_calcium_oscillations(db, pathogen="pathogen")
    message = str(excinfo.value)
    assert "cell_id" in message and "cell_mask_dim" in message


# ---------------------------------------------------------------------------
# generate_score_heatmap: one implementation, two import paths
# ---------------------------------------------------------------------------

ROWS = [f"r{i}" for i in range(1, 7)]
HEATMAP_COLUMN = "c3"
CONTROL_SGRNAS = ["sgA", "sgB"]


def _heatmap_inputs(tmp_path):
    """Write canonical (``columnID``-keyed) score and read-count CSVs."""
    rng = np.random.default_rng(7)

    folder = tmp_path / "models"
    for i, model in enumerate(("modelA", "modelB")):
        sub = folder / model
        sub.mkdir(parents=True)
        pd.DataFrame({
            "columnID": [HEATMAP_COLUMN] * len(ROWS),
            "rowID": ROWS,
            "pred": rng.uniform(0, 1, len(ROWS)),
        }).to_csv(sub / "scores.csv", index=False)

    cv = tmp_path / "cv.csv"
    pd.DataFrame({
        "columnID": [HEATMAP_COLUMN] * len(ROWS),
        "rowID": ROWS,
        "pred_cv": rng.uniform(0, 1, len(ROWS)),
    }).to_csv(cv, index=False)

    counts = tmp_path / "counts.csv"
    pd.DataFrame([
        {"columnID": HEATMAP_COLUMN, "rowID": row, "grna_name": grna,
         "count": int(rng.integers(10, 500))}
        for row in ROWS for grna in CONTROL_SGRNAS
    ]).to_csv(counts, index=False)

    return {
        "folders": [str(folder)], "csv_name": "scores.csv",
        "data_column": "pred", "csv": str(counts), "cv_csv": str(cv),
        "data_column_cv": "pred_cv", "plateID": 1,
        "columnID": HEATMAP_COLUMN, "control_sgrnas": CONTROL_SGRNAS,
        "fraction_grna": "sgA", "cmap": "viridis", "dst": None,
    }


def test_generate_score_heatmap_agrees_between_toxo_and_submodules(tmp_path):
    """toxo's copy was stale; it now delegates, so both must agree exactly."""
    from spacr.submodules import generate_score_heatmap as sub_heatmap
    from spacr.toxo import generate_score_heatmap as toxo_heatmap

    settings = _heatmap_inputs(tmp_path)

    from_sub = sub_heatmap(dict(settings))
    from_toxo = toxo_heatmap(dict(settings))

    assert isinstance(from_toxo, pd.DataFrame)
    assert not from_toxo.empty
    # keyed on columnID throughout, never on the pre-rename column_name
    assert "columnID" in from_toxo.columns
    assert "column_name" not in from_toxo.columns
    pd.testing.assert_frame_equal(from_toxo, from_sub)


def test_generate_score_heatmap_from_toxo_defaults_the_colormap(tmp_path):
    """The old toxo copy hard-coded viridis; settings without cmap still work."""
    from spacr.toxo import generate_score_heatmap as toxo_heatmap

    settings = _heatmap_inputs(tmp_path)
    del settings["cmap"]

    out = toxo_heatmap(settings)
    assert isinstance(out, pd.DataFrame) and not out.empty
    # the caller's dict is not mutated by the default
    assert "cmap" not in settings


def test_generate_score_heatmap_from_toxo_writes_the_artifacts(tmp_path):
    from spacr.toxo import generate_score_heatmap as toxo_heatmap

    settings = _heatmap_inputs(tmp_path)
    dst = tmp_path / "out"
    dst.mkdir()
    settings["dst"] = str(dst)

    toxo_heatmap(settings)

    assert (dst / "scores_comparison_plate_1.pdf").is_file()
    saved = pd.read_csv(dst / "scores_comparison_plate_1_data.csv")
    assert "columnID" in saved.columns
    mae = pd.read_csv(dst / "mae_scores_comparison_plate_1.csv")
    assert set(mae["Channel"]) == {"modelA_pred", "modelB_pred", "pred_cv"}


# ---------------------------------------------------------------------------
# db_table_name may not name a spaCR-owned table
# ---------------------------------------------------------------------------

def _motility_frames():
    all_df = pd.DataFrame({
        "plateID": [PLATE] * 4, "wellID": ["A01", "A01", "B01", "B01"],
        "fieldID": [1, 1, 1, 1], "cellID": [1, 2, 1, 2],
        "frame": [0, 0, 0, 0], "velocity": [1.0, 2.0, 3.0, 4.0],
    })
    well_df = pd.DataFrame({
        "plateID": [PLATE, PLATE], "wellID": ["A01", "B01"],
        "n_tracks": [2, 2], "mean_velocity_all": [1.5, 3.5],
    })
    return all_df, well_df


@pytest.mark.parametrize("name", [
    "cell", "nucleus", "pathogen", "cytoplasm", "organelle",
    "png_list", "object_counts", "settings", "run_status",
])
def test_save_measurements_refuses_every_spacr_owned_table(tmp_path, name):
    from spacr.timelapse import _save_measurements_and_well_summary

    all_df, well_df = _motility_frames()
    with pytest.raises(ValueError) as excinfo:
        _save_measurements_and_well_summary(all_df, well_df, str(tmp_path), name)

    message = str(excinfo.value)
    assert "db_table_name" in message           # names the setting
    assert repr(name) in message
    assert "if_exists='replace'" in message     # says why
    assert "timelapse_object_measurements" in message


def test_save_measurements_refuses_a_reserved_name_case_insensitively(tmp_path):
    """SQLite table names are case-insensitive, so 'CELL' replaces 'cell'."""
    from spacr.timelapse import _save_measurements_and_well_summary

    all_df, well_df = _motility_frames()
    for spelling in ("CELL", " Cell "):
        with pytest.raises(ValueError, match="db_table_name"):
            _save_measurements_and_well_summary(all_df, well_df,
                                                str(tmp_path), spelling)


def test_reserved_db_table_name_refusal_leaves_the_cell_table_intact(tmp_path):
    """The whole point: the measurement table survives the attempt."""
    from spacr.timelapse import _save_measurements_and_well_summary

    src = tmp_path / "plate"
    db = _build_measurements_db(src, timelapse=True)
    before = _read(db, "cell")
    assert len(before) == N_TIME * len(WELLS) * N_CELLS

    all_df, well_df = _motility_frames()
    with pytest.raises(ValueError):
        _save_measurements_and_well_summary(all_df, well_df, str(src), "cell")

    pd.testing.assert_frame_equal(_read(db, "cell"), before)


def test_save_measurements_writes_a_legitimate_table(tmp_path):
    """A name of the user's own still writes, alongside the measurements."""
    from spacr.timelapse import _save_measurements_and_well_summary

    src = tmp_path / "plate"
    db = _build_measurements_db(src, timelapse=True)
    cell_before = _read(db, "cell")

    all_df, well_df = _motility_frames()
    measurements_dir, db_path = _save_measurements_and_well_summary(
        all_df, well_df, str(src), "timelapse_object_measurements")

    assert db_path == db
    assert measurements_dir == os.path.join(str(src), "measurements")
    pd.testing.assert_frame_equal(_read(db, "timelapse_object_measurements"),
                                  all_df)
    assert len(_read(db, "timelapse_object_measurements_well_motility")) == 2
    # the spaCR-owned tables are untouched
    pd.testing.assert_frame_equal(_read(db, "cell"), cell_before)


def test_save_measurements_with_no_well_summary_skips_the_companion(tmp_path):
    from spacr.timelapse import _save_measurements_and_well_summary

    src = tmp_path / "plate"
    all_df, _well_df = _motility_frames()
    _save_measurements_and_well_summary(all_df, pd.DataFrame(), str(src),
                                        "motility_measurements")

    db = os.path.join(str(src), "measurements", "measurements.db")
    con = sqlite3.connect(db)
    try:
        tables = {row[0] for row in
                  con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()
    assert tables == {"motility_measurements"}


def test_automated_motility_assay_refuses_a_reserved_table_up_front(tmp_path):
    """Refused before the hours of regionprops, not after them."""
    from spacr.timelapse import automated_motility_assay

    # no merged/ directory at all: reaching the FileNotFoundError below would
    # mean the guard fired too late to be worth anything.
    with pytest.raises(ValueError, match="db_table_name"):
        automated_motility_assay({"src": str(tmp_path), "db_table_name": "cell"})


def test_automated_motility_assay_accepts_a_legitimate_table_name(tmp_path):
    """The guard passes the default through and the pipeline gets going."""
    from spacr.timelapse import automated_motility_assay

    with pytest.raises(FileNotFoundError, match="merged"):
        automated_motility_assay({"src": str(tmp_path),
                                  "db_table_name": "my_motility_table"})


def test_reserved_db_table_names_are_the_spacr_owned_tables():
    from spacr.timelapse import RESERVED_DB_TABLE_NAMES

    assert set(RESERVED_DB_TABLE_NAMES) == {
        "cell", "cytoplasm", "nucleus", "object_counts", "organelle",
        "pathogen", "png_list", "run_status", "settings"}
    assert list(RESERVED_DB_TABLE_NAMES) == sorted(RESERVED_DB_TABLE_NAMES)
