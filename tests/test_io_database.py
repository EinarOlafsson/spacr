"""Table-driven tests of the SQLite readers and writers in ``spacr.io``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: ``_read_db``, ``_create_database``, the settings and
object-count writers, the training-progress CSVs and curves, the table
joins (``_read_and_join_tables`` / ``_read_and_merge_data``), the
plate-name migration and the crop rows read back from an object table.
Every database is a tiny synthetic file under ``tmp_path``.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io as IO  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _write_db(path, tables):
    """Write ``{name: frame}`` into a fresh SQLite file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        for name, df in tables.items():
            df.to_sql(name, con, index=False, if_exists="replace")
    return str(path)


def _query(db, sql):
    """Run ``sql`` against ``db`` and return the frame."""
    con = sqlite3.connect(str(db))
    try:
        return pd.read_sql(sql, con)
    finally:
        con.close()


def _meta(key):
    """Well metadata: odd keys land in r2/c2, even keys in r1/c1."""
    r, c = f"r{(key % 2) + 1}", f"c{(key % 2) + 1}"
    return {"plateID": "plate1", "rowID": r, "columnID": c, "fieldID": "f1",
            "prcf": f"plate1_{r}_{c}_f1"}


def _objects(role, pairs):
    """A measurement table: ``pairs`` is a list of (object_label, parent cell)."""
    rows = []
    for label, cell in pairs:
        row = dict(_meta(cell), object_label=label)
        if role not in ("cell", "cytoplasm"):
            row["cell_id"] = cell
        row[f"{role}_area"] = 10.0 + label
        row[f"{role}_channel_0_mean_intensity"] = 100.0 + label
        rows.append(row)
    return pd.DataFrame(rows)


def _own(role, n=6):
    """``n`` objects keyed on their own label (cell / cytoplasm)."""
    return _objects(role, [(i, i) for i in range(1, n + 1)])


def _png_rows(n=6, field="f1"):
    """png_list rows for objects 1..n, cropped from ``field``."""
    rows = []
    for obj in range(1, n + 1):
        row = dict(_meta(obj))
        row["fieldID"] = field
        row["prcf"] = row["prcf"].replace("_f1", f"_{field}")
        row.update({"cell_id": f"o{obj}", "png_path": f"/x/cell_png/o{obj}.png",
                    "file_name": f"o{obj}.png", "test": obj % 2})
        rows.append(row)
    return pd.DataFrame(rows)


def _stats_frame(n=4):
    """A per-epoch stats frame with every column read_plot_model_stats plots."""
    return pd.DataFrame({
        "epoch": list(range(1, n + 1)),
        "accuracy": np.linspace(0.50, 0.90, n),
        "neg_accuracy": np.linspace(0.40, 0.80, n),
        "pos_accuracy": np.linspace(0.60, 0.95, n),
        "loss": np.linspace(1.00, 0.20, n),
        "prauc": np.linspace(0.30, 0.80, n),
        "optimal_threshold": np.linspace(0.40, 0.60, n),
    })


def _suffix_counts(series):
    """``{'o<N>': count}`` from a per-object count column."""
    return {k.rsplit("_", 1)[-1]: int(v) for k, v in series.to_dict().items()}


# ---------------------------------------------------------------------------
# _read_db
# ---------------------------------------------------------------------------

def test_read_db_returns_one_frame_per_table_whatever_the_path_or_chunking(
        subtests, tmp_path, monkeypatch):
    """Each named table comes back whole, re-indexed, for text and bytes paths."""
    db = _write_db(tmp_path / "m.db", {"cell": pd.DataFrame({
        "object_label": np.arange(1, 7), "plateID": ["plate1"] * 6,
        "cell_area": np.linspace(100.0, 200.0, 6)})})
    real = pd.read_sql_query

    def chunked(parts):
        def fake(sql, con, *args, **kwargs):
            if "chunksize" in kwargs:
                kwargs.pop("chunksize")
                full = real(sql, con, *args, **kwargs)
                return iter([full.iloc[a:b] for a, b in parts])
            return real(sql, con, *args, **kwargs)
        return fake

    cases = [
        ("text path", db, None, list(range(1, 7))),
        ("bytes path", os.fsencode(db), None, list(range(1, 7))),
        ("two chunks", db, [(0, 2), (2, 6)], list(range(1, 7))),
        ("no chunks at all", db, [], []),
    ]
    for case_id, path, parts, labels in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as m:
                if parts is not None:
                    m.setattr(pd, "read_sql_query", chunked(parts))
                frames = IO._read_db(path, ["cell"])
            assert len(frames) == 1
            [df] = frames
            assert df["object_label"].tolist() == labels
            assert list(df.columns) == ["object_label", "plateID", "cell_area"]
            assert list(df.index) == list(range(len(labels)))
            if labels:
                assert df["cell_area"].tolist() == pytest.approx(
                    np.linspace(100.0, 200.0, 6).tolist())

    with subtests.test(msg="an empty table name is refused, not interpolated"):
        bad = tmp_path / "bad.db"
        con = sqlite3.connect(bad)
        con.execute('CREATE TABLE "" (png_path TEXT)')
        con.execute("INSERT INTO \"\" VALUES ('a.png')")
        con.commit()
        con.close()
        with pytest.raises(ValueError, match="Invalid table name"):
            IO._read_db(str(bad), [""])


# ---------------------------------------------------------------------------
# _create_database
# ---------------------------------------------------------------------------

def test_create_database_makes_a_usable_file_or_reports_why_not(
        subtests, tmp_path, capsys):
    """A writable path becomes a working SQLite file; a bad one is printed."""
    with subtests.test(msg="writable path"):
        db = tmp_path / "new.db"
        IO._create_database(str(db))
        assert db.is_file()
        con = sqlite3.connect(str(db))
        try:
            con.execute("CREATE TABLE t (a INTEGER)")
            con.execute("INSERT INTO t VALUES (1)")
            assert con.execute("SELECT a FROM t").fetchall() == [(1,)]
        finally:
            con.close()
    with subtests.test(msg="missing parent folder"):
        capsys.readouterr()
        bad = tmp_path / "missing_dir" / "nested.db"
        IO._create_database(str(bad))
        assert not bad.exists()
        assert "unable to open database file" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _save_settings_to_db
# ---------------------------------------------------------------------------

def test_save_settings_to_db_replaces_latest_and_appends_history(
        subtests, tmp_path):
    """``settings`` holds the last run as text; the history keeps every stage."""
    src = tmp_path / "plate1" / "images"
    src.mkdir(parents=True)
    db = tmp_path / "plate1" / "measurements" / "measurements.db"
    runs = [
        ("first stage", {"src": str(src), "diameter": 30, "plot": False,
                         "channels": [0, 1]}, "measure_crop",
         {"src": str(src), "diameter": "30", "plot": "False",
          "channels": "[0, 1]"}, ["measure_crop"]),
        ("second stage replaces", {"src": str(src), "diameter": 99},
         "preprocess", {"src": str(src), "diameter": "99"},
         ["measure_crop", "preprocess"]),
        ("no stage", {"src": str(src), "diameter": 45}, None,
         {"src": str(src), "diameter": "45"}, None),
    ]
    for case_id, settings, stage, latest, stages in runs:
        with subtests.test(msg=case_id):
            if stage is None:
                IO._save_settings_to_db(settings)
            else:
                IO._save_settings_to_db(settings, stage=stage)
            df = _query(db, f'SELECT * FROM "{IO.SETTINGS_TABLE}"')
            assert dict(zip(df["setting_key"], df["setting_value"])) == latest
            hist = _query(db, f'SELECT * FROM "{IO.SETTINGS_HISTORY_TABLE}"')
            if stages is not None:
                assert sorted(set(hist["stage"])) == stages
                assert hist["run_id"].nunique() == len(stages)


# ---------------------------------------------------------------------------
# _save_object_counts_to_database
# ---------------------------------------------------------------------------

def test_save_object_counts_counts_labels_and_upserts(subtests, tmp_path):
    """Background 0 is not an object, and a re-save updates the same row."""
    db = tmp_path / "counts.db"
    two = np.zeros((8, 8), np.int32)
    two[1:3, 1:3] = 1
    two[5:7, 5:7] = 2
    cases = [
        ("fully tiled mask counts every label",
         np.array([[1, 1, 2], [2, 3, 3], [3, 1, 2]], np.uint16), "cell",
         "full.npy", "_before_filtration", 3, "cell_before_filtration"),
        ("background is not counted",
         np.array([[0, 0, 1], [1, 2, 0], [0, 0, 2]], np.uint16), "cell",
         "sparse.npy", "_before_filtration", 2, "cell_before_filtration"),
        ("two blobs", two, "cell", "a.npy", "_test", 2, "cell_test"),
        ("all background", np.zeros((4, 4), np.uint16), "pathogen",
         "empty.npy", "_after", 0, "pathogen_after"),
        ("first save", np.array([[0, 1], [1, 0]], np.uint16), "cell", "f.npy",
         "", 1, "cell"),
        ("re-save upserts", np.array([[0, 1], [2, 3]], np.uint16), "cell",
         "f.npy", "", 3, "cell"),
    ]
    for case_id, mask, role, name, added, count, count_type in cases:
        with subtests.test(msg=case_id):
            IO._save_object_counts_to_database([mask], role, [name], str(db),
                                               added)
            df = _query(db, "SELECT * FROM object_counts")
            row = df[(df["file_name"] == name)
                     & (df["count_type"] == count_type)]
            assert len(row) == 1
            assert int(row["object_count"].iloc[0]) == count
    with subtests.test(msg="one row per (file, count type)"):
        df = _query(db, "SELECT * FROM object_counts")
        assert sorted(zip(df["file_name"], df["count_type"])) == [
            ("a.npy", "cell_test"), ("empty.npy", "pathogen_after"),
            ("f.npy", "cell"), ("full.npy", "cell_before_filtration"),
            ("sparse.npy", "cell_before_filtration")]


# ---------------------------------------------------------------------------
# _save_progress / read_plot_model_stats
# ---------------------------------------------------------------------------

def test_save_progress_appends_csvs_and_plots_only_with_validation(
        subtests, tmp_path):
    """train.csv appends; validation.csv and the six curves need validation."""
    metrics = ["accuracy", "loss", "neg_accuracy", "optimal_threshold",
               "pos_accuracy", "prauc"]
    cases = [
        ("no validation", "a", [None, None], [1, 2, 3, 1, 2, 3], None, []),
        ("with validation", "b", [_stats_frame(3), _stats_frame(3)],
         [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], metrics),
    ]
    for case_id, sub, vals, train_epochs, val_epochs, pdfs in cases:
        with subtests.test(msg=case_id):
            dst = tmp_path / sub / "nested"
            for val in vals:
                IO._save_progress(str(dst), _stats_frame(3), val)
            back = pd.read_csv(dst / "train.csv", index_col=0)
            assert back["epoch"].tolist() == train_epochs
            assert list(back.columns) == list(_stats_frame(3).columns)
            if val_epochs is None:
                assert not (dst / "validation.csv").exists()
            else:
                vback = pd.read_csv(dst / "validation.csv", index_col=0)
                assert vback["epoch"].tolist() == val_epochs
            assert sorted(p.stem for p in dst.glob("*.pdf")) == pdfs


def test_read_plot_model_stats_draws_six_curves_without_restyling(
        subtests, tmp_path, monkeypatch):
    """One figure per metric, saved or shown, and rcParams left untouched."""
    train, val = tmp_path / "train.csv", tmp_path / "validation.csv"
    _stats_frame().to_csv(train)
    _stats_frame().to_csv(val)
    with subtests.test(msg="save=False shows six figures, writes nothing"):
        shown = []
        monkeypatch.setattr(plt, "show",
                            lambda *a, **k: shown.append(len(plt.get_fignums())))
        IO.read_plot_model_stats(str(train), str(val), save=False)
        assert len(shown) == 6
        assert all(n >= 1 for n in shown)
        assert list(tmp_path.glob("*.pdf")) == []
    with subtests.test(msg="save=True writes six PDFs and keeps rcParams"):
        plt.close("all")
        before = {k: repr(v) for k, v in matplotlib.rcParams.items()}
        IO.read_plot_model_stats(str(train), str(val), save=True)
        after = {k: repr(v) for k, v in matplotlib.rcParams.items()}
        assert sorted(k for k in after if before.get(k) != after[k]) == []
        assert sorted(p.name for p in tmp_path.glob("*.pdf")) == [
            "accuracy.pdf", "loss.pdf", "neg_accuracy.pdf",
            "optimal_threshold.pdf", "pos_accuracy.pdf", "prauc.pdf"]
        assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# _read_and_join_tables
# ---------------------------------------------------------------------------

def test_read_and_join_tables_joins_roles_onto_the_cell_table(
        subtests, tmp_path, capsys):
    """Cell + png_list + cytoplasm + child roles join on (object_label, prcf)."""
    prcf = "plate1_r01_c01_1"
    png = pd.DataFrame({
        "cell_id": ["o1", "o2", "o3"],
        "png_path": [f"/data/plate1_A01_1_{i}.png" for i in (1, 2, 3)],
        "plateID": ["plate1"] * 3, "rowID": ["r01"] * 3,
        "columnID": ["c01"] * 3, "fieldID": ["1"] * 3})
    cell = pd.DataFrame({"object_label": [1, 2], "prcf": [prcf] * 2,
                         "plateID": ["plate1"] * 2, "rowID": ["r01"] * 2,
                         "columnID": ["c01"] * 2, "fieldID": ["1"] * 2,
                         "cell_area": [100.0, 200.0]})
    nucleus = pd.DataFrame({"cell_id": [1, 1, 2], "prcf": [prcf] * 3,
                            "nucleus_area": [10.0, 20.0, 50.0]})
    cyto_cell = pd.DataFrame({"object_label": [1, 2], "prcf": [prcf] * 2,
                              "plateID": ["plate1"] * 2,
                              "area": [100.0, 200.0]})
    cyto = pd.DataFrame({"object_label": [1, 2], "prcf": [prcf] * 2,
                         "plateID": ["plate1"] * 2, "area": [70.0, 80.0]})
    lone = pd.DataFrame({"object_label": [1], "prcf": ["p_r_c_1"],
                         "cell_area": [3.0]})
    cases = [
        # id, tables, kwargs, expected {column: values}, printed fragment
        ("png_list only", {"png_list": png}, {},
         {"object_label": [1, 2, 3],
          "png_path": png["png_path"].tolist()},
         "Cell table not found in database tables."),
        ("cell + png_list + nucleus", {"png_list": png.iloc[:2],
                                       "cell": cell, "nucleus": nucleus}, {},
         {"object_label": [1, 2], "cell_area": [100.0, 200.0],
          "png_path": png["png_path"].tolist()[:2],
          "count_nucleus": [2, 1], "nucleus_area": [15.0, 50.0]}, None),
        ("cytoplasm suffix, left join",
         {"cell": cyto_cell, "cytoplasm": cyto.iloc[:1]},
         {"table_names": ["cell", "cytoplasm"]},
         {"area": [100.0, 200.0], "area_cytoplasm": [70.0, np.nan]}, None),
        ("collapse off keeps both copies",
         {"cell": cyto_cell, "cytoplasm": cyto,
          "nucleus": nucleus.iloc[[0, 2]]},
         {"table_names": ["cell", "cytoplasm", "nucleus"],
          "collapse_duplicate_identity": False},
         {"area": [100.0, 200.0], "area_cytoplasm": [70.0, 80.0],
          "plateID_cytoplasm": ["plate1", "plate1"],
          "count_nucleus": [1, 1]}, None),
        ("a missing table is announced and skipped", {"cell": lone},
         {"table_names": ["cell", "cytoplasm"]},
         {"object_label": [1], "cell_area": [3.0]},
         "Table cytoplasm not found in the database."),
    ]
    for i, (case_id, tables, kwargs, expected, printed) in enumerate(cases):
        with subtests.test(msg=case_id):
            db = _write_db(tmp_path / f"j{i}.db", tables)
            capsys.readouterr()
            out = IO._read_and_join_tables(db, **kwargs)
            text = capsys.readouterr().out
            for column, values in expected.items():
                got = out[column].tolist()
                if any(isinstance(v, float) for v in values):
                    np.testing.assert_allclose(got, values)
                else:
                    assert got == values, column
            if printed:
                assert printed in text
    with subtests.test(msg="png_list-only frame shape"):
        db = _write_db(tmp_path / "png.db", {"png_list": png})
        out = IO._read_and_join_tables(db)
        assert list(out.columns) == ["object_label", "png_path", "plateID",
                                     "rowID", "columnID", "fieldID"]
        assert np.issubdtype(out["object_label"].dtype, np.integer)
    with subtests.test(msg="only an absent table leaves nothing to join"):
        db = _write_db(tmp_path / "absent.db", {"cell": lone})
        capsys.readouterr()
        with pytest.raises(AttributeError):
            IO._read_and_join_tables(db, table_names=["does_not_exist"])
        assert "Table does_not_exist not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _read_and_merge_data
# ---------------------------------------------------------------------------

def test_read_and_merge_data_anchors_groups_and_limits_every_role(
        subtests, tmp_path, capsys):
    """Each role combination yields one row per parent cell, keyed on prcf."""
    two_nuclei = _objects("nucleus", [(1, 1), (7, 1), (2, 2), (3, 3), (4, 4),
                                      (5, 5), (6, 6)])
    two_paths = _objects("pathogen", [(1, 1), (7, 1), (2, 2), (3, 3), (4, 4),
                                      (5, 5), (6, 6)])
    six = ["o1", "o2", "o3", "o4", "o5", "o6"]
    five = six[1:]
    org = _objects("organelle", [(1, 1), (2, 1), (3, 2), (4, 3)])
    cases = [
        # id, tables, roles, kwargs, row suffixes, {count column: {o: n}},
        # printed fragments, fragments that must NOT be printed
        ("cell only", {"cell": _own("cell")}, ["cell"], {}, six, {}, [], []),
        ("change_plate keeps plate1", {"cell": _own("cell")}, ["cell"],
         {"change_plate": True}, six, {}, [], []),
        ("cytoplasm anchors loudly", {"cytoplasm": _own("cytoplasm")},
         ["cytoplasm"], {"verbose": True}, six, {},
         ["cytoplasms grouped"], []),
        ("cytoplasm anchors quietly", {"cytoplasm": _own("cytoplasm", 3)},
         ["cytoplasm"], {}, six[:3], {}, [], ["cytoplasms grouped"]),
        ("nucleus labels differ from cell labels",
         {"nucleus": _objects("nucleus", [(10 + i, i) for i in range(1, 7)])},
         ["nucleus"], {"nuclei_limit": None}, six,
         {"nucleus_prcfo_count": dict.fromkeys(six, 1)}, [], []),
        ("nuclei_limit=True drops the two-nucleus cell",
         {"nucleus": two_nuclei}, ["nucleus"],
         {"verbose": True, "nuclei_limit": True}, five,
         {"nucleus_prcfo_count": dict.fromkeys(five, 1)},
         ["nucleus grouped"], []),
        ("pathogen_limit=True drops the doubly infected cell",
         {"pathogen": two_paths}, ["pathogen"],
         {"verbose": True, "pathogen_limit": True}, five,
         {"pathogen_prcfo_count": dict.fromkeys(five, 1)},
         ["pathogens grouped"], []),
        ("a text pathogen_limit filters nothing, quietly",
         {"pathogen": _objects("pathogen", [(1, 1), (2, 1), (3, 2)])},
         ["pathogen"], {"pathogen_limit": "lots"}, ["o1", "o2"],
         {"pathogen_prcfo_count": {"o1": 2, "o2": 1}}, [],
         ["pathogens grouped"]),
        ("a text nuclei_limit filters nothing",
         {"cell": _own("cell", 2),
          "nucleus": _objects("nucleus", [(1, 1), (2, 1), (3, 2)])},
         ["cell", "nucleus"], {"nuclei_limit": "2"}, ["o1", "o2"],
         {"nucleus_prcfo_count": {"o1": 2, "o2": 1}}, [], []),
        ("nuclei_limit=True on a cell table",
         {"cell": _own("cell", 2),
          "nucleus": _objects("nucleus", [(1, 1), (2, 1), (3, 2)])},
         ["cell", "nucleus"], {"nuclei_limit": True}, ["o2"],
         {"nucleus_prcfo_count": {"o2": 1}}, [], []),
        ("organelle anchors on its parent cell loudly", {"organelle": org},
         ["organelle"], {"verbose": True}, ["o1", "o2", "o3"],
         {"organelle_prcfo_count": {"o1": 2, "o2": 1, "o3": 1}},
         ["organelle grouped"], []),
        ("organelle anchors quietly", {"organelle": org}, ["organelle"], {},
         ["o1", "o2", "o3"], {}, [], ["organelle grouped"]),
        ("organelle merges onto a cell table",
         {"cell": _own("cell", 3), "organelle": org}, ["cell", "organelle"],
         {"verbose": True}, ["o1", "o2", "o3"],
         {"organelle_prcfo_count": {"o1": 2, "o2": 1, "o3": 1}},
         ["organelle: 4, organelle grouped: 3"], []),
        ("all core roles with numeric limits",
         {"cell": _own("cell"), "cytoplasm": _own("cytoplasm"),
          "nucleus": _objects("nucleus", [(i, i) for i in range(1, 7)]),
          "pathogen": _objects("pathogen", [(i, i) for i in range(1, 7)])},
         ["cell", "cytoplasm", "nucleus", "pathogen"],
         {"nuclei_limit": 2, "pathogen_limit": 2, "verbose": True}, six, {},
         ["cytoplasms: 6, cytoplasms grouped: 6",
          "nucleus: 6, nucleus grouped: 6",
          "pathogens: 6, pathogens grouped: 6"], []),
        ("matching png_list adds its crop columns",
         {"cell": _own("cell"), "png_list": _png_rows()},
         ["cell", "png_list"], {"verbose": True}, six, {},
         ["png_list grouped"], []),
        ("crops from another field empty the merge loudly",
         {"cell": _own("cell"), "png_list": _png_rows(field="f9")},
         ["cell", "png_list"], {}, [], {},
         ["6 of 6 objects have no row in png_list",
          "gap in the database rather than a filter"], []),
    ]
    for i, (case_id, tables, roles, kwargs, suffixes, counts, said,
            unsaid) in enumerate(cases):
        with subtests.test(msg=case_id):
            db = _write_db(tmp_path / f"m{i}.db", tables)
            capsys.readouterr()
            merged, frames = IO._read_and_merge_data([db], roles, **kwargs)
            out = capsys.readouterr().out
            assert sorted(ix.rsplit("_", 1)[-1] for ix in merged.index) == \
                suffixes
            assert [len(f) for f in frames] == [
                len(tables[r]) for r in roles if r != "png_list"]
            for column, expected in counts.items():
                assert _suffix_counts(merged[column]) == expected
            if suffixes:
                assert set(merged["plateID"]) == {"plate1"}
            for fragment in said:
                assert fragment in out
            for fragment in unsaid:
                assert fragment not in out

    with subtests.test(msg="png_list columns arrive pruned"):
        db = _write_db(tmp_path / "png.db",
                       {"cell": _own("cell"), "png_list": _png_rows()})
        merged, _ = IO._read_and_merge_data([db], ["cell", "png_list"])
        assert sorted(merged["png_path"]) == [
            f"/x/cell_png/o{i}.png" for i in range(1, 7)]
        assert sorted(merged["test"].astype(int)) == [0, 0, 0, 1, 1, 1]
        assert "file_name" not in merged.columns
        assert {"cell_area", "cell_channel_0_mean_intensity"} <= set(
            merged.columns)

    stamp = {"measurement_ndim": 2, "measurement_units": "px", "n_z": 1,
             "voxel_size_z_um": 1.0, "voxel_size_xy_um": 0.5}
    stamped_cases = [
        # id, png_list cropped from field, expected rows
        ("stamped tables keep one copy of the stamp", "f1", 6),
        ("stamped crops from another field empty the merge", "f9", 0),
    ]
    for i, (case_id, field, n_rows) in enumerate(stamped_cases):
        with subtests.test(msg=case_id):
            db = _write_db(tmp_path / f"stamp{i}.db",
                           {"cell": _own("cell").assign(**stamp),
                            "png_list": _png_rows(field=field).assign(**stamp)})
            merged, _ = IO._read_and_merge_data([db], ["cell", "png_list"])
            assert len(merged) == n_rows
            for col, value in (stamp.items() if n_rows else ()):
                assert merged[col].tolist() == [value] * n_rows, col
                assert f"{col}_x" not in merged.columns

    with subtests.test(msg="pathogen counts are true integers"):
        db = _write_db(tmp_path / "pint.db", {"pathogen": two_paths})
        merged, _ = IO._read_and_merge_data([db], ["pathogen"],
                                            pathogen_limit=True)
        assert str(merged["pathogen_prcfo_count"].dtype) == "Int64"

    with subtests.test(msg="the key is prcf + object, timepoint included"):
        cells = _objects("cell", [(1, 1), (2, 2)])
        db = _write_db(tmp_path / "k.db", {"cell": cells})
        merged, _ = IO._read_and_merge_data([db], ["cell"])
        assert sorted(merged.index) == ["plate1_r1_c1_f1_o2",
                                        "plate1_r2_c2_f1_o1"]
        frames = _objects("cell", [(1, 1), (1, 1)])
        frames["timeID"] = [1, 2]
        frames["prcf"] = frames["prcf"] + "_" + frames["timeID"].astype(str)
        tdb = _write_db(tmp_path / "t.db", {"cell": frames})
        merged, _ = IO._read_and_merge_data([tdb], ["cell"])
        assert sorted(merged.index) == ["plate1_r2_c2_f1_1_o1",
                                        "plate1_r2_c2_f1_2_o1"]

    with subtests.test(msg="cells_per_well counts objects across fields"):
        rows = [{"plateID": "p1", "rowID": "r1", "columnID": "c1",
                 "fieldID": f, "prcf": f"p1_r1_c1_{f}", "object_label": lab,
                 "area": 10.0 * lab}
                for f in ("f1", "f2") for lab in (1, 2, 3)]
        db = _write_db(tmp_path / "w.db", {"cell": pd.DataFrame(rows)})
        merged, _ = IO._read_and_merge_data([db], ["cell"])
        assert len(merged) == 6
        assert sorted(set(merged["cells_per_well"])) == [6]

    refusals = [
        ("numeric well metadata loses plateID",
         {"cell": _objects("cell", [(1, 1), (2, 2)]).assign(
             plateID=1, rowID=2, columnID=3, fieldID=4)}, KeyError, "plateID"),
        ("an empty table loses plateID",
         {"cell": _objects("cell", [(1, 1)]).iloc[:0]}, KeyError, "plateID"),
    ]
    for i, (case_id, tables, exc, match) in enumerate(refusals):
        with subtests.test(msg=case_id):
            db = _write_db(tmp_path / f"r{i}.db", tables)
            with pytest.raises(exc, match=match):
                IO._read_and_merge_data([db], ["cell"])

    with subtests.test(msg="no databases at all is not a silent empty result"):
        capsys.readouterr()
        with pytest.raises(AttributeError):
            IO._read_and_merge_data([], ["cell"], verbose=True)
        assert "cell: 0" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# migrate_unescaped_plate_names
# ---------------------------------------------------------------------------

def test_migrate_unescaped_plate_names_escapes_only_split_plate_names(
        subtests, tmp_path):
    """Plate names with ``_`` are escaped once; everything else is left alone."""
    cases = [
        # id, {folder: [file names]}, dry_run, planned {old: new} basenames,
        # file names left in each folder afterwards
        ("underscore plate renamed", {"merged": ["exp_1_A01_1_1.npy"]}, False,
         {"exp_1_A01_1_1.npy": "exp%5F1_A01_1_1.npy"},
         {"merged": ["exp%5F1_A01_1_1.npy"]}),
        ("dry run moves nothing", {"masks": ["exp_1_A01_1_1.npy"]}, True,
         {"exp_1_A01_1_1.npy": "exp%5F1_A01_1_1.npy"},
         {"masks": ["exp_1_A01_1_1.npy"]}),
        ("ordinary plates need nothing",
         {"masks": ["exp1_A01_1_1.npy", "exp1_A02_1_1.npy"]}, False, {},
         {"masks": ["exp1_A01_1_1.npy", "exp1_A02_1_1.npy"]}),
        ("a dotted non-stem is skipped beside a real one",
         {"stack": ["a_b.c_d_e_f.npy", "exp_1_A01_1_1.npy"]}, True,
         {"exp_1_A01_1_1.npy": "exp%5F1_A01_1_1.npy"},
         {"stack": ["a_b.c_d_e_f.npy", "exp_1_A01_1_1.npy"]}),
        ("other file kinds are not touched",
         {"merged": ["exp_1_A01_1_1.json", "notes_about_exp_1_and_more.txt"]},
         False, {},
         {"merged": ["exp_1_A01_1_1.json", "notes_about_exp_1_and_more.txt"]}),
    ]
    for i, (case_id, layout, dry, planned, after) in enumerate(cases):
        with subtests.test(msg=case_id):
            root = tmp_path / f"c{i}"
            for folder, names in layout.items():
                (root / folder).mkdir(parents=True)
                for name in names:
                    if name.endswith(".npy"):
                        np.save(root / folder / name, np.zeros((2, 2), np.uint8))
                    else:
                        (root / folder / name).write_text("{}", encoding="utf-8")
            got = IO.migrate_unescaped_plate_names(str(root), dry_run=dry)
            assert {os.path.basename(o): os.path.basename(n)
                    for o, n in got} == planned
            assert all(os.path.dirname(o) == os.path.dirname(n)
                       for o, n in got)
            for folder, names in after.items():
                assert sorted(p.name for p in (root / folder).iterdir()) == \
                    sorted(names)
            if not dry:
                assert IO.migrate_unescaped_plate_names(str(root)) == [], \
                    "a second run must be a no-op"

    with subtests.test(msg="an occupied destination refuses before moving"):
        merged = tmp_path / "occ" / "merged"
        merged.mkdir(parents=True)
        np.save(merged / "exp_1_A01_1_1.npy", np.zeros((2, 2), np.uint8))
        np.save(merged / "exp%5F1_A01_1_1.npy", np.zeros((2, 2), np.uint8))
        with pytest.raises(FileExistsError, match="refusing to migrate"):
            IO.migrate_unescaped_plate_names(str(tmp_path / "occ"))
        assert (merged / "exp_1_A01_1_1.npy").exists()


# ---------------------------------------------------------------------------
# crop_rows_from_object_table
# ---------------------------------------------------------------------------

def test_crop_rows_from_object_table_names_crops_after_their_parent(
        subtests, tmp_path, capsys):
    """Rows become crop names ``<field>_<parent>_<label>.png`` or an empty frame."""
    cols = ["object_label", "plateID", "rowID", "columnID", "fieldID", "prcf",
            "file_name", "path_name"]
    base = (4, "plate1", "A", "01", "1", "plate1_A_01_1", "plate1_A01_1.npy",
            "/data/plate1_A01_1.npy")
    cases = [
        # id, table, columns, rows, asked role, expected png names, printed
        ("nucleus without parent links", "nucleus", cols, [base], "nucleus",
         ["plate1_A01_1_none_4.png"], "1 'nucleus' objects"),
        ("nucleus with parent links", "nucleus", cols + ["cell_id"],
         [base + (9,)], "nucleus", ["plate1_A01_1_9_4.png"], None),
        ("no table for the asked role", "cell", cols, [base], "nucleus", [],
         "no 'nucleus' table"),
        ("an empty table", "cell", cols, [], "cell", [], None),
    ]
    for i, (case_id, table, columns, rows, role, names, said) in enumerate(cases):
        with subtests.test(msg=case_id):
            db = tmp_path / f"o{i}.db"
            con = sqlite3.connect(db)
            con.execute('CREATE TABLE "%s" (%s)'
                        % (table, ", ".join(f'"{c}"' for c in columns)))
            con.executemany('INSERT INTO "%s" VALUES (%s)'
                            % (table, ", ".join("?" * len(columns))), rows)
            con.commit()
            con.close()
            capsys.readouterr()
            frame = IO.crop_rows_from_object_table(str(db), object_type=role,
                                                   verbose=said is not None)
            out = capsys.readouterr().out
            if names:
                assert list(frame["png_name"]) == names
                assert frame["png_path"].iloc[0].endswith(
                    os.path.join(f"{role}_png", names[0]))
            else:
                assert frame.empty
                assert "png_name" not in frame.columns
            if said:
                assert said in out
            else:
                assert out == ""
