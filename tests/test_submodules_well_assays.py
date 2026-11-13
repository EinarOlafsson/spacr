"""Table-driven tests of the per-well phenotype assays in ``spacr.submodules``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture:

* ``analyze_recruitment``: the exact pathogen / host ratios per cell, the
  per-well table, every object filter, the database location rules and the
  overlay / control-panel plotting;
* ``analyze_plaques``: the summary, stats and details tables written to
  ``plaques_analysis.db`` from existing or freshly segmented masks;
* ``analyze_percent_positive``: per-well above / below counts against the
  threshold and the well-name translation through ``rename_log.csv``;
* ``analyze_replication``: parasites per vacuole and per well.

``spacr.io._read_and_merge_data`` and the plotting / model collaborators are
recording fakes where the real ones would need a GPU, a download or a
display; the databases and CSVs are real.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io as SIO  # noqa: E402
import spacr.plot as SPLT  # noqa: E402
import spacr.submodules as SUB  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _case_dir(tmp_path, name):
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


class _Recorder:
    """Callable recording every call; optionally raising."""

    def __init__(self, exc=None, result=None):
        self.calls, self.exc, self.result = [], exc, result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.result


# ---------------------------------------------------------------------------
# analyze_recruitment
# ---------------------------------------------------------------------------

def _merged(wells=(("r1", "c1"), ("r4", "c4")), n_cells=4, extra_wells=()):
    """A merged four-compartment table; cell i of a well has area 1000 + i and
    channel-c intensities 100+c (cell), 200+c (nucleus), 50+c (cytoplasm),
    400+c (pathogen)."""
    rows = []
    for rid, cid in list(wells) + list(extra_wells):
        for i in range(n_cells):
            prc = f"plate1_{rid}_{cid}"
            rec = {"prcfo": f"{prc}_1_o{i + 1}", "plateID": "plate1", "rowID": rid,
                   "columnID": cid, "fieldID": "1", "object_label": f"o{i + 1}", "prc": prc,
                   "prcf": f"{prc}_1", "file_name": f"{prc}_1", "cells_per_well": float(n_cells),
                   "cell_area": 1000.0 + i, "nucleus_area": 300.0 + i,
                   "pathogen_area": 120.0 + i, "cytoplasm_area": 700.0 + i}
            for c in range(4):
                rec.update({f"cell_channel_{c}_mean_intensity": 100.0 + c,
                            f"nucleus_channel_{c}_mean_intensity": 200.0 + c,
                            f"cytoplasm_channel_{c}_mean_intensity": 50.0 + c,
                            f"pathogen_channel_{c}_mean_intensity": 400.0 + c,
                            f"pathogen_channel_{c}_percentile_75": 500.0 + c,
                            f"pathogen_channel_{c}_outside_mean": 10.0 + c,
                            f"pathogen_channel_{c}_outside_75_percentile": 20.0 + c,
                            f"pathogen_channel_{c}_periphery_mean": 30.0 + c,
                            f"cell_channel_{c}_percentile_95": 1000.0 + 10 * i})
            rows.append(rec)
    return pd.DataFrame(rows).set_index("prcfo")


def _zero_first_intensity():
    df = _merged()
    df.iloc[0, df.columns.get_loc("cell_channel_0_mean_intensity")] = 0.0
    return df


def _loose_db(folder):
    db = folder / "measurements.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE t (a INTEGER)")
    return db


def test_analyze_recruitment_reports_exact_ratios_per_cell_and_well(subtests, tmp_path, capsys):
    """Each surviving cell carries the literal pathogen / host intensity
    ratios, each filter keeps exactly the cells inside its bounds, and the
    database path, plots and CSVs follow the settings."""
    unbounded = [0, 10 ** 100]
    huge = [10 ** 6, 10 ** 7]
    cases = [
        # (id, merged frame, src builder, settings overrides, expected
        #  (n cells, n wells, extra checks))
        ("defaults keep every cell", _merged(), "dir", {}, (8, 2, {"full": True})),
        ("a zero signal survives the default intensity range", _zero_first_intensity(), "dir",
         {}, (8, 2, {"zero_kept": 1})),
        ("the cell size filter is an open interval", _merged(), "dir",
         {"cell_size_range": [1001, 1003]}, (2, 2, {"areas": {1002.0}})),
        ("None size ranges become unbounded", _merged(), "dir",
         {"cell_size_range": None, "nucleus_size_range": None, "pathogen_size_range": None},
         (8, 2, {"ranges": unbounded})),
        ("no channel dims skip every object filter", _merged(), "dir",
         {"cell_chann_dim": None, "nucleus_chann_dim": None, "pathogen_chann_dim": None,
          "cell_size_range": huge, "nucleus_size_range": huge, "pathogen_size_range": huge},
         (8, 2, {})),
        ("wells below cells_per_well are dropped",
         pd.concat([_merged(wells=(("r1", "c1"),), n_cells=2),
                    _merged(wells=(("r4", "c4"),), n_cells=5)]), "dir",
         {"cells_per_well": 3}, (5, 1, {"prcs": {"plate1_r4_c4"}})),
        ("unannotated wells are dropped", _merged(extra_wells=(("r9", "c9"),)), "dir",
         {"cell_types": ["HeLa"], "cell_plate_metadata": [["c1", "c4"]]}, (8, 2, {})),
        ("target_intensity_min filters on the 95th percentile", _merged(), "dir",
         {"target_intensity_min": 1015}, (4, 2, {"p95_min": 1020.0,
                                                 "printed": ["After channel 2 filtration"]})),
        ("a loose database is moved into measurements/", _merged(), "loose", {},
         (8, 2, {"moved": True})),
        ("an existing measurements/ is not clobbered", _merged(), "loose+existing", {},
         (8, 2, {"kept": True})),
        ("the canonical database path is read as given", _merged(), "canonical", {},
         (8, 2, {})),
        ("plot without a merged folder plots nothing", _merged(), "dir", {"plot": True},
         (8, 2, {"overlays": 0})),
        ("plot overlays the first plot_nr + 1 merged files", _merged(), "merged5",
         {"plot": True, "plot_nr": 1}, (8, 2, {"overlays": 2})),
        ("a failing overlay is reported, not raised", _merged(), "merged1",
         {"plot": True, "overlay_fails": True},
         (8, 2, {"overlays": 1, "printed": ["Failed to plot images with outlines",
                                            "no display for you"]})),
        ("plot_control writes control panels", _merged(), "dir", {"plot_control": True},
         (8, 2, {"controls": True})),
    ]
    for case_id, frame, src_kind, over, (n_cells, n_wells, check) in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id) / "plate"
            folder.mkdir()
            src, db = folder, folder / "measurements" / "measurements.db"
            if src_kind.startswith("loose"):
                if src_kind.endswith("existing"):
                    (folder / "measurements").mkdir()
                    db.write_bytes(b"already here")
                src = _loose_db(folder)
            elif src_kind == "canonical":
                (folder / "measurements").mkdir()
                db.write_bytes(b"db")
                src = db
            elif src_kind.startswith("merged"):
                (folder / "merged").mkdir()
                for i in range(int(src_kind[-1])):
                    (folder / "merged" / f"field_{i}.npy").write_bytes(b"")
            over = dict(over)
            fails = over.pop("overlay_fails", False)
            settings = {"src": str(src), "plot": False, "plot_control": False,
                        "target_intensity_min": 0, "figuresize": 4, **over}
            reads = []
            overlay = _Recorder(exc=RuntimeError("no display for you") if fails else None)
            controls, recruit = _Recorder(), _Recorder()
            capsys.readouterr()
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(SIO, "_read_and_merge_data", lambda locs, tables, **kw: (
                    reads.append({"locs": list(locs), "tables": list(tables)}) or frame.copy(), []))
                mp.setattr(SPLT, "plot_image_mask_overlay", overlay)
                mp.setattr(SPLT, "_plot_controls", controls)
                mp.setattr(SPLT, "_plot_recruitment", recruit)
                cells, wells = SUB.analyze_recruitment(settings)
            printed = capsys.readouterr().out
            assert (len(cells), len(wells)) == (n_cells, n_wells)
            assert settings["src"] == str(folder)
            assert reads == [{"locs": [str(db)],
                              "tables": ["cell", "nucleus", "pathogen", "cytoplasm"]}]
            np.testing.assert_allclose(cells["recruitment"], 402.0 / 52.0)
            assert [c[0][1] for c in recruit.calls] == ["by PV", "by well"]
            assert controls.calls == []
            assert (folder / "results" / "cells.csv").exists()
            assert len(pd.read_csv(folder / "results" / "wells.csv")) == n_wells
            assert (folder / "settings" / "recruitment.csv").exists()
            assert len(overlay.calls) == check.get("overlays", 0)
            if check.get("full"):
                for column, value in (("pathogen_channel_2_cytoplasm_mean_ratio", 402 / 52),
                                      ("pathogen_channel_3_cytoplasm_mean_ratio", 403 / 53),
                                      ("pathogen_channel_3_cell_q75_ratio", 503 / 103),
                                      ("pathogen_channel_3_nucleus_outside_mean_ratio", 13 / 203),
                                      ("pathogen_channel_3_cytoplasm_periphery_mean_ratio",
                                       33 / 53)):
                    np.testing.assert_allclose(cells[column], value)
                assert set(cells["condition"]) == {"HeLa_pathogen_1_cm",
                                                   "HeLa_pathogen_2_lovastatin"}
                np.testing.assert_allclose(wells["cells_per_well"], 4.0)
                assert settings["cell_intensity_range"] is None
            if "zero_kept" in check:
                assert int((cells["cell_channel_0_mean_intensity"] == 0.0).sum()) == 1
            if "areas" in check:
                assert set(np.round(cells["cell_area"])) == check["areas"]
            if "ranges" in check:
                for kind in ("cell", "nucleus", "pathogen"):
                    assert settings[f"{kind}_size_range"] == check["ranges"]
            if "prcs" in check:
                assert set(cells["prc"]) == check["prcs"]
            if "p95_min" in check:
                assert cells["cell_channel_2_percentile_95"].min() == check["p95_min"]
            for text in check.get("printed", []):
                assert text in printed
            if check.get("moved"):
                assert not (folder / "measurements.db").exists()
                with sqlite3.connect(db) as con:
                    assert [r[0] for r in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'")] == ["t"]
            if check.get("kept"):
                assert (folder / "measurements.db").exists()
                assert db.read_bytes() == b"already here"
            if check.get("overlays") == 2:
                made = {str(folder / "merged" / f"field_{i}.npy") for i in range(5)}
                assert {c[0][0] for c in overlay.calls} <= made
                args, kwargs = overlay.calls[0]
                assert args[1:] == ([0, 1, 2, 3], 3, 0, 2)
                assert kwargs == {"figuresize": 10, "percentiles": (1, 99), "thickness": 3,
                                  "save_pdf": True, "outline_palette": "default"}
            if check.get("controls"):
                panels = folder / "results" / "panels" / "controls"
                folders = sorted(p.name for p in panels.iterdir())
                assert len(folders) == 16 + 1 + 20
                first = panels / "C01_channel_0_cell_mean_intensity"
                assert {p.name for p in first.iterdir()} >= set([
                    "C01_channel_0_cell_mean_intensity.pdf",
                    "C01_channel_0_cell_mean_intensity.png", "data_replicates.csv",
                    "stats_test.csv", "stats_assumptions.csv", "stats_posthoc.csv",
                    "stats_descriptive.csv"])
                assert len(pd.read_csv(first / "data_replicates.csv")) == 2
                figures = folder / "results" / "figures"
                assert (figures / "controls_channel_3_intensity.pdf").exists()
                assert (figures / "controls_objects_per_well.png").exists()


# ---------------------------------------------------------------------------
# analyze_plaques
# ---------------------------------------------------------------------------

def _label_tif(path, blobs=((10, 10, 4), (40, 40, 6))):
    img = np.zeros((64, 64), np.uint16)
    yy, xx = np.mgrid[:64, :64]
    for idx, (cy, cx, r) in enumerate(blobs, start=1):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = idx
    tifffile.imwrite(str(path), img)
    return sorted(np.bincount(img.ravel())[1:].tolist())


def test_analyze_plaques_tabulates_every_plaque_of_every_mask(subtests, tmp_path):
    """Each mask tif gives one summary / stats row and one details row per
    plaque with its exact pixel size, from existing masks or after
    segmentation; a rerun replaces the tables."""
    import spacr.spacr_cellpose as SCP
    import spacr.utils as SU

    three = ((15, 15, 3), (45, 45, 3), (15, 45, 3))
    cases = [
        # (id, settings, existing masks {name: blobs}, segmentation writes, runs)
        ("existing masks, decoys ignored", {"masks": False},
         {"a.tif": ((10, 10, 4), (40, 40, 6)), "b.tif": ((20, 20, 5),)}, None, 1),
        ("a rerun replaces the tables", {"masks": False},
         {"a.tif": ((10, 10, 4), (40, 40, 6))}, None, 2),
        ("masks=True segments first", {"masks": True, "diameter": 30}, {}, {"seg.tif": three}, 1),
    ]
    for case_id, over, existing, segmented, runs in cases:
        with subtests.test(msg=case_id):
            src = _case_dir(tmp_path, case_id) / "plaques"
            masks = src / "masks"
            masks.mkdir(parents=True)
            sizes = {name: _label_tif(masks / name, blobs) for name, blobs in existing.items()}
            (masks / "notes.txt").write_text("ignore me", encoding="utf-8")
            (masks / "sub.tif").mkdir()
            seen, finetune = {}, _Recorder()

            def fake_segment(settings, model_path):
                seen.update(settings, model_path=model_path)
                for name, blobs in segmented.items():
                    sizes[name] = _label_tif(Path(settings["dst"]) / name, blobs)
                return len(segmented)

            settings = {"src": str(src), **over}
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(SU, "download_models", _Recorder(result="/nowhere"))
                mp.setattr(SCP, "identify_masks_finetune", finetune)
                mp.setattr(SUB, "_resolve_plaque_model",
                           lambda settings, fetch=True: "/models/plaque.CP_model")
                mp.setattr(SUB, "_segment_plaque_folder", fake_segment)
                for _ in range(runs):
                    assert SUB.analyze_plaques(dict(settings) if runs > 1 else settings) is None
            assert finetune.calls == []
            if segmented:
                assert (seen["dst"], seen["diameter"]) == (str(masks), 30)
                assert seen["model_path"] == seen["custom_model"] == "/models/plaque.CP_model"
            else:
                assert seen == {}
                if runs == 1:
                    assert settings["dst"] == str(masks)
                    assert settings["custom_model"] == "/models/plaque.CP_model"
            with sqlite3.connect(masks / "plaques_analysis.db") as con:
                summary = pd.read_sql("SELECT * FROM summary", con).set_index("file")
                stats = pd.read_sql("SELECT * FROM stats", con).set_index("file")
                details = pd.read_sql("SELECT * FROM details", con)
            assert sorted(summary.index) == sorted(stats.index) == sorted(sizes)
            assert len(details) == sum(len(s) for s in sizes.values())
            for name, s in sizes.items():
                assert summary.loc[name, "object_count"] == len(s)
                assert summary.loc[name, "average_size"] == pytest.approx(np.mean(s))
                assert stats.loc[name, "plaque_count"] == len(s)
                assert stats.loc[name, "average_size"] == pytest.approx(np.mean(s))
                assert stats.loc[name, "std_dev_size"] == pytest.approx(np.std(s))
                assert sorted(details.loc[details["file"] == name, "plaque_size"]) == s


# ---------------------------------------------------------------------------
# analyze_percent_positive
# ---------------------------------------------------------------------------

ROWS2, COLS4 = ["r1", "r2"], ["c1", "c2", "c3", "c4"]


def _letter(row_id):
    return "ABCDEFGH"[int(row_id[1:]) - 1]


def _screen(folder):
    """cell table: f1 above threshold, f2 above only in c1, f3 below and small;
    rename_log: two rows per well."""
    rows, label = [], 0
    for r in ROWS2:
        for c in COLS4:
            for f in ("f1", "f2", "f3"):
                label += 1
                value = {"f1": 3000.0, "f2": 3000.0 if c == "c1" else 1000.0, "f3": 500.0}[f]
                rows.append({"object_label": label, "plateID": "plate1", "rowID": r,
                             "columnID": c, "fieldID": f, "prcf": f"plate1_{r}_{c}_{f}",
                             "cell_area": 500.0 if f == "f3" else 1500.0,
                             "cell_channel_1_mean_intensity": value})
    (folder / "measurements").mkdir(parents=True)
    with sqlite3.connect(folder / "measurements" / "measurements.db") as con:
        pd.DataFrame(rows).to_sql("cell", con, index=False)
    log = [{"Original File": f"raw_{_letter(r)}{int(c[1:]):02d}_{f}_ch1.tif",
            "Renamed TIFF": f"plate1_{_letter(r)}{int(c[1:]):02d}_{f}.tif"}
           for r in ROWS2 for c in COLS4 for f in ("f1", "f2")]
    pd.DataFrame(log).to_csv(folder / "rename_log.csv", index=False)


def test_analyze_percent_positive_counts_objects_above_the_threshold_per_well(
        subtests, tmp_path):
    """Each well's above / below counts and fractions are exact (with and
    without the area filter), every well is translated back to its plate
    well name, and a well id that is not plate_row_column stops the run."""
    from spacr import schema

    cases = [
        # (id, filter_1, expected (c1 above, c1 below, others above, others below))
        ("the area filter drops the small f3 objects", ["cell_area", 1000], (2, 0, 1, 1)),
        ("no filter counts every object", None, (2, 1, 1, 2)),
    ]
    for case_id, filter_1, (c1a, c1b, oa, ob) in cases:
        with subtests.test(msg=case_id):
            src = _case_dir(tmp_path, case_id) / "screen"
            _screen(src)
            captured = {}
            real_merge = pd.merge

            def spy(left, right, *a, **kw):
                out = real_merge(left, right, *a, **kw)
                if kw.get("on") == ["rowID", "column_name"]:
                    captured.update(count=left.copy(), translate=right.copy())
                return out

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(pd, "merge", spy)
                merged = SUB.analyze_percent_positive({
                    "src": str(src), "tables": ["cell"], "filter_1": filter_1,
                    "value_col": "cell_channel_1_mean_intensity", "threshold": 2000.0})
            counts = captured["count"].set_index("column_name")
            assert len(counts) == 8 and set(counts["condition"]) == {"none"}
            assert set(counts["plateID"]) == {"plate1"} and set(counts["rowID"]) == set(ROWS2)
            for col in COLS4:
                above, below = (c1a, c1b) if col == "c1" else (oa, ob)
                rows = counts.loc[[col]]
                assert rows["above"].tolist() == [above] * 2
                assert rows["below"].tolist() == [below] * 2
                assert rows["total"].tolist() == [above + below] * 2
                np.testing.assert_allclose(rows["fraction_above"], above / (above + below))
                np.testing.assert_allclose(rows["fraction_below"], below / (above + below))
            tdf = captured["translate"]
            assert len(tdf) == 8
            assert set(tdf["plate_well"]) == {f"plate1_{_letter(r)}{int(c[1:]):02d}"
                                              for r in ROWS2 for c in COLS4}
            assert set(tdf["prc"]) == {f"p1_{r}_{c}" for r in ROWS2 for c in COLS4}
            assert set(tdf["fieldID"]) == {"f1"}
            row = tdf[tdf["well"] == "A03"].iloc[0]
            assert (row["plateID"], row["rowID"], row["column_name"]) == ("plate1", "r1", "c3")
            assert len(merged) == 8
            assert {"well", "plate_well", "fieldID", "rowID", "column_name", "Original File",
                    "Renamed TIFF", "above", "below", "fraction_above",
                    "fraction_below"} <= set(merged.columns)
            pd.testing.assert_frame_equal(pd.read_csv(src / "result.csv")[["above", "below"]],
                                          merged[["above", "below"]].reset_index(drop=True))
            assert (src / "settings" / "analyze_percent_positive.csv").is_file()

    with subtests.test(msg="a well id that is not a prc key is named"):
        frame = pd.DataFrame({"prc": ["plate1_r1_c1", "plate1_r1_c1", "plate1", "plate1"],
                              "cell_area": [1500.0] * 4,
                              "cell_channel_1_mean_intensity": [3000.0, 500.0, 3000.0, 500.0]})
        src = _case_dir(tmp_path, "short key") / "screen"
        (src / "measurements").mkdir(parents=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(SIO, "_read_and_merge_data", lambda locs, tables, **kw: (frame.copy(), []))
            with pytest.raises(schema.KeyParseError) as excinfo:
                SUB.analyze_percent_positive({
                    "src": str(src), "tables": ["cell"], "filter_1": ["cell_area", 1000],
                    "value_col": "cell_channel_1_mean_intensity", "threshold": 2000.0})
        message = str(excinfo.value)
        for fragment in ("1 well id(s)", "'plate1'", "plate_row_column"):
            assert fragment in message


# ---------------------------------------------------------------------------
# analyze_replication
# ---------------------------------------------------------------------------

TWO_CONDITIONS = [{"row": "r1", "column": "c1", "cell": 1, "n": 2},
                  {"row": "r1", "column": "c1", "cell": 2, "n": 4},
                  {"row": "r2", "column": "c1", "cell": 1, "n": 2},
                  {"row": "r1", "column": "c2", "cell": 1, "n": 8},
                  {"row": "r2", "column": "c2", "cell": 1, "n": 4}]


def _replication_source(root, spec, drop_area=False, parasite_columns=None):
    from tests.test_replication_assay import write_db

    src = write_db(root, spec)
    if drop_area or parasite_columns:
        db = os.path.join(src, "measurements", "measurements.db")
        with sqlite3.connect(db) as con:
            frame = pd.read_sql("SELECT * FROM pathogen", con)
            if drop_area:
                frame = frame.drop(columns=["pathogen_area"])
            for column, value in (parasite_columns or {}).items():
                frame[column] = value
            frame.to_sql("pathogen", con, index=False, if_exists="replace")
    return src


def test_analyze_replication_counts_parasites_per_vacuole_and_well(
        subtests, tmp_path, monkeypatch, capsys):
    """Every vacuole's parasite count is exact with or without an area
    column, a group the cell table lacks seeds no empty wells, and QC names
    only the wells it flags."""
    from tests.test_replication_assay import settings_for

    cases = [
        # (id, spec, source kwargs, settings overrides, expected n_parasites, n wells)
        ("two conditions", TWO_CONDITIONS, {}, {}, [2, 2, 4, 4, 8], 4),
        ("no area column", TWO_CONDITIONS, {"drop_area": True}, {}, [2, 2, 4, 4, 8], 4),
        ("a parasite-only group column", TWO_CONDITIONS, {"parasite_columns": {"strain": "RH"}},
         {"group_column": "strain"}, [2, 2, 4, 4, 8], 4),
    ]
    for case_id, spec, src_kw, over, n_parasites, n_wells in cases:
        with subtests.test(msg=case_id):
            src = _replication_source(_case_dir(tmp_path, case_id) / "plate1", spec, **src_kw)
            out = SUB.analyze_replication(settings_for(src, **over))
            assert sorted(out["vacuoles"]["n_parasites"]) == n_parasites
            assert len(out["wells"]) == n_wells
            assert ("total_parasite_area" in out["vacuoles"].columns) is not src_kw.get("drop_area",
                                                                                        False)
            if "group_column" in over:
                assert set(out["wells"]["strain"]) == {"RH"}
            else:
                assert "condition" in out["wells"].columns

    with subtests.test(msg="verbose QC names only flagged wells"):
        clean = _replication_source(tmp_path / "clean", TWO_CONDITIONS)
        capsys.readouterr()
        SUB.analyze_replication(settings_for(clean, verbose=True))
        assert "QC:" not in capsys.readouterr().out
        odd = _replication_source(tmp_path / "odd", [{"row": "r1", "column": "c1", "cell": 1, "n": 3},
                                                     {"row": "r1", "column": "c2", "cell": 1, "n": 4}])
        SUB.analyze_replication(settings_for(odd, verbose=True))
        flagged = capsys.readouterr().out
        assert "QC:" in flagged and "plate1_r1_c1" in flagged

    with subtests.test(msg="the API runs before the GUI registers its defaults"):
        from spacr import settings as settings_module
        src = _replication_source(tmp_path / "hook", TWO_CONDITIONS)
        with_hook = SUB.analyze_replication(settings_for(src))
        with monkeypatch.context() as mp:
            mp.delattr(settings_module, "set_analyze_replication_defaults")
            without_hook = SUB.analyze_replication(settings_for(src))
        pd.testing.assert_frame_equal(without_hook["wells"], with_hook["wells"])
        assert list(without_hook["vacuoles"]["n_parasites"]) == list(
            with_hook["vacuoles"]["n_parasites"])
