"""Four table-driven screens: where a table comes from, and what is said about it.

Each screen takes a source (a CSV, a measurement database, a folder of runs,
a search folder) and has exactly one line that names it. These tests drive the
real screen inline and read that line back to the character, together with
what was or was not loaded, so a refusal can never pass for a load.

Screens: Outliers (load, table picking, refused methods, unwritable exports),
Small Multiples (load, table picking, formulas, closing mid-read), Project
Browser (search folders and the chooser's start folder), Training Runs (the
clicked-curve caption).
"""
from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

pytestmark = pytest.mark.qt


def _measurement_db(path, tables):
    """A SQLite measurement database holding ``tables``."""
    with sqlite3.connect(str(path)) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return str(path)


def _picker_items(picker):
    return [picker.itemText(i) for i in range(picker.count())]


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def _planted_plate(seed=3, bad_well=3, shift=1.4, n_wells=8, per_well=25):
    """Eight wells, one of them shifted; small enough to scan instantly."""
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(n_wells):
        factor = shift if well == bad_well else 1.0
        area = factor * rng.lognormal(0.0, 0.2, per_well)
        perimeter = rng.lognormal(0.0, 0.2, per_well)
        for i in range(per_well):
            rows.append(("p1", "r1", f"c{well + 1}", "f1", i,
                         area[i], perimeter[i]))
    return pd.DataFrame(rows, columns=[
        "plateID", "rowID", "columnID", "fieldID", "object_label",
        "cell_area", "cell_perimeter"])


def test_the_outliers_screen_names_its_source_and_every_refusal(
        subtests, qtbot, monkeypatch, tmp_path):
    """CSV and database loads label the source; refusals replace the label."""
    from spacr.qt.screens.outliers import OutliersScreen

    def fresh():
        widget = OutliersScreen(threaded=False)
        qtbot.addWidget(widget)
        return widget

    frame = _planted_plate()
    csv = tmp_path / "objects.csv"
    frame.to_csv(csv, index=False)

    with subtests.test(msg="a CSV loads and scans itself"):
        screen = fresh()
        screen.load_path(str(csv))
        assert screen.result is not None
        assert list(screen.frame.columns) == list(frame.columns)
        assert len(screen.frame) == 200
        assert screen._table_picker.isVisibleTo(screen) is False
        assert screen._source.text() == screen.result.headline()
        assert screen._export.isEnabled() is True

    with subtests.test(msg="the label names the file before the scan"), \
            monkeypatch.context() as patch:
        screen = fresh()
        scans = []
        patch.setattr(screen, "scan", lambda: scans.append(True))
        screen.load_path(str(csv))
        assert scans == [True]
        assert screen._source.text() == "objects.csv · 200 rows × 7 columns"

    with subtests.test(msg="a database offers its tables and reads the one picked"), \
            monkeypatch.context() as patch:
        other = frame.assign(cell_area=frame["cell_area"] * 10.0)
        path = _measurement_db(tmp_path / "measurements.db",
                               {"object": frame, "second_pass": other})
        screen = fresh()
        scans = []
        patch.setattr(screen, "scan", lambda: scans.append(True))
        screen.load_path(path)
        assert _picker_items(screen._table_picker) == ["object", "second_pass"]
        assert screen._table_picker.isVisibleTo(screen) is True
        assert screen._source.text() == (
            "measurements.db · object · 200 rows × 7 columns")
        first = screen.frame["cell_area"].mean()
        screen._on_table_picked("second_pass")
        assert screen._table_picker.currentText() == "second_pass"
        assert screen.frame["cell_area"].mean() == pytest.approx(first * 10.0)
        assert screen._source.text() == (
            "measurements.db · second_pass · 200 rows × 7 columns")
        assert scans == [True, True]
        screen._on_table_picked("")
        assert scans == [True, True]

    with subtests.test(msg="picking another table rescans it in place"):
        screen = fresh()
        screen.load_path(path)
        first = screen.result
        assert screen._source.text() == first.headline()
        assert screen.object_table.rowCount() > 0
        screen._on_table_picked("second_pass")
        assert screen.result is not first
        assert screen._source.text() == screen.result.headline()
        assert screen.frame["cell_area"].tolist() == pytest.approx(
            (frame["cell_area"] * 10.0).tolist())
        assert screen._export.isEnabled() is True

    with subtests.test(msg="a file that is not a database is refused"):
        screen = fresh()
        failures = []
        screen.failed.connect(failures.append)
        broken = tmp_path / "broken.db"
        broken.write_bytes(b"this is not a SQLite file")
        screen.load_path(str(broken))
        assert screen.frame is None
        assert len(failures) == 1
        assert failures[0].startswith("could not read broken.db: ")
        assert screen._source.text() == failures[0]

    with subtests.test(msg="a method this build does not have is refused"), \
            monkeypatch.context() as patch:
        screen = fresh()
        screen.set_frame(frame, scan=False)
        patch.setattr(screen, "current_method", lambda: "isolation_forest")
        failures = []
        screen.failed.connect(failures.append)
        screen.scan()
        assert screen.result is None
        assert len(failures) == 1
        assert failures[0].startswith(
            "unknown method 'isolation_forest'; choose one of ")
        assert screen._source.text() == failures[0]
        assert screen.report.toPlainText() == failures[0]
        assert screen._export.isEnabled() is False

    with subtests.test(msg="a scan with no table asks for one"):
        screen = fresh()
        screen.scan()
        assert screen._source.text() == "Load a table first."

    with subtests.test(msg="an export with no scan asks for one"), \
            monkeypatch.context() as patch:
        screen = fresh()
        patch.setattr(QFileDialog, "getSaveFileName", staticmethod(
            lambda *a, **k: pytest.fail("a save dialog was opened")))
        screen.export_csv()
        assert screen._source.text() == "Nothing to export — run a Scan first."

    with subtests.test(msg="an unwritable export says so and stops"), \
            monkeypatch.context() as patch:
        screen = fresh()
        screen.set_frame(frame)
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        patch.setattr(QFileDialog, "getSaveFileName", staticmethod(
            lambda *a, **k: (str(blocker / "scan.csv"), "")))
        screen.export_csv()
        assert screen._source.text().startswith(
            "could not write those files: ")
        assert "not-a-directory" in screen._source.text()
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "broken.db", "measurements.db", "not-a-directory", "objects.csv"]

    with subtests.test(msg="a written export lists what it wrote"), \
            monkeypatch.context() as patch:
        screen = fresh()
        screen.set_frame(frame)
        out = tmp_path / "out"
        out.mkdir()
        patch.setattr(QFileDialog, "getSaveFileName", staticmethod(
            lambda *a, **k: (str(out / "scan.csv"), "")))
        screen.export_csv()
        assert screen._source.text() == (
            "wrote scan_objects / _flagged / _wells .csv and _report.txt")
        assert sorted(p.name for p in out.iterdir()) == [
            "scan_flagged.csv", "scan_objects.csv", "scan_report.txt",
            "scan_wells.csv"]


# ---------------------------------------------------------------------------
# Small Multiples
# ---------------------------------------------------------------------------

def test_small_multiples_names_the_table_it_loaded_and_ignores_empty_picks(
        subtests, qtbot, tmp_path):
    """CSV and database loads reach the grid; unreadable files and blank picks do not."""
    from spacr.qt.screens import trellis as T
    from spacr.qt.widgets.formula import ColumnFormula

    def fresh():
        widget = T.TrellisScreen(threaded=False)
        qtbot.addWidget(widget)
        return widget

    csv = tmp_path / "measurements.csv"
    pd.DataFrame({"plate": ["p1", "p1", "p2", "p2"],
                  "area": [10.0, 20.0, 30.0, 40.0],
                  "perimeter": [4.0, 8.0, 12.0, 16.0]}).to_csv(csv, index=False)
    db = _measurement_db(tmp_path / "measurements.db", {
        "object": pd.DataFrame({"plate": ["p1", "p2"], "area": [1.0, 2.0]}),
        "well": pd.DataFrame({"plate": ["p1"], "count": [7]})})

    with subtests.test(msg="a CSV reaches the grid"):
        screen = fresh()
        screen.load_path(str(csv))
        assert list(screen._frame.columns) == ["plate", "area", "perimeter"]
        assert screen._source.text() == "measurements.csv · 4 rows × 3 columns"
        assert screen._table_picker.isVisibleTo(screen) is False

    with subtests.test(msg="a computed column reaches the grid without a reload"):
        screen = fresh()
        screen.load_path(str(csv))
        assert screen.formulas.add_formula(
            ColumnFormula("compactness", "area / perimeter ** 2")) is True
        screen._on_formulas_changed()
        computed = screen.formulas.computed_frame()
        assert list(computed["compactness"]) == pytest.approx(
            [10.0 / 16.0, 20.0 / 64.0, 30.0 / 144.0, 40.0 / 256.0])
        assert screen._source.text() == "measurements.csv · 4 rows × 3 columns"

    with subtests.test(msg="a database offers its tables and loads the pick"):
        screen = fresh()
        screen.load_path(db)
        assert _picker_items(screen._table_picker) == ["object", "well"]
        assert list(screen._frame.columns) == ["plate", "area"]
        assert screen._source.text() == "measurements.db · object · 2 rows × 2 columns"
        screen._on_table_picked("well")
        assert list(screen._frame.columns) == ["plate", "count"]
        assert screen._source.text() == "measurements.db · well · 1 rows × 2 columns"

    with subtests.test(msg="an explicit table is honoured"):
        screen = fresh()
        screen.load_path(db, table="well")
        assert screen._table_picker.currentText() == "well"
        assert list(screen._frame.columns) == ["plate", "count"]
        assert screen._source.text() == "measurements.db · well · 1 rows × 2 columns"

    with subtests.test(msg="an empty table name is ignored with a file loaded"):
        screen = fresh()
        screen.load_path(db)
        before = screen._source.text()
        loaded = screen._frame
        screen._on_table_picked("")
        assert screen._frame is loaded
        assert screen._source.text() == before

    with subtests.test(msg="a file whose tables cannot be listed is reported"):
        screen = fresh()
        broken = tmp_path / "not_a_database.db"
        broken.write_bytes(b"this is not a SQLite file")
        screen.load_path(str(broken))
        assert screen._source.text().startswith(
            "could not read not_a_database.db: ")
        assert screen._frame is None
        assert screen._table_picker.count() == 0

    with subtests.test(msg="closing abandons the read in flight"):
        screen = fresh()
        screen.load_path(str(csv))
        screen.close()
        assert screen.active_jobs() == 0
        assert screen.is_busy() is False


# ---------------------------------------------------------------------------
# Project Browser
# ---------------------------------------------------------------------------

def test_the_project_browser_searches_and_offers_only_probed_folders(
        subtests, qtbot, monkeypatch, tmp_path):
    """Roots are remembered when possible; the chooser starts on a probed one."""
    import spacr.qt.prefs as prefs
    from spacr.qt import path_probe
    from spacr.qt.screens.project_browser import ProjectBrowserScreen

    def pushed():
        return prefs.get_recent_sources("project_browser")

    def fresh(roots=()):
        widget = ProjectBrowserScreen(threaded=False, roots=roots)
        qtbot.addWidget(widget)
        return widget

    def chooser(patch, answer):
        seen = []

        def _chooser(_parent, title, start):
            seen.append((title, start))
            return answer

        patch.setattr(QFileDialog, "getExistingDirectory", _chooser)
        return seen

    title = "Search this folder for spaCR projects"
    older = os.path.abspath(str(tmp_path / "older"))
    newer = os.path.abspath(str(tmp_path / "newer"))
    another = os.path.abspath(str(tmp_path / "another"))
    for folder in (older, newer, another):
        os.mkdir(folder)

    with subtests.test(msg="an added folder is searched and remembered"):
        screen = fresh()
        assert screen.add_root(str(tmp_path / "older"), scan=False) is True
        assert screen.roots() == (older,)
        assert pushed() == [older]
        assert screen.add_root(older, scan=False) is False
        assert screen.roots() == (older,)
        assert pushed() == [older]

    with subtests.test(msg="a folder that cannot be remembered is still searched"), \
            monkeypatch.context() as patch:
        def _explode(*_args, **_kwargs):
            raise OSError("preferences are read-only")

        patch.setattr(prefs, "push_recent_source", _explode)
        screen = fresh()
        assert screen.add_root(older, scan=False) is True
        assert screen.roots() == (older,)

    with subtests.test(msg="the chooser starts on the newest probed root"), \
            monkeypatch.context() as patch:
        screen = fresh()
        screen._roots.extend([older, newer])
        path_probe.forget(newer)
        path_probe.prime(older, True, want_dir=True)
        seen = chooser(patch, another)
        screen.choose_root()
        assert seen == [(title, older)]
        assert screen.roots() == (older, newer, another)
        assert pushed() == [another, older]

    with subtests.test(msg="no probed root starts the chooser at home"), \
            monkeypatch.context() as patch:
        unseen = os.path.abspath(str(tmp_path / "never-asked-about"))
        screen = fresh()
        screen._roots.append(unseen)
        path_probe.forget(unseen)
        seen = chooser(patch, "")
        screen.choose_root()
        assert seen == [(title, os.path.expanduser("~"))]
        assert screen.roots() == (unseen,)

    with subtests.test(msg="a cancelled chooser adds nothing"), \
            monkeypatch.context() as patch:
        screen = fresh()
        screen._roots.append(older)
        chooser(patch, "")
        screen.choose_root()
        assert screen.roots() == (older,)
        assert pushed() == [another, older]


# ---------------------------------------------------------------------------
# Training Runs
# ---------------------------------------------------------------------------

def _series(**kwargs):
    base = dict(label="run-a train", run_id="gone", epoch_range=lambda: (1, 1),
                best=lambda metric: None, last=lambda metric: None)
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_a_clicked_training_curve_is_captioned_with_what_is_known(
        subtests, qtbot, qt_theme_applied, monkeypatch, tmp_path):
    """Epochs, last and best values and the run path, each only when known."""
    import spacr.train_compare as _tc
    from spacr.qt.screens import train_compare as tcs
    from tests.test_train_compare import BASE_SETTINGS, make_run

    monkeypatch.setattr(_tc, "_SETTINGS_SEARCH_DEPTH", 5, raising=True)

    def fresh():
        widget = tcs.TrainCompareScreen(threaded=False)
        qtbot.addWidget(widget)
        clicked = []
        widget.series_clicked.connect(clicked.append)
        return widget, clicked

    with subtests.test(msg="an unknown label clears the caption"):
        screen, clicked = fresh()
        screen._picked.setText("stale")
        assert screen.identify_series("nope") == ""
        assert screen._picked.text() == ""
        assert clicked == []

    with subtests.test(msg="a run no longer listed has no path"):
        screen, clicked = fresh()
        screen._figure.spacr_series_by_label = {"run-a train": _series()}
        assert screen.identify_series("run-a train") == "run-a train · epoch 1"
        assert screen._picked.text() == "run-a train · epoch 1"
        assert clicked == ["run-a train"]

    with subtests.test(msg="best, last and the run path are all named"):
        screen, clicked = fresh()
        screen._runs = [SimpleNamespace(run_id="r1", path="/runs/r1")]
        metric = screen.selected_metric()
        screen._figure.spacr_series_by_label = {"s": _series(
            label="s", run_id="r1", epoch_range=lambda: (1, 9),
            best=lambda m: {"epoch": 4, "value": 0.5},
            last=lambda m: {"epoch": 9, "value": 0.25})}
        assert screen.identify_series("s") == (
            f"s · epochs 1–9 · last {metric} 0.2500 @ 9 · best {metric} "
            f"0.5000 @ 4 (chosen on this same curve, so optimistic) · /runs/r1")
        assert clicked == ["s"]

    with subtests.test(msg="a metric with no direction has a last but no best"):
        screen, _clicked = fresh()
        root = str(tmp_path / "tree")
        os.makedirs(root)
        make_run(root, "dsB", epochs=25,
                 train=np.linspace(0.50, 0.93, 25),
                 val=np.linspace(0.48, 0.86, 25),
                 settings={**BASE_SETTINGS, "epochs": 25})
        assert screen.scan(root) is True
        run = next(r for r in screen.runs() if "dsB" in str(r.path))
        assert screen.select_runs([run.run_id]) is True
        assert screen.overlay() is True
        label = f"{run.run_id} · val"
        accuracy = screen.identify_series(label)
        assert accuracy.startswith(f"{label} · epochs 1–25 · last accuracy ")
        assert "best accuracy " in accuracy
        assert screen.set_metric("optimal_threshold") is True
        assert screen.identify_series(label) == (
            f"{label} · epochs 1–25 · last optimal_threshold 0.5000 @ 25 · "
            f"{run.path}")
        assert screen.picked_text() == screen._picked.text()
