"""The Outliers screen, offscreen and synchronous.

Every number the screen shows is the engine's, so this file checks the wiring
and nothing else: that the frame reaches the engine, that both result tables
populate, that the report is the engine's ``report()`` verbatim, that the
export writes the whole table rather than the flagged part of it, and that a
refusal from the engine arrives as a readable sentence instead of a traceback.

The fixture is the same planted plate ``tests/test_outlier_model.py`` pins by
hand: 8 wells x 25 objects of seeded lognormal with one well multiplied by 1.4.
That well is loud enough to be found in 200 objects, which keeps this file's
assertions about *what is displayed* independent of the engine's own
arithmetic — the numbers themselves are asserted over there.

``threaded=False`` throughout: the JobRunner then runs each job inline through
the same signals in the same order, so a scan has finished by the time
``set_frame`` returns and there is no waiting to get flaky.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.outliers import (
    APP_CLI_NOTE, APP_INTRO, APP_KEY, APP_NAME, MAX_TABLE_ROWS,
    OutliersScreen, make_outliers_screen, register,
)
from spacr.qt.widgets.outlier_model import (
    METHOD_IQR, METHOD_MAD, METHOD_MAHALANOBIS, TRANSFORM_LOG10,
    TRANSFORM_NONE, OBJECT_COLUMNS,
)

pytestmark = pytest.mark.qt


def planted_plate(seed: int = 3, bad_well: int = 3, shift: float = 1.4,
                  n_wells: int = 8, per_well: int = 25) -> pd.DataFrame:
    """Eight wells, one of them shifted; small enough to draw entirely."""
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


@pytest.fixture
def screen(qtbot):
    widget = OutliersScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# It builds, and it says so when there is nothing to say
# ---------------------------------------------------------------------------

def test_it_constructs_offscreen_with_nothing_loaded(screen):
    assert screen.objectName() == "OutliersScreen"
    assert screen.result is None
    assert screen.frame is None
    assert screen.object_table.rowCount() == 0
    assert screen.well_table.rowCount() == 0
    assert screen.report.toPlainText() == ""
    assert not screen.is_busy()


def test_scanning_without_a_table_says_so_rather_than_raising(screen):
    screen.scan()
    assert "Load a table first" in screen._source.text()
    assert screen.result is None


def test_the_factory_returns_a_screen():
    widget = make_outliers_screen()
    try:
        assert isinstance(widget, OutliersScreen)
    finally:
        widget.close()
        widget.deleteLater()


def test_the_screen_shows_and_paints_with_results_in_it(screen, qtbot):
    """The check ``tests/qt/test_all_module_smoke.py`` makes of every app.

    A QWidget method that shadows a Qt virtual — ``TrainCompareScreen.metric``
    did exactly this — passes every unit test and terminates the process the
    moment Qt calls it with a paint-device argument. Showing and grabbing is
    what catches it, and this screen has a ``frame`` property, a ``report``
    attribute and a ``scan`` method that all had to be checked against Qt's
    own names.
    """
    screen.set_frame(planted_plate())
    screen.resize(900, 600)
    screen.show()
    qtbot.waitExposed(screen)
    assert not screen.grab().isNull()
    screen.close()


# ---------------------------------------------------------------------------
# set_frame runs the engine and both tables populate
# ---------------------------------------------------------------------------

def test_set_frame_scans_and_fills_both_result_tables(screen):
    frame = planted_plate()
    screen.set_frame(frame)

    result = screen.result
    assert result is not None
    assert result.n_rows_in == 200

    # Objects: one row per object up to the cap, and the columns the engine
    # added are the ones on screen.
    assert screen.object_table.rowCount() == min(200, MAX_TABLE_ROWS)
    headers = [screen.object_table.horizontalHeaderItem(i).text()
               for i in range(screen.object_table.columnCount())]
    assert "cell_area" in headers
    assert OBJECT_COLUMNS["outlier"] in headers
    assert OBJECT_COLUMNS["score"] in headers
    assert OBJECT_COLUMNS["reason"] in headers

    # Wells: every well, including the ones with nothing wrong.
    assert screen.well_table.rowCount() == 8
    well_headers = [screen.well_table.horizontalHeaderItem(i).text()
                    for i in range(screen.well_table.columnCount())]
    for name in ("columnID", "n_objects", "n_flagged_objects",
                 "flagged_share", "well_outlier_score", "well_outlier"):
        assert name in well_headers


def test_the_report_tab_is_the_engines_report_verbatim(screen):
    screen.set_frame(planted_plate())
    assert screen.report.toPlainText() == screen.result.report()
    assert "Flagged is not deleted" in screen.report.toPlainText()


def test_the_header_counts_come_from_the_result_not_from_the_table(screen):
    screen.set_frame(planted_plate())
    result = screen.result
    assert screen.tabs.tabText(0) == f"Objects ({result.n_flagged:,})"
    assert screen.tabs.tabText(1) == f"Wells ({len(result.flagged_wells())})"
    assert screen._source.text() == result.headline()


def test_the_planted_well_is_the_top_row_of_the_well_table(screen):
    screen.set_frame(planted_plate())
    result = screen.result
    # bad_well = 3 is the fourth column of the plate.
    assert result.flagged_wells() == (("p1", "r1", "c4"),)
    # The table is sorted flagged-first, so the planted well is row 0 and its
    # "flagged" cell reads yes.
    headers = [screen.well_table.horizontalHeaderItem(i).text()
               for i in range(screen.well_table.columnCount())]
    flagged_column = headers.index("well_outlier")
    assert screen.well_table.item(0, flagged_column).text() == "yes"
    assert screen.well_table.item(1, flagged_column).text() == "no"
    assert screen.well_table.item(
        0, headers.index("columnID")).text() == "c4"


def test_the_worst_object_is_the_top_row_of_the_object_table(screen):
    screen.set_frame(planted_plate())
    headers = [screen.object_table.horizontalHeaderItem(i).text()
               for i in range(screen.object_table.columnCount())]
    score = headers.index(OBJECT_COLUMNS["score"])
    first = float(screen.object_table.item(0, score).text())
    second = float(screen.object_table.item(1, score).text())
    assert first >= second
    assert first == pytest.approx(float(np.nanmax(screen.result.scores)),
                                 rel=1e-4)


def test_set_frame_can_load_without_scanning(screen):
    screen.set_frame(planted_plate(), scan=False)
    assert screen.result is None
    assert screen.features.available()          # the picker is populated
    assert screen.object_table.rowCount() == 0
    screen.scan()
    assert screen.result is not None


# ---------------------------------------------------------------------------
# The controls reach the spec
# ---------------------------------------------------------------------------

def test_the_controls_are_the_spec(screen):
    screen.set_frame(planted_plate(), scan=False)
    spec = screen.spec()
    assert spec.method == METHOD_MAD
    assert spec.k == 3.5
    assert spec.transform == TRANSFORM_NONE
    assert spec.per_well is True
    assert spec.min_well_objects == 20
    assert set(spec.features) == {"cell_area", "cell_perimeter"}


def test_choosing_a_method_repoints_the_single_threshold_box(screen):
    for index, (method, k) in enumerate(
            ((METHOD_MAD, 3.5), (METHOD_IQR, 1.5),
             (METHOD_MAHALANOBIS, 0.001))):
        screen.method.setCurrentIndex(index)
        assert screen.current_method() == method
        assert screen.threshold.value() == pytest.approx(k)
    assert "α" in screen.threshold_label.text()
    screen.method.setCurrentIndex(0)
    assert "k" in screen.threshold_label.text()


def test_a_stricter_threshold_flags_fewer_objects(screen):
    screen.set_frame(planted_plate())
    loose = screen.result.n_flagged
    screen.threshold.setValue(10.0)
    screen.scan()
    assert screen.result.threshold == 10.0
    assert screen.result.n_flagged <= loose


def test_turning_the_well_pass_off_empties_the_well_table(screen):
    screen.set_frame(planted_plate())
    assert screen.well_table.rowCount() == 8
    screen.per_well.setChecked(False)
    screen.scan()
    assert screen.result.has_wells is False
    assert screen.well_table.rowCount() == 0
    assert screen.tabs.tabText(1) == "Wells"


def test_the_multivariate_method_runs_from_the_screen(screen):
    screen.set_frame(planted_plate(), scan=False)
    screen.method.setCurrentIndex(2)
    screen.scan()
    assert screen.result.method == METHOD_MAHALANOBIS
    # chi2.ppf(0.999, 2) = 13.8155.
    assert screen.result.threshold == pytest.approx(13.8155, abs=1e-3)
    assert screen.object_table.rowCount() == 200


def test_the_log_transform_is_offered_and_reaches_the_engine(screen):
    screen.set_frame(planted_plate(), scan=False)
    screen.transform.setCurrentIndex(1)
    screen.scan()
    assert screen.result.transform == TRANSFORM_LOG10
    assert "log10-transformed" in screen.report.toPlainText()


# ---------------------------------------------------------------------------
# Refusals arrive as sentences
# ---------------------------------------------------------------------------

def test_an_engine_refusal_is_shown_and_not_raised(screen, qtbot):
    frame = planted_plate()
    frame["cell_area"] = frame["cell_area"] - frame["cell_area"].max()
    with qtbot.waitSignal(screen.failed, timeout=1000) as blocker:
        screen.set_frame(frame)
        screen.transform.setCurrentIndex(1)
        screen.scan()
    assert "log10" in blocker.args[0]
    assert "log10" in screen._source.text()
    assert screen.report.toPlainText() == blocker.args[0]


def test_a_table_with_no_well_columns_refuses_with_a_way_out(screen, qtbot):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"alpha": rng.normal(size=60),
                          "beta": rng.normal(size=60)})
    with qtbot.waitSignal(screen.failed, timeout=1000) as blocker:
        screen.set_frame(frame)
    assert "per_well=False" in blocker.args[0]
    assert screen.result is None or not screen.result.has_wells
    # And the way out works from the screen.
    screen.per_well.setChecked(False)
    screen.scan()
    assert screen.result is not None
    assert screen.result.has_wells is False


def test_a_failed_load_is_reported_inline(screen, qtbot, tmp_path):
    missing = tmp_path / "not_here.db"
    with qtbot.waitSignal(screen.failed, timeout=2000):
        screen.load_path(str(missing))
    assert screen.result is None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_writes_the_whole_table_the_flagged_rows_and_the_wells(
        screen, tmp_path, monkeypatch):
    screen.set_frame(planted_plate())
    target = tmp_path / "scan.csv"
    monkeypatch.setattr(
        "spacr.qt.screens.outliers.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "CSV (*.csv)")))
    screen.export_csv()

    objects = pd.read_csv(tmp_path / "scan_objects.csv")
    flagged = pd.read_csv(tmp_path / "scan_flagged.csv")
    wells = pd.read_csv(tmp_path / "scan_wells.csv")
    report = (tmp_path / "scan_report.txt").read_text(encoding="utf-8")

    # The whole table, not the flagged part of it.
    assert len(objects) == 200
    assert len(flagged) == screen.result.n_flagged
    assert len(flagged) < len(objects)
    assert len(wells) == 8
    for name in OBJECT_COLUMNS.values():
        assert name in objects.columns
    assert report.strip() == screen.result.report().strip()
    assert "wrote scan_objects" in screen._source.text()


def test_export_before_a_scan_says_so(screen, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "spacr.qt.screens.outliers.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "x.csv"), "")))
    screen.export_csv()
    assert "run a Scan first" in screen._source.text()
    assert not (tmp_path / "x_objects.csv").exists()


def test_a_cancelled_export_dialog_writes_nothing(screen, tmp_path,
                                                  monkeypatch):
    screen.set_frame(planted_plate())
    monkeypatch.setattr(
        "spacr.qt.screens.outliers.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")))
    screen.export_csv()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_is_idempotent_and_lands_in_explore():
    """One row in `spacr.qt.SELF_REGISTERING_MODULES` turns the screen on."""
    from spacr.qt import app as app_mod

    if any(row[0] == APP_KEY for row in app_mod.APPS):
        # Something already ran the registration (a launch test, most likely).
        # The only thing left to assert is that a second call is a no-op.
        assert register() is False
        return
    try:
        assert register() is True
        assert register() is False          # never raises on the duplicate
        rows = [row for row in app_mod.APPS if row[0] == APP_KEY]
        assert len(rows) == 1
        assert rows[0][1] == APP_NAME
        # Explore, not Results & QC. The screen answers a QC question, but it
        # answers it the way the Gate Editor beside it does — pick features,
        # move a threshold, get a COLUMN back — and Results & QC is where the
        # screens that hand back a verdict live. The cap decided it in
        # practice: that section was at 12 of 13 with Control Charts, the
        # campaign-level verdict, arriving in the same batch.
        assert rows[0][3] == app_mod.SECTION_EXPLORE
        assert app_mod.APP_STAGE[APP_KEY] == app_mod.STAGE_ALPHA
        assert callable(app_mod.APP_FACTORIES[APP_KEY])
        meta = app_mod.APP_META[APP_KEY]
        assert meta["intro"] == APP_INTRO
        assert meta["cli_note"] == APP_CLI_NOTE
        assert meta["api_module"] == "qt/screens/outliers"
        assert len(meta["translations"]) == 9
    finally:
        # Leave the process-wide registry as it was found. A stray row is a
        # stray tile, a stray sidebar entry and a stray "GUI-only" excuse for
        # every test that runs after this one.
        app_mod.unregister_app(APP_KEY)


def test_the_module_does_not_register_itself_at_import():
    """Importing a screen must not mutate the registry — see register()."""
    import importlib
    import spacr.qt.screens.outliers as module
    from spacr.qt.app import APPS

    before = len([row for row in APPS if row[0] == APP_KEY])
    importlib.reload(module)
    after = len([row for row in APPS if row[0] == APP_KEY])
    assert after == before
