"""The two new screens, driven headlessly.

Both follow the same contract as the rest of the Qt layer: a ``threaded``
kwarg that runs work inline without the behaviour diverging, a ``JobRunner``
for anything that touches the disk, a QSS block registered through the theme
seam rather than set inline, and an idempotent ``register()``.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import experiment_design as design_module  # noqa: E402
from spacr.qt.screens import qc_dashboard as qc_module  # noqa: E402
from spacr.qt.widgets.plate_layout import (  # noqa: E402
    EDGE_LEAVE_EMPTY, ROLE_NEGATIVE, ROLE_POSITIVE, Condition,
)

pytestmark = pytest.mark.qt


@pytest.fixture
def registry_sandbox():
    """Restore the app registry after the test.

    A leaked row is a leaked tile, a leaked sidebar button and a leaked
    keyboard binding for every test that runs afterwards.
    """
    from spacr.qt import app as app_mod

    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    meta = dict(app_mod.APP_META)
    yield app_mod
    app_mod.APPS[:] = apps
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod.APP_META.clear()
    app_mod.APP_META.update(meta)
    app_mod._refresh_sections()


# -- experiment design ------------------------------------------------------


@pytest.fixture
def designer(qtbot):
    screen = design_module.ExperimentDesignScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


def test_the_designer_opens_with_a_design_worth_starting_from(designer):
    """An empty table teaches nothing. The default has both controls and
    enough replicates to mean something."""
    roles = {condition.role for condition in designer.conditions()}
    assert ROLE_NEGATIVE in roles
    assert ROLE_POSITIVE in roles
    assert all(c.replicates >= 2 for c in designer.conditions())
    assert designer.design().plate_format == 96


def test_the_plate_draws_one_label_per_well(designer):
    assert len(designer._well_labels) == 96
    designer._format.setCurrentIndex(designer._format.findData(384))
    assert len(designer._well_labels) == 384


def test_moving_the_controls_inward_clears_the_edge_warning(designer, qtbot):
    """The screen's whole reason for existing, exercised through the form."""
    designer._layout_box.setCurrentText("column")
    designer._set_conditions([Condition("neg", 8, ROLE_NEGATIVE),
                              Condition("drug", 24)])
    designer.refresh()
    assert "plate edge" in designer.findings_text()

    designer._edge.setCurrentIndex(
        designer._edge.findData(EDGE_LEAVE_EMPTY))
    designer.refresh()
    assert "Every control well sits on the plate edge" not in \
        designer.findings_text()


def test_a_half_typed_row_does_not_break_the_plate(designer):
    """The user is in the middle of typing. Dropping the row and redrawing is
    the only behaviour that does not fight them."""
    from PySide6.QtWidgets import QTableWidgetItem

    before = len(designer.conditions())
    designer._add_row()
    row = designer._table.rowCount() - 1

    # Name cleared: the row is not a condition yet.
    designer._table.setItem(row, 0, QTableWidgetItem(""))
    designer.refresh()
    assert len(designer.conditions()) == before

    # Replicates half-typed as a bare minus sign.
    designer._table.setItem(row, 0, QTableWidgetItem("new_condition"))
    designer._table.setItem(row, 1, QTableWidgetItem("-"))
    designer.refresh()
    assert len(designer.conditions()) == before

    # Finished typing: it counts.
    designer._table.setItem(row, 1, QTableWidgetItem("4"))
    designer.refresh()
    assert len(designer.conditions()) == before + 1
    assert designer.status_text()


def test_a_design_that_does_not_fit_says_so_instead_of_drawing_it(designer):
    designer._set_conditions([Condition("a", 500)])
    designer.refresh()
    assert "usable" in designer.status_text()
    assert designer._status.property("spacrError") == "true"


def test_export_writes_the_three_files_and_names_the_join_key(
        designer, tmp_path):
    assert designer.export_to(tmp_path) is True
    names = {path.name for path in tmp_path.iterdir()}
    assert names == {"plate_map.csv", "plate_map.json",
                     "plate_map_settings.json"}
    assert "rowID" in designer.status_text()
    assert designer.is_busy() is False


def test_exporting_an_empty_design_refuses(designer, tmp_path):
    designer._set_conditions([])
    designer.refresh()
    assert designer.export_to(tmp_path) is False
    assert "no conditions" in designer.status_text()
    assert list(tmp_path.iterdir()) == []


def test_the_designer_registers_once(registry_sandbox):
    app_mod = registry_sandbox
    if any(row[0] == design_module.APP_KEY for row in app_mod.APPS):
        assert design_module.register() is False
        return
    assert design_module.register() is True
    assert design_module.register() is False
    meta = app_mod.APP_META[design_module.APP_KEY]
    assert meta["intro"] == design_module.APP_INTRO
    assert len(meta["translations"]) == 9


def test_the_designer_contributes_its_qss_through_the_theme_seam():
    from spacr.qt import theme as theme_mod

    assert "ExperimentDesign" in theme_mod.widget_qss_names()
    dark = theme_mod._WIDGET_QSS["ExperimentDesign"](
        theme_mod.palette_for("dark"), None)
    light = theme_mod._WIDGET_QSS["ExperimentDesign"](
        theme_mod.palette_for("light"), None)
    assert "spacrWellRole" in dark
    assert "spacrWellEdge" in dark
    assert light != dark, "the block must follow the theme"


# -- QC dashboard -----------------------------------------------------------


def _project(tmp_path, *, mixed_units=False):
    folder = tmp_path / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(folder / "measurements.db"))
    try:
        connection.execute(
            "CREATE TABLE cell (object_label INTEGER, cell_area REAL, "
            "measurement_ndim INTEGER, measurement_units TEXT)")
        rows = [(1, 120.0, 2, "px")]
        if mixed_units:
            rows.append((2, 608.4, 3, "um"))
        connection.executemany("INSERT INTO cell VALUES (?, ?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()
    return tmp_path


def test_the_dashboard_reads_a_project_and_shows_every_card(qtbot, tmp_path):
    screen = qc_module.QCDashboardScreen(
        threaded=False, src=str(_project(tmp_path)))
    qtbot.addWidget(screen)

    dashboard = screen.dashboard()
    assert dashboard is not None
    assert [card.key for card in dashboard.cards] == [
        "segmentation", "units", "leakage", "plate", "agreement"]
    assert "Measurement units" in screen.visible_text()
    assert "nothing was recomputed" in screen.status_text()


def test_a_missing_card_shows_what_to_run(qtbot, tmp_path):
    screen = qc_module.QCDashboardScreen(
        threaded=False, src=str(_project(tmp_path)))
    qtbot.addWidget(screen)
    text = screen.visible_text()
    assert "[missing]" in text
    assert "->" in text, "a missing card must say how to produce it"


def test_a_failing_check_sets_the_headline(qtbot, tmp_path):
    screen = qc_module.QCDashboardScreen(
        threaded=False, src=str(_project(tmp_path, mixed_units=True)))
    qtbot.addWidget(screen)
    assert screen.dashboard().verdict == "fail"
    assert "FAIL" in screen.as_text()
    assert "different units" in screen.as_text()


def test_a_second_read_of_unchanged_files_is_skipped(qtbot, tmp_path):
    """The cost rule: returning to the screen must not re-parse."""
    calls = []

    def _reader(src):
        from spacr.qt.widgets.qc_summary import read_dashboard
        calls.append(src)
        return read_dashboard(src)

    screen = qc_module.QCDashboardScreen(
        threaded=False, src=str(_project(tmp_path)), reader=_reader)
    qtbot.addWidget(screen)
    assert len(calls) == 1

    assert screen.refresh() is False
    assert len(calls) == 1
    assert "has changed" in screen.status_text()

    assert screen.refresh(force=True) is True
    assert len(calls) == 2


def test_a_rewritten_artifact_is_picked_up_on_the_next_visit(qtbot, tmp_path):
    calls = []

    def _reader(src):
        from spacr.qt.widgets.qc_summary import read_dashboard
        calls.append(src)
        return read_dashboard(src)

    root = _project(tmp_path)
    screen = qc_module.QCDashboardScreen(
        threaded=False, src=str(root), reader=_reader)
    qtbot.addWidget(screen)
    assert len(calls) == 1

    connection = sqlite3.connect(
        str(root / "measurements" / "measurements.db"))
    try:
        connection.execute("INSERT INTO cell VALUES (2, 608.4, 3, 'um')")
        connection.commit()
    finally:
        connection.close()

    assert screen.refresh() is True
    assert len(calls) == 2
    assert screen.dashboard().verdict == "fail"


def test_a_stale_card_is_announced_in_the_status_line(qtbot, tmp_path):
    from spacr.qt.widgets.qc_summary import Dashboard, QCCard

    stale = Dashboard(
        root=str(tmp_path), verdict="ok", headline="all clean",
        cards=[QCCard("segmentation", "Segmentation", "ok", "clean",
                      stale=True)])
    screen = qc_module.QCDashboardScreen(
        threaded=False, src=str(tmp_path), reader=lambda s: stale)
    qtbot.addWidget(screen)
    assert "out of date" in screen.status_text()
    assert "ok (out of date)" in screen.visible_text()


def test_a_missing_folder_says_so_rather_than_reading(qtbot, tmp_path):
    screen = qc_module.QCDashboardScreen(threaded=False)
    qtbot.addWidget(screen)
    assert screen.refresh() is False
    assert "Pick a project folder" in screen.status_text()

    screen._src_edit.setText(str(tmp_path / "nope"))
    assert screen.refresh() is False
    assert screen._status.property("spacrError") == "true"


def test_a_reader_that_raises_reaches_the_status_line(qtbot, tmp_path):
    def _boom(_src):
        raise RuntimeError("disk on fire")

    screen = qc_module.QCDashboardScreen(
        threaded=False, src=str(_project(tmp_path)), reader=_boom)
    qtbot.addWidget(screen)
    assert "disk on fire" in screen.status_text()
    assert screen._status.property("spacrError") == "true"


def test_the_dashboard_registers_once(registry_sandbox):
    app_mod = registry_sandbox
    if any(row[0] == qc_module.APP_KEY for row in app_mod.APPS):
        assert qc_module.register() is False
        return
    assert qc_module.register() is True
    assert qc_module.register() is False
    meta = app_mod.APP_META[qc_module.APP_KEY]
    assert len(meta["translations"]) == 9


def test_the_dashboard_contributes_its_qss_through_the_theme_seam():
    from spacr.qt import theme as theme_mod

    assert "QCDashboard" in theme_mod.widget_qss_names()
    dark = theme_mod._WIDGET_QSS["QCDashboard"](
        theme_mod.palette_for("dark"), None)
    assert "spacrQCVerdictLevel" in dark
    assert "spacrQCStale" in dark
    assert dark != theme_mod._WIDGET_QSS["QCDashboard"](
        theme_mod.palette_for("light"), None)


def test_both_factories_build_a_working_screen(qtbot):
    for factory in (design_module.make_experiment_design_screen,
                    qc_module.make_qc_dashboard_screen):
        screen = factory()
        qtbot.addWidget(screen)
        assert screen.is_busy() is False
        assert screen.active_jobs() == 0
        screen.close()
