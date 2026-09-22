"""Every module that reads a measurements database offers the example plate.

Item 460. Asked on 2026-09-21: "Modules that read a measurements DB" should get
a "Load test data" button. The button reuses the Annotate example (a real
plate: ``measurements/measurements.db`` and the crops it indexes), which Annotate
and Classify already download into the shared example plate folder.

The download is replaced here by a function that writes a small plate with
the same tables, so these tests say what the button does to each screen --
fills its source and opens it -- without the 280 MB archive.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtWidgets import QPushButton

from spacr.qt.widgets import measurements_example as mx

#: Screens that read a measurements database and get the button, with the
#: check that the example reached them.
WIRED = ("tabulate", "graph_builder", "gate_editor", "plate_view",
         "db_browser", "lineage", "qc_dashboard", "report", "umap")

#: Screens considered and left without the button, with the reason. A button
#: here would load a plate the screen has nothing to show for.
NOT_WIRED = {
    "dose_response": "the example has no concentration column to fit",
    "profiler": "reads a regression coefficient table, not a database",
    "investigate_hit": "starts from a regression hit, which needs Regression",
    "run_compare": "compares registered runs; the example has no registry",
    "pipeline_graph": "draws provenance; the example has no registry",
    "project_browser": "lists project folders; it does not read a database",
    "layer_viewer": "opens images and masks, not a database",
    "power": "a design calculator; it reads no data at all",
}


def _write_plate(folder: Path) -> Path:
    """Write a four-well plate with cells, nuclei and pathogens inside them."""
    database = folder / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    wells = [("r2", "c2"), ("r2", "c3"), ("r3", "c2"), ("r3", "c3")]
    cells, nuclei, pathogens = [], [], []
    for index, (row, column) in enumerate(wells):
        for label in range(1, 6):
            key = dict(plateID="plate1", rowID=row, columnID=column,
                       fieldID="f1", prcf=f"plate1_{row}_{column}_f1")
            cells.append(dict(key, object_label=label,
                              cell_area=100.0 * (index + 1) + label))
            nuclei.append(dict(key, object_label=label, cell_id=label,
                               nucleus_area=20.0 + label))
            pathogens.append(dict(key, object_label=label, cell_id=label,
                                  pathogen_area=5.0 + label))
    with sqlite3.connect(database) as connection:
        pd.DataFrame(cells).to_sql("cell", connection, index=False)
        pd.DataFrame(nuclei).to_sql("nucleus", connection, index=False)
        pd.DataFrame(pathogens).to_sql("pathogen", connection, index=False)
        pd.DataFrame([dict(annotation_column="x")]).iloc[:0].to_sql(
            "annotation_round_log", connection, index=False)
    return database


@pytest.fixture
def example_folder(tmp_path, monkeypatch):
    """Point the shared example folder at an empty temporary one."""
    folder = tmp_path / "example_data" / "plate1"
    monkeypatch.setattr(mx, "example_measurements_folder", lambda: folder)
    return folder


@pytest.fixture
def fake_download():
    """A stand-in for ``download_annotate_example`` that writes the plate."""
    calls = []

    def ask(parent, destination, on_done):
        """Write the plate into ``destination`` and report success."""
        calls.append(Path(destination))
        _write_plate(Path(destination))
        on_done(object(), "")

    ask.calls = calls
    return ask


def _build(qtbot, key):
    """Build the screen the app would build for ``key``."""
    from spacr.qt.app import MainWindow

    from .test_all_module_smoke import _FactoryHost

    screen = MainWindow._build_screen(_FactoryHost(), key)
    qtbot.addWidget(screen)
    return screen


def _test_data_buttons(screen):
    """Every "Load test data…" button the shared helper put on ``screen``."""
    return [button for button in screen.findChildren(QPushButton)
            if button.objectName() == mx.BUTTON_OBJECT_NAME]


def _loaded(key, screen, folder, database):
    """Whether ``screen`` has the example in its source and has opened it."""
    if key in ("tabulate", "graph_builder", "gate_editor"):
        return (screen._path == str(database)
                and screen._frame is not None and len(screen._frame) == 20)
    if key == "plate_view":
        return (screen._path_edit.text() == str(database)
                and screen.current_value_column() == mx.EXAMPLE_MEASUREMENT
                and screen._frame is not None)
    if key == "db_browser":
        return (screen.database_path() == str(database)
                and screen._table == mx.EXAMPLE_TABLE
                and screen._explicit_path == ""
                and not screen.edit_mode_enabled())
    if key == "lineage":
        return (screen._db.text() == str(database)
                and screen.tree.topLevelItemCount() > 0)
    if key == "qc_dashboard":
        return screen.source() == str(folder)
    if key == "report":
        return (screen._path_edit.text() == str(folder)
                and screen.report is not None)
    if key == "umap":
        return screen._settings_model.collect().get("src") == str(folder)
    raise AssertionError(key)


@pytest.mark.parametrize("key", WIRED)
def test_the_button_downloads_fills_the_source_and_opens_it(
        qtbot, example_folder, fake_download, key):
    """One button; pressing it fetches the plate and the screen opens it."""
    screen = _build(qtbot, key)
    buttons = _test_data_buttons(screen)
    assert len(buttons) == 1, f"{key} has {len(buttons)} test-data buttons"
    assert "test data" in buttons[0].text().lower()

    handed = mx.load_test_data(screen, ask=fake_download)

    database = example_folder / "measurements" / "measurements.db"
    assert fake_download.calls == [example_folder]
    assert handed == {"folder": str(example_folder), "db": str(database)}
    qtbot.waitUntil(
        lambda: _loaded(key, screen, example_folder, database), timeout=20_000)
    assert buttons[0].isEnabled()


@pytest.mark.parametrize("key", ("tabulate", "lineage"))
def test_a_cached_plate_is_reused_without_downloading(
        qtbot, example_folder, key):
    """The second press, or a plate another module fetched, costs nothing."""
    database = _write_plate(example_folder)
    screen = _build(qtbot, key)

    def refuse(*_args):
        """Fail the test if a download is attempted."""
        raise AssertionError("downloaded a plate that was already cached")

    handed = mx.load_test_data(screen, ask=refuse)
    assert handed["db"] == str(database)
    qtbot.waitUntil(
        lambda: _loaded(key, screen, example_folder, database), timeout=20_000)


def test_a_failed_download_says_so_and_gives_the_button_back(
        qtbot, example_folder):
    """No plate, no pretending: the reason is shown and the button restored."""
    screen = _build(qtbot, "lineage")
    button = _test_data_buttons(screen)[0]

    def fail(_parent, _destination, on_done):
        """Report the failure the real worker would."""
        assert not button.isEnabled()
        on_done(None, "no network")

    assert mx.load_test_data(screen, ask=fail) == {}
    assert button.isEnabled()
    assert "test data" in button.text().lower()
    assert "no network" in screen.status.text()
    assert screen._db.text() == ""


@pytest.mark.parametrize("key", sorted(NOT_WIRED))
def test_the_modules_the_example_does_not_fit_have_no_button(qtbot, key):
    """A decision, recorded: see NOT_WIRED for why each one was left out."""
    screen = _build(qtbot, key)
    assert _test_data_buttons(screen) == [], NOT_WIRED[key]


def test_the_umap_control_sits_in_its_input_section(qtbot):
    """Built from ``EXAMPLE_DATA_SECTIONS``, so it must name a real section."""
    from spacr.qt.screens.app_screen import EXAMPLE_DATA_SECTIONS

    screen = _build(qtbot, "umap")
    wanted = EXAMPLE_DATA_SECTIONS["umap"].upper()
    holding = [sec for sec in screen._settings_sections
               if getattr(sec, "_header", None) is not None
               and sec._header.text().replace("&&", "&").upper()
               .startswith(wanted)]
    assert len(holding) == 1
    assert len(_test_data_buttons(holding[0])) == 1


def test_every_plate_template_loads_into_experiment_design(qtbot):
    """One menu pick replaces the design; the plate name is kept."""
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen
    from spacr.qt.widgets.plate_layout import assign_wells, plate_templates

    templates = plate_templates()
    formats = {t.design.plate_format for t in templates}
    assert {96, 384} <= formats
    assert len(templates) >= 4

    screen = ExperimentDesignScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._plate_id.setText("my_plate")
    actions = screen._template_button.menu().actions()
    assert [a.text() for a in actions] == [t.title for t in templates]
    for action, template in zip(actions, templates):
        assert template.description
        action.trigger()
        design = screen.design()
        assert design.plate_id == "my_plate"
        assert design.plate_format == template.design.plate_format
        assert design.layout == template.design.layout
        assert design.edge_policy == template.design.edge_policy
        assert design.conditions == template.design.conditions
        assert len(assign_wells(design)) == template.design.wells_requested
        assert "usable wells assigned" in screen.status_text()
