"""
Import Project — the Tools screen in front of :mod:`spacr.foreign`.

Everything runs offscreen against a real temporary tree of somebody
else's TIFFs, label images and ``results.csv``, because the thing this
screen has to get right is the **mapping table it shows before anything
is written**. A mapping the user did not read is how their ``Area`` in
µm² becomes spaCR's ``cell_area`` in px².

The properties pinned here:

* it **builds offscreen** with nothing selected and Import disabled;
* Preview **writes nothing** and fills the mapping table with the
  *inferred proposal* — never a spaCR feature name;
* **editing a row changes what would be applied**, and the conflict and
  unit checks re-run in place without rescanning the disk;
* the **unmapped** columns and the **conflicting** ones are visible on
  screen, by name;
* a **blocking conflict leaves Import disabled**;
* **save/load** round-trips the mapping through the same CSV the module
  writes;
* the **import runs off the GUI thread** through the same ``finished`` →
  bound-method relay as the Plate Viewer and the Format Converter;
* **no modal dialogs anywhere** — the autouse fixture fires if one opens,
  because a QMessageBox would hang a headless run forever.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from PySide6.QtCore import Qt

from spacr import crops as cropping
from spacr import foreign as fgn
from spacr.qt.screens.foreign import (
    CONFLICT_CHOICES,
    ColumnMapModel,
    ForeignScreen,
    MAP_COLUMNS,
    OBJECT_CHOICES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIZE = 16


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    ``MakeMasksScreen._load_current`` once hung the whole headless suite
    on a QMessageBox; this fixture makes that failure mode impossible to
    reintroduce here without a red test.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


@pytest.fixture
def theirs(tmp_path):
    """A third party's images, cell masks and measurement table."""
    images = tmp_path / "their_images"
    masks = tmp_path / "their_cell_masks"
    images.mkdir()
    masks.mkdir()

    mask = np.zeros((SIZE, SIZE), np.uint16)
    mask[1:5, 1:5] = 1
    mask[8:14, 8:14] = 2

    rows = []
    for field in (1, 2):
        for channel in (1, 2):
            tifffile.imwrite(str(images / f"fov{field:02d}_C{channel}.tif"),
                             np.full((SIZE, SIZE), field * 10 + channel,
                                     np.uint16))
        tifffile.imwrite(str(masks / f"fov{field:02d}_cell_mask.tif"), mask)
        for label, area in ((1, 16.0), (2, 36.0)):
            rows.append({"ImageNumber": f"fov{field:02d}_C1.tif",
                         "ObjectNumber": label,
                         "AreaShape_Area_um2": area * 0.25,
                         "Metadata_Treatment": "wt",
                         "cell_area": 1.0})
    table = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(table, index=False)

    return {"images": str(images), "masks": {"cell": str(masks)},
            "table": str(table)}


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — jobs run inline so assertions are exact."""
    widget = ForeignScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def loaded(screen, theirs):
    """A screen pointed at the data, previewed, with the proposal on show."""
    screen.set_images(theirs["images"])
    screen.add_mask_folder("cell", theirs["masks"]["cell"])
    screen.set_measurements(theirs["table"])
    screen.set_pixel_size(0.5)
    assert screen.preview() is True
    return screen


@pytest.fixture
def threaded_screen(qtbot, qt_theme_applied):
    """The real thing: jobs go through a QThread."""
    widget = ForeignScreen(threaded=True)
    qtbot.addWidget(widget)
    yield widget
    # Never let a QThread outlive the test — one collected while running
    # takes the whole process down.
    for thread, _worker in list(widget._jobs):
        thread.quit()
        thread.wait(5000)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_it_builds_offscreen_with_nothing_selected(screen):
    assert screen.mapping_row_count() == 0
    assert screen.plan() is None
    assert screen.result() is None
    assert not screen.can_import()
    assert screen.mask_folders() == {}
    assert screen.pixel_size() is None
    assert "Preview" in screen.status_text()
    assert screen.last_error == ""


def test_the_choices_are_the_ones_the_importer_understands(screen):
    # A combo entry the backend rejects would only be discovered by a
    # user picking it.
    assert OBJECT_CHOICES == cropping.MASK_PLANE_ORDER
    assert {value for _label, value in CONFLICT_CHOICES} == set(fgn.ON_CONFLICT)
    assert screen.on_conflict() == "refuse"          # the safe default


def test_the_mapping_columns_are_the_column_map_fields():
    keys = [key for key, _label, _editable in MAP_COLUMNS]
    assert keys == list(fgn.COLUMN_MAP_COLUMNS)
    editable = {key for key, _l, e in MAP_COLUMNS if e}
    # their column name is the one thing that must not be edited: renaming
    # it would point the mapping at a column that does not exist.
    assert "source" not in editable
    assert editable == set(fgn.COLUMN_MAP_COLUMNS) - {"source"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def test_choosing_images_proposes_a_destination(screen, theirs):
    screen.set_images(theirs["images"])
    assert screen.images_path() == theirs["images"]
    assert screen.destination_path() == theirs["images"] + "_spacr"


def test_mask_folders_are_added_removed_and_kept_in_plane_order(screen, tmp_path):
    a, b = tmp_path / "nuc", tmp_path / "cell"
    a.mkdir()
    b.mkdir()
    assert screen.add_mask_folder("nucleus", str(a)) is True
    assert screen.add_mask_folder("cell", str(b)) is True
    # cell before nucleus, which is the order their planes are appended in
    assert list(screen.mask_folders()) == ["cell", "nucleus"]
    assert screen._mask_list.count() == 2

    assert screen.remove_mask_folder("nucleus") is True
    assert list(screen.mask_folders()) == ["cell"]
    assert screen.remove_mask_folder("nucleus") is False


def test_a_bad_mask_class_is_reported_inline(screen, tmp_path):
    assert screen.add_mask_folder("mitochondrion", str(tmp_path)) is False
    assert "not a spaCR mask class" in screen.status_text()
    assert screen.last_error
    assert screen.mask_folders() == {}

    assert screen.add_mask_folder("cell", "") is False
    assert "Choose a mask folder" in screen.status_text()


def test_preview_refuses_missing_inputs_inline(screen, theirs, tmp_path):
    assert screen.preview() is False
    assert "image folder" in screen.status_text()

    screen.set_images(str(tmp_path / "absent"))
    assert screen.preview() is False
    assert "Not a folder" in screen.status_text()

    screen.set_images(theirs["images"])
    assert screen.preview() is False
    assert "mask folder" in screen.status_text()

    screen.add_mask_folder("cell", str(tmp_path / "absent"))
    assert screen.preview() is False
    assert "Not a folder" in screen.status_text()

    screen.add_mask_folder("cell", theirs["masks"]["cell"])
    assert screen.preview() is False
    assert "measurement table" in screen.status_text()

    screen.set_measurements(str(tmp_path / "absent.csv"))
    assert screen.preview() is False
    assert "Not a file" in screen.status_text()
    # every one of those was inline: the autouse fixture would have fired
    assert screen.plan() is None


def test_a_blank_pixel_size_means_unknown_not_one(screen):
    screen.set_pixel_size("")
    assert screen.pixel_size() is None
    screen.set_pixel_size("not a number")
    assert screen.pixel_size() is None
    screen.set_pixel_size(0.65)
    assert screen.pixel_size() == 0.65


# ---------------------------------------------------------------------------
# The preview and the inferred mapping
# ---------------------------------------------------------------------------

def test_preview_shows_the_inferred_mapping_and_writes_nothing(loaded, tmp_path):
    plan = loaded.plan()
    assert plan is not None and plan.ok
    assert plan.proposed

    # one row per column of their table, minus the two join keys
    assert loaded.mapping_row_count() == 3
    sources = [m.source for m in loaded.column_maps()]
    assert sources == ["AreaShape_Area_um2", "Metadata_Treatment", "cell_area"]

    # never a spaCR feature name in the proposal
    assert all(not fgn.is_spacr_name(m.target) for m in loaded.column_maps())
    assert loaded.can_import()
    # nothing on disk
    assert not os.path.exists(loaded.destination_path())
    assert loaded.result() is None


def test_the_report_names_the_join_and_the_counts(loaded):
    text = loaded.report_text()
    assert "Join key" in text
    assert "ImageNumber" in text and "ObjectNumber" in text
    assert "4/4 measurement row(s) matched" in text
    assert "matched an object" in loaded.status_text()


def test_a_conflicting_column_is_visible_on_screen(loaded):
    """Their table has a literal ``cell_area``; the screen has to say so."""
    lines = loaded.conflict_lines()
    assert any("shadows_spacr" in line and "cell_area" in line
               for line in lines)
    assert "shadows_spacr" in loaded.report_text()
    # not blocking — nothing is overwritten — so Import stays live
    assert loaded.can_import()


def test_an_unmapped_column_is_visible_on_screen(loaded):
    row = next(i for i in range(loaded.mapping_row_count())
               if loaded._model.map_at(i).source == "Metadata_Treatment")
    assert loaded.set_mapping_value(row, "target", "") is True

    assert loaded.unmapped_columns() == ["Metadata_Treatment"]
    text = loaded.report_text()
    assert "COLUMNS WITH NO MAPPING (1)" in text
    assert "Metadata_Treatment" in text
    assert "1 column(s) unmapped" in loaded.status_text()


def test_an_uncalibrated_column_is_visible_when_the_scale_is_removed(loaded):
    assert loaded.plan().uncalibrated == []
    loaded.set_pixel_size("")            # the scale is now unknown
    plan = loaded.plan()
    assert len(plan.uncalibrated) == 1
    assert plan.uncalibrated[0].source == "AreaShape_Area_um2"
    assert not plan.uncalibrated[0].calibrated
    assert plan.uncalibrated[0].factor is None      # never 1.0
    assert "UNCALIBRATED COLUMNS" in loaded.report_text()
    assert "1 uncalibrated" in loaded.status_text()


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def test_editing_a_row_changes_what_would_be_applied(loaded):
    row = next(i for i in range(loaded.mapping_row_count())
               if loaded._model.map_at(i).source == "AreaShape_Area_um2")
    before = loaded.plan().target_for("AreaShape_Area_um2")
    assert before.startswith(fgn.FOREIGN_PREFIX)

    assert loaded.set_mapping_value(row, "target", "foreign_my_area") is True
    assert loaded.column_maps()[row].target == "foreign_my_area"
    assert loaded.plan().target_for("AreaShape_Area_um2") == "foreign_my_area"

    # and the edit survives into the resolution the import would run
    resolution = next(r for r in loaded.plan().resolved
                      if r.source == "AreaShape_Area_um2")
    assert resolution.target == "foreign_my_area"
    assert resolution.factor == pytest.approx(4.0)


def test_editing_a_transform_re_runs_the_unit_arithmetic(loaded):
    row = next(i for i in range(loaded.mapping_row_count())
               if loaded._model.map_at(i).source == "AreaShape_Area_um2")
    assert loaded.set_mapping_value(row, "transform", "*10") is True
    resolution = next(r for r in loaded.plan().resolved
                      if r.source == "AreaShape_Area_um2")
    assert resolution.factor == pytest.approx(10.0)
    assert resolution.calibrated


def test_editing_a_target_onto_a_spacr_name_disables_import(loaded):
    row = next(i for i in range(loaded.mapping_row_count())
               if loaded._model.map_at(i).source == "AreaShape_Area_um2")
    assert loaded.set_mapping_value(row, "target", "cell_area") is True

    assert not loaded.plan().ok
    assert not loaded.can_import()
    assert any("spacr_name" in line for line in loaded.conflict_lines())
    assert "blocking problem" in loaded.status_text()
    assert loaded.last_error

    # switching the policy to rename resolves it, in place
    loaded.set_on_conflict("rename")
    assert loaded.plan().ok
    assert loaded.can_import()
    assert loaded.plan().target_for(
        "AreaShape_Area_um2").startswith(fgn.FOREIGN_PREFIX)


def test_an_edit_that_changes_nothing_is_not_an_edit(loaded):
    row = 0
    current = loaded._model.map_at(row).target
    assert loaded.set_mapping_value(row, "target", current) is False
    assert loaded.set_mapping_value(row, "source", "nope") is False
    assert loaded.set_mapping_value(row, "not_a_field", "x") is False
    assert loaded._model.map_at(row).source != "nope"


def test_changing_an_input_invalidates_the_mapping_on_screen(loaded, theirs):
    assert loaded.mapping_row_count() > 0
    loaded.set_measurements(theirs["table"] + ".other")
    assert loaded.plan() is None
    assert loaded.mapping_row_count() == 0
    assert loaded.report_text() == ""
    assert "press Preview again" in loaded.status_text()
    assert not loaded.can_import()


# ---------------------------------------------------------------------------
# The mapping file
# ---------------------------------------------------------------------------

def test_the_mapping_saves_and_loads_through_the_module_format(loaded, tmp_path):
    path = str(tmp_path / "map.csv")
    assert loaded.save_mapping(path) is True
    assert os.path.isfile(path)
    # it really is spacr.foreign's format
    assert [m.to_row() for m in fgn.load_column_map(path)] == \
        [m.to_row() for m in loaded.column_maps()]

    # edit the file behind the screen's back, then load it
    edited = fgn.load_column_map(path)
    edited = [fgn.ColumnMap(source=m.source, target="foreign_edited",
                            transform="identity")
              if m.source == "Metadata_Treatment" else m for m in edited]
    fgn.save_column_map(edited, path)

    assert loaded.load_mapping(path) is True
    assert loaded.plan().target_for("Metadata_Treatment") == "foreign_edited"
    assert loaded._model.map_at(
        loaded._model.row_of("Metadata_Treatment")).target == "foreign_edited"


def test_saving_with_no_mapping_is_reported_inline(screen, tmp_path):
    assert screen.save_mapping(str(tmp_path / "m.csv")) is False
    assert "press Preview first" in screen.status_text()
    assert not screen._btn_save_map.isEnabled()


def test_loading_a_broken_mapping_is_reported_inline(loaded, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    assert loaded.load_mapping(str(bad)) is False
    assert "Could not load the mapping" in loaded.status_text()
    assert loaded.last_error
    # the mapping on screen is untouched
    assert loaded.mapping_row_count() == 3


def test_saving_to_an_unwritable_path_is_reported_inline(loaded, tmp_path):
    unwritable = tmp_path / "map.csv" / "nested.csv"
    (tmp_path / "map.csv").write_text("", encoding="utf-8")
    assert loaded.save_mapping(str(unwritable)) is False
    assert "Could not save the mapping" in loaded.status_text()


def test_a_mapping_can_be_loaded_before_a_preview(screen, tmp_path):
    path = str(tmp_path / "m.csv")
    fgn.save_column_map([fgn.ColumnMap(source="A", target="foreign_a")], path)
    assert screen.load_mapping(path) is True
    assert screen.mapping_row_count() == 1
    assert "Press Preview" in screen.status_text()
    assert not screen.can_import()


# ---------------------------------------------------------------------------
# Importing
# ---------------------------------------------------------------------------

def test_import_refuses_before_a_preview(screen, tmp_path):
    assert screen.run_import() is False
    assert "Press Preview first" in screen.status_text()


def test_import_refuses_a_blocking_plan_and_a_missing_destination(loaded):
    loaded.set_destination("")
    assert loaded.run_import() is False
    assert "destination folder" in loaded.status_text()

    row = loaded._model.row_of("AreaShape_Area_um2")
    loaded.set_mapping_value(row, "target", "cell_area")
    assert loaded.run_import() is False
    assert "blocking problems" in loaded.status_text()


def test_a_full_import_from_the_screen(loaded, tmp_path):
    dst = str(tmp_path / "imported")
    loaded.set_destination(dst)
    assert loaded.run_import() is True

    result = loaded.result()
    assert result is not None and result.is_complete
    assert result.n_fields == 2
    assert os.path.isfile(result.db_path)
    assert "Imported 2 field(s)" in loaded.status_text()
    assert "column(s) in the object tables are THEIRS" in loaded.report_text()
    # the map that ran is written beside the project
    assert os.path.isfile(os.path.join(dst, fgn.COLUMN_MAP_FILENAME))


def test_the_import_applies_the_edited_mapping_not_the_inferred_one(
        loaded, tmp_path):
    import sqlite3

    row = loaded._model.row_of("AreaShape_Area_um2")
    loaded.set_mapping_value(row, "target", "foreign_reviewed_area")
    loaded.set_destination(str(tmp_path / "imported"))
    assert loaded.run_import() is True

    connection = sqlite3.connect(loaded.result().db_path)
    try:
        cells = pd.read_sql_query("SELECT * FROM cell", connection)
    finally:
        connection.close()
    assert "foreign_reviewed_area" in cells.columns
    assert "foreign_areashape_area_um2" not in cells.columns
    # 4 µm² at 0.5 µm/px is 16 px²
    assert sorted(cells["foreign_reviewed_area"].unique()) == [16.0, 36.0]


def test_a_failing_job_lands_in_the_status_label_not_a_dialog(
        screen, theirs, monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("their masks are on a dead NFS mount")

    monkeypatch.setattr(fgn, "plan_import", _boom)
    screen.set_images(theirs["images"])
    screen.add_mask_folder("cell", theirs["masks"]["cell"])
    screen.set_measurements(theirs["table"])
    assert screen.preview() is False
    assert "dead NFS mount" in screen.status_text()
    assert screen.last_error
    assert not screen.is_busy()


def test_a_none_result_is_reported_rather_than_crashing(screen):
    screen._on_plan_ready(None)
    assert "no plan" in screen.status_text()
    screen._on_result_ready(None)
    assert "no result" in screen.status_text()


def test_an_incomplete_import_says_so(loaded, tmp_path, monkeypatch):
    real = fgn.run_import

    def _partial(plan, dst, **kwargs):
        result = real(plan, dst, **kwargs)
        result.ledger.record_failure("a field", stage="stack", exc="disk full")
        return result

    monkeypatch.setattr(fgn, "run_import", _partial)
    loaded.set_destination(str(tmp_path / "imported"))
    assert loaded.run_import() is True
    assert not loaded.result().is_complete
    assert "INCOMPLETE" in loaded.status_text()
    assert loaded.last_error


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

def test_the_import_runs_off_the_gui_thread(threaded_screen, theirs, tmp_path,
                                            qtbot):
    dst = str(tmp_path / "imported")
    threaded_screen.set_images(theirs["images"])
    threaded_screen.add_mask_folder("cell", theirs["masks"]["cell"])
    threaded_screen.set_measurements(theirs["table"])
    threaded_screen.set_pixel_size(0.5)
    threaded_screen.set_destination(dst)

    with qtbot.waitSignal(threaded_screen.job_finished,
                          timeout=60000) as blocker:
        assert threaded_screen.preview() is True
        assert threaded_screen.is_busy()
    assert blocker.args == [True]
    assert threaded_screen.mapping_row_count() == 3
    assert not threaded_screen.is_busy()

    with qtbot.waitSignal(threaded_screen.job_finished, timeout=120000):
        assert threaded_screen.run_import() is True
    assert threaded_screen.result().n_fields == 2
    assert os.path.isfile(threaded_screen.result().db_path)


def test_completion_is_delivered_through_a_bound_method(threaded_screen):
    """A closure on ``finished`` would run the handler in the worker
    thread; ``_job_settled`` is what keeps it on the GUI thread."""
    assert threaded_screen._on_job_settled.__self__ is threaded_screen
    assert hasattr(threaded_screen, "_job_settled")


def test_controls_are_disabled_while_a_job_is_in_flight(
        threaded_screen, theirs, qtbot):
    threaded_screen.set_images(theirs["images"])
    threaded_screen.add_mask_folder("cell", theirs["masks"]["cell"])
    threaded_screen.set_measurements(theirs["table"])
    with qtbot.waitSignal(threaded_screen.job_finished, timeout=60000):
        threaded_screen.preview()
        assert not threaded_screen._btn_preview.isEnabled()
        assert not threaded_screen._images_edit.isEnabled()
        assert not threaded_screen._table.isEnabled()
    assert threaded_screen._btn_preview.isEnabled()


def test_worker_threads_are_retired_after_they_finish(
        threaded_screen, theirs, qtbot):
    threaded_screen.set_images(theirs["images"])
    threaded_screen.add_mask_folder("cell", theirs["masks"]["cell"])
    threaded_screen.set_measurements(theirs["table"])
    with qtbot.waitSignal(threaded_screen.job_finished, timeout=60000):
        threaded_screen.preview()
    qtbot.waitUntil(lambda: threaded_screen.active_jobs() == 0, timeout=60000)
    assert threaded_screen._thread is None
    assert threaded_screen._worker is None


def test_the_worker_body_stashes_its_result_for_the_gui_thread(screen):
    """``_capture`` is a named method precisely so it can be called here."""
    payload = {}
    ForeignScreen._capture(lambda: "done", payload)
    assert payload == {"result": "done"}


def test_a_worker_traceback_is_reported_inline(screen):
    screen._on_worker_error_text("Traceback…\nValueError: no such column\n")
    assert "ValueError: no such column" in screen.status_text()
    assert not screen.is_busy()
    screen._on_worker_error_text("")
    assert "unknown error" in screen.status_text()


def test_a_settled_job_with_nothing_pending_does_not_explode(screen):
    screen._on_job_settled(True)
    assert not screen.is_busy()
    screen._on_job_settled(False)
    assert not screen.is_busy()


# ---------------------------------------------------------------------------
# The model on its own
# ---------------------------------------------------------------------------

def test_the_model_is_empty_and_addressable_when_it_has_no_maps(qtbot):
    model = ColumnMapModel()
    assert model.rowCount() == 0
    assert model.columnCount() == len(MAP_COLUMNS)
    assert model.map_at(0) is None
    assert model.row_of("anything") == -1
    assert model.data(model.index(0, 0)) is None
    assert model.setData(model.index(0, 0), "x") is False
    model.set_status(None)                    # no rows: emits nothing


def test_the_model_renders_and_edits_one_mapping(qtbot):
    model = ColumnMapModel()
    model.set_maps([fgn.ColumnMap(source="Area", target="foreign_area",
                                  transform="area", unit_in="um^2",
                                  unit_out="px^2", note="checked")])
    assert model.rowCount() == 1
    assert model.headerData(0, Qt.Horizontal) == "Their column"
    assert model.headerData(0, Qt.Vertical) == "1"
    assert model.headerData(0, Qt.Horizontal, Qt.ToolTipRole) is None
    assert model.data(model.index(0, 0)) == "Area"
    assert model.data(model.index(0, 1)) == "foreign_area"
    assert model.data(model.index(0, 2), Qt.EditRole) == "area"
    assert model.data(model.index(0, 0), Qt.SizeHintRole) is None

    # 'source' is not editable, the rest are
    assert not (model.flags(model.index(0, 0)) & Qt.ItemIsEditable)
    assert model.flags(model.index(0, 1)) & Qt.ItemIsEditable
    assert model.setData(model.index(0, 0), "Other", Qt.EditRole) is False
    assert model.setData(model.index(0, 1), "foreign_x", Qt.EditRole) is True
    assert model.maps()[0].target == "foreign_x"
    assert model.setData(model.index(0, 1), "foreign_x",
                         Qt.DisplayRole) is False
    assert model.row_of("Area") == 0


def test_a_status_refresh_is_not_an_edit(qtbot):
    """Otherwise one keystroke recurses: edit → resolve → status →
    dataChanged → edit → …  ``mapping_edited`` exists to break that."""
    model = ColumnMapModel()
    model.set_maps([fgn.ColumnMap(source="Area", target="foreign_area")])
    edits = []
    model.mapping_edited.connect(lambda: edits.append(1))

    model.set_status({"Area": "mapped"})
    assert edits == []                      # a tooltip refresh is not an edit
    model.set_maps([fgn.ColumnMap(source="B", target="foreign_b")])
    assert edits == []                      # nor is a wholesale replacement
    model.setData(model.index(0, 1), "foreign_c", Qt.EditRole)
    assert edits == [1]


def test_the_model_shows_the_resolution_status_as_a_tooltip(qtbot):
    model = ColumnMapModel()
    model.set_maps([fgn.ColumnMap(source="Area", target="foreign_area")])
    model.set_status({"Area": "uncalibrated: no pixel size was given"})
    tip = model.data(model.index(0, 0), Qt.ToolTipRole)
    assert "Area  ->  foreign_area" in tip
    assert "no pixel size" in tip


# ---------------------------------------------------------------------------
# The pickers
# ---------------------------------------------------------------------------

def _stub_dialogs(monkeypatch, folder="", open_file="", save_file=""):
    """Replace the file dialogs with fixed answers.

    Overrides the autouse guard on purpose: these tests exercise what the
    *buttons* do, and the point of the guard is that nothing else opens a
    dialog behind the user's back.
    """
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: folder))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (open_file, "")))
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (save_file, "")))


def test_the_pickers_apply_what_the_user_chose(loaded, theirs, tmp_path,
                                               monkeypatch):
    saved = str(tmp_path / "picked_map.csv")
    _stub_dialogs(monkeypatch, folder="/chosen/folder",
                  open_file=theirs["table"], save_file=saved)

    # Save first: the input pickers below invalidate the plan on screen.
    loaded._pick_save_mapping()
    assert os.path.isfile(saved)

    loaded._pick_mask()
    assert loaded._mask_edit.text() == "/chosen/folder"
    loaded._pick_destination()
    assert loaded.destination_path() == "/chosen/folder"
    loaded._pick_images()
    assert loaded.images_path() == "/chosen/folder"
    loaded._pick_table()
    assert loaded.measurements_path() == theirs["table"]
    assert loaded.plan() is None                 # the inputs moved

    _stub_dialogs(monkeypatch, open_file=saved)
    loaded._pick_load_mapping()
    assert loaded.mapping_row_count() == 3
    assert "Press Preview" in loaded.status_text()


def test_a_cancelled_picker_changes_nothing(loaded, monkeypatch, tmp_path):
    _stub_dialogs(monkeypatch)           # every dialog returns ""
    before = (loaded.images_path(), loaded.measurements_path(),
              loaded.destination_path(), loaded._mask_edit.text(),
              loaded.mapping_row_count())
    loaded._pick_images()
    loaded._pick_mask()
    loaded._pick_table()
    loaded._pick_destination()
    loaded._pick_save_mapping()
    loaded._pick_load_mapping()
    assert (loaded.images_path(), loaded.measurements_path(),
            loaded.destination_path(), loaded._mask_edit.text(),
            loaded.mapping_row_count()) == before


# ---------------------------------------------------------------------------
# The mask list widget
# ---------------------------------------------------------------------------

def test_the_add_button_uses_the_combo_and_the_line_edit(screen, tmp_path):
    screen._object_box.setCurrentIndex(screen._object_box.findData("nucleus"))
    screen._mask_edit.setText(str(tmp_path))
    screen._btn_add_mask.click()
    assert screen.mask_folders() == {"nucleus": str(tmp_path)}


def test_removing_needs_a_selection_and_says_so(screen, tmp_path):
    screen._btn_remove_mask.click()
    assert "Select a mask folder" in screen.status_text()

    screen.add_mask_folder("cell", str(tmp_path))
    screen._mask_list.setCurrentRow(0)
    screen._btn_remove_mask.click()
    assert screen.mask_folders() == {}


def test_an_unknown_conflict_policy_is_a_programming_error(screen):
    with pytest.raises(ValueError, match="Unknown on_conflict"):
        screen.set_on_conflict("shrug")


# ---------------------------------------------------------------------------
# Empty-state and error plumbing
# ---------------------------------------------------------------------------

def test_the_report_is_empty_with_no_plan(screen):
    screen._refresh_report()
    assert screen.report_text() == ""
    assert screen.unmapped_columns() == []
    assert screen.conflict_lines() == []
    assert not screen.can_import()


def test_editing_before_a_preview_changes_nothing_but_the_model(screen,
                                                                tmp_path):
    path = str(tmp_path / "m.csv")
    fgn.save_column_map([fgn.ColumnMap(source="A", target="foreign_a")], path)
    screen.load_mapping(path)
    assert screen.set_mapping_value(0, "target", "foreign_b") is True
    assert screen.column_maps()[0].target == "foreign_b"
    assert screen.plan() is None            # no plan to re-resolve against


def test_a_completion_handler_that_raises_is_reported_inline(screen):
    def _boom(_result):
        raise RuntimeError("the handler fell over")

    screen._pending.append(({"result": None}, _boom))
    settled = []
    screen.job_finished.connect(settled.append)
    screen._on_job_settled(True)
    assert settled == [False]
    assert "the handler fell over" in screen.status_text()


def test_the_model_flags_an_invalid_index_as_merely_enabled(qtbot):
    from PySide6.QtCore import QModelIndex

    model = ColumnMapModel()
    flags = model.flags(QModelIndex())
    assert flags & Qt.ItemIsEnabled
    assert not (flags & Qt.ItemIsEditable)
