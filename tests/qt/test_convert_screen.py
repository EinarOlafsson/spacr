"""
Format Converter — the Tools screen in front of :mod:`spacr.convert`.

Everything runs offscreen against real temporary TIFF trees, because the
one thing this screen must get right is the table it shows *before*
anything is written. A preview that disagrees with what the conversion
would do is worse than no preview at all.

The properties pinned here:

* it **builds offscreen** and previews a temp tree into the exact
  ``plate1_A01_T0001F001L01A01Z01C01.tif`` naming;
* the preview **writes nothing** — Convert is a separate press, and it
  stays disabled until there is a plan that can actually run;
* a **bad source is reported inline** — no modal dialog anywhere (the
  autouse fixture fires if one opens, because a QMessageBox would hang a
  headless run forever);
* a **collision leaves Convert disabled** and puts both filenames on
  screen;
* the **conversion runs off the GUI thread** through the same
  ``finished`` → bound-method relay as the Plate Viewer, and the summary
  lands on the GUI thread afterwards.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import Qt

from spacr import convert as cvt
from spacr.qt.screens.convert import (
    LAYOUT_CHOICES,
    PLATE_NAME_CHOICES,
    PlanTableModel,
    Z_CHOICES,
    ConvertScreen,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


def _write(path, value=1, shape=(8, 8), **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.full(shape, value, np.uint16), **kwargs)
    return path


@pytest.fixture
def run1(tmp_path):
    """``run1/wt/`` with four field-sets of two channels."""
    root = tmp_path / "src"
    for field in range(1, 5):
        for channel in (1, 2):
            _write(str(root / "run1" / "wt" / f"fov{field:02d}_C{channel}.tif"),
                   value=field * 10 + channel)
    return str(root)


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — jobs run inline so assertions are exact."""
    widget = ConvertScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def threaded_screen(qtbot, qt_theme_applied):
    """The real thing: jobs go through a QThread."""
    widget = ConvertScreen(threaded=True)
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
    assert screen.preview_row_count() == 0
    assert screen.plan() is None
    assert screen.result() is None
    assert not screen.can_convert()
    assert "Preview" in screen.status_text()
    assert screen.last_error == ""
    assert not screen.resume_enabled()
    assert "field" in screen._resume.toolTip().lower()


def test_the_option_choices_are_the_ones_the_converter_understands(screen):
    # Every choice on screen is one the converter accepts, and every one
    # it accepts is on screen — a combo entry the backend rejects would
    # only be discovered by a user picking it.
    assert {value for _label, value in LAYOUT_CHOICES} == set(cvt.LAYOUTS)
    assert {value for _label, value in Z_CHOICES} == set(cvt.Z_HANDLING)
    assert screen.layout_mode() == "auto"
    # The default must be the lossless one.
    assert screen.z_handling() == cvt.Z_KEEP
    assert screen.plate_naming() == PLATE_NAME_CHOICES[0][1] == "index"


def test_the_lossy_z_choices_say_so_in_their_labels():
    labels = dict((value, label) for label, value in Z_CHOICES)
    assert "discarded" in labels[cvt.Z_MAX]
    assert "discarded" in labels[cvt.Z_FIRST]
    assert "Keep every plane" in labels[cvt.Z_KEEP]


def test_an_unknown_option_value_is_refused(screen):
    with pytest.raises(ValueError):
        screen.set_layout_mode("sideways")
    with pytest.raises(ValueError):
        screen.set_z_handling("average")


def test_it_is_registered_under_data_as_alpha():
    """Registration lives in spacr.qt.app.APPS.

    It was briefly filed under "Alpha modules", a section #16i added and
    #16j removed: "where is the format converter" is a question about
    the subject, not about how finished it is. Alpha is now a stage the
    tile draws as a hover colour."""
    from spacr.qt.app import APPS
    entry = next((a for a in APPS if a[0] == "convert"), None)
    if entry is None:
        pytest.skip("convert not registered in spacr.qt.app.APPS yet")
    assert entry[1] == "Format Converter"
    from spacr.qt.app import SECTION_DATA, app_stage
    assert entry[3] == SECTION_DATA
    # `spacr.qt.maturity` reassessed every alpha module against the
    # evidence in the repository and this one no longer qualifies; the
    # reason is recorded beside the decision. Applied here because the
    # promotions land in `register_self_registering_modules`, which every
    # launch calls but a bare test process may not have. `apply` alone,
    # not the whole registration pass: it touches only APP_STAGE, so it
    # cannot re-register a module a test has deliberately removed.
    from spacr.qt import maturity
    maturity.apply()
    assert app_stage(entry[0]) == "stable"
    assert entry[2].strip()


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def test_previewing_a_temp_tree_produces_the_specified_naming(screen, run1):
    screen.set_source(run1)
    assert screen.preview() is True
    assert screen.preview_row_count() == 8
    targets = screen.preview_targets()
    assert targets[0] == "plate1_A01_T0001F001L01A01Z01C01.tif"
    assert targets[1] == "plate1_A01_T0001F001L01A01Z01C02.tif"
    assert screen.preview_value(0, "plate") == "plate1"
    assert screen.preview_value(0, "well") == "A01"
    assert screen.preview_value(0, "source_well") == "wt"
    assert screen.preview_value(0, "source_field") == "fov01"
    assert "Nothing has been written" in screen.status_text()
    assert screen.last_error == ""


def test_the_preview_writes_nothing(screen, run1, tmp_path):
    before = sorted(os.walk(run1))
    screen.set_source(run1)
    screen.set_destination(str(tmp_path / "out"))
    screen.preview()
    assert sorted(os.walk(run1)) == before
    assert not (tmp_path / "out").exists()


def test_the_preview_table_shows_source_and_target_side_by_side(screen, run1):
    screen.set_source(run1)
    screen.preview()
    model = screen._model
    headers = [model.headerData(c, Qt.Horizontal)
               for c in range(model.columnCount())]
    assert headers[:2] == ["Source", "Target"]
    # The source cell shows the basename; the tooltip has the full path.
    index = model.index(0, 0)
    assert model.data(index, Qt.DisplayRole) == "fov01_C1.tif"
    assert run1 in model.data(index, Qt.ToolTipRole)


def test_changing_an_option_invalidates_the_preview(screen, run1):
    screen.set_source(run1)
    screen.preview()
    assert screen.can_convert()
    screen.set_z_handling(cvt.Z_MAX)
    assert screen.plan() is None
    assert screen.preview_row_count() == 0
    assert not screen.can_convert()
    assert "Preview again" in screen.status_text()


def test_max_projection_changes_the_plan_and_says_so(screen, tmp_path):
    root = tmp_path / "src"
    os.makedirs(root)
    tifffile.imwrite(str(root / "fov01.tif"),
                     np.zeros((5, 8, 8), np.uint16), metadata={"axes": "ZYX"})
    screen.set_source(str(root))
    screen.preview()
    assert screen.preview_row_count() == 5

    screen.set_z_handling(cvt.Z_MAX)
    screen.preview()
    assert screen.preview_row_count() == 1
    assert "max-projects" in screen.summary_text()
    assert "will NOT be written" in screen.summary_text()


def test_a_missing_source_is_reported_inline(screen):
    assert screen.preview() is False
    assert screen.last_error
    assert "Choose a source folder" in screen.status_text()


def test_a_bad_source_path_is_reported_inline(screen, tmp_path):
    screen.set_source(str(tmp_path / "does_not_exist"))
    assert screen.preview() is False
    assert "Not a folder" in screen.status_text()
    assert screen.last_error
    # The screen is still usable afterwards.
    assert not screen.can_convert()


def test_a_source_folder_with_no_images_is_reported_inline(screen, tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "readme.txt").write_text("no images here")
    screen.set_source(str(root))
    assert screen.preview() is True
    assert screen.preview_row_count() == 0
    assert "No readable images" in screen.status_text()
    assert not screen.can_convert()


def test_an_unreadable_file_shows_up_in_the_preview_as_a_skip(screen, tmp_path):
    root = tmp_path / "src"
    _write(str(root / "good.tif"))
    (root / "corrupt.tif").write_bytes(b"definitely not a TIFF")
    screen.set_source(str(root))
    screen.preview()
    assert screen.preview_row_count() == 2
    statuses = [screen.preview_value(r, "status") for r in range(2)]
    assert any(s.startswith("SKIP") for s in statuses)
    assert "cannot be read" in screen.status_text()
    # Still convertible: one bad file does not block the other seven.
    assert screen.can_convert()


def test_a_collision_disables_convert_and_names_both_files(screen, tmp_path):
    root = tmp_path / "src"
    _write(str(root / "fov01_C1.tif"))
    _write(str(root / "fov01_C1.tiff"))
    screen.set_source(str(root))
    screen.set_destination(str(tmp_path / "out"))
    screen.preview()

    assert screen.plan() is not None and not screen.plan().ok
    assert not screen.can_convert()
    assert screen.last_error
    assert "blocking problem" in screen.status_text()
    summary = screen.summary_text()
    assert "fov01_C1.tif" in summary
    assert "fov01_C1.tiff" in summary

    # Pressing Convert anyway refuses inline and writes nothing.
    assert screen.run_convert() is False
    assert not (tmp_path / "out").exists()


def test_a_scan_that_raises_lands_in_the_status_label(screen, run1, monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("the disk fell over")

    monkeypatch.setattr(cvt, "scan", _boom)
    screen.set_source(run1)
    assert screen.preview() is False
    assert "the disk fell over" in screen.status_text()
    assert screen.last_error


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------

def test_convert_is_refused_before_a_preview(screen, run1, tmp_path):
    screen.set_source(run1)
    screen.set_destination(str(tmp_path / "out"))
    assert screen.run_convert() is False
    assert "Press Preview first" in screen.status_text()
    assert not (tmp_path / "out").exists()


def test_convert_is_refused_without_a_destination(screen, run1):
    screen.set_source(run1)
    screen.set_destination("")
    screen.preview()
    assert screen.run_convert() is False
    assert "destination" in screen.status_text()


def test_a_destination_is_suggested_from_the_source(screen, run1):
    screen.set_source(run1)
    assert screen.destination_path().endswith("_yokogawa")
    assert screen.destination_path() != run1


def test_converting_writes_the_files_the_preview_promised(screen, run1, tmp_path):
    dst = str(tmp_path / "out")
    screen.set_source(run1)
    screen.set_destination(dst)
    screen.preview()
    promised = screen.preview_targets()

    assert screen.run_convert() is True
    result = screen.result()
    assert result is not None
    assert result.n_written == 8
    assert sorted(f for f in os.listdir(dst) if f.endswith(".tif")) == \
        sorted(promised)
    assert "Converted 8 file(s)" in screen.status_text()
    assert "conversion_map.csv" in screen.status_text()


def test_resume_switch_reuses_the_converter_field_checkpoint(
        screen, run1, tmp_path):
    dst = str(tmp_path / "out")
    screen.set_source(run1)
    screen.set_destination(dst)
    screen.preview()
    screen.run_convert()

    screen.set_resume(True)
    screen.preview()
    assert screen.run_convert()

    assert screen.resume_enabled()
    assert screen.result().n_written == 0
    assert len(screen.result().resumed_fields) == 4
    assert "Resumed 4 completed field" in screen.summary_text()


def test_the_summary_names_what_was_skipped_and_where_the_map_went(
        screen, tmp_path):
    root = tmp_path / "src"
    _write(str(root / "good.tif"))
    (root / "corrupt.tif").write_bytes(b"not a TIFF")
    dst = str(tmp_path / "out")
    screen.set_source(str(root))
    screen.set_destination(dst)
    screen.preview()
    screen.run_convert()

    summary = screen.summary_text()
    assert "Skipped 1 source(s)" in summary
    assert "corrupt.tif" in summary
    assert "conversion_map.csv" in summary
    # The partial run is reported as an error, not as a clean finish.
    assert screen.last_error
    assert "skipped 1" in screen.status_text()
    assert os.path.isfile(screen.result().map_path)


def test_the_map_written_by_the_screen_round_trips(screen, run1, tmp_path):
    screen.set_source(run1)
    screen.set_destination(str(tmp_path / "out"))
    screen.preview()
    screen.run_convert()
    frame = cvt.read_map(screen.result().map_path)
    assert len(frame) == 8
    assert frame["target"].is_unique
    assert frame.groupby("target")["source"].nunique().max() == 1
    assert set(frame["source_well"]) == {"wt"}


def test_a_convert_that_raises_lands_in_the_status_label(
        screen, run1, tmp_path, monkeypatch):
    screen.set_source(run1)
    screen.set_destination(str(tmp_path / "out"))
    screen.preview()

    def _boom(*_a, **_k):
        raise OSError("the destination is read-only")

    monkeypatch.setattr(cvt, "convert", _boom)
    assert screen.run_convert() is False
    assert "read-only" in screen.status_text()
    assert screen.last_error
    assert not screen._progress_bar.isVisible()


def test_progress_updates_the_bar_and_the_status(screen, run1, tmp_path):
    screen.set_source(run1)
    screen.set_destination(str(tmp_path / "out"))
    screen.preview()
    screen._on_progress(3, 8, "fov02_C1.tif")
    assert screen._progress_bar.value() == 3
    assert screen._progress_bar.maximum() == 8
    assert "3/8" in screen.status_text()
    assert "fov02_C1.tif" in screen.status_text()


def test_rerunning_from_the_screen_does_not_overwrite(screen, run1, tmp_path):
    dst = str(tmp_path / "out")
    screen.set_source(run1)
    screen.set_destination(dst)
    screen.preview()
    screen.run_convert()
    assert screen.result().n_written == 8

    screen.preview()
    screen.run_convert()
    assert screen.result().n_written == 0
    assert len(screen.result().existing) == 8


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

def test_the_conversion_runs_off_the_gui_thread(threaded_screen, run1, tmp_path,
                                                qtbot):
    dst = str(tmp_path / "out")
    threaded_screen.set_source(run1)
    threaded_screen.set_destination(dst)

    with qtbot.waitSignal(threaded_screen.job_finished, timeout=15000) as blocker:
        assert threaded_screen.preview() is True
        assert threaded_screen.is_busy()
    assert blocker.args == [True]
    assert threaded_screen.preview_row_count() == 8
    assert not threaded_screen.is_busy()

    with qtbot.waitSignal(threaded_screen.job_finished, timeout=30000):
        assert threaded_screen.run_convert() is True
    assert threaded_screen.result().n_written == 8
    assert sorted(f for f in os.listdir(dst) if f.endswith(".tif")) == \
        sorted(threaded_screen.preview_targets())


def test_completion_is_delivered_through_a_bound_method(threaded_screen):
    """A closure on ``finished`` would run the handler in the worker
    thread; ``_job_settled`` is what keeps it on the GUI thread."""
    assert threaded_screen._on_job_settled.__self__ is threaded_screen
    assert hasattr(threaded_screen, "_job_settled")


def test_controls_are_disabled_while_a_job_is_in_flight(
        threaded_screen, run1, qtbot):
    threaded_screen.set_source(run1)
    with qtbot.waitSignal(threaded_screen.job_finished, timeout=15000):
        threaded_screen.preview()
        assert not threaded_screen._btn_preview.isEnabled()
        assert not threaded_screen._src_edit.isEnabled()
    assert threaded_screen._btn_preview.isEnabled()
    assert threaded_screen.can_convert()


def test_worker_threads_are_retired_after_they_finish(
        threaded_screen, run1, qtbot):
    threaded_screen.set_source(run1)
    with qtbot.waitSignal(threaded_screen.job_finished, timeout=15000):
        threaded_screen.preview()
    qtbot.waitUntil(lambda: threaded_screen.active_jobs() == 0, timeout=15000)
    assert threaded_screen._thread is None
    assert threaded_screen._worker is None


def test_a_worker_error_string_lands_inline(screen):
    screen._on_worker_error_text("Traceback…\nValueError: nope")
    assert "Conversion failed" in screen.status_text()
    assert "ValueError: nope" in screen.status_text()
    assert screen.last_error
    screen._on_worker_error_text("")
    assert "unknown error" in screen.status_text()


# ---------------------------------------------------------------------------
# The table model
# ---------------------------------------------------------------------------

def test_the_model_is_empty_and_safe_before_any_plan(qtbot):
    model = PlanTableModel()
    assert model.rowCount() == 0
    assert model.columnCount() == len(model._columns)
    assert model.data(model.index(0, 0)) is None
    model.set_frame(None)
    assert model.rowCount() == 0


def test_preview_helpers_are_safe_off_the_end_of_the_table(screen, run1):
    screen.set_source(run1)
    screen.preview()
    assert screen.preview_value(-1, "target") == ""
    assert screen.preview_value(999, "target") == ""
    assert screen.preview_value(0, "no_such_column") == ""
    # A frame with no target column at all.
    import pandas as pd
    screen._model.set_frame(pd.DataFrame({"plate": ["plate1"]}))
    assert screen.preview_targets() == []


def test_plate_naming_is_settable(screen, run1):
    screen.set_plate_naming("name")
    assert screen.plate_naming() == "name"
    screen.set_source(run1)
    screen.preview()
    assert screen.preview_value(0, "plate") == "run1"


def test_an_empty_plan_refuses_to_convert(screen, tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    screen.set_source(str(root))
    screen.set_destination(str(tmp_path / "out"))
    screen.preview()
    assert screen.run_convert() is False
    assert "empty" in screen.status_text()
    assert not (tmp_path / "out").exists()


def test_a_scan_that_yields_no_plan_is_reported_inline(screen):
    screen._on_plan_ready(None)
    assert "no plan" in screen.status_text()
    assert screen.last_error
    assert screen.preview_row_count() == 0


def test_a_conversion_that_yields_no_result_is_reported_inline(screen):
    screen._on_result_ready(None)
    assert "no result" in screen.status_text()
    assert screen.last_error
    assert not screen._progress_bar.isVisible()


def test_the_pickers_feed_the_line_edits(screen, run1, tmp_path, monkeypatch):
    """The one place a dialog is allowed — and it is not modal-blocking."""
    from PySide6.QtWidgets import QFileDialog

    chosen = {"value": run1}
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: chosen["value"]))
    screen._pick_source()
    assert screen.source_path() == run1

    chosen["value"] = str(tmp_path / "picked")
    screen._pick_destination()
    assert screen.destination_path() == str(tmp_path / "picked")

    # Cancelling changes nothing.
    chosen["value"] = ""
    screen._pick_source()
    screen._pick_destination()
    assert screen.source_path() == run1
    assert screen.destination_path() == str(tmp_path / "picked")


def test_the_worker_body_stashes_its_result_for_the_gui_thread(screen):
    """``_capture`` is what runs on the QThread; the GUI thread reads
    ``box['result']`` afterwards in ``_on_job_settled``."""
    box = {}
    screen._capture(lambda: 42, box)
    assert box == {"result": 42}


def test_a_completion_handler_that_raises_lands_inline(screen):
    def _boom(_result):
        raise RuntimeError("the handler blew up")

    screen._pending.append(({"result": None}, _boom))
    screen._on_job_settled(True)
    assert "the handler blew up" in screen.status_text()
    assert screen.last_error
    assert not screen.is_busy()


def test_settling_with_nothing_pending_is_harmless(screen):
    screen._on_job_settled(True)
    assert not screen.is_busy()
    assert screen.last_error == ""


def test_the_model_ignores_columns_it_does_not_display(run1):
    plan = cvt.plan(cvt.scan(run1))
    frame = plan.to_frame()
    frame["something_else"] = 1
    model = PlanTableModel()
    model.set_frame(frame)
    headers = [model.headerData(c, Qt.Horizontal)
               for c in range(model.columnCount())]
    assert "something_else" not in headers
    assert model.rowCount() == len(frame)
    assert model.headerData(0, Qt.Vertical) == "1"
    assert model.headerData(0, Qt.Horizontal, Qt.FontRole) is None
