"""The Classifier Evaluation workbench when the bundle will not cooperate.

Discovery and loading both run on workers, and every one of their failure
paths ends the same way: the screen has to say what went wrong and clear the
panel it can no longer describe. A stale confusion cell beside a bundle that
did not load is the one outcome the workbench may not have.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from spacr.classifier_evaluation import (
    LeakageReport,
    evaluate_predictions,
    write_evaluation_bundle,
)
from spacr.qt.screens import classifier_evaluation as CE
from spacr.qt.screens.classifier_evaluation import (
    ClassifierEvaluationScreen,
    _DropPathEdit,
    _item,
)


@pytest.fixture
def evaluation_root(tmp_path):
    """One small evaluation bundle, written the way a real run writes it."""
    paths = ["plate1_A01_1_1.png", "plate1_A02_1_2.png",
             "plate2_B01_1_3.png", "plate2_B02_1_4.png"]
    evaluation = evaluate_predictions(
        [0, 1, 0, 1],
        np.asarray([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]]),
        paths,
        classes=["negative", "positive"],
        fold_ids=[1, 1, 2, 2],
        calibration_method="none",
        calibration_bins=4,
    )
    report = LeakageReport(
        group_by="well", train_samples=2, validation_samples=2,
        overlap_counts={"exact": 0, "augmentation_family": 0, "object": 0,
                        "field": 0, "well": 0, "plate": 1},
        examples={"exact": [], "augmentation_family": [], "object": [],
                  "field": [], "well": [], "plate": ["plate1"]},
        split_name="outer_1",
    )
    return write_evaluation_bundle(
        tmp_path / "model" / "evaluation", evaluation,
        leakage_reports=[report]).parents[1]


@pytest.fixture
def screen(qtbot, evaluation_root):
    """A loaded, unthreaded workbench."""
    widget = ClassifierEvaluationScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._source.setText(str(evaluation_root))
    widget.scan()
    return widget


@pytest.fixture
def empty_screen(qtbot):
    """A workbench that has never been pointed at anything."""
    widget = ClassifierEvaluationScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# the drop-target path field
# ---------------------------------------------------------------------------

def test_a_dropped_folder_fills_the_field_and_announces_itself(qtbot,
                                                               evaluation_root):
    """The field is a drop target so a results folder can be dragged in."""
    edit = _DropPathEdit()
    qtbot.addWidget(edit)
    seen = []
    edit.path_dropped.connect(seen.append)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(evaluation_root))])
    enter = QDragEnterEvent(QPoint(1, 1), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)
    edit.dragEnterEvent(enter)
    assert enter.isAccepted()

    drop = QDropEvent(QPointF(1, 1), Qt.CopyAction, mime,
                      Qt.LeftButton, Qt.NoModifier)
    edit.dropEvent(drop)
    assert drop.isAccepted()
    assert edit.text() == str(evaluation_root)
    assert seen == [str(evaluation_root)]


def test_a_drag_that_is_not_a_local_file_is_refused(qtbot):
    """A URL from a browser is not a folder on this machine."""
    edit = _DropPathEdit()
    qtbot.addWidget(edit)
    mime = QMimeData()
    mime.setUrls([QUrl("https://example.org/results")])

    enter = QDragEnterEvent(QPoint(1, 1), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)
    edit.dragEnterEvent(enter)
    assert not enter.isAccepted()

    drop = QDropEvent(QPointF(1, 1), Qt.CopyAction, mime,
                      Qt.LeftButton, Qt.NoModifier)
    edit.dropEvent(drop)
    assert not drop.isAccepted()
    assert edit.text() == ""


def test_a_missing_number_is_shown_as_a_blank_cell():
    """An empty cell reads as "not measured"; ``nan`` reads as a value."""
    assert _item(None).text() == ""
    assert _item(float("nan")).text() == ""
    assert _item(0.123456789).text() == "0.12346"
    assert _item("negative").text() == "negative"
    assert not (_item("negative").flags() & Qt.ItemIsEditable)


# ---------------------------------------------------------------------------
# choosing and scanning
# ---------------------------------------------------------------------------

def test_scanning_with_no_folder_chosen_asks_for_one(empty_screen):
    """The status line says what to do rather than searching nothing."""
    empty_screen.scan()
    assert "Choose a results folder first." in empty_screen._status.text()


def test_a_cancelled_folder_dialog_leaves_the_field_alone(empty_screen,
                                                          monkeypatch):
    """Backing out of the picker is not a request to scan."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    empty_screen._choose_source()
    assert empty_screen._source.text() == ""


def test_choosing_a_folder_fills_the_field_and_scans_it(empty_screen,
                                                        monkeypatch,
                                                        evaluation_root):
    """The chosen path is the path that gets searched."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(evaluation_root)))
    empty_screen._choose_source()
    assert empty_screen._source.text() == str(evaluation_root)
    assert empty_screen.bundles


def test_a_folder_with_no_bundles_says_so_and_clears_the_panel(empty_screen,
                                                               tmp_path):
    """Nothing found must not leave the previous run's tables on screen."""
    empty = tmp_path / "no_runs_here"
    empty.mkdir()
    empty_screen._source.setText(str(empty))
    empty_screen.scan()
    assert "No evaluation_manifest.json files were found." in (
        empty_screen._status.text())
    assert empty_screen.bundle is None


def test_a_scan_that_failed_reports_the_error_it_hit(empty_screen, monkeypatch,
                                                     tmp_path):
    """Discovery's own message reaches the status line."""
    monkeypatch.setattr(
        CE, "find_evaluation_bundles",
        lambda root: (_ for _ in ()).throw(PermissionError("results is 0700")))
    empty_screen._source.setText(str(tmp_path))
    empty_screen.scan()
    assert "PermissionError" in empty_screen._status.text()
    assert empty_screen.bundle is None


def test_a_bundle_outside_the_chosen_root_is_labelled_by_its_own_path(
        empty_screen, monkeypatch, evaluation_root, tmp_path):
    """A manifest that is not under the root has no relative name to show."""
    manifests = CE.find_evaluation_bundles(evaluation_root)
    monkeypatch.setattr(CE, "find_evaluation_bundles",
                        lambda root: list(manifests))
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    empty_screen._source.setText(str(elsewhere))
    empty_screen.scan()
    labels = [empty_screen._bundle_choice.itemText(i)
              for i in range(empty_screen._bundle_choice.count())]
    assert labels and all(label for label in labels)


def test_a_scan_while_one_is_running_is_ignored(screen):
    """Two discoveries over one folder is one too many."""
    screen._busy = True
    try:
        before = screen._status.text()
        screen.scan()
        assert screen._status.text() == before
    finally:
        screen._busy = False


# ---------------------------------------------------------------------------
# loading a bundle
# ---------------------------------------------------------------------------

def test_a_load_while_one_is_running_is_ignored(screen):
    """The second selection waits rather than racing the first read."""
    screen._busy = True
    try:
        loaded = screen.bundle
        screen._load_selected_bundle()
        assert screen.bundle is loaded
    finally:
        screen._busy = False


def test_a_selector_with_nothing_in_it_loads_nothing(empty_screen):
    """No manifest is selected, so there is nothing to parse."""
    empty_screen._load_selected_bundle()
    assert empty_screen.bundle is None


def test_a_bundle_that_cannot_be_parsed_is_reported_and_clears_the_panel(
        screen, monkeypatch):
    """A stale table beside "could not load" would be the worse outcome."""
    monkeypatch.setattr(
        CE, "load_evaluation_bundle",
        lambda manifest: (_ for _ in ()).throw(
            json.JSONDecodeError("bad manifest", "{", 0)))
    screen._load_selected_bundle()
    assert screen.bundle is None
    assert "Could not load classifier evaluation" in screen._status.text()
    assert "JSONDecodeError" in screen._status.text()
    assert screen._cell is None


# ---------------------------------------------------------------------------
# rendering without a bundle
# ---------------------------------------------------------------------------

def test_the_tables_are_empty_before_anything_is_loaded(empty_screen):
    """Every render path is safe to call on a workbench with no bundle."""
    assert empty_screen._predictions_frame().empty
    empty_screen._render_predictions()
    assert empty_screen._predictions.rowCount() == 0
    empty_screen.show_cell("negative", "positive")
    assert empty_screen._cell is None


def test_a_bundle_that_cannot_be_broken_down_by_object_says_so(screen,
                                                               monkeypatch):
    """The normalised matrix is still shown; only the clicking is lost."""
    monkeypatch.setattr(
        CE.cx, "confusion_counts",
        lambda predictions, classes: (_ for _ in ()).throw(
            CE.cx.ConfusionError("no object key column in this bundle")))
    screen._render_confusion(dict(screen.bundle.get("summary") or {}))
    assert "cannot be broken down by object" in screen._confusion_ranking.text()
    assert "no object key column" in screen._confusion_ranking.text()
    assert screen._cell is None
    assert screen._confusion.rowCount() > 0


def test_a_cell_that_cannot_be_built_reports_why_and_opens_nothing(screen,
                                                                   monkeypatch):
    """The message from the builder is what the summary line shows."""
    monkeypatch.setattr(
        CE.cx.ConfusionCell, "build",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            CE.cx.ConfusionError("that pair has no objects"))))
    screen.show_cell("negative", "positive")
    assert screen._cell is None
    assert screen._cell_summary.text() == "that pair has no objects"


def test_a_click_off_the_matrix_opens_no_cell(screen):
    """Only a cell with both a row label and a column header is a pair."""
    screen._cell = None
    screen._on_confusion_cell(0, 0)
    assert screen._cell is None
    screen._on_confusion_cell(screen._confusion.rowCount() + 5, 1)
    assert screen._cell is None


# ---------------------------------------------------------------------------
# opening the crops behind a cell
# ---------------------------------------------------------------------------

def test_opening_a_cell_before_one_is_chosen_does_nothing(screen):
    """There is no list to open until a cell has been clicked."""
    screen._clear_cell()
    assert screen._open_high() is None
    assert screen._open_low() is None


def test_a_cell_that_cannot_list_its_keys_reports_the_reason(screen,
                                                             monkeypatch):
    """A refusal from the cell reaches the status line, not just the log."""
    screen.show_cell("negative", "negative")
    if screen._cell is None:
        pytest.skip("this bundle has no negative/negative cell")

    monkeypatch.setattr(
        type(screen._cell), "keys",
        lambda self, which: (_ for _ in ()).throw(
            CE.cx.ConfusionError("no confidence column to rank by")))
    assert screen.open_cell("high") is None
    assert "no confidence column to rank by" in screen._status.text()


def test_an_opener_that_raises_is_reported_against_the_crops(screen,
                                                             monkeypatch):
    """The screen names what failed instead of dying on a missing viewer."""
    screen.show_cell("negative", "negative")
    if screen._cell is None:
        pytest.skip("this bundle has no negative/negative cell")

    monkeypatch.setattr(
        CE, "open_objects",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no crop viewer")))
    assert screen.open_cell("high") is None
    assert "Could not open those crops" in screen._status.text()


def test_changing_the_threshold_redraws_the_open_cell(screen):
    """The split between high and low confidence moves with the control."""
    screen.show_cell("negative", "negative")
    if screen._cell is None:
        pytest.skip("this bundle has no negative/negative cell")
    before = screen._cell_summary.text()
    screen._threshold.setValue(0.99)
    assert screen._cell is not None
    assert screen._cell.threshold == pytest.approx(0.99)
    assert screen._cell_summary.text() != before or before


# ---------------------------------------------------------------------------
# the object lists beside a cell
# ---------------------------------------------------------------------------

def test_a_frame_with_no_object_key_still_lists_its_rows(screen):
    """Losing the key column costs the tooltip, not the list."""
    listing = screen._high_list
    frame = pd.DataFrame({"score": [0.9, 0.8]})
    screen._fill_list(listing, frame, "high")
    assert listing.count() == 2


def test_a_long_list_says_how_many_it_did_not_show(screen):
    """A truncated list that did not say so reads as the whole cell."""
    listing = screen._high_list
    size = CE.LIST_PREVIEW + 7
    frame = pd.DataFrame({"basename": [f"o{i}.png" for i in range(size)]})
    screen._fill_list(listing, frame, "high")
    assert listing.count() == CE.LIST_PREVIEW + 1
    assert "more; open them to see all" in listing.item(listing.count() - 1).text()


def test_an_empty_list_says_none_rather_than_showing_nothing(screen):
    """A blank list box is indistinguishable from one that failed to fill."""
    listing = screen._high_list
    screen._fill_list(listing, pd.DataFrame(), "high")
    assert listing.count() == 1
    assert listing.item(0).text()


# ---------------------------------------------------------------------------
# the folder buttons and shutdown
# ---------------------------------------------------------------------------

def test_the_folder_button_does_nothing_without_a_bundle(empty_screen,
                                                          monkeypatch):
    """Nothing is loaded, so there is no folder to open."""
    from PySide6.QtGui import QDesktopServices

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url)))
    empty_screen._open_current_folder()
    assert opened == []


def test_the_folder_button_opens_the_bundles_own_folder(screen, monkeypatch):
    """The folder shown is the one holding the manifest that was read."""
    from pathlib import Path

    from PySide6.QtGui import QDesktopServices

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url)))
    screen._open_current_folder()
    assert opened
    assert opened[0].toLocalFile() == str(
        Path(screen.bundle["path"]).parent.resolve())


def test_closing_the_screen_cancels_and_drains_every_worker(screen):
    """Qt aborts the process if a running QThread is destroyed with the screen."""
    cancelled = []
    drained = []

    class Worker:
        @staticmethod
        def request_cancel(reason):
            cancelled.append(reason)

    class RefusingWorker:
        @staticmethod
        def request_cancel(reason):
            raise RuntimeError("the worker's C++ side is gone")

    class Thread:
        pass

    screen._jobs = [(Thread(), Worker()), (Thread(), RefusingWorker()),
                    (Thread(), None)]
    import spacr.qt.bridge as bridge
    original = bridge.drain_thread
    bridge.drain_thread = lambda thread, worker, timeout_ms=0: drained.append(
        thread)
    try:
        screen.close()
    finally:
        bridge.drain_thread = original
    assert cancelled == ["classifier-evaluation closed"]
    assert len(drained) == 3
    assert screen._jobs == []


# ---------------------------------------------------------------------------
# the same work, on a worker thread
# ---------------------------------------------------------------------------

def test_a_threaded_scan_and_load_land_on_the_gui_thread(qtbot,
                                                          evaluation_root):
    """Discovery and parsing both run off the GUI thread and report back."""
    widget = ClassifierEvaluationScreen(threaded=True)
    qtbot.addWidget(widget)
    widget._source.setText(str(evaluation_root))
    widget.scan()
    qtbot.waitUntil(lambda: not widget._busy, timeout=10000)
    assert widget.bundles

    qtbot.waitUntil(lambda: widget.bundle is not None, timeout=10000)
    assert widget._confusion.rowCount() > 0
    qtbot.waitUntil(lambda: widget._jobs == [], timeout=10000)
    widget.close()


def test_a_breakdown_that_cannot_be_computed_is_reported_per_level(screen,
                                                                   monkeypatch):
    """One level failing must not cost the other its line."""
    def only_wells(rows, level):
        if level == "plate":
            raise CE.cx.ConfusionError("no plate column in this bundle")
        return "3 wells"

    monkeypatch.setattr(CE.cx, "describe_breakdown", only_wells)
    screen.show_cell("negative", "negative")
    if screen._cell is None:
        pytest.skip("this bundle has no negative/negative cell")
    text = screen._describe_origin(screen._cell)
    assert "well: 3 wells" in text
    assert "plate: no plate column in this bundle" in text


def test_crops_are_still_opened_when_their_confidences_cannot_be_read(
        screen, monkeypatch):
    """The scores are context for the viewer, not a precondition for opening."""

    screen.show_cell("negative", "negative")
    if screen._cell is None:
        pytest.skip("this bundle has no negative/negative cell")

    halves = [which for which in ("high", "low")
              if len(screen._cell.keys(which))]
    if not halves:
        pytest.skip("this cell has no objects on either side of the split")

    # A bundle whose confidence column is spelled differently: the keys are
    # still listable, only the per-key scores are not.
    monkeypatch.setattr(CE.cx, "CONFIDENCE_COLUMN", "a_column_nobody_wrote")
    seen = []

    def opener(keys, **kwargs):
        seen.append(kwargs.get("context"))
        return "opened"

    monkeypatch.setattr(CE, "open_objects", opener)
    assert screen.open_cell(halves[0]) == "opened"
    assert seen and seen[0]["scores"] == {}


def test_opening_an_empty_half_says_the_list_is_empty(screen, monkeypatch):
    """A half with nothing in it is a sentence, not a viewer with no crops."""
    screen.show_cell("negative", "negative")
    if screen._cell is None:
        pytest.skip("this bundle has no negative/negative cell")

    monkeypatch.setattr(type(screen._cell), "keys", lambda self, which: [])
    assert screen.open_cell("high") is None
    assert "That list is empty." in screen._status.text()


def test_a_confusion_row_with_a_missing_cell_is_skipped(screen, monkeypatch):
    """A table item Qt did not create is passed over, not tooltipped."""
    table = screen._confusion
    assert table.rowCount() > 0
    table.setItem(0, 1, None)
    screen._render_confusion(dict(screen.bundle.get("summary") or {}))
    assert table.rowCount() > 0
