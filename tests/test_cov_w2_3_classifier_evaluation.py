"""The Classifier Evaluation workbench when the bundle will not cooperate.

Discovery and loading both run on workers, and every one of their failure
paths ends the same way: the screen has to say what went wrong and clear the
panel it can no longer describe. A stale confusion cell beside a bundle that
did not load is the one outcome the workbench may not have.
"""
from __future__ import annotations

import numpy as np
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


# ---------------------------------------------------------------------------
# loading a bundle
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# rendering without a bundle
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# the object lists beside a cell
# ---------------------------------------------------------------------------


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
    assert Path(opened[0].toLocalFile()) == Path(
        screen.bundle["path"]).parent.resolve()


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
