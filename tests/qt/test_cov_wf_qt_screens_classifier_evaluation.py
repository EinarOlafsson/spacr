"""The Classifier Evaluation workbench when a control fires with nothing open.

Three seams of the C8 confusion workbench that only the *empty* or the
*damaged* case reaches: the "sure at" spin box moving before any cell has
been clicked, the copy-path button pressed before a bundle is loaded, and a
matrix cell whose widget is gone while the tooltip pass walks the table.
Each one is a control the user can operate at a moment the screen has no
data for it, so each has to do nothing rather than raise -- a traceback out
of a spin box or a stale path on the clipboard is a bug the person sees.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.classifier_evaluation import (
    LeakageReport,
    evaluate_predictions,
    write_evaluation_bundle,
)
from spacr.qt.screens.classifier_evaluation import ClassifierEvaluationScreen


@pytest.fixture
def evaluation_root(tmp_path):
    """One small bundle with all four confusion cells populated.

    Two objects are classified correctly and two are not, so every (true,
    predicted) pair in the 2x2 matrix holds exactly one object and each of
    them can be inspected.
    """
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
def screen(qtbot, qt_theme_applied, evaluation_root):
    """A workbench with that bundle scanned and loaded, off the thread pool."""
    widget = ClassifierEvaluationScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._source.setText(str(evaluation_root))
    widget.scan()
    return widget


@pytest.fixture
def empty_screen(qtbot, qt_theme_applied):
    """A workbench that has never been pointed at a results folder."""
    widget = ClassifierEvaluationScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


class _Clipboard:
    """A stand-in clipboard that records what the screen tried to copy."""

    def __init__(self):
        self.copied = []

    def setText(self, value):
        self.copied.append(value)


def test_the_sure_at_control_only_resplits_a_cell_that_is_open(screen):
    """Moving the threshold with no cell open must not resurrect the last one.

    The spin box is live from the moment the page is built, and a user
    reaches for it while reading the matrix -- before clicking any cell, and
    again right after a new bundle clears the inspector. If the handler ran
    unconditionally it would rebuild whatever ``self._cell`` still pointed
    at, so a threshold nudge would silently repopulate the two crop lists
    with the *previous* cell's objects and the "Open in Annotate" buttons
    would hand Annotate crops from a confusion the user is no longer looking
    at. With nothing open the control has to be inert; with a cell open it
    has to re-split it where "sure" now starts.
    """
    screen._clear_cell()
    prompt = screen._cell_summary.text()

    screen._threshold.setValue(0.40)

    assert screen._cell is None, "an inert control opened a cell"
    assert screen._cell_summary.text() == prompt
    assert screen._high_list.count() == 0 and screen._low_list.count() == 0

    # The same control, one click later, on the cell it is meant to move:
    # plate2_B01 is a negative called positive at 0.60 confidence, so it is
    # "sure and wrong" under a 0.40 threshold ...
    screen.show_cell("negative", "positive")
    assert screen._cell is not None, "the bundle lost its negative/positive cell"
    assert screen._cell.threshold == pytest.approx(0.40)
    assert len(screen._cell.high) == 1 and screen._cell.low.empty

    # ... and "unsure and wrong" once the bar is raised above 0.60.
    screen._threshold.setValue(0.90)
    assert screen._cell.threshold == pytest.approx(0.90)
    assert screen._cell.high.empty and len(screen._cell.low) == 1
    assert "B01" in screen._low_list.item(0).text()


def test_copy_path_stays_silent_until_a_bundle_is_loaded(
        empty_screen, screen, monkeypatch):
    """An empty workbench must not put a stale or crashing path on the clipboard.

    ``Copy path`` is disabled until a bundle loads, but the slot is also the
    seam a shortcut or a script calls, and ``self.bundle["path"]`` on an
    unloaded screen is a ``TypeError``. Worse than the traceback would be a
    success message: the status line saying "Evaluation path copied" while
    the clipboard still holds whatever the user copied last is a lie they
    then paste into a manuscript. Nothing copied, nothing said -- and the
    moment a bundle IS loaded, the manifest path and the confirmation.
    """
    board = _Clipboard()
    monkeypatch.setattr(
        "spacr.qt.screens.classifier_evaluation.QApplication.clipboard",
        staticmethod(lambda: board),
    )

    assert empty_screen.bundle is None
    assert not empty_screen._copy_path.isEnabled()
    empty_screen._copy_current_path()
    assert board.copied == [], "an unloaded workbench copied something"
    assert "copied" not in empty_screen._status.text().casefold()

    # Same slot, same clipboard, one loaded bundle later.
    assert screen.bundle is not None and screen._copy_path.isEnabled()
    screen._copy_current_path()
    assert board.copied == [str(screen.bundle["path"])]
    assert "copied" in screen._status.text().casefold()


def test_a_matrix_cell_with_no_item_is_skipped_not_a_traceback(
        screen, monkeypatch):
    """One missing cell widget must not cost the whole matrix its tooltips.

    Every number in the matrix carries its share of the row on a tooltip --
    "1 object(s) - 50.0% of this row" -- which is what tells the user
    whether a cell is worth opening. The pass that writes those tooltips
    reads each widget back out of the table, and ``QTableWidget.item``
    returns ``None`` for any cell that was never filled or was taken out
    from under it. Without the skip that read is an ``AttributeError`` on
    ``None`` raised in the middle of rendering, which would abandon the
    matrix half-annotated and leave the ranking line and the cell inspector
    showing the previous bundle.
    """
    original = screen._render_frame

    def holed(table, frame, **kwargs):
        original(table, frame, **kwargs)
        if table is screen._confusion and table.rowCount():
            table.takeItem(0, 1)

    monkeypatch.setattr(screen, "_render_frame", holed)
    screen._render_confusion(dict(screen.bundle.get("summary") or {}))

    assert screen._confusion.item(0, 1) is None, "the hole was not punched"
    # The cell after the hole, and the whole second row, still got theirs.
    assert "50.0% of this row" in screen._confusion.item(0, 2).toolTip()
    assert "1 object(s)" in screen._confusion.item(1, 1).toolTip()
    # And the work that follows the loop still ran.
    assert screen._confusion_ranking.text()
    assert screen._cell is None
    assert screen._threshold.value() == pytest.approx(0.75)
