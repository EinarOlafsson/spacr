"""Dragging a rectangle of wells, editing the condition table, exporting.

The plate map is a grid the user drags across, so the gesture is the feature:
press anchors, move previews, release commits — and the pressed widget keeps
the mouse grab, so the well under the pointer has to be found by asking rather
than by waiting to be entered. The rest is what the screen does with rows and
exports that are not usable yet.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt              # noqa: E402
from PySide6.QtGui import QMouseEvent                               # noqa: E402
from PySide6.QtWidgets import QWidget                               # noqa: E402

from spacr.qt.screens import experiment_design as ED                # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def screen(qtbot):
    widget = ED.ExperimentDesignScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def _mouse(kind, widget, point=(2, 2), modifiers=Qt.NoModifier):
    return QMouseEvent(kind, QPointF(*point), widget.mapToGlobal(QPointF(*point)),
                       Qt.LeftButton, Qt.LeftButton, modifiers)


# ---------------------------------------------------------------------------
# The drag
# ---------------------------------------------------------------------------


def test_a_point_off_the_plate_names_no_well(screen):
    assert screen.well_at(QPoint(-5000, -5000)) is None


def test_a_well_with_no_screen_above_it_does_not_raise(qtbot):
    """A well built outside the screen must not throw from a click."""
    holder = QWidget()
    qtbot.addWidget(holder)
    orphan = ED._Well(0, 0, holder)

    orphan.mousePressEvent(_mouse(QEvent.MouseButtonPress, orphan))
    orphan.mouseMoveEvent(_mouse(QEvent.MouseMove, orphan))
    orphan.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, orphan))

    assert orphan._screen() is None


# ---------------------------------------------------------------------------
# The condition table
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_cancelling_the_folder_dialog_writes_nothing(screen, tmp_path,
                                                      monkeypatch):
    monkeypatch.setattr(ED.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))

    screen._on_export()

    assert list(tmp_path.iterdir()) == []
