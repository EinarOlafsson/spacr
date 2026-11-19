"""Shared table-screen and backdrop contracts with the original case factories."""
from contextlib import nullcontext

import pytest
from PySide6.QtCore import QEvent, Qt

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.feature_explorer import FeatureExplorerScreen
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.screens.tabulate import TabulateScreen
from tests.qt.test_ambient import make_widget as make_ambient
from tests.qt.test_dna_rain import _at_design_scale, make_widget as make_rain


_SCREEN_TYPES = (FeatureExplorerScreen, GraphBuilderScreen, TabulateScreen)


@pytest.fixture
def screen(qtbot, screen_type):
    """Build each original screen inline with a private selection and Qt cleanup."""
    link = LinkedSelection()
    widget = screen_type(link=link, threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture(autouse=True)
def _original_backdrop_scale(request, qapp):
    """Keep the original DNA scale context active through Qt widget cleanup."""
    case = getattr(request.node, "callspec", None)
    factory = case.params.get("make_widget") if case is not None else None
    context = _at_design_scale() if factory is make_rain else nullcontext()
    with context:
        yield


@pytest.mark.parametrize(
    "screen_type", _SCREEN_TYPES,
    ids=[f"{kind.__name__}-test_picking_a_table_with_no_file_loaded_does_nothing"
         for kind in _SCREEN_TYPES],
)
def test_picking_a_table_with_no_file_loaded_does_nothing(screen):
    """The combo is populated programmatically; its signal must be harmless."""
    screen._on_table_picked("cell")

    assert screen._frame is None


@pytest.mark.parametrize(
    "screen_type", _SCREEN_TYPES,
    ids=[f"{kind.__name__}-test_a_failed_read_is_reported_on_the_source_line"
         for kind in _SCREEN_TYPES],
)
def test_a_failed_read_is_reported_on_the_source_line(screen, tmp_path):
    """The failure arrives from the worker and lands on the label."""
    screen._path = str(tmp_path / "cells.csv")

    screen._on_load_failed("No columns to parse from file")

    assert screen._source.text() == (
        "could not read cells.csv: No columns to parse from file")


@pytest.mark.parametrize("make_widget", [
    pytest.param(make_ambient,
                 id="test_a_focus_event_that_reaches_the_backdrop_is_refused"),
    pytest.param(make_rain,
                 id="test_a_focus_event_that_reaches_the_rain_is_refused"),
])
def test_a_focus_event_that_reaches_a_backdrop_is_refused(qtbot, make_widget):
    """Deliver FocusIn to each original backdrop and verify that it hands focus back."""
    from PySide6.QtGui import QFocusEvent
    from PySide6.QtWidgets import QApplication

    widget = make_widget(qtbot)
    handed_back = []
    widget.clearFocus = lambda: handed_back.append(True)
    event = QFocusEvent(QEvent.Type.FocusIn, Qt.FocusReason.OtherFocusReason)
    event.accept()

    QApplication.sendEvent(widget, event)

    assert not event.isAccepted()
    assert handed_back == [True]
