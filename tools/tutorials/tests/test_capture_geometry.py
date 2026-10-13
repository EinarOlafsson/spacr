"""A dialog's width/height editors must not shadow native geometry access."""
from pathlib import Path
import sys

import pytest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QWidget

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_geometry import capture_rect, foreground_dialogs


def test_active_modal_is_above_its_background_dialog_and_does_not_mutate_input():
    background, modal = object(), object()
    original = [modal, background]
    assert foreground_dialogs(original) == original
    assert foreground_dialogs(original, modal) == [background, modal]
    assert original == [modal, background]


def test_active_popup_is_above_modal_without_adding_hidden_windows():
    background, modal, popup, hidden = object(), object(), object(), object()
    assert foreground_dialogs([popup, modal, background], modal, popup) == [background, modal, popup]
    assert foreground_dialogs([modal, background], hidden, hidden) == [modal, background]


@pytest.fixture
def widgets():
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.resize(400, 300)
    child = QWidget(window)
    child.setGeometry(10, 20, 80, 60)
    window.show()
    child.show()
    app.processEvents()
    yield app, window, child
    window.close()
    window.deleteLater()
    app.processEvents()


def test_visible_widget_has_its_actual_geometry(widgets):
    _, window, child = widgets
    assert capture_rect(child, window) == [10, 20, 80, 60]


def test_export_dialog_style_width_and_height_editors_do_not_break_geometry(widgets):
    _, window, child = widgets
    child.width = QDoubleSpinBox(child)
    child.height = QDoubleSpinBox(child)
    assert capture_rect(child, window) == [10, 20, 80, 60]


def test_hidden_widget_is_not_presented_as_visible(widgets):
    _, window, child = widgets
    assert capture_rect(child, window) == [10, 20, 80, 60]
    child.hide()
    assert capture_rect(child, window) is None


def test_partial_widget_is_clipped_to_the_recorded_window(widgets):
    _, window, child = widgets
    child.setGeometry(-10, 280, 80, 60)
    assert capture_rect(child, window) == [0, 280, 70, 20]


def test_widget_outside_the_recorded_window_has_no_focus_rect(widgets):
    _, window, child = widgets
    assert capture_rect(child, window) is not None
    child.move(450, 350)
    assert capture_rect(child, window) is None
