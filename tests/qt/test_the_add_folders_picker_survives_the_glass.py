"""The Overview's Add folders / Add plates picker under the app's own glass.

On the maintainer's desktop the picker opened as a small clipped corner with
no Choose or Cancel button and no edge that would drag, while Qt reported it
at four fifths of the screen. Under an application style sheet QFileDialog's
constructor polishes itself before Python has bound the object it is
building, so the glass filter was handed a temporary wrapper: the rounded
mask was cut at the constructor's default size and the backdrop that should
have refitted it listened for a wrapper the dialog never delivered again.

These tests run the picker the way spaCR runs it: the glass filter and a
style sheet on the application, the dialog built by the Overview's own
handler and shown, with the event loop turning, before anything is read.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QPushButton

from spacr.qt import theme
from spacr.qt.widgets import glass
from spacr.qt.widgets import qc_field_browser as browser

pytestmark = pytest.mark.qt

PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"


class _PolishWatch(QObject):
    """Counts the QFileDialog polishes that arrive on a wrapper Python never returned."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Polish and isinstance(watched, QFileDialog):
            self.seen.append(watched)
        return False


@pytest.fixture
def glassed_app(monkeypatch):
    """The app's dark sheet and glass filter, as spaCR runs with them."""
    app = QApplication.instance()
    from spacr.qt.widgets import setup_card

    monkeypatch.setattr(setup_card.SetupCard, "_tick", lambda self: None)
    previous = app.styleSheet()
    app.setStyleSheet(theme.stylesheet(theme="dark"))
    installed_here = glass.install_glass_everywhere(app)
    yield app
    if installed_here:
        glass.uninstall_glass_everywhere(app)
    app.setStyleSheet(previous)


def _plate(folder):
    import numpy as np
    import tifffile

    folder.mkdir(parents=True, exist_ok=True)
    for well in ("A01", "A02"):
        for channel in (1, 2):
            tifffile.imwrite(folder / f"p1-{well}-f1-c{channel}.tif",
                             np.full((24, 32), channel, dtype=np.uint16))
    return folder


def _shown(qtbot, states):
    """An ``exec`` that shows the picker, lets the loop run, records it and cancels."""

    def run(dialog):
        dialog.show()
        qtbot.wait(400)
        states.append(_state(dialog))
        dialog.reject()
        return 0

    return run


def _state(dialog):
    mask = dialog.mask()
    buttons = {}
    for button in dialog.findChildren(QPushButton):
        if button.isHidden():
            continue
        top_left = button.mapTo(dialog, QPoint(0, 0))
        bottom_right = button.mapTo(dialog, QPoint(button.width() - 1, button.height() - 1))
        buttons[button.text().replace("&", "")] = {
            "visible": button.isVisible(),
            "region": not button.visibleRegion().isEmpty(),
            "inside": dialog.rect().contains(top_left) and dialog.rect().contains(bottom_right),
            "unmasked": mask.isEmpty() or (mask.contains(top_left) and mask.contains(bottom_right)),
        }
    return {"size": (dialog.width(), dialog.height()), "mask": mask.boundingRect(),
            "buttons": buttons, "glassed": bool(dialog.property(glass.GLASSED)),
            "frameless": bool(dialog.windowFlags() & Qt.FramelessWindowHint),
            "rect": dialog.rect(), "dialog": dialog}


def _assert_big_with_every_button(state):
    width, height = state["size"]
    assert width >= 720 and height >= 480
    assert state["glassed"] and state["frameless"]
    mask = state["mask"]
    assert mask.isEmpty() or (mask.width() >= width - 2 and mask.height() >= height - 2), (
        f"a {mask.width()}x{mask.height()} window mask cuts the {width}x{height} picker")
    for name in ("Choose", "Cancel"):
        button = state["buttons"].get(name)
        assert button is not None, f"no {name} button among {sorted(state['buttons'])}"
        assert button == {"visible": True, "region": True, "inside": True, "unmasked": True}, (name, button)


def _drag(dialog, edge, delta):
    start = dialog.mapToGlobal(edge)
    QApplication.sendEvent(dialog, QMouseEvent(QEvent.MouseButtonPress, edge, start, Qt.LeftButton,
                                               Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(dialog, QMouseEvent(QEvent.MouseMove, edge + delta, start + delta, Qt.NoButton,
                                               Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(dialog, QMouseEvent(QEvent.MouseButtonRelease, edge + delta, start + delta,
                                               Qt.LeftButton, Qt.NoButton, Qt.NoModifier))


def test_the_file_dialog_polishes_before_python_holds_it(glassed_app):
    watch = _PolishWatch()
    glassed_app.installEventFilter(watch)
    try:
        dialog = QFileDialog(None, "probe")
    finally:
        glassed_app.removeEventFilter(watch)
    assert watch.seen, "the precondition this file guards against no longer happens"
    assert all(seen is not dialog for seen in watch.seen)
    assert not dialog.property(glass.GLASSED)
    dialog.deleteLater()


def test_the_wizard_add_folders_picker_shows_big_with_every_button(glassed_app, qtbot, tmp_path, monkeypatch):
    plate = _plate(tmp_path / "plate1")
    wizard = browser._OverviewMetadataWizard(str(plate), {"custom_regex": PATTERN})
    wizard._startup_timer.stop()
    qtbot.addWidget(wizard)
    states = []
    monkeypatch.setattr(QFileDialog, "exec", _shown(qtbot, states))
    assert wizard._choose_more_sources() == []
    assert len(states) == 1
    _assert_big_with_every_button(states[0])


def test_the_add_plates_picker_shows_big_with_every_button_and_drags_by_its_edges(
        glassed_app, qtbot, tmp_path, monkeypatch):
    plate = _plate(tmp_path / "plate1")
    view = browser._FieldOverview(settings={
        "source": str(plate), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (3, 3), "representatives": 1})
    qtbot.addWidget(view)
    view._cancel()
    states = []

    def run(dialog):
        dialog.show()
        qtbot.wait(400)
        states.append(_state(dialog))
        width = dialog.width()
        edge = QPointF(dialog.width() - 3, dialog.height() / 2)
        assert dialog.mask().isEmpty() or dialog.mask().contains(edge.toPoint())
        _drag(dialog, edge, QPointF(-60, 0))
        qtbot.wait(50)
        states.append(width - dialog.width())
        dialog.reject()
        return 0

    monkeypatch.setattr(QFileDialog, "exec", run)
    assert view._choose_plates() == []
    _assert_big_with_every_button(states[0])
    assert states[1] == 60


def test_a_dialog_python_holds_is_still_glassed_on_its_first_polish(glassed_app, qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.ensurePolished()
    assert dialog.property(glass.GLASSED)
