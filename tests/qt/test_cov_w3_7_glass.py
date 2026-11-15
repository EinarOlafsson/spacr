"""The glass treatment where it has to give way: the edges of every helper.

The look itself is asserted by ``test_every_popup_has_the_card_and_the_rim``
and its neighbours — a card behind every dialog, a rim with room to run,
no square corners. This file drives the same functions at the points where
something is missing or refuses: a dialog with no layout, a window with no
native handle, a theme helper that will not import, an ambient engine that
is not there.

The rule the module states is that decoration is never load-bearing, so
every one of these has the same shape: the helper answers False (or 0, or
None), the dialog is still a working dialog, and nothing propagates.
"""
from __future__ import annotations


import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QWidget)

from spacr.qt.widgets import glass
from spacr.qt.widgets.cursor_policy import arrow_cursor


@pytest.fixture(autouse=True)
def _no_installer_left_behind():
    """This file installs and removes the application filter; leave none on."""
    yield
    glass.uninstall_glass_everywhere()


@pytest.fixture
def dialog(qtbot):
    """A plain dialog with a vertical layout, the shape a settings popup has."""
    dlg = QDialog()
    qtbot.addWidget(dlg)
    dlg.resize(320, 240)
    QVBoxLayout(dlg)
    return dlg


def _move(pos, glob=(500, 500), buttons=Qt.NoButton):
    return QMouseEvent(QEvent.Type.MouseMove, QPointF(*pos), QPointF(*glob),
                       Qt.NoButton, buttons, Qt.NoModifier)


# ---------------------------------------------------------------------------
# Which dialogs are treated
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Clearing the containers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Resizing a window that has no frame
# ---------------------------------------------------------------------------

def test_hovering_an_edge_shows_a_line_and_leaving_hides_it(
        dialog):
    """A resize you cannot see is one nobody finds."""
    assert glass.let_the_user_resize(dialog) is True
    watcher = dialog._spacr_resizer

    watcher.eventFilter(dialog, _move((1, 120)))
    assert dialog.cursor().shape() == Qt.CursorShape.ArrowCursor
    assert watcher._hint.edges == Qt.LeftEdge

    watcher.eventFilter(dialog, _move((160, 120)))
    assert not watcher._hint.edges
    assert dialog.cursor().shape() == Qt.CursorShape.ArrowCursor or (
        dialog.cursor().pixmap().cacheKey() == arrow_cursor(False).pixmap().cacheKey())

    watcher.eventFilter(dialog, _move((1, 120)))
    watcher.eventFilter(dialog, QEvent(QEvent.Type.Leave))
    assert not watcher._hint.edges
    assert dialog.cursor().shape() == Qt.CursorShape.ArrowCursor or (
        dialog.cursor().pixmap().cacheKey() == arrow_cursor(False).pixmap().cacheKey())


# ---------------------------------------------------------------------------
# Dragging a window that has no title bar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The window itself
# ---------------------------------------------------------------------------

def test_a_dialog_that_was_showing_is_put_back(dialog, qtbot):
    """``setWindowFlags`` hides a visible widget; Qt requires a re-show."""
    dialog.show()
    qtbot.waitExposed(dialog)
    assert glass.make_frameless(dialog) is True
    assert not dialog.isHidden(), \
        "opening a settings window hid the settings window"
    assert bool(dialog.windowFlags() & Qt.FramelessWindowHint)
    assert dialog.property(glass.DETACHED) is True
    dialog.hide()


def test_a_dialog_that_refuses_the_flags_keeps_the_frame(dialog, monkeypatch):
    monkeypatch.setattr(type(dialog), "setWindowFlags",
                        lambda self, flags: (_ for _ in ()).throw(
                            RuntimeError("no native window")))
    assert glass.make_frameless(dialog) is False


def test_a_window_with_no_size_cannot_be_masked(qtbot):
    empty = QDialog()
    qtbot.addWidget(empty)
    empty.resize(0, 0)
    assert glass.round_the_corners(empty) is False


def test_a_window_that_will_not_take_a_mask_keeps_square_corners(dialog,
                                                                 monkeypatch):
    monkeypatch.setattr(type(dialog), "setMask",
                        lambda self, region: (_ for _ in ()).throw(
                            RuntimeError("no shape extension")))
    assert glass.round_the_corners(dialog) is False


def test_the_backdrop_that_cannot_fit_does_not_take_the_dialog_down(dialog,
                                                                    qtbot):
    card = QWidget(dialog)
    backdrop = glass._Backdrop(dialog, card)
    assert card.geometry() == dialog.rect()

    backdrop._card = None
    backdrop._fit()                       # nothing to fit; nothing raised

    # Something that is not a widget at all: the card is somebody else's
    # object, and a backdrop that raises while fitting one is a backdrop
    # that takes the dialog's show event with it.
    backdrop._card = object()
    backdrop._fit()

    backdrop._dialog = None
    assert backdrop.eventFilter(dialog, QEvent(QEvent.Type.Resize)) is False


def test_the_backdrop_refits_the_card_when_the_dialog_resizes(dialog, qtbot):
    """The card is kept at the dialog's size without subclassing it."""
    card = QWidget(dialog)
    glass._Backdrop(dialog, card)
    dialog.show()
    qtbot.waitExposed(dialog)
    dialog.resize(400, 300)
    qtbot.waitUntil(lambda: card.geometry() == dialog.rect(), timeout=2000)
    assert card.geometry() == dialog.rect()
    assert card.width() > 0 and card.height() > 0
    dialog.hide()


# ---------------------------------------------------------------------------
# Room for the rim, and the way out
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The backdrop behind the card
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Glassing one dialog
# ---------------------------------------------------------------------------

@pytest.fixture
def no_backdrop(monkeypatch):
    """No drifting strata: this file is about the card, not the animation."""
    monkeypatch.setattr(glass, "_install_the_backdrop", lambda dialog: None)


def test_a_dialog_with_no_verdict_to_follow_is_still_glassed(qtbot,
                                                             no_backdrop):
    """``glass`` is called on anything a filter sees, not only real dialogs."""
    class Refuses:
        def connect(self, _slot):
            raise RuntimeError("no verdict to follow")

    class Bare(QDialog):
        def __init__(self):
            super().__init__()
            self.accepted = Refuses()

    odd = Bare()
    qtbot.addWidget(odd)
    QVBoxLayout(odd)
    assert glass.glass(odd) is True
    assert odd.property(glass.GLASSED) is True


def test_a_dialog_that_cannot_be_glassed_opens_looking_as_it_did(dialog,
                                                                 no_backdrop,
                                                                 monkeypatch):
    """Decoration is never load-bearing: the dialog is still a dialog."""
    monkeypatch.setattr(glass, "_make_room_for_the_rim",
                        lambda dlg: (_ for _ in ()).throw(
                            RuntimeError("the layout is frozen")))
    assert glass.glass(dialog) is False
    assert not dialog.property(glass.GLASSED)
    assert dialog.isEnabled()


# ---------------------------------------------------------------------------
# The one install point
# ---------------------------------------------------------------------------

def test_an_event_the_installer_cannot_read_is_swallowed(qapp):
    installer = glass._GlassInstaller(qapp)

    class Unreadable:
        def type(self):
            raise RuntimeError("event is gone")

    assert installer.eventFilter(qapp, Unreadable()) is False


@pytest.fixture
def window_sheet(qtbot):
    """A per-window application sheet in force, removed afterwards."""
    from PySide6.QtWidgets import QApplication

    from spacr.qt import theme

    app = QApplication.instance()
    app.setStyleSheet("")
    theme.apply_stylesheet_per_window(app, "QLabel { color: rgb(11, 22, 33); }")
    yield app
    theme.apply_stylesheet_per_window(app, "")
    app.setStyleSheet("")


def test_the_card_rule_survives_a_theme_change(dialog, window_sheet, qtbot):
    """It is the dialog's own rule now, so a new window sheet keeps it."""
    from spacr.qt import theme

    assert glass._paint_nothing_behind_the_card(dialog) is True
    dialog.show()
    window_sheet.processEvents()
    theme.apply_stylesheet_per_window(
        window_sheet, "QLabel { color: rgb(200, 100, 50); }")
    window_sheet.processEvents()
    sheet = dialog.styleSheet()
    assert "rgb(200, 100, 50)" in sheet
    assert "rgb(11, 22, 33)" not in sheet
    assert sheet.count(glass.NO_BACKGROUND) == 1
