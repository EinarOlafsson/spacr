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

import logging
import sys
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from spacr.qt.widgets import glass


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


def _press(pos, glob=(500, 500), button=Qt.LeftButton):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(*pos),
                       QPointF(*glob), button, button, Qt.NoModifier)


def _move(pos, glob=(500, 500), buttons=Qt.NoButton):
    return QMouseEvent(QEvent.Type.MouseMove, QPointF(*pos), QPointF(*glob),
                       Qt.NoButton, buttons, Qt.NoModifier)


def _release(pos=(0, 0), glob=(500, 500)):
    return QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(*pos),
                       QPointF(*glob), Qt.LeftButton, Qt.NoButton,
                       Qt.NoModifier)


# ---------------------------------------------------------------------------
# Which dialogs are treated
# ---------------------------------------------------------------------------

def test_a_widget_that_is_not_a_dialog_is_never_glassed(qtbot):
    plain = QWidget()
    qtbot.addWidget(plain)
    assert glass.wants_glass(plain) is False
    assert glass.glass(plain) is False


def test_a_dialog_that_says_no_keeps_its_own_painting(dialog):
    dialog.setProperty(glass.NO_GLASS, True)
    assert glass.wants_glass(dialog) is False


def test_a_dialog_treated_once_is_not_treated_again(dialog):
    dialog.setProperty(glass.GLASSED, True)
    assert glass.wants_glass(dialog) is False


def test_without_the_card_class_a_dialog_is_still_a_candidate(dialog,
                                                              monkeypatch):
    """Not being able to LOOK for a card is not evidence there is one."""
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.setup_card", None)
    assert glass.wants_glass(dialog) is True
    assert glass.glass(dialog) is False, \
        "no card class means nothing to put behind the dialog"


# ---------------------------------------------------------------------------
# Clearing the containers
# ---------------------------------------------------------------------------

def test_a_dialog_with_no_containers_clears_nothing(dialog):
    """The dialog's own body is not a container it holds."""
    assert glass.clear_the_containers(dialog) == 0


def test_a_control_and_its_internals_are_left_opaque(dialog, qtbot):
    """A control you can see through is a control you cannot read."""
    from PySide6.QtWidgets import QComboBox

    holder = QWidget(dialog)
    QHBoxLayout(holder).addWidget(QComboBox(holder))
    assert glass.clear_the_containers(dialog) == 1, \
        "the combo, or something inside it, was made transparent"


def test_without_the_theme_helper_nothing_is_made_transparent(dialog,
                                                              monkeypatch):
    QWidget(dialog)
    monkeypatch.setitem(sys.modules, "spacr.qt.theme", None)
    assert glass.clear_the_containers(dialog) == 0


def test_a_container_that_will_not_go_transparent_is_reported_as_none(
        dialog, monkeypatch):
    from spacr.qt import theme

    QWidget(dialog)
    monkeypatch.setattr(theme, "make_transparent",
                        lambda *w: (_ for _ in ()).throw(
                            RuntimeError("stylesheet is locked")))
    assert glass.clear_the_containers(dialog) == 0


# ---------------------------------------------------------------------------
# Resizing a window that has no frame
# ---------------------------------------------------------------------------

def test_a_point_in_the_middle_is_not_an_edge(dialog):
    assert glass._edges_at(dialog, dialog.rect().center()) == Qt.Edge(0)
    assert glass._cursor_for(Qt.Edge(0)) is None


@pytest.mark.parametrize("point,shape", [
    ((0, 0), Qt.CursorShape.SizeFDiagCursor),
    ((319, 0), Qt.CursorShape.SizeBDiagCursor),
    ((0, 239), Qt.CursorShape.SizeBDiagCursor),
    ((319, 239), Qt.CursorShape.SizeFDiagCursor),
    ((0, 120), Qt.CursorShape.SizeHorCursor),
    ((160, 0), Qt.CursorShape.SizeVerCursor),
])
def test_each_edge_says_which_way_it_will_move(dialog, point, shape):
    from PySide6.QtCore import QPoint

    assert glass._cursor_for(glass._edges_at(dialog, QPoint(*point))) == shape


def test_hovering_an_edge_changes_the_pointer_and_leaving_puts_it_back(
        dialog):
    """A resize you cannot see is one nobody finds."""
    assert glass.let_the_user_resize(dialog) is True
    watcher = dialog._spacr_resizer

    watcher.eventFilter(dialog, _move((1, 120)))
    assert dialog.cursor().shape() == Qt.CursorShape.SizeHorCursor

    watcher.eventFilter(dialog, _move((160, 120)))
    assert dialog.cursor().shape() == Qt.CursorShape.ArrowCursor

    watcher.eventFilter(dialog, _move((1, 120)))
    watcher.eventFilter(dialog, QEvent(QEvent.Type.Leave))
    assert dialog.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_a_press_in_the_middle_does_not_start_a_resize(dialog):
    glass.let_the_user_resize(dialog)
    watcher = dialog._spacr_resizer
    assert watcher.eventFilter(dialog, _press((160, 120))) is False


def test_a_press_on_an_edge_is_taken_by_the_resize(dialog, monkeypatch):
    """Handed to the compositor: computing the geometry here walks away."""
    glass.let_the_user_resize(dialog)
    watcher = dialog._spacr_resizer

    # No native window yet — nothing to hand the drag to, and no crash.
    assert dialog.windowHandle() is None
    assert watcher.eventFilter(dialog, _press((1, 120))) is False

    asked = []
    handle = types.SimpleNamespace(
        startSystemResize=lambda edges: asked.append(edges))
    monkeypatch.setattr(type(dialog), "windowHandle", lambda self: handle)
    assert watcher.eventFilter(dialog, _press((1, 1))) is True
    assert asked == [Qt.Edge.LeftEdge | Qt.Edge.TopEdge]


def test_an_event_the_resizer_cannot_read_is_swallowed(dialog):
    glass.let_the_user_resize(dialog)
    watcher = dialog._spacr_resizer

    class Unreadable:
        def type(self):
            raise RuntimeError("event is gone")

    assert watcher.eventFilter(dialog, Unreadable()) is False


def test_a_filter_whose_window_has_gone_answers_nothing(dialog, qtbot):
    glass.let_the_user_resize(dialog)
    watcher = dialog._spacr_resizer
    other = QWidget()
    qtbot.addWidget(other)
    assert watcher.eventFilter(other, _move((1, 1))) is False
    watcher._window = None
    assert watcher.eventFilter(dialog, _move((1, 1))) is False


def test_there_is_nothing_to_make_resizable_without_a_window():
    assert glass.let_the_user_resize(None) is False


def test_a_window_keeps_the_one_resizer_it_has(dialog):
    assert glass.let_the_user_resize(dialog) is True
    first = dialog._spacr_resizer
    assert glass.let_the_user_resize(dialog) is False
    assert dialog._spacr_resizer is first


def test_a_resizer_that_cannot_be_built_leaves_the_window_alone(dialog,
                                                                monkeypatch):
    monkeypatch.setattr(glass, "_ResizeByEdge",
                        lambda window: (_ for _ in ()).throw(
                            RuntimeError("no event loop")))
    assert glass.let_the_user_resize(dialog) is False


# ---------------------------------------------------------------------------
# Dragging a window that has no title bar
# ---------------------------------------------------------------------------

def test_dragging_the_background_moves_the_window(dialog):
    """The title bar was where a window was dragged from."""
    dialog.move(100, 100)
    dragger = glass._DragByBackground(dialog)

    dragger.eventFilter(dialog, _press((160, 120), glob=(500, 500)))
    assert dragger._grab is not None
    dragger.eventFilter(dialog, _move((160, 120), glob=(540, 530)))
    assert dialog.pos().x() == 140
    assert dialog.pos().y() == 130

    dragger.eventFilter(dialog, _release())
    assert dragger._grab is None
    dragger.eventFilter(dialog, _move((160, 120), glob=(700, 700)))
    assert dialog.pos().x() == 140, "the window kept moving after the release"


def test_a_press_on_a_control_is_left_entirely_alone(dialog):
    button = QPushButton("Save", dialog)
    button.setGeometry(10, 10, 80, 24)
    dragger = glass._DragByBackground(dialog)
    dragger.eventFilter(dialog, _press((20, 20)))
    assert dragger._grab is None


def test_a_drag_that_goes_wrong_forgets_the_grab(dialog, monkeypatch):
    dragger = glass._DragByBackground(dialog)
    dragger.eventFilter(dialog, _press((160, 120)))
    assert dragger._grab is not None
    monkeypatch.setattr(type(dialog), "move",
                        lambda self, point: (_ for _ in ()).throw(
                            RuntimeError("no window manager")))
    dragger.eventFilter(dialog, _move((160, 120), glob=(600, 600)))
    assert dragger._grab is None


def test_a_dragger_whose_dialog_has_gone_answers_nothing(dialog, qtbot):
    dragger = glass._DragByBackground(dialog)
    other = QWidget()
    qtbot.addWidget(other)
    assert dragger.eventFilter(other, _press((1, 1))) is False
    dragger._dialog = None
    assert dragger.eventFilter(dialog, _press((1, 1))) is False


# ---------------------------------------------------------------------------
# The window itself
# ---------------------------------------------------------------------------

def test_the_dialog_stops_painting_its_own_background_once(dialog):
    dialog.setStyleSheet("QLabel { color: red; }")
    assert glass._paint_nothing_behind_the_card(dialog) is True
    assert "QLabel { color: red; }" in dialog.styleSheet()
    assert glass.NO_BACKGROUND in dialog.styleSheet()
    assert glass._paint_nothing_behind_the_card(dialog) is False


def test_a_dialog_that_will_not_take_the_stylesheet(dialog, monkeypatch):
    monkeypatch.setattr(type(dialog), "styleSheet",
                        lambda self: (_ for _ in ()).throw(
                            RuntimeError("gone")))
    assert glass._paint_nothing_behind_the_card(dialog) is False


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

def test_a_dialog_with_no_layout_gets_no_extra_margin(qtbot):
    bare = QDialog()
    qtbot.addWidget(bare)
    assert bare.layout() is None
    assert glass._make_room_for_the_rim(bare) is False


def test_the_rim_gets_its_band_added_to_the_dialogs_own_margins(dialog):
    dialog.layout().setContentsMargins(4, 4, 4, 4)
    assert glass._make_room_for_the_rim(dialog) is True
    assert dialog.layout().contentsMargins().left() == 4 + glass.RIM_ROOM


def test_a_layout_that_refuses_the_margins(dialog, monkeypatch):
    layout = dialog.layout()
    monkeypatch.setattr(type(layout), "setContentsMargins",
                        lambda self, *a: (_ for _ in ()).throw(
                            RuntimeError("fixed layout")))
    assert glass._make_room_for_the_rim(dialog) is False


def test_a_dialog_with_no_button_is_told_how_to_close_it(dialog):
    assert glass._say_how_to_close_it(dialog) is True
    hints = [w for w in dialog.findChildren(QLabel)
             if "Escape" in w.text()]
    assert len(hints) == 1
    assert dialog.property(glass.CLOSE_HINT) is True
    assert glass._say_how_to_close_it(dialog) is False, \
        "a second pass added a second hint"


def test_a_dialog_with_a_button_already_has_a_way_out(dialog):
    dialog.layout().addWidget(QDialogButtonBox(QDialogButtonBox.Ok, dialog))
    assert glass._say_how_to_close_it(dialog) is False


def test_a_dialog_laid_out_sideways_gets_no_hint(qtbot):
    """The hint goes under the form; a row is not a form."""
    sideways = QDialog()
    qtbot.addWidget(sideways)
    QHBoxLayout(sideways)
    assert glass._say_how_to_close_it(sideways) is False


def test_a_hint_that_cannot_be_built_is_not_a_dialog_that_cannot_open(
        dialog, monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.i18n", None)
    assert glass._say_how_to_close_it(dialog) is False
    assert not dialog.property(glass.CLOSE_HINT)


# ---------------------------------------------------------------------------
# The backdrop behind the card
# ---------------------------------------------------------------------------

def test_the_user_who_turned_the_backdrop_off_keeps_it_off(dialog,
                                                           monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_ambient_enabled", lambda: False)
    assert glass._install_the_backdrop(dialog) is None

    monkeypatch.setattr(preferences, "get_ambient_enabled", lambda: True)
    monkeypatch.setattr(preferences, "get_popup_backdrop", lambda: "off")
    assert glass._install_the_backdrop(dialog) is None


def test_an_unreadable_preference_still_leaves_a_backdrop_to_install(
        dialog, monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_ambient_enabled",
                        lambda: (_ for _ in ()).throw(RuntimeError("no store")))
    installed = []
    monkeypatch.setattr("spacr.qt.widgets.ambient.install_ambient",
                        lambda dlg, **kwargs: installed.append(kwargs) or
                        QWidget(dlg))
    assert glass._install_the_backdrop(dialog) is not None
    assert installed[0]["theme"] == "aurora"
    assert installed[0]["corner_radius"] == glass.CARD_RADIUS


def test_with_no_ambient_engine_the_card_is_still_a_card(dialog, monkeypatch):
    """INVARIANTS 10: without the engine the dialog still works."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_ambient_enabled", lambda: True)
    monkeypatch.setattr(preferences, "get_popup_backdrop", lambda: "aurora")
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.ambient", None)
    assert glass._install_the_backdrop(dialog) is None


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

def test_the_filter_is_installed_once_and_comes_off_again(qapp):
    glass.uninstall_glass_everywhere()
    assert glass.install_glass_everywhere() is True
    assert glass.install_glass_everywhere() is False
    assert glass.uninstall_glass_everywhere() is True
    assert glass.uninstall_glass_everywhere() is False


def test_there_is_nothing_to_install_the_filter_on(monkeypatch):
    from PySide6.QtWidgets import QApplication

    glass.uninstall_glass_everywhere()
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    assert glass.install_glass_everywhere() is False
    assert glass._INSTALLED is None


def test_a_filter_that_will_not_install_leaves_nothing_behind(monkeypatch):
    glass.uninstall_glass_everywhere()
    monkeypatch.setattr(glass, "_GlassInstaller",
                        lambda parent: (_ for _ in ()).throw(
                            RuntimeError("no application thread")))
    assert glass.install_glass_everywhere() is False
    assert glass._INSTALLED is None


def test_a_filter_that_will_not_come_off_is_still_forgotten(qapp,
                                                            monkeypatch):
    """Forgotten either way, or a second install could never happen."""
    glass.uninstall_glass_everywhere()
    glass.install_glass_everywhere()
    installed = glass._INSTALLED
    # Bound before the patch, so this file can put the application back the
    # way it found it: a filter left on the QApplication would glass every
    # dialog built by every test after this one.
    really_remove = qapp.removeEventFilter

    monkeypatch.setattr(type(qapp), "removeEventFilter",
                        lambda self, obj: (_ for _ in ()).throw(
                            RuntimeError("already gone")))
    try:
        assert glass.uninstall_glass_everywhere() is True
        assert glass._INSTALLED is None
    finally:
        really_remove(installed)


def test_an_event_the_installer_cannot_read_is_swallowed(qapp):
    installer = glass._GlassInstaller(qapp)

    class Unreadable:
        def type(self):
            raise RuntimeError("event is gone")

    assert installer.eventFilter(qapp, Unreadable()) is False
