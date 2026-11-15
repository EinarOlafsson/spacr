"""The glass treatment of dialogs at every point where something is missing or refuses.

The module's rule is that decoration is never load-bearing: a dialog with no
layout, a window with no native handle, a theme helper that will not import,
an ambient engine that is not there, an application that refuses to give an
event filter back -- each answers False, 0 or None, the dialog stays a working
dialog, and nothing propagates.  The application-wide filter's bookkeeping must
always end on exactly one filter and one owner, or the next install would be a
false "already installed".

One general test per unit, each walking a case table with pytest's
``subtests`` fixture; every case builds its own dialog.
"""
from __future__ import annotations

import logging
import sys
import types
from contextlib import contextmanager

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog,  # noqa: E402
                               QDialogButtonBox, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from spacr.qt import preferences, theme  # noqa: E402
from spacr.qt.widgets import ambient, glass  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _no_installer_left_behind():
    """Leave neither a filter nor a remembered owner for the next test."""
    yield
    glass.uninstall_glass_everywhere()
    glass._INSTALLED = None
    glass._INSTALLED_APP = None


@pytest.fixture
def make_dialog(qtbot):
    """Build a dialog with a vertical layout (or the given one, or none)."""
    def build(layout=QVBoxLayout):
        dlg = QDialog()
        qtbot.addWidget(dlg)
        dlg.resize(320, 240)
        if layout is not None:
            layout(dlg)
        return dlg
    return build


def _raises(message):
    def boom(*_args, **_kwargs):
        raise RuntimeError(message)
    return boom


def _press(pos, glob=(500, 500), button=Qt.LeftButton):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(*pos),
                       QPointF(*glob), button, button, Qt.NoModifier)


def _move(pos, glob=(500, 500), buttons=Qt.NoButton):
    return QMouseEvent(QEvent.Type.MouseMove, QPointF(*pos), QPointF(*glob),
                       Qt.NoButton, buttons, Qt.NoModifier)


def _release(pos=(0, 0), glob=(500, 500)):
    return QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(*pos),
                       QPointF(*glob), Qt.LeftButton, Qt.NoButton, Qt.NoModifier)


def test_wants_glass_picks_only_untreated_dialogs(make_dialog, qtbot, monkeypatch, subtests):
    """wants_glass is False for non-dialogs, opted-out or already-glassed dialogs; a missing card class still says True."""
    with subtests.test(msg="plain_widget_is_never_glassed"):
        plain = QWidget()
        qtbot.addWidget(plain)
        assert glass.wants_glass(plain) is False
        assert glass.glass(plain) is False

    for case_id, prop in [("dialog_that_says_no", glass.NO_GLASS),
                          ("dialog_already_treated", glass.GLASSED)]:
        with subtests.test(msg=case_id):
            dialog = make_dialog()
            dialog.setProperty(prop, True)
            assert glass.wants_glass(dialog) is False

    with subtests.test(msg="no_card_class_is_a_candidate_but_cannot_be_glassed"):
        dialog = make_dialog()
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "spacr.qt.widgets.setup_card", None)
            assert glass.wants_glass(dialog) is True
            assert glass.glass(dialog) is False


def test_clear_the_containers_counts_only_containers_it_made_transparent(make_dialog, monkeypatch, subtests):
    """clear_the_containers clears plain containers, never controls, and reports 0 when the theme helper fails."""
    with subtests.test(msg="no_containers"):
        assert glass.clear_the_containers(make_dialog()) == 0

    with subtests.test(msg="a_control_and_its_internals_stay_opaque"):
        dialog = make_dialog()
        holder = QWidget(dialog)
        QHBoxLayout(holder).addWidget(QComboBox(holder))
        assert glass.clear_the_containers(dialog) == 1

    with subtests.test(msg="no_theme_helper"):
        dialog = make_dialog()
        QWidget(dialog)
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "spacr.qt.theme", None)
            assert glass.clear_the_containers(dialog) == 0

    with subtests.test(msg="container_refusing_transparency"):
        dialog = make_dialog()
        QWidget(dialog)
        with monkeypatch.context() as mp:
            mp.setattr(theme, "make_transparent", _raises("stylesheet is locked"))
            assert glass.clear_the_containers(dialog) == 0


def test_let_the_user_resize_installs_one_edge_resizer(make_dialog, qtbot, monkeypatch, subtests):
    """let_the_user_resize installs one resizer that takes edge presses only and ignores foreign or unreadable events."""
    with subtests.test(msg="no_window"):
        assert glass.let_the_user_resize(None) is False

    with subtests.test(msg="a_window_keeps_its_one_resizer"):
        dialog = make_dialog()
        assert glass.let_the_user_resize(dialog) is True
        first = dialog._spacr_resizer
        assert glass.let_the_user_resize(dialog) is False
        assert dialog._spacr_resizer is first

    with subtests.test(msg="a_resizer_that_cannot_be_built"):
        dialog = make_dialog()
        with monkeypatch.context() as mp:
            mp.setattr(glass, "_ResizeByEdge", _raises("no event loop"))
            assert glass.let_the_user_resize(dialog) is False
        assert not hasattr(dialog, "_spacr_resizer")

    with subtests.test(msg="press_in_the_middle_is_not_a_resize"):
        dialog = make_dialog()
        glass.let_the_user_resize(dialog)
        assert dialog._spacr_resizer.eventFilter(dialog, _press((160, 120))) is False

    with subtests.test(msg="press_on_an_edge_is_taken"):
        dialog = make_dialog()
        glass.let_the_user_resize(dialog)
        watcher = dialog._spacr_resizer
        assert dialog.windowHandle() is None
        assert watcher.eventFilter(dialog, _press((1, 120))) is False
        asked = []
        handle = types.SimpleNamespace(startSystemResize=asked.append)
        with monkeypatch.context() as mp:
            mp.setattr(type(dialog), "windowHandle", lambda self: handle)
            assert watcher.eventFilter(dialog, _press((1, 1))) is True
        assert asked == []
        assert watcher._grab[0] == Qt.Edge.LeftEdge | Qt.Edge.TopEdge

    with subtests.test(msg="unreadable_event_is_swallowed"):
        dialog = make_dialog()
        glass.let_the_user_resize(dialog)

        class Unreadable:
            def type(self):
                raise RuntimeError("event is gone")

        assert dialog._spacr_resizer.eventFilter(dialog, Unreadable()) is False

    with subtests.test(msg="foreign_widget_or_gone_window_answers_nothing"):
        dialog = make_dialog()
        glass.let_the_user_resize(dialog)
        watcher = dialog._spacr_resizer
        other = QWidget()
        qtbot.addWidget(other)
        assert watcher.eventFilter(other, _move((1, 1))) is False
        watcher._window = None
        assert watcher.eventFilter(dialog, _move((1, 1))) is False


def test_dragging_the_dialog_background_moves_the_window(make_dialog, qtbot, monkeypatch, subtests):
    """_DragByBackground moves the window by the drag, ignores controls, and forgets a failed grab."""
    with subtests.test(msg="drag_moves_until_release"):
        dialog = make_dialog()
        dialog.move(100, 100)
        dragger = glass._DragByBackground(dialog)
        dragger.eventFilter(dialog, _press((160, 120), glob=(500, 500)))
        assert dragger._grab is not None
        dragger.eventFilter(dialog, _move((160, 120), glob=(540, 530), buttons=Qt.LeftButton))
        assert (dialog.pos().x(), dialog.pos().y()) == (140, 130)
        dragger.eventFilter(dialog, _release())
        assert dragger._grab is None
        dragger.eventFilter(dialog, _move((160, 120), glob=(700, 700)))
        assert (dialog.pos().x(), dialog.pos().y()) == (140, 130)

    with subtests.test(msg="press_on_a_control_is_left_alone"):
        dialog = make_dialog()
        QPushButton("Save", dialog).setGeometry(10, 10, 80, 24)
        dragger = glass._DragByBackground(dialog)
        dragger.eventFilter(dialog, _press((20, 20)))
        assert dragger._grab is None

    with subtests.test(msg="a_failed_move_forgets_the_grab"):
        dialog = make_dialog()
        dragger = glass._DragByBackground(dialog)
        dragger.eventFilter(dialog, _press((160, 120)))
        assert dragger._grab is not None
        with monkeypatch.context() as mp:
            mp.setattr(type(dialog), "move", _raises("no window manager"))
            dragger.eventFilter(dialog, _move((160, 120), glob=(600, 600), buttons=Qt.LeftButton))
        assert dragger._grab is None

    with subtests.test(msg="foreign_widget_or_gone_dialog_answers_nothing"):
        dialog = make_dialog()
        dragger = glass._DragByBackground(dialog)
        other = QWidget()
        qtbot.addWidget(other)
        assert dragger.eventFilter(other, _press((1, 1))) is False
        dragger._dialog = None
        assert dragger.eventFilter(dialog, _press((1, 1))) is False


@contextmanager
def _window_sheet():
    """A per-window application sheet in force, removed afterwards."""
    app = QApplication.instance()
    app.setStyleSheet("")
    theme.apply_stylesheet_per_window(app, "QLabel { color: rgb(11, 22, 33); }")
    try:
        yield app
    finally:
        theme.apply_stylesheet_per_window(app, "")
        app.setStyleSheet("")


def test_the_card_rule_is_added_to_the_dialog_sheet_once(make_dialog, monkeypatch, subtests):
    """_paint_nothing_behind_the_card appends NO_BACKGROUND once, in one styling pass, or refuses."""
    with subtests.test(msg="rule_joins_the_own_sheet_once"):
        dialog = make_dialog()
        dialog.setStyleSheet("QLabel { color: red; }")
        assert glass._paint_nothing_behind_the_card(dialog) is True
        assert "QLabel { color: red; }" in dialog.styleSheet()
        assert glass.NO_BACKGROUND in dialog.styleSheet()
        assert glass._paint_nothing_behind_the_card(dialog) is False

    with subtests.test(msg="dialog_refusing_the_stylesheet"):
        dialog = make_dialog()
        with monkeypatch.context() as mp:
            mp.setattr(type(dialog), "styleSheet", _raises("gone"))
            assert glass._paint_nothing_behind_the_card(dialog) is False

    with subtests.test(msg="window_sheet_and_rule_in_one_styling_pass"):
        dialog = make_dialog()
        with _window_sheet() as app:
            QLabel("row", dialog)
            dialog.setStyleSheet("QLabel { font-weight: bold; }")
            calls = []
            real = QWidget.setStyleSheet

            def counting(self, sheet):
                if self is dialog:
                    calls.append(sheet)
                return real(self, sheet)

            QWidget.setStyleSheet = counting
            try:
                assert glass._paint_nothing_behind_the_card(dialog) is True
                dialog.show()
                app.processEvents()
            finally:
                QWidget.setStyleSheet = real
            assert len(calls) == 1
            sheet = dialog.styleSheet()
            assert "rgb(11, 22, 33)" in sheet
            assert "QLabel { font-weight: bold; }" in sheet
            assert glass.NO_BACKGROUND in sheet
            assert glass._paint_nothing_behind_the_card(dialog) is False


def test_the_rim_gets_room_and_a_buttonless_dialog_gets_a_close_hint(make_dialog, monkeypatch, subtests):
    """_make_room_for_the_rim adds RIM_ROOM to the margins; _say_how_to_close_it adds one Escape hint when needed."""
    with subtests.test(msg="rim_no_layout"):
        assert glass._make_room_for_the_rim(make_dialog(layout=None)) is False

    with subtests.test(msg="rim_band_added_to_own_margins"):
        dialog = make_dialog()
        dialog.layout().setContentsMargins(4, 4, 4, 4)
        assert glass._make_room_for_the_rim(dialog) is True
        margins = dialog.layout().contentsMargins()
        assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == \
            (4 + glass.RIM_ROOM,) * 4

    with subtests.test(msg="rim_layout_refusing_margins"):
        dialog = make_dialog()
        with monkeypatch.context() as mp:
            mp.setattr(type(dialog.layout()), "setContentsMargins", _raises("fixed layout"))
            assert glass._make_room_for_the_rim(dialog) is False

    with subtests.test(msg="hint_for_a_dialog_with_no_button"):
        dialog = make_dialog()
        assert glass._say_how_to_close_it(dialog) is True
        hints = [w for w in dialog.findChildren(QLabel) if "Escape" in w.text()]
        assert len(hints) == 1
        assert dialog.property(glass.CLOSE_HINT) is True
        assert glass._say_how_to_close_it(dialog) is False

    with subtests.test(msg="no_hint_with_a_button_box"):
        dialog = make_dialog()
        dialog.layout().addWidget(QDialogButtonBox(QDialogButtonBox.Ok, dialog))
        assert glass._say_how_to_close_it(dialog) is False

    with subtests.test(msg="no_hint_for_a_sideways_layout"):
        assert glass._say_how_to_close_it(make_dialog(layout=QHBoxLayout)) is False

    with subtests.test(msg="hint_that_cannot_be_built"):
        dialog = make_dialog()
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "spacr.qt.i18n", None)
            assert glass._say_how_to_close_it(dialog) is False
        assert not dialog.property(glass.CLOSE_HINT)


class _Fractal(QWidget):
    """Stands in for a fractal backdrop: pause/resume, no set_animating."""

    def __init__(self, parent):
        super().__init__(parent)
        self.calls = []

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def lower(self):
        self.calls.append("lower")
        super().lower()


def test_the_popup_backdrop_follows_its_own_preference(make_dialog, monkeypatch, subtests):
    """_install_the_backdrop installs, pauses or resumes per the popup-backdrop preference alone."""
    def recording_install(mp, dialog):
        installed = []
        mp.setattr(ambient, "install_ambient",
                   lambda dlg, **kwargs: installed.append(kwargs) or QWidget(dlg))
        return installed

    for ambient_on in (False, True):
        with subtests.test(msg=f"off_stays_off_with_main_animation_{ambient_on}"):
            dialog = make_dialog()
            with monkeypatch.context() as mp:
                mp.setattr(preferences, "get_ambient_enabled", lambda: ambient_on)
                mp.setattr(preferences, "get_popup_backdrop", lambda: "off")
                installed = recording_install(mp, dialog)
                assert glass._install_the_backdrop(dialog) is None
            assert installed == []

    for ambient_on in (False, True):
        with subtests.test(msg=f"drift_installs_with_main_animation_{ambient_on}"):
            dialog = make_dialog()
            with monkeypatch.context() as mp:
                mp.setattr(preferences, "get_ambient_enabled", lambda: ambient_on)
                mp.setattr(preferences, "get_popup_backdrop", lambda: "drift")
                installed = recording_install(mp, dialog)
                assert isinstance(glass._install_the_backdrop(dialog), QWidget)
            assert len(installed) == 1
            assert installed[0]["theme"] == "drift"
            assert installed[0]["corner_radius"] == glass.CARD_RADIUS

    with subtests.test(msg="unreadable_preference_invents_nothing"):
        dialog = make_dialog()
        with monkeypatch.context() as mp:
            mp.setattr(preferences, "get_popup_backdrop", _raises("no store"))
            installed = recording_install(mp, dialog)
            assert glass._install_the_backdrop(dialog) is None
        assert installed == []

    with subtests.test(msg="no_ambient_engine_still_a_card"):
        dialog = make_dialog()
        with monkeypatch.context() as mp:
            mp.setattr(preferences, "get_ambient_enabled", lambda: True)
            mp.setattr(preferences, "get_popup_backdrop", lambda: "drift")
            mp.setitem(sys.modules, "spacr.qt.widgets.ambient", None)
            assert glass._install_the_backdrop(dialog) is None

    with subtests.test(msg="off_pauses_and_hides_a_fractal"):
        dialog = make_dialog()
        backdrop = _Fractal(dialog)
        backdrop.show()
        dialog._spacr_popup_backdrop = backdrop
        with monkeypatch.context() as mp:
            mp.setattr(preferences, "get_popup_backdrop", lambda: "off")
            assert glass._install_the_backdrop(dialog) is None
        assert backdrop.calls == ["pause"]
        assert backdrop.isHidden()

    with subtests.test(msg="same_fractal_theme_resumes_the_existing_backdrop"):
        dialog = make_dialog()
        backdrop = _Fractal(dialog)
        backdrop.hide()
        dialog._spacr_popup_backdrop = backdrop
        with monkeypatch.context() as mp:
            mp.setattr(preferences, "get_popup_backdrop", lambda: ambient.SPACEOUT_THEME)
            built = recording_install(mp, dialog)
            assert glass._install_the_backdrop(dialog) is backdrop
        assert built == []
        assert backdrop.calls == ["resume", "lower"]
        assert not backdrop.isHidden()
        assert dialog._spacr_popup_backdrop is backdrop
        assert dialog.property("spacrIndependentBackdrop") is True


class _StandInApplication(QObject):
    """An application-shaped object recording filters put on and taken off.

    A stand-in rather than the live ``QApplication``: a filter left on the real
    one would glass every dialog built by every later test.
    """

    def __init__(self, refuse_removal: bool = False):
        super().__init__()
        self.installed = []
        self.removed = []
        self._refuse_removal = refuse_removal

    @property
    def glass_filters(self):
        return [item for item in self.installed if isinstance(item, glass._GlassInstaller)]

    def installEventFilter(self, event_filter):     # noqa: N802 - Qt naming
        self.installed.append(event_filter)

    def removeEventFilter(self, event_filter):      # noqa: N802 - Qt naming
        if self._refuse_removal:
            raise RuntimeError("that filter is already gone")
        self.removed.append(event_filter)


def _forget_installation():
    glass.uninstall_glass_everywhere()
    glass._INSTALLED = None
    glass._INSTALLED_APP = None


def test_install_glass_everywhere_keeps_exactly_one_filter_and_owner(qapp, monkeypatch, caplog, subtests):
    """install_glass_everywhere installs once per application and survives refusing, orphaned or absent owners."""
    caplog.set_level(logging.DEBUG, logger="spacr.qt.glass")

    with subtests.test(msg="live_application_installs_once"):
        _forget_installation()
        assert glass.install_glass_everywhere() is True
        assert glass.install_glass_everywhere() is False
        assert glass.uninstall_glass_everywhere() is True
        assert glass.uninstall_glass_everywhere() is False

    with subtests.test(msg="no_application_to_install_on"):
        _forget_installation()
        with monkeypatch.context() as mp:
            mp.setattr(QApplication, "instance", staticmethod(lambda: None))
            assert glass.install_glass_everywhere() is False
        assert glass._INSTALLED is None

    with subtests.test(msg="filter_that_will_not_build_leaves_nothing"):
        _forget_installation()
        with monkeypatch.context() as mp:
            mp.setattr(glass, "_GlassInstaller", _raises("no application thread"))
            assert glass.install_glass_everywhere() is False
        assert glass._INSTALLED is None

    with subtests.test(msg="old_owner_refusing_removal_does_not_block_the_new_one"):
        _forget_installation()
        caplog.clear()
        stubborn = _StandInApplication(refuse_removal=True)
        assert glass.install_glass_everywhere(stubborn) is True
        old_filter = glass._INSTALLED
        assert stubborn.glass_filters == [old_filter]
        successor = _StandInApplication()
        assert glass.install_glass_everywhere(successor) is True
        assert glass._INSTALLED is not old_filter
        assert successor.glass_filters == [glass._INSTALLED]
        assert glass._INSTALLED_APP is successor
        assert "the old glass filter would not come off" in caplog.text

    with subtests.test(msg="orphaned_filter_yields_and_a_remembered_owner_gives_it_back"):
        _forget_installation()
        orphan = _StandInApplication()
        assert glass.install_glass_everywhere(orphan) is True
        orphaned_filter = glass._INSTALLED
        glass._INSTALLED_APP = None
        successor = _StandInApplication()
        assert glass.install_glass_everywhere(successor) is True
        assert orphan.removed == []
        assert glass._INSTALLED is not orphaned_filter
        assert successor.glass_filters == [glass._INSTALLED]
        assert glass._INSTALLED_APP is successor
        successors_filter = glass._INSTALLED
        third = _StandInApplication()
        assert glass.install_glass_everywhere(third) is True
        assert successor.removed == [successors_filter]
        assert third.glass_filters == [glass._INSTALLED]

    with subtests.test(msg="whole_messy_lifecycle_ends_on_one_pair"):
        _forget_installation()
        caplog.clear()
        first = _StandInApplication()
        assert glass.install_glass_everywhere(first) is True
        glass._INSTALLED_APP = None
        second = _StandInApplication(refuse_removal=True)
        assert glass.install_glass_everywhere(second) is True
        assert first.removed == []
        assert second.glass_filters == [glass._INSTALLED]
        third = _StandInApplication()
        assert glass.install_glass_everywhere(third) is True
        assert "the old glass filter would not come off" in caplog.text
        thirds_filter = glass._INSTALLED
        assert third.glass_filters == [thirds_filter]
        fourth = _StandInApplication()
        assert glass.install_glass_everywhere(fourth) is True
        assert third.removed == [thirds_filter]
        with monkeypatch.context() as mp:
            mp.setattr(QApplication, "instance", staticmethod(lambda: None))
            glass._INSTALLED_APP = None
            assert glass.uninstall_glass_everywhere() is True
        assert fourth.removed == []
        assert glass._INSTALLED is None and glass._INSTALLED_APP is None
        fifth = _StandInApplication()
        assert glass.install_glass_everywhere(fifth) is True
        assert fifth.glass_filters == [glass._INSTALLED]
        assert glass.uninstall_glass_everywhere() is True
        assert fifth.removed == fifth.glass_filters


def test_uninstall_glass_everywhere_always_forgets_the_filter(qapp, monkeypatch, caplog, subtests):
    """uninstall_glass_everywhere forgets the pair even when removal, the theme or the application is gone."""
    with subtests.test(msg="filter_that_will_not_come_off_is_forgotten"):
        _forget_installation()
        glass.install_glass_everywhere()
        installed = glass._INSTALLED
        from spacr.qt.gil_priority import _stop_watching_application_events as really_remove
        try:
            with monkeypatch.context() as mp:
                mp.setattr(glass, "_stop_watching_application_events", _raises("already gone"))
                assert glass.uninstall_glass_everywhere() is True
            assert glass._INSTALLED is None
        finally:
            really_remove(qapp, installed)

    with subtests.test(msg="theme_keeping_its_claim_is_logged"):
        _forget_installation()
        removed = []
        installed = object()
        caplog.clear()
        with monkeypatch.context() as mp:
            mp.setattr(glass, "_INSTALLED", installed)
            mp.setattr(glass, "_INSTALLED_APP", qapp)
            mp.setattr(glass, "_stop_watching_application_events",
                       lambda app, target: removed.append((app, target)))
            mp.setattr(theme, "_stop_letting_a_decorator_sheet",
                       _raises("the theme is already torn down"))
            with caplog.at_level(logging.DEBUG, logger=glass.LOG.name):
                assert glass.uninstall_glass_everywhere() is True
            assert removed == [(qapp, installed)]
            assert glass._INSTALLED is None and glass._INSTALLED_APP is None
        assert "the theme kept the glass filter's claim" in caplog.text

    with subtests.test(msg="no_application_and_no_owner"):
        _forget_installation()
        stray = _StandInApplication()
        with monkeypatch.context() as mp:
            mp.setattr(QApplication, "instance", staticmethod(lambda: None))
            glass._INSTALLED = glass._GlassInstaller(stray)
            glass._INSTALLED_APP = None
            assert glass.uninstall_glass_everywhere() is True
            assert stray.removed == []
            assert glass._INSTALLED is None and glass._INSTALLED_APP is None
            assert glass.uninstall_glass_everywhere() is False
            owner = _StandInApplication()
            assert glass.install_glass_everywhere(owner) is True
            owners_filter = glass._INSTALLED
            assert glass.uninstall_glass_everywhere() is True
            assert owner.removed == [owners_filter]
