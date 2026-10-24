"""Lesson 05 clip 14 and lesson 08 stalls: glassing a dialog restyled it three times.

The capture host's stall watcher caught the GUI thread in
``glass.make_frameless`` under the theme's window filter, 0.36-0.39 s at a
time. The dialog was the kept Preferences window, built ahead by the
warm-up and polished by Qt's posted ``PolishRequest`` before it was ever
shown: the window sheet went on (one full restyle), the glass added its
own rule (a second), the frameless flags made Qt repolish every styled
widget (a third) and each transparent container was polished once more,
over every widget of every tab.

Now the glass holds the dialog's sheet while it decorates it and puts the
same text on once, last, and Preferences sends its unseen tabs away before
that first polish as it does before a show. These tests pin the look
(pixel for pixel) and the cost (self-calibrated against the old path).
"""
from __future__ import annotations

import contextlib
import time

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QFrame, QLabel,
                               QLineEdit, QMainWindow, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from spacr.qt import gui_scale, preferences, theme
from spacr.qt.widgets import glass

pytestmark = pytest.mark.qt


@pytest.fixture
def glassed_app(monkeypatch):
    """A per-window dark sheet in force and the glass filter installed."""
    app = QApplication.instance()
    from spacr.qt.widgets import setup_card

    monkeypatch.setattr(glass, "_install_the_backdrop", lambda dialog: None)
    monkeypatch.setattr(setup_card.SetupCard, "_tick", lambda self: None)
    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))
    installed_here = glass.install_glass_everywhere(app)
    yield app
    if installed_here:
        glass.uninstall_glass_everywhere(app)
    theme._forget_window_stylesheets(app)


@pytest.fixture
def owner(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.resize(900, 700)
    return window


def _dialog(parent, rows: int = 40) -> QDialog:
    """A settings-like dialog: a form of rows in a scroll area, and buttons."""
    dialog = QDialog(parent)
    dialog.setObjectName("GlassProbe")
    outer = QVBoxLayout(dialog)
    outer.addWidget(QLabel("Settings that need a decision", dialog))
    scroll = QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    page = QWidget()
    form = QFormLayout(page)
    for row in range(rows):
        holder = QFrame(page)
        line = QVBoxLayout(holder)
        line.setContentsMargins(0, 0, 0, 0)
        if row % 3 == 0:
            editor = QComboBox(holder)
            editor.addItems(["first", "second", "third"])
        elif row % 3 == 1:
            editor = QLineEdit(f"value {row}", holder)
        else:
            editor = QCheckBox(f"switch {row}", holder)
        line.addWidget(editor)
        form.addRow(QLabel(f"Row {row}", page), holder)
    scroll.setWidget(page)
    outer.addWidget(scroll)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                               dialog)
    outer.addWidget(buttons)
    extra = QPushButton("Apply", dialog)
    outer.addWidget(extra)
    dialog.resize(520, 560)
    return dialog


def _the_old_way(monkeypatch) -> None:
    """Glass as before: no hold, and the theme filter sheets the dialog itself."""
    @contextlib.contextmanager
    def _no_hold(window):
        yield window

    monkeypatch.setattr(theme, "_one_sheet_for", _no_hold)
    monkeypatch.setattr(theme, "_DECORATORS_THAT_SHEET", [])


def _shown_and_grabbed(dialog):
    dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    dialog.show()
    for _ in range(3):
        QApplication.processEvents()
    image = dialog.grab().toImage()
    state = (dialog.styleSheet(), int(dialog.windowFlags()), dialog.size(),
             [(type(w).__name__, w.geometry().getRect(),
               w.palette().color(w.backgroundRole()).name(),
               w.palette().color(w.foregroundRole()).name())
              for w in dialog.findChildren(QWidget)])
    dialog.hide()
    return image, state


def test_a_glassed_dialog_looks_exactly_as_before(glassed_app, owner,
                                                   monkeypatch):
    new_image, new_state = _shown_and_grabbed(_dialog(owner))
    _the_old_way(monkeypatch)
    old_image, old_state = _shown_and_grabbed(_dialog(owner))
    assert glass.NO_BACKGROUND in new_state[0]
    assert new_state == old_state
    assert new_image == old_image


def test_the_dialog_is_sheeted_once_whoever_hears_the_polish_first(
        glassed_app, owner):
    """The theme filter re-registered after the glass is asked first."""
    theme.apply_stylesheet_per_window(glassed_app, theme.window_stylesheet())
    dialog = _dialog(owner)
    sheets = []
    real = dialog.setStyleSheet
    dialog.setStyleSheet = lambda text: sheets.append(len(text)) or real(text)
    dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    dialog.show()
    QApplication.processEvents()
    assert dialog.property(glass.GLASSED)
    assert len(sheets) == 1, sheets
    assert glass.NO_BACKGROUND in dialog.styleSheet()
    assert not dialog.property(theme._SHEET_HELD)
    dialog.hide()


def _glass_seconds(owner, rows: int) -> float:
    dialog = _dialog(owner, rows)
    started = time.perf_counter()
    dialog.ensurePolished()
    gui_scale._flush_pending_sheets()
    elapsed = time.perf_counter() - started
    assert dialog.property(glass.GLASSED)
    dialog.deleteLater()
    QApplication.processEvents()
    return elapsed


def test_glassing_costs_one_restyle_not_three(glassed_app, owner,
                                              monkeypatch):
    """Self-calibrated: the same dialog glassed the new way and the old."""
    _glass_seconds(owner, 120)
    new = min(_glass_seconds(owner, 120) for _ in range(3))
    _the_old_way(monkeypatch)
    old = min(_glass_seconds(owner, 120) for _ in range(3))
    assert new < old * 0.75, (new, old)


def test_a_held_sheet_still_goes_on_when_the_glass_fails(glassed_app, owner,
                                                         monkeypatch):
    def _broken(dialog):
        raise RuntimeError("no card")

    monkeypatch.setattr(glass, "_glass_held", _broken)
    dialog = _dialog(owner, 3)
    with pytest.raises(RuntimeError):
        glass.glass(dialog)
    assert not dialog.property(theme._SHEET_HELD)
    assert dialog.styleSheet().startswith(theme.window_stylesheet())


def test_a_tagged_container_is_repolished_when_no_window_sheet_is_in_force(
        qtbot):
    assert theme.window_stylesheet() is None
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.setStyleSheet(
        'QWidget[spacrTransparent="true"] { background-color: rgb(1, 2, 3); }')
    holder = QWidget(dialog)
    dialog.ensurePolished()
    holder.ensurePolished()
    before = holder.palette().color(QPalette.Window).name()
    with theme._one_sheet_for(dialog):
        theme.make_transparent(holder)
        assert dialog.property(theme._SHEET_OWES_A_POLISH)
        assert holder.palette().color(QPalette.Window).name() == before
    assert holder.property(theme.TRANSPARENT_PROPERTY) is True
    assert not dialog.property(theme._SHEET_OWES_A_POLISH)
    assert holder.palette().color(QPalette.Window).name() == "#010203"


def _warm_preferences(owner, ahead: bool = True):
    """Build Preferences as the warm-up does and let Qt polish it."""
    dialog = preferences.PreferencesDialog(owner)
    dialog._pages_leave_at_the_polish = ahead
    started = time.perf_counter()
    QApplication.processEvents()
    gui_scale._flush_pending_sheets()
    return dialog, time.perf_counter() - started


def test_preferences_built_ahead_polishes_only_its_open_tab(glassed_app,
                                                            owner):
    dialog, _ = _warm_preferences(owner)
    try:
        assert dialog.testAttribute(Qt.WA_WState_Polished)
        assert dialog.property(glass.GLASSED)
        assert dialog._pages_away, "the unseen tabs stayed for the polish"
        assert not dialog.isVisible()
    finally:
        dialog.deleteLater()
        QApplication.processEvents()


def test_preferences_polish_ahead_is_cheaper_than_before(glassed_app, owner,
                                                         monkeypatch):
    """Self-calibrated: the warm-up's polish with and without the early send-off."""
    _warm_preferences(owner)[0].deleteLater()
    new = []
    for _ in range(2):
        dialog, seconds = _warm_preferences(owner)
        new.append(seconds)
        dialog.deleteLater()
    QApplication.processEvents()
    old = []
    for _ in range(2):
        dialog, seconds = _warm_preferences(owner, ahead=False)
        old.append(seconds)
        assert not dialog._pages_away
        dialog.deleteLater()
    QApplication.processEvents()
    assert min(new) < min(old) * 0.75, (new, old)


def _every_tab_grabbed(dialog):
    """Show ``dialog`` and grab it on each tab, with what is visible there."""
    dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    dialog.show()
    QApplication.processEvents()
    tabs = dialog._page_tabs
    seen = [dialog.styleSheet(), int(dialog.windowFlags()), dialog.size()]
    for index in range(tabs.count()):
        tabs.setCurrentIndex(index)
        for _ in range(3):
            QApplication.processEvents()
        seen.append((index, dialog.grab().toImage(), [
            (type(w).__name__, w.mapTo(dialog, w.rect().topLeft()).toTuple(),
             w.size().toTuple(),
             w.palette().color(w.backgroundRole()).name(),
             w.palette().color(w.foregroundRole()).name())
            for w in QWidget.findChildren(dialog, QWidget)
            if w.isVisible()]))
    dialog.hide()
    return seen


def test_preferences_looks_the_same_opened_after_a_polish_ahead(
        glassed_app, owner, monkeypatch):
    """Every tab, pixel for pixel, against a window polished at its show."""
    warm, _ = _warm_preferences(owner)
    assert warm._pages_away
    warm_seen = _every_tab_grabbed(warm)
    cold, _ = _warm_preferences(owner, ahead=False)
    assert not cold._pages_away
    cold_seen = _every_tab_grabbed(cold)
    assert warm_seen[:3] == cold_seen[:3]
    assert len(warm_seen) == len(cold_seen) > 4
    for (index, warm_image, warm_widgets), (_, cold_image, cold_widgets) in zip(
            warm_seen[3:], cold_seen[3:]):
        assert warm_widgets == cold_widgets, index
        assert warm_image == cold_image, index
    warm.deleteLater()
    cold.deleteLater()
    QApplication.processEvents()


def test_preferences_built_by_anyone_else_keeps_its_pages_until_the_show(
        glassed_app, owner):
    """The help search index walks every tab after letting Qt polish."""
    dialog = preferences.PreferencesDialog(owner)
    QApplication.processEvents()
    try:
        assert dialog.testAttribute(Qt.WA_WState_Polished)
        assert not dialog._pages_away
    finally:
        dialog.deleteLater()
        QApplication.processEvents()


def test_the_warm_up_asks_for_the_pages_to_leave_at_the_polish():
    source = open(preferences.__file__, encoding="utf-8").read()
    start = source.index("def _warm_one_slice(")
    body = source[start:source.index("\ndef ", start + 1)]
    assert "_pages_leave_at_the_polish = True" in body


def test_the_window_filter_skips_only_what_a_decorator_claims():
    probe = QWidget()
    assert not theme._a_decorator_sheets_it(probe)
    probe.setProperty(theme._SHEET_HELD, True)
    assert theme._a_decorator_sheets_it(probe)
    probe.setProperty(theme._SHEET_HELD, None)
    claim = lambda widget: widget is probe
    theme._let_a_decorator_sheet(claim)
    theme._let_a_decorator_sheet(claim)
    try:
        assert theme._DECORATORS_THAT_SHEET.count(claim) == 1
        assert theme._a_decorator_sheets_it(probe)
        assert not theme._a_decorator_sheets_it(QWidget())
    finally:
        theme._stop_letting_a_decorator_sheet(claim)
    assert claim not in theme._DECORATORS_THAT_SHEET
    assert QEvent.Polish in theme._SHEETING_MOMENTS
