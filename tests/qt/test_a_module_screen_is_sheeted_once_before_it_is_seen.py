"""A module screen is repolished once before its first paint, not four times.

ITEMS 284 AND 380. Opening a module froze the interface for its whole build,
and the largest single share of that freeze was the same screen being
restyled over and over before anybody could see it. Counted on one
Regression open (1,385 widgets) by wrapping ``QWidget.setStyleSheet``:

    1. ``AppScreen._sync_page_palette``, still inside ``__init__``, put
       the whole window sheet on the parentless screen             668 ms
    2. ``MainWindow._theme_screen`` appended the late widget blocks  829 ms
    3. ``QStackedWidget.addWidget`` reparented a screen that now
       carried a sheet, and Qt repolishes the whole subtree for that
    4. the page's ``Show`` put the same text on again               806 ms

Every one of those repolished all 1,385 widgets. Only the fourth is ever
seen, so the first two now write the sheet down instead of applying it,
the third costs nothing because a widget without a sheet joining a parent
without one is not restyled by Qt, and the fourth applies the whole debt
at once. Regression's worst stall went from about 7.5 s to 4.8 s and
Mask's from about 4.0 s to 2.3 s on the startup benchmark's own path.

THE OTHER HALF IS THAT NOTHING GOES UNSHEETED, which is what the unit
tests below hold: the rule written down lands on the first show whoever
the widget ends up belonging to, a late block lands with it, a rule the
widget set on itself in between is kept exactly as before, and a widget
that can already be seen is sheeted at once.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QEvent, QObject                     # noqa: E402
from PySide6.QtGui import QPalette                             # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel,           # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt import theme                                     # noqa: E402

#: One colour nothing else uses, so a widget wearing it is unambiguous.
SHEET = "QLabel { color: rgb(11, 22, 33); }"
SHEET_HEX = "#0b1621"
OWN = "QWidget#DeferredProbe { background-color: #123456; }"


@pytest.fixture
def sheeted_app(qtbot):
    """A QApplication carrying the per-window sheet and its filter."""
    app = QApplication.instance()
    theme.apply_stylesheet_per_window(app, SHEET)
    yield app


@pytest.fixture
def registry_sandbox():
    """Put the widget-block registry back as it was."""
    saved = dict(theme._WIDGET_QSS)
    try:
        yield
    finally:
        theme._WIDGET_QSS.clear()
        theme._WIDGET_QSS.update(saved)


def _pump(app, rounds: int = 20) -> None:
    for _ in range(rounds):
        app.processEvents()


def _resolved(label) -> str:
    label.ensurePolished()
    _pump(QApplication.instance(), 5)
    return label.palette().color(QPalette.WindowText).name()


def _a_screen_being_built(qtbot):
    """A parentless widget with a label in it, not yet shown or created."""
    widget = QWidget()
    widget.setObjectName("DeferredProbe")
    qtbot.addWidget(widget)
    QVBoxLayout(widget).addWidget(QLabel("probe", widget))
    return widget


def test_a_rule_given_mid_construction_is_written_down_not_applied(
        sheeted_app, qtbot):
    """No ``setStyleSheet`` on a widget nobody can see yet."""
    widget = _a_screen_being_built(qtbot)

    theme.set_a_sheeted_widgets_own_rule(widget, OWN)

    assert widget.styleSheet() == "", (
        "a widget still being built was restyled for a sheet it will be "
        "given again at its first show")
    assert widget.property(theme._SHEET_TARGET), (
        "the widget was not marked, so nothing would pay the debt at its "
        "show")
    assert widget.property(theme._WINDOW_SHEET_WAITS)


def test_the_written_down_rule_lands_with_the_sheet_on_the_first_show(
        sheeted_app, qtbot):
    widget = _a_screen_being_built(qtbot)
    theme.set_a_sheeted_widgets_own_rule(widget, OWN)

    widget.show()
    _pump(sheeted_app)

    sheet = widget.styleSheet()
    assert sheet.startswith(SHEET), "the window sheet never arrived"
    assert "#123456" in sheet, "the widget's own rule was lost"
    assert _resolved(widget.findChild(QLabel)) == SHEET_HEX
    assert not widget.property(theme._WINDOW_SHEET_WAITS), (
        "the debt is paid but the widget still says it is owed")


def test_a_page_parented_before_its_show_is_sheeted_on_that_show(
        sheeted_app, qtbot):
    """The module-screen path: built, parented into a bare host, shown."""
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    host.show()
    _pump(sheeted_app)

    widget = _a_screen_being_built(qtbot)
    theme.set_a_sheeted_widgets_own_rule(widget, OWN)
    widget.hide()
    layout.addWidget(widget)
    assert widget.styleSheet() == "", "joining the host applied the debt"

    widget.show()
    _pump(sheeted_app)
    assert widget.styleSheet().startswith(SHEET)
    assert "#123456" in widget.styleSheet()
    assert _resolved(widget.findChild(QLabel)) == SHEET_HEX


def test_a_late_block_joins_the_debt_and_lands_with_the_sheet(
        sheeted_app, qtbot, registry_sandbox):
    """``ensure_widget_qss_applied`` does not restyle a root that is owed."""
    theme.register_widget_qss(
        "DeferredLateBlock",
        lambda palette, opacity: "QLabel#DeferredLateBlock { color: #654321; }")
    try:
        widget = _a_screen_being_built(qtbot)
        theme.set_a_sheeted_widgets_own_rule(widget, OWN)

        applied = theme.ensure_widget_qss_applied(root=widget)

        assert applied is False
        assert widget.styleSheet() == "", (
            "the late blocks restyled a screen that is sheeted at its show")
        suffix = getattr(widget, theme._LOCAL_WIDGET_QSS_ATTRIBUTE, "")
        assert "DeferredLateBlock" in suffix, "the block was not kept"

        widget.show()
        _pump(sheeted_app)
        sheet = widget.styleSheet()
        assert sheet.startswith(SHEET)
        assert sheet.endswith(suffix), (
            "the late block did not arrive with the sheet it was waiting for")
    finally:
        theme.unregister_widget_qss("DeferredLateBlock")


def test_a_rule_the_widget_set_itself_meanwhile_wins_as_it_always_did(
        sheeted_app, qtbot):
    """Same answer as when the sheet was applied at once.

    Applied at once, the widget carried sheet + OWN; a later direct
    ``setStyleSheet(mine)`` replaced that, and the next sheeting read
    ``mine`` back as the widget's own rule because the digest no longer
    matched. The written-down path records the digest of what the widget
    carried when the rule was given, so the same thing happens.
    """
    widget = _a_screen_being_built(qtbot)
    theme.set_a_sheeted_widgets_own_rule(widget, OWN)
    mine = "QWidget#DeferredProbe { border: 1px solid #0a0b0c; }"
    widget.setStyleSheet(mine)

    widget.show()
    _pump(sheeted_app)

    assert widget.styleSheet() == SHEET + mine


def test_forgetting_the_sheet_pays_the_debt_with_the_rule_alone(
        sheeted_app, qtbot):
    """Test isolation takes the sheet down; the own rule must survive it."""
    widget = _a_screen_being_built(qtbot)
    theme.set_a_sheeted_widgets_own_rule(widget, OWN)

    theme._forget_window_stylesheets(sheeted_app)

    assert widget.styleSheet() == OWN
    assert not widget.property(theme._WINDOW_SHEET_WAITS)


def test_with_no_sheet_in_force_the_rule_lands_at_once(qtbot):
    app = QApplication.instance()
    theme._forget_window_stylesheets(app)
    widget = _a_screen_being_built(qtbot)

    theme.set_a_sheeted_widgets_own_rule(widget, OWN)

    assert widget.styleSheet() == OWN, (
        "nothing will sheet this widget at its show, so the rule has to "
        "land now")


def test_a_widget_already_on_show_is_sheeted_at_once(sheeted_app, qtbot):
    widget = _a_screen_being_built(qtbot)
    widget.show()
    _pump(sheeted_app)

    theme.set_a_sheeted_widgets_own_rule(widget, OWN)

    assert widget.styleSheet().startswith(SHEET)
    assert "#123456" in widget.styleSheet()


class _StyleChanges(QObject):
    """Counts ``StyleChange`` per receiver, keyed by the C++ object."""

    def __init__(self):
        super().__init__()
        self.counts = {}

    def eventFilter(self, watched, event):  # noqa: N802 - Qt naming
        if event.type() == QEvent.StyleChange:
            from shiboken6 import getCppPointer
            key = getCppPointer(watched)[0]
            self.counts[key] = self.counts.get(key, 0) + 1
        return False


@pytest.mark.slow
@pytest.mark.parametrize("key", ["measure", "classify_merged"])
def test_opening_a_module_repolishes_its_screen_once(sheeted_app, qtbot, key):
    """THE RATCHET. Counted, not timed: a loaded runner cannot move it.

    Before this change the screen root received a ``StyleChange`` for each
    of the four restyles in this file's header. A whole-subtree restyle
    reaches the root every time, so the root's count IS the number of
    times every widget on the screen was repolished before first paint.
    """
    from shiboken6 import getCppPointer

    from spacr.qt import register_self_registering_modules
    from spacr.qt.app import MainWindow

    register_self_registering_modules()
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    _pump(sheeted_app, 60)
    theme.apply_stylesheet_per_window(sheeted_app, SHEET)
    _pump(sheeted_app)

    counter = _StyleChanges()
    sheeted_app.installEventFilter(counter)
    try:
        window._on_nav_selected(key)
        _pump(sheeted_app, 60)
    finally:
        sheeted_app.removeEventFilter(counter)

    screen = window._screens[key]
    restyles = counter.counts.get(getCppPointer(screen)[0], 0)
    assert restyles == 1, (
        f"the {key} screen was restyled {restyles} times before its "
        "first paint; each one repolishes every widget on it")
    assert screen.styleSheet().startswith(SHEET), (
        "the screen was restyled once, but not with the sheet")


def test_classify_controls_exist_before_the_first_screen_sheet(
        qtbot, monkeypatch):
    from spacr.qt.app import MainWindow
    from spacr.qt.preferences import apply_preferences_to_app

    apply_preferences_to_app(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()
    observed = []
    original = theme._sheet_one_window

    def sheet(widget):
        if getattr(widget, "app_key", None) == "classify_merged":
            observed.append((getattr(widget, "_fold_strip", None) is not None,
                             getattr(widget, "_flowview_section", None) is not None))
        return original(widget)

    monkeypatch.setattr(theme, "_sheet_one_window", sheet)
    window._on_nav_selected("classify_merged")
    _pump(QApplication.instance())
    assert observed and all(strip and flow for strip, flow in observed)
