"""The theme module's edges: what it does when it is asked for the impossible.

Almost everything here is called from a ``paintEvent``, from a widget
constructor, or from a module import, so none of it may raise. That shape
covers three groups: the contrast and hue solvers, which must return a
defensible number even when no colour satisfies the rules; the styling
helpers, which must survive a widget whose C++ half has already gone; and
``ensure_widget_qss_applied``, which is called at import time and has to end
in a plain ``False`` and an untouched application whatever is missing --
including ``PySide6.QtWidgets`` itself, under the packaging smoke checks and
the documentation build.
"""
from __future__ import annotations

import builtins

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QAbstractScrollArea,   # noqa: E402
                               QWidget)

from spacr.qt import theme                            # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def styled_app(qapp):
    """An application carrying a stylesheet with no registered-block marker."""
    before = qapp.styleSheet()
    qapp.setStyleSheet("QWidget { color: red; }")
    yield qapp
    qapp.setStyleSheet(before)


@pytest.fixture
def no_qt_widgets(monkeypatch):
    """Make ``import PySide6.QtWidgets`` fail, and nothing else."""
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PySide6.QtWidgets":
            raise ImportError("no module named 'PySide6.QtWidgets'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_without_the_qt_widgets_module_the_sheet_is_left_alone(
        styled_app, no_qt_widgets):
    """No Qt to ask for the application means nothing to restyle."""
    assert theme.ensure_widget_qss_applied("SomeBlock") is False
    assert styled_app.styleSheet() == "QWidget { color: red; }"


def test_registering_a_block_without_qt_widgets_still_returns_the_block(
        styled_app, no_qt_widgets):
    """Registration is an import-time act; it must survive a headless one."""
    name = "ChromeHeadlessRegistrationBlock"

    def block(palette, opacity=None):
        return "QWidget#Chrome { color: red; }"

    try:
        assert theme.register_widget_qss(name, block) is block
        assert name in theme.widget_qss_names()
        assert styled_app.styleSheet() == "QWidget { color: red; }"
    finally:
        theme.unregister_widget_qss(name)


# --- solvers asked for something they cannot deliver -----------------------

def test_a_surface_no_scrim_can_rescue_reports_a_full_floor():
    """``page`` fails every contrast rule even painted fully opaque."""
    assert theme.legible_scrim_floor("dark", "page") == 1.0


def test_a_role_with_no_contrast_rule_gets_no_ink_band():
    """Bands are spent contrast; a role nothing is read against has none."""
    assert theme._ink_band("dark", "border_soft") is None


def test_a_hue_with_no_colour_in_the_band_falls_back_to_a_plain_shift():
    """The ramps are discrete, so a narrow band can hold no candidate."""
    fallback = "#336699"

    assert theme._hue_ink(0.0, 0.9931, 0.9999, fallback) == \
        theme._hue_shift(fallback, 0.0)


def test_a_role_with_no_hue_seat_comes_back_from_spaceout_unchanged():
    """Only the seated roles are re-hued; the rest are copied through."""
    unseated = "chrome_role_with_no_seat"
    assert unseated not in theme.SPACEOUT_HUES
    original = {unseated: "#123456"}

    assert theme.spaceout_palette(original, 30.0, "dark") == original


# --- the alpha and fade curves --------------------------------------------

def test_an_unspecified_page_opacity_is_the_documented_default():
    """``None`` means :data:`DEFAULT_PANE_OPACITY`, not zero and not one."""
    assert theme.pane_alpha("dark") == \
        theme.pane_alpha("dark", theme.DEFAULT_PANE_OPACITY)
    assert theme.pane_alpha("glass") == \
        theme.pane_alpha("glass", theme.DEFAULT_PANE_OPACITY)


def test_the_field_fade_profile_runs_edge_to_edge():
    """The sampled ramp starts at the left edge and ends at the right."""
    profile = theme.field_fade_profile(5)

    assert [t for t, _ in profile] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [a for _, a in profile] == [
        theme.field_fade_alpha(t) for t, _ in profile]


def test_a_one_stop_fade_profile_is_widened_to_two():
    """A gradient needs two stops; asking for fewer gets the ends."""
    assert [t for t, _ in theme.field_fade_profile(1)] == [0.0, 1.0]


# --- the spaceout clock ---------------------------------------------------

def test_the_drift_step_is_zero_while_spaceout_is_off(monkeypatch):
    """No spaceout means no hue rotation to quantise."""
    monkeypatch.setattr(theme, "_SPACEOUT", False)
    monkeypatch.setattr(theme, "_DRIFT_SECONDS", 9999.0)

    assert theme.spaceout_drift_step() == 0.0


def test_the_clock_advances_only_while_spaceout_is_on(monkeypatch):
    """A disabled spaceout must not accumulate time it will replay later."""
    monkeypatch.setattr(theme, "_SPACEOUT", True)
    monkeypatch.setattr(theme, "_DRIFT_SECONDS", 0.0)
    theme.advance_spaceout_drift(2.5)
    assert theme.spaceout_drift_seconds() == 2.5

    theme.advance_spaceout_drift(-1.0)
    assert theme.spaceout_drift_seconds() == 2.5

    monkeypatch.setattr(theme, "_SPACEOUT", False)
    theme.advance_spaceout_drift(4.0)
    assert theme.spaceout_drift_seconds() == 2.5


def test_the_clock_can_be_set_and_never_runs_backwards_past_zero(monkeypatch):
    """Restoring a saved clock is how a reopened window resumes the drift."""
    monkeypatch.setattr(theme, "_DRIFT_SECONDS", 0.0)

    theme.set_spaceout_drift_seconds(31.5)
    assert theme.spaceout_drift_seconds() == 31.5

    theme.set_spaceout_drift_seconds(-5.0)
    assert theme.spaceout_drift_seconds() == 0.0


def test_enabling_spaceout_twice_does_not_restart_the_drift(monkeypatch):
    """The second call is a no-op, so an already-running drift is not jerked."""
    monkeypatch.setattr(theme, "_SPACEOUT", True)
    monkeypatch.setattr(theme, "_DRIFT_SECONDS", 12.5)

    theme.enable_spaceout()

    assert theme.spaceout_drift_seconds() == 12.5


# --- painting a panel -----------------------------------------------------

def test_an_explicit_theme_and_opacity_ask_no_preference(monkeypatch):
    """Both arguments given, the preference module is never imported."""
    import spacr.qt.preferences as prefs

    def refuse():
        raise AssertionError("the preference store must not be consulted")

    monkeypatch.setattr(prefs, "resolve_effective_theme", refuse)
    monkeypatch.setattr(prefs, "get_pane_opacity", refuse)

    colour = theme.panel_qcolor("surface", "dark", 0.5)

    assert colour.isValid()
    assert colour.alphaF() == pytest.approx(
        theme.panel_alpha("dark", "surface", 0.5), abs=1e-3)


def test_a_page_colour_survives_an_unreadable_preference_store(monkeypatch):
    """A backdrop must never be the thing that stops a screen opening."""
    import spacr.qt.preferences as prefs

    def refuse():
        raise RuntimeError("the preference store is unavailable")

    monkeypatch.setattr(prefs, "resolve_effective_theme", refuse)

    assert theme.active_page_colour() == theme.page_colour("dark")


def test_a_borderless_panel_leaves_the_hairline_off(qtbot):
    """``border=False`` is for panels that already sit inside a frame."""
    from PySide6.QtGui import QImage, QPainter

    widget = QWidget()
    qtbot.addWidget(widget)
    widget.resize(40, 30)

    def render(border):
        image = QImage(40, 30, QImage.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        try:
            theme.paint_panel(painter, widget, theme="dark", border=border,
                              radius=8)
        finally:
            painter.end()
        return image

    with_border = render(True)
    without_border = render(False)

    assert with_border != without_border
    assert without_border.pixelColor(20, 15).alpha() > 0


# --- declaring what does and does not paint -------------------------------

class _ViewportlessScrollArea(QAbstractScrollArea):
    """A scroll area caught after its viewport's C++ half has gone."""

    def viewport(self):
        return None


def test_a_none_is_skipped_without_stopping_the_widgets_after_it(qtbot):
    """Callers pass a widget that may not have been built yet.

    ``make_transparent`` is variadic and screens call it with a whole layout at
    once -- header, splitter, scroll area. One of those can legitimately be
    None on a screen that builds its header lazily, and the guard is a
    ``continue`` precisely so the containers listed after it are still tagged.
    Were it a ``break`` or an early return the backdrop would be buried by
    whichever container happened to follow the None in the argument list.
    """
    from PySide6.QtWidgets import QWidget

    after = QWidget()
    qtbot.addWidget(after)
    assert not after.property(theme.TRANSPARENT_PROPERTY)

    theme.make_transparent(None, after)

    assert after.property(theme.TRANSPARENT_PROPERTY) is True


def test_a_scroll_area_with_no_viewport_still_gets_tagged(qtbot):
    """The area itself is tagged even when the half that paints is gone."""
    area = _ViewportlessScrollArea()
    qtbot.addWidget(area)

    theme.make_transparent(area)

    assert area.property(theme.TRANSPARENT_PROPERTY) is True


def test_a_viewportless_scroll_area_does_not_stop_the_container_sweep(qtbot):
    """One broken child must not leave the rest of the screen opaque."""
    root = QWidget()
    qtbot.addWidget(root)
    broken = _ViewportlessScrollArea(root)
    plain = QWidget(root)

    tagged = theme.clear_container_surfaces(root)

    assert tagged >= 2
    assert broken.property(theme.TRANSPARENT_PROPERTY) is True
    assert plain.property(theme.TRANSPARENT_PROPERTY) is True


# --- registering a QSS block ----------------------------------------------

def test_a_widget_qss_block_must_be_callable():
    """A block is called with the palette; a string could never be."""
    with pytest.raises(TypeError, match="is not callable"):
        theme.register_widget_qss("ChromeNotCallable",
                                  "QWidget { color: red; }")


def test_an_unstyled_application_is_not_given_a_stylesheet(qapp):
    """Nothing has styled the app, so there is nothing to be missing from."""
    before = qapp.styleSheet()
    qapp.setStyleSheet("")
    try:
        assert theme.ensure_widget_qss_applied("SomeBlock") is False
        assert qapp.styleSheet() == ""
    finally:
        qapp.setStyleSheet(before)


# --- close marks ----------------------------------------------------------

def test_a_close_mark_keeps_one_resizer_however_often_it_is_applied(qtbot):
    """Re-applying must not stack event filters on the same button."""
    from PySide6.QtWidgets import QToolButton

    button = QToolButton()
    qtbot.addWidget(button)

    theme.apply_close_mark(button)
    first = button._spacr_close_mark_resizer
    theme.apply_close_mark(button)

    assert button._spacr_close_mark_resizer is first


def test_a_close_mark_without_a_tooltip_keeps_the_one_it_had(qtbot):
    """``None`` preserves the existing tooltip rather than clearing it."""
    from PySide6.QtWidgets import QToolButton

    button = QToolButton()
    qtbot.addWidget(button)
    button.setToolTip("Close this plate")

    theme.apply_close_mark(button)

    assert button.toolTip() == "Close this plate"


def test_a_close_mark_already_the_right_size_is_left_alone(qtbot):
    """The second measurement changes nothing, so it must not re-fix the box."""
    from PySide6.QtWidgets import QToolButton

    button = QToolButton()
    qtbot.addWidget(button)
    theme.apply_close_mark(button)
    fixed = button.size()

    button.setMinimumWidth(0)
    theme.size_close_mark(button)

    assert button.size() == fixed


def test_a_resizer_whose_button_has_gone_declines_the_event(qtbot):
    """Qt keeps delivering events to a filter after its widget is destroyed."""
    import shiboken6
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QToolButton

    button = QToolButton()
    qtbot.addWidget(button)
    theme.apply_close_mark(button)
    resizer = button._spacr_close_mark_resizer
    shiboken6.delete(button)

    assert resizer.eventFilter(button, QEvent(QEvent.FontChange)) is False


# --- tab bars -------------------------------------------------------------

@pytest.fixture
def closable_tabs(qtbot):
    """A tab widget with three closable tabs."""
    from PySide6.QtWidgets import QTabWidget

    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.setTabsClosable(True)
    for name in ("plate1", "plate2", "plate3"):
        tabs.addTab(QWidget(), name)
    return tabs


def test_a_close_mark_closes_the_tab_it_is_sitting_on_now(qtbot,
                                                          closable_tabs):
    """Closing an earlier tab renumbers the rest, so the mark is looked up."""
    from PySide6.QtWidgets import QTabBar

    assert theme.install_close_marks(closable_tabs) == 3
    bar = closable_tabs.tabBar()
    mark = (bar.tabButton(2, QTabBar.RightSide)
            or bar.tabButton(2, QTabBar.LeftSide))

    with qtbot.waitSignal(bar.tabCloseRequested, timeout=1000) as caught:
        mark.click()

    assert caught.args == [2]


def test_a_tab_bar_being_marked_does_not_mark_itself_again(qtbot,
                                                           closable_tabs):
    """``setTabButton`` re-enters the sweep; the guard makes that a no-op."""
    bar = closable_tabs.tabBar()
    bar._spacr_marking_tabs = True
    try:
        assert theme.mark_tab_bar(bar) == 0
    finally:
        bar._spacr_marking_tabs = False


def test_close_marks_install_on_a_bare_tab_bar(qtbot, closable_tabs):
    """``root`` may be the bar itself, not only the widget around it."""
    assert theme.install_close_marks(closable_tabs.tabBar()) == 3


def test_close_marks_install_on_everything_below_a_container(qtbot):
    """A screen hands in its page and expects every tab strip under it."""
    from PySide6.QtWidgets import QTabWidget

    container = QWidget()
    qtbot.addWidget(container)
    tabs = QTabWidget(container)
    tabs.setTabsClosable(True)
    for name in ("plate1", "plate2"):
        tabs.addTab(QWidget(), name)

    assert theme.install_close_marks(container) == 2


def test_the_scroll_arrows_come_off_a_tab_widget(qtbot, closable_tabs):
    """Given a tab widget, its own bar is the one to strip."""
    assert theme.take_the_scroll_arrows_off(closable_tabs) == 1
    assert closable_tabs.tabBar().usesScrollButtons() is False


def test_the_scroll_arrows_come_off_a_bare_tab_bar(qtbot, closable_tabs):
    """Given a bar, that bar is the one to strip."""
    bar = closable_tabs.tabBar()

    assert theme.take_the_scroll_arrows_off(bar) == 1
    assert bar.usesScrollButtons() is False


def test_a_close_mark_whose_tab_has_gone_asks_for_nothing(qtbot,
                                                          closable_tabs):
    """A mark left over from a removed tab must not close somebody else's."""
    from PySide6.QtWidgets import QTabBar

    theme.install_close_marks(closable_tabs)
    bar = closable_tabs.tabBar()
    side = (QTabBar.RightSide if bar.tabButton(1, QTabBar.RightSide)
            else QTabBar.LeftSide)
    mark = bar.tabButton(1, side)
    mark.setParent(None)
    closable_tabs.removeTab(1)

    with qtbot.assertNotEmitted(bar.tabCloseRequested):
        theme._request_tab_close(bar, mark, side)

    assert closable_tabs.count() == 2
