"""Item 641, 2026-10-25: two GUI-thread freezes while Home idles.

* The Preferences warm-up recomposes the window sheet once screens built
  ahead register their blocks, and each recompose restyled the VISIBLE Home
  page in full for rules naming widgets Home does not have. A root missing
  only blocks whose every selector names an absent id keeps its sheet.
* A recompose also sheeted a screen still being built ahead (parentless,
  ``WA_DontShowOnScreen``), and its join to the stack then repolished it
  twice more. Such a screen is left to the prewarm, which sheets it after
  the join.

Counted in ``setStyleSheet`` calls and ``StyleChange`` events, not timed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget  # noqa: E402

from spacr.qt import theme  # noqa: E402

pytestmark = pytest.mark.qt

_BLOCK = "LateInertProbeBlock"


@pytest.fixture
def isolated_sheet(qapp):
    saved_attr = getattr(qapp, theme._WINDOW_SHEET_ATTRIBUTE, None)
    saved_serial = getattr(qapp, theme._WINDOW_SHEET_SERIAL, 0)
    saved_history = getattr(qapp, theme._WINDOW_SHEET_HISTORY, None)
    yield qapp
    _HOSTS.clear()
    theme.unregister_widget_qss(_BLOCK)
    setattr(qapp, theme._WINDOW_SHEET_ATTRIBUTE, saved_attr)
    setattr(qapp, theme._WINDOW_SHEET_SERIAL, saved_serial)
    setattr(qapp, theme._WINDOW_SHEET_HISTORY, saved_history)


_HOSTS = []


def _shown_page(qtbot, *, host_name="", label_name="HomeProbeLabel"):
    """A visible Home-like page inside a window, marked as a sheet target."""
    host = QWidget()
    host.setObjectName(host_name)
    qtbot.addWidget(host)
    _HOSTS.append(host)
    page = QWidget(host)
    QVBoxLayout(page).addWidget(QLabel("probe", objectName=label_name))
    theme.mark_as_a_sheet_target(page)
    host.show()
    return page


def _counted(widget, monkeypatch):
    calls = []
    real = type(widget).setStyleSheet

    def counting(self, text):
        if self is widget:
            calls.append(len(text))
        return real(self, text)

    monkeypatch.setattr(type(widget), "setStyleSheet", counting)
    return calls


def _recompose_with(app, rule):
    theme.register_widget_qss(_BLOCK, lambda palette, opacity: rule,
                              replace=True)
    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))


def test_a_shown_page_keeps_its_sheet_for_a_block_naming_only_absent_ids(
        isolated_sheet, qtbot, monkeypatch):
    app = isolated_sheet
    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))
    page = _shown_page(qtbot)
    theme._sheet_one_window(page)
    worn = page.styleSheet()

    _recompose_with(app, "QWidget#ScreenOnlyBox QLabel, "
                         "QLabel#ScreenOnlyLabel:hover { color: red; }")
    calls = _counted(page, monkeypatch)

    assert not theme._sheet_one_window(page)
    assert calls == []
    assert page.styleSheet() == worn
    assert page.property(theme._WINDOW_SHEET_SERIAL) == getattr(
        app, theme._WINDOW_SHEET_SERIAL)


def test_a_block_naming_a_widget_of_the_page_still_restyles_it(
        isolated_sheet, qtbot):
    app = isolated_sheet
    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))
    page = _shown_page(qtbot)
    theme._sheet_one_window(page)

    _recompose_with(app, "QLabel#HomeProbeLabel { color: red; }")

    assert theme._sheet_one_window(page)
    assert f"registered widget QSS: {_BLOCK}" in page.styleSheet()


def test_a_block_naming_an_ancestor_of_the_page_still_restyles_it(
        isolated_sheet, qtbot):
    app = isolated_sheet
    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))
    page = _shown_page(qtbot, host_name="ProbeWindowHost")
    theme._sheet_one_window(page)

    _recompose_with(app, "QWidget#ProbeWindowHost QLabel { color: red; }")

    assert theme._sheet_one_window(page)


@pytest.mark.parametrize("rule", [
    "QLabel { color: red; }",
    '*[spacrProbe="true"] { color: red; }',
    "QLabel#ScreenOnlyLabel, QToolTip { color: red; }",
])
def test_a_selector_without_an_id_still_restyles_the_page(
        isolated_sheet, qtbot, rule):
    app = isolated_sheet
    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))
    page = _shown_page(qtbot)
    theme._sheet_one_window(page)

    _recompose_with(app, rule)

    assert theme._sheet_one_window(page)


def test_an_id_inside_an_attribute_value_is_not_taken_for_a_selector_id():
    needs = theme._selector_id_requirements(
        '/* #Commented */ QLabel[text="#NotAnId"] { color: #fff; }')
    assert needs is None
    needs = theme._selector_id_requirements(
        "QWidget#A > QLabel#B, QFrame#C { color: #fff; } QLabel#D {}")
    assert needs == (frozenset({"A", "B"}), frozenset({"C"}),
                     frozenset({"D"}))


def test_a_recompose_leaves_a_screen_being_built_ahead_alone(
        isolated_sheet, qtbot):
    """Its prewarm sheets it once it joins the stack; sheeting it before
    that made the join repolish the whole screen twice more."""
    from PySide6.QtCore import Qt

    app = isolated_sheet
    ahead, plain = QWidget(), QWidget()
    for widget in (ahead, plain):
        qtbot.addWidget(widget)
        QVBoxLayout(widget).addWidget(QLabel("probe"))
    ahead.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)

    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))

    assert ahead.styleSheet() == ""
    assert ahead.property(theme._WINDOW_SHEET_SERIAL) is None
    assert plain.styleSheet(), "an ordinary hidden window is still sheeted"
    assert theme._sheet_one_window(ahead), (
        "the prewarm's own sheeting after the join still applies")
