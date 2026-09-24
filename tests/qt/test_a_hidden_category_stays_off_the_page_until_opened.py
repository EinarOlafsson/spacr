"""A settings category nobody can see is kept off the page until it is opened.

Opening a module styles every widget under its page against the whole
stylesheet, hidden or not. Counted on 2026-09-21, most of a settings form is
in categories the user cannot see at open -- collapsed, or hidden by the
Essentials view or the maturity preference: 533 of Classify's widgets, 392
of Mask's, 303 of Measure's. `Section._detach_body_while_hidden` takes those
bodies out of the page just before it is first styled, and
`Section._attach_body` puts each back the moment it could be seen.

NOTHING IS REBUILT, and these tests hold that half as hard as the speed
half: every control still exists and is still the one the settings model
reads, the run still collects every value, the search still finds and
reveals settings in a category that is away, and a category that comes back
is translated into the language the window is in NOW.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtGui import QPalette                              # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel,            # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt.widgets.section import (Section, _logical_parent,  # noqa: E402
                                      _sections_below)

#: One colour nothing else uses, so a label wearing it is unambiguous.
SHEET = "QLabel { color: rgb(11, 22, 33); }"
SHEET_HEX = "#0b1621"


@contextmanager
def _language(code):
    from spacr.qt import i18n

    before = os.environ.get(i18n.ENV_LANGUAGE)
    os.environ[i18n.ENV_LANGUAGE] = code
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(i18n.ENV_LANGUAGE, None)
        else:
            os.environ[i18n.ENV_LANGUAGE] = before


def _pump(rounds: int = 20) -> None:
    app = QApplication.instance()
    for _ in range(rounds):
        app.processEvents()


def _page(qtbot):
    page = QWidget()
    qtbot.addWidget(page)
    layout = QVBoxLayout(page)
    open_one = Section("Open")
    open_one.add_row("Seen", QLabel("seen"))
    open_one.set_expanded(True)
    shut = Section("Shut")
    field = QLabel("field")
    shut.add_row("Hidden", field)
    layout.addWidget(open_one)
    layout.addWidget(shut)
    return page, open_one, shut, field


# -- the section --------------------------------------------------------------

def test_a_collapsed_body_leaves_the_page_and_an_open_one_stays(qtbot):
    page, open_one, shut, field = _page(qtbot)
    assert open_one._detach_body_while_hidden() == 0
    assert shut._detach_body_while_hidden() > 0
    assert shut._body_is_detached()
    assert not page.isAncestorOf(field)
    assert shut._holds(field), "the category no longer knows its own row"
    assert shut.header().parentWidget() is shut, "the header left too"


def test_opening_the_category_puts_the_body_back_and_styles_it(qtbot):
    from spacr.qt import theme

    app = QApplication.instance()
    page, _open_one, shut, field = _page(qtbot)
    shut._detach_body_while_hidden()
    try:
        theme.apply_stylesheet_per_window(app, SHEET)
        page.show()
        _pump()
        shut.set_expanded(True)
        _pump()
        assert not shut._body_is_detached()
        assert page.isAncestorOf(field)
        assert field.isVisible()
        assert field.palette().color(QPalette.WindowText).name() == SHEET_HEX
    finally:
        theme.apply_stylesheet_per_window(app, "")


def test_a_hidden_open_category_comes_back_when_it_is_shown(qtbot):
    page, _open_one, shut, field = _page(qtbot)
    shut.set_expanded(True)
    shut.hide()
    assert shut._detach_body_while_hidden() > 0
    page.show()
    _pump()
    shut.setVisible(True)
    _pump()
    assert not shut._body_is_detached()
    assert field.isVisible()


def test_a_collapsed_category_shown_again_keeps_its_body_away(qtbot):
    page, _open_one, shut, _field = _page(qtbot)
    shut.hide()
    shut._detach_body_while_hidden()
    page.show()
    shut.setVisible(True)
    _pump()
    assert shut.isVisible()
    assert shut._body_is_detached(), "a collapsed header brought its body"


def test_the_body_goes_back_where_it_was(qtbot):
    _page_widget, _open_one, shut, _field = _page(qtbot)
    index = shut.layout().indexOf(shut._body)
    shut._detach_body_while_hidden()
    shut._attach_body()
    assert shut.layout().indexOf(shut._body) == index


def test_nested_headings_are_still_found_through_a_detached_body(qtbot):
    page = QWidget()
    qtbot.addWidget(page)
    outer = Section("Outer")
    inner = Section("Inner")
    deep = QLabel("deep")
    inner.add_row("Deep", deep)
    outer.add_prose(inner)
    QVBoxLayout(page).addWidget(outer)
    outer._detach_body_while_hidden()

    assert outer._nested_sections() == [inner]
    assert inner in _sections_below(page)
    assert _logical_parent(inner.parentWidget()) is outer
    assert outer._holds(deep)


def test_attaching_twice_and_detaching_an_open_body_are_no_ops(qtbot):
    _page_widget, open_one, shut, _field = _page(qtbot)
    assert shut._attach_body() is False
    assert open_one._detach_body_while_hidden() == 0
    shut._detach_body_while_hidden()
    assert shut._detach_body_while_hidden() == 0
    assert shut._attach_body() is True
    assert shut._attach_body() is False


def test_whoever_asked_is_told_the_body_came_back(qtbot):
    heard = []
    _page_widget, _open_one, shut, _field = _page(qtbot)
    shut._body_came_back = heard.append
    shut._detach_body_while_hidden()
    shut.set_expanded(True)
    assert heard == [shut]


# -- on a real module ---------------------------------------------------------

def _window(qtbot, key):
    from spacr.qt import register_self_registering_modules
    from spacr.qt.app import MainWindow

    register_self_registering_modules()
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    _pump(40)
    window._on_nav_selected(key)
    _pump(60)
    return window, window._screens[key]


def _away(screen) -> list:
    """Categories whose body is off the page: detached, or not built yet.

    Since 2026-09-22 a category closed at open is not built at all until it
    is opened (``AppScreen._build_a_waiting_heading``); detaching is what
    happens to the categories that ARE built and closed.
    """
    return [section for section in screen.rendered_settings_sections()
            if section._body_is_detached()
            or screen._heading_is_waiting(section)]


def _detach_every_built_category(screen) -> None:
    """Build every category, close them all, and detach them as at open."""
    screen._open_every_waiting_heading()
    for section in screen.rendered_settings_sections():
        section.set_expanded(False)
    screen._detach_what_the_form_hides()


def _key_in_a_detached_body(screen):
    model = screen._settings_model
    for key, widget in model._widgets.items():
        if not screen.isAncestorOf(widget):
            return key, widget
    raise AssertionError("no setting sat in a detached category")


def test_opening_a_module_takes_what_it_hides_off_the_page(qtbot):
    _window_, screen = _window(qtbot, "classify_merged")
    away = _away(screen)
    assert away, "no category left the page"
    for section in away:
        assert (not screen.isAncestorOf(section._body)
                or not section._body.findChildren(QWidget))
    unbuilt = len(screen._waiting_heading_of)
    detached = sum(len(s._body.findChildren(QWidget)) for s in away
                   if s._body_is_detached())
    assert unbuilt > 50 or detached > 200


def test_every_setting_still_reaches_the_run(qtbot):
    _window_, screen = _window(qtbot, "classify_merged")
    model = screen._settings_model
    values = model.collect()
    key, _widget = _key_in_a_detached_body(screen)
    assert key in values
    assert set(model._widgets) <= set(values)


def test_a_value_set_while_its_category_is_away_is_collected(qtbot):
    _window_, screen = _window(qtbot, "classify_merged")
    model = screen._settings_model
    key = next(key for key, widget in model._widgets.items()
               if not screen.isAncestorOf(widget)
               and type(widget).__name__ in ("QSpinBox", "QDoubleSpinBox"))
    widget = model._widgets[key]
    target = widget.maximum()
    assert model.set_value_for_key(key, target)
    assert model.collect()[key] == target


def test_searching_reveals_a_setting_whose_category_was_away(qtbot):
    _window_, screen = _window(qtbot, "classify_merged")
    bar = screen._settings_search
    key, widget = _key_in_a_detached_body(screen)
    if not bar._disclosure.isChecked():
        bar._disclosure.click()
    bar._input.setText(key)
    _pump(30)
    assert screen.isAncestorOf(widget), "the search left the row off the page"
    assert widget.isVisible(), f"the search did not reveal {key}"


def test_a_category_that_comes_back_speaks_the_language_of_now(qtbot):
    """Switched to Swedish while a category was away, it opens in Swedish."""
    from spacr.qt.i18n import retranslate_widget_tree, tr

    window, screen = _window(qtbot, "mask")
    _detach_every_built_category(screen)
    found = None
    for section in _away(screen):
        for label in section._body.findChildren(QLabel):
            source = str(label.text() or "")
            if source and tr(source, "sv") != source:
                found = (section, label, source)
                break
        if found:
            break
    assert found, "no translatable caption in a detached category"
    section, label, source = found
    assert label.text() == source
    with _language("sv"):
        retranslate_widget_tree(window, "sv")
        assert label.text() == source, (
            "the window's language pass reached a body that was away -- "
            "then this test is not testing the came-back pass")
        section.setVisible(True)
        section.set_expanded(True)
        _pump(10)
        came_back = label.text()
        assert came_back != source, "the category came back in English"
        retranslate_widget_tree(window, "sv")
        assert label.text() == came_back, (
            "the window's own pass renders this caption differently from "
            "the pass the category got when it came back")
