"""Navigating a BUILT preferences dialog by the two handles it leaves behind.

Instruction 422. A preference result in the Help search field has to land on
the row the user asked for, and the dialog it lands in is assembled by
procedure with no schema to address. ``spacr.qt.preferences_navigation`` finds
its way by page object name and by row caption, and the tests in
``test_the_help_search_field_lands_where_it_says.py`` drive that against the
real dialog.

THIS FILE IS THE OTHER HALF: what the same functions do when the dialog is
not the one they expected. A page that is not in any tab, a caption nobody is
showing, a row that went away between being found and being marked -- each of
those is a plausible afternoon's refactor of ``preferences.py``, and none of
them may reach the user as a traceback or as a dialog that claims to have
landed somewhere it did not. A synthetic dialog is used on purpose: the real
one cannot be made to go wrong in these particular ways without breaking it
for every other test.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@pytest.fixture
def dialog(qtbot):
    """A dialog shaped like Preferences: tabs, a page, one captioned row."""
    from spacr.qt.preferences_navigation import TABS_NAME

    holder = QWidget()
    qtbot.addWidget(holder)
    column = QVBoxLayout(holder)
    tabs = QTabWidget(holder)
    tabs.setObjectName(TABS_NAME)
    column.addWidget(tabs)

    for name, caption in (("PreferencesTabProbe", "PNG resolution"),
                          ("PreferencesTabOther", "Something else")):
        scroll = QScrollArea(tabs)
        page = QWidget(scroll)
        page.setObjectName(name)
        form = QFormLayout(page)
        form.addRow(QLabel(caption, page), QLineEdit(page))
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        tabs.addTab(scroll, name)
    holder.show()
    qtbot.waitExposed(holder)
    return holder


def test_the_tab_bar_is_found_even_when_it_is_not_the_named_one(qtbot):
    """The object name is the handle; one tab bar is the fallback.

    A dialog that stopped naming its tab bar would otherwise make every
    preference result silently unreachable.
    """
    from spacr.qt.preferences_navigation import tab_widget

    holder = QWidget()
    qtbot.addWidget(holder)
    column = QVBoxLayout(holder)
    anonymous = QTabWidget(holder)
    column.addWidget(anonymous)
    assert tab_widget(holder) is anonymous
    assert tab_widget(QWidget()) is None


def test_a_page_is_addressed_by_name_and_only_by_name(dialog):
    """An empty name addresses nothing rather than the first page."""
    from spacr.qt.preferences_navigation import page_named

    assert page_named(dialog, "PreferencesTabProbe") is not None
    assert page_named(dialog, "") is None
    assert page_named(dialog, "PreferencesTabNowhere") is None


def test_a_page_that_is_in_no_tab_is_not_shown(qtbot, dialog):
    """Found in the dialog is not the same as reachable from the tab bar."""
    from spacr.qt.preferences_navigation import show_tab

    orphan = QWidget(dialog)
    orphan.setObjectName("PreferencesTabOrphan")
    assert show_tab(dialog, "PreferencesTabOrphan") is False
    assert show_tab(QWidget(), "PreferencesTabProbe") is False
    assert show_tab(dialog, "PreferencesTabNowhere") is False


def test_landing_without_a_row_lands_on_the_page(qtbot, dialog):
    """A tab with no row named still has to come to the front."""
    from spacr.qt.preferences_navigation import show_tab, tab_widget

    tabs = tab_widget(dialog)
    tabs.setCurrentIndex(1)
    assert show_tab(dialog, "PreferencesTabProbe") is True
    assert tabs.currentIndex() == 0


def test_a_caption_is_looked_for_in_both_languages(monkeypatch, dialog):
    """The dialog shows the translated caption; the index holds the English.

    So both spellings are looked for, and a catalog that cannot answer at
    all leaves the English one still being looked for.
    """
    from spacr.qt import preferences_navigation as navigation

    assert navigation._captions("PNG resolution") == ["PNG resolution"]

    monkeypatch.setattr(navigation, "tr", lambda text: "Auflösung")
    assert navigation._captions("PNG resolution") == ["Auflösung",
                                                      "PNG resolution"]

    monkeypatch.setattr(navigation, "tr", lambda text: "")
    assert navigation._captions("PNG resolution") == ["PNG resolution"]

    def explode(_text):
        raise RuntimeError("no catalog")

    monkeypatch.setattr(navigation, "tr", explode)
    assert navigation._captions("PNG resolution") == ["PNG resolution"]


def test_a_row_nobody_is_showing_is_not_found(dialog):
    """The drift answer: no row rather than the wrong row."""
    from spacr.qt.preferences_navigation import page_named, reveal_row, row_field

    page = page_named(dialog, "PreferencesTabProbe")
    assert row_field(page, "PNG resolution") is not None
    assert row_field(page, "A setting nobody has") is None
    assert reveal_row(page, "A setting nobody has") is False


def test_the_marked_row_is_put_back_the_way_it_was_found(qtbot, monkeypatch,
                                                         dialog):
    """A static outline for a few seconds, and then the row as it was.

    Static on purpose: a mark that moved would owe an answer to the
    Animation preferences, and this one owes none.
    """
    from spacr.qt import preferences_navigation as navigation

    page = navigation.page_named(dialog, "PreferencesTabProbe")
    widget = navigation.row_field(page, "PNG resolution")
    before = widget.styleSheet()
    monkeypatch.setattr(navigation, "MARK_MS", 10)
    assert navigation.reveal_row(page, "PNG resolution") is True
    assert widget.property("spacrRevealed") is True
    qtbot.waitUntil(lambda: widget.property("spacrRevealed") is False,
                    timeout=2000)
    assert widget.styleSheet() == before


def test_a_row_that_goes_away_while_it_is_being_marked_is_not_a_crash(
        qtbot, monkeypatch, dialog):
    """Found and then deleted is a refusal, not a traceback on the GUI thread."""
    from spacr.qt import preferences_navigation as navigation

    page = navigation.page_named(dialog, "PreferencesTabProbe")

    class _Gone(QWidget):
        def setFocus(self, *_args):
            raise RuntimeError("the row was deleted")

    gone = _Gone(page)
    qtbot.addWidget(gone)
    monkeypatch.setattr(navigation, "row_field", lambda _page, _label: gone)
    assert navigation.reveal_row(page, "PNG resolution") is False


def test_a_row_deleted_before_the_mark_expires_is_not_a_crash(qtbot,
                                                              monkeypatch,
                                                              dialog):
    """The timer outlives the page when Preferences is closed on top of it."""
    from spacr.qt import preferences_navigation as navigation

    page = navigation.page_named(dialog, "PreferencesTabProbe")
    widget = navigation.row_field(page, "PNG resolution")
    monkeypatch.setattr(navigation, "MARK_MS", 20)
    assert navigation.reveal_row(page, "PNG resolution") is True
    widget.setParent(None)
    widget.deleteLater()
    qtbot.wait(120)


def test_a_page_with_nothing_to_scroll_is_still_marked(qtbot):
    """Not every page is inside a scroll area, and the mark does not need one."""
    from spacr.qt import preferences_navigation as navigation

    page = QWidget()
    qtbot.addWidget(page)
    page.setObjectName("PreferencesTabBare")
    form = QFormLayout(page)
    form.addRow(QLabel("Interface font size", page), QLineEdit(page))
    page.show()
    qtbot.waitExposed(page)
    assert navigation.reveal_row(page, "Interface font size") is True
    assert navigation.row_field(
        page, "Interface font size").property("spacrRevealed") is True


def test_the_row_named_on_the_way_in_is_the_one_that_is_marked(qtbot, dialog):
    """``show_tab`` schedules the mark, so the event loop has to be let run."""
    from spacr.qt import preferences_navigation as navigation

    assert navigation.show_tab(dialog, "PreferencesTabProbe",
                               "PNG resolution") is True
    page = navigation.page_named(dialog, "PreferencesTabProbe")
    widget = navigation.row_field(page, "PNG resolution")
    qtbot.waitUntil(lambda: widget.property("spacrRevealed") is True,
                    timeout=2000)
