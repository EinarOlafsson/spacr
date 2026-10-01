"""The Help search list says when it is not showing everything.

Instruction 422, finished 2026-09-30. The per-kind cap of eight keeps one
query's list mixed, and it also clipped "one row per module": ``src`` is in
36 modules and was offered for 8 of them, with nothing on screen saying
another 28 existed. A row reading "and 28 more" now follows the last shown
row of a clipped kind, and opening it -- click or Return -- reveals the next
rows of that kind without closing the popup.

Each test drives the real field with a small hand-built index so the
arithmetic is visible at the call site.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest


@pytest.fixture
def window(qtbot, qt_theme_applied):
    """A real main window with the field installed."""
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    return win


@pytest.fixture
def field(window):
    """The installed field, loaded with 20 module rows of one setting."""
    from spacr.qt.help_index import HelpEntry
    from spacr.qt.help_search import field_of

    found = field_of(window)
    assert found is not None
    rows = [HelpEntry(kind="setting", title="zzsrc", subtitle=f"M{i}",
                      payload={"app": "mask", "key": "zzsrc",
                               "category": ""})
            for i in range(20)]
    rows.append(HelpEntry(kind="module", title="Zzsrc module",
                          subtitle="Module", payload={"app": "mask"}))
    found.set_index(rows)
    return found


def _texts(field):
    """Every row's text, in order."""
    return [field._list.item(i).text() for i in range(field._list.count())]


def test_a_clipped_kind_is_followed_by_a_row_that_says_how_many(field):
    """Eight rows shown, twelve left out, and the list says twelve."""
    from spacr.qt.help_search import MORE_ROLE

    results = field.type_and_search("zzsrc")
    assert sum(e.kind == "setting" for e in results) == 8
    assert field._left_out == {"setting": 12}
    texts = _texts(field)
    assert "and 12 more" in texts
    more = texts.index("and 12 more")
    assert field._list.item(more).data(MORE_ROLE) == "setting"
    # It sits under its own kind: the row before it is a setting row.
    assert texts[more - 1].startswith("zzsrc")
    # The more row is not a result; results() stays entries only.
    assert len(field.results()) == field._list.count() - 1


def test_return_on_the_more_row_reveals_the_rest_and_keeps_the_list(
        qtbot, field):
    """Keyboard alone: walk down to it, press Return, the rows arrive."""
    field.setFocus()
    field.type_and_search("zzsrc")
    more = _texts(field).index("and 12 more")
    field._list.setCurrentRow(more)
    QTest.keyClick(field, Qt.Key_Return)
    qtbot.waitUntil(lambda: "and 12 more" not in _texts(field),
                    timeout=2000)
    assert sum(e.kind == "setting" for e in field.results()) == 20
    assert field._left_out == {}
    assert field.popup().isVisible()
    assert field._list.currentRow() == more


def test_clicking_the_more_row_opens_nothing(qtbot, field):
    """It reveals rows; it never counts as a result being opened."""
    field.type_and_search("zzsrc")
    more = _texts(field).index("and 12 more")
    with qtbot.assertNotEmitted(field.opened, wait=100):
        field._on_activated(field._list.item(more))
        qtbot.waitUntil(lambda: field._left_out == {}, timeout=2000)


def test_typing_again_puts_the_cap_back(field):
    """A lifted cap belongs to one query, not to the session."""
    field.type_and_search("zzsrc")
    field._show_more("setting")
    assert field._left_out == {}
    field.type_and_search("zzsr")
    assert field._left_out == {"setting": 12}


def test_a_list_that_shows_everything_has_no_more_row(field):
    """No count, no row: "and 0 more" would be noise."""
    from spacr.qt.help_search import MORE_ROLE

    field.type_and_search("Zzsrc module")
    assert field._left_out == {}
    assert field._list.count() >= 1
    assert all(field._list.item(i).data(MORE_ROLE) is None
               for i in range(field._list.count()))


def test_the_more_row_is_rendered_through_the_catalog(monkeypatch, field):
    """Its words go through ``tr`` like every other caption in the field."""
    from spacr.qt import help_search

    seen = []

    def fake_tr(source, **values):
        """Record what was asked for and answer in brackets."""
        seen.append(source)
        return "[" + source.format(**values) + "]"

    monkeypatch.setattr(help_search, "tr", fake_tr)
    field.type_and_search("zzsrc")
    assert "and {count} more" in seen
    assert "[and 12 more]" in _texts(field)


def test_an_empty_query_forgets_what_was_left_out(field):
    """Clearing the box clears the count with the list."""
    field.type_and_search("zzsrc")
    assert field._left_out
    field.type_and_search("")
    assert field._left_out == {}
    assert field._list.count() == 0
