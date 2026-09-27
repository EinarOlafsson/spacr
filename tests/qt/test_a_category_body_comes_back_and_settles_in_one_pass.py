"""A settings category's body, header and toggle at their edges.

Pinned here, each as what the user sees or gets:

* a hidden category whose body is no longer on its form takes nothing off
  the page; a detached body comes back even when nothing is watching for
  the category being shown;
* a closed category nested inside another is still found by the outer one,
  body and all, while it is off the page;
* a row label handed in as a ready-made widget is the label shown;
* the header shows no tooltip of its own, and other objects' events pass
  through the category untouched;
* toggling inside a scroll area whose content has gone, or inside a toggle
  that is already settling, still opens the category and leaves the
  scrollbar policy alone.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget  # noqa: E402

from spacr.qt.widgets import section as sec  # noqa: E402
from spacr.qt.widgets.section import Section  # noqa: E402

pytestmark = pytest.mark.qt


def _hidden(qtbot, title="Shut"):
    category = Section(title)
    qtbot.addWidget(category)
    category.add_row("Field", QLabel("field"))
    assert category.isHidden()
    return category


# ---------------------------------------------------------------------------
# The body off the page
# ---------------------------------------------------------------------------

def test_a_body_no_longer_on_the_form_is_left_where_it_is(qtbot):
    category = _hidden(qtbot)
    category.layout().removeWidget(category._body)
    assert category._detach_body_while_hidden() == 0
    assert category._body.parent() is category
    assert not category._body_is_detached()


def test_a_body_comes_back_with_nothing_watching_for_the_show(qtbot,
                                                             monkeypatch):
    category = _hidden(qtbot)
    assert category._detach_body_while_hidden() > 0
    monkeypatch.setattr(sec, "_BACK_ON_SHOW", None)

    assert category._attach_body() is True
    assert category._body.parent() is category
    assert category.layout().indexOf(category._body) >= 0


def test_a_closed_nested_category_is_found_while_it_is_away(qtbot):
    outer = Section("Outer")
    qtbot.addWidget(outer)
    inner = Section("Inner", parent=outer._body)
    outer.add_widget(inner)
    deepest = Section("Deepest", parent=inner._body)
    inner.add_widget(deepest)
    inner.hide()
    assert inner._detach_body_while_hidden() > 0
    assert deepest.window() is not outer.window()

    found = outer._nested_sections()
    assert inner in found
    assert deepest in found


# ---------------------------------------------------------------------------
# Rows and the header
# ---------------------------------------------------------------------------

def test_a_widget_label_is_the_label_shown(qtbot):
    category = Section("Downloads", expanded=True)
    qtbot.addWidget(category)
    label = QLabel("Fetch")
    category.add_prose_row(label, QLabel("buttons"))
    assert label.parent() is not None
    assert label.parent().objectName() == "SettingLabelWithInfo"


def test_the_header_shows_no_tooltip_and_others_pass_through(qtbot):
    category = Section("Tips")
    qtbot.addWidget(category)
    tooltip = QEvent(QEvent.Type.ToolTip)
    assert category.eventFilter(category.header(), tooltip) is True

    other = QLabel("elsewhere")
    qtbot.addWidget(other)
    assert category.eventFilter(other, QEvent(QEvent.Type.ToolTip)) is False


# ---------------------------------------------------------------------------
# Toggling inside a scroll area
# ---------------------------------------------------------------------------

def _in_a_scroll(qtbot):
    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    content = QWidget()
    column = QVBoxLayout(content)
    category = Section("Scrolled")
    category.add_row("Field", QLabel("field"))
    column.addWidget(category)
    scroll.setWidget(content)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return scroll, category


def test_a_scroll_whose_content_has_gone_still_opens_the_category(qtbot):
    scroll, category = _in_a_scroll(qtbot)
    seen = []
    category.toggled.connect(seen.append)
    content = scroll.takeWidget()
    category.setParent(None)
    qtbot.addWidget(category)
    category.setParent(scroll.viewport())
    content.deleteLater()
    assert scroll.widget() is None

    category._on_toggle(True)
    assert seen == [True]
    assert category.is_expanded()


def test_a_toggle_inside_a_settling_toggle_leaves_the_policy_alone(qtbot):
    scroll, category = _in_a_scroll(qtbot)
    seen = []
    category.toggled.connect(seen.append)
    scroll.setProperty(sec.SETTLING_DEPTH, 1)

    category._on_toggle(True)
    assert seen == [True]
    assert scroll.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert int(scroll.property(sec.SETTLING_DEPTH)) == 1
