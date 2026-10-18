"""A name the accessibility pass derived gives way once Qt can name the control.

The pass names a control that Qt would announce with no name, and marks
that name as its own. When the control later gains something Qt reads
itself -- a button caption, a window title, a titled group box, a label
that is its buddy -- the derived name is cleared, so the reader hears the
one Qt gives. A menu whose title is removed loses the derived name with it,
and a menu deleted mid-pass is passed over.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spacr.qt import i18n

pytestmark = pytest.mark.qt


def test_qt_can_name_a_captioned_button_a_titled_window_or_group_and_a_buddy(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    button = QToolButton(host)
    assert not i18n._a11y_qt_may_name(button, set())
    button.setText("Run")
    assert i18n._a11y_qt_may_name(button, set())

    window = QWidget()
    qtbot.addWidget(window)
    assert not i18n._a11y_qt_may_name(window, set())
    window.setWindowTitle("Settings")
    assert i18n._a11y_qt_may_name(window, set())

    group = QGroupBox(host)
    field = QLineEdit(group)
    assert not i18n._a11y_qt_may_name(field, set())
    assert i18n._a11y_qt_may_name(field, {field})
    group.setTitle("Object filters")
    assert i18n._a11y_qt_may_name(field, set())


def test_a_derived_name_is_cleared_when_the_field_moves_into_a_titled_group(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    group = QGroupBox(host)
    field = QLineEdit(group)
    field.setPlaceholderText("Cell diameter")
    assert i18n._name_for_assistive_tech([field]) >= 1
    assert field.accessibleName() == "Cell diameter"
    group.setTitle("Cellpose")
    i18n._name_for_assistive_tech([field])
    assert field.accessibleName() == ""


def test_a_menu_whose_title_is_removed_loses_its_derived_name(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    menu = QMenu("Export", host)
    assert i18n._name_for_assistive_tech([menu]) == 1
    assert menu.accessibleName() == "Export"
    menu.setTitle("")
    assert i18n._name_for_assistive_tech([menu]) == 1
    assert menu.accessibleName() == ""


def test_a_menu_deleted_before_the_pass_reaches_it_is_passed_over(qtbot):
    import shiboken6

    host = QWidget()
    qtbot.addWidget(host)
    gone = QMenu("Recent", host)
    kept = QMenu("Open", host)
    shiboken6.delete(gone)
    assert i18n._name_for_assistive_tech([gone, kept]) == 1
    assert kept.accessibleName() == "Open"


def test_a_scroll_area_looks_past_an_untitled_group_for_its_caption(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    area = QScrollArea(host)
    content = QWidget()
    group = QGroupBox(content)
    QVBoxLayout(content).addWidget(group)
    QVBoxLayout(group).addWidget(QLabel("Illumination correction"))
    area.setWidget(content)
    assert i18n._a11y_scroll_area_name(area) == "Illumination correction"
