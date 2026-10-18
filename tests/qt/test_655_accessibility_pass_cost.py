"""The accessibility naming pass costs a bounded amount of work per pass.

Lesson 43 froze for a third of a second while the Measure screen's
Illumination fold opened on the GNOME capture host. Every late-arriving
subtree runs a language pass, and each pass cleared and rewrote the derived
name of every control it reached, so a screen reader bus was sent two name
changes per control per pass even when nothing had changed. These tests
count that work instead of timing it: a repeat pass writes nothing, a pass
for new arrivals re-derives only the scroll areas, and a derived name still
follows a language change.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spacr.qt import i18n

from .test_accessibility_pass import _open

#: The module screen with the most widgets on it.
HEAVIEST_SCREEN = "make_masks"


@pytest.fixture
def writes(monkeypatch):
    """Count every accessible name or description written from Python."""
    counts = {"name": 0, "description": 0}
    set_name = QWidget.setAccessibleName
    set_description = QWidget.setAccessibleDescription

    def name(widget, text):
        counts["name"] += 1
        set_name(widget, text)

    def description(widget, text):
        counts["description"] += 1
        set_description(widget, text)

    monkeypatch.setattr(QWidget, "setAccessibleName", name)
    monkeypatch.setattr(QWidget, "setAccessibleDescription", description)
    return counts


@pytest.fixture
def derivations(monkeypatch):
    """Count how many controls the pass derives a name for."""
    seen = []
    source = i18n._a11y_source

    def counted(widget, labels_by_key):
        seen.append(widget)
        return source(widget, labels_by_key)

    monkeypatch.setattr(i18n, "_a11y_source", counted)
    return seen


def test_a_repeat_pass_on_the_heaviest_screen_writes_no_names(
        qtbot, qt_theme_applied, monkeypatch, writes):
    screen = _open(qtbot, monkeypatch, HEAVIEST_SCREEN)
    i18n.retranslate_widget_tree(screen)
    writes.update(name=0, description=0)
    i18n.retranslate_widget_tree(screen)
    assert writes == {"name": 0, "description": 0}


def test_a_pass_for_new_arrivals_derives_only_scroll_area_names(
        qtbot, qt_theme_applied, monkeypatch, writes, derivations):
    screen = _open(qtbot, monkeypatch, HEAVIEST_SCREEN)
    named = [widget for widget in screen.findChildren(QWidget)
             if widget.property(i18n._A11Y_DERIVED_NAME)]
    assert len(named) > 50
    derivations.clear()
    writes.update(name=0, description=0)
    i18n.retranslate_widget_tree(screen, only_new=True)
    scroll_areas = screen.findChildren(QScrollArea)
    assert len(derivations) <= len(scroll_areas)
    assert all(isinstance(widget, QScrollArea) for widget in derivations)
    assert writes == {"name": 0, "description": 0}


def _deep_panel(qtbot, first=None):
    area = QScrollArea()
    qtbot.addWidget(area)
    content = QWidget()
    column = QVBoxLayout(content)
    if first is not None:
        column.addWidget(QGroupBox(first))
    for _row in range(400):
        holder = QWidget()
        QVBoxLayout(holder).addWidget(QLabel(""))
        column.addWidget(holder)
    column.addWidget(QGroupBox("Buried"))
    area.setWidget(content)
    return area


def test_the_scroll_area_fallback_stops_short_of_a_deep_panel(qtbot):
    assert (i18n._a11y_scroll_area_name(_deep_panel(qtbot))
            == i18n.tr("Details"))
    assert (i18n._a11y_scroll_area_name(_deep_panel(qtbot, "Segmentation"))
            == "Segmentation")


def test_a_scroll_area_named_before_its_contents_follows_them(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    area = QScrollArea(host)
    i18n.retranslate_widget_tree(host)
    assert area.accessibleName() == i18n.tr("Details")
    content = QWidget()
    QVBoxLayout(content).addWidget(QGroupBox("Object filters"))
    area.setWidget(content)
    i18n.retranslate_widget_tree(host, only_new=True)
    assert area.accessibleName() == "Object filters"


def test_a_derived_name_kept_between_passes_follows_the_language(
        qtbot, monkeypatch):
    host = QWidget()
    qtbot.addWidget(host)
    button = QToolButton(host)
    button.setToolTip("Details")
    row = QHBoxLayout(host)
    row.addWidget(button)
    row.addWidget(QLabel("Details"))
    field = QLineEdit()
    row.addWidget(field)
    monkeypatch.setenv(i18n.ENV_LANGUAGE, "en")
    i18n.retranslate_widget_tree(host)
    assert button.accessibleName() == "Details"
    assert field.accessibleName() == "Details"
    monkeypatch.setenv(i18n.ENV_LANGUAGE, "de")
    german = i18n.tr("Details", "de")
    assert german != "Details"
    i18n.retranslate_widget_tree(host, only_new=True)
    assert button.accessibleName() == german
    assert field.accessibleName() == german
    button.setAccessibleName("Open the panel")
    i18n.retranslate_widget_tree(host)
    assert button.accessibleName() == "Open the panel"
