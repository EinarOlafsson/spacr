"""185, through the user's path: the button is actually on the field.

THE GAP THIS CLOSES. 185's tests drove `PlateMapPicker` and `pick_wells_for`
directly, and both worked -- so the instruction was closed while the button
appeared on NOTHING. `build_sections` hands the row builder
`('Control wells', widget)`, a title-cased sentence for a human, and the rule
that decides which fields get a picker matches on the SETTING NAME. They were
never equal, so every field took the no-op branch.

A control that is unreachable is not a control, and the only test that can
tell is one that starts from the built panel.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.qt

from spacr.well_spec import WELL_ONLY_SETTINGS


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


def _buttons_beside(screen, key: str) -> list:
    """Every button in the row the setting's field was placed in."""
    from PySide6.QtWidgets import QPushButton

    field = screen._settings_model._widgets.get(key)
    if field is None:
        return []
    holder = field.parent()
    if holder is None or holder.layout() is None:
        return []
    layout = holder.layout()
    return [layout.itemAt(i).widget() for i in range(layout.count())
            if isinstance(layout.itemAt(i).widget(), QPushButton)]


class TestTheWellFieldsHaveTheirPicker:

    def test_the_key_is_read_off_the_widget_not_the_label(self, screen):
        """The bug in one line: the label is a sentence, the rule wants a
        key."""
        field = screen._settings_model._widgets.get("positive_control_wells")

        assert screen._key_of(field) == "positive_control_wells"

    def test_a_widget_the_panel_does_not_hold_has_no_key(self, screen):
        from PySide6.QtWidgets import QLineEdit

        assert screen._key_of(QLineEdit()) == ""

    @pytest.mark.parametrize(
        "key", ["positive_control_wells", "negative_control_wells",
                "mixed_control_wells", "filter_value"])
    def test_a_well_field_has_a_plate_button(self, screen, key):
        if screen._settings_model._widgets.get(key) is None:
            pytest.skip(f"{key} is not on the regression panel")

        labels = [b.text() for b in _buttons_beside(screen, key)]
        assert "Plate…" in labels, labels

    def test_the_button_is_to_the_RIGHT_of_the_field(self, screen):
        """The plate map fills the field it sits beside; the advisor (192)
        sits on the LEFT of `inference` and proposes a dozen. Reading which
        is which off the position is the whole reason they differ."""
        from PySide6.QtWidgets import QPushButton

        field = screen._settings_model._widgets.get("positive_control_wells")
        layout = field.parent().layout()
        order = [layout.itemAt(i).widget() for i in range(layout.count())]
        button = next(w for w in order
                      if isinstance(w, QPushButton) and w.text() == "Plate…")

        assert order.index(field) < order.index(button)

    def test_a_field_that_is_not_wells_gets_no_picker(self, screen):
        """The picker writes the WHOLE field, so one on `negative_control` --
        which mixes wells with another vocabulary -- would destroy a value it
        does not understand."""
        assert "negative_control" not in WELL_ONLY_SETTINGS
        labels = [b.text() for b in _buttons_beside(screen, "negative_control")]

        assert "Plate…" not in labels

    def test_pressing_it_opens_the_map_on_the_fields_value(self, screen,
                                                           monkeypatch):
        """End to end: the button the user sees reaches the picker."""
        from PySide6.QtWidgets import QPushButton

        seen = {}
        monkeypatch.setattr(
            screen, "pick_wells_for",
            lambda field, key="": seen.update(key=key) or "A01")
        button = next(
            b for b in _buttons_beside(screen, "positive_control_wells")
            if b.text() == "Plate…")

        button.click()

        assert seen.get("key") == "positive_control_wells"


class TestTheTwoButtonsAreNotTheSameButton:

    def test_the_advisor_is_only_on_inference(self, screen):
        labels = [b.text() for b in _buttons_beside(
            screen, "positive_control_wells")]

        assert "Settings for my data…" not in labels

    def test_and_the_plate_map_is_not_on_inference(self, screen):
        labels = [b.text() for b in _buttons_beside(screen, "inference")]

        assert "Plate…" not in labels
        assert "Settings for my data…" in labels
