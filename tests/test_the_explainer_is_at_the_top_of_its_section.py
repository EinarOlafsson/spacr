"""A section's prose box opens it, rather than following the controls.

Asked for on 2026-08-17: "just ad the text box i asked for (at the top)", and
"Permutation test is good it just needs a text box at the top briefly
explaining what it does".

The box used to be appended to the PANE after the section, which put it BELOW
every control it describes -- so a user read eleven settings and then found out
what they were choosing between.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    panel = AppScreen("regression")
    qtbot.addWidget(panel)
    return panel


def _sections_with_a_box(screen):
    by_upper = {t.upper(): (t, b)
                for t, b in screen._section_explainers.items()}
    return [(section,) + by_upper[section.title().upper()]
            for section in screen._settings_sections
            if section.title().upper() in by_upper]


def test_both_sections_that_should_have_a_box_have_one(screen):
    from spacr.qt.screens.settings_model import SECTION_EXPLAINERS

    assert set(screen._section_explainers) == set(
        SECTION_EXPLAINERS["regression"])
    assert set(screen._section_explainers) == {"Model & Inference",
                                               "Permutation Test"}


def test_the_box_is_the_first_row_of_its_section(screen):
    from PySide6.QtWidgets import QFormLayout

    found = _sections_with_a_box(screen)
    assert len(found) == 2
    for section, title, box in found:
        form = section._form
        # A SET, because a spanning row answers to more than one role: Qt
        # returns the same item for SpanningRole and FieldRole, so collecting
        # a list finds the one box twice and says nothing about placement.
        rows = {row
                for row in range(form.rowCount())
                for role in (QFormLayout.SpanningRole, QFormLayout.FieldRole,
                             QFormLayout.LabelRole)
                if (form.itemAt(row, role) is not None
                    and form.itemAt(row, role).widget() is box)}
        assert rows == {0}, f"{title}: box at rows {sorted(rows)}, not the top"


def test_the_box_is_not_registered_as_a_setting_row(screen):
    """`Section._row_widgets` is taken to BE the labelled setting rows.

    Registering a prose box there would either fail
    tests/qt/test_all_module_smoke.py's row contract or force a fake
    settingKey onto a non-setting -- which would then be pushed into the
    tooltip and API-documentation machinery.
    """
    for section, _title, box in _sections_with_a_box(screen):
        assert box not in [field for _label, field in section._row_widgets]
        assert all(field.property("settingKey")
                   for _label, field in section._row_widgets)


def test_the_model_box_states_the_formula_for_the_current_model(screen):
    text = screen._section_explainers["Model & Inference"].toPlainText()
    assert text
    # `mixed` is the default since instruction 132, and the box names it.
    assert "mixed" in text.lower()
    assert "~" in text, "a box that states a formula has to contain one"


def test_the_permutation_box_says_what_the_test_does(screen):
    text = screen._section_explainers["Permutation Test"].toPlainText()
    assert text
    assert len(text) < len(
        screen._section_explainers["Model & Inference"].toPlainText()), (
        "the permutation box was asked for as a BRIEF explanation")


def test_the_permutation_box_does_not_chase_the_panel(screen):
    """Only the model box depends on the panel's values.

    A static box that re-rendered on every keystroke would be work for
    nothing; worse, it would be a second thing to keep in step.
    """
    before = screen._section_explainers["Permutation Test"].toPlainText()
    screen._refresh_model_explainer()
    assert screen._section_explainers[
        "Permutation Test"].toPlainText() == before


def test_add_prose_is_the_primitive_and_add_widget_is_not(qtbot):
    """The two have the same signature and opposite bookkeeping."""
    from PySide6.QtWidgets import QLabel

    from spacr.qt.widgets.section import Section

    section = Section("Demo")
    qtbot.addWidget(section)
    prose, row = QLabel("prose"), QLabel("row")

    section.add_prose(prose)
    assert section._row_widgets == []

    section.add_widget(row)
    assert [w for _l, w in section._row_widgets] == [row]


def test_add_prose_can_go_above_the_rows(qtbot):
    from PySide6.QtWidgets import QFormLayout, QLabel

    from spacr.qt.widgets.section import Section

    section = Section("Demo")
    qtbot.addWidget(section)
    first, box = QLabel("a"), QLabel("box")
    section.add_row(QLabel("label"), first)
    section.add_prose(box, at_top=True)

    item = (section._form.itemAt(0, QFormLayout.SpanningRole)
            or section._form.itemAt(0, QFormLayout.FieldRole))
    assert item is not None and item.widget() is box
