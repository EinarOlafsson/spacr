"""Mask's repeated settings are one table, and the file behind it is unchanged.

Instruction 364, measured: 78 of Mask's 201 settings are the same twenty-odd
questions asked once per object type, so the module asks 203 questions before
anything is segmented. The maintainer chose a TABLE over tabs and over
leaving the names flat.

The property that makes the table safe, and the one every test here is about:
**what the user sees is the settings file, rearranged and not transformed**.
The stored keys never change, so no settings file, notebook, tutorial or
`spacr-run` invocation migrates. If a round trip through this widget altered
one value or one type, the table would be a rewrite of everybody's saved
settings wearing the clothes of a layout change.

It is also what lets instruction 326 land: one more organelle is one more
COLUMN, where the flat vocabulary needed twenty new settings that every
tooltip table and translation catalog had to learn.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.object_settings_table import to_table                # noqa: E402
from spacr.qt.widgets.object_settings_grid import (             # noqa: E402
    AUTO_TEXT, ObjectSettingsGrid)


@pytest.fixture
def mask_settings():
    """The real thing: Mask's own defaults, not a fixture written to suit."""
    from spacr.settings import get_timelapse_settings

    return get_timelapse_settings()


@pytest.fixture
def grid(qtbot, qt_theme_applied, mask_settings):
    widget = ObjectSettingsGrid()
    qtbot.addWidget(widget)
    widget.set_settings(mask_settings)
    return widget


# ---------------------------------------------------------------------------
# The file behind the table
# ---------------------------------------------------------------------------

def test_showing_a_settings_dict_changes_nothing_in_it(grid, mask_settings):
    """The round trip is the whole safety argument."""
    out = grid.settings()

    assert set(out) == set(mask_settings)
    for key, value in mask_settings.items():
        assert out[key] == value, key
        assert type(out[key]) is type(value), f"{key} changed type"


def test_the_settings_the_table_does_not_cover_are_carried_through(
        grid, mask_settings):
    """`src` and `verbose` are not per-object questions and must survive."""
    out = grid.settings()
    for key in ("src", "verbose"):
        if key in mask_settings:
            assert out[key] == mask_settings[key]


def test_every_per_object_key_is_on_screen_exactly_once(grid, mask_settings):
    """78 settings become one table -- and none of them goes missing on the
    way, which a table of the wrong shape would do silently."""
    expected = to_table(mask_settings)
    assert set(grid.questions()) == set(expected)
    cells = sum(len(row) for row in grid.table().values())
    assert cells == sum(len(row) for row in expected.values())


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def test_an_edited_value_keeps_the_type_it_had(grid):
    """A table hands back strings. Writing "12" where 12 was is a settings
    file that has quietly changed meaning."""
    assert grid.set_value("background", "cell", "42") is True

    out = grid.settings()
    assert out["cell_background"] == 42
    assert type(out["cell_background"]) is int


def test_a_float_setting_does_not_become_an_int(grid):
    """`organelle_cellprob_threshold` is a float and `cell_cellprob_threshold`
    an int -- the same question with two types, which is exactly the case a
    single "cast to the row's type" rule would get wrong."""
    assert grid.set_value("cellprob_threshold", "organelle", "0.5") is True
    out = grid.settings()
    assert out["organelle_cellprob_threshold"] == 0.5
    assert isinstance(out["organelle_cellprob_threshold"], float)


def test_clearing_a_cell_restores_auto_rather_than_an_empty_string(grid):
    """`None` means "work it out" -- a diameter of None is Cellpose
    estimating it. An empty string is not the same claim, and is not a
    number the pipeline can use."""
    grid.set_value("background", "cell", "7")
    assert grid.set_value("background", "cell", "") is True
    assert grid.settings()["cell_background"] is None


def test_an_unset_value_reads_as_auto_and_not_as_blank(grid):
    """A blank cell would read as "nobody has filled this in yet"."""
    from PySide6.QtCore import Qt

    model = grid._model
    for row in range(model.rowCount()):
        for column in range(model.columnCount()):
            index = model.index(row, column)
            question = model.question_at(row)
            obj = model.objects()[column]
            if model.asks(question, obj) and model.value_at(question, obj) is None:
                assert model.data(index, Qt.DisplayRole) == AUTO_TEXT
                return
    pytest.skip("this settings dict has no unset per-object value")


def test_a_question_an_object_does_not_ask_cannot_be_typed_into(grid):
    """Cytoplasm is DERIVED -- cell minus the rest -- so it has no channel to
    be found in. Writing a value there invents a key nothing reads."""
    from PySide6.QtCore import Qt

    model = grid._model
    absent = [(model.question_at(r), o)
              for r in range(model.rowCount())
              for o in model.objects()
              if not model.asks(model.question_at(r), o)]
    if not absent:
        pytest.skip("every object asks every question in this settings dict")
    question, obj = absent[0]
    index = model.index(grid.questions().index(question),
                        model.objects().index(obj))

    assert not (model.flags(index) & Qt.ItemIsEditable)
    assert grid.set_value(question, obj, "3") is False
    assert f"{obj}_{question}" not in grid.settings()


# ---------------------------------------------------------------------------
# What 326 needs
# ---------------------------------------------------------------------------

def test_one_more_organelle_is_one_more_column(grid):
    """The operation the table exists for. In the flat vocabulary this was
    twenty new settings; here the question count does not move."""
    before_questions = len(grid.questions())
    before_objects = len(grid.objects())

    assert grid.add_organelle() is True

    assert len(grid.questions()) == before_questions
    assert len(grid.objects()) == before_objects + 1
    assert "organelleb" in grid.objects()


def test_a_new_organelle_starts_where_the_first_one_is(grid):
    """Not at a global default nobody chose: a second mitochondrion is
    configured like the first until the user says otherwise.

    Driven through a question the first organelle ACTUALLY ASKS, which is not
    every question: `background` is asked of cell, nucleus and pathogen and
    not of organelle, so widening copies nothing there -- correctly, and a
    test that used it was asserting against a key neither column has.
    """
    asked = [q for q in grid.questions()
             if grid._model.asks(q, "organelle")
             and isinstance(grid._model.value_at(q, "organelle"), (int, float))
             and not isinstance(grid._model.value_at(q, "organelle"), bool)]
    assert asked, "the first organelle asks no numeric question"
    question = asked[0]
    assert grid.set_value(question, "organelle", "11") is True

    grid.add_organelle()

    out = grid.settings()
    assert out[f"organelleb_{question}"] == out[f"organelle_{question}"] == 11


def test_the_columns_are_numbered_and_never_lettered(grid):
    """Instruction 326, in the maintainer's words: "i dont want to see
    organellea b c d anywhere". The letters are a storage spelling."""
    grid.add_organelle()
    headers = [grid._model.headerData(i, __import__("PySide6.QtCore",
                                                    fromlist=["Qt"]).Qt.Horizontal)
               for i in range(grid._model.columnCount())]
    assert "Organelle 1" in headers and "Organelle 2" in headers
    assert not [h for h in headers if str(h).startswith("organelle")]


def test_the_ceiling_is_said_rather_than_hit_silently(grid):
    """Twenty-six is the alphabet running out, not a number somebody chose,
    and a button that stops working without saying why is worse than one
    that refuses out loud."""
    while grid.add_organelle():
        pass

    assert grid.next_organelle() == ""
    assert "ceiling" in grid.status_text()
    assert "alphabet" in grid.status_text()


def test_the_status_line_counts_what_364_is_about(grid):
    """The instruction's own number: how many settings this is, asked once
    each instead of once per object."""
    text = grid.status_text()
    assert "question(s)" in text and "object(s)" in text


# ---------------------------------------------------------------------------
# The table can be made taller
# ---------------------------------------------------------------------------

class TestTheTableIsExpandableDown:
    """The per-object table sizes to its rows and can be dragged taller.

    It sits as one row of a scrolling settings form, so without a height of
    its own it gets whatever the form gives it -- which put twenty-odd
    questions behind an inner scrollbar inside an outer one.
    """

    def _grid(self, qtbot):
        from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid
        from spacr.settings import get_measure_crop_settings
        grid = ObjectSettingsGrid()
        qtbot.addWidget(grid)
        grid.set_settings(get_measure_crop_settings({}))
        return grid

    def test_it_opens_tall_enough_to_show_its_rows(self, qtbot,
                                                   qt_theme_applied):
        grid = self._grid(qtbot)
        assert grid._model.rowCount() > 0, "nothing to size against"
        assert grid._table.height() >= min(grid.content_height(),
                                           grid.AUTO_TABLE_H)

    def test_a_short_table_still_has_something_to_grab(self, qtbot,
                                                       qt_theme_applied):
        """A table collapsed to its header cannot be dragged bigger."""
        grid = self._grid(qtbot)
        grid.set_user_height(1)
        assert grid._table.height() == grid.MIN_TABLE_H

    def test_a_very_long_table_stops_at_the_cap(self, qtbot,
                                                qt_theme_applied):
        """Past the cap the form is one table and the rest stops being findable."""
        grid = self._grid(qtbot)
        assert grid._table.height() <= grid.AUTO_TABLE_H

    def test_the_grip_drags_the_table_and_double_click_gives_it_back(
            self, qtbot, qt_theme_applied):
        grid = self._grid(qtbot)
        fitted = grid._table.height()
        grid.set_user_height(fitted + 120)
        assert grid._table.height() == fitted + 120
        grid.reset_user_height()
        assert grid._table.height() == fitted

    def test_a_dragged_height_survives_a_content_change(self, qtbot,
                                                        qt_theme_applied):
        """Adding an organelle must not throw away the height the user set.

        `_announce` re-fits after every content change, and a re-fit that
        ignored the user's answer would undo the drag the moment the table
        grew a column.
        """
        from spacr.settings import get_measure_crop_settings
        grid = self._grid(qtbot)
        grid.set_user_height(333)
        grid.set_settings(get_measure_crop_settings({}))
        assert grid._table.height() == 333

    def test_the_grip_is_the_same_affordance_as_the_console_handle(
            self, qtbot, qt_theme_applied):
        """One object name, so the two resize handles cannot drift apart."""
        grid = self._grid(qtbot)
        assert grid._grip.objectName() == "ConsoleSectionResizeHandle"
