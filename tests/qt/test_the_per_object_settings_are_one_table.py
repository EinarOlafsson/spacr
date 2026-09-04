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

from spacr.object_settings_table import to_table  # noqa: E402
from spacr.organelle_types import NUMBER_OF_ORGANELLES, organelle_role  # noqa: E402
from spacr.qt.widgets.object_settings_grid import (  # noqa: E402
    AUTO_TEXT,
    MAX_ORGANELLES,
    ObjectSettingsGrid,
    _kind_of,
)


@pytest.fixture
def mask_settings():
    """The real thing: Mask's own defaults, not a fixture written to suit.

    WITH ONE ORGANELLE DECLARED. `number_of_organelles` is what decides how
    many organelle columns the table draws, and the shipped defaults leave
    every slot unset -- which is a count of zero and correctly no organelle
    column at all. A fixture that left it there would be testing the
    no-organelle case in every test that mentions "the first organelle".
    """
    from spacr.organelle_types import NUMBER_OF_ORGANELLES
    from spacr.settings import get_timelapse_settings

    settings = get_timelapse_settings()
    settings[NUMBER_OF_ORGANELLES] = 1
    return settings


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


def test_the_table_claims_only_what_more_than_one_kind_asks(
        grid, mask_settings):
    """The table holds the shared questions; the rest stay in the form.

    A table is a claim that its rows and columns are independent, and a
    question a single KIND of object asks is not a row -- it is one setting,
    and it belongs in the form beside that object's others. The organelle's
    own detection settings are 34 of these: ridge filters, hysteresis, LoG
    sigmas, which no cell, nucleus or pathogen has, and which would be rows
    of mostly blank cells.

    KINDS, NOT COLUMNS, and this test was wrong twice before it was right.
    It first demanded EVERY object ask, which emptied Measure's table
    completely -- Measure asks `mask_dim` of four of its five objects and
    `max_size` of three, so requiring all five left nothing. Counting
    columns instead made the row set depend on the organelle count, since a
    second organelle is seeded from the first. Counting kinds is stable
    under both.
    """
    roles = {_kind_of(obj) for obj in grid.objects()}
    for question, row in grid.table().items():
        asking = {_kind_of(obj) for obj in row}
        assert len(asking) > 1, (
            f"{question!r} is in the table but only {sorted(asking)} asks it")
    shown = to_table(mask_settings)
    for question, row in shown.items():
        asking = {_kind_of(obj) for obj in row}
        if len(asking & roles) > 1 and question not in grid.questions():
            raise AssertionError(f"{question!r} is shared and was dropped")


def test_nothing_the_table_drops_is_lost(grid, mask_settings):
    """A dropped question stays in the settings dict, so the form still has it.

    The grid claims only the keys it shows. A key it dropped but still claimed
    would be hidden from the form as well and reachable from nowhere at all,
    which is worse than either place on its own.
    """
    out = grid.settings()
    missing = [key for key in mask_settings if key not in out]
    assert not missing, f"the table lost {missing[:5]}"


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def test_an_edited_value_keeps_the_type_it_had(grid):
    """A table hands back strings. Writing "12" where 12 was is a settings
    file that has quietly changed meaning."""
    assert grid.set_value("min_area", "cell", "42") is True

    out = grid.settings()
    assert out["cell_min_area"] == 42
    assert type(out["cell_min_area"]) is int


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
    grid.set_value("min_area", "cell", "7")
    assert grid.set_value("min_area", "cell", "") is True
    assert grid.settings()["cell_min_area"] is None


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
    """The ceiling is the naming scheme running out, not a number somebody
    chose, and a button that stops working without saying why is worse than
    one that refuses out loud.

    SEEDED AT THE CEILING RATHER THAN WALKED TO IT. This used to press Add
    until it refused, which was fine at 26 slots; 326 raised the ceiling to
    702 and each Add rebuilds the whole table, so the walk became quadratic
    and the test stopped finishing at all. The refusal is what is under
    test, and it does not care how the grid arrived at a full house.
    """
    full = {NUMBER_OF_ORGANELLES: MAX_ORGANELLES, "cell_mask_dim": 0}
    for number in range(1, MAX_ORGANELLES + 1):
        full[f"{organelle_role(number)}_mask_dim"] = 0
    grid.set_settings(full)

    assert grid.next_organelle() == ""
    assert grid.add_organelle() is False
    assert "ceiling" in grid.status_text()
    assert "lettered" in grid.status_text()


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


# ---------------------------------------------------------------------------
# `number_of_organelles` is the source of truth for the columns
# ---------------------------------------------------------------------------

class TestTheCountDecidesTheColumns:
    """One place decides how many organelle columns there are, and it is the
    setting every other reader of these settings already goes by."""

    def _grid_at(self, qtbot, count):
        from spacr.organelle_types import NUMBER_OF_ORGANELLES
        from spacr.settings import get_timelapse_settings
        settings = get_timelapse_settings()
        settings[NUMBER_OF_ORGANELLES] = count
        grid = ObjectSettingsGrid()
        qtbot.addWidget(grid)
        grid.set_settings(settings)
        return grid

    def _organelles(self, grid):
        return [o for o in grid.objects() if o.startswith("organelle")]

    def test_zero_organelles_is_no_organelle_column(self, qtbot,
                                                    qt_theme_applied):
        """The settings dict keeps a placeholder for every slot -- that is what
        makes lowering the count reversible -- so a table built off the keys
        showed an Organelle 1 column for something the run will not segment.
        """
        assert self._organelles(self._grid_at(qtbot, 0)) == []

    @pytest.mark.parametrize("count", [1, 2, 3, 5])
    def test_the_column_count_is_the_setting(self, qtbot, qt_theme_applied,
                                             count):
        assert len(self._organelles(self._grid_at(qtbot, count))) == count

    def test_lowering_the_count_hides_rather_than_deletes(self, qtbot,
                                                          qt_theme_applied):
        """A hidden slot's answers survive, or lowering the count would be a
        destructive edit disguised as a display change."""
        from spacr.organelle_types import NUMBER_OF_ORGANELLES
        grid = self._grid_at(qtbot, 2)
        question = next(q for q in grid.questions()
                        if grid._model.asks(q, "organelleb"))
        grid.set_value(question, "organelleb", "17")
        kept = grid.settings()

        kept[NUMBER_OF_ORGANELLES] = 1
        grid.set_settings(kept)
        assert self._organelles(grid) == ["organelle"]
        assert grid.settings()[f"organelleb_{question}"] == 17

    def test_adding_an_organelle_raises_the_count(self, qtbot,
                                                  qt_theme_applied):
        """Otherwise the column is one the rest of the application does not
        believe in, and vanishes the next time the table is rebuilt."""
        from spacr.organelle_types import organelle_count
        grid = self._grid_at(qtbot, 1)
        assert grid.add_organelle() is True
        assert organelle_count(grid.settings()) == 2
        assert len(self._organelles(grid)) == 2

    def test_adding_an_organelle_keeps_unsaved_edits(self, qtbot,
                                                     qt_theme_applied):
        """The rebuild reads the settings dict, and what is on screen may not
        be in it yet. Without folding the edits back first, pressing Add
        reverts every cell the user has typed into."""
        grid = self._grid_at(qtbot, 1)
        question = next(q for q in grid.questions()
                        if grid._model.asks(q, "cell"))
        grid.set_value(question, "cell", "23")
        grid.add_organelle()
        assert grid.settings()[f"cell_{question}"] == 23


# ---------------------------------------------------------------------------
# "auto" is not what an unset channel means
# ---------------------------------------------------------------------------

class TestAnUnsetChannelSaysOffNotAuto:
    """`cell_channel = None` produces no cell masks, no cell table and no cell
    crops. Drawn as "auto" that reads as a promise to work a channel out,
    which is the opposite of what it does."""

    def test_an_unset_channel_reads_off(self, grid):
        from PySide6.QtCore import Qt
        row = grid.questions().index("channel")
        col = grid.objects().index("cell")
        index = grid._model.index(row, col)
        grid.set_value("channel", "cell", "")
        assert grid._model.data(index, Qt.DisplayRole) == "off"

    def test_an_unset_diameter_still_reads_auto(self, grid):
        """Most questions DO mean "work it out" -- a diameter of None is
        Cellpose estimating it. Only the channel is a switch."""
        from PySide6.QtCore import Qt
        question = next(q for q in grid.questions()
                        if "diameter" in q and grid._model.asks(q, "cell"))
        row = grid.questions().index(question)
        col = grid.objects().index("cell")
        grid.set_value(question, "cell", "")
        assert grid._model.data(grid._model.index(row, col),
                                Qt.DisplayRole) == "auto"

    def test_typing_either_word_back_clears_the_cell(self, grid):
        for word in ("off", "auto", "OFF", ""):
            grid.set_value("channel", "cell", "2")
            assert grid.set_value("channel", "cell", word) is True
            assert grid.settings()["cell_channel"] is None

    def test_every_cell_carries_the_help(self, grid):
        """Asked for: "all the rows need to have tooltips".

        On the CELLS, which is where the help ended up: each one carries the
        typed body, the API link and the setting's animation.
        """
        from PySide6.QtCore import Qt
        thin = []
        for row, question in enumerate(grid.questions()):
            for column, obj in enumerate(grid.objects()):
                if not grid._model.asks(question, obj):
                    continue
                tip = str(grid._model.data(grid._model.index(row, column),
                                           Qt.ToolTipRole) or "")
                if len(tip) <= 40:
                    thin.append(f"{obj}_{question}")
        assert not thin, f"cells with no real help: {thin[:5]}"

    def test_the_row_header_does_not_repeat_the_cells(self, grid):
        """One explanation, in one place.

        The row header is the name; the cells carry the help. A tooltip on
        the name as well is the same thing twice, in the place the pointer
        crosses on its way to the cell it wants.
        """
        from PySide6.QtCore import Qt
        for row in range(len(grid.questions())):
            assert not grid._model.headerData(row, Qt.Vertical,
                                              Qt.ToolTipRole)


# ---------------------------------------------------------------------------
# The model zoo, per object column
# ---------------------------------------------------------------------------

class TestTheModelZooIsPerColumn:
    """A cell and a pathogen are not segmented by the same checkpoint, so the
    control that picks one has to know which column it is in.

    THE WHOLE CELL IS THE CONTROL. A button small enough to fit in a table
    cell has no room for a word, so nothing is drawn and the cell itself is
    clicked.
    """

    def _view_index(self, grid, question, obj):
        source = grid._model.index(list(grid.questions()).index(question),
                                   grid.objects().index(obj))
        mapper = getattr(grid._table.model(), "mapFromSource", None)
        return mapper(source) if mapper else source

    def test_nothing_is_drawn_in_the_cell(self, grid):
        from spacr.qt.widgets.object_settings_grid import MODEL_QUESTION
        for obj in grid.objects():
            index = self._view_index(grid, MODEL_QUESTION, obj)
            assert grid._table.indexWidget(index) is None, (
                f"{obj} has a widget in its model cell")

    def test_clicking_a_model_cell_picks_for_that_object(self, grid,
                                                         monkeypatch):
        import spacr.qt.widgets.model_zoo_picker as zoo
        from spacr.qt.widgets.object_settings_grid import MODEL_QUESTION
        monkeypatch.setattr(zoo, "choose_model", lambda *a, **k: "/m/x.pt")
        before = grid.settings().get("nucleus_model_name")

        grid._table.clicked.emit(self._view_index(grid, MODEL_QUESTION,
                                                  "cell"))
        out = grid.settings()
        assert out["cell_model_name"] == "/m/x.pt"
        assert out.get("nucleus_model_name") == before, (
            "picking a model for one object changed another")

    def test_clicking_any_other_row_opens_nothing(self, grid, monkeypatch):
        """Only the model row is a control; every other cell is just a cell."""
        import spacr.qt.widgets.model_zoo_picker as zoo
        opened = []
        monkeypatch.setattr(zoo, "choose_model",
                            lambda *a, **k: opened.append(1) or "/m/x.pt")
        grid._table.clicked.emit(self._view_index(grid, "channel", "cell"))
        assert not opened

    def test_a_cancelled_picker_leaves_the_model_alone(self, grid,
                                                       monkeypatch):
        """Cancelling is not an instruction to forget the model already set."""
        import spacr.qt.widgets.model_zoo_picker as zoo
        from spacr.qt.widgets.object_settings_grid import MODEL_QUESTION
        grid.set_value(MODEL_QUESTION, "cell", "cpsam")
        monkeypatch.setattr(zoo, "choose_model", lambda *a, **k: None)
        assert grid.choose_model_for("cell") is False
        assert grid.settings()["cell_model_name"] == "cpsam"

    def test_the_cell_says_it_is_clickable(self, grid):
        """A control that is not drawn has to be findable some other way."""
        from PySide6.QtCore import Qt

        from spacr.qt.widgets.object_settings_grid import MODEL_QUESTION
        index = grid._model.index(
            list(grid.questions()).index(MODEL_QUESTION),
            grid.objects().index("cell"))
        tip = str(grid._model.data(index, Qt.ToolTipRole) or "")
        assert "Click this cell" in tip


# ---------------------------------------------------------------------------
# The tooltips are the form's tooltips
# ---------------------------------------------------------------------------

class TestTheTooltipsMatchTheForm:
    """Asked for: the table's help should be structured like the rest of the
    application's, carry an API reference, and show the setting's animation
    where the flat form shows one.

    So it goes through the SAME popup rather than a second one. A table has no
    widget per cell, so the anchor is the view and the cell under the pointer
    decides which setting it speaks for.
    """

    def _rect_of(self, grid, question, obj):
        source = grid._model.index(list(grid.questions()).index(question),
                                   grid.objects().index(obj))
        mapper = getattr(grid._table.model(), "mapFromSource", None)
        index = mapper(source) if mapper else source
        return grid._table.visualRect(index)

    def test_the_cell_under_the_pointer_names_its_setting(self, grid):
        grid.resize(900, 500)
        rect = self._rect_of(grid, "channel", "cell")
        assert grid._key_under(rect.center()) == "cell_channel"

    def test_the_anchor_carries_the_setting_and_the_module(self, grid):
        """The popup looks up its ANIMATION by the anchor's settingKey, so an
        anchor that does not carry one gets help with no animation."""
        grid.set_app_key("mask")
        grid.resize(900, 500)
        grid._offer_tooltip(self._rect_of(grid, "channel", "cell").center())
        assert grid._table.property("settingKey") == "cell_channel"
        assert grid._table.property("settingsAppKey") == "mask"

    def test_the_help_carries_an_api_reference(self, grid):
        from spacr.qt.screens.settings_model import format_tooltip, get_tooltips
        html = format_tooltip(str(get_tooltips().get("cell_channel") or ""),
                              "mask", "cell_channel")
        assert "<a" in html and "href" in html, "no API reference in the help"

    def test_the_native_tooltip_is_swallowed(self, grid):
        """It disappears the moment the pointer moves toward the API link, and
        that link is the point."""
        from PySide6.QtCore import QEvent
        assert grid.eventFilter(grid._table.viewport(),
                                QEvent(QEvent.Type.ToolTip)) is True

    def test_a_cell_nobody_asks_offers_no_help(self, grid):
        """Cytoplasm has no channel; a tooltip there would explain a setting
        that does not exist for it."""
        objects = [o for o in grid.objects()
                   if not grid._model.asks("channel", o)]
        if not objects:
            pytest.skip("every object in this settings dict asks for a channel")
        rect = self._rect_of(grid, "channel", objects[0])
        assert grid._key_under(rect.center()) == ""
