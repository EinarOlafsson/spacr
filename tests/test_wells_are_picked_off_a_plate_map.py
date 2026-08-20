"""185: rows, columns and wells in one vocabulary, and a map to pick them on.

"the underlying function(s) needs to be able to handle row (e.g. r1), column
(e.g. c1) and well (A01)."

WHY THE PARSER IS THE HARD HALF. A well specification that does not match is
not an error anyone sees: it is silently zero wells, and whatever was going
to be measured on them is measured on nothing while the run finishes and the
figures draw. Same failure mode as a control that matches nothing (184), same
treatment -- one reader, whole values, and a refusal that names what it
looked for AND the layout it was read against.
"""
from __future__ import annotations

import pytest

from spacr.well_spec import (DEFAULT_LAYOUT, LAYOUTS, WellSpecError, parse,
                             parse_one, row_label, row_number, shape, to_text,
                             well_label)


class TestTheFourSpellings:

    def test_a_row_is_the_whole_row(self):
        rows, columns = shape(96)

        assert len(parse_one("r1", 96)) == columns

    def test_a_column_is_the_whole_column(self):
        rows, _columns = shape(96)

        assert len(parse_one("c1", 96)) == rows

    def test_a_plate_map_well(self):
        assert parse_one("A01", 96) == {(1, 1)}

    def test_the_rc_well_spacr_already_writes(self):
        assert parse_one("r1_c1", 96) == {(1, 1)}
        assert parse_one("r2c3", 96) == {(2, 3)}

    def test_padding_does_not_matter(self):
        assert parse_one("A1", 96) == parse_one("A01", 96)

    def test_case_decides_between_a_column_and_a_well(self):
        """`C04` is row C column 4 on every plate map ever printed, and `c4`
        is column 4 in spaCR's own vocabulary (`control_wells` documents "a
        column ('c12')"). Those collide the moment the patterns are
        case-insensitive -- and they did: `C04` on a 12-well plate came back
        as the whole of column 4.
        """
        assert parse_one("c4", 96) == {(r, 4) for r in range(1, 9)}
        assert parse_one("C4", 96) == {(3, 4)}
        assert parse_one("C04", 96) == {(3, 4)}

    def test_the_refusal_teaches_the_rule(self):
        with pytest.raises(WellSpecError) as raised:
            parse_one("hello", 96)

        assert "'c4' is column 4" in str(raised.value)

    def test_the_three_forms_mix_in_one_string(self):
        """"i whould be able to choose which wells to include" -- as typed."""
        both = parse("r1, c1, A01", 96)

        assert both == parse("r1", 96) | parse("c1", 96) | {(1, 1)}


class TestTheLayoutIsPartOfTheQuestion:

    def test_a_well_beyond_the_plate_is_refused(self):
        with pytest.raises(WellSpecError, match="24-well"):
            parse_one("H12", 24)

    def test_the_refusal_names_the_layout_and_its_corner(self):
        """"H12 is not a well" is unhelpful to a reader who believes they are
        on a 96."""
        with pytest.raises(WellSpecError) as raised:
            parse_one("H12", 24)

        message = str(raised.value)
        assert "24" in message and "D06" in message

    def test_the_same_well_is_fine_on_a_bigger_plate(self):
        assert parse_one("H12", 96) == {(8, 12)}

    def test_an_unknown_layout_is_refused_with_the_offer(self):
        with pytest.raises(WellSpecError, match="1536"):
            shape(100)

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_every_layout_has_a_first_and_a_last_well(self, layout):
        rows, columns = shape(layout)

        assert parse_one("A01", layout) == {(1, 1)}
        assert parse_one(well_label(rows, columns), layout) \
            == {(rows, columns)}
        assert rows * columns == layout

    def test_a_1536_needs_two_letter_rows(self):
        assert row_label(27) == "AA"
        assert row_number("AA") == 27
        assert parse_one("AF48", 1536) == {(32, 48)}


class TestNonsenseIsRefusedNotGuessed:

    @pytest.mark.parametrize("token", ["", "  "])
    def test_blank_names_nothing(self, token):
        assert parse_one(token, 96) == set()

    @pytest.mark.parametrize("token", ["row1", "1A", "r", "c", "hello", "-"])
    def test_a_spelling_it_does_not_read_is_an_error(self, token):
        with pytest.raises(WellSpecError):
            parse_one(token, 96)

    def test_none_is_no_wells_rather_than_an_error(self):
        assert parse(None, 96) == set()


class TestWritingItBack:
    """D: "a value the dialog writes must parse back to the same selection"."""

    @pytest.mark.parametrize("text", ["r1", "c1", "A01", "r1,c2",
                                      "A01,B02,C03"])
    def test_it_round_trips(self, text):
        cells = parse(text, 96)

        assert parse(to_text(cells, 96), 96) == cells

    def test_a_whole_row_collapses_to_the_compact_form(self):
        assert to_text(parse("r3", 96), 96) == "r3"

    def test_a_whole_column_collapses_too(self):
        assert to_text(parse("c5", 96), 96) == "c5"

    def test_anything_else_is_listed_as_wells(self):
        """Not a range: inventing A01:A06 would be a fifth spelling that
        `parse` does not read, and a value the field cannot read back is
        worse than a long one."""
        assert to_text({(1, 1), (2, 2)}, 96) == "A01,B02"

    def test_nothing_selected_is_the_empty_string(self):
        assert to_text(set(), 96) == ""


class TestThePicker:

    @pytest.fixture
    def picker(self, qtbot):
        pytest.importorskip("PySide6")
        from spacr.qt.widgets.plate_map_picker import PlateMapPicker

        widget = PlateMapPicker("", layout=96)
        qtbot.addWidget(widget)
        return widget

    def test_the_three_buttons_the_ask_named(self, picker):
        assert picker.plate_button.text() == "Plate"
        assert picker.done_button.text() == "Done"
        assert picker.close_button.text() == "Close"

    def test_it_opens_on_384_by_default(self, qtbot):
        from spacr.qt.widgets.plate_map_picker import PlateMapPicker

        widget = PlateMapPicker()
        qtbot.addWidget(widget)

        assert widget._layout_size == DEFAULT_LAYOUT == 384

    def test_a_drag_selects_a_rectangle(self, picker):
        picker.select_region((2, 2), (4, 5))

        assert picker.selection() == {(r, c) for r in range(2, 5)
                                      for c in range(2, 6)}

    def test_a_drag_over_a_whole_column_writes_the_compact_form(self, picker):
        picker.select_region((1, 1), (8, 1))

        assert picker.value() == "c1"

    def test_a_drag_begun_on_a_chosen_well_clears(self, picker):
        """The spreadsheet idiom the instruction proposed."""
        picker.select_region((1, 1), (3, 3))
        picker.select_region((1, 1), (2, 2))

        assert (1, 1) not in picker.selection()
        assert (3, 3) in picker.selection()

    def test_it_opens_showing_the_field_it_was_given(self, qtbot):
        from spacr.qt.widgets.plate_map_picker import PlateMapPicker

        widget = PlateMapPicker("r2", layout=96)
        qtbot.addWidget(widget)

        assert widget.value() == "r2"

    def test_a_field_that_will_not_parse_opens_empty(self, qtbot):
        """The picker is how a user FIXES a value they typed wrong, so it
        must not refuse to open on one."""
        from spacr.qt.widgets.plate_map_picker import PlateMapPicker

        widget = PlateMapPicker("nonsense!", layout=96)
        qtbot.addWidget(widget)

        assert widget.value() == ""

    def test_changing_the_layout_keeps_what_still_fits_and_says_what_did_not(
            self, picker):
        picker.select_region((1, 1), (8, 1))       # a whole 96 column

        picker.ask_for_layout(24)

        assert len(picker.selection()) == 4, "a 24 has four rows"
        assert "dropped" in picker._caption.text(), (
            "a well that vanished with the layout is one the user still "
            "believes is chosen")


class TestTheAuditIsWrittenDown:
    """185 C: "The audit is part of the deliverable: a button next to only
    some of them is worse than none.\""""

    def test_every_well_setting_is_a_real_setting(self):
        from spacr.settings import tooltips
        from spacr.well_spec import WELL_SETTINGS

        unknown = [key for key in WELL_SETTINGS if key not in tooltips]

        assert not unknown, f"named but not settings: {unknown}"

    def test_the_well_only_list_is_a_subset(self):
        from spacr.well_spec import WELL_ONLY_SETTINGS, WELL_SETTINGS

        assert set(WELL_ONLY_SETTINGS) <= set(WELL_SETTINGS)

    def test_the_audit_still_matches_what_the_tooltips_say(self):
        """The list was FOUND by reading spaCR's own tooltips, so it has to
        keep matching them -- a setting that starts documenting wells and is
        not on the list is a field with no button beside it."""
        import re

        from spacr.settings import tooltips
        from spacr.well_spec import WELL_SETTINGS

        pattern = r"'c\d+'|'r\d+'|well location|a column \(|a row \("
        documented = {key for key, text in tooltips.items()
                      if re.search(pattern, str(text))}

        assert documented == set(WELL_SETTINGS), (
            f"missing from the audit: {sorted(documented - set(WELL_SETTINGS))}; "
            f"no longer documenting wells: "
            f"{sorted(set(WELL_SETTINGS) - documented)}")

    def test_the_mixed_vocabulary_settings_are_deliberately_left_out(self):
        """`classes` names classes and `negative_control` may name a gene or
        a guide (184). A picker that overwrote one would destroy a value it
        does not understand."""
        from spacr.well_spec import WELL_ONLY_SETTINGS

        for key in ("classes", "class_metadata", "negative_control",
                    "positive_control"):
            assert key not in WELL_ONLY_SETTINGS


class TestTheButtonReachesTheField:

    @pytest.fixture
    def screen(self, qtbot):
        pytest.importorskip("PySide6")
        from spacr.qt.screens.app_screen import AppScreen

        widget = AppScreen("regression")
        qtbot.addWidget(widget)
        return widget

    def test_the_field_is_still_what_the_panel_collects(self, screen):
        """Wrapping a field in a row must not change which object
        `collect()` reads -- a picker that made the value unreadable would be
        worse than no picker."""
        collected = screen._settings_model.collect() or {}

        assert "filter_value" in collected

    def test_a_setting_that_does_not_take_wells_is_untouched(self, screen):
        from PySide6.QtWidgets import QWidget

        widget = screen._settings_model._widgets.get("regression_type")

        assert not hasattr(widget, "_spacr_field")

    def test_picking_writes_the_specification_into_the_field(self, screen,
                                                             monkeypatch):
        from spacr.qt.widgets import plate_map_picker

        class _Chosen(plate_map_picker.PlateMapPicker):
            def exec(self):                       # noqa: A003 - Qt naming
                self.select_region((1, 1), (16, 1))
                return 1

        monkeypatch.setattr(plate_map_picker, "PlateMapPicker", _Chosen)
        field = screen._settings_model._widgets["filter_value"]

        written = screen.pick_wells_for(field, "filter_value")

        assert written == "c1"

    def test_closing_without_choosing_leaves_the_field_alone(self, screen,
                                                             monkeypatch):
        from spacr.qt.widgets import plate_map_picker

        class _Closed(plate_map_picker.PlateMapPicker):
            def exec(self):                       # noqa: A003 - Qt naming
                return 0

        monkeypatch.setattr(plate_map_picker, "PlateMapPicker", _Closed)
        field = screen._settings_model._widgets["filter_value"]
        before = field.text()

        written = screen.pick_wells_for(field, "filter_value")

        assert written == ""
        assert field.text() == before
