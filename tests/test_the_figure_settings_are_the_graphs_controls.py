"""The figure settings groups C, D and E (instruction 200).

C. THE MARKS -- palette, colour scheme, size, style.
D. THE FURNITURE -- ONE control for every line, not one per line.
E. THE SHAPE OF THE PAGE -- a named ratio rather than two boxes of inches.

A TAB WHOSE CONTROLS DO NOTHING IS WORSE THAN A MISSING TAB, so each of
these is asserted against the resolver that reads it rather than against
its presence in a list.
"""
from __future__ import annotations

import pytest

from spacr.figure_style import (GENERAL_DEFAULTS, PAGE_SHAPES, STYLE_CHOICES,
                                chrome_of, page_size)


class TestCTheMarks:

    def test_the_colour_scheme_offers_the_three_asked_for(self):
        assert STYLE_CHOICES["mark_colouring"] == ("group", "uniform",
                                                   "random")

    def test_by_group_is_the_default(self):
        """The house rule is that everything is grey except what the
        sentence is about."""
        assert GENERAL_DEFAULTS["mark_colouring"] == "group"

    def test_the_marker_style_is_a_closed_list(self):
        """A named set from the house shapes, not a free-text matplotlib
        code the user has to know."""
        assert len(STYLE_CHOICES["marker_style"]) >= 4
        assert "o" in STYLE_CHOICES["marker_style"]

    def test_the_palette_is_a_named_set(self):
        """Not a colour picker per point."""
        assert "colorblind" in STYLE_CHOICES["palette"]

    def test_the_size_is_already_a_setting(self):
        assert "marker_size" in GENERAL_DEFAULTS


class TestDTheFurnitureIsOneInk:
    """"ONE CONTROL FOR THE LINES, not one per line. The axis spines, the
    periphery box and the grid are the same ink -- they are the frame"."""

    def test_there_is_one_control(self):
        assert "chrome_colour" in GENERAL_DEFAULTS

    def test_it_reaches_every_piece(self):
        style = {"chrome_colour": "#888888"}
        for element in ("grid", "spine", "box"):
            assert chrome_of(style, element) == "#888888"

    def test_a_per_element_override_wins_where_it_is_set(self):
        """"A per-element override can exist under it; the one control is
        what a user reaches for"."""
        style = {"chrome_colour": "#888888", "grid_colour": "#EEEEEE"}
        assert chrome_of(style, "grid") == "#EEEEEE"

    def test_and_only_where_it_is_set(self):
        style = {"chrome_colour": "#888888", "grid_colour": "#EEEEEE"}
        assert chrome_of(style, "spine") == "#888888"

    def test_the_default_follows_the_ink_rather_than_a_literal(self):
        """A frame pinned to a literal while the text follows the theme is a
        figure that looks wrong in one of the two themes -- the reason 178
        measured eleven times."""
        assert GENERAL_DEFAULTS["chrome_colour"] == ""
        assert chrome_of({}) == ""

    def test_an_empty_override_does_not_win(self):
        assert chrome_of({"chrome_colour": "#888", "grid_colour": ""},
                         "grid") == "#888"


class TestETheShapeOfThePage:

    def test_the_shapes_are_named(self):
        """"A named ratio -- square, portrait, landscape, wide -- rather
        than two boxes of inches"."""
        for name in ("square", "portrait", "landscape", "wide"):
            assert name in STYLE_CHOICES["page_shape"]

    def test_square_is_square(self):
        assert page_size("square", 8.0) == (8.0, 8.0)

    def test_landscape_is_wider_than_tall(self):
        width, height = page_size("landscape", 8.0)
        assert width > height

    def test_portrait_is_taller_than_wide(self):
        width, height = page_size("portrait", 8.0)
        assert height > width

    def test_the_ratio_holds_when_the_size_changes(self):
        """"it keeps the two axes consistent when the size changes"."""
        small = page_size("wide", 4.0)
        large = page_size("wide", 8.0)
        assert small[0] / small[1] == pytest.approx(large[0] / large[1])

    def test_custom_keeps_the_inches(self):
        """"The inches stay for a user who wants a journal's exact column
        width", and returning a square for 'custom' would overwrite them."""
        assert "custom" in STYLE_CHOICES["page_shape"]
        assert "custom" not in PAGE_SHAPES
        with pytest.raises(KeyError):
            page_size("custom", 8.0)


class TestTheDialogPicksThemUp:
    """A setting the dialog cannot render is a setting the user cannot
    reach."""

    @pytest.mark.parametrize("name", ["mark_colouring", "marker_style",
                                      "page_shape"])
    def test_the_choices_reach_the_dialog(self, name):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from spacr.qt.widgets.figure_settings import style_choices_for

        QApplication.instance() or QApplication([])
        assert style_choices_for(name) == STYLE_CHOICES[name]

    def test_every_new_default_is_a_style_key(self):
        """F: one vocabulary, two scopes -- a figure setting with no default
        is a setting that will drift out of one of the two."""
        for name in ("mark_colouring", "marker_style", "page_shape",
                     "chrome_colour"):
            assert name in GENERAL_DEFAULTS
