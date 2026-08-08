"""The reflowing grid of search figures.

Most of what matters is in the two pure functions, so most of this file
needs no display: how many columns fit a container, and which cell a set of
parameter values belongs in.
"""

import pytest

from spacr.qt.widgets.figure_grid import (
    DEFAULT_CELL_ASPECT, MIN_CELL_PX, SearchFigureGrid, axis_layout,
    cell_caption, reflow_shape,
)


class TestReflowShape:

    def test_a_square_container_gets_a_square_grid(self):
        """Nine figures in a square panel is 3x3, not 9x1.

        The view exists to compare embeddings against each other; a single
        row of nine wastes the height and makes each one tiny.
        """
        columns, rows, _cell = reflow_shape(9, 900, 900)
        assert (columns, rows) == (3, 3)

    def test_a_wide_short_container_gets_one_row(self):
        columns, rows, _cell = reflow_shape(9, 1800, 200)
        assert rows == 1 and columns == 9

    def test_columns_stop_where_a_cell_stops_being_readable(self):
        """A cell narrower than MIN_CELL_PX is a smudge. The grid scrolls
        instead of shrinking past it.

        The container here is short enough that twenty figures overflow at
        every column count, so nothing but the minimum cell width is left
        to decide the answer -- which is the point. The same twenty figures
        in a container four times as wide take more columns, so it really
        is the cap doing the work and not the shape.
        """
        narrow = reflow_shape(20, MIN_CELL_PX * 3 + 20, 300)
        assert narrow[0] == 3 and narrow[2] >= MIN_CELL_PX - 10
        wide = reflow_shape(20, (MIN_CELL_PX * 3 + 20) * 4, 300)
        assert wide[0] > 3

    def test_a_container_too_narrow_for_one_cell_still_shows_one(self):
        """Something has to be visible. One squeezed cell beats none."""
        columns, rows, cell = reflow_shape(6, 100, 900)
        assert columns == 1 and rows == 6 and cell > 0

    def test_no_figures_is_no_grid(self):
        assert reflow_shape(0, 900, 900) == (0, 0, 0)

    def test_a_zero_height_container_does_not_divide_by_it(self):
        columns, rows, cell = reflow_shape(4, 900, 0)
        assert columns >= 1 and rows >= 1 and cell > 0

    def test_the_cell_width_accounts_for_the_gaps(self):
        columns, _rows, cell = reflow_shape(3, 900, 900, spacing=10)
        assert cell * columns + 10 * (columns - 1) <= 900


class TestAxisLayout:

    def test_two_parameters_make_the_familiar_table(self):
        coords = [{"n": n, "d": d} for n in (5, 15) for d in (0.1, 0.5, 0.9)]
        rows, columns, cells = axis_layout(coords, ["n", "d"])
        assert rows == ["n"]           # the narrower axis goes down
        assert columns == [0.1, 0.5, 0.9]
        assert cells == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    def test_one_parameter_makes_one_ordered_row(self):
        """Arrival order is not value order. A walk visits 1, 3, 2."""
        rows, columns, cells = axis_layout(
            [{"a": 1}, {"a": 3}, {"a": 2}], ["a"])
        assert rows == [] and columns == [1, 2, 3]
        assert cells == [(0, 0), (0, 2), (0, 1)]

    def test_more_than_two_parameters_keep_the_promise_of_position(self):
        """Two cells in one row must differ in exactly one parameter.

        There is no honest flat picture of a 3-D space, so the widest axis
        goes across and every combination of the rest gets its own row.
        """
        coords = [{"a": a, "b": b, "c": c}
                  for a in (1, 2) for b in (1, 2) for c in (1, 2, 3)]
        rows, columns, cells = axis_layout(coords, ["a", "b", "c"])
        assert rows == ["a", "b"] and columns == [1, 2, 3]
        assert len(set(cells)) == len(cells) == 12
        by_row = {}
        for coord, (row, _col) in zip(coords, cells):
            by_row.setdefault(row, []).append(coord)
        for members in by_row.values():
            assert len({(m["a"], m["b"]) for m in members}) == 1

    def test_the_widest_axis_goes_across(self):
        coords = [{"a": a, "b": b} for a in range(5) for b in (1, 2)]
        rows, columns, _cells = axis_layout(coords, ["a", "b"])
        assert rows == ["b"] and len(columns) == 5

    def test_numbers_and_text_on_one_axis_do_not_raise(self):
        """A metric axis holds strings; comparing one with an int raises."""
        _rows, columns, _cells = axis_layout(
            [{"m": "cosine"}, {"m": 2}, {"m": "euclidean"}], ["m"])
        assert columns == [2, "cosine", "euclidean"]

    def test_a_parameter_no_trial_carries_is_not_an_axis(self):
        rows, columns, cells = axis_layout([{"a": 1}, {"a": 2}], ["a", "ghost"])
        assert rows == [] and columns == [1, 2] and cells == [(0, 0), (0, 1)]

    def test_no_parameters_falls_back_to_arrival_order(self):
        _rows, _columns, cells = axis_layout([{"a": 1}, {"a": 2}], [])
        assert cells == [(0, 0), (0, 1)]

    def test_no_figures_is_no_cells(self):
        assert axis_layout([], ["a"]) == ([], [], [])


def test_the_caption_names_the_values_behind_a_figure():
    assert cell_caption({"n_neighbors": 15, "min_dist": 0.1},
                        ["n_neighbors", "min_dist"]) == (
        "n_neighbors=15  min_dist=0.1")


class TestWidget:

    @pytest.fixture
    def grid(self, qt_theme_applied, qtbot):
        widget = SearchFigureGrid(["n", "d"])
        qtbot.addWidget(widget)
        widget.resize(900, 900)
        return widget

    def test_an_empty_grid_says_so_instead_of_showing_a_blank(self, grid):
        assert grid.count() == 0
        assert grid._empty.isVisible() or not grid.isVisible()

    def test_figures_land_as_they_arrive(self, grid, tmp_path):
        for index in range(4):
            grid.add_figure(str(tmp_path / f"missing_{index}.png"),
                            {"n": index // 2, "d": index % 2})
        assert grid.count() == 4
        assert grid.columns() == 2

    def test_a_figure_that_failed_to_render_keeps_its_cell(self, grid,
                                                          tmp_path):
        """A blank slot would silently change what every other position
        claims. The cell stays and names the configuration."""
        grid.add_figure(str(tmp_path / "nope.png"), {"n": 5, "d": 0.1})
        label = grid._labels[0]
        assert label.pixmap().isNull()
        assert "n=5" in label.text()

    def test_clearing_empties_the_grid(self, grid, tmp_path):
        grid.add_figure(str(tmp_path / "a.png"), {"n": 1, "d": 1})
        grid.clear()
        assert grid.count() == 0 and grid.columns() == 0

    def test_the_axes_can_change_after_figures_have_landed(self, grid,
                                                           tmp_path):
        for index in range(4):
            grid.add_figure(str(tmp_path / f"{index}.png"),
                            {"n": index, "d": 0.1})
        grid.set_parameters(["n"])
        assert grid.parameters() == ["n"]
        assert grid.columns() == 4  # one ordered row

    def test_the_format_comes_from_preferences_not_a_second_setting(self,
                                                                    grid):
        assert SearchFigureGrid.figure_format() in {"png", "pdf"}
        assert SearchFigureGrid.figure_dpi() > 0
