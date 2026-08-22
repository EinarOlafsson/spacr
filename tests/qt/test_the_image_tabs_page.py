"""The image tabs page instead of scrolling (instruction 211).

THE VISIBLE AREA IS THE PAGE. `cells_per_page` is gone -- not hidden,
removed: the count is a CONSEQUENCE of the container size and the image
size, and a setting that contradicts the geometry produces a half-empty page
or a clipped row with no way for the user to tell which.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt              # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets.cell_montage_view import fits_on_a_page  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class TestTheCountFollowsTheGeometry:

    def test_a_bigger_viewport_holds_more(self):
        _, small = fits_on_a_page(400, 300, 96)
        _, large = fits_on_a_page(800, 600, 96)
        assert large > small

    def test_a_bigger_thumbnail_holds_fewer(self):
        _, big_thumbs = fits_on_a_page(800, 600, 128)
        _, small_thumbs = fits_on_a_page(800, 600, 48)
        assert small_thumbs > big_thumbs

    def test_the_columns_follow_the_width(self):
        narrow, _ = fits_on_a_page(400, 600, 96)
        wide, _ = fits_on_a_page(1200, 600, 96)
        assert wide > narrow

    def test_a_viewport_too_small_still_shows_one(self):
        """An empty page is indistinguishable from a well with no cells."""
        columns, count = fits_on_a_page(10, 10, 96)
        assert columns == 1 and count == 1

    def test_the_page_holds_whole_rows(self):
        columns, count = fits_on_a_page(800, 600, 96)
        assert count % columns == 0, (
            "a page ending mid-row is the clipped row this replaces")


class TestTheSettingIsGone:

    def test_not_in_the_picture_settings(self):
        from spacr.picture_settings import ALL_KEYS, OWN_DEFAULTS

        assert "cells_per_page" not in ALL_KEYS
        assert "cells_per_page" not in OWN_DEFAULTS

    def test_not_in_the_tooltips(self):
        from spacr.settings import tooltips

        assert "cells_per_page" not in tooltips

    def test_a_saved_preference_carrying_it_is_migrated_out(self):
        """"the setting's current value has to be migrated out rather than
        left in the saved preferences to confuse a later reader"."""
        from spacr.picture_settings import drop_retired

        out, notes = drop_retired({"cells_per_page": 60, "img_size": 96})
        assert "cells_per_page" not in out
        assert out["img_size"] == 96
        assert notes and "cells_per_page" in notes[0]

    def test_the_note_says_why_rather_than_only_that(self):
        """A reader who meets an unexplained key in their own settings file
        has no way to find out it was deliberate."""
        from spacr.picture_settings import RETIRED

        assert "contradict" in RETIRED["cells_per_page"]

    def test_nothing_to_drop_says_nothing(self):
        from spacr.picture_settings import drop_retired

        out, notes = drop_retired({"img_size": 96})
        assert out == {"img_size": 96} and notes == []


class TestTheTab:

    @pytest.fixture
    def tab(self, app):
        import pandas as pd

        from spacr.qt.widgets.cell_montage_view import _WellTab

        one = _WellTab("p1_r1_c1", "A01")
        rows = pd.DataFrame({"prcfo": [f"p_r_c_f_o{i}" for i in range(200)]})
        one.set_content(rows, [None] * 200, "caption", columns=5)
        one.resize(800, 600)
        return one

    def test_there_is_no_scrollbar(self, tab):
        """A page that scrolls is not a page -- it is a grid with a smaller
        window over it, which is what this replaces."""
        assert tab._scroll.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert tab._scroll.horizontalScrollBarPolicy() == \
            Qt.ScrollBarAlwaysOff

    def test_the_page_size_is_measured_not_configured(self, tab):
        assert tab.per_page() == fits_on_a_page(
            tab._scroll.viewport().width(),
            tab._scroll.viewport().height(), tab._thumb_px)[1]

    def test_next_moves_by_one_page(self, tab):
        if tab.page_count() < 2:
            pytest.skip("this viewport holds every crop")
        assert tab.show_page(1) == 1

    def test_neither_end_wraps(self, tab):
        assert tab.show_page(-5) == 0
        assert tab.show_page(9999) == tab.page_count() - 1

    def test_the_page_count_is_stated(self, tab):
        """"paging without a total is navigation without a map"."""
        tab._refresh_pager()
        if tab.page_count() < 2:
            pytest.skip("one page, so no pager")
        assert "of" in tab._page_label.text()

    def test_the_reader_keeps_their_place_across_a_resize(self, tab):
        """Keeping the PAGE NUMBER teleports the reader: the same page
        number is a different set of cells once the page holds a different
        number of them."""
        if tab.page_count() < 3:
            pytest.skip("too few pages to move within")
        tab.show_page(2)
        anchor = tab.first_on_page()
        tab.resize(500, 400)
        assert tab.first_on_page() <= anchor
        assert tab.first_on_page() + tab.per_page() > anchor, (
            "the crop the reader was looking at is no longer on the page")


class TestThePageAndTheLayoutAgree:
    """Reported 2026-08-21: "the cells kind of fit into the container but i
    see cells to the right . and never more that two rows ... where there
    could be 3 almost 4 instead of the next page".

    TWO NUMBERS MEANT ONE THING. The view computed columns from a fixed cell
    size over the whole tab width; the page size came from the real
    thumbnail size over the scroll area's viewport. The page held one
    number's worth of cells laid out at the other's.
    """

    @pytest.fixture
    def tab(self, app):
        import pandas as pd

        from spacr.qt.widgets.cell_montage_view import _WellTab

        one = _WellTab("p1_r1_c1", "A01")
        one.resize(700, 420)
        one.show()
        QApplication.processEvents()
        rows = pd.DataFrame({"prcfo": [f"x{i}" for i in range(60)]})
        one.set_content(rows, [None] * 60, "caption", columns=3)
        QApplication.processEvents()
        return one

    def test_the_layout_uses_the_measured_column_count(self, tab):
        columns, _per = tab.geometry_page()
        assert tab._columns == columns

    def test_the_hint_from_the_caller_does_not_win(self, tab):
        """It is derived from a fixed cell size and the whole tab width,
        which is not what the grid is drawn into."""
        tab.fill(2)
        assert tab._columns == tab.geometry_page()[0]

    def test_a_page_is_whole_rows_at_that_count(self, tab):
        columns, per_page = tab.geometry_page()
        assert per_page % columns == 0

    def test_nothing_is_drawn_past_the_page(self, tab):
        """Cells to the right of the container is the overflow this fixes."""
        assert len(tab.thumbs()) <= tab.per_page()

    def test_the_last_row_is_not_charged_for_spacing(self):
        """n items span n*thumb + (n-1)*spacing, so a page with room for
        exactly three rows must offer three, not two."""
        from spacr.qt.widgets.cell_montage_view import fits_on_a_page

        thumb, gap = 96, 6
        exactly_three = 3 * thumb + 2 * gap
        _columns, per_page = fits_on_a_page(700, exactly_three, thumb,
                                            spacing=gap)
        columns, _ = fits_on_a_page(700, exactly_three, thumb, spacing=gap)
        assert per_page // columns == 3

    def test_and_the_same_for_the_columns(self):
        from spacr.qt.widgets.cell_montage_view import fits_on_a_page

        thumb, gap = 96, 6
        exactly_five = 5 * thumb + 4 * gap
        columns, _per = fits_on_a_page(exactly_five, 400, thumb, spacing=gap)
        assert columns == 5
