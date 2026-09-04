"""The dock is an icon, a name and a category heading, and nothing else.

Written 2026-09-03 with the rewrite it covers, replacing five files that
asserted behaviour the maintainer asked to be removed:

  test_the_dock_is_a_translucent_slab       the slab is gone
  test_the_dock_and_menu_show_the_folded_modules   the second level is gone
  test_no_dock_widget_fills_a_background    the row has no paintEvent at all
  test_the_dock_does_not_relayout_on_hover  kept here, as one test
  test_the_dock_names_itself_on_hover       the name is now ALWAYS drawn

The two invariants worth carrying over are kept: hovering must not move any
geometry, and a row must not paint a box behind itself. Both were real bugs
-- the blink and the "black box" -- and both are now structural rather than
defended, so each is one test instead of a file.
"""
from __future__ import annotations

import pytest

from spacr.qt.app import MainWindow
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QLabel

from spacr.qt.widgets.dock import HOME_KEY, Dock, DockRow

ROWS = [
    ("__home__", "Home", "Back to the tiles", "Core"),
    ("mask", "Generate Masks", "Segment objects", "Core"),
    ("measure", "Measure", "Extract features", "Core"),
    ("regress", "Regression", "Fit models", "Data"),
    ("alpha_only", "Alpha Only", "Not stable yet", "Tools"),
]


@pytest.fixture()
def dock(qtbot):
    """A dock over known rows, with everything visible."""
    bar = Dock(ROWS, icon_for=lambda key: None)
    qtbot.addWidget(bar)
    return bar


def _hover(row, entered: bool) -> None:
    point = QPointF(1, 1)
    if entered:
        row.enterEvent(QEnterEvent(point, point, point))
    else:
        row.leaveEvent(QEvent(QEvent.Type.Leave))


# -- what it draws ---------------------------------------------------------
def test_every_row_shows_its_name_at_rest(dock):
    """The old dock drew the name only while hovered, so a resting dock was
    a column of unlabelled glyphs. Every row now carries its name."""
    assert [row.text() for row in dock.rows()] == [r[1] for r in ROWS]
    for row in dock.rows():
        assert row.text(), f"{row.key} has no name at rest"


def test_a_row_carries_its_key_for_everything_outside_this_module(dock):
    """``navKey`` is how the tutorial highlighter, the icon refresh and the
    maturity tests find a row."""
    assert [row.property("navKey") for row in dock.rows()] == [r[0] for r in ROWS]


def test_a_row_has_no_popup_tooltip(dock):
    """The explanation belongs in the strip along the bottom. A popup here
    would be a second one, in the place the pointer is already covering."""
    for row in dock.rows():
        assert row.toolTip() == ""


def test_a_row_paints_no_box_of_its_own(dock):
    """The "black box" four commits chased was the dock painting itself.
    A row that does not override paintEvent cannot grow one again."""
    assert "paintEvent" not in vars(DockRow)
    assert "paintEvent" not in vars(Dock)


# -- the one effect --------------------------------------------------------
def test_hovering_a_row_is_the_only_effect_and_it_is_the_accent(dock):
    from spacr.qt.theme import active_palette
    rule = dock.styleSheet()
    accent = active_palette()["accent"]
    assert f"QPushButton#SidebarItem:hover {{ color: {accent}; }}" in rule


def test_a_row_keeps_the_object_name_the_theme_styles(dock):
    """The theme carries eight ``QPushButton#SidebarItem`` rules; renaming
    the row would silently un-style the whole dock."""
    assert {row.objectName() for row in dock.rows()} == {"SidebarItem"}


def test_hovering_names_the_module_for_the_bottom_strip(dock, qtbot):
    seen = []
    dock.module_hovered.connect(seen.append)
    _hover(dock.rows()[1], True)
    assert seen == ["mask"]
    assert dock.rows()[1].is_hovered()


def test_leaving_does_not_clear_the_bottom_strip(dock):
    """The bar keeps its last module for thirty seconds on purpose: a link
    that vanishes when the pointer sets off toward it cannot be clicked."""
    seen = []
    dock.module_hovered.connect(seen.append)
    row = dock.rows()[1]
    _hover(row, True)
    _hover(row, False)
    assert seen == ["mask"], "leaving must not emit, or the bar would empty"
    assert not row.is_hovered()


def test_a_hover_sweep_moves_no_geometry(dock, qtbot):
    """The blink was the old dock resizing icons under the pointer, which
    relaid the column out. Nothing may move now."""
    dock.show()
    qtbot.waitExposed(dock)
    before = [row.geometry() for row in dock.rows()]
    for row in dock.rows():
        _hover(row, True)
        _hover(row, False)
    assert [row.geometry() for row in dock.rows()] == before


def test_clicking_a_row_asks_for_that_module(dock):
    chosen = []
    dock.nav_selected.connect(chosen.append)
    dock.rows()[2].click()
    assert chosen == ["measure"]


# -- categories ------------------------------------------------------------
def test_the_headings_are_the_sections_in_order(dock):
    assert dock.sections() == ["Core", "Data", "Tools"]


def test_a_heading_keeps_the_legacy_object_name(dock):
    """The theme styles ``SidebarSection`` and the maturity test looks
    headings up by it."""
    names = {label.text() for label in dock.findChildren(QLabel)
             if label.objectName() == "SidebarSection"}
    assert names == {"Core", "Data", "Tools"}


def test_a_category_collapses_and_reopens(dock, qtbot):
    dock.show()
    qtbot.waitExposed(dock)
    assert dock.section_is_open("Core")
    assert dock.toggle_section("Core") is False
    assert all(row.isHidden() for row in dock.rows() if row.key != "regress"
               and dock._section_of[row.key] == "Core")
    assert dock.toggle_section("Core") is True
    assert not dock.rows()[0].isHidden()


def test_a_shut_heading_stays_so_it_can_be_clicked_open(dock, qtbot):
    dock.show()
    qtbot.waitExposed(dock)
    dock.toggle_section("Core")
    header = next(label for label in dock.findChildren(QLabel)
                  if label.text() == "Core")
    assert not header.isHidden()


def test_releasing_on_a_heading_toggles_it(dock, qtbot):
    """On release rather than press, so a drag that starts on a heading and
    ends elsewhere does not shut the section behind you."""
    from PySide6.QtGui import QMouseEvent

    header = next(label for label in dock.findChildren(QLabel)
                  if label.text() == "Core")
    release = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(2, 2),
                          QPointF(2, 2), Qt.MouseButton.LeftButton,
                          Qt.MouseButton.LeftButton,
                          Qt.KeyboardModifier.NoModifier)
    assert dock.eventFilter(header, release) is True
    assert not dock.section_is_open("Core")


# -- there is no second level ---------------------------------------------
def test_there_are_no_folded_child_rows(dock):
    """The sub categories were removed on request."""
    assert not any(row.property("isFoldChild") for row in dock.rows())
    assert dock.host_is_expanded("mask") is False
    assert dock.expand_host("mask") is None


# -- maturity --------------------------------------------------------------
def test_the_maturity_filter_hides_a_row_inside_an_open_section(qtbot):
    bar = Dock(ROWS, is_visible=lambda key: key != "alpha_only")
    qtbot.addWidget(bar)
    bar.show()
    qtbot.waitExposed(bar)
    hidden = {row.key for row in bar.rows() if row.isHidden()}
    assert "alpha_only" in hidden


def test_a_heading_hides_when_every_module_under_it_is_filtered_out(qtbot):
    bar = Dock(ROWS, is_visible=lambda key: key != "alpha_only")
    qtbot.addWidget(bar)
    bar.show()
    qtbot.waitExposed(bar)
    header = next(label for label in bar.findChildren(QLabel)
                  if label.text() == "Tools")
    assert header.isHidden(), "Tools holds only the filtered module"


def test_home_survives_any_filter(qtbot):
    """Home is how you get back. It is never what a maturity setting hides."""
    bar = Dock(ROWS, is_visible=lambda key: False)
    qtbot.addWidget(bar)
    bar.show()
    qtbot.waitExposed(bar)
    home = next(row for row in bar.rows() if row.key == HOME_KEY)
    assert not home.isHidden()


# -- width -----------------------------------------------------------------
def test_the_column_fits_its_longest_name_within_bounds(dock):
    from spacr.qt.preferences import scaled_px
    width = dock.fitting_width()
    assert scaled_px(Dock.WIDTH_MIN) <= width <= scaled_px(Dock.WIDTH_MAX)
    assert dock.clipped_items() == []


def test_refresh_icons_survives_a_provider_that_has_none(dock):
    dock.refresh_icons()            # must not raise
    assert dock.row_height() > 0


def test_sync_hover_reports_the_row_under_the_pointer(dock):
    assert dock.sync_hover() is None
    _hover(dock.rows()[1], True)
    assert dock.sync_hover() == "mask"


# -- carried over from test_the_dock_behaves_like_a_dock, which was retired
# with the design it described. These four invariants outlived it.
def test_every_icon_is_the_same_size_in_every_state(dock):
    """The old dock grew the icon under the pointer and shrank it again,
    and that is what relaid the column out and made it blink."""
    sizes = {(row.iconSize().width(), row.iconSize().height())
             for row in dock.rows()}
    assert len(sizes) == 1, f"rows disagree about icon size: {sizes}"
    side = sizes.pop()[0]
    assert side >= 16, "an icon this small reads as the bullet it replaced"
    _hover(dock.rows()[1], True)
    assert dock.rows()[1].iconSize().width() == side


def test_every_row_says_what_it_is(dock):
    """A long name elides to fit the column, so the full one has to survive
    somewhere a screen reader can reach."""
    for row in dock.rows():
        assert row.accessibleName(), f"{row.key} has no accessible name"
        assert row.property("navKey"), f"{row.key} lost its navKey"


def test_a_row_carries_the_key_the_hint_strip_reads(dock):
    """``moduleAppKey`` is :data:`spacr.qt.module_hints.KEY_PROPERTY`. Setting
    it is what makes the dock explain itself through the SAME mechanism as
    the menus and the tiles rather than a second one of its own."""
    from spacr.qt.module_hints import KEY_PROPERTY

    for row in dock.rows():
        assert row.property(KEY_PROPERTY) == row.key


def test_the_collapsed_dock_fits_a_900px_laptop(dock, qtbot):
    """Sections were collapsed in instruction 330 precisely to stop the dock
    asking for more height than the window has."""
    dock.show()
    qtbot.waitExposed(dock)
    tall = (sum(header.sizeHint().height()
                for header in dock._headers.values()
                if not header.isHidden())
            + sum(r.sizeHint().height() for r in dock.rows()
                  if not r.isHidden()))
    assert tall <= 900, f"the collapsed dock asks for {tall} px of 900"


# ---------------------------------------------------------------------------
# The column has a ground of its own
# ---------------------------------------------------------------------------

def test_the_container_paints_no_box_behind_the_rounded_one(qtbot,
                                                            qt_theme_applied):
    """The blanket `QWidget` rule was the box, and this is what stops it.

    The application sheet carries `QWidget { background-color: bg }`, so any
    untagged container paints an opaque rectangle -- and a plain QWidget
    holding a rounded panel is a square of `bg` behind rounded corners.
    Colouring it only changes which colour the rectangle is; three attempts
    at that proved it. `Panel` in home.py already had the answer and says why
    in as many words: untagged wrappers "read as one large black column
    behind every panel".
    """
    from PySide6.QtCore import Qt
    from spacr.qt.widgets.dock import PANEL_RADIUS, Dock

    dock = Dock([("mask", "Mask", "", "Segment")])
    qtbot.addWidget(dock)

    # make_transparent's mark. Without it the container fills.
    assert not dock.autoFillBackground()
    assert dock.testAttribute(Qt.WidgetAttribute.WA_StyledBackground) is False \
        or "background: transparent" in dock.styleSheet() \
        or dock.property("spacrTransparent"), (
            "the container was not made transparent")

    # And the rounded box is a frame INSIDE it, styled as HomePanelBox is.
    sheet = dock.styleSheet()
    assert "QFrame#DockPanel {" in sheet
    assert f"border-radius: {PANEL_RADIUS}px" in sheet


def test_the_dock_row_escapes_an_ampersand(qtbot, qt_theme_applied):
    """Qt reads a bare `&` as a mnemonic: "Align & Stitch" drew as
    "Align _Stitch", the ampersand gone and the S underlined."""
    from spacr.qt.widgets.dock import DockRow

    row = DockRow("align", "Align & Stitch", "", None)
    qtbot.addWidget(row)
    assert "&&" in row.full_text()
    # The accessible name keeps the real character -- a screen reader must
    # not say the escape.
    assert row.accessibleName() == "Align & Stitch"
