"""One close mark, everywhere, read from the theme rather than written out.

"whenever you have a symbol for closing, e.g. a tab after opening annotation
aggreement the close symbol should just be a large X that turns red when
hovered and is other wise white (in dark mode). this goes for all close icons
throghout the spacr application"

What these tests hold in place:

* ONE DEFINITION. The glyph, the square it occupies and its two colours live
  in :mod:`spacr.qt.theme`; a site asks for the mark and says what pressing it
  closes. A sweep of the Qt package finds no second copy of the glyph and no
  site colouring its own X.
* THE THEME'S INK, NOT ``#FFFFFF``. Measured on the DRAWN widget in both
  themes: white on dark, near-black on light, and the theme's error red under
  a real pointer in both.
* A HIT TARGET AS WELL AS A GLYPH. The mark is larger than the pixmap Qt
  draws on a closable tab, and the square a user has to hit grew with it.
* A LONG TITLE AT A NARROW WINDOW still shows its mark, inside its own tab,
  without the title being drawn over it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QLabel, QTabBar, QTabWidget, QVBoxLayout, QWidget,
)

from spacr.qt import theme

QT_PACKAGE = Path(theme.__file__).resolve().parent


# ---------------------------------------------------------------------------
# Measuring the drawn widget
# ---------------------------------------------------------------------------

def _painted(widget) -> dict:
    """Every opaque colour in the widget's own rendering, by pixel count."""
    image = widget.grab().toImage()
    counts: dict = {}
    for y in range(image.height()):
        for x in range(image.width()):
            colour = image.pixelColor(x, y)
            if colour.alpha() < 200:
                continue
            key = (colour.red(), colour.green(), colour.blue())
            counts[key] = counts.get(key, 0) + 1
    return counts


def _ink(widget, background) -> tuple:
    """The colour the glyph is drawn in: the most-painted non-background one."""
    counts = _painted(widget)
    bg = QColor(background)
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    for (red, green, blue), _n in ranked:
        distance = (abs(red - bg.red()) + abs(green - bg.green())
                    + abs(blue - bg.blue()))
        if distance > 60:
            return (red, green, blue)
    return ranked[0][0] if ranked else (0, 0, 0)


def _near(measured, expected, tolerance=12) -> bool:
    """Whether a measured RGB triple is the expected hex colour."""
    want = QColor(expected)
    return (abs(measured[0] - want.red()) <= tolerance
            and abs(measured[1] - want.green()) <= tolerance
            and abs(measured[2] - want.blue()) <= tolerance)


class _Bench:
    """A real window holding a close mark and somewhere else to point at."""

    def __init__(self, qtbot, qapp, name):
        from spacr.qt.preferences import get_font_scale

        self.saved = qapp.styleSheet()
        self.qapp = qapp
        # THE SCALE THE APP RUNS AT. A sheet built for one Zoom on a widget
        # sized for another measures a glyph nobody sees.
        qapp.setStyleSheet(theme.stylesheet(name, font_scale=get_font_scale()))
        self.palette = theme.palette_for(name)
        self.host = QWidget()
        qtbot.addWidget(self.host)
        column = QVBoxLayout(self.host)
        column.setContentsMargins(24, 24, 24, 24)
        self.away = QLabel("elsewhere", self.host)
        column.addWidget(self.away)
        self.mark = theme.close_mark_button(self.host, tooltip="Close")
        column.addWidget(self.mark)
        self.host.resize(220, 180)
        self.host.show()
        qapp.processEvents()
        # ONE PRIMING CYCLE, and it is not ceremony. A widget that happens
        # to be under the cursor when its window is SHOWN is marked as under
        # the mouse without an enter event, and Qt then has no record of it
        # to send the matching leave to -- so it stays hovered for the rest
        # of the process however far the pointer moves. Entering it once for
        # real puts that bookkeeping straight, and leaving works from then on.
        self.point_at_mark()
        self.point_away()

    def _point_at(self, widget):
        """Move the pointer onto ``widget`` through real move events.

        Parked at the window's corner first. Qt DROPS a mouse move whose
        position and buttons are unchanged, and a previous test that left
        the pointer at these very coordinates would otherwise make this
        move a no-op -- no enter, no leave, and a hover state carried in
        from a window that no longer exists.
        """
        QTest.mouseMove(self.host, QPoint(1, 1))
        self.qapp.processEvents()
        QTest.mouseMove(widget, widget.rect().center())
        self.qapp.processEvents()

    def point_away(self):
        """Put the pointer somewhere that is not the mark, and settle."""
        self._point_at(self.away)

    def point_at_mark(self):
        """Put the pointer on the mark, through a real move event."""
        self._point_at(self.mark)

    def close(self):
        self.host.close()
        self.qapp.setStyleSheet(self.saved)


@pytest.fixture(params=["dark", "light"])
def bench(request, qtbot, qapp):
    """A close mark drawn under one of the two flat themes."""
    made = _Bench(qtbot, qapp, request.param)
    yield made
    made.close()


# ---------------------------------------------------------------------------
# The two colours, measured
# ---------------------------------------------------------------------------

def test_the_mark_is_the_themes_own_ink_at_rest(bench):
    """White on dark, near-black on light -- never a literal #FFFFFF."""
    bench.point_away()
    assert not bench.mark.underMouse()

    drawn = _ink(bench.mark, bench.palette["bg"])

    assert _near(drawn, bench.palette["fg"]), (
        f"resting mark drawn {drawn}, expected {bench.palette['fg']}")


def test_the_mark_turns_red_under_a_real_pointer(bench):
    """Hover is driven with a mouse move, not by setting a flag."""
    bench.point_away()
    resting = _ink(bench.mark, bench.palette["bg"])

    bench.point_at_mark()
    assert bench.mark.underMouse()
    hovered = _ink(bench.mark, bench.palette["bg"])

    assert _near(hovered, bench.palette["error"]), (
        f"hovered mark drawn {hovered}, expected {bench.palette['error']}")
    assert hovered != resting


def test_the_mark_goes_back_to_the_themes_ink_when_the_pointer_leaves(bench):
    """Red is the hover state, not a state the mark gets stuck in."""
    bench.point_at_mark()
    assert _near(_ink(bench.mark, bench.palette["bg"]),
                 bench.palette["error"])

    bench.point_away()

    assert _near(_ink(bench.mark, bench.palette["bg"]), bench.palette["fg"])


def test_the_two_colours_are_read_from_the_theme_not_written_down():
    """`close_mark_colours` follows the palette in both directions."""
    dark = theme.close_mark_colours("dark")
    light = theme.close_mark_colours("light")

    assert dark["rest"] == theme.palette_for("dark")["fg"]
    assert light["rest"] == theme.palette_for("light")["fg"]
    assert dark["hover"] == theme.palette_for("dark")["error"]
    assert light["hover"] == theme.palette_for("light")["error"]
    # The point of reading them: the light theme does NOT get a white X.
    assert dark["rest"] != light["rest"]


# ---------------------------------------------------------------------------
# A hit target as well as a glyph
# ---------------------------------------------------------------------------

def test_the_mark_is_larger_than_the_one_qt_draws_on_a_closable_tab(
        qtbot, qt_theme_applied):
    """Both the glyph and the square a user has to hit."""
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.addTab(QWidget(), "One")
    tabs.setTabsClosable(True)
    bar = tabs.tabBar()
    default = bar.tabButton(0, QTabBar.RightSide)
    assert default is not None
    was = max(default.sizeHint().width(), default.sizeHint().height())

    assert theme.install_close_marks(tabs) == 1
    mark = bar.tabButton(0, QTabBar.RightSide)

    assert theme.is_close_mark(mark)
    assert mark.width() >= theme.CLOSE_MARK_HIT_PX
    assert mark.width() > was, (
        f"the mark's target shrank: {mark.width()} px against Qt's {was} px")


def _glyph_box(widget):
    """The bounding box of the ink a widget actually painted."""
    image = widget.grab().toImage()
    painted = [(x, y)
               for y in range(image.height())
               for x in range(image.width())
               if image.pixelColor(x, y).alpha() >= 200
               and image.pixelColor(x, y).red() > 100]
    assert painted, "nothing was drawn"
    xs = [x for x, _ in painted]
    ys = [y for _, y in painted]
    return min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def test_the_glyph_is_drawn_larger_than_the_mark_it_replaces(qtbot, qapp):
    """"a LARGE X" -- measured against the `×` the sites used to draw."""
    from PySide6.QtWidgets import QToolButton
    from spacr.qt.preferences import get_font_scale

    saved = qapp.styleSheet()
    qapp.setStyleSheet(theme.stylesheet("dark", font_scale=get_font_scale()))
    try:
        host = QWidget()
        qtbot.addWidget(host)
        column = QVBoxLayout(host)
        was = QToolButton(host)          # what a site drew for itself before
        was.setText("\u00d7")
        column.addWidget(was)
        mark = theme.close_mark_button(host)
        column.addWidget(mark)
        host.resize(140, 140)
        host.show()
        qapp.processEvents()

        _x, _y, old_w, old_h = _glyph_box(was)
        _x, _y, new_w, new_h = _glyph_box(mark)

        # Rasterisation can round one axis to the same pixel on two fonts.
        # It is still a larger drawn mark when neither axis contracts and
        # the painted bounding box grows on the other axis (7x7 versus 7x6
        # on Ubuntu/PySide 6.11.2, for example).
        assert (new_w >= old_w and new_h >= old_h
                and new_w * new_h > old_w * old_h), (
            f"the mark is {new_w}x{new_h}; the one it replaces is "
            f"{old_w}x{old_h}")
    finally:
        qapp.setStyleSheet(saved)


def test_the_glyph_is_never_clipped_by_its_box(qtbot, qapp):
    """A larger X inside a box too small for it would lose its arms."""
    from spacr.qt.preferences import get_font_scale

    saved = qapp.styleSheet()
    qapp.setStyleSheet(theme.stylesheet("dark", font_scale=get_font_scale()))
    try:
        host = QWidget()
        qtbot.addWidget(host)
        column = QVBoxLayout(host)
        mark = theme.close_mark_button(host)
        column.addWidget(mark)
        host.resize(120, 90)
        host.show()
        qapp.processEvents()

        left, top, width, height = _glyph_box(mark)

        assert left > 0 and top > 0
        assert left + width < mark.width()
        assert top + height < mark.height()
        # A square glyph, drawn square: the X is not squashed on either axis.
        assert abs(width - height) <= 3
    finally:
        qapp.setStyleSheet(saved)


def test_a_mark_never_shrinks_a_target_a_site_already_asked_for(
        qtbot, qt_theme_applied):
    """A site with a bigger button keeps it; the mark is a floor."""
    from PySide6.QtWidgets import QToolButton

    big = QToolButton()
    qtbot.addWidget(big)
    big.setMinimumSize(40, 40)

    theme.apply_close_mark(big)

    assert big.width() >= 40 and big.height() >= 40


def test_a_mark_is_never_smaller_than_the_control_qt_would_draw(
        qtbot, qt_theme_applied):
    """The rule that keeps a hit target from shrinking anywhere.

    The box is the largest of three things: the floor, the glyph, and what
    the control itself asks for -- so no site's X can come back smaller
    than the one it replaced.
    """
    from PySide6.QtWidgets import QPushButton, QToolButton

    for cls in (QToolButton, QPushButton):
        button = cls()
        qtbot.addWidget(button)
        theme.apply_close_mark(button)
        hint = button.sizeHint()

        assert button.height() >= hint.height(), cls.__name__
        assert button.height() >= theme.CLOSE_MARK_HIT_PX, cls.__name__
        assert button.width() >= theme.CLOSE_MARK_HIT_PX, cls.__name__
        # And never wider than tall: the glyph is square, and a mark as wide
        # as a word would push the label beside it out of its row.
        assert button.width() <= button.height(), cls.__name__


# ---------------------------------------------------------------------------
# A long title at a narrow window
# ---------------------------------------------------------------------------

def test_a_long_tab_title_at_a_narrow_window_keeps_its_mark(
        qtbot, qt_theme_applied, qapp):
    """The mark stays inside its own tab and off the title."""
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.setTabsClosable(True)
    tabs.addTab(QWidget(), "Annotator Agreement Between Two Raters, Plate 1")
    tabs.addTab(QWidget(), "Map Barcodes")
    theme.install_close_marks(tabs)
    tabs.resize(240, 160)
    tabs.show()
    qapp.processEvents()

    bar = tabs.tabBar()
    for index in range(bar.count()):
        mark = bar.tabButton(index, QTabBar.RightSide)
        assert theme.is_close_mark(mark)
        tab = bar.tabRect(index)
        assert not tab.isEmpty()
        # The mark is drawn INSIDE the tab it closes.
        assert tab.contains(mark.geometry()), (
            f"tab {index}: mark {mark.geometry()} outside tab {tab}")
        # And the tab is still wide enough for some title beside it.
        assert tab.width() - mark.width() >= 12


def test_a_narrow_tab_elides_its_title_rather_than_dropping_the_mark(
        qtbot, qt_theme_applied, qapp):
    """Qt shortens the words; the control is what must survive."""
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.setTabsClosable(True)
    tabs.setElideMode(Qt.ElideRight)
    long_title = "Annotator Agreement Between Two Raters, Plate 1"
    tabs.addTab(QWidget(), long_title)
    theme.install_close_marks(tabs)
    tabs.show()
    qapp.processEvents()
    wide = tabs.tabBar().tabRect(0).width()

    tabs.resize(160, 140)
    qapp.processEvents()
    bar = tabs.tabBar()
    mark = bar.tabButton(0, QTabBar.RightSide)

    assert bar.tabText(0) == long_title       # the title itself is untouched
    assert mark.isVisibleTo(bar)
    assert bar.tabRect(0).width() <= wide     # the tab did give ground
    assert bar.tabRect(0).contains(mark.geometry())


# ---------------------------------------------------------------------------
# One definition
# ---------------------------------------------------------------------------

#: A button being handed a close glyph as its text -- what a site does when
#: it draws its own mark instead of asking the theme for one.
_OWN_GLYPH = re.compile(
    r"""(?:setText\(\s*|Q(?:Push|Tool)Button\(\s*)"""
    r"""["'](?:×|✕|✖|✗|✘|\\u00d7|\\u2715)""")


def _qt_sources():
    """Every module of the Qt package except the theme that owns the mark."""
    for path in sorted(QT_PACKAGE.rglob("*.py")):
        if path.name == "theme.py" or "i18n" in path.name:
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def test_no_module_draws_a_close_glyph_of_its_own():
    """The sweep the instruction asked for, kept runnable."""
    offenders = []
    for path in _qt_sources():
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if _OWN_GLYPH.search(line):
                offenders.append(f"{path.relative_to(QT_PACKAGE)}:{number}")
    assert offenders == [], (
        "these write their own close glyph instead of asking the theme: "
        + ", ".join(offenders))


def test_every_closable_tab_strip_asks_the_theme_for_its_mark():
    """`setTabsClosable(True)` and no `install_close_marks` is a drift."""
    missing = []
    for path in _qt_sources():
        text = path.read_text(encoding="utf-8")
        if "setTabsClosable(True)" in text and "close_marks" not in text:
            missing.append(str(path.relative_to(QT_PACKAGE)))
    assert missing == [], (
        "these make closable tabs without the shared mark: "
        + ", ".join(missing))


def test_the_marks_colours_are_stated_once():
    """No module re-states the rest/hover pair the theme already holds."""
    rule = theme.close_mark_rules("dark")
    assert rule.count(theme.palette_for("dark")["fg"]) == 1
    assert theme.CLOSE_MARK_PROPERTY in rule
    # And the sheet carries it, after the contributed widget blocks, so no
    # block can win the tie by arriving later.
    sheet = theme.stylesheet("dark")
    selector = f'*[{theme.CLOSE_MARK_PROPERTY}="true"]'
    assert selector in sheet
    assert sheet.index(selector) > sheet.index("QStatusBar")
    assert sheet.count(f"{selector} {{") == 1


# ---------------------------------------------------------------------------
# The sites, as they are actually built
# ---------------------------------------------------------------------------

def _series(values):
    """A one-column frame slice, for the clause rows that read their range."""
    import pandas as pd

    return pd.Series(values)


def _marks_under(root):
    """Every widget below ``root`` drawn by the shared close-mark rules."""
    return [child for child in root.findChildren(QWidget)
            if theme.is_close_mark(child)]


def test_a_folded_page_closes_with_the_shared_mark(qtbot, qt_theme_applied):
    """Where the maintainer saw it: closing Annotator Agreement."""
    from spacr.qt.screens import map_barcodes
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="map_barcodes")
    qtbot.addWidget(screen)
    assert map_barcodes.install_folds(screen) is not None
    opener = screen._fold_openers[0]
    folded = opener.open()
    pages = screen._fold_pages
    bar = pages.tabBar()
    index = pages.indexOf(folded)

    mark = bar.tabButton(index, QTabBar.RightSide)
    assert theme.is_close_mark(mark), "the folded page draws its own close"
    assert mark.text() == theme.CLOSE_MARK

    # The host's own page keeps no usable mark: there is nothing behind it.
    host_mark = bar.tabButton(0, QTabBar.RightSide)
    assert host_mark is None or host_mark.isHidden()

    # And the mark closes the page it sits on, not an index it remembered.
    QTest.mouseClick(mark, Qt.LeftButton)
    assert pages.indexOf(folded) < 0
    assert pages.count() == 1


def test_a_mark_closes_the_tab_it_moved_to(qtbot, qt_theme_applied):
    """Closing an earlier tab must not redirect a later mark.

    The mark is looked up on the bar when it is pressed rather than
    remembering the index it was built at, because removing a tab renumbers
    every tab behind it.
    """
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.setTabsClosable(True)
    first, second = QWidget(), QWidget()
    tabs.addTab(first, "First")
    tabs.addTab(second, "Second")
    theme.install_close_marks(tabs)
    closed = []
    tabs.tabCloseRequested.connect(
        lambda index: closed.append(tabs.widget(index)))
    bar = tabs.tabBar()
    mark = bar.tabButton(1, QTabBar.RightSide)

    tabs.removeTab(0)                       # the tab in front goes away
    QTest.mouseClick(mark, Qt.LeftButton)

    assert closed == [second]


def test_the_montage_well_tab_carries_the_shared_mark(qtbot, qt_theme_applied):
    """Its x is on the LEFT and it is the application's mark."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)
    tab = view._open_well_tab(("plate1", "r1", "c1"), "plate1_r1_c1 · GRA14",
                              "a well")
    bar = view._tabs.tabBar()
    index = view._tabs.indexOf(tab)

    mark = bar.tabButton(index, QTabBar.LeftSide)

    assert theme.is_close_mark(mark)
    assert bar.tabButton(index, QTabBar.RightSide) is None
    assert "closes from here and nowhere else" in mark.toolTip()
    assert mark.width() >= theme.CLOSE_MARK_HIT_PX


def test_the_montage_summary_tab_still_has_no_mark(qtbot, qt_theme_applied):
    """A tab that must always exist gets no X, shared or otherwise."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)
    bar = view._tabs.tabBar()

    for side in (QTabBar.LeftSide, QTabBar.RightSide):
        assert bar.tabButton(0, side) is None


def test_the_working_set_chip_carries_the_shared_mark(qtbot, qt_theme_applied):
    """And is still wide enough for the table name beside it."""
    from spacr.qt.widgets.table_chip import TableChip

    chip = TableChip("cell_measurements_plate_1", removable=True)
    qtbot.addWidget(chip)
    chip.show()

    marks = _marks_under(chip)
    assert len(marks) == 1
    assert marks[0].text() == theme.CLOSE_MARK
    # The name is not cropped by the mark that grew beside it.
    assert chip.minimumWidth() >= (
        chip.fontMetrics().horizontalAdvance("cell_measurements_plate_1")
        + marks[0].width())


def test_the_value_chip_and_its_group_drop_carry_the_shared_mark(
        qtbot, qt_theme_applied):
    """The settings chips were the last place with a mark of their own."""
    from spacr.qt.screens import settings_model as sm

    strip = sm._ChipStrip(removable=True)
    qtbot.addWidget(strip)
    strip.set_values(["cell", "nucleus"])
    strip.show()

    marks = _marks_under(strip)
    assert len(marks) == 3, [m.objectName() for m in marks]
    assert {m.text() for m in marks} == {theme.CLOSE_MARK}


@pytest.mark.parametrize("build", [
    pytest.param(
        lambda: __import__(
            "spacr.qt.widgets.data_filter_panel", fromlist=["x"]
        )._RangeRow("cell_area", _series([1.0, 2.0, 3.0])),
        id="filter-clause"),
    pytest.param(
        lambda: __import__(
            "spacr.qt.widgets.graph_builder", fromlist=["x"]
        ).DropZone("x"),
        id="graph-drop-zone"),
    pytest.param(
        lambda: __import__(
            "spacr.qt.widgets.pivot_builder", fromlist=["x"]
        ).DropWell("rows"),
        id="pivot-well"),
    pytest.param(
        lambda: __import__(
            "spacr.qt.widgets.row_exclusion", fromlist=["x"]
        )._ExclusionRuleRow(),
        id="exclusion-rule"),
])
def test_each_swept_site_asks_the_theme_for_its_mark(
        build, qtbot, qt_theme_applied):
    """One definition means every one of these resolves to it."""
    widget = build()
    qtbot.addWidget(widget)
    widget.show()

    marks = _marks_under(widget)

    assert len(marks) == 1, [m.objectName() for m in marks]
    assert marks[0].text() == theme.CLOSE_MARK
    assert marks[0].width() >= theme.CLOSE_MARK_HIT_PX
    assert marks[0].height() >= theme.CLOSE_MARK_HIT_PX


def test_the_class_chip_carries_the_shared_mark(qtbot, qt_theme_applied):
    """The Classify selector's chips removed themselves with their own x."""
    from spacr.qt.widgets.class_editor import ClassChip
    from spacr.classify_classes import ClassRule
    from spacr.qt.theme import active_palette

    chip = ClassChip(0, ClassRule(name="positive", column="annot_1", value=1),
                     active_palette())
    qtbot.addWidget(chip)
    chip.show()

    marks = _marks_under(chip)

    assert len(marks) == 1
    assert marks[0].text() == theme.CLOSE_MARK
    assert marks[0].width() >= theme.CLOSE_MARK_HIT_PX


def test_the_layer_viewers_remove_control_is_the_shared_mark(
        qtbot, qt_theme_applied):
    """The one control in that row that destroys something now says so."""
    from spacr.qt.layer_viewer import LayerViewer

    viewer = LayerViewer()
    qtbot.addWidget(viewer)

    assert theme.is_close_mark(viewer.remove_button)
    assert viewer.remove_button.text() == theme.CLOSE_MARK
    assert viewer.remove_button.width() >= theme.CLOSE_MARK_HIT_PX


def test_the_mark_re_measures_when_zoom_changes_under_it(qtbot, qapp):
    """A box fixed to yesterday's font would clip today's glyph.

    The Zoom preference rebuilds the sheet under marks that already exist,
    so the mark has to notice the larger font arriving rather than keep the
    square it was born with.
    """
    saved = qapp.styleSheet()
    qapp.setStyleSheet(theme.stylesheet("dark", font_scale=1.0))
    try:
        host = QWidget()
        qtbot.addWidget(host)
        column = QVBoxLayout(host)
        mark = theme.close_mark_button(host)
        column.addWidget(mark)
        host.resize(160, 120)
        host.show()
        qapp.processEvents()
        small = mark.size()

        qapp.setStyleSheet(theme.stylesheet("dark", font_scale=2.0))
        qapp.processEvents()

        assert mark.font().pixelSize() > 0
        assert mark.height() > small.height(), (
            f"the box stayed {small} while the glyph grew to "
            f"{mark.font().pixelSize()} px")
        left, top, width, height = _glyph_box(mark)
        assert left > 0 and top > 0
        assert left + width < mark.width()
        assert top + height < mark.height()
    finally:
        qapp.setStyleSheet(saved)


def test_a_folded_page_with_a_long_name_keeps_its_mark_when_squeezed(
        qtbot, qt_theme_applied, qapp):
    """The strip the maintainer was looking at, at a width that hurts."""
    from spacr.qt.screens import map_barcodes
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="map_barcodes")
    qtbot.addWidget(screen)
    assert map_barcodes.install_folds(screen) is not None
    folded = screen._fold_openers[0].open()
    pages = screen._fold_pages
    index = pages.indexOf(folded)
    pages.setTabText(index, "Annotator Agreement Between Two Raters")
    screen.resize(1100, 700)
    screen.show()
    qapp.processEvents()
    bar = pages.tabBar()
    mark = bar.tabButton(index, QTabBar.RightSide)
    roomy = mark.size()

    screen.resize(420, 700)
    qapp.processEvents()

    assert theme.is_close_mark(mark)
    assert mark.isVisibleTo(bar)
    assert mark.size() == roomy, "the mark shrank when the window did"
    assert bar.tabRect(index).contains(mark.geometry()), (
        f"mark {mark.geometry()} left tab {bar.tabRect(index)}")


def test_the_settings_caption_does_not_name_a_dot(qtbot):
    """The information dots were removed; the caption still pointed at one.

    "Hover any setting for details, or select (i) for documentation" was
    shown under the settings panel and was wrong in ten languages once the
    dot stopped being drawn. The API link did not go anywhere -- it is in
    the hover tooltip, which is what the sentence says now.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    for name in dir(screen):
        if "hint" not in name.lower():
            continue
        attribute = getattr(screen, name, None)
        if not callable(attribute):
            continue
        try:
            text = attribute()
        except Exception:                                    # noqa: BLE001
            continue
        if isinstance(text, str):
            assert "ⓘ" not in text, f"{name}() still names the dot"


def test_no_widget_on_a_screen_draws_the_dot(qtbot):
    """The acceptance point as written: not that the flag is gone, but that
    nothing DRAWS one."""
    from PySide6.QtWidgets import QWidget

    from spacr.qt.screens.app_screen import AppScreen

    for key in ("regression", "measure"):
        screen = AppScreen(key)
        qtbot.addWidget(screen)
        carrying = []
        for widget in screen.findChildren(QWidget):
            for getter in ("text", "toolTip"):
                function = getattr(widget, getter, None)
                if not callable(function):
                    continue
                try:
                    value = function()
                except Exception:                            # noqa: BLE001
                    continue
                if isinstance(value, str) and "ⓘ" in value:
                    carrying.append((type(widget).__name__, getter))
        assert not carrying, f"{key} still draws the dot: {carrying[:3]}"
