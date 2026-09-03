"""The dock draws icons on translucent plates, and no names.

Instruction 348, parts 2 and 3 -- asked for verbatim on 2026-09-02: "in the
dock enlarge the icons and add a magnification effect so the dock functions
like the osx dock and remove the text that can be moved to the category
tooltip location upon hover."

THE MAGNIFICATION IS GONE, removed on 2026-09-03 at the maintainer's
request: "remove the icon magnefication effect from the dock and replace the
black box behind thicons with a translucent box with rounded edges. the
translucent box should also highlight upon hover." Every icon now rests at
one size whatever the pointer is doing, and hover is the plate stepping up
plus the name and the accent ink that 369 added. The five tests that drove
`Sidebar.magnify_from` were replaced by the five below them, which assert
what the dock does instead.

Part 1 (Help is a dock heading, and it is last) landed earlier and is covered
by ``test_the_dock_and_menu_show_the_folded_modules.py``.

MEASURED, one 1440x900 MainWindow with the dark stylesheet applied:

    icon, every row    16 px  ->  26 px, constant
    row height         38 px  ->  48 px, and it never moves
    dock width        220 px  -> 220 px  (`WIDTH_MIN`, unchanged)
    plate behind a row  none  ->  +16 lift at rest, +17 more on hover
    rows carrying a painted name  71 -> 0
    rows carrying an accessible name  71 -> 71

The last two lines are the pair that matters. Removing the name from a row
is only safe while every other way of identifying it survives, so the tests
below check the accessible name, the tooltip, the ``navKey`` property, the
status strip AND the icon -- 34 folded children had a NULL icon before this
change, and a row with neither a name nor a picture is a blank strip.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QHelpEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication, QLabel

from spacr.qt.app import MainWindow, Sidebar


@pytest.fixture(scope="module")
def window(qapp, qt_theme_applied):
    """One themed 1440x900 window, shared: building it costs seconds.

    The stylesheet has to be on the application, not just imported: an
    unstyled dock reports a 25 px row where the shipped one is 48 px, and
    every height asserted below would be measuring the wrong widget.
    """
    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    qapp.processEvents()
    yield win
    win.hide()
    qapp.processEvents()


@pytest.fixture
def dock(window):
    """The dock, put back to its resting state after each test.

    The window is module-scoped, so a test that leaves a row hovered or
    selected hands that row to the next test. Cleared in teardown rather
    than in setup so a failure reads as the test that caused it.
    """
    bar = window._sidebar
    yield bar
    for row in bar._items:
        row._hovered = False
        row.setProperty("selected", None)
    bar.leaveEvent(QEvent(QEvent.Type.Leave))


def _visible(bar):
    return [row for row in bar._items if not row.isHidden()]


# ---------------------------------------------------------------------------
# Part 2 -- the icons grow
# ---------------------------------------------------------------------------

def test_a_dock_icon_is_bigger_than_the_bullet_it_used_to_be(dock):
    """Every row rests at ``ICON_PX``, and that is well past the old 16 px.

    16 px was the size of a glyph sitting in front of a word. With the word
    gone the icon IS the row, and 16 px in a 48 px row reads as a bullet
    point rather than a dock tile.
    """
    from spacr.qt.preferences import scaled_px

    assert Sidebar.ICON_PX > 16, "348 asked for the icons to be enlarged"
    hosts = [row for row in _visible(dock) if not row.property("isFoldChild")]
    assert hosts, "no dock rows to measure"
    sizes = {row.iconSize().width() for row in hosts}
    assert sizes == {scaled_px(Sidebar.ICON_PX)}, (
        f"rows rest at {sorted(sizes)} px, expected "
        f"{scaled_px(Sidebar.ICON_PX)} px")


def test_every_icon_is_the_same_size_whatever_the_pointer_does(dock):
    """No magnifier. Removed 2026-09-03 -- see this module's docstring.

    Asserted through the events a real pointer delivers, not by checking
    that a method is absent: a magnifier could come back as anything, and
    what was asked for is that moving the pointer over the dock does not
    resize an icon.
    """
    rows = _visible(dock)
    assert len(rows) >= 5, "need a row with two neighbours on each side"
    before = [row.iconSize().width() for row in rows]

    for row in (rows[1], rows[3], rows[-1]):
        local = QPointF(row.width() / 2, row.height() / 2)
        globally = QPointF(row.mapToGlobal(QPoint(int(local.x()),
                                                  int(local.y()))))
        QApplication.sendEvent(row, QEnterEvent(local, local, globally))
        QApplication.sendEvent(row, QMouseEvent(
            QEvent.Type.MouseMove, local, globally,
            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier))
        assert [r.iconSize().width() for r in rows] == before, (
            f"the pointer over {row.property('navKey')!r} resized an icon")
        QApplication.sendEvent(row, QEvent(QEvent.Type.Leave))

    hosts = {r.iconSize().width() for r in rows
             if not r.property("isFoldChild")}
    children = {r.iconSize().width() for r in rows
                if r.property("isFoldChild")}
    assert len(hosts) == 1, f"hosts disagree on icon size: {sorted(hosts)}"
    assert len(children) <= 1, f"children disagree: {sorted(children)}"
    if children:
        assert children < hosts, "a folded child is not drawn smaller"


def _paint_only(widget, background="#ff00ff"):
    """``widget`` rendered over ``background`` WITHOUT the window fill.

    `QWidget.render` defaults to `DrawWindowBackground | DrawChildren`, and
    the window fill comes from the widget's PALETTE whatever its `paintEvent`
    does -- so a plain render reports `#161719` for a row that paints nothing
    at all, which is exactly the colour the defect being tested used to
    paint. Dropping that flag leaves only what the widget itself draws, which
    is the question.
    """
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QColor, QPixmap, QRegion
    from PySide6.QtWidgets import QWidget

    shot = QPixmap(widget.size())
    shot.fill(QColor(background))
    widget.render(shot, QPoint(), QRegion(),
                  QWidget.RenderFlag.DrawChildren)
    return shot.toImage()


def test_a_row_paints_nothing_behind_its_icon(dock, qapp):
    """No plate, no button, no box. Asked for on 2026-09-03.

    "the icon with text also has fields which appear whne hovered, remove
    these, i just want the transparent dock holder with rounded edges, the
    icons and when hovered the icons turn blue and you see the text which is
    also blue. nothing else."

    Measured by rendering the row over a colour nothing in the palette uses
    and asking whether that colour survives beside the icon. This has failed
    for two different reasons on the same day: `drawControl(CE_PushButton)`
    rendering a NATIVE button panel from the palette's Button role -- an
    opaque `#161719` behind every icon, drawn over the dock's own slab --
    and a plate this class painted itself for a few hours.
    """
    row = next(r for r in _visible(dock) if not r.property("isFoldChild"))
    image = _paint_only(row)

    beside = image.pixelColor(row.width() - 12, row.height() // 2)
    assert beside.name().lower() == "#ff00ff", (
        f"the row painted {beside.name()} behind its icon")
    corner = image.pixelColor(0, 0)
    assert corner.name().lower() == "#ff00ff", (
        f"the row painted {corner.name()} in its corner")


def test_hovering_a_row_inks_it_in_the_accent(dock, qapp):
    """The whole of a row's hover: blue icon, blue name, nothing else."""
    from PySide6.QtGui import QColor, QPixmap

    from spacr.qt.theme import active_palette

    row = next(r for r in _visible(dock) if not r.property("isFoldChild"))

    def bluish_pixels():
        image = _paint_only(row, "#000000")
        return sum(
            1
            for x in range(0, row.width(), 2)
            for y in range(0, row.height(), 2)
            if (lambda c: c.blue() > c.red() + 20 and c.blue() > 40)(
                image.pixelColor(x, y)))

    row._hovered = False
    rest = bluish_pixels()
    row._hovered = True
    hot = bluish_pixels()
    row._hovered = False

    assert rest == 0, "the row is already blue with nothing hovering it"
    assert hot > 0, "hovering the row did not ink anything in the accent"
    accent = QColor(active_palette()["accent"])
    assert accent.blue() > accent.red(), "the accent is not a blue"


def test_the_open_module_keeps_its_icon_inked(dock):
    """Selection has to be visible, or the dock forgets where you are.

    IT IS THE ICON, not a bar and not a plate. Every box in the dock came
    off on 2026-09-03; what marks the open module is the same accent ink a
    hover uses, left on. Its NAME is not drawn -- that would put a permanent
    label back in the column the dock stopped drawing labels in.
    """
    from PySide6.QtGui import QColor, QPixmap

    row = next(r for r in _visible(dock) if not r.property("isFoldChild"))

    def bluish_pixels():
        image = _paint_only(row, "#000000")
        return sum(
            1
            for x in range(0, row.width(), 2)
            for y in range(0, row.height(), 2)
            if (lambda c: c.blue() > c.red() + 20 and c.blue() > 40)(
                image.pixelColor(x, y)))

    plain = bluish_pixels()
    row.setProperty("selected", "true")
    marked = bluish_pixels()
    row.setProperty("selected", None)
    assert marked > plain, (
        "the open module is not marked at all: "
        f"{plain} accent pixels unselected, {marked} selected")


def test_every_row_is_taller_than_the_icon_it_shows(dock):
    """The plate needs room around the icon, or it reads as an outline.

    The height comes from ``ICON_MAX_PX``, which was the magnifier's peak
    size and is now simply the icon box each row reserves.
    """
    from spacr.qt.preferences import scaled_px

    wanted = scaled_px(Sidebar.ICON_MAX_PX + 2 * Sidebar.ROW_PAD_PX)
    heights = {row.height() for row in _visible(dock)}
    assert heights == {wanted}, f"row heights {sorted(heights)}, want {wanted}"
    assert wanted > scaled_px(Sidebar.ICON_PX), (
        "the row is no taller than its icon, leaving the plate no margin")


def test_the_rows_never_move(dock):
    """A dock row is a target, and a target that moves is a misclick.

    This dock has been reported flickering once (2026-09-01) for a hover
    that moved geometry, which is why 348 pinned the row height and why
    the magnifier grew icons about a fixed centre. With the magnifier gone
    the claim is simpler and stronger: hovering changes no geometry at all.
    """
    rows = _visible(dock)
    tops = [row.y() for row in rows]
    heights = [row.height() for row in rows]
    width = dock.width()
    for index in (0, len(rows) // 2, len(rows) - 1):
        row = rows[index]
        local = QPointF(row.width() / 2, row.height() / 2)
        QApplication.sendEvent(row, QEnterEvent(
            local, local,
            QPointF(row.mapToGlobal(QPoint(int(local.x()), int(local.y()))))))
    assert [row.y() for row in rows] == tops, "a row moved"
    assert [row.height() for row in rows] == heights, "a row changed height"
    assert dock.width() == width, "the dock changed width"


# ---------------------------------------------------------------------------
# Part 3 -- the names come off, and everything else that names a row stays
# ---------------------------------------------------------------------------

def test_a_dock_row_paints_no_text(dock, qapp):
    """Rendered AND UNHOVERED, a row is a plate and an icon and nothing else.

    Probed by rendering rather than by reading ``text()``, because the row
    deliberately still HOLDS its name -- see the class docstring on
    ``_DockRow`` -- and the maintainer's ask was about what is on screen.

    UNHOVERED IS NOW PART OF THE CLAIM. 348 took the name off the row; 369
    put it back for as long as the pointer is on it, in the accent colour,
    beside the icon. The two are not in conflict: the complaint 348 fixed
    was a permanent column of names duplicating the status strip, and a
    label that exists only under the pointer is the opposite of permanent.
    `test_the_dock_names_itself_on_hover.py` holds the hovered half.

    The second half is a control. A probe that reports "no ink" against a
    row that is in fact drawing its name would pass for the wrong reason
    forever, so the same probe is run over an ordinary button carrying the
    same text and must find the ink it missed on the dock row.
    """
    from spacr.qt.widgets.eliding import ElidingPushButton

    row = next(r for r in _visible(dock)
               if str(r.property("navKey")) == "mask")

    def rendered(widget):
        pixmap = QPixmap(widget.size())
        pixmap.fill()
        widget.render(pixmap)
        return pixmap.toImage()

    def differs_right_of(first, second, from_x):
        """Columns where two renders of the same widget disagree."""
        return sum(
            1 for x in range(from_x, first.width())
            if any(first.pixelColor(x, y) != second.pixelColor(x, y)
                   for y in range(first.height())))

    # THE CLAIM, ASKED DIRECTLY: give the row a different name and nothing
    # about it changes. That is what "paints no text" means, and it is
    # measurable without deciding which pixels are plate, which are the
    # border and which are glyphs -- the earlier probe counted every
    # non-white pixel, so the plate alone reported 177 columns of "text"
    # and this test never passed from the day it was written.
    gap = row.icon_rect().right() + 6
    before = rendered(row)
    was = row.text()
    row.setText("A NAME THAT WOULD BE VERY WIDE INDEED")
    qapp.processEvents()
    after = rendered(row)
    row.setText(was)
    qapp.processEvents()

    assert differs_right_of(before, after, gap) == 0, (
        "the dock row painted its name: changing the text changed the "
        "pixels to the right of the icon")

    # The control: the same comparison over an ordinary button MUST see the
    # difference, or the probe's verdict on the dock row means nothing.
    control = ElidingPushButton(was, dock)
    control.setObjectName("SidebarItem")
    control.resize(row.size())
    control.setIcon(row.icon())
    control.setIconSize(row.iconSize())
    qapp.processEvents()
    control_before = rendered(control)
    control.setText("A NAME THAT WOULD BE VERY WIDE INDEED")
    qapp.processEvents()
    control_after = rendered(control)
    assert differs_right_of(control_before, control_after, gap) > 0, (
        "the probe cannot see painted text at all, so its verdict on the "
        "dock row means nothing")
    control.deleteLater()


def test_every_dock_row_still_says_what_it_is(dock):
    """Screen reader, hover and code each keep a way to identify a row.

    A row with no visible name, no accessible name and no tooltip is a row
    a user cannot identify -- 348's own WATCH says so.
    """
    nameless, described, keyless, popups = [], [], [], []
    for row in dock._items:
        key = str(row.property("navKey") or "")
        if not key:
            keyless.append(row)
        if not row.accessibleName().strip():
            nameless.append(key)
        if row.toolTip().strip():
            popups.append(key)
        if key != "__home__" and not row.accessibleDescription().strip():
            described.append(key)
    assert dock._items, "no dock rows at all"
    assert not keyless, "a dock row carries no navKey"
    assert not nameless, f"rows with no accessible name: {nameless}"
    assert not described, f"rows that describe themselves to nobody: {described}"
    # AND NO TOOLTIP, which is the change of 2026-09-03: "remove the popup
    # window tooltip on the moduals. the tooltip is shown at the botom of
    # the screen." The sentence moved to the accessible description above
    # and to the strip at the foot of the window, which -- unlike a popup --
    # can carry the API and Tutorial links and hold them long enough to be
    # pressed. This assertion is the direction that matters now: a row that
    # grew a tooltip back has put the popup back.
    assert not popups, f"these rows still pop a tooltip: {popups}"


def test_no_dock_row_is_blank(dock):
    """Every row draws SOMETHING, which since 348 means an icon.

    The 34 folded children carried a null icon and were identified purely
    by their indented label. With the labels off the rows that would have
    made them 34 identical empty strips, so they were given the icons their
    keys already resolve to.
    """
    blank = [str(row.property("navKey")) for row in dock._items
             if row.icon().isNull()]
    assert not blank, f"dock rows that would draw nothing at all: {blank}"


def test_hovering_a_dock_row_explains_it_in_the_hint_strip(window, dock,
                                                          qapp):
    """The strip is where the description went, and it has to be there.

    NOT THE STATUS BAR ANY MORE. It went to the bottom-LEFT status line with
    a four-second linger until 2026-09-03, and that is what the maintainer
    reported: "in the bottom of the screen to the left is text that also
    flickers sometimes like its going to what is hovered and something else
    back and forthe." Two writers, alternating -- this filter on every hover,
    and Qt restoring the permanent message four seconds later.

    It goes to the page's hint strip now, which holds the last module for
    thirty seconds and carries its API and Tutorial links.

    Driven through the filter Qt would hand the ToolTip event to, so this
    fails if the row stops carrying `moduleAppKey` as well as if the strip
    stops listening.
    """
    hints = window._module_hints
    assert hints is not None, "module hints are not installed on the window"
    home = window._startup
    window._stack.setCurrentWidget(home)
    home._hint_bar.release()
    row = next(r for r in _visible(dock)
               if str(r.property("navKey")) == "mask")
    window.statusBar().clearMessage()
    hints.eventFilter(row, QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1),
                                      row.mapToGlobal(QPoint(1, 1))))
    qapp.processEvents()
    assert home._hint_bar.module_key == "mask", (
        f"the strip is explaining {home._hint_bar.module_key!r}")
    assert window.statusBar().currentMessage() == "", (
        "the status bar is being written on hover again")


def test_a_folded_child_is_inset_and_smaller_than_its_host(dock, qapp):
    """330's three leading spaces were the only mark of a child row.

    With no label to indent, the indent moved into the icon: a child is
    drawn further right and smaller than the row it hangs off.
    """
    from spacr.qt.preferences import scaled_px

    # A host in Core, because Core is the only section open by default and a
    # child in a closed section stays hidden however far its host expands --
    # `make_masks` (6 children, the widest fold) is in Data and showed none.
    visible = {str(r.property("navKey")) for r in _visible(dock)}
    host = next(key for key in ("regression", "classify_merged", "mask")
                if key in visible
                and any(str(r.property("foldParent")) == key
                        for r in dock._items))
    dock.expand_host(host)
    qapp.processEvents()
    children = [r for r in dock._items
                if str(r.property("foldParent")) == host and not r.isHidden()]
    parent = next(r for r in dock._items
                  if str(r.property("navKey")) == host)
    assert children, f"{host} showed no folded children"

    for child in children:
        assert child.icon_rect().x() > parent.icon_rect().x(), (
            f"{child.property('navKey')} is not indented under {host}")
        assert child.iconSize().width() < parent.iconSize().width(), (
            f"{child.property('navKey')} is not drawn smaller than its host")
    assert children[0].iconSize().width() == max(
        1, round(scaled_px(Sidebar.ICON_PX) * Sidebar.FOLD_ICON_SCALE))
    dock.expand_host("__home__")
    qapp.processEvents()


# ---------------------------------------------------------------------------
# The thing bigger rows put at risk
# ---------------------------------------------------------------------------

def test_the_collapsed_dock_still_fits_a_900px_laptop(dock, qapp):
    """348's WATCH: bigger rows make the collapsed default matter more.

    Measured, not asserted from the code: the title, the five headings and
    every row Core shows, plus the widest host's folded children opened
    underneath, against the 900 px laptop the scroll area was introduced
    for.
    """
    for section in list(dock._section_headers):
        if section != "Core" and dock.section_is_open(section):
            dock.toggle_section(section)
    dock.expand_host("mask")
    qapp.processEvents()

    title = next(w for w in dock.findChildren(QLabel)
                 if w.objectName() == "SidebarTitle")
    tall = (title.sizeHint().height()
            + sum(h.sizeHint().height()
                  for h in dock._section_headers.values() if h.isVisible())
            + sum(r.height() for r in _visible(dock)))
    assert tall <= 900, (
        f"the collapsed dock asks for {tall} px of a 900 px window; the "
        "sections were collapsed in 330 precisely to stop that")
