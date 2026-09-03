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


def test_a_row_sits_on_a_translucent_rounded_plate(dock, qapp):
    """The box asked for on 2026-09-03, measured off the rendered row.

    Three claims, and all three are what the request named. TRANSLUCENT and
    VISIBLE: the plate is a step off whatever is behind the dock rather
    than either invisible or an opaque slab. ROUNDED: the row's corner is
    NOT the plate, because the plate's corner is cut away there. HIGHLIGHTS
    ON HOVER: the step gets bigger.

    Read off pixels rather than off the stylesheet on purpose. The QSS
    rule that used to be believed to draw this reaches no dock row at all
    (see `_DockRow._paint_plate`), and a test that had asserted the QSS
    text would have passed for the whole time the dock had no plate.
    """
    row = next(r for r in _visible(dock) if not r.property("isFoldChild"))

    def sample():
        shot = QPixmap(row.size())
        shot.fill()
        row.render(shot)
        image = shot.toImage()
        inside = image.pixelColor(row.width() - 10, row.height() // 2)
        corner = image.pixelColor(0, 0)
        return inside, corner

    row._hovered = False
    rest, behind = sample()
    row._hovered = True
    hover, hover_corner = sample()
    row._hovered = False

    def step(a, b):
        return max(abs(x - y) for x, y in
                   ((a.red(), b.red()), (a.green(), b.green()),
                    (a.blue(), b.blue())))

    assert step(rest, behind) >= 8, (
        f"the resting plate {rest.name()} is invisible against "
        f"{behind.name()}")
    assert rest.alpha() == 255 and behind.alpha() == 255, (
        "sampled a transparent pixel -- the render never happened")
    assert step(hover, rest) >= 8, (
        f"hover does not highlight: {rest.name()} -> {hover.name()}")
    assert behind.name() == hover_corner.name(), (
        "the corner changed on hover, so the plate is not inset there")
    assert step(rest, behind) < 200, (
        f"the resting plate {rest.name()} is opaque, not translucent")


def test_the_open_module_is_marked_by_its_plate_and_an_accent_bar(dock):
    """Selection has to survive a hover, or the dock forgets where you are.

    Before the plate existed, `:checked` was an opaque QSS background that
    never reached the row -- so the accent bar was the only mark, and it
    was drawn square against a plate that is now rounded.
    """
    from spacr.qt.theme import active_palette

    row = next(r for r in _visible(dock) if not r.property("isFoldChild"))
    row.setProperty("selected", "true")
    shot = QPixmap(row.size())
    shot.fill()
    row.render(shot)
    image = shot.toImage()
    accent = active_palette()["accent"]
    bar = image.pixelColor(row.PLATE_INSET_PX + 1, row.height() // 2)
    row.setProperty("selected", None)
    assert bar.name().lower() == accent.lower(), (
        f"the selected row's left edge is {bar.name()}, not the accent "
        f"{accent}")


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
    nameless, tipless, keyless = [], [], []
    for row in dock._items:
        key = str(row.property("navKey") or "")
        if not key:
            keyless.append(row)
        if not row.accessibleName().strip():
            nameless.append(key)
        if not row.toolTip().strip():
            tipless.append(key)
    assert dock._items, "no dock rows at all"
    assert not keyless, "a dock row carries no navKey"
    assert not nameless, f"rows with no accessible name: {nameless}"
    assert not tipless, f"rows with no tooltip: {tipless}"


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


def test_hovering_a_dock_row_puts_its_name_in_the_status_strip(window, dock):
    """The strip is where the name went, and it has to be there.

    Driven through the filter Qt would hand the ToolTip event to, so this
    fails if the row stops carrying `moduleNameSource` as well as if the
    strip stops listening.
    """
    hints = window._module_hints
    assert hints is not None, "module hints are not installed on the window"
    row = next(r for r in _visible(dock)
               if str(r.property("navKey")) == "mask")
    window.statusBar().clearMessage()
    hints.eventFilter(row, QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1),
                                      row.mapToGlobal(QPoint(1, 1))))
    assert "Mask" in window.statusBar().currentMessage()


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
