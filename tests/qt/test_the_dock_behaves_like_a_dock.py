"""The dock draws icons, magnifies them under the pointer, and no names.

Instruction 348, parts 2 and 3 -- asked for verbatim on 2026-09-02: "in the
dock enlarge the icons and add a magnification effect so the dock functions
like the osx dock and remove the text that can be moved to the category
tooltip location upon hover."

Part 1 (Help is a dock heading, and it is last) landed earlier and is covered
by ``test_the_dock_and_menu_show_the_folded_modules.py``.

MEASURED, one 1440x900 MainWindow with the dark stylesheet applied:

    icon at rest       16 px  ->  26 px
    icon under pointer 16 px  ->  40 px  (36 one row out, 30 two rows out)
    row height         38 px  ->  48 px, and it does not move while
                                  magnifying -- see the note on
                                  `Sidebar.magnify_from`
    dock width        220 px  -> 220 px  (`WIDTH_MIN`, unchanged)
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

    The teardown is deliberately the only place the magnifier is called
    from the fixture: run against the version of ``app.py`` that has no
    magnifier, a fixture that called it in SETUP would turn every test in
    this file into the same error and hide which of them were already
    passing before the change.
    """
    bar = window._sidebar
    yield bar
    reset = getattr(bar, "magnify_from", None)
    if reset is not None:
        reset(None)


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


def test_the_pointer_magnifies_the_row_under_it_and_its_neighbours(dock):
    """The OS X curve: biggest under the pointer, smaller either side.

    Asserted on the sizes rather than on a screenshot, because the sizes are
    the whole observable effect and a screenshot of a themed icon is a test
    of the icon.
    """
    from spacr.qt.preferences import scaled_px

    dock.magnify_from(None)
    rows = _visible(dock)
    assert len(rows) >= 5, "need a row with two neighbours on each side"
    focus = 3
    dock.magnify_from(rows[focus].y() + rows[focus].height() // 2)
    sizes = [row.iconSize().width() for row in rows]

    base, peak = scaled_px(Sidebar.ICON_PX), scaled_px(Sidebar.ICON_MAX_PX)
    assert sizes[focus] == peak, (
        f"the row under the pointer is {sizes[focus]} px, not {peak}")
    assert base < sizes[focus - 1] < peak, "the row above did not swell"
    assert base < sizes[focus + 1] < peak, "the row below did not swell"
    assert sizes[focus - 2] < sizes[focus - 1], "the falloff is not monotonic"
    assert sizes[focus + 2] < sizes[focus + 1], "the falloff is not monotonic"
    far = [s for i, s in enumerate(sizes) if abs(i - focus) > 3
           and not rows[i].property("isFoldChild")]
    assert far and set(far) == {base}, (
        f"rows out of reach of the pointer are not at rest: {sorted(set(far))}")


def test_the_swell_falls_away_when_the_pointer_leaves_the_dock(dock):
    """``magnify_from(None)`` is what the dock's ``leaveEvent`` calls."""
    rows = _visible(dock)
    dock.magnify_from(rows[2].y() + rows[2].height() // 2)
    assert max(row.iconSize().width() for row in rows) > Sidebar.ICON_PX

    dock.magnify_from(None)
    resting = {row: dock.resting_icon_px(row) for row in rows}
    left_big = {str(row.property("navKey")): row.iconSize().width()
                for row in rows if row.iconSize().width() != resting[row]}
    assert not left_big, f"these rows stayed swollen: {left_big}"


def test_magnifying_never_changes_a_row_height(dock):
    """A fixed row height is what keeps the dock from chasing the pointer.

    If a swollen row grew taller it would push every row below it down, so
    the row under a stationary pointer would stop being the row under the
    pointer -- and this dock has already been reported flickering once
    (2026-09-01) for a hover that moved geometry.
    """
    rows = _visible(dock)
    before = [row.height() for row in rows]
    tops = [row.y() for row in rows]
    for index in (0, len(rows) // 2, len(rows) - 1):
        dock.magnify_from(rows[index].y() + rows[index].height() // 2)
    assert [row.height() for row in rows] == before, "a row changed height"
    assert [row.y() for row in rows] == tops, "a row moved"


def test_a_real_mouse_move_over_a_row_reaches_the_magnifier(dock):
    """Driven through the event the window manager would deliver.

    Everything else in this file calls `magnify_from` directly, which
    proves the arithmetic and proves nothing about the wiring. This sends
    the QMouseEvent a pointer actually generates, so it fails if the dock
    stops installing itself as a filter on its rows, if the rows lose
    mouse tracking, or if the coordinate mapping into the scrolled row
    host goes wrong.
    """
    from spacr.qt.preferences import scaled_px

    dock.magnify_from(None)
    rows = _visible(dock)
    row = rows[4]
    local = QPointF(row.width() / 2, row.height() / 2)
    QApplication.sendEvent(row, QMouseEvent(
        QEvent.Type.MouseMove, local,
        QPointF(row.mapToGlobal(QPoint(int(local.x()), int(local.y())))),
        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier))
    assert row.iconSize().width() == scaled_px(Sidebar.ICON_MAX_PX)

    # And an Enter, which is the event a pointer arriving from off the
    # dock delivers before it ever moves inside a row.
    other = rows[1]
    QApplication.sendEvent(other, QEnterEvent(
        local, local,
        QPointF(other.mapToGlobal(QPoint(int(local.x()), int(local.y()))))))
    assert other.iconSize().width() == scaled_px(Sidebar.ICON_MAX_PX)
    assert row.iconSize().width() < scaled_px(Sidebar.ICON_MAX_PX)

    # Leaving the dock is the dock's own event, not a row's: a pointer can
    # leave the last row downwards onto the empty stretch below it, where
    # no other row will ever be entered.
    QApplication.sendEvent(dock, QEvent(QEvent.Type.Leave))
    assert {r.iconSize().width() for r in rows} == {
        dock.resting_icon_px(r) for r in rows}


def test_every_row_is_as_tall_as_the_largest_icon_it_can_show(dock):
    """Otherwise a magnified icon would be clipped by its own row."""
    from spacr.qt.preferences import scaled_px

    wanted = scaled_px(Sidebar.ICON_MAX_PX + 2 * Sidebar.ROW_PAD_PX)
    heights = {row.height() for row in _visible(dock)}
    assert heights == {wanted}, f"row heights {sorted(heights)}, want {wanted}"
    assert wanted > scaled_px(Sidebar.ICON_MAX_PX)


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

    def ink_right_of(widget, from_x):
        pixmap = QPixmap(widget.size())
        pixmap.fill()
        widget.render(pixmap)
        image = pixmap.toImage()
        marked = 0
        for x in range(from_x, image.width()):
            for y in range(image.height()):
                colour = image.pixelColor(x, y)
                if (colour.red(), colour.green(), colour.blue()) != (255,) * 3:
                    marked += 1
                    break
        return marked

    gap = row.icon_rect().right() + 6
    assert ink_right_of(row, gap) == 0, (
        "the dock row painted something to the right of its icon")

    control = ElidingPushButton(row.text(), dock)
    control.setObjectName("SidebarItem")
    control.resize(row.size())
    control.setIcon(row.icon())
    control.setIconSize(row.iconSize())
    qapp.processEvents()
    assert ink_right_of(control, gap) > 0, (
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
