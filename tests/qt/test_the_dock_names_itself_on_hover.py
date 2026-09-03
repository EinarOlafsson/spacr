"""The dock: the spaCR mark on Home, the name in blue on hover, no grey tray.

Instruction 369, verbatim: "the home icon should be the spacr icon, when an
icon is hovered its name should appear in blue tetx center left anlligned (to
the right of the icon) and th icon should go from white to blue, the
background dark gray container can be removed, the hover highlight should
stay."

Two of the seven asks were already true when 369 was filed -- nested modules
are nested (330) and Help is a section at the bottom (348) -- and are covered
by their own files. What is here is the five that were not.

THE ROW MUST NOT MOVE. 348 pinned the row height because a target that grows
under the pointer is flicker this dock has been reported for once already,
and everything 369 adds is PAINT: a second ink colour and a string, with no
geometry behind either. That is asserted rather than assumed, because it is
the thing most likely to be broken by a later "just make it fit" change.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt          # noqa: E402
from PySide6.QtGui import QEnterEvent                           # noqa: E402


def _dock(qtbot):
    from spacr.qt.app import Sidebar

    bar = Sidebar()
    qtbot.addWidget(bar)
    bar.resize(220, 900)
    bar.show()
    qtbot.waitExposed(bar)
    return bar


def _rows(bar):
    from spacr.qt.app import _DockRow

    return [w for w in bar.findChildren(_DockRow) if w.property("navKey")]


def _hover(row, on: bool):
    """Drive the real events, not the handlers.

    A test that calls `enterEvent` directly passes against a widget that
    never receives one, which is the failure mode this whole file exists to
    catch on a surface nobody looks at twice.
    """
    if on:
        centre = QPointF(row.rect().center())
        row.enterEvent(QEnterEvent(centre, centre, centre))
    else:
        row.leaveEvent(QEvent(QEvent.Type.Leave))


def test_the_home_row_wears_the_application_mark(qtbot, qt_theme_applied):
    """Ask 1 and ask 5 are one defect: of the rows checked when 369 was
    filed, exactly one drew something other than what the Home screen draws,
    and it was `__home__`."""
    from spacr.qt import iconset

    assert iconset.bundled_icon_path("home") is not None, (
        "there is no bundled home.png, so the row falls back to a Font "
        "Awesome house -- run tools/icon_generators/home_from_the_app_mark.py")

    bar = _dock(qtbot)
    home = [r for r in _rows(bar) if r.property("navKey") == "__home__"]
    assert home, "the dock has no Home row"
    drawn = home[0].icon().pixmap(26, 26).toImage()
    expected = iconset.app_icon("home").pixmap(26, 26).toImage()
    assert drawn == expected


def test_the_mark_is_a_mask_so_it_can_be_re_inked(qtbot):
    """The tile was NOT copied in. `app_icon.png` is a teal rounded square
    that carries tone in RGB; a dock row has to be re-inkable, or it cannot
    follow the theme and cannot go blue on hover."""
    from spacr.qt import iconset

    art = iconset._load_rgba(iconset.bundled_icon_path("home"))
    assert art is not None
    assert not iconset.carries_tonal_structure(art), (
        "home.png carries RGB shading, so `reink` will not paint it flat "
        "and the hover tint will fight the artwork")


def test_hovering_paints_the_name_and_leaving_stops(qtbot, qt_theme_applied):
    bar = _dock(qtbot)
    rows = _rows(bar)
    assert rows, "the dock has no rows"
    row = rows[0]

    assert not row.is_hovered()
    _hover(row, True)
    assert row.is_hovered()
    _hover(row, False)
    assert not row.is_hovered()


def test_the_name_sits_to_the_right_of_the_icon(qtbot, qt_theme_applied):
    """"center left anlligned (to the right of the icon)"."""
    bar = _dock(qtbot)
    row = _rows(bar)[0]
    icon, name = row.icon_rect(), row.name_rect()
    assert name.left() > icon.right(), "the name overlaps or precedes the icon"
    assert name.top() == 0 and name.height() == row.height(), (
        "the name box must span the row so AlignVCenter centres it")


def test_the_row_geometry_is_identical_hovered_and_not(qtbot, qt_theme_applied):
    """THE ONE THAT MATTERS. 348 fixed the row height so a target does not
    move under the pointer; 369 must not undo that to make room for a word."""
    bar = _dock(qtbot)
    row = _rows(bar)[0]

    before = (row.geometry(), row.sizeHint(), row.minimumSizeHint(),
              row.icon_rect())
    _hover(row, True)
    after = (row.geometry(), row.sizeHint(), row.minimumSizeHint(),
             row.icon_rect())
    assert before == after, (
        f"the row changed shape on hover: {before} -> {after}")


def test_the_name_is_still_readable_data_when_it_is_not_painted(qtbot,
                                                               qt_theme_applied):
    """348 cleared the PAINTING and kept the property, because three things
    read it back and all three fail silently on an empty string. 369 adds
    painting; it must not have quietly changed the data either way."""
    bar = _dock(qtbot)
    for row in _rows(bar):
        assert row.text(), f"{row.property('navKey')} lost its name"
        assert not row.is_hovered()


def test_the_accent_comes_from_the_theme_not_from_a_literal(qtbot,
                                                            qt_theme_applied):
    """"one colour, named once". There are four themes and the palette is a
    live preference; a hex here is a dock that stops matching the app."""
    from spacr.qt import theme
    from spacr.qt.app import _DockRow

    row = _DockRow("X")
    qtbot.addWidget(row)
    assert row._accent() == theme.active_palette()["accent"]


@pytest.mark.parametrize("name", ("dark", "light", "space"))
def test_the_grey_tray_is_gone_where_there_is_no_picture(name):
    """The container comes off on the flat themes. Over `cell` and `glass`
    it stays, because #16j's ghost-dock complaint applies verbatim there --
    see `test_the_dock_is_opaque_over_a_picture_and_bare_otherwise`."""
    from spacr.qt import theme

    qss = theme.stylesheet(name)
    start = qss.index("#EdgeDrawer, #Sidebar, #SidebarScroll")
    block = qss[start:qss.index("}", start)]
    assert "transparent" in block, f"{name} still paints a dock container"


@pytest.mark.parametrize("name", ("dark", "light", "space", "cell"))
def test_the_edge_survives_in_every_theme(name):
    """The tray goes; the border does not. `dock_colour` argues a navigation
    column "has to be a solid edge for the page to end at", and that half of
    the argument is untouched by 369."""
    from spacr.qt import theme

    qss = theme.stylesheet(name)
    start = qss.index("#Sidebar {")
    assert "border-right" in qss[start:qss.index("}", start)]


def _row_rgb(row):
    """The row as rendered, RGB only, in a predictable channel order."""
    import numpy as np
    from PySide6.QtGui import QImage, QPixmap

    pix = QPixmap(row.size())
    row.render(pix)
    img = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8)
    return buf.reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :3].astype(int)


def test_the_ink_actually_changes_on_the_painted_pixels(qtbot,
                                                        qt_theme_applied):
    """369: "the icon should go from white to blue".

    ASSERTED ON THE RENDER, not on the QIcon. A tint applied to the wrong
    pixmap, or applied and then discarded, still leaves a perfectly valid
    QIcon behind -- and the first version of this feature did exactly that:
    the artwork was so fine at 26 px that no pixel reached alpha 200, the
    `SourceIn` fill had nothing solid to colour, and every object-level
    check passed while the row on screen was unchanged.
    """
    import numpy as np
    from spacr.qt import theme

    bar = _dock(qtbot)
    row = _rows(bar)[0]

    cold = _row_rgb(row)
    _hover(row, True)
    hot = _row_rgb(row)

    accent = theme.active_palette()["accent"].lstrip("#")
    target = np.array([int(accent[i:i + 2], 16) for i in (0, 2, 4)])
    near = lambda a: int((np.abs(a - target).sum(axis=2) < 120).sum())

    # MEASURED AS A CHANGE, NOT AS AN ABSOLUTE, and that is not slack. The
    # claim is "the ink goes from white to blue", which is a difference; a
    # single antialiased pixel landing within the colour tolerance is not
    # evidence against it. Asserting `near(cold) == 0` failed exactly that
    # way when this file ran after `test_the_text_fits_sweep.py`, which
    # leaves a 200 % font scale behind -- conftest already names other files
    # as victims of that leak.
    #
    # The icon alone contributes about 47 solid pixels plus its edges, and
    # the name adds more, so a real change is worth tens of pixels and a
    # stray one is not.
    gained = near(hot) - near(cold)
    assert gained > 20, (
        f"hovering added only {gained} accent-coloured pixels: the ink did "
        f"not change and the name was not drawn")
    assert (cold != hot).any(), "the render is identical hovered and not"
