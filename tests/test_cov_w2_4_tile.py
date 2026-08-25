"""Home-screen tiles — the hover tween, the icon branch, and the geometry.

Three groups, each of which is a thing a user sees go wrong when it breaks:

* the classic :class:`Tile`'s hover animation. It exists so the icon grows
  without the tile moving, so what is asserted is that the OUTER size is
  unchanged while the icon size follows the animated property;
* the icon branch of the constructor, against the initials branch beside it
  -- a tile with an icon must not also stamp two letters on itself;
* :class:`HTile`'s width reporting. The name lives in a child label, so
  ``QPushButton.sizeHint()`` never measures it; that is the defect these
  overrides exist for, and it is asserted with a genuinely long name and a
  genuinely short one, plus the mid-construction case where the label does
  not exist yet and the plain button hint is all there is to report.
"""
from __future__ import annotations


from PySide6.QtCore import QEvent, QPointF, QSize, Qt
from PySide6.QtGui import QEnterEvent, QIcon, QPixmap

from spacr.qt.widgets.tile import HTile, Tile, _TileButton


def _icon(colour=Qt.red, side=32):
    pixmap = QPixmap(side, side)
    pixmap.fill(colour)
    return QIcon(pixmap)


# ---------------------------------------------------------------------------
# _TileButton
# ---------------------------------------------------------------------------

def test_the_animated_property_drives_the_icon_size(qtbot):
    button = _TileButton(64)
    qtbot.addWidget(button)
    button.setProperty("iconPixels", 40)
    assert button.property("iconPixels") == 40
    assert button.iconSize() == QSize(40, 40)


def test_hovering_aims_the_icon_at_its_zoomed_size(qtbot):
    button = _TileButton(64)
    qtbot.addWidget(button)
    button.setFixedSize(120, 120)

    point = QPointF(10, 10)
    button.enterEvent(QEnterEvent(point, point, point))

    assert button._anim.endValue() == int(64 * 1.18)
    assert button._anim.startValue() == 64
    # The tile itself must not move or resize -- only the icon inside it.
    assert button.size() == QSize(120, 120)


def test_leaving_aims_the_icon_back_at_its_base_size(qtbot):
    button = _TileButton(64)
    qtbot.addWidget(button)
    point = QPointF(10, 10)
    button.enterEvent(QEnterEvent(point, point, point))
    button.setProperty("iconPixels", 75)

    button.leaveEvent(QEvent(QEvent.Type.Leave))

    assert button._anim.startValue() == 75
    assert button._anim.endValue() == 64


# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------

def test_a_tile_with_an_icon_shows_the_icon_and_no_initials(qtbot):
    tile = Tile("Annotator Agreement", icon=_icon(), icon_size=48,
                tile_size=110)
    qtbot.addWidget(tile)
    assert tile._button.icon().isNull() is False
    assert tile._button.text() == ""
    assert tile._button.iconSize() == QSize(48, 48)
    assert tile.text == "Annotator Agreement"


def test_a_tile_with_no_icon_falls_back_to_two_initials(qtbot):
    tile = Tile("Annotator Agreement")
    qtbot.addWidget(tile)
    assert tile._button.icon().isNull() is True
    assert tile._button.text() == "AA"


def test_a_one_word_tile_shows_one_initial(qtbot):
    """Initials come from WORDS, so a single-word name gives a single letter."""
    tile = Tile("Curate")
    qtbot.addWidget(tile)
    assert tile._button.text() == "C"


def test_a_nameless_tile_still_has_something_on_it(qtbot):
    """No words at all falls through to the first two characters."""
    tile = Tile("3d")
    qtbot.addWidget(tile)
    assert tile._button.text() == "3"


def test_the_caption_falls_back_to_the_text(qtbot):
    tile = Tile("Curate")
    qtbot.addWidget(tile)
    assert tile._caption.text() == "Curate"
    assert tile._button.toolTip() == "Curate"


# ---------------------------------------------------------------------------
# HTile geometry
# ---------------------------------------------------------------------------

def test_an_htile_with_an_icon_sizes_it_to_the_font_scale(qtbot):
    """The icon side length is a scaled px, not the raw argument."""
    from spacr.qt.preferences import scaled_px

    tile = HTile("Curate", "review masks", icon=_icon(), icon_size=52)
    qtbot.addWidget(tile)
    assert tile.icon().isNull() is False
    assert tile.iconSize() == QSize(scaled_px(52), scaled_px(52))


def test_a_name_only_htile_centres_its_label_and_keeps_a_plain_tooltip(qtbot):
    tile = HTile("Curate")
    qtbot.addWidget(tile)
    assert tile.toolTip() == "Curate"
    assert tile.accessibleDescription() == ""
    assert tile.name_label.text() == "Curate"
    # Stretch, label, stretch -- three items, no description row.
    assert tile.layout().itemAt(0).layout().count() == 3


def test_a_long_name_widens_the_tile_beyond_the_plain_button_hint(qtbot):
    """The defect these overrides exist for: every tile hinted the same."""
    from PySide6.QtWidgets import QPushButton

    short = HTile("QC", "quality control")
    long = HTile("Annotator Agreement Explorer", "quality control")
    qtbot.addWidget(short)
    qtbot.addWidget(long)

    assert long.required_width() > short.required_width()
    # The plain button hint cannot tell them apart, which is why sizeHint
    # is overridden rather than left alone.
    assert long.sizeHint().width() > QPushButton.sizeHint(long).width()
    assert long.sizeHint().width() >= long.required_width()


def test_the_tile_stays_shrinkable_so_the_name_can_elide(qtbot):
    """A minimum hint as wide as the name would make the grid unshrinkable."""
    tile = HTile("Annotator Agreement Explorer", "quality control")
    qtbot.addWidget(tile)
    minimum = tile.minimumSizeHint()
    assert minimum.width() <= tile.sizeHint().width()
    assert minimum.height() >= tile.minimumHeight()


def test_a_narrow_tile_reports_its_name_as_elided(qtbot):
    tile = HTile("Annotator Agreement Explorer", "quality control")
    qtbot.addWidget(tile)
    tile.resize(tile.required_width() + 40, tile.minimumHeight())
    tile.show()
    qtbot.waitExposed(tile)
    assert tile.is_name_elided() is False

    tile.resize(60, tile.minimumHeight())
    tile.name_label.resize(30, tile.name_label.height())
    assert tile.is_name_elided() is True


def test_the_name_label_and_text_are_reachable(qtbot):
    tile = HTile("Curate", "review masks")
    qtbot.addWidget(tile)
    assert tile.name_label.text() == "Curate"
    assert tile.text_label == "Curate"
    assert tile.accessibleName() == "Curate"
    assert tile.accessibleDescription() == "review masks"


def test_a_width_asked_for_mid_construction_is_the_plain_button_hint(qtbot):
    """Qt can ask for geometry before the name label exists.

    ``setToolTip`` is called from the constructor before the layout and the
    label are built, so overriding it asks the real question at the real
    moment rather than blanking the attribute afterwards.
    """
    from PySide6.QtWidgets import QPushButton

    seen = []

    class AsksTooEarly(HTile):
        def setToolTip(self, text):                       # noqa: N802 - Qt
            seen.append((self._name_lbl, self.layout(),
                         self.required_width()))
            super().setToolTip(text)

    tile = AsksTooEarly("Annotator Agreement Explorer", "quality control")
    qtbot.addWidget(tile)

    label, layout, width = seen[0]
    assert label is None and layout is None
    assert width == QPushButton.sizeHint(tile).width()
    # And once built it reports the real, larger requirement.
    assert tile.required_width() > width
