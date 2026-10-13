"""A tile asked to rescale without a scale follows the one the user saved.

``_rescale_icon_sizes`` hands each tile the new scale, but a tile can also
be asked to re-apply "whatever is current" with no argument. Then the
square tile's frame and caption cap, and its button's resting icon size,
come from the saved interface scale -- the same pixels a tile built at that
scale would have.
"""
from spacr.qt.preferences import _scaled_side, set_font_scale
from spacr.qt.widgets.tile import CAPTION_SLACK_PX, Tile, _TileButton


def test_a_square_tile_follows_the_saved_scale(qtbot):
    tile = Tile("Measure", icon_size=64, tile_size=120)
    qtbot.addWidget(tile)
    set_font_scale(1.5)

    tile._apply_icon_scale()

    side = _scaled_side(120, 1.5)
    assert tile._button.width() == side and tile._button.height() == side
    assert tile._caption.maximumWidth() == _scaled_side(
        120 + CAPTION_SLACK_PX, 1.5)


def test_a_tile_button_rests_at_the_saved_scale(qtbot):
    button = _TileButton(40)
    qtbot.addWidget(button)
    set_font_scale(2.0)

    button._apply_icon_scale()

    assert button._base_size == _scaled_side(40, 2.0)
    assert button.iconSize().width() == _scaled_side(40, 2.0)
