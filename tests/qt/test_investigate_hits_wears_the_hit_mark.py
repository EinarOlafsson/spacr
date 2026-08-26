"""The hit mark belongs to the hit, and two modules draw it.

``hit_list.png`` was drawn for the Hit List tile. The Hit List has folded
onto Regression, so the picture now names *a hit* rather than a tile, and
Investigate Hit -- the module that takes one hit apart -- is what a user
reaches for it expecting. Investigate Hit drew the shared puzzle piece
instead, which is the artwork every unfiled key draws.

Measured in pixels rather than in table lookups: an icon that resolves to
a path but paints nothing, or paints the same silhouette as the fallback,
is the bug this is about. Both keys are checked in every theme, because
the bundled PNGs are re-inked per theme and a mark that only survives on
one of them is half a fix.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pytest

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter

from spacr.qt import iconset
from spacr.qt.theme import THEMES

#: Edge the icons are rasterised at. Larger than any slot they are drawn
#: into, so two silhouettes that genuinely differ cannot come out equal
#: through sheer lack of resolution.
RENDER_PX = 48

#: The two keys that share one asset, on purpose.
HIT_KEYS = ("investigate_hit", "hit_list")


def render(icon, px: int = RENDER_PX):
    """``icon`` painted onto transparency as an ``(px, px, 4)`` array."""
    image = QImage(px, px, QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        icon.paint(painter, 0, 0, px, px)
    finally:
        painter.end()
    buffer = image.constBits()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(px, px, 4).copy()


def coverage(pixels) -> float:
    """Fraction of the square the artwork actually inks."""
    return float((pixels[:, :, 3] > 8).mean())


def silhouette_difference(first, second) -> float:
    """Fraction of pixels where two icons disagree about being there."""
    delta = np.abs(first[:, :, 3].astype(int) - second[:, :, 3].astype(int))
    return float((delta > 64).mean())


@pytest.fixture
def puzzle_piece(qapp):
    """The shared fallback glyph an unfiled key falls through to."""
    qta = iconset._try_qta()
    if qta is None:
        pytest.skip("qtawesome is not installed")
    glyph = qta.icon("fa5s.puzzle-piece", color="#888888")
    assert not glyph.isNull()
    return render(glyph)


class TestOneAssetTwoKeys:
    """The sharing is declared, not a filename coincidence."""

    def test_both_keys_resolve_to_the_same_bundled_file(self):
        paths = {key: iconset.bundled_icon_path(key) for key in HIT_KEYS}
        assert paths["investigate_hit"] is not None
        assert paths["investigate_hit"] == paths["hit_list"]
        assert os.path.basename(paths["hit_list"]) == "hit_list.png"
        assert os.path.isfile(paths["hit_list"])

    def test_investigate_hit_says_out_loud_whose_picture_it_borrows(self):
        """A collision nobody wrote down is the thing this table prevents."""
        assert iconset.SHARED_ICON_ASSETS["investigate_hit"] == "hit_list.png"

    def test_dedicated_artwork_would_retire_the_alias(self, tmp_path,
                                                      monkeypatch):
        """``<key>.png`` outranks the alias, so installing art needs no code.

        The alias exists because Investigate Hit has no drawing of its
        own. The day somebody draws one it must win, or the borrowed
        picture becomes a lock on the key.
        """
        (tmp_path / "hit_list.png").write_bytes(b"")
        (tmp_path / "investigate_hit.png").write_bytes(b"")
        monkeypatch.setattr(iconset, "RESOURCE_DIR", str(tmp_path))
        resolved = iconset.bundled_icon_path("investigate_hit")
        assert os.path.basename(resolved) == "investigate_hit.png"


class TestTheMarkSurvivesEveryTheme:
    """The bundled PNGs are re-inked per theme; both keys must survive it."""

    @pytest.mark.parametrize("theme", THEMES)
    @pytest.mark.parametrize("key", HIT_KEYS)
    def test_the_icon_paints_real_artwork(self, qapp, theme, key):
        icon = iconset.app_icon(key, theme=theme)
        assert not icon.isNull()
        pixels = render(icon)
        # A QIcon that resolves and paints nothing is indistinguishable
        # from a blank button, which is what this whole module exists to
        # stop. hit_list.png inks about 18 % of its square.
        assert coverage(pixels) > 0.05

    @pytest.mark.parametrize("theme", ["light", "dark"])
    @pytest.mark.parametrize("key", HIT_KEYS)
    def test_the_mark_clears_the_contrast_floor(self, theme, key):
        path = iconset.bundled_icon_path(key)
        assert iconset.icon_contrast(path, theme) >= iconset.MIN_ICON_CONTRAST

    def test_the_light_and_dark_marks_are_inked_differently(self):
        """Proof the re-inking ran, rather than one baked colour twice."""
        path = iconset.bundled_icon_path("investigate_hit")
        assert (iconset.icon_ink_color(path, "light")
                != iconset.icon_ink_color(path, "dark"))

    @pytest.mark.parametrize("theme", THEMES)
    def test_both_keys_paint_the_identical_picture(self, qapp, theme):
        first = render(iconset.app_icon("investigate_hit", theme=theme))
        second = render(iconset.app_icon("hit_list", theme=theme))
        assert np.array_equal(first, second)


class TestNotTheFallback:
    """Investigate Hit used to draw the artwork of every unfiled key."""

    @pytest.mark.parametrize("theme", THEMES)
    def test_it_is_not_the_puzzle_piece(self, qapp, puzzle_piece, theme):
        pixels = render(iconset.app_icon("investigate_hit", theme=theme))
        assert silhouette_difference(pixels, puzzle_piece) > 0.2

    def test_it_is_not_the_fontless_fallback_diamond(self, qapp):
        """The other fallback: the glyph drawn when no icon font loads."""
        diamond = render(iconset._fallback_icon("investigate_hit",
                                                "dark", size=RENDER_PX))
        pixels = render(iconset.app_icon("investigate_hit", theme="dark"))
        assert silhouette_difference(pixels, diamond) > 0.1

    def test_the_home_tile_draws_it_too(self, qapp, puzzle_piece):
        """The tile goes through ``app`` and its own override table."""
        from spacr.qt.app import _icon_for_app

        icon = _icon_for_app("investigate_hit")
        assert icon is not None and not icon.isNull()
        pixels = render(icon)
        assert coverage(pixels) > 0.05
        assert silhouette_difference(pixels, puzzle_piece) > 0.2


class TestTheHitListKeepsItsFoldButton:
    """Investigate Hit gaining the mark must not cost the Hit List one."""

    def test_the_hit_list_is_still_a_fold_on_regression(self):
        from spacr.qt.screens import regression

        assert "hit_list" in regression.FOLDED_APPS

    def test_the_fold_button_wears_the_mark_rather_than_an_initial(self,
                                                                   qapp):
        """A FoldButton with no icon falls back to the module's letter."""
        from spacr.qt.widgets.fold_strip import FoldButton

        button = FoldButton("hit_list")
        assert not button.icon().isNull()
        assert button.text() == ""
        pixels = render(button.icon())
        assert coverage(pixels) > 0.05

    def test_the_folded_settings_heading_gets_a_mark_as_well(self, qapp):
        """``module_mark`` answers None for a key with no picture."""
        from spacr.qt.widgets.section import module_mark

        for key in HIT_KEYS:
            mark = module_mark(key)
            assert mark is not None and not mark.isNull()


class TestTheMakeMasksToolRow:
    """Three canvas tools shared the puzzle piece with each other."""

    TOOLS = ("recrop", "draw", "divide")

    @pytest.mark.parametrize("key", TOOLS)
    def test_the_tool_has_a_glyph_of_its_own(self, qapp, key):
        assert key in iconset._NAME_TO_GLYPH

    @pytest.mark.parametrize("key", TOOLS)
    def test_the_tool_is_not_the_puzzle_piece(self, qapp, puzzle_piece, key):
        pixels = render(iconset.icon(key))
        assert coverage(pixels) > 0.05
        assert silhouette_difference(pixels, puzzle_piece) > 0.2

    def test_no_two_tools_draw_the_same_picture(self, qapp):
        """Three buttons wearing one mark is worse than a row with a gap."""
        drawn = {key: render(iconset.icon(key)) for key in self.TOOLS}
        for first, second in itertools.combinations(self.TOOLS, 2):
            assert silhouette_difference(drawn[first], drawn[second]) > 0.1

    def test_every_tool_in_the_row_has_a_mark_of_its_own(self, qapp):
        """The whole row, so the next tool added cannot slip through."""
        from spacr.qt.screens.make_masks import tool_row_entries

        keys = [entry[2] for entry in tool_row_entries()]
        assert set(self.TOOLS) <= set(keys)
        missing = [key for key in keys
                   if key not in iconset._NAME_TO_GLYPH
                   and iconset.bundled_icon_path(key) is None]
        assert missing == []
