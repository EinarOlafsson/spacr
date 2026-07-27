"""Tests for the Batch H home-screen fixes: hero layout, tile icons,
reserved surface, plaque icon override, and tile-text-not-cut-off."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel


@pytest.fixture
def _empty_journal(tmp_path, monkeypatch):
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    yield tmp_path


class TestHeroLayout:
    def test_home_has_logo_and_subtitle(self, qtbot, _empty_journal):
        from spacr.qt.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        labels = [w.text() for w in win._startup.findChildren(QLabel)]
        # Subtitle is present on Home
        assert any("End-to-end" in lbl for lbl in labels)
        # Wordmark is present
        assert any(lbl == "spaCR" for lbl in labels)


class TestReservedSurface:
    def test_reserved_placeholder_shows_caption(self, qtbot,
                                                    _empty_journal):
        """The reserved surface survived the Home redesign — it moved.

        It used to be a full-width grey box captioned "FEATURED" below
        the app rows. It is now the News panel in the right-hand column:
        same escape hatch (``set_reserved_content``), same placeholder
        wording, but a heading that says what would actually go there.
        A panel headed FEATURED with nothing featured in it was 140 px
        of page saying nothing.
        """
        from spacr.qt.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        labels = [w.text() for w in win._startup.findChildren(QLabel)]
        assert any("Reserved for featured" in lbl for lbl in labels)
        assert any(lbl.startswith("NEWS") for lbl in labels)

    def test_set_reserved_content_swaps_widget(self, qtbot,
                                                   _empty_journal):
        from spacr.qt.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        marker = QLabel("REPLACED")
        win._startup.set_reserved_content(marker)
        labels = [w.text() for w in win._startup.findChildren(QLabel)]
        assert "REPLACED" in labels
        # The panel now knows the new content
        assert win._startup._reserved_content is marker


class TestPlaqueIconOverride:
    def test_plaque_key_finds_bundled_icon(self):
        from spacr.qt.app import _icon_for_app
        icon = _icon_for_app("analyze_plaques")
        assert icon is not None
        # A returned QIcon should be non-null (has pixmap)
        assert not icon.pixmap(16, 16).isNull()


class TestTileText:
    def test_tiles_omit_the_description(self, qtbot, _empty_journal):
        """Name only, no wrapped description — that eliminates the
        cut-off text symptom.

        The tile class changed under this test: it asserted no
        ``QLabel#HTileDesc`` on the horizontal ``HTile`` rows Home used
        to draw. Home draws ``AppTile`` now — icon over name — and the
        equivalent statement is that a tile carries exactly two labels,
        one of which is the icon."""
        from spacr.qt.app import MainWindow
        from spacr.qt.widgets.home import AppTile
        win = MainWindow()
        qtbot.addWidget(win)
        tiles = win._startup.findChildren(AppTile)
        assert tiles, "no AppTile widgets under home"
        for t in tiles:
            labels = t.findChildren(QLabel)
            assert len(labels) == 2, (
                f"{t.text_label} draws {[lbl.text() for lbl in labels]} — "
                "a tile is an icon and a name, nothing else")
            assert sum(1 for lbl in labels if lbl.pixmap()) == 1

    def test_tile_icon_size_is_larger(self, qtbot, _empty_journal):
        """Icons should be big in the new layout, not 28 px."""
        from spacr.qt.app import MainWindow
        from spacr.qt.widgets.home import AppTile
        win = MainWindow()
        qtbot.addWidget(win)
        tiles = win._startup.findChildren(AppTile)
        assert tiles
        for t in tiles:
            glyph = next(lbl for lbl in t.findChildren(QLabel) if lbl.pixmap())
            assert glyph.width() >= 40
