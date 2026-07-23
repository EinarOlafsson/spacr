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
        from spacr.qt.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        labels = [w.text() for w in win._startup.findChildren(QLabel)]
        assert any("Reserved for featured" in lbl for lbl in labels)
        assert any(lbl == "FEATURED" for lbl in labels)

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
    def test_horizontal_row_tiles_omit_description(self, qtbot,
                                                       _empty_journal):
        """The horizontal rows now show name only, no wrapped
        description — that eliminates the cut-off text symptom."""
        from spacr.qt.app import MainWindow
        from spacr.qt.widgets.tile import HTile
        win = MainWindow()
        qtbot.addWidget(win)
        tiles = win._startup.findChildren(HTile)
        assert tiles, "no HTile widgets under home"
        for t in tiles:
            desc_labels = [c for c in t.findChildren(QLabel)
                             if c.objectName() == "HTileDesc"]
            assert not desc_labels, (
                "horizontal-row tiles should not carry a description "
                "label (name only)")

    def test_tile_icon_size_is_larger(self, qtbot, _empty_journal):
        """Icons should be 44 px in the new layout, not 28."""
        from spacr.qt.app import MainWindow
        from spacr.qt.widgets.tile import HTile
        win = MainWindow()
        qtbot.addWidget(win)
        tiles = win._startup.findChildren(HTile)
        assert tiles
        for t in tiles:
            assert t.iconSize().width() >= 40
