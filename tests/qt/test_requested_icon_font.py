"""Semantic icons load one family while retaining QtAwesome's exact renderer."""
import types

import pytest
from PySide6.QtGui import QFontDatabase, QIcon, QImage

from spacr.qt import iconset


@pytest.fixture
def provider(qapp, monkeypatch):
    qta = pytest.importorskip("qtawesome")
    monkeypatch.setattr(iconset, "_try_qta", lambda: qta)
    monkeypatch.setattr(qapp, "_spacr_glyph_providers", {}, raising=False)
    monkeypatch.setitem(qta._resource, "iconic", None)
    return qta


def test_only_requested_family_is_loaded_and_reused(qapp, monkeypatch, provider):
    loaded = []
    original = provider.IconicFont.load_font

    def load(self, prefix, *args, **kwargs):
        loaded.append(prefix)
        return original(self, prefix, *args, **kwargs)

    monkeypatch.setattr(provider.IconicFont, "load_font", load)
    monkeypatch.setitem(provider._resource, "iconic", None)
    assert not iconset.icon("settings").isNull()
    assert not iconset.icon("info").isNull()
    assert loaded == ["fa5s"]
    assert provider._resource["iconic"] is None
    assert len(qapp._spacr_glyph_providers) == 1


@pytest.mark.parametrize("theme", ["dark", "light", "space"])
def test_all_semantic_glyphs_match_existing_engine_pixels(provider, theme):
    fill = iconset._theme_palette(theme)["fg_muted"]
    for name in [*iconset._NAME_TO_GLYPH, "unknown-glyph"]:
        actual = iconset.icon(name, theme=theme)
        expected = provider.icon(iconset._NAME_TO_GLYPH.get(name, "fa5s.puzzle-piece"), color=fill)
        assert not actual.isNull()
        for size in (16, 48):
            for mode in (QIcon.Normal, QIcon.Disabled, QIcon.Active, QIcon.Selected):
                for state in (QIcon.On, QIcon.Off):
                    got = actual.pixmap(size, mode, state).toImage()
                    wanted = expected.pixmap(size, mode, state).toImage()
                    assert not got.isNull()
                    if size == 16 and mode == QIcon.Normal and state == QIcon.Off:
                        rgba = got.convertToFormat(QImage.Format_RGBA8888)
                        assert any(bytes(rgba.constBits())[3::4]), (theme, name)
                    assert got == wanted, (theme, name, size, mode, state)


def test_font_database_reset_reloads_the_requested_provider(provider):
    first = iconset._glyph_provider(provider, "fa5s.cog")
    assert first is not provider
    for font_id in first.fontids.values():
        assert QFontDatabase.removeApplicationFont(font_id)
    second = iconset._glyph_provider(provider, "fa5s.cog")
    assert second is not first and second is not provider
    assert not second.icon("fa5s.cog", color="white").isNull()


@pytest.mark.parametrize("change", ["missing_metadata", "custom_directory", "system_fonts", "bad_checksum", "failed_load", "empty_ids", "invalid_ids"])
def test_unrecognized_or_failed_font_setup_retains_original_api(provider, change):
    fake = types.SimpleNamespace(**{key: value for key, value in vars(provider).items()
                                   if key not in ("__name__", "__dict__")})
    if change == "missing_metadata":
        del fake._BUNDLED_FONTS
    elif change == "custom_directory":
        fake._BUNDLED_FONTS = [("fa5s", "custom.ttf", "custom.json", "/custom")]
    elif change == "system_fonts":
        fake._SYSTEM_FONTS = True
    elif change == "bad_checksum":
        fake._MD5_HASHES = {row[1]: "not-the-checksum" for row in fake._BUNDLED_FONTS}
    elif change == "empty_ids":
        fake.IconicFont = lambda *args: types.SimpleNamespace(fontids={})
    elif change == "invalid_ids":
        fake.IconicFont = lambda *args: types.SimpleNamespace(fontids={"fa5s": -9999})
    else:
        def fail(*args):
            raise RuntimeError("font unavailable")
        fake.IconicFont = fail
    assert iconset._glyph_provider(fake, "fa5s.cog") is fake


def test_unknown_font_family_retains_original_api(provider):
    assert iconset._glyph_provider(provider, "unknown.font") is provider


def test_explicit_color_keeps_the_existing_renderer(provider):
    actual = iconset.icon("settings", color="#19bca4")
    expected = provider.icon("fa5s.cog", color="#19bca4")
    assert actual.pixmap(32, 32).toImage() == expected.pixmap(32, 32).toImage()


def test_existing_global_fonts_are_reused_without_another_registration(provider, monkeypatch):
    assert not provider.icon("fa5s.cog").isNull()

    def fail(*args):
        pytest.fail("an already initialized font must not load again")

    monkeypatch.setattr(provider, "IconicFont", fail)
    assert iconset._glyph_provider(provider, "fa5s.cog") is provider
    assert not iconset.icon("settings").isNull()


def test_no_application_defers_to_the_original_api(provider, monkeypatch):
    from PySide6.QtWidgets import QApplication

    with monkeypatch.context() as patch:
        patch.setattr(QApplication, "instance", lambda: None)
        assert iconset._glyph_provider(provider, "fa5s.cog") is provider


def test_an_upstream_font_without_a_known_checksum_can_still_render(provider, monkeypatch):
    monkeypatch.setattr(provider, "_MD5_HASHES", {})
    selected = iconset._glyph_provider(provider, "fa5s.cog")
    assert selected is not provider
    assert not selected.icon("fa5s.cog", color="white").isNull()
