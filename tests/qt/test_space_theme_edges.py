"""Defensive paths in the Space theme and the themed-icon loader.

Every branch here exists because something can genuinely go wrong on a
user's machine — a read-only home directory, a half-written cache file,
no qtawesome, a display that reports nonsense. Exercising them for real
(rather than annotating them away) is the only way to know the fallback
is a fallback and not a second crash.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr.qt import space


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    target = tmp_path / "backgrounds"
    monkeypatch.setenv(space.ENV_CACHE_DIR, str(target))
    return target


# ---------------------------------------------------------------------------
# numpy helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_upsample_to_the_same_size_is_a_no_op(self):
        src = np.zeros((4, 6, 3), dtype=np.float32)
        assert space._bilinear_upsample(src, 6, 4) is src

    def test_upsample_from_a_single_row_or_column(self):
        src = np.ones((1, 1, 3), dtype=np.float32)
        out = space._bilinear_upsample(src, 5, 3)
        assert out.shape == (3, 5, 3)
        assert np.allclose(out, 1.0)

    def test_box_blur_with_zero_radius_is_a_no_op(self):
        img = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        assert space._box_blur(img, 0) is img

    def test_box_blur_preserves_a_constant_field(self):
        img = np.full((8, 8, 3), 0.5, dtype=np.float32)
        assert np.allclose(space._box_blur(img, 2), 0.5)

    def test_area_downsample_averages(self):
        img = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
        out = space._area_downsample(img, 2)
        assert out.shape == (2, 2, 1)
        assert out[0, 0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))

    def test_area_downsample_handles_a_ragged_size(self):
        img = np.ones((5, 5, 3), dtype=np.float32)
        assert space._area_downsample(img, 2).shape == (2, 2, 3)

    def test_area_downsample_factor_below_one_is_clamped(self):
        img = np.ones((4, 4, 3), dtype=np.float32)
        assert space._area_downsample(img, 0).shape == (4, 4, 3)


class TestSplat:
    def test_no_stars_gives_an_empty_buffer(self):
        empty = np.zeros(0)
        buf = space._splat(8, 8, empty, empty, empty,
                           np.zeros((0, 3)), np.zeros(0))
        assert buf.shape == (8, 8, 3)
        assert not buf.any()

    def test_stars_entirely_off_canvas_contribute_nothing(self):
        xs = np.array([-500.0, 900.0])
        ys = np.array([-500.0, 900.0])
        buf = space._splat(16, 16, xs, ys, np.array([1.0, 1.0]),
                           np.ones((2, 3)), np.array([1.0, 1.0]))
        assert not buf.any()

    def test_a_star_at_the_edge_is_clipped_not_wrapped(self):
        xs = np.array([0.5])
        ys = np.array([0.5])
        buf = space._splat(16, 16, xs, ys, np.array([4.0]),
                           np.ones((1, 3)), np.array([1.2]))
        assert buf[0, 0].sum() > 0
        assert buf[15, 15].sum() == 0, "flux wrapped around the edge"

    def test_negligible_flux_is_dropped(self):
        xs = np.array([8.0])
        ys = np.array([8.0])
        buf = space._splat(16, 16, xs, ys, np.array([1e-9]),
                           np.ones((1, 3)), np.array([1.0]))
        assert not buf.any()

    def test_no_spikes_without_stars(self):
        empty = np.zeros(0)
        buf = space._diffraction_spikes(8, 8, empty, empty, empty,
                                        np.zeros((0, 3)))
        assert not buf.any()

    def test_spikes_off_canvas_are_skipped(self):
        buf = space._diffraction_spikes(
            8, 8, np.array([-90.0]), np.array([-90.0]),
            np.array([1.0]), np.ones((1, 3)))
        assert not buf.any()


class TestToneMap:
    def test_a_black_frame_does_not_divide_by_zero(self):
        black = np.zeros((8, 8, 3), dtype=np.float32)
        out = space._tone_map(black)
        assert out.shape == black.shape
        assert not out.any()

    def test_exposure_falls_back_when_there_is_no_signal(self):
        assert space._solve_exposure(np.zeros((4, 4)), 0.1, 40.0) == 1.0
        assert space._solve_exposure(np.zeros((0, 0)), 0.1, None) == 1.0

    def test_the_mean_ceiling_overrides_the_sky_anchor(self):
        """A frame that is bright everywhere must be pulled down by the
        mean cap, not left at the sky-percentile exposure."""
        bright = np.full((64, 64, 3), 40.0, dtype=np.float32)
        out = space._tone_map(bright)
        luma = (0.2126 * out[:, :, 0] + 0.7152 * out[:, :, 1]
                + 0.0722 * out[:, :, 2])
        assert luma.mean() <= space.MAX_MEAN_LUMA + 1e-3


class TestCacheEdges:
    def test_a_cache_entry_of_the_wrong_size_is_rejected(self, cache_dir,
                                                         qapp):
        good = space.background_path(64, 48, "stars", seed=2)
        wrong = space.background_path(96, 64, "stars", seed=2)
        # Put the wrong-sized image under the right name.
        good.write_bytes(wrong.read_bytes())
        assert not space._load_cached(good, 64, 48)
        recovered = space.background_path(64, 48, "stars", seed=2)
        assert space._load_cached(recovered, 64, 48)

    def test_a_zero_byte_cache_entry_is_rejected(self, cache_dir, qapp):
        path = cache_dir / "empty.png"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        assert not space._load_cached(path, 10, 10)

    def test_a_missing_cache_entry_is_rejected(self, cache_dir):
        assert not space._load_cached(cache_dir / "nope.png", 10, 10)

    def test_a_big_but_undecodable_cache_entry_is_rejected(self, cache_dir,
                                                           qapp):
        """Large enough to pass the size check, still not an image — a
        half-synced file, or a disk that returned garbage."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "garbage.png"
        path.write_bytes(b"NOT-A-PNG" * 200)
        assert not space._load_cached(path, 10, 10)

    def test_load_cached_never_raises(self, cache_dir, monkeypatch):
        monkeypatch.setattr(space.Path, "is_file",
                            lambda self: (_ for _ in ()).throw(OSError("nope")))
        assert not space._load_cached(cache_dir / "x.png", 10, 10)

    def test_a_failed_save_returns_none(self, cache_dir, qapp, monkeypatch):
        class Unsaveable:
            def save(self, *args, **kwargs):
                return False

        monkeypatch.setattr(space, "to_qimage", lambda arr: Unsaveable())
        assert space.background_path(64, 64, "stars", seed=9) is None

    def test_clear_cache_survives_an_undeletable_file(self, cache_dir, qapp,
                                                     monkeypatch):
        space.background_path(48, 48, "stars", seed=4)

        def refuse(self, missing_ok=False):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(space.Path, "unlink", refuse)
        assert space.clear_cache() == 0

    def test_clear_cache_survives_an_unreadable_directory(self, monkeypatch):
        monkeypatch.setattr(space, "cache_dir",
                            lambda: (_ for _ in ()).throw(OSError("gone")))
        assert space.clear_cache() == 0

    def test_cache_dir_defaults_under_the_home_directory(self, monkeypatch):
        monkeypatch.delenv(space.ENV_CACHE_DIR, raising=False)
        assert space.cache_dir().parts[-2:] == (".spacr", "backgrounds")


class _FakeGeometry:
    def __init__(self, width, height):
        self._w, self._h = width, height

    def width(self):
        return self._w

    def height(self):
        return self._h


class _FakeScreen:
    def __init__(self, width, height, ratio=1.0):
        self._geo = _FakeGeometry(width, height)
        self._ratio = ratio

    def geometry(self):
        return self._geo

    def devicePixelRatio(self):
        return self._ratio


class _FakeApp:
    def __init__(self, screen):
        self._screen = screen

    def primaryScreen(self):
        return self._screen


class TestScreenSize:
    """Faked through ``space._gui_app`` rather than by patching
    ``QGuiApplication.instance``, which would break every later test
    that needs a real application."""

    def test_reads_the_real_screen(self, qapp):
        assert space.screen_size() == space.screen_size()

    def test_falls_back_without_a_gui_application(self, monkeypatch):
        monkeypatch.setattr(space, "_gui_app", lambda: None)
        assert space.screen_size(default=(111, 222)) == (111, 222)

    def test_falls_back_without_a_primary_screen(self, monkeypatch):
        monkeypatch.setattr(space, "_gui_app", lambda: _FakeApp(None))
        assert space.screen_size(default=(123, 456)) == (123, 456)

    def test_falls_back_on_a_nonsense_geometry(self, monkeypatch):
        monkeypatch.setattr(space, "_gui_app",
                            lambda: _FakeApp(_FakeScreen(1, 1)))
        assert space.screen_size(default=(321, 654)) == (321, 654)

    def test_falls_back_when_qt_raises(self, monkeypatch):
        def boom():
            raise RuntimeError("no display")

        monkeypatch.setattr(space, "_gui_app", boom)
        assert space.screen_size(default=(9, 9)) == (9, 9)

    def test_device_pixel_ratio_is_honoured(self, monkeypatch):
        """A 1920-logical-px screen at 2x is 3840 real pixels — the whole
        reason for generating at native resolution."""
        monkeypatch.setattr(
            space, "_gui_app",
            lambda: _FakeApp(_FakeScreen(1920, 1080, ratio=2.0)))
        assert space.screen_size() == (3840, 2160)

    def test_an_8k_panel_is_clamped(self, monkeypatch):
        monkeypatch.setattr(
            space, "_gui_app",
            lambda: _FakeApp(_FakeScreen(7680, 4320, ratio=1.0)))
        assert space.screen_size() == space.MAX_DIM

    def test_a_zero_device_pixel_ratio_does_not_collapse_the_size(
            self, monkeypatch):
        monkeypatch.setattr(
            space, "_gui_app",
            lambda: _FakeApp(_FakeScreen(2000, 1300, ratio=0.0)))
        assert space.screen_size() == (2000, 1300)

    def test_a_small_virtual_screen_still_covers_a_normal_window(
            self, monkeypatch):
        """The QSS centres the background without repeating it, so an
        image narrower than the window shows hard-edged bands of flat
        colour on either side."""
        monkeypatch.setattr(
            space, "_gui_app",
            lambda: _FakeApp(_FakeScreen(800, 600, ratio=1.0)))
        assert space.screen_size() == space.MIN_BACKGROUND


class TestUrlOpener:
    def test_the_default_opener_uses_urllib(self, monkeypatch):
        """Never actually reaches the network: urlopen is replaced."""
        import urllib.request

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"payload"

        seen = {}

        def fake_urlopen(url, timeout=None):
            seen["url"] = url
            seen["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        assert space._urlopen_bytes("https://example.invalid/x", 3.0) == b"payload"
        assert seen == {"url": "https://example.invalid/x", "timeout": 3.0}


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------

class TestIconEdges:
    def test_missing_qtawesome_degrades_to_an_empty_icon(self, monkeypatch,
                                                         qapp):
        from spacr.qt import iconset
        monkeypatch.setattr(iconset, "_try_qta", lambda: None)
        assert iconset.icon("run").isNull()
        assert iconset.accent_icon("run").isNull()
        assert iconset.contrast_icon("run").isNull()

    def test_a_broken_qtawesome_degrades_to_an_empty_icon(self, monkeypatch,
                                                          qapp):
        from spacr.qt import iconset

        class Broken:
            def icon(self, *args, **kwargs):
                raise RuntimeError("font not found")

        monkeypatch.setattr(iconset, "_try_qta", lambda: Broken())
        assert iconset.icon("run").isNull()

    def test_try_qta_returns_none_when_the_import_fails(self, monkeypatch):
        import builtins

        from spacr.qt import iconset
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "qtawesome":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        iconset._try_qta.cache_clear()
        try:
            assert iconset._try_qta() is None
        finally:
            iconset._try_qta.cache_clear()

    def test_accent_and_contrast_icons_use_different_colours(self, qapp):
        from spacr.qt import iconset
        from spacr.qt.theme import palette_for
        palette = palette_for("dark")
        assert palette["accent"] != palette["button_accent_ink"]
        # Both must resolve without raising even if qtawesome is absent.
        assert iconset.accent_icon("run") is not None
        assert iconset.contrast_icon("run") is not None

    def test_veil_falls_back_to_full_ink_on_an_impossible_palette(
            self, monkeypatch, qapp):
        """A palette whose foreground barely differs from its surfaces
        cannot produce a 3:1 veil; use the ink itself rather than loop
        forever or emit something invisible."""
        from spacr.qt import iconset
        flat = {"fg": "#808080", "bg": "#7f7f7f", "surface": "#7f7f7f",
                "surface_alt": "#7f7f7f", "surface_hi": "#7f7f7f"}
        monkeypatch.setattr(iconset, "_theme_palette", lambda theme: flat)
        monkeypatch.setattr(iconset, "effective_surface",
                            lambda theme, role, **kw: flat[role])
        iconset.veil_color.cache_clear()
        try:
            assert iconset.veil_color("flat") == "#808080"
        finally:
            iconset.veil_color.cache_clear()

    def test_structure_test_on_empty_artwork(self, qapp):
        from spacr.qt import iconset
        assert not iconset.carries_tonal_structure(np.zeros((4, 4, 4)))

    def test_fully_transparent_artwork_has_no_ink_colour(self, monkeypatch,
                                                          qapp):
        from spacr.qt import iconset
        blank = np.zeros((6, 6, 4), dtype=np.uint8)
        monkeypatch.setattr(iconset, "themed_array",
                            lambda path, theme=None: blank)
        assert iconset.icon_ink_color("anything.png") is None
        assert iconset.icon_contrast("anything.png") == 0.0

    def test_themed_pixmap_is_none_for_an_empty_image(self, monkeypatch, qapp):
        from spacr.qt import iconset
        from PySide6.QtGui import QImage
        monkeypatch.setattr(iconset, "themed_qimage",
                            lambda path, theme=None: QImage())
        assert iconset.themed_pixmap("whatever.png") is None

    def test_file_stamp_of_a_missing_file(self, qapp):
        from spacr.qt import iconset
        assert iconset._file_stamp("/no/such/icon.png") == ("/no/such/icon.png",
                                                             0, 0)

    def test_bundled_icon_path_prefers_the_override(self, qapp):
        from spacr.qt import iconset
        path = iconset.bundled_icon_path("analyze_plaques",
                                         override="plaque.png")
        assert path is not None and path.endswith("plaque.png")

    def test_bundled_icon_path_falls_back_to_the_key(self, qapp):
        from spacr.qt import iconset
        path = iconset.bundled_icon_path("mask")
        assert path is not None and path.endswith("mask.png")

    def test_bundled_icon_path_is_none_for_an_unknown_key(self, qapp):
        from spacr.qt import iconset
        assert iconset.bundled_icon_path("no-such-thing") is None

    def test_app_icon_falls_back_when_the_png_is_unreadable(self, monkeypatch,
                                                            qapp):
        from spacr.qt import iconset
        monkeypatch.setattr(iconset, "themed_pixmap",
                            lambda path, theme=None: None)
        assert iconset.app_icon("mask") is not None

    def test_the_app_registry_and_iconset_agree(self, qapp):
        """Every override in app.py must point at a file that exists."""
        from spacr.qt import app, iconset
        for key, filename in app._ICON_OVERRIDES.items():
            assert os.path.isfile(os.path.join(iconset.RESOURCE_DIR, filename)), \
                f"{key} -> {filename} does not exist"

    def test_icon_for_app_still_honours_force_glyph(self, qapp):
        from spacr.qt import app
        assert app._icon_for_app("invasion") is not None
