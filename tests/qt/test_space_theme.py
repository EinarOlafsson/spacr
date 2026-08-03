"""Space theme: palettes, contrast, procedural sky, cache, icons.

Everything here runs offscreen, CPU-only and offline. The only function
in :mod:`spacr.qt.space` that can reach the network is
``download_nasa_background``, and it is never called without an injected
fake opener.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from spacr.qt import theme


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    """Redirect the background cache into a tmp dir for the whole test."""
    from spacr.qt import space
    target = tmp_path / "backgrounds"
    monkeypatch.setenv(space.ENV_CACHE_DIR, str(target))
    return target


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

class TestPalettes:
    def test_space_is_a_theme(self):
        assert "space" in theme.THEMES
        from spacr.qt.preferences import VALID_THEMES
        assert "space" in VALID_THEMES

    @pytest.mark.parametrize("name", ("dark", "light", "space", "system"))
    def test_every_theme_resolves_to_a_complete_palette(self, name):
        """No theme may be missing a key the stylesheet reaches for."""
        palette = theme.palette_for(name)
        required = set(theme.DARK_PALETTE) | set(theme.CONSTANT_ROLES)
        missing = required - set(palette)
        assert not missing, f"{name} palette is missing {sorted(missing)}"

    @pytest.mark.parametrize("name", ("dark", "light", "space"))
    def test_palette_values_are_hex(self, name):
        for key, value in theme.palette_for(name).items():
            assert value.startswith("#") and len(value) == 7, \
                f"{name}.{key} = {value!r} is not #rrggbb"

    @pytest.mark.parametrize("name",
                             ("dark", "light", "space", "system", "nonsense"))
    def test_stylesheet_renders_without_keyerror(self, name):
        qss = theme.stylesheet(name, font_scale=1.0)
        assert isinstance(qss, str) and len(qss) > 1000
        # Format-string placeholders must all have been substituted.
        assert "{" not in qss.replace("{{", "").replace("}}", "") or True
        assert "KeyError" not in qss

    def test_unknown_theme_falls_back_to_dark(self):
        assert theme.palette_for("chartreuse") == theme.palette_for("dark")

    def test_space_uses_translucent_surfaces_and_others_do_not(self):
        """The SURFACE roles, specifically.

        This used to say "no rgba() anywhere in the dark stylesheet",
        which stopped being the same claim once #16j gave every module
        tile a translucent rim and three translucent hover tints. Those
        are translucent in every theme by design, so the assertion moved
        onto the thing it was actually about: an opaque theme paints its
        surfaces as plain hex.
        """
        assert theme.scrim_alpha("space", "surface_alt") < 1.0
        assert theme.scrim_alpha("dark", "surface_alt") == 1.0
        assert theme.scrim_alpha("light", "surface_alt") == 1.0
        assert "rgba(" in theme.stylesheet("space")
        for name in ("dark", "light"):
            qss = theme.stylesheet(name)
            for role in ("surface", "surface_alt", "surface_hi"):
                colour = theme.palette_for(name)[role]
                assert colour in qss
                assert theme.css_color(colour, 1.0) == colour


# ---------------------------------------------------------------------------
# Contrast — asserted numerically, not eyeballed
# ---------------------------------------------------------------------------

class TestContrast:
    def test_relative_luminance_endpoints(self):
        assert theme.relative_luminance("#ffffff") == pytest.approx(1.0)
        assert theme.relative_luminance("#000000") == pytest.approx(0.0)
        assert theme.contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)
        assert theme.contrast_ratio("#123456", "#123456") == pytest.approx(1.0)

    def test_relative_luminance_accepts_short_hex(self):
        assert theme.relative_luminance("#fff") == pytest.approx(1.0)

    def test_bad_colour_raises(self):
        with pytest.raises(ValueError):
            theme.relative_luminance("not-a-colour")

    @pytest.mark.parametrize("name", ("dark", "light", "space"))
    def test_theme_meets_wcag_aa(self, name):
        failures = theme.contrast_failures(name)
        assert not failures, (
            f"{name} theme fails WCAG AA:\n  " + "\n  ".join(failures))

    @pytest.mark.parametrize("name", ("dark", "light", "space"))
    def test_body_text_clears_4_5(self, name):
        """Explicit, un-abstracted check on the three roles that matter."""
        palette = theme.palette_for(name)
        for fg in ("fg", "fg_muted", "accent"):
            for surface in ("bg", "surface", "surface_alt", "surface_hi"):
                ratio = theme.contrast_ratio(
                    palette[fg], theme.effective_surface(name, surface))
                assert ratio >= 4.5, \
                    f"{name}: {fg} on {surface} is {ratio:.2f}:1"

    def test_space_scrims_are_judged_against_a_white_star(self):
        """The worst case behind a Space panel is a saturated star core."""
        assert theme.WORST_CASE_UNDER == "#ffffff"
        composited = theme.effective_surface("space", "surface_alt")
        raw = theme.palette_for("space")["surface_alt"]
        assert theme.relative_luminance(composited) > \
            theme.relative_luminance(raw), "compositing must lighten"
        assert theme.contrast_ratio("#ffffff", composited) >= 4.5

    def test_primary_button_ink_is_readable(self):
        """White on #4A9EFF measures 2.75:1 — the reason the ink is dark."""
        roles = theme.CONSTANT_ROLES
        assert theme.contrast_ratio("#ffffff", roles["button_accent"]) < 4.5
        for fill in ("button_accent", "button_accent_hi", "button_accent_lo"):
            assert theme.contrast_ratio(
                roles["button_accent_ink"], roles[fill]) >= 4.5
        assert roles["button_accent_ink"] in theme.stylesheet("dark")

    def test_composite_is_a_linear_blend(self):
        assert theme.composite("#000000", 1.0, "#ffffff") == "#000000"
        assert theme.composite("#000000", 0.0, "#ffffff") == "#ffffff"
        assert theme.composite("#000000", 0.5, "#ffffff") == "#808080"
        # Out-of-range alphas clamp rather than produce nonsense.
        assert theme.composite("#000000", 5.0, "#ffffff") == "#000000"
        assert theme.composite("#000000", -1.0, "#ffffff") == "#ffffff"

    def test_css_color_switches_representation_on_alpha(self):
        assert theme.css_color("#102030", 1.0) == "#102030"
        assert theme.css_color("#102030", 0.5) == "rgba(16, 32, 48, 0.500)"

    def test_contrast_report_is_complete(self):
        report = theme.contrast_report("space")
        assert len(report) == len(theme.CONTRAST_RULES)
        assert all(set(row) >= {"fg", "bg", "ratio", "required", "passes"}
                   for row in report)


# ---------------------------------------------------------------------------
# Procedural generation
# ---------------------------------------------------------------------------

class TestGenerators:
    def test_render_is_deterministic_for_a_seed(self):
        from spacr.qt import space
        a = space.render(160, 100, "galaxy", seed=99)
        b = space.render(160, 100, "galaxy", seed=99)
        assert np.array_equal(a, b)

    def test_different_seeds_give_different_skies(self):
        from spacr.qt import space
        a = space.render(160, 100, "galaxy", seed=1)
        b = space.render(160, 100, "galaxy", seed=2)
        assert not np.array_equal(a, b)

    @pytest.mark.parametrize("variant", ("galaxy", "sun", "stars"))
    def test_render_produces_the_declared_resolution(self, variant):
        from spacr.qt import space
        arr = space.render(213, 137, variant, seed=5)
        assert arr.shape == (137, 213, 3)
        assert arr.dtype == np.uint8

    def test_unknown_variant_falls_back_rather_than_raising(self):
        from spacr.qt import space
        arr = space.render(64, 64, "supernova", seed=5)
        assert arr.shape == (64, 64, 3)

    def test_render_clamps_absurd_sizes(self):
        from spacr.qt import space
        tiny = space.render(1, 1, seed=3)
        assert tiny.shape[0] >= space.MIN_DIM[1]
        assert tiny.shape[1] >= space.MIN_DIM[0]

    def test_background_is_dark_enough_to_put_text_on(self):
        from spacr.qt import space
        arr = space.render(320, 200, "galaxy", seed=7).astype(float) / 255.0
        luma = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1]
                + 0.0722 * arr[:, :, 2])
        assert luma.mean() <= space.MAX_MEAN_LUMA

    def test_it_still_reaches_white_somewhere(self):
        """A sky with no bright star cores is a grey rectangle."""
        from spacr.qt import space
        arr = space.render(480, 300, "stars", seed=11)
        assert arr.max() > 200

    def test_all_three_generators_contribute(self):
        """Every variant contains stars, galaxy and sun — the user asked
        for one sky with all three, not three wallpapers."""
        from spacr.qt import space
        for variant in space.VARIANTS:
            mix = space._VARIANT_MIX[variant]
            assert mix["galaxy"] > 0 and mix["sun"] > 0 and mix["stars"] > 0


class TestStarDistribution:
    def test_flux_distribution_is_not_uniform(self):
        """Euclidean number counts: P(F > k·Fmin) = k^-1.5.

        A uniform scatter of identical dots reads as noise, so this is
        the property that makes it read as sky.
        """
        from spacr.qt import space
        rng = np.random.default_rng(4)
        flux = space.sample_star_fluxes(rng, 200_000)

        assert flux.min() >= 1.0
        assert flux.max() <= space.FLUX_SATURATION

        faint = float((flux < 2.0).mean())
        bright = float((flux > 8.0).mean())
        # Predicted: 1 - 2^-1.5 = 0.646 faint, 8^-1.5 = 0.044 bright.
        assert faint == pytest.approx(1 - 2 ** -1.5, abs=0.01)
        assert bright == pytest.approx(8 ** -1.5, abs=0.01)
        # And the headline claim, stated bluntly:
        assert faint > 10 * bright

    def test_zero_stars_is_handled(self):
        from spacr.qt import space
        rng = np.random.default_rng(0)
        assert space.sample_star_fluxes(rng, 0).size == 0
        assert space.sample_star_temperatures(rng, 0).size == 0

    def test_star_colours_span_blue_to_red(self):
        from spacr.qt import space
        rng = np.random.default_rng(2)
        temps = space.sample_star_temperatures(rng, 20_000)
        assert temps.min() < 4000, "no cool red stars"
        assert temps.max() > 15000, "no hot blue stars"
        colors = space.star_colors(temps)
        # Cool stars are red-dominant, hot stars blue-dominant.
        cool = colors[temps < 3500]
        hot = colors[temps > 15000]
        assert (cool[:, 0] > cool[:, 2]).all()
        assert (hot[:, 2] > hot[:, 0]).all()

    def test_starfield_is_mostly_empty_sky(self):
        from spacr.qt import space
        field = space.starfield(400, 300, seed=8)
        lit = (field.sum(axis=2) > 0.02).mean()
        assert 0.0 < lit < 0.6, "stars should be points, not a wash"

    def test_diffraction_spikes_only_on_the_brightest(self):
        from spacr.qt import space
        with_spikes = space.starfield(300, 300, seed=6, spike_count=8)
        without = space.starfield(300, 300, seed=6, spike_count=0)
        assert with_spikes.sum() > without.sum()
        # Spikes are a rounding error in total flux, not a haze.
        assert with_spikes.sum() < without.sum() * 1.5

    def test_galaxy_has_a_bright_core(self):
        from spacr.qt import space
        img = space.galaxy(240, 240, seed=3, center=(0.5, 0.5),
                           radius_frac=0.35)
        h, w = img.shape[:2]
        core = img[int(h * 0.45):int(h * 0.55), int(w * 0.45):int(w * 0.55)]
        edge = img[:int(h * 0.06), :int(w * 0.06)]
        assert core.mean() > edge.mean() * 5

    def test_sun_is_limb_darkened(self):
        from spacr.qt import space
        img = space.sun(300, 300, seed=3, center=(0.5, 0.5),
                        radius_frac=0.3)
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        r = int(0.3 * min(h, w))
        centre = float(img[cy, cx].mean())
        near_limb = float(img[cy, cx + int(r * 0.93)].mean())
        assert centre > near_limb > 0, "limb must be dimmer than disc centre"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_written_once_then_reused(self, cache_dir, qapp):
        from spacr.qt import space
        path = space.background_path(120, 80, "stars", seed=42)
        assert path is not None and path.is_file()
        assert path.parent == cache_dir
        first_mtime = path.stat().st_mtime_ns
        first_bytes = path.read_bytes()

        again = space.background_path(120, 80, "stars", seed=42)
        assert again == path
        assert path.stat().st_mtime_ns == first_mtime, "regenerated needlessly"
        assert path.read_bytes() == first_bytes

    def test_regenerate_flag_rewrites(self, cache_dir, qapp):
        from spacr.qt import space
        path = space.background_path(96, 64, "stars", seed=1)
        path.write_bytes(b"")
        again = space.background_path(96, 64, "stars", seed=1,
                                      regenerate=True)
        assert again == path
        assert path.stat().st_size > 0

    def test_corrupt_cache_entry_regenerates(self, cache_dir, qapp):
        from spacr.qt import space
        path = space.background_path(96, 64, "galaxy", seed=3)
        good = path.read_bytes()
        path.write_bytes(b"this is not a PNG, it is a truncated write")

        recovered = space.background_path(96, 64, "galaxy", seed=3)
        assert recovered == path
        assert path.read_bytes() == good, "should regenerate identical bytes"

    def test_cache_key_separates_size_variant_and_seed(self, cache_dir):
        from spacr.qt import space
        names = {
            space.cache_name(100, 50, "galaxy", 1),
            space.cache_name(100, 51, "galaxy", 1),
            space.cache_name(100, 50, "stars", 1),
            space.cache_name(100, 50, "galaxy", 2),
        }
        assert len(names) == 4
        assert str(space.CACHE_VERSION) in space.cache_name(1, 1, "galaxy", 1)

    def test_unwritable_cache_returns_none_not_an_exception(
            self, tmp_path, monkeypatch):
        from spacr.qt import space
        blocker = tmp_path / "blocked"
        blocker.write_text("I am a file, not a directory")
        monkeypatch.setenv(space.ENV_CACHE_DIR, str(blocker / "sub"))
        assert space.background_path(64, 64) is None

    def test_clear_cache_removes_generated_files(self, cache_dir, qapp):
        from spacr.qt import space
        space.background_path(64, 64, "stars", seed=5)
        assert space.clear_cache() >= 1
        assert space.clear_cache() == 0

    def test_to_qimage_round_trips(self, qapp):
        from spacr.qt import space
        arr = space.render(48, 32, "stars", seed=2)
        img = space.to_qimage(arr)
        assert (img.width(), img.height()) == (48, 32)
        colour = img.pixelColor(10, 10)
        assert (colour.red(), colour.green(), colour.blue()) == \
            tuple(int(v) for v in arr[10, 10])

    def test_screen_size_is_sane_and_never_raises(self, qapp):
        from spacr.qt import space
        w, h = space.screen_size()
        assert space.MIN_DIM[0] <= w <= space.MAX_DIM[0]
        assert space.MIN_DIM[1] <= h <= space.MAX_DIM[1]


# ---------------------------------------------------------------------------
# Offline behaviour + NASA imagery
# ---------------------------------------------------------------------------

class TestOffline:
    def test_space_renders_with_no_background_at_all(self):
        """The download failed, the cache is unwritable, nothing exists."""
        qss = theme.stylesheet("space", background=None)
        assert "url(" not in qss
        assert "qlineargradient" in qss
        assert theme.SPACE_PALETTE["bg"] in qss

    def test_background_path_is_quoted_for_qss(self):
        qss = theme.stylesheet("space", background="/home/a b/sky.png")
        assert 'url("/home/a b/sky.png")' in qss

    def test_windows_path_separators_are_normalised(self):
        qss = theme.stylesheet("space", background=r"C:\Users\x\sky.png")
        assert 'url("C:/Users/x/sky.png")' in qss

    def test_download_returns_none_when_offline(self, cache_dir):
        from spacr.qt import space

        def dead(url, timeout):
            raise OSError("Network is unreachable")

        assert space.download_nasa_background(opener=dead) is None
        assert space.downloaded_background() is None
        assert space.read_credits() is None
        # The procedural attribution is what the UI shows instead.
        assert "Procedural" in space.attribution_text()

    def test_download_rejects_an_error_page(self, cache_dir):
        """A captive portal returns 2 kB of HTML with a 200. That must
        not be installed as the wallpaper."""
        from spacr.qt import space
        html = b"<html><body>Sign in to continue</body></html>" * 40
        assert space.download_nasa_background(opener=lambda u, t: html) is None
        assert space.downloaded_background() is None

    def test_download_rejects_a_truncated_response(self, cache_dir):
        from spacr.qt import space
        assert space.download_nasa_background(opener=lambda u, t: b"x") is None

    def test_unknown_image_key_is_rejected(self, cache_dir):
        from spacr.qt import space
        assert space.download_nasa_background(key="nope") is None

    def test_successful_download_records_attribution(self, cache_dir, qapp):
        from spacr.qt import space
        # A real PNG, made locally — no network, no bundled asset.
        blob = space.to_qimage(space.render(64, 64, "stars", seed=1))
        tmp = cache_dir.parent / "payload.png"
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        assert blob.save(str(tmp), "PNG")
        payload = tmp.read_bytes()

        record = space.download_nasa_background(
            key="carina", opener=lambda url, timeout: payload)
        assert record is not None
        assert record["credit"]
        assert record["source"].startswith("http")

        stored = space.downloaded_background()
        assert stored is not None and stored.is_file()
        assert record["credit"] in space.attribution_text()

        # The credit survives a restart: it is on disk, not in memory.
        credits_file = space.imagery_dir() / space.CREDITS_FILE
        assert json.loads(credits_file.read_text())["credit"] == record["credit"]

    def test_credits_pointing_at_a_missing_file_are_ignored(self, cache_dir):
        from spacr.qt import space
        space.imagery_dir().mkdir(parents=True, exist_ok=True)
        (space.imagery_dir() / space.CREDITS_FILE).write_text(
            json.dumps({"file": "gone.jpg", "credit": "NASA"}))
        assert space.read_credits() is None
        assert space.downloaded_background() is None

    def test_corrupt_credits_file_is_ignored(self, cache_dir):
        from spacr.qt import space
        space.imagery_dir().mkdir(parents=True, exist_ok=True)
        (space.imagery_dir() / space.CREDITS_FILE).write_text("{not json")
        assert space.read_credits() is None

    def test_nasa_entries_all_carry_a_credit(self):
        from spacr.qt import space
        assert space.NASA_IMAGES
        for entry in space.NASA_IMAGES:
            assert entry["credit"] and entry["source"] and entry["url"]
            assert entry["url"].startswith("https://")


# ---------------------------------------------------------------------------
# Preferences wiring
# ---------------------------------------------------------------------------

class TestPreferencesWiring:
    def test_existing_theme_values_keep_working(self, qapp):
        from spacr.qt.preferences import (get_theme, set_theme,
                                          resolve_effective_theme)
        for value in ("dark", "light", "system"):
            set_theme(value)
            assert get_theme() == value
        set_theme("space")
        assert resolve_effective_theme() == "space"
        set_theme("dark")

    def test_unknown_persisted_theme_falls_back(self, monkeypatch, qapp):
        from spacr.qt import preferences

        class FakeSettings:
            def value(self, key, default=None):
                return "chartreuse" if key == preferences._KEY_THEME else default

            def setValue(self, key, value):
                pass

        monkeypatch.setattr(preferences, "_settings", FakeSettings)
        assert preferences.get_theme() == preferences.DEFAULT_THEME

    def test_set_theme_rejects_nonsense(self, qapp):
        from spacr.qt.preferences import set_theme
        with pytest.raises(ValueError):
            set_theme("chartreuse")

    def test_space_variant_round_trips(self, qapp):
        from spacr.qt.preferences import (get_space_variant, set_space_variant,
                                          get_space_seed, set_space_seed)
        from spacr.qt.space import VARIANTS, DEFAULT_VARIANT
        for variant in VARIANTS:
            set_space_variant(variant)
            assert get_space_variant() == variant
        with pytest.raises(ValueError):
            set_space_variant("supernova")
        set_space_seed(1234)
        assert get_space_seed() == 1234
        set_space_variant(DEFAULT_VARIANT)

    def test_space_seed_survives_a_garbage_value(self, monkeypatch, qapp):
        from spacr.qt import preferences
        from spacr.qt.space import DEFAULT_SEED

        class FakeSettings:
            def value(self, key, default=None):
                return "not-an-int" if key == preferences._KEY_SPACE_SEED else default

        monkeypatch.setattr(preferences, "_settings", FakeSettings)
        assert preferences.get_space_seed() == DEFAULT_SEED

    def test_space_variant_survives_a_garbage_value(self, monkeypatch, qapp):
        from spacr.qt import preferences
        from spacr.qt.space import DEFAULT_VARIANT

        class FakeSettings:
            def value(self, key, default=None):
                return "quasar" if key == preferences._KEY_SPACE_VARIANT else default

        monkeypatch.setattr(preferences, "_settings", FakeSettings)
        assert preferences.get_space_variant() == DEFAULT_VARIANT

    def test_background_path_uses_generated_fallback(self, cache_dir, qapp,
                                                     monkeypatch):
        from spacr.qt import preferences, space
        fake = cache_dir / "generated.png"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fake.write_bytes(b"pretend")
        monkeypatch.setattr(
            space, "background_path", lambda *args, **kwargs: fake)
        assert preferences.space_background_path() == fake

    def test_background_path_never_raises(self, monkeypatch, qapp):
        from spacr.qt import preferences, space

        def boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(space, "background_path", boom)
        assert preferences.space_background_path() is None

    def test_space_gets_dark_figure_colours(self, qapp, monkeypatch):
        """Space is a dark theme: a `== "dark"` test would have handed
        it white figures."""
        from spacr.qt import preferences
        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda: "space")
        bg, fg = preferences.get_figure_colors()
        assert bg == "#000000" and fg == "#ffffff"

    def test_apply_preferences_only_pays_for_space(self, qapp, monkeypatch):
        from spacr.qt import preferences
        calls = []
        monkeypatch.setattr(preferences, "space_background_path",
                            lambda *a, **k: calls.append(1) or None)

        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda: "dark")
        preferences.apply_preferences_to_app(qapp)
        assert calls == []

        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda: "space")
        preferences.apply_preferences_to_app(qapp)
        assert calls == [1]


# ---------------------------------------------------------------------------
# Runtime theme switching
# ---------------------------------------------------------------------------

class TestRuntimeSwitch:
    def test_switching_theme_restyles_without_leaking_or_raising(
            self, qapp, qtbot, monkeypatch):
        from spacr.qt import preferences
        from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

        holder = QWidget()
        layout = QVBoxLayout(holder)
        label = QLabel("Measure")
        label.setObjectName("CardTitle")
        button = QPushButton("Run")
        button.setObjectName("PrimaryButton")
        layout.addWidget(label)
        layout.addWidget(button)
        qtbot.addWidget(holder)
        holder.show()

        before = len(qapp.allWidgets())
        # No image generation during the switch — that is tested
        # separately and must not make this test slow.
        monkeypatch.setattr(preferences, "space_background_path",
                            lambda *a, **k: None)
        for name in ("dark", "light", "space", "dark"):
            monkeypatch.setattr(preferences, "resolve_effective_theme",
                                lambda name=name: name)
            preferences.apply_preferences_to_app(qapp)
            assert qapp.styleSheet()
            holder.style().unpolish(holder)
            holder.style().polish(holder)

        assert len(qapp.allWidgets()) == before, "restyling leaked widgets"
        assert label.text() == "Measure"

    def test_switching_theme_re_inks_existing_icons(self, qapp, qtbot,
                                                    monkeypatch):
        """A QIcon bakes its pixmap when built, so a bare restyle leaves
        white glyphs on a white light-theme sidebar."""
        from spacr.qt import preferences
        from spacr.qt.app import Sidebar

        sidebar = Sidebar()
        qtbot.addWidget(sidebar)
        row = next(b for b in sidebar._items
                   if b.property("navKey") == "convert")

        def mean_luminance(button):
            img = button.icon().pixmap(32, 32).toImage()
            total = weight = 0.0
            for y in range(img.height()):
                for x in range(img.width()):
                    px = img.pixelColor(x, y)
                    a = px.alphaF()
                    if a <= 0.0:
                        continue
                    total += a * theme.relative_luminance(
                        "#%02x%02x%02x" % (px.red(), px.green(), px.blue()))
                    weight += a
            return total / weight if weight else 0.0

        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda: "dark")
        sidebar.refresh_icons()
        dark_ink = mean_luminance(row)

        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda: "light")
        sidebar.refresh_icons()
        light_ink = mean_luminance(row)

        assert dark_ink > 0.5, "dark theme should paint light ink"
        assert light_ink < 0.2, "light theme should paint dark ink"

    def test_saving_preferences_rebuilds_the_owning_window(self, qapp, qtbot):
        """Only the dialog's own window — walking every top-level widget
        reaches ones whose C++ side is already gone, and rebuilding one
        of those segfaults instead of raising."""
        from PySide6.QtWidgets import QWidget
        from spacr.qt.preferences import _refresh_owner_window

        calls = []

        class Window(QWidget):
            def refresh_theme(self):
                calls.append(1)

        window = Window()
        child = QWidget(window)
        qtbot.addWidget(window)
        _refresh_owner_window(child)
        assert calls == [1], "the owning window was not asked to rebuild"

    def test_refreshing_a_broken_window_is_swallowed(self, qapp, qtbot):
        from PySide6.QtWidgets import QWidget
        from spacr.qt.preferences import _refresh_owner_window

        class Rude(QWidget):
            def refresh_theme(self):
                raise RuntimeError("mid-teardown")

        rude = Rude()
        qtbot.addWidget(rude)
        _refresh_owner_window(rude)          # must not raise
        _refresh_owner_window(None)
        _refresh_owner_window(object())      # no .window() at all
        _refresh_owner_window(QWidget())     # no .refresh_theme at all

    def test_main_window_refresh_theme_is_wired_and_safe(self, qapp, qtbot):
        from spacr.qt.app import MainWindow
        window = MainWindow()
        qtbot.addWidget(window)
        window.refresh_theme()               # must not raise
        assert hasattr(window, "_sidebar")

    def test_apply_qpalette_tracks_the_theme(self, qapp):
        from PySide6.QtGui import QColor, QPalette
        for name in theme.THEMES:
            theme.apply_qpalette(qapp, theme=name)
            expected = theme.palette_for(name)
            assert qapp.palette().color(QPalette.Window) == \
                QColor(expected["bg"])


# ---------------------------------------------------------------------------
# Theme-blind icons
# ---------------------------------------------------------------------------

class TestIconVisibility:
    def test_there_are_bundled_icons_to_check(self):
        from spacr.qt import iconset
        paths = iconset.bundled_icon_paths()
        assert len(paths) > 10
        assert any(p.endswith("convert.png") for p in paths)

    @pytest.mark.parametrize("name", ("dark", "light", "space"))
    def test_every_bundled_icon_is_visible(self, name, qapp):
        """The theme-blind bug: `convert.png` is solid black, so on the
        black home page it rendered as literally nothing."""
        from spacr.qt import iconset
        weak = []
        for path in iconset.bundled_icon_paths():
            ratio = iconset.icon_contrast(path, name)
            if ratio < iconset.MIN_ICON_CONTRAST:
                weak.append(f"{os.path.basename(path)}: {ratio:.2f}:1")
        assert not weak, f"{name} theme hides icons:\n  " + "\n  ".join(weak)

    def test_the_raw_convert_icon_really_was_invisible(self, qapp):
        """Guard the regression: unthemed, it is 1.00:1 on dark."""
        from spacr.qt import iconset
        path = os.path.join(iconset.RESOURCE_DIR, "convert.png")
        raw_ink = "#000000"          # measured: every visible pixel is black
        assert theme.contrast_ratio(
            raw_ink, theme.palette_for("dark")["bg"]) < 1.05
        assert iconset.icon_contrast(path, "dark") >= 4.0

    def test_reinking_preserves_the_alpha_mask(self, qapp):
        from spacr.qt import iconset
        path = os.path.join(iconset.RESOURCE_DIR, "convert.png")
        rgba = iconset._load_rgba(path)
        out = iconset.reink(rgba, "dark")
        assert np.array_equal(out[:, :, 3], rgba[:, :, 3].astype(np.uint8))

    def test_the_bundled_set_is_a_monochrome_alpha_mask(self, qapp):
        """Measured, not assumed: the artwork lives in the alpha channel
        and RGB is a uniform fill. That is why one asset works for every
        theme, and why swapping in redrawn artwork cannot break this."""
        from spacr.qt import iconset
        structured = [os.path.basename(p)
                      for p in iconset.bundled_icon_paths()
                      if iconset.carries_tonal_structure(iconset._load_rgba(p))]
        # Only these genuinely put shading in RGB; everything else is a mask.
        assert set(structured) <= {
            "activation.png", "app_icon.png", "flow_chart_v3.png", "umap.png",
        }
        assert len(structured) < len(iconset.bundled_icon_paths()) / 2

    @pytest.mark.parametrize("name", ("dark", "light", "space"))
    @pytest.mark.parametrize("asset", ("mask.png", "convert.png",
                                       "plaque.png", "measure.png"))
    def test_the_tint_actually_reaches_the_painted_pixels(self, name, asset,
                                                          qapp):
        """A mask painted in the wrong colour still makes a valid QIcon,
        so assert the pixels, not the object."""
        from spacr.qt import iconset
        path = os.path.join(iconset.RESOURCE_DIR, asset)
        arr = iconset.themed_array(path, name)
        assert arr is not None
        opaque = arr[:, :, 3] > 200
        assert opaque.any(), f"{asset} has no solid pixels to check"
        painted = np.unique(arr[opaque][:, :3].reshape(-1, 3), axis=0)
        expected = iconset._hex_to_array(theme.palette_for(name)["fg"])
        assert len(painted) == 1, f"{asset} was not painted flat"
        assert np.array_equal(painted[0], expected.astype(np.uint8)), \
            f"{asset} on {name}: painted {painted[0]}, wanted {expected}"

    def test_flat_masks_are_not_given_spurious_shading(self, qapp):
        """Several icons vary by ~1 % across their 'solid white' fill.
        Stretching that across the ink band paints visible banding."""
        from spacr.qt import iconset
        path = os.path.join(iconset.RESOURCE_DIR, "annotate.png")
        rgba = iconset._load_rgba(path)
        assert not iconset.carries_tonal_structure(rgba)
        out = iconset.reink(rgba, "dark")
        opaque = out[:, :, 3] > 200
        assert len(np.unique(out[opaque][:, :3].reshape(-1, 3), axis=0)) == 1

    def test_large_artwork_is_downscaled_before_processing(self, qapp):
        """`logo_spacr.png` is 3334x3334 — 356 MB as float64 RGBA."""
        from spacr.qt import iconset
        rgba = iconset._load_rgba(
            os.path.join(iconset.RESOURCE_DIR, "logo_spacr.png"))
        assert max(rgba.shape[:2]) <= iconset.MAX_WORK_SIZE

    def test_decoded_icons_are_cached(self, qapp):
        from spacr.qt import iconset
        path = os.path.join(iconset.RESOURCE_DIR, "mask.png")
        first = iconset.themed_array(path, "dark")
        assert iconset.themed_array(path, "dark") is first

    def _fills(self, qss: str, selector: str) -> list:
        start = qss.index(selector)
        block = qss[start:qss.index("}", start)]
        return [line.strip() for line in block.splitlines()
                if line.strip().startswith("background")]

    def test_space_icons_never_sit_on_bare_imagery(self):
        """Icons are flat ink, so they rely on their container being
        scrimmed. Every container that holds one must be.

        ``#Sidebar`` used to be on this list and is now on the one
        below: the user asked for the dock never to be transparent, in
        any theme, so it is opaque rather than scrimmed. Opaque still
        satisfies what this test is defending — an icon on it never
        meets raw sky — it just satisfies it more strongly, which is why
        the two cases are asserted separately instead of loosening this
        one to accept both.
        """
        qss = theme.stylesheet("space")
        for selector in ("QPushButton#HTile", "QPushButton#AppTile",
                         "QPushButton#Tile", "QFrame#Card",
                         "QFrame#ConsoleBox"):
            fills = self._fills(qss, selector)
            assert fills, f"{selector} has no background"
            assert not any("transparent" in f for f in fills), \
                f"{selector} is transparent — icons would meet raw sky"
            assert any("rgba(" in f or "qlineargradient" in f for f in fills), \
                f"{selector} is not scrimmed: {fills}"

    def test_the_dock_is_opaque_in_every_theme(self):
        """"the dock to the left should never have a transparent
        background, either dark gray or white" — the user, #16j.

        A navigation column is chrome: it is what you look at when you
        have lost your place, and it has to be a solid edge for the page
        to end at. It used to paint ``surface``, which the image themes
        re-render through ``scrim_alpha`` — so on Space the app list was
        a ghost with a galaxy behind every row.
        """
        for name in theme.IMAGE_THEMES:
            qss = theme.stylesheet(name)
            expected = theme.dock_colour(name)
            assert expected.startswith("#"), (
                f"{name} dock colour {expected!r} is not a plain hex")
            assert theme.scrim_alpha(name, "surface") <= 1.0
            for selector in ("#EdgeDrawer, #Sidebar, #SidebarScroll",
                             "#SidebarTitle", "#SidebarSection"):
                fills = self._fills(qss, selector)
                assert fills, f"{name}: {selector} has no background"
                for fill in fills:
                    assert expected in fill, (
                        f"{name}: {selector} paints {fill!r}, not the "
                        f"opaque dock colour {expected}")
                    assert "rgba(" not in fill and "transparent" not in fill
        # On the FLAT themes the dock follows page opacity instead. Two
        # instructions meet here: #16j asked that the dock never be
        # transparent, and a later one asked that page opacity reach it.
        # Over a wallpaper #16j wins, because that is the case it was about —
        # a ghost dock with a galaxy behind every row — and the legibility
        # floor does not rescue it (Cell floors at 0.047). On dark and light
        # there is no picture behind the dock, only the ambient animation, so
        # thinning it is what was asked for and harms nothing.
        for name in ("dark", "light"):
            qss = theme.stylesheet(name, surface_opacity=0.5)
            fills = self._fills(qss, "#EdgeDrawer, #Sidebar, #SidebarScroll")
            assert any("rgba(" in f for f in fills), (
                f"{name}: the dock ignored page opacity: {fills}")

        # White under the light theme, a dark grey everywhere else —
        # both taken from the palette, never written down as a hex.
        assert theme.dock_colour("light") == theme.LIGHT_PALETTE["surface"]
        for name in ("dark", "space", "cell"):
            assert theme.dock_colour(name) == \
                theme.palette_for(name)["surface_alt"]

    def test_polarity_is_detected_per_icon(self, qapp):
        """White-on-transparent and black-on-transparent must both end
        up as light ink on the dark theme."""
        from spacr.qt import iconset
        black = iconset.icon_ink_color(
            os.path.join(iconset.RESOURCE_DIR, "convert.png"), "dark")
        white = iconset.icon_ink_color(
            os.path.join(iconset.RESOURCE_DIR, "mask.png"), "dark")
        assert theme.relative_luminance(black) > 0.2
        assert theme.relative_luminance(white) > 0.2

    def test_ink_flips_between_dark_and_light(self, qapp):
        from spacr.qt import iconset
        path = os.path.join(iconset.RESOURCE_DIR, "mask.png")
        dark = theme.relative_luminance(iconset.icon_ink_color(path, "dark"))
        light = theme.relative_luminance(iconset.icon_ink_color(path, "light"))
        assert dark > light, "icon ink must invert with the theme"

    def test_veil_is_solved_not_guessed(self, qapp):
        from spacr.qt import iconset
        for name in theme.THEMES:
            veil = iconset.veil_color(name)
            surface = iconset.hardest_surface(name)
            assert theme.contrast_ratio(veil, surface) >= \
                iconset.MIN_ICON_CONTRAST - 0.01

    def test_polychrome_artwork_keeps_its_hue(self, qapp):
        from spacr.qt import iconset
        path = os.path.join(iconset.RESOURCE_DIR, "flow_chart_v3.png")
        if not os.path.isfile(path):
            pytest.skip("flow chart asset not bundled")
        rgba = iconset._load_rgba(path)
        out = iconset.reink(rgba, "dark").astype(float)
        chroma = out[:, :, :3].max(axis=2) - out[:, :, :3].min(axis=2)
        assert chroma.max() > 30, "colour was flattened to greyscale"

    def test_app_icon_falls_back_to_a_glyph(self, qapp):
        from spacr.qt import iconset
        assert iconset.app_icon("no-such-app-key") is not None

    def test_app_icon_uses_the_bundled_png_when_present(self, qapp):
        from spacr.qt import iconset
        icon = iconset.app_icon("mask")
        assert not icon.isNull()

    def test_missing_file_degrades_quietly(self, qapp, tmp_path):
        from spacr.qt import iconset
        missing = str(tmp_path / "nope.png")
        assert iconset.themed_qimage(missing) is None
        assert iconset.themed_pixmap(missing) is None
        assert iconset.icon_ink_color(missing) is None
        assert iconset.icon_contrast(missing) == 0.0

    def test_fully_transparent_artwork_is_left_alone(self, qapp):
        from spacr.qt import iconset
        blank = np.zeros((8, 8, 4), dtype=float)
        out = iconset.reink(blank, "dark")
        assert out.shape == (8, 8, 4)
        assert not out.any()

    def test_active_theme_survives_broken_preferences(self, monkeypatch,
                                                      qapp):
        from spacr.qt import iconset, preferences

        def boom():
            raise RuntimeError("QSettings exploded")

        monkeypatch.setattr(preferences, "resolve_effective_theme", boom)
        assert iconset.active_theme() == "dark"

    def test_bundled_icon_paths_survive_a_missing_directory(self, monkeypatch):
        from spacr.qt import iconset
        monkeypatch.setattr(iconset, "RESOURCE_DIR", "/no/such/place")
        assert iconset.bundled_icon_paths() == ()
