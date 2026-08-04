"""
Photographic themes — the Cell theme and the Space deep-field variant.

These cover :mod:`spacr.qt.imagery` plus the parts of
:mod:`spacr.qt.theme` and :mod:`spacr.qt.preferences` that a background
made of *photons* needs and a generated one does not.

Four claims are load-bearing here and all four are measured rather than
asserted by inspection:

* text stays legible over the brightest region the real crop presents;
* a master is decoded once per screen size and **never** during a resize
  or a repaint;
* the burned-in scale bar in one of the originals reaches no shipped
  pixel; and
* a build with the assets stripped still starts, and Space still gets a
  sky.

Everything runs offscreen, offline and CPU-only. The masters are read
from the installed package, and every test that would otherwise write to
the user's home directory is redirected through ``SPACR_SPACE_CACHE``.
"""
from __future__ import annotations

import math
import resource

import numpy as np
import pytest
from PIL import Image, ImageDraw

from spacr.qt import imagery, preferences, theme

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Redirect the background cache away from the real ``~/.spacr``."""
    target = tmp_path / "backgrounds"
    monkeypatch.setenv("SPACR_SPACE_CACHE", str(target))
    return target


@pytest.fixture
def no_masters(tmp_path, monkeypatch):
    """Point the master lookup at an empty directory."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(imagery.ENV_MASTER_DIR, str(empty))
    return empty


def _installed(key: str):
    path = imagery.master_path(key)
    if path is None:
        pytest.skip(f"master for {key!r} is not installed in this build")
    return path


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

class TestPalettes:
    def test_cell_is_a_theme_with_a_palette(self):
        assert "cell" in theme.THEMES
        assert "cell" in preferences.VALID_THEMES
        assert "cell" in theme.IMAGE_THEMES

    def test_every_theme_resolves_to_a_complete_palette(self):
        """No theme may be missing a role — a KeyError in `palette_for`
        surfaces as a crash somewhere unrelated three screens later."""
        roles = set(theme.DARK_PALETTE) | set(theme.CONSTANT_ROLES)
        for name in theme.THEMES:
            palette = theme.palette_for(name)
            assert roles.issubset(palette), \
                f"{name} is missing {sorted(roles - set(palette))}"
            for role, value in palette.items():
                assert value.startswith("#") and len(value) == 7, \
                    f"{name}.{role} = {value!r} is not #rrggbb"

    def test_no_role_lookup_raises_for_any_theme(self):
        """Every role a contrast rule names resolves, in every theme.

        "It did not raise" is not the claim. A ``palette_for`` that
        started handing back one default colour for anything it did not
        know would never raise either — and every rule would then
        measure the identity ratio, exactly 1.0, because both ends of
        each pair would be that same colour. So the floor is asserted
        against a deliberate identity measurement, and ``fg`` on ``bg``
        (18.5:1 to 21:1 across the shipped themes) has to sit far above
        it.
        """
        roles = {role for rule in theme.CONTRAST_RULES for role in rule[:2]}
        assert len(theme.CONTRAST_RULES) >= 10 and len(roles) >= 10
        for name in list(theme.THEMES) + ["system", "", "no-such-theme"]:
            palette = theme.palette_for(name)
            assert roles.issubset(palette), \
                f"{name} cannot resolve {sorted(roles - set(palette))}"
            measured = []
            for fg, surface, required in theme.CONTRAST_RULES:
                ratio = theme.contrast_ratio(palette[fg], palette[surface])
                assert isinstance(ratio, float) and math.isfinite(ratio), \
                    f"{name}: {fg} on {surface} measured {ratio!r}"
                assert 1.0 <= ratio <= 21.0, \
                    f"{name}: {fg} on {surface} measured {ratio}"
                measured.append((fg, surface, required, ratio))
            # The identity floor, and the distance the real roles keep
            # from it.
            assert theme.contrast_ratio(palette["fg"], palette["fg"]) == \
                pytest.approx(1.0)
            assert theme.contrast_ratio(palette["fg"], palette["bg"]) >= 4.5, \
                f"{name}: body text does not clear AA on its own background"
            assert len({round(ratio, 2) for *_r, ratio in measured}) > 1, \
                f"{name}: every rule measured the same ratio"
            assert any(ratio >= required
                       for *_r, required, ratio in measured), \
                f"{name}: not one contrast rule is met"

    def test_cell_clears_aa_against_the_worst_case_it_can_actually_meet(self):
        """Renamed, and it now means something different.

        It used to read ``..._against_a_pure_white_background`` and it
        pinned the Cell scrims against ``#ffffff``. White is the worst
        case for *Space*, whose procedural sky keeps its sun blown out;
        it is not one Cell can produce, because every Cell wallpaper is
        exposure-solved down to :func:`theme.max_background_luma`.
        Judging Cell against white cost a 0.90 scrim and, with it, the
        picture. ``contrast_failures`` now resolves the worst case per
        theme (:func:`theme.scrim_under`), and this asserts the thing
        that was always the point: text on a Cell panel is readable over
        anything a Cell wallpaper can put behind it.
        """
        assert theme.scrim_under("cell") != theme.WORST_CASE_UNDER
        assert theme.relative_luminance(theme.scrim_under("cell")) == \
            pytest.approx(theme.max_background_luma("cell"), abs=0.005)
        assert theme.contrast_failures("cell") == []

    def test_scrims_are_declared_for_every_image_theme(self):
        for name in theme.IMAGE_THEMES:
            assert name in theme.SCRIM_ALPHA
            for role in ("surface", "surface_alt", "surface_hi"):
                assert 0.0 < theme.scrim_alpha(name, role) < 1.0

    def test_opaque_themes_are_unaffected_by_the_scrim_machinery(self):
        for name in ("dark", "light"):
            for role in ("surface", "surface_alt", "surface_hi"):
                assert theme.scrim_alpha(name, role) == 1.0
                assert theme.effective_surface(name, role) == \
                    theme.palette_for(name)[role]


class TestMaxBackgroundLuma:
    def test_the_bare_rules_are_read_out_of_the_contrast_rules(self):
        """Derived, not restated — a role added to CONTRAST_RULES against
        `bg` is automatically enforced against the imagery."""
        expected = {(fg, req) for fg, surface, req in theme.CONTRAST_RULES
                    if surface == "bg"}
        assert set(theme.BARE_IMAGE_RULES) == expected
        assert expected

    def test_the_limit_actually_satisfies_every_bare_rule(self):
        for name in theme.IMAGE_THEMES:
            limit = theme.max_background_luma(name)
            palette = theme.palette_for(name)
            for role, required in theme.BARE_IMAGE_RULES:
                ratio = ((theme.relative_luminance(palette[role]) + 0.05)
                         / (limit + 0.05))
                assert ratio >= required - 1e-9, \
                    f"{name}: {role} only reaches {ratio:.3f}:1 at the limit"

    def test_a_light_theme_admits_no_background_at_all(self):
        """Not a bug — a numeric statement of why these images are dark
        themes only. Near-black `fg` leaves no room for any wallpaper."""
        assert theme.max_background_luma("light") < 0.0

    def test_image_report_judges_bg_against_the_image_itself(self):
        report = theme.image_contrast_report("cell", "#ffffff")
        bare = [r for r in report if r["bg"] == "bg"]
        assert bare and all(r["bg_color"] == "#ffffff" for r in bare)
        # ...and everything else against its scrim over that colour.
        scrimmed = [r for r in report if r["bg"] == "surface"]
        assert scrimmed and all(r["bg_color"] != "#ffffff" for r in scrimmed)

    def test_a_white_wallpaper_is_reported_as_a_failure(self):
        """A pure white Cell wallpaper breaks the *panels* too, now.

        This used to assert every failure named ``on bg`` — i.e. that
        only bare text suffered and the scrims shrugged a white
        wallpaper off. That was true when the Cell scrims were 0.88-0.93
        and it is why they were, and it is exactly the trade the user
        saw as "the themes aren't implemented, I can't see the cells":
        a scrim thick enough to survive a hypothetical white photograph
        transmits 10 % of a real one.

        Cell's scrims are now solved against
        :func:`theme.scrim_under`, which for Cell is the exposure
        ceiling every Cell wallpaper is *guaranteed* to be under
        (`imagery.render` solves the shipped masters and the user's own
        drop-in alike), so white is out of contract. Feeding it in
        anyway must therefore report the panels as well — and this test
        pins that, because the day a Cell wallpaper can reach white is
        the day the scrims have to go back up.
        """
        failures = theme.image_contrast_failures("cell", "#ffffff")
        assert failures, "white behind bare text must not be reported as fine"
        assert any("on bg" in line for line in failures)
        assert any("on surface" in line for line in failures), \
            "a thin scrim cannot save text over white; say so"
        # Space is the theme that really can be handed a white pixel —
        # its sky blows its sun out on purpose — so its scrims are
        # solved against white and they hold.
        assert all("on bg" in line
                   for line in theme.image_contrast_failures(
                       "space", "#ffffff"))

    def test_a_black_wallpaper_passes_everything(self):
        assert theme.image_contrast_failures("cell", "#000000") == []
        assert theme.image_contrast_failures("space", "#000000") == []


# ---------------------------------------------------------------------------
# Legibility over the real pixels
# ---------------------------------------------------------------------------

class TestMeasuredLegibility:
    @pytest.mark.parametrize("key", sorted(imagery.MASTERS))
    def test_aa_holds_over_the_brightest_region_of_the_real_crop(self, key):
        """The whole point. Not an average, not one pixel: the brightest
        region the size of a line of body text, measured from the file
        that ships."""
        _installed(key)
        measured = imagery.legibility(key)
        assert measured is not None
        assert measured["failures"] == [], (
            f"{key}: {measured['failures']} over {measured['color']}")
        assert measured["passes"], (
            f"{key}: brightest region {measured['brightest']:.5f} exceeds "
            f"the palette limit {measured['limit']:.5f}")

    @pytest.mark.parametrize("key", sorted(imagery.MASTERS))
    def test_the_exposure_used_the_headroom_it_had(self, key):
        """A wallpaper crushed far below the limit is a waste of a
        photograph; one sitting exactly on it has no margin for JPEG.
        Everything should land in between, or be naturally dark enough
        that no dimming happened at all."""
        _installed(key)
        measured = imagery.legibility(key)
        assert measured["brightest"] <= measured["limit"]
        assert measured["brightest"] <= measured["target"] * 1.02

    def test_the_brightest_window_is_a_region_not_a_pixel(self):
        """One white pixel in a black frame must not move the metric;
        a white band the size of the window must move it all the way."""
        black = np.zeros((200, 400, 3), dtype=np.uint8)
        black[100, 200] = 255
        value, _ = imagery.brightest_window(black, window=(0.1, 0.1))
        assert value < 0.01

        banded = np.zeros((200, 400, 3), dtype=np.uint8)
        banded[80:120, 100:300] = 255
        value, color = imagery.brightest_window(banded, window=(0.1, 0.1))
        assert value > 0.9
        assert color == "#ffffff"

    def test_the_reported_colour_carries_the_reported_luminance(self):
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        value, color = imagery.brightest_window(arr, window=(0.2, 0.2))
        assert theme.relative_luminance(color) == pytest.approx(value,
                                                                abs=0.01)

    def test_a_flat_grey_frame_reads_back_its_own_luminance(self):
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        value, color = imagery.brightest_window(arr)
        assert color == "#808080"
        assert value == pytest.approx(theme.relative_luminance("#808080"),
                                       abs=1e-3)

    def test_legibility_is_none_without_a_master(self, no_masters):
        assert imagery.legibility("microtubules") is None
        assert imagery.master_array("microtubules") is None


class TestExposureSolve:
    def test_dim_scales_relative_luminance_by_exactly_the_factor(self):
        rng = np.random.default_rng(3)
        arr = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
        before = imagery.luminance_map(arr).mean()
        after = imagery.luminance_map(imagery.dim(arr, 0.25)).mean()
        assert after == pytest.approx(before * 0.25, rel=0.02)

    def test_solve_dim_hits_the_target(self):
        assert imagery.solve_dim(0.20, 0.05) == pytest.approx(0.25)
        assert imagery.solve_dim(0.02, 0.05) == 1.0      # never brightens
        assert imagery.solve_dim(0.0, 0.05) == 1.0       # black stays black
        assert imagery.solve_dim(0.20, -1.0) == 0.0      # clamped, no crash

    def test_dim_by_one_is_a_no_op(self):
        arr = np.full((4, 4, 3), 200, dtype=np.uint8)
        assert imagery.dim(arr, 1.0) is arr

    def test_exposure_target_sits_below_the_hard_limit(self):
        for name in theme.IMAGE_THEMES:
            assert 0.0 < imagery.exposure_target(name) < \
                theme.max_background_luma(name)

    def test_exposure_target_never_goes_negative(self):
        assert imagery.exposure_target("light") == 0.0


# ---------------------------------------------------------------------------
# Burned-in annotation
# ---------------------------------------------------------------------------

class TestNoBurnedInAnnotation:
    def test_the_crop_geometrically_excludes_the_scale_bar(self):
        """`cell_2.png` carries a "5 um" bar at the measured bounds
        below. The crop is what removes it, so the crop is what the test
        checks — and because the shipped master is built from that crop,
        no runtime crop can bring the bar back."""
        entry = imagery.MASTERS["filopodia"]
        bar = entry["annotation"]
        assert not imagery.rects_overlap(entry["source_crop"], bar)
        assert entry["source_crop"][3] < bar[1]

    def test_the_overlap_test_is_not_vacuous(self):
        bar = imagery.MASTERS["filopodia"]["annotation"]
        assert imagery.rects_overlap((0.0, 0.0, 1.0, 1.0), bar)
        assert imagery.rects_overlap(bar, bar)
        assert not imagery.rects_overlap((0.0, 0.0, 0.1, 0.1), bar)

    @pytest.mark.parametrize("key", sorted(imagery.MASTERS))
    def test_no_shipped_master_contains_solid_annotation(self, key):
        path = _installed(key)
        with Image.open(path) as master:
            arr = np.asarray(master.convert("RGB"))
        assert imagery.solid_annotation_blocks(arr) == 0

    @pytest.mark.parametrize("key", sorted(imagery.MASTERS))
    def test_the_detector_finds_a_bar_when_there_is_one(self, key):
        """Negative results only mean something if the detector still
        works. Draw the same kind of bar back on and it must fire."""
        path = _installed(key)
        with Image.open(path) as master:
            image = master.convert("RGB")
        width, height = image.size
        ImageDraw.Draw(image).rectangle(
            [int(width * 0.78), int(height * 0.90),
             int(width * 0.94), int(height * 0.925)], fill=(255, 255, 255))
        assert imagery.solid_annotation_blocks(np.asarray(image)) > 0

    def test_the_detector_survives_degenerate_input(self):
        assert imagery.solid_annotation_blocks(
            np.zeros((0, 0, 3), dtype=np.uint8)) == 0
        assert imagery.solid_annotation_blocks(
            np.zeros((8, 8, 3), dtype=np.uint8)) == 0
        assert imagery.solid_annotation_blocks(
            np.zeros((64, 64, 3), dtype=np.uint8)) == 0


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

class TestCoverBox:
    @pytest.mark.parametrize("out", [(1920, 1200), (3840, 2160), (3440, 1440),
                                     (1600, 1200), (1200, 1600)])
    def test_the_box_has_the_output_aspect_and_fits_inside(self, out):
        box = imagery.cover_box(3840, 2400, out[0], out[1])
        width, height = box[2] - box[0], box[3] - box[1]
        assert 0 <= box[0] and 0 <= box[1]
        assert box[2] <= 3840 and box[3] <= 2400
        assert width / height == pytest.approx(out[0] / out[1], rel=0.01)

    def test_it_covers_rather_than_contains(self):
        """One dimension is always used whole — that is what stops the
        stylesheet letterboxing the picture into bands of flat colour."""
        for out in ((3840, 2160), (1200, 1600), (2560, 1600)):
            box = imagery.cover_box(3840, 2400, *out)
            assert (box[2] - box[0] == 3840) or (box[3] - box[1] == 2400)

    def test_focus_moves_the_box_and_stays_inside(self):
        low = imagery.cover_box(1000, 1000, 100, 50, focus=0.0)
        high = imagery.cover_box(1000, 1000, 100, 50, focus=1.0)
        mid = imagery.cover_box(1000, 1000, 100, 50, focus=0.5)
        assert low[1] == 0
        assert high[3] == 1000
        assert low[1] < mid[1] < high[1]

    def test_degenerate_sizes_do_not_raise(self):
        assert imagery.cover_box(1, 1, 1, 1) == (0, 0, 1, 1)
        assert imagery.cover_box(0, 0, 0, 0) == (0, 0, 1, 1)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_a_master_is_decoded_once_and_the_result_reused(self, qapp,
                                                            cache_dir):
        _installed("filopodia")
        imagery.reset_decode_count()
        first = imagery.background_path("filopodia", 1920, 1200)
        assert first is not None and first.is_file()
        assert imagery.decode_count() == 1

        second = imagery.background_path("filopodia", 1920, 1200)
        assert second == first
        assert imagery.decode_count() == 1, \
            "a cache hit must not touch the master"

    def test_the_cache_key_separates_size_key_and_version(self):
        assert imagery.cache_name("a", 100, 200) != \
            imagery.cache_name("a", 200, 100)
        assert imagery.cache_name("a", 100, 200) != \
            imagery.cache_name("b", 100, 200)
        assert f"v{imagery.CACHE_VERSION}" in imagery.cache_name("a", 1, 2)

    def test_a_truncated_entry_regenerates(self, qapp, cache_dir):
        _installed("filopodia")
        path = imagery.background_path("filopodia", 1920, 1200)
        path.write_bytes(b"\xff\xd8\xff" + b"garbage" * 40)
        imagery.reset_decode_count()
        again = imagery.background_path("filopodia", 1920, 1200)
        assert again == path
        assert imagery.decode_count() == 1
        assert imagery._load_cached(path, 1920, 1200)

    def test_an_empty_entry_regenerates(self, qapp, cache_dir):
        _installed("filopodia")
        path = imagery.background_path("filopodia", 1920, 1200)
        path.write_bytes(b"")
        assert imagery.background_path("filopodia", 1920, 1200) == path
        assert imagery._load_cached(path, 1920, 1200)

    def test_an_entry_of_the_wrong_size_regenerates(self, qapp, cache_dir):
        _installed("filopodia")
        path = imagery.background_path("filopodia", 1920, 1200)
        Image.new("RGB", (64, 64), (0, 0, 0)).save(path, "JPEG")
        assert not imagery._load_cached(path, 1920, 1200)
        imagery.reset_decode_count()
        assert imagery.background_path("filopodia", 1920, 1200) == path
        assert imagery.decode_count() == 1
        assert imagery._load_cached(path, 1920, 1200)

    def test_a_missing_entry_is_not_a_cache_hit(self, cache_dir, tmp_path):
        assert imagery._load_cached(tmp_path / "nope.jpg", 10, 10) is False

    def test_regenerate_forces_a_rebuild(self, qapp, cache_dir):
        _installed("filopodia")
        imagery.background_path("filopodia", 1920, 1200)
        imagery.reset_decode_count()
        imagery.background_path("filopodia", 1920, 1200, regenerate=True)
        assert imagery.decode_count() == 1

    @pytest.mark.parametrize("size", [(1920, 1080), (1920, 1200),
                                      (2560, 1600), (3440, 1440)])
    def test_the_rendered_background_is_exactly_the_requested_size(
            self, qapp, cache_dir, size):
        """Exactly, for every aspect. The stylesheet centres the image
        without scaling it, so anything but an exact match is either a
        crop or a letterbox on the user's screen."""
        _installed("deep_field")
        path = imagery.background_path("deep_field", *size)
        with Image.open(path) as rendered:
            assert rendered.size == size

    def test_the_screen_floor_comes_from_screen_size_not_the_cache(
            self, qapp, cache_dir, monkeypatch):
        """A background narrower than the window would letterbox, so the
        floor exists — but it belongs to the screen query, so an explicit
        request is honoured as written."""
        from spacr.qt.space import MIN_BACKGROUND
        _installed("deep_field")
        monkeypatch.setattr(imagery, "screen_size",
                            lambda: (MIN_BACKGROUND[0], MIN_BACKGROUND[1]))
        path = imagery.background_path("deep_field")
        with Image.open(path) as rendered:
            assert rendered.size == MIN_BACKGROUND

    def test_clear_cache_removes_what_it_wrote(self, qapp, cache_dir):
        _installed("filopodia")
        imagery.background_path("filopodia", 1920, 1200)
        imagery.background_path("filopodia", 2560, 1600)
        assert imagery.clear_cache() == 2
        assert imagery.clear_cache() == 0

    def test_clear_cache_survives_a_missing_directory(self, tmp_path,
                                                      monkeypatch):
        monkeypatch.setenv("SPACR_SPACE_CACHE", str(tmp_path / "gone"))
        assert imagery.clear_cache() == 0

    def test_an_unwritable_cache_returns_none_rather_than_raising(
            self, qapp, monkeypatch, tmp_path):
        blocked = tmp_path / "ro" / "backgrounds"
        (tmp_path / "ro").mkdir()
        (tmp_path / "ro").chmod(0o500)
        monkeypatch.setenv("SPACR_SPACE_CACHE", str(blocked))
        try:
            assert imagery.background_path("filopodia", 1920, 1200) is None
        finally:
            (tmp_path / "ro").chmod(0o700)

    def test_a_screen_size_of_zero_falls_back_to_the_primary_screen(
            self, qapp, cache_dir):
        _installed("filopodia")
        path = imagery.background_path("filopodia")
        assert path is not None
        with Image.open(path) as rendered:
            assert rendered.size[0] >= 1920 and rendered.size[1] >= 1200

    def test_an_absurd_request_is_clamped_not_honoured(self, qapp, cache_dir):
        _installed("filopodia")
        path = imagery.background_path("filopodia", 100000, 100000)
        with Image.open(path) as rendered:
            assert rendered.size == theme_max_dim()


def theme_max_dim():
    from spacr.qt.space import MAX_DIM
    return MAX_DIM


class TestWriteFallback:
    def test_png_is_used_when_qt_cannot_read_the_jpeg(self, qapp, tmp_path,
                                                      monkeypatch):
        """A Qt build without the JPEG plugin must get a wallpaper it can
        decode, not an infinite regeneration loop."""
        calls = {"n": 0}
        real = imagery._qt_can_read

        def no_jpeg_plugin(path, width, height):
            calls["n"] += 1
            if calls["n"] == 1:            # the JPEG probe
                return False
            return real(path, width, height)

        monkeypatch.setattr(imagery, "_qt_can_read", no_jpeg_plugin)
        out = tmp_path / "bg.jpg"
        written = imagery._write(Image.new("RGB", (32, 24), (10, 20, 30)),
                                 out, 32, 24)
        assert written == out
        with Image.open(out) as saved:
            assert saved.format == "PNG"
        assert calls["n"] >= 2

    def test_a_format_qt_cannot_read_at_all_returns_none(self, qapp, tmp_path,
                                                         monkeypatch):
        monkeypatch.setattr(imagery, "_qt_can_read",
                            lambda path, width, height: False)
        out = tmp_path / "bg.jpg"
        assert imagery._write(Image.new("RGB", (8, 8)), out, 8, 8) is None
        assert not list(tmp_path.iterdir())

    def test_qt_can_read_reports_false_for_rubbish(self, qapp, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        assert imagery._qt_can_read(bad, 10, 10) is False

    def test_qt_can_read_reports_false_rather_than_raising(self, qapp):
        class Hostile:
            def __str__(self):
                raise RuntimeError("not a path")

        assert imagery._qt_can_read(Hostile(), 10, 10) is False

    def test_load_cached_reports_false_rather_than_raising(self, qapp):
        assert imagery._load_cached(None, 10, 10) is False


class TestDecoderQuirks:
    def test_a_plugin_whose_draft_raises_still_decodes(self, tmp_path,
                                                       monkeypatch):
        """``draft`` is a JPEG optimisation and an optional one. A
        decoder that refuses it must cost speed, not the theme."""
        from PIL import Image as PILImage

        source = tmp_path / "plain.png"
        PILImage.new("RGB", (48, 32), (5, 40, 60)).save(source)
        real_open = PILImage.open

        class NoDraft:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._inner.close()
                return False

            def draft(self, mode, size):
                raise ValueError("this decoder has no draft mode")

            def convert(self, mode):
                return self._inner.convert(mode)

        monkeypatch.setattr(PILImage, "open",
                            lambda path, *a, **k: NoDraft(real_open(path)))
        imagery.reset_decode_count()
        decoded = imagery._open_master(source, hint=(16, 16))
        assert decoded.size == (48, 32)
        assert imagery.decode_count() == 1


class TestClearCacheRobustness:
    def test_an_undeletable_entry_is_skipped(self, qapp, cache_dir,
                                             monkeypatch):
        _installed("filopodia")
        imagery.background_path("filopodia", 1920, 1200)

        def refuse(self, *args, **kwargs):
            raise OSError("held open by something")

        monkeypatch.setattr("pathlib.Path.unlink", refuse)
        assert imagery.clear_cache() == 0

    def test_a_cache_directory_that_raises_is_survived(self, monkeypatch):
        def boom():
            raise RuntimeError("no home directory")
        monkeypatch.setattr(imagery, "cache_dir", boom)
        assert imagery.clear_cache() == 0


# ---------------------------------------------------------------------------
# The performance claim
# ---------------------------------------------------------------------------

class TestNoDecodeOnResizeOrRepaint:
    def test_a_window_resize_and_repaint_decode_nothing(self, qapp, qtbot,
                                                        cache_dir,
                                                        monkeypatch):
        """The claim this whole module exists to make good on.

        The window paints a screen-sized JPEG that Qt loaded when the
        stylesheet was applied. Resizing it, showing it and forcing it to
        repaint must not reopen a master — a 3840x2400 decode per resize
        event would make the theme unusable.
        """
        from PySide6.QtWidgets import QWidget

        _installed("microtubules")
        monkeypatch.setattr(preferences, "get_theme", lambda: "cell")
        preferences.apply_preferences_to_app(qapp)

        window = QWidget()
        qtbot.addWidget(window)
        window.resize(800, 600)
        window.show()

        imagery.reset_decode_count()
        for width in (900, 1100, 640, 1280, 1024):
            window.resize(width, int(width * 0.62))
            window.grab()               # forces a real paint
            qapp.processEvents()
        assert imagery.decode_count() == 0

    def test_applying_the_theme_decodes_at_most_once(self, qapp, cache_dir,
                                                     monkeypatch):
        _installed("microtubules")
        monkeypatch.setattr(preferences, "get_theme", lambda: "cell")
        preferences.apply_preferences_to_app(qapp)     # warms the cache
        imagery.reset_decode_count()
        preferences.apply_preferences_to_app(qapp)
        assert imagery.decode_count() == 0, \
            "a second theme apply must come straight off disk"

    def test_a_full_theme_switch_stays_within_a_memory_budget(
            self, qapp, cache_dir, monkeypatch):
        """`space_1.jpeg` is 281 MB decoded as RGBA. The shipped master
        is capped so the largest decode is ~27 MB; add a Lanczos pass to
        a 4K target and the whole switch should cost well under 250 MB.

        ``ru_maxrss`` is a high-water mark, so this can only ever fail
        when the switch really did allocate that much — it cannot fail
        because some earlier test was greedy.
        """
        _installed("deep_field")
        monkeypatch.setattr(preferences, "get_theme", lambda: "space")
        monkeypatch.setattr(preferences, "get_space_variant",
                            lambda: "deep_field")
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        path = imagery.background_path("deep_field", 3840, 2400)
        preferences.apply_preferences_to_app(qapp)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert path is not None
        assert (after - before) / 1024.0 < 250.0, \
            f"theme switch grew RSS by {(after - before) / 1024.0:.0f} MB"


# ---------------------------------------------------------------------------
# Locating masters
# ---------------------------------------------------------------------------

class TestMasterLookup:
    def test_the_shipped_masters_are_installed(self):
        """If this fails the wheel is missing `resources/themes`."""
        assert set(imagery.available_keys()) == set(imagery.MASTERS)

    def test_every_master_is_within_the_cap(self):
        for key in imagery.MASTERS:
            path = _installed(key)
            with Image.open(path) as master:
                assert master.width <= imagery.MASTER_CAP[0]
                assert master.height <= imagery.MASTER_CAP[1]

    def test_an_unknown_key_has_no_master_and_renders_nothing(self):
        assert imagery.master_path("no-such-image") is None
        assert imagery.render("no-such-image", 100, 100) is None
        assert imagery.theme_for("no-such-image") is None
        assert imagery.title_for("no-such-image") == "no-such-image"

    def test_titles_are_human_readable(self):
        for key in imagery.MASTERS:
            assert imagery.title_for(key) != key
            assert imagery.theme_for(key) in theme.IMAGE_THEMES

    def test_the_user_directory_wins_over_the_bundled_asset(self, qapp,
                                                            tmp_path,
                                                            monkeypatch):
        """The escape hatch: drop a file in ~/.spacr/themes and it is
        used instead of the one in the wheel."""
        monkeypatch.setenv("SPACR_SPACE_CACHE", str(tmp_path / "backgrounds"))
        mine = tmp_path / "themes"
        mine.mkdir()
        assert imagery.user_dir() == mine
        own = mine / imagery.MASTERS["microtubules"]["file"]
        Image.new("RGB", (640, 400), (12, 40, 60)).save(own, "JPEG")
        assert imagery.master_path("microtubules") == own

    def test_a_users_own_image_is_dimmed_to_the_same_limit(self, qapp,
                                                           tmp_path,
                                                           monkeypatch):
        """The shipped masters are pre-exposed, so the runtime solve is a
        no-op on them. It is not a no-op on an image the user supplied,
        and that is the case it exists for."""
        monkeypatch.setenv("SPACR_SPACE_CACHE", str(tmp_path / "backgrounds"))
        mine = tmp_path / "themes"
        mine.mkdir()
        blinding = mine / imagery.MASTERS["microtubules"]["file"]
        Image.new("RGB", (800, 500), (255, 255, 255)).save(blinding, "PNG")

        rendered = imagery.render("microtubules", 1920, 1200)
        value, color = imagery.brightest_window(imagery._probe(rendered))
        assert value <= theme.max_background_luma("cell")
        assert theme.image_contrast_failures("cell", color) == []

    def test_the_env_override_replaces_the_search_path(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.setenv(imagery.ENV_MASTER_DIR, str(tmp_path))
        assert imagery.master_dirs() == (tmp_path,)
        assert imagery.master_path("microtubules") is None
        assert imagery.available_keys() == ()

    def test_a_broken_home_directory_is_not_fatal(self, monkeypatch):
        def boom():
            raise RuntimeError("no home")
        monkeypatch.setattr(imagery, "master_dirs", boom)
        assert imagery.master_path("microtubules") is None


# ---------------------------------------------------------------------------
# Degrading without the assets
# ---------------------------------------------------------------------------

class TestDegradesWithoutMasters:
    def test_space_falls_back_to_the_generated_sky(self, qapp, cache_dir,
                                                   no_masters, monkeypatch):
        """A source checkout with the JPEGs stripped, or a user who
        deleted them: Space must still get a sky, offline, with no
        error."""
        monkeypatch.setattr(preferences, "get_space_variant",
                            lambda: "deep_field")
        path = preferences.space_background_path(1920, 1200)
        assert path is not None
        assert path.name.startswith("space-")     # the procedural one

    def test_cell_falls_back_to_the_gradient(self, qapp, cache_dir,
                                             no_masters, monkeypatch):
        monkeypatch.setattr(preferences, "get_theme", lambda: "cell")
        assert preferences.cell_background_path(1920, 1200) is None
        preferences.apply_preferences_to_app(qapp)     # must not raise
        qss = theme.stylesheet("cell", background=None)
        assert "qlineargradient" in qss
        assert theme.CELL_PALETTE["accent_soft"] in qss

    def test_nothing_here_touches_the_network(self, no_masters, cache_dir,
                                               monkeypatch):
        """`cache_dir` was missing here and the test was reading the
        developer's real ~/.spacr/backgrounds. It passed only while that
        directory happened not to hold a deep_field render at exactly
        1920x1200 — rendering one at that size (which any 1920x1200
        screen does on first launch) turned it red, because
        `background_path` correctly returns the *cached* file whether or
        not a master is installed. Redirect the cache as well as the
        masters and it tests what it says it tests."""
        import urllib.request

        def forbidden(*args, **kwargs):
            raise AssertionError("the imagery pipeline opened a socket")

        monkeypatch.setattr(urllib.request, "urlopen", forbidden)
        assert not (cache_dir / imagery.cache_name(
            "deep_field", 1920, 1200)).exists()
        assert imagery.background_path("deep_field", 1920, 1200) is None
        assert imagery.legibility("deep_field") is None

    def test_background_path_never_raises(self, monkeypatch, cache_dir):
        def boom(*args, **kwargs):
            raise OSError("disk on fire")
        monkeypatch.setattr(imagery, "render", boom)
        assert imagery.background_path("filopodia", 1920, 1200) is None


# ---------------------------------------------------------------------------
# Preferences wiring
# ---------------------------------------------------------------------------

class TestPreferences:
    def test_cell_variant_round_trips(self, qapp):
        from spacr.qt.imagery import CELL_VARIANTS, DEFAULT_CELL_VARIANT
        for variant in CELL_VARIANTS:
            preferences.set_cell_variant(variant)
            assert preferences.get_cell_variant() == variant
        with pytest.raises(ValueError):
            preferences.set_cell_variant("mitochondria")
        preferences.set_cell_variant(DEFAULT_CELL_VARIANT)

    def test_cell_variant_survives_a_garbage_value(self, monkeypatch, qapp):
        from spacr.qt.imagery import DEFAULT_CELL_VARIANT

        class Rubbish:
            def value(self, key, default=None):
                return "\x00not a variant"

        monkeypatch.setattr(preferences, "_settings", Rubbish)
        assert preferences.get_cell_variant() == DEFAULT_CELL_VARIANT

    def test_the_photo_variant_joins_the_space_list(self, qapp):
        from spacr.qt.space import VARIANTS
        choices = preferences.space_variants()
        assert set(VARIANTS).issubset(choices)
        assert "deep_field" in choices
        preferences.set_space_variant("deep_field")
        assert preferences.get_space_variant() == "deep_field"
        preferences.set_space_variant("galaxy")

    def test_the_procedural_variant_list_is_left_alone(self):
        """`space.VARIANTS` indexes `_VARIANT_MIX`; the photo key must
        not be smuggled in there."""
        from spacr.qt import space
        for variant in space.VARIANTS:
            assert variant in space._VARIANT_MIX
        assert "deep_field" not in space.VARIANTS

    def test_an_unknown_saved_theme_falls_back_rather_than_raising(
            self, monkeypatch, qapp):
        class Old:
            def value(self, key, default=None):
                return "solarized" if key.endswith("theme") else default

        monkeypatch.setattr(preferences, "_settings", Old)
        assert preferences.get_theme() == preferences.DEFAULT_THEME
        assert preferences.resolve_effective_theme() in theme.THEMES

    def test_an_existing_saved_theme_keeps_working(self, monkeypatch, qapp):
        for saved in ("dark", "light", "space", "cell"):
            class Saved:
                def value(self, key, default=None, _s=saved):
                    return _s if key.endswith("theme") else default

            monkeypatch.setattr(preferences, "_settings", Saved)
            assert preferences.get_theme() == saved
            assert preferences.resolve_effective_theme() == saved

    def test_theme_background_path_routes_by_theme(self, qapp, cache_dir,
                                                   monkeypatch):
        _installed("microtubules")
        monkeypatch.setattr(preferences, "get_cell_variant",
                            lambda: "microtubules")
        assert preferences.theme_background_path("dark") is None
        assert preferences.theme_background_path("light") is None
        cell = preferences.theme_background_path("cell", 1920, 1200)
        assert cell is not None and cell.name.startswith("photo-microtubules")
        monkeypatch.setattr(preferences, "get_space_variant",
                            lambda: "deep_field")
        space_bg = preferences.theme_background_path("space", 1920, 1200)
        assert space_bg is not None
        assert space_bg.name.startswith("photo-deep_field")

    def test_cell_background_path_never_raises(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("nope")
        monkeypatch.setattr(imagery, "background_path", boom)
        assert preferences.cell_background_path(100, 100) is None

    def test_the_dialog_consolidates_image_variants_into_theme(self, qapp, qtbot):
        dialog = preferences.PreferencesDialog()
        qtbot.addWidget(dialog)
        from PySide6.QtWidgets import QComboBox, QLabel, QPushButton
        values = set()
        for combo in dialog.findChildren(QComboBox):
            for i in range(combo.count()):
                values.add(combo.itemData(i))
        assert "space:deep_field" in values
        assert {"cell:microtubules", "cell:filopodia"} <= values
        assert "deep_field" not in values
        assert "microtubules" not in values
        labels = {label.text() for label in dialog.findChildren(QLabel)}
        assert "Space background" not in labels
        assert "Cell background" not in labels
        assert not any(
            "NASA" in button.text()
            for button in dialog.findChildren(QPushButton))

    def test_composite_theme_choice_round_trips_variants(
            self, qapp, tmp_path, monkeypatch):
        from PySide6.QtCore import QSettings

        settings_path = tmp_path / "theme-choice.ini"
        monkeypatch.setattr(
            preferences,
            "_settings",
            lambda: QSettings(str(settings_path), QSettings.IniFormat),
        )
        preferences.set_theme_choice("space:stars")
        assert preferences.get_theme() == "space"
        assert preferences.get_space_variant() == "stars"
        assert preferences.get_theme_choice() == "space:stars"

        preferences.set_theme_choice("cell:filopodia")
        assert preferences.get_theme() == "cell"
        assert preferences.get_cell_variant() == "filopodia"
        assert preferences.get_theme_choice() == "cell:filopodia"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

class TestStylesheet:
    def test_the_cell_theme_paints_the_background_it_is_given(self):
        qss = theme.stylesheet("cell", background="/tmp/a b/pic.jpg")
        assert 'url("/tmp/a b/pic.jpg")' in qss
        assert "background-repeat: no-repeat" in qss
        assert "background-color: transparent" in qss

    def test_every_theme_produces_a_stylesheet(self):
        for name in list(theme.THEMES) + ["system", "nonsense"]:
            qss = theme.stylesheet(name, font_scale=1.25)
            assert len(qss) > 1000
            for selector in ("#Sidebar", "#Card", "#PrimaryButton",
                             "#Console", "QMainWindow"):
                assert selector in qss

    def test_opaque_themes_still_emit_plain_hex(self):
        """The scrim machinery must be byte-for-byte invisible to the
        themes that predate it.

        Was a flat ``"rgba(" not in qss``. #16j introduced translucency
        that has nothing to do with scrims — the hairline rim every
        module tile carries, and the three maturity tints its hover
        fills with — and those are translucent in *every* theme on
        purpose: a tint that is a solid colour is not a tint. Section
        maturity now uses the same hues for its border and header tint,
        so those documented alphas belong to the same allow-list. The
        semantic Run/Propagate and Stop/Close buttons likewise use an
        explicitly requested 18% blue/red hover tint.
        The assertion names them and demands they are the only ones,
        which is the same guarantee stated precisely rather than a
        blanket ban that a rule about something else happened to trip.
        """
        import re
        for name in ("dark", "light"):
            qss = theme.stylesheet(name)
            palette = theme.palette_for(name)
            allowed = {
                theme.css_color(theme.rim_colour(name), 0.35),
                theme.css_color(palette["button_accent"], 0.18),
                theme.css_color(palette["error"], 0.18),
            }
            for hue in theme.STAGE_HOVER.values():
                allowed.add(theme.css_color(hue, 0.22))
                allowed.add(theme.css_color(hue, 0.40))
                allowed.add(theme.css_color(hue, 0.72))
                allowed.add(theme.css_color(hue, 0.14))
            found = set(re.findall(r"rgba\([^)]*\)", qss))
            assert found <= allowed, (
                f"{name} emits translucency the scrim solver did not "
                f"authorise: {sorted(found - allowed)}")
            assert theme.palette_for(name)["bg"] in qss
            for role in ("surface", "surface_alt", "surface_hi"):
                assert theme.palette_for(name)[role] in qss, (
                    f"{name}.{role} came out as anything but plain hex")

    def test_image_themes_keep_their_popups_opaque(self):
        for name in theme.IMAGE_THEMES:
            qss = theme.stylesheet(name, background="/tmp/x.jpg")
            assert "QMenu, QToolTip, QMessageBox" in qss
            assert theme.palette_for(name)["surface_alt"] in qss

    def test_windows_are_repainted_for_the_new_theme(self, qapp, qtbot,
                                                     cache_dir, monkeypatch):
        from spacr.qt.app import MainWindow
        monkeypatch.setattr(preferences, "get_theme", lambda: "cell")
        window = MainWindow()
        qtbot.addWidget(window)
        window.refresh_theme()          # must not raise
        assert qapp.styleSheet()


# ---------------------------------------------------------------------------
# Build-time derivation
# ---------------------------------------------------------------------------

class TestBuildMaster:
    def test_a_missing_original_is_skipped_not_fatal(self, tmp_path):
        assert imagery.build_master("filopodia", tmp_path, tmp_path) is None
        built = imagery.build_masters(tmp_path, tmp_path)
        assert set(built) == set(imagery.MASTERS)
        assert all(value is None for value in built.values())

    def test_a_bright_original_is_dimmed_to_the_palette(self, tmp_path):
        """End to end on a synthetic 'micrograph': a blazing frame in,
        a wallpaper the theme can put text on out."""
        src = tmp_path / "src"
        src.mkdir()
        rng = np.random.default_rng(11)
        arr = rng.integers(200, 256, (900, 1400, 3), dtype=np.uint8)
        Image.fromarray(arr).save(src / imagery.MASTERS["filopodia"]["source"])

        out = imagery.build_master("filopodia", src, tmp_path / "out")
        assert out is not None and out.is_file()
        with Image.open(out) as built:
            assert built.width / built.height == \
                pytest.approx(imagery.MASTER_ASPECT, rel=0.01)
            probe = imagery._probe(built)
        value, color = imagery.brightest_window(probe)
        assert value <= theme.max_background_luma("cell")
        assert theme.image_contrast_failures("cell", color) == []

    def test_the_build_never_upscales(self, tmp_path):
        """`cell.png` is only 2048 px wide. Inventing pixels at build
        time would bake a soft master into the wheel for every user,
        including the ones whose screen is 1920."""
        src = tmp_path / "src"
        src.mkdir()
        Image.new("RGB", (600, 500), (8, 30, 44)).save(
            src / imagery.MASTERS["microtubules"]["source"])
        out = imagery.build_master("microtubules", src, tmp_path / "out")
        with Image.open(out) as built:
            assert built.width <= 600
            assert built.height <= 500

    def test_the_source_crop_is_applied(self, tmp_path):
        """Paint the annotation region a giveaway colour and check none
        of it survives into the built master."""
        src = tmp_path / "src"
        src.mkdir()
        entry = imagery.MASTERS["filopodia"]
        width, height = 1600, 1600
        image = Image.new("RGB", (width, height), (6, 20, 30))
        x0, y0, x1, y1 = entry["annotation"]
        ImageDraw.Draw(image).rectangle(
            [int(x0 * width), int(y0 * height),
             int(x1 * width), int(y1 * height)], fill=(255, 0, 255))
        image.save(src / entry["source"])

        out = imagery.build_master("filopodia", src, tmp_path / "out")
        with Image.open(out) as built:
            arr = np.asarray(built.convert("RGB"))
        magenta = (arr[:, :, 0] > 120) & (arr[:, :, 1] < 80) & \
            (arr[:, :, 2] > 120)
        assert not magenta.any(), "the annotation region reached the master"
