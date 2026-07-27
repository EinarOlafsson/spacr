"""The generated sky, measured against the rule the photographs obey.

:mod:`spacr.qt.imagery` has always solved every photographic master down
to :func:`spacr.qt.imagery.exposure_target` — the brightest a bare window
background may be before white text on it drops under WCAG AA — and has
always been able to *report* on any wallpaper with
:func:`spacr.qt.imagery.legibility`. The procedurally generated sky had
never been through either. Measured at 1440x900 before this change, the
brightest text-line-sized region of the sky was:

===========  ==========  ==============  =====================
variant      brightest   vs 0.0586 limit bare white text
===========  ==========  ==============  =====================
``galaxy``   0.4879      8.3x            1.96:1 (needs 4.5:1)
``sun``      0.8201      14.0x           1.20:1
``stars``    0.5041      8.6x            1.90:1
===========  ==========  ==============  =====================

Every test here is about one of three things:

1. that the sky is now inside the limit, at every size, variant and seed
   the app can ask for, measured with the same function and the same
   window used on the photographs;
2. that the price was paid where the rule was actually broken — the
   *sun*, which is a caption-sized bright patch — and not where it was
   not: the empty sky's exposure anchor and the star cores come out
   byte-for-byte identical; and
3. that the two routes that would have destroyed the picture were
   rejected on measurement rather than taste. `test_a_global_dim_would_
   have_blacked_the_sky_out` runs the obvious one and shows what it does.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt import imagery, space, theme


LIMIT = theme.max_background_luma("space")


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    """Redirect the background cache into a tmp dir for the whole test."""
    target = tmp_path / "backgrounds"
    monkeypatch.setenv(space.ENV_CACHE_DIR, str(target))
    return target

#: Sizes that between them cover the clamps, the small-frame regime where
#: the bloom is proportionally widest, and a real panel.
SIZES = ((16, 16), (213, 137), (320, 200), (720, 450), (1440, 900))


def measure(arr):
    """``(brightest text-window luminance, its colour)`` of a frame."""
    return imagery.brightest_window(space._measure_probe(arr))


def linear_luma(arr):
    return imagery.luminance_map(arr)


# ---------------------------------------------------------------------------
# The bug, reproduced — so that nothing below can pass vacuously
# ---------------------------------------------------------------------------

class TestTheUnboundedSkyReallyWasIllegible:
    """``legible=False`` is the "before" picture, kept renderable.

    Without these the assertions further down would be satisfied by a
    generator that had simply always been dark enough.
    """

    @pytest.mark.parametrize("variant,expected", [
        ("galaxy", 8.3), ("sun", 14.0), ("stars", 8.6)])
    def test_the_measured_overshoot_is_the_one_that_was_reported(
            self, variant, expected):
        arr = space.render(1440, 900, variant, legible=False)
        value, _ = imagery.brightest_window(arr)
        assert value / LIMIT == pytest.approx(expected, abs=0.15)

    @pytest.mark.parametrize("variant", space.VARIANTS)
    def test_bare_text_over_the_unbounded_sky_fails_wcag(self, variant):
        arr = space.render(1440, 900, variant, legible=False)
        _, color = imagery.brightest_window(arr)
        failures = theme.image_contrast_failures("space", color)
        assert failures, "the unbounded sky must fail, or nothing is fixed"
        # The headline rule: white body text on the wallpaper.
        assert any(line.startswith("fg (#ffffff)") for line in failures)

    def test_the_ai_toggle_colour_is_among_the_casualties(self):
        """The reported symptom: the title-bar AI/LP toggle is painted
        with ``accent`` straight onto the wallpaper, and it landed on the
        sun's halo."""
        arr = space.render(1440, 900, "sun", legible=False)
        _, color = imagery.brightest_window(arr)
        palette = theme.palette_for("space")
        assert theme.contrast_ratio(palette["accent"], color) < 4.5


# ---------------------------------------------------------------------------
# The fix, measured
# ---------------------------------------------------------------------------

class TestTheSkyIsNowInsideTheLimit:
    @pytest.mark.parametrize("variant", space.VARIANTS)
    def test_every_variant_passes_at_a_real_panel_size(self, variant):
        report = space.legibility(variant, 1440, 900)
        assert report["passes"], report
        assert report["failures"] == []
        assert report["brightest"] <= report["limit"]

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("variant", space.VARIANTS)
    def test_it_holds_at_every_size(self, size, variant):
        value, _ = measure(space.render(*size, variant))
        assert value <= LIMIT, f"{variant} at {size} measures {value:.4f}"

    @pytest.mark.parametrize("seed", (1, 7, 11, 20250726))
    def test_it_holds_for_any_seed(self, seed):
        """The seed is a user-facing knob, so the guarantee cannot be a
        property of one pleasing sky."""
        for variant in space.VARIANTS:
            value, _ = measure(space.render(720, 450, variant, seed=seed))
            assert value <= LIMIT

    def test_an_unknown_variant_is_bounded_too(self):
        """It falls back to the default mix; the fallback must not be a
        hole in the guarantee."""
        value, _ = measure(space.render(720, 450, "supernova"))
        assert value <= LIMIT

    def test_the_photographic_space_wallpaper_is_held_to_the_same_number(self):
        """``deep_field`` is the fourth thing the Space theme can show.
        It went through this solve already; the point is that it is the
        *same* solve, so the theme has one ceiling and not two."""
        report = imagery.legibility("deep_field")
        if report is None:
            pytest.skip("deep_field master not installed in this build")
        assert report["theme"] == "space"
        assert report["limit"] == LIMIT
        assert report["target"] == space.exposure_target()
        assert report["passes"] and report["failures"] == []


# ---------------------------------------------------------------------------
# What it did NOT cost — the reconciliation, made assertable
# ---------------------------------------------------------------------------

class TestTheSkyAnchorSurvived:
    """``TARGET_SKY_PERCENTILE`` anchors the exposure on empty sky. The
    claim is that the limit and that anchor are about different things
    and can both be honoured; these are the measurements behind it."""

    @pytest.mark.parametrize("variant", space.VARIANTS)
    def test_the_empty_sky_is_exactly_as_dark_as_it_was(self, variant):
        before = linear_luma(space.render(1440, 900, variant, legible=False))
        after = linear_luma(space.render(1440, 900, variant))
        assert float(np.percentile(after, space.TARGET_SKY_PERCENTILE)) == \
            pytest.approx(float(np.percentile(
                before, space.TARGET_SKY_PERCENTILE)), rel=1e-6)

    @pytest.mark.parametrize("variant", space.VARIANTS)
    def test_star_cores_are_untouched_to_the_byte(self, variant):
        """The exemption the whole design rests on: a point source does
        not move a text-window mean, so it does not have to pay."""
        cy, cx, r = int(0.74 * 900), int(0.80 * 1440), int(0.22 * 900)

        def peak_away_from_the_sun(arr):
            gray = arr.max(axis=2).copy()
            gray[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 0
            return int(gray.max())

        before = space.render(1440, 900, variant, legible=False)
        after = space.render(1440, 900, variant)
        assert peak_away_from_the_sun(after) == peak_away_from_the_sun(before)

    def test_the_starfield_on_its_own_is_nowhere_near_the_limit(self,
                                                                monkeypatch):
        """The number that justifies exempting it: 1-11 % of the limit
        for the whole field, against a sun that was 1400 %.

        Measured at the exposure the *composed* frame actually uses —
        tone-mapping a bare starfield would re-solve the exposure for a
        near-empty image and answer a different question.
        """
        seen = []
        real = space.tone_exposure
        monkeypatch.setattr(space, "tone_exposure",
                            lambda luma: seen.append(real(luma)) or seen[-1])
        for variant in space.VARIANTS:
            seen.clear()
            space.render(1440, 900, variant)
            mix = space._VARIANT_MIX[variant]
            stars = space.starfield(1440, 900, seed=space.DEFAULT_SEED,
                                    density=space.STAR_DENSITY * mix["stars"])
            ldr = space._apply_tone_curve(stars, seen[0])
            arr = np.clip(ldr * 255.0 + 0.5, 0, 255).astype(np.uint8)
            value, _ = imagery.brightest_window(arr)
            assert value < 0.15 * LIMIT, \
                f"{variant} starfield alone is {value / LIMIT:.2f}x the limit"

    def test_the_sky_still_reaches_white_where_the_generator_draws_a_star(self):
        """A sky with no bright cores is a grey rectangle. 720x450 is the
        smallest size at which this seed draws a star past 200 — below it
        the frame's brightest pixel used to be the *sun disc*, which is
        the thing that had to come down."""
        arr = space.render(720, 450, "stars", seed=11)
        assert int(arr.max()) > 200

    def test_the_galaxy_barely_paid_anything(self):
        """It was never the offender — the 0.4879 the galaxy variant
        measured was its *sun*. Its own arms come through within a hair
        of untouched, which is why the highlight curve is a shoulder and
        not a power law."""
        before = space.render(1440, 900, "galaxy", legible=False)
        after = space.render(1440, 900, "galaxy")
        # Left half of the frame is galaxy; the sun sits bottom right.
        lhs = slice(None), slice(0, 700)
        kept = float(linear_luma(after)[lhs].mean()
                     / linear_luma(before)[lhs].mean())
        assert kept > 0.9, f"the galaxy lost {(1 - kept) * 100:.0f}% of its light"


class TestTheRejectedAlternatives:
    """Both obvious routes were built, rendered and looked at. These keep
    the reasons in the suite rather than only in a docstring."""

    def test_a_global_dim_would_have_blacked_the_sky_out(self):
        """Calling ``solve_dim`` on the finished frame — the literal
        photographic recipe — needs a factor near 0.1 and takes the
        sky's own 40th percentile to zero."""
        arr = space.render(1440, 900, "galaxy", legible=False)
        measured, _ = imagery.brightest_window(arr)
        factor = imagery.solve_dim(measured, space.exposure_target())
        assert factor < 0.15
        dimmed = imagery.dim(arr, factor)
        assert float(np.percentile(linear_luma(dimmed),
                                   space.TARGET_SKY_PERCENTILE)) == 0.0
        assert int(dimmed.max()) < 100, "no star core survives it"

    def test_the_real_path_keeps_both(self):
        arr = space.render(1440, 900, "galaxy")
        assert float(np.percentile(linear_luma(arr),
                                   space.TARGET_SKY_PERCENTILE)) > 0.0
        assert int(arr.max()) > 150


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------

class TestSrgbEncode:
    def test_it_is_the_transfer_function_and_not_the_identity(self):
        """The unit confusion this exists to prevent: Space's 0.0586
        *linear* limit is ``#444444``, not a 6 % signal — the two
        readings are a factor of 4.6 apart."""
        assert imagery.srgb_encode(LIMIT) == pytest.approx(0.2685, abs=1e-3)
        assert imagery.srgb_encode(LIMIT) / LIMIT == pytest.approx(4.58,
                                                                   abs=0.05)
        assert imagery.srgb_encode(space.exposure_target()) == \
            pytest.approx(0.2546, abs=1e-3)

    def test_it_round_trips_the_lut_the_module_measures_with(self):
        for level in (0, 1, 17, 64, 128, 200, 255):
            linear = float(imagery.linear_rgb(
                np.full((1, 1, 3), level, dtype=np.uint8))[0, 0, 0])
            assert imagery.srgb_encode(linear) * 255.0 == \
                pytest.approx(level, abs=0.51)

    def test_the_toe_and_the_clamps(self):
        assert imagery.srgb_encode(0.0) == 0.0
        assert imagery.srgb_encode(0.001) == pytest.approx(0.001 * 12.92)
        assert imagery.srgb_encode(1.0) == pytest.approx(1.0)
        assert imagery.srgb_encode(-1.0) == 0.0
        assert imagery.srgb_encode(9.0) == pytest.approx(1.0)


class TestHighlightCeiling:
    def test_it_inverts_the_tone_curve_at_that_exposure(self):
        exposure = 0.37
        ceiling = space.highlight_ceiling(exposure)
        encoded = 1.0 - np.exp(-ceiling * exposure)
        assert float(encoded) == pytest.approx(
            imagery.srgb_encode(space.exposure_target()), abs=1e-9)

    def test_a_hotter_exposure_needs_a_lower_ceiling(self):
        assert space.highlight_ceiling(2.0) < space.highlight_ceiling(0.5)

    @pytest.mark.parametrize("exposure,target", [
        (0.5, 0.0), (0.5, -1.0), (0.0, None), (-1.0, None)])
    def test_a_palette_that_admits_no_wallpaper_means_no_ceiling(
            self, exposure, target):
        """``max_background_luma`` is negative for the light theme. That
        must read as "nothing to solve", not as "clamp to zero"."""
        assert space.highlight_ceiling(exposure, target) == float("inf")

    def test_a_palette_that_admits_white_compresses_nothing(self):
        ceiling = space.highlight_ceiling(0.5, 1.0)
        img = np.full((4, 4, 3), 200.0, dtype=np.float32)
        out = space._compress_highlights(img.copy(), ceiling)
        mapped = space._apply_tone_curve(out, 0.5)
        assert float(mapped.min()) == pytest.approx(1.0)


class TestCompressHighlights:
    def _ramp(self, top):
        ramp = np.linspace(0.0, top, 256, dtype=np.float32)
        return np.repeat(ramp[None, :, None], 3, axis=2).copy()

    def test_it_bounds_the_luminance(self):
        out = space._compress_highlights(self._ramp(50.0), 2.0)
        assert float(space._luma(out).max()) <= 2.0 + 1e-6

    def test_it_is_the_identity_below_the_knee(self):
        img = self._ramp(50.0)
        before = img.copy()
        out = space._compress_highlights(img, 2.0)
        foot = 2.0 * space.HIGHLIGHT_KNEE
        under = space._luma(before) <= foot
        assert under.any()
        assert np.array_equal(out[under], before[under])

    def test_it_is_monotone_so_nothing_inverts(self):
        out = space._luma(space._compress_highlights(self._ramp(50.0), 2.0))
        # float32 ulp at 2.0 is 2.4e-7; the shoulder is flat up there.
        assert np.all(np.diff(out[0]) >= -1e-6)

    def test_it_preserves_hue(self):
        img = np.zeros((1, 4, 3), dtype=np.float32)
        img[0] = [[9.0, 3.0, 1.0], [1.0, 9.0, 3.0],
                  [0.01, 0.02, 0.03], [40.0, 40.0, 40.0]]
        out = space._compress_highlights(img.copy(), 2.0)
        for i in range(4):
            ratio = out[0, i] / img[0, i]
            assert ratio.max() - ratio.min() < 1e-5

    def test_a_frame_already_under_the_ceiling_is_returned_untouched(self):
        img = self._ramp(0.5)
        assert space._compress_highlights(img.copy(), 2.0).max() == \
            pytest.approx(img.max())

    @pytest.mark.parametrize("ceiling", (float("inf"), 0.0, -1.0))
    def test_a_degenerate_ceiling_does_nothing(self, ceiling):
        img = self._ramp(50.0)
        out = space._compress_highlights(img.copy(), ceiling)
        assert np.array_equal(out, img)

    def test_the_join_has_no_step_in_it(self):
        """C1 at the knee — value and slope both match — or a smooth
        gradient crossing it gains a Mach band."""
        ceiling = 2.0
        foot = ceiling * space.HIGHLIGHT_KNEE
        span = np.linspace(foot * 0.98, foot * 1.02, 4001, dtype=np.float32)
        img = np.repeat(span[None, :, None], 3, axis=2).copy()
        out = space._luma(space._compress_highlights(img, ceiling))
        slope = np.diff(out) / np.diff(space._luma(
            np.repeat(span[None, :, None], 3, axis=2)))
        assert float(np.abs(np.diff(slope)).max()) < 1e-3


class TestMeasureProbe:
    def test_a_small_frame_is_measured_as_is(self):
        arr = np.zeros((100, 200, 3), dtype=np.uint8)
        assert space._measure_probe(arr) is arr

    def test_a_large_frame_is_box_averaged_not_point_sampled(self):
        """Point sampling would let the probe miss or duplicate a star;
        the box average is what makes the thumbnail's window mean the
        same number as the full frame's."""
        arr = np.zeros((900, 1440, 3), dtype=np.uint8)
        arr[:, ::2] = 255
        probe = space._measure_probe(arr)
        assert max(probe.shape[:2]) <= 480
        assert 120 <= int(probe.mean()) <= 135

    def test_the_probe_agrees_with_the_full_resolution_measurement(self):
        arr = space.render(1440, 900, "sun")
        full, _ = imagery.brightest_window(arr)
        probed, _ = imagery.brightest_window(space._measure_probe(arr))
        assert probed == pytest.approx(full, abs=0.004)


class TestEnforceLegibility:
    def test_it_dims_something_that_is_over(self):
        """Not a vacuous guard: hand it white and it must come back
        inside the limit."""
        white = np.full((120, 400, 3), 255, dtype=np.uint8)
        out = space._enforce_legibility(white)
        value, _ = imagery.brightest_window(out)
        assert value <= LIMIT
        assert int(out.max()) < 255

    def test_it_leaves_something_already_dark_alone(self):
        dark = np.full((120, 400, 3), 8, dtype=np.uint8)
        assert space._enforce_legibility(dark) is dark

    def test_a_target_of_zero_is_a_no_op_rather_than_a_black_frame(self):
        white = np.full((120, 400, 3), 255, dtype=np.uint8)
        assert space._enforce_legibility(white, target=0.0) is white

    def test_it_really_fires_inside_render_at_the_sizes_that_need_it(
            self, monkeypatch):
        """The highlight ceiling bounds the smooth layers; the starfield,
        the bloom and the vignette land afterwards. At 320x200 the bloom
        is proportionally widest and they do push it over, so the
        measured guard is load-bearing and not decoration."""
        seen = []
        real = imagery.solve_dim
        monkeypatch.setattr(imagery, "solve_dim",
                            lambda m, t: seen.append(real(m, t)) or seen[-1])
        space.render(320, 200, "stars")
        assert seen and seen[-1] < 1.0
        seen.clear()
        space.render(1440, 900, "galaxy")
        assert seen and seen[-1] == 1.0


class TestExposureTargetAgreesWithTheImageryPipeline:
    def test_one_number_from_one_function(self):
        assert space.exposure_target() == imagery.exposure_target("space")
        assert space.exposure_target() == pytest.approx(
            LIMIT * imagery.SAFETY_MARGIN)

    def test_it_moves_with_the_palette_rather_than_being_tabulated(self,
                                                                   monkeypatch):
        monkeypatch.setattr(theme, "max_background_luma", lambda name: 0.02)
        assert space.exposure_target() == pytest.approx(
            0.02 * imagery.SAFETY_MARGIN)


class TestLegibilityReport:
    def test_it_matches_the_shape_imagery_returns_for_a_photograph(self):
        generated = space.legibility("galaxy", 320, 200)
        photographed = imagery.legibility_of(
            np.zeros((40, 60, 3), dtype=np.uint8), "space", key="x")
        assert set(generated) == set(photographed)
        assert generated["theme"] == "space"
        assert generated["key"] == "space:galaxy"

    def test_it_defaults_to_the_screen_size(self, monkeypatch):
        monkeypatch.setattr(space, "screen_size", lambda: (240, 150))
        assert space.legibility("stars")["passes"]

    def test_legibility_of_reports_a_failing_image_honestly(self):
        white = np.full((40, 60, 3), 255, dtype=np.uint8)
        report = imagery.legibility_of(white, "space")
        assert not report["passes"]
        assert report["failures"]
        assert report["color"] == "#ffffff"


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

class TestCacheVersion:
    def test_it_was_bumped_past_the_too_bright_sky(self):
        """Size, variant and seed are all unchanged, so nothing else in
        the cache key would ever invalidate a v1 PNG — every existing
        user would keep the 8-14x sky forever."""
        assert space.CACHE_VERSION >= 2
        assert f"v{space.CACHE_VERSION}" in \
            space.cache_name(100, 100, "galaxy", 1)

    def test_a_v1_file_is_not_reused(self, tmp_path, monkeypatch, qapp):
        monkeypatch.setenv(space.ENV_CACHE_DIR, str(tmp_path))
        stale = tmp_path / space.cache_name(64, 48, "stars", 3).replace(
            f"v{space.CACHE_VERSION}", "v1")
        tmp_path.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"x" * 4096)
        path = space.background_path(64, 48, "stars", seed=3)
        assert path is not None and path != stale
        assert stale.read_bytes() == b"x" * 4096, "the old file is not touched"


class TestTheDownloadedImageIsBoundedToo:
    """``preferences.space_background_path`` returns a downloaded NASA
    frame *ahead of* the generated sky, and hands it to the stylesheet
    unrendered. It is therefore the one remaining way an unbounded
    picture could get behind the app's text — and one of the three
    offered images is a solar flare."""

    def _payload(self, arr):
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "PNG")
        return buf.getvalue()

    def test_a_blown_out_download_is_dimmed_before_it_is_installed(
            self, cache_dir, qapp):
        bright = np.full((400, 700, 3), 250, dtype=np.uint8)
        payload = self._payload(bright)
        assert imagery.brightest_window(bright)[0] > LIMIT

        record = space.download_nasa_background(
            key="sun_flare", opener=lambda url, timeout: payload)
        assert record is not None
        stored = space.downloaded_background()
        assert stored is not None
        report = imagery.legibility_of(
            np.asarray(imagery._open_master(stored), dtype=np.uint8), "space")
        assert report["passes"], report
        assert report["failures"] == []

    def test_an_already_dark_download_is_left_alone(self, cache_dir, qapp):
        dark = space.render(200, 140, "stars", seed=4)
        payload = self._payload(dark)
        assert space.download_nasa_background(
            key="carina", opener=lambda url, timeout: payload) is not None
        stored = space.downloaded_background()
        assert stored.read_bytes() == payload, \
            "a frame already under the limit must not be re-encoded"

    def test_an_image_that_cannot_be_solved_is_refused_not_installed(
            self, cache_dir, qapp, monkeypatch):
        payload = self._payload(space.render(120, 90, "stars", seed=4))
        monkeypatch.setattr(imagery, "solve_image_file",
                            lambda *a, **k: False)
        assert space.download_nasa_background(
            key="carina", opener=lambda url, timeout: payload) is None
        assert space.downloaded_background() is None

    def test_solve_image_file_reports_failure_rather_than_raising(
            self, tmp_path):
        assert imagery.solve_image_file(tmp_path / "nope.jpg", "space") is False
        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"not an image at all" * 20)
        assert imagery.solve_image_file(broken, "space") is False


class TestDeterminismSurvivedTheSolve:
    def test_the_same_arguments_give_the_same_bytes(self):
        a = space.render(213, 137, "sun", seed=99)
        b = space.render(213, 137, "sun", seed=99)
        assert np.array_equal(a, b)

    def test_the_bound_is_not_silently_a_no_op(self):
        a = space.render(213, 137, "sun", seed=99)
        b = space.render(213, 137, "sun", seed=99, legible=False)
        assert not np.array_equal(a, b)
