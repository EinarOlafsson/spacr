"""The scrim solver, and the page surfaces the backdrop shows through.

Two bugs are pinned here, both reported the same way by the user:
"the themes don't seem to be implemented correctly, I can't see the
cells", and "the matrix effect ... should be visible" behind the
`map_barcodes` title and its black page background.

They had one cause between them — *everything painted on top of the
backdrop was opaque* — and two mechanisms:

1. The image themes' scrims were a hand-picked 0.86-0.93. A 0.90 scrim
   transmits a 1.10:1 range of the picture under it, which is right at
   the threshold of a visible difference. The theme worked; it was a
   10 % ghost of itself. :func:`theme.solve_scrim_alpha` now derives
   the alpha instead, and :data:`theme.MIN_PICTURE_CONTRAST` is the
   number it is derived against.
2. Under dark and light, ``QWidget`` is an opaque ``bg``, so the
   *first* container between the DNA rain and the eye buried it — the
   header carrying the screen title, the splitter, the settings scroll
   area and its viewport. :func:`theme.make_transparent` is the opt-out
   and ``AppScreen._clear_page_surfaces`` is where it is applied.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QScrollArea, QWidget

from spacr.qt import theme
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets import Section


# ---------------------------------------------------------------------------
# The worst case each theme can actually present
# ---------------------------------------------------------------------------

class TestScrimUnder:
    def test_grey_inverts_the_luminance_it_is_given(self):
        for target in (0.0, 0.002, 0.0586, 0.1088, 0.5, 1.0):
            grey = theme._grey_for_luminance(target)
            assert theme.relative_luminance(grey) == \
                pytest.approx(target, abs=0.004)

    def test_grey_takes_the_linear_branch_for_near_black(self):
        """Below 0.0031308 sRGB is linear, not a power law. Getting this
        wrong only shows up on the darkest greys, which is exactly where
        an image theme lives."""
        assert theme._grey_for_luminance(0.0) == "#000000"
        # 0.001 * 12.92 * 255 = 3.3 -> #030303. Through the power law it
        # would have been #0f0f0f, five times too bright.
        assert theme._grey_for_luminance(0.001) == "#030303"

    def test_grey_clamps_rather_than_producing_nonsense(self):
        assert theme._grey_for_luminance(-5.0) == "#000000"
        assert theme._grey_for_luminance(9.0) == "#ffffff"

    def test_space_is_judged_against_white_because_its_sun_is_white(self):
        """`spacr.qt.space` anchors its exposure on the 40th percentile
        precisely so a sun stays blown out, so Space really can put a
        near-white text-line-sized region behind a panel. Measured on a
        generated 1440x900 galaxy sky: 0.49 luminance, ``#bab9b9``."""
        assert theme.scrim_under("space") == "#ffffff"

    def test_cell_is_judged_against_the_ceiling_its_imagery_is_solved_to(self):
        """Every Cell wallpaper comes out of `imagery.render`, which
        exposure-solves it. White is out of contract; the ceiling is the
        honest worst case, and it is what lets Cell's scrims be thin
        enough to see the micrograph through."""
        under = theme.scrim_under("cell")
        assert under != "#ffffff"
        assert theme.relative_luminance(under) == pytest.approx(
            theme.max_background_luma("cell"), abs=0.005)

    def test_opaque_themes_get_the_white_default(self):
        """Never used — their alphas are 1.0 — but it must not raise, and
        `max_background_luma` is negative for light."""
        for name in ("dark", "light", "system", "no-such-theme"):
            assert theme.scrim_under(name) == theme.WORST_CASE_UNDER

    def test_only_an_exposure_bounded_theme_may_relax_its_worst_case(self):
        """The relaxation is earned by the wallpaper pipeline, not
        granted by the palette. A theme in this set whose picture is not
        actually solved would get panels too thin for its own
        background."""
        assert set(theme.EXPOSURE_BOUNDED_THEMES) <= set(theme.IMAGE_THEMES)
        assert "space" not in theme.EXPOSURE_BOUNDED_THEMES
        for name in theme.THEMES:
            relaxed = theme.scrim_under(name) != theme.WORST_CASE_UNDER
            assert relaxed == (name in theme.EXPOSURE_BOUNDED_THEMES)


# ---------------------------------------------------------------------------
# The two bounds
# ---------------------------------------------------------------------------

class TestScrimBounds:
    def test_the_floor_is_the_thinnest_legible_scrim(self):
        for name in theme.IMAGE_THEMES:
            palette = theme.palette_for(name)
            for role in ("surface", "surface_alt", "surface_hi"):
                floor = theme.legible_scrim_floor(name, role)
                rules = theme._scrim_rules(role)
                assert rules
                at = theme.composite(palette[role], floor,
                                     theme.scrim_under(name))
                for fg, required in rules:
                    assert theme.contrast_ratio(palette[fg], at) >= required
                if floor > 0.0:
                    below = theme.composite(palette[role], floor - 0.002,
                                            theme.scrim_under(name))
                    assert any(
                        theme.contrast_ratio(palette[fg], below)
                        < required * theme.SCRIM_HEADROOM
                        for fg, required in rules), \
                        f"{name}.{role}: the floor is not tight"

    def test_an_unsolvable_surface_falls_back_to_fully_opaque(self):
        """Demand a headroom no alpha can buy and the floor must say
        1.0 — fully opaque, the safest answer — rather than run off the
        end of the sweep and return whatever the last iteration left."""
        original = theme.SCRIM_HEADROOM
        try:
            theme.SCRIM_HEADROOM = 100.0
            assert theme.legible_scrim_floor("space", "surface") == 1.0
            assert theme.legible_scrim_floor("cell", "surface_hi") == 1.0
        finally:
            theme.SCRIM_HEADROOM = original

    def test_picture_contrast_is_1_to_1_when_the_panel_is_opaque(self):
        for name in theme.IMAGE_THEMES:
            assert theme.picture_contrast(name, "surface", 1.0) == \
                pytest.approx(1.0)
            assert theme.picture_contrast(name, "surface", 0.0) > 1.0

    def test_picture_contrast_falls_as_the_scrim_thickens(self):
        previous = None
        for step in range(0, 11):
            value = theme.picture_contrast("cell", "surface_alt", step / 10)
            if previous is not None:
                assert value <= previous + 1e-9
            previous = value

    def test_the_ceiling_is_the_thickest_see_through_scrim(self):
        for name in theme.IMAGE_THEMES:
            for role in ("surface", "surface_alt", "surface_hi"):
                ceiling = theme.present_scrim_ceiling(name, role)
                assert theme.picture_contrast(name, role, ceiling) >= \
                    theme.MIN_PICTURE_CONTRAST
                assert theme.picture_contrast(name, role, ceiling + 0.002) < \
                    theme.MIN_PICTURE_CONTRAST

    def test_an_unreachable_target_leaves_no_ceiling_at_all(self):
        """21:1 is the most any two colours can differ, so a target above
        it is satisfied by no alpha and the sweep must bottom out at 0.0
        rather than run off the end."""
        original = theme.MIN_PICTURE_CONTRAST
        try:
            theme.MIN_PICTURE_CONTRAST = 25.0
            assert theme.present_scrim_ceiling("space", "surface") == 0.0
        finally:
            theme.MIN_PICTURE_CONTRAST = original


# ---------------------------------------------------------------------------
# What the solver produced
# ---------------------------------------------------------------------------

class TestSolvedScrims:
    def test_every_image_theme_is_solved_for_every_role(self):
        for name in theme.IMAGE_THEMES:
            assert set(theme.SCRIM_ALPHA[name]) == \
                set(theme.SCRIM_ROLES) | {"elevated"}
            for role in theme.SCRIM_ROLES:
                assert 0.0 < theme.scrim_alpha(name, role) < 1.0

    def test_popups_stay_opaque(self):
        """A translucent top-level window without a compositor shows the
        desktop, not the wallpaper."""
        for name in theme.IMAGE_THEMES:
            assert theme.scrim_alpha(name, "elevated") == 1.0

    def test_the_solved_alpha_sits_between_its_two_bounds(self):
        for name in theme.IMAGE_THEMES:
            for role, colour_role in theme.SCRIM_ROLES.items():
                solved = theme.solve_scrim_alpha(name, role, colour_role)
                assert solved == theme.scrim_alpha(name, role)
                assert solved >= theme.legible_scrim_floor(
                    name, role, colour_role)

    def test_the_picture_now_actually_reads_through_every_panel(self):
        """The regression this whole change exists to prevent. At the
        old hand-picked alphas these numbers were 1.07-1.12:1 — a
        difference at the threshold of visibility, which is why the
        themes were reported as unimplemented."""
        for name in theme.IMAGE_THEMES:
            assert theme.scrim_failures(name) == []
            for row in theme.scrim_report(name):
                assert row["picture"] >= theme.MIN_PICTURE_CONTRAST
                assert row["shows_picture"]

    def test_every_panel_is_still_legible_over_the_worst_case(self):
        """Legibility is the constraint the picture is not allowed to
        break. Asserted numerically, per role, not eyeballed."""
        for name in theme.IMAGE_THEMES:
            for row in theme.scrim_report(name):
                assert row["legible"], (
                    f"{name}.{row['role']}: {row['worst_fg']} only reaches "
                    f"{row['worst_ratio']:.2f}:1 of {row['required']:.1f}:1")
                assert row["worst_ratio"] >= row["required"]

    def test_the_scrims_came_down_from_the_hand_picked_originals(self):
        """The values this replaced, spelled out: surface 0.88,
        surface_alt 0.90, surface_hi 0.93, tile 0.86, for both themes."""
        was = {"surface": 0.88, "surface_alt": 0.90,
               "surface_hi": 0.93, "tile": 0.86}
        for name in theme.IMAGE_THEMES:
            for role, old in was.items():
                assert theme.scrim_alpha(name, role) < old, \
                    f"{name}.{role} did not come down from {old}"

    def test_cell_ends_up_far_thinner_than_space(self):
        """Not a coincidence and not a preference: Space is pinned by
        legibility because its sky blows out, Cell is pinned by the
        picture because its wallpaper is exposure-solved."""
        for role in theme.SCRIM_ROLES:
            assert theme.scrim_alpha("cell", role) < \
                theme.scrim_alpha("space", role)
        # ...and that is visible in which bound binds.
        for row in theme.scrim_report("space"):
            assert row["alpha"] > 0.5
        for row in theme.scrim_report("cell"):
            assert row["floor"] < 0.2, \
                "Cell legibility must not bind — its wallpaper is solved"

    def test_tile_is_solved_against_the_colour_it_is_painted_with(self):
        assert theme.SCRIM_ROLES["tile"] == "surface"
        for name in theme.IMAGE_THEMES:
            assert theme.scrim_alpha(name, "tile") == \
                theme.scrim_alpha(name, "surface")

    def test_the_report_carries_the_audit_trail(self):
        report = theme.scrim_report("cell")
        assert len(report) == len(theme.SCRIM_ROLES)
        assert all(set(row) >= {"role", "alpha", "floor", "ceiling",
                                "picture", "worst_fg", "worst_ratio",
                                "required", "legible", "shows_picture"}
                   for row in report)

    def test_a_scrim_that_cannot_do_both_is_reported_not_hidden(self):
        """Raise the bar past what any panel can transmit and the
        failure must be a sentence naming both bounds, not silence.

        Measured on Cell rather than Space: Space was retired, and this test
        needs any theme with a wallpaper behind its panels — which is what
        makes the two bounds pull against each other in the first place.
        """
        original = theme.MIN_PICTURE_CONTRAST
        try:
            theme.MIN_PICTURE_CONTRAST = 20.0
            failures = theme.scrim_failures("cell")
            assert len(failures) == len(theme.SCRIM_ROLES)
            assert all("shows the picture at" in line for line in failures)
            assert all("legibility floor" in line for line in failures)
        finally:
            theme.MIN_PICTURE_CONTRAST = original
        assert theme.scrim_failures("cell") == []


# ---------------------------------------------------------------------------
# Page surfaces
# ---------------------------------------------------------------------------

class TestMakeTransparent:
    def test_the_qss_carries_the_rule_in_every_theme(self):
        for name in theme.THEMES:
            qss = theme.stylesheet(name)
            assert f'*[{theme.TRANSPARENT_PROPERTY}="true"]' in qss

    def test_it_tags_the_widget(self, qapp):
        widget = QWidget()
        assert not widget.property(theme.TRANSPARENT_PROPERTY)
        theme.make_transparent(widget)
        assert widget.property(theme.TRANSPARENT_PROPERTY) is True

    def test_a_scroll_areas_viewport_is_tagged_too(self, qapp):
        """The viewport is the widget that actually paints. Tagging only
        the QScrollArea leaves an opaque rectangle exactly the size of
        the settings panel."""
        scroll = QScrollArea()
        theme.make_transparent(scroll)
        assert scroll.property(theme.TRANSPARENT_PROPERTY) is True
        assert scroll.viewport().property(theme.TRANSPARENT_PROPERTY) is True

    def test_none_entries_are_skipped(self, qapp):
        """Callers pass `getattr(self, "_header", None)` — a screen that
        failed to build one must not take the app down."""
        widget = QWidget()
        theme.make_transparent(None, widget, None)
        assert widget.property(theme.TRANSPARENT_PROPERTY) is True

    def test_a_widget_with_no_style_is_survivable(self, qapp):
        widget = QWidget()
        widget.style = lambda: None
        theme.make_transparent(widget)
        assert widget.property(theme.TRANSPARENT_PROPERTY) is True

    def test_a_scroll_area_with_no_viewport_is_survivable(self, qapp):
        scroll = QScrollArea()
        scroll.viewport = lambda: None
        theme.make_transparent(scroll)
        assert scroll.property(theme.TRANSPARENT_PROPERTY) is True

    def test_tagged_widgets_paint_nothing_in_the_opaque_themes(self, qapp):
        """The actual regression: under dark, an untagged container is a
        solid `bg` and buries whatever is behind the page."""
        for name in ("dark", "light"):
            qapp.setStyleSheet(theme.stylesheet(name))
            # The host stands in for the backdrop: it is tagged too, so
            # whatever it is rendered onto is what shows where nothing
            # paints. Magenta is a colour no palette contains.
            host = QWidget()
            host.resize(60, 40)
            theme.make_transparent(host)
            plain = QWidget(host)
            plain.setGeometry(0, 0, 60, 20)
            clear = QWidget(host)
            clear.setGeometry(0, 20, 60, 20)
            theme.make_transparent(clear)

            image = QImage(QSize(60, 40), QImage.Format_RGB32)
            image.fill(QColor("#ff00ff"))
            host.render(image)

            bg = QColor(theme.palette_for(name)["bg"]).rgb()
            assert QColor(image.pixel(30, 10)).rgb() == bg, \
                f"{name}: an untagged container must paint its background"
            assert QColor(image.pixel(30, 30)).rgb() == \
                QColor("#ff00ff").rgb(), \
                f"{name}: a tagged container must paint nothing at all"
        qapp.setStyleSheet(theme.stylesheet("dark"))


class TestAppScreenPageSurfaces:
    def test_the_rain_screen_clears_every_container_over_the_rain(
            self, qtbot, qt_theme_applied):
        screen = AppScreen("map_barcodes")
        qtbot.addWidget(screen)
        for name in ("_header", "_body_splitter", "_settings_scroll",
                     "_settings_content", "_runtime_wrap", "_console_wrap"):
            widget = getattr(screen, name)
            assert widget.property(theme.TRANSPARENT_PROPERTY) is True, \
                f"{name} still paints over the DNA rain"
        assert screen._settings_scroll.viewport().property(
            theme.TRANSPARENT_PROPERTY) is True

    def test_the_cards_on_it_are_left_opaque(self, qtbot, qt_theme_applied):
        """The user asked for the grey settings categories to stay on
        top. Tagging them too would have dissolved the form into the
        animation."""
        screen = AppScreen("map_barcodes")
        qtbot.addWidget(screen)
        sections = screen.findChildren(Section)
        assert sections
        for section in sections:
            assert not section.property(theme.TRANSPARENT_PROPERTY)

    def test_a_screen_with_no_backdrop_at_all_is_untouched(
            self, qtbot, qt_theme_applied, monkeypatch):
        """Surfaces are only dissolved when something is painting behind them.

        This used to build ``AppScreen("measure")`` and assert its surfaces
        were opaque, on the reasoning that only ``map_barcodes`` had an
        animation behind it. That premise is gone: every non-sequencing screen
        now installs the ambient backdrop, so ``measure``'s surfaces are
        cleared for exactly the same reason the rain screen's are, and the old
        assertion was pinning a fact rather than a rule.

        The rule it was actually protecting is the one kept here — clearing a
        page surface is only ever correct when there IS a backdrop, because a
        transparent container over a plain themed page paints nothing and the
        form loses its background. So the case to pin is the one with the
        ambient background switched off.
        """
        from spacr.qt import preferences as prefs
        monkeypatch.setattr(prefs, "get_ambient_enabled", lambda: False)

        screen = AppScreen("measure")
        qtbot.addWidget(screen)
        assert getattr(screen, "_ambient", None) is None, \
            "the preference is off, so no backdrop should have been installed"
        assert not screen._header.property(theme.TRANSPARENT_PROPERTY)
        assert not screen._body_splitter.property(theme.TRANSPARENT_PROPERTY)

    def test_a_screen_with_the_ambient_backdrop_clears_its_surfaces(
            self, qtbot, qt_theme_applied, monkeypatch):
        """The mirror of the rain case, for every other module.

        Without this the backdrop runs, costs its frames, and reaches the eye
        only through the few pixels of layout spacing between widgets — the
        exact failure the DNA rain hit first and left a comment about.
        """
        from spacr.qt import preferences as prefs
        monkeypatch.setattr(prefs, "get_ambient_enabled", lambda: True)

        screen = AppScreen("measure")
        qtbot.addWidget(screen)
        assert getattr(screen, "_ambient", None) is not None
        assert screen._header.property(theme.TRANSPARENT_PROPERTY) is True
        assert screen._body_splitter.property(
            theme.TRANSPARENT_PROPERTY) is True


class TestThemeWallpaper:
    def test_the_opaque_themes_have_no_wallpaper_to_hand_the_rain(
            self, qapp, monkeypatch):
        from spacr.qt.screens import app_screen
        from spacr.qt import preferences
        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda: "dark")
        assert app_screen._theme_wallpaper() is None

    def test_an_image_theme_hands_over_the_file_the_qss_points_at(
            self, qapp, monkeypatch):
        from spacr.qt.screens import app_screen
        from spacr.qt import preferences
        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda: "cell")
        monkeypatch.setattr(preferences, "theme_background_path",
                            lambda name, *a: f"/tmp/{name}.jpg")
        assert app_screen._theme_wallpaper() == "/tmp/cell.jpg"

    def test_a_broken_preferences_store_is_not_fatal(self, qapp, monkeypatch):
        from spacr.qt.screens import app_screen
        from spacr.qt import preferences

        def boom():
            raise RuntimeError("no settings")
        monkeypatch.setattr(preferences, "resolve_effective_theme", boom)
        assert app_screen._theme_wallpaper() is None
