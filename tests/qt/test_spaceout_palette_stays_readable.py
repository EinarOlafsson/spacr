"""The rainbow palette the ``spaceout`` launcher dresses spaCR in.

Two things have to be true at once, and they pull against each other: the
palette has to be a *rainbow*, and text has to stay readable on it. The
construction is what reconciles them —
:func:`spacr.qt.theme.spaceout_palette` moves every role in HUE ONLY and
leaves its WCAG relative luminance where it was — so these tests measure the
same three things the shipped themes are already measured on:

* :func:`spacr.qt.theme.contrast_failures`, the check the colour-blind work
  used (``tests/qt/test_theme_blind_widgets.py`` calls it "layer 1: the
  palettes themselves are clean"), over every surface it already covers;
* :func:`spacr.qt.theme.page_separation_failures` — can you see the *panel*,
  which the ratio alone cannot answer down at the black end;
* :func:`spacr.qt.theme.scrim_failures` — can you still see the *wallpaper*
  through an image theme's translucent panels.

They also assert the theme contract is untouched, because every screen in
the application reads it: ``THEMES`` still names four themes,
``resolve_effective_theme`` still answers one of them, and a light start is
still light.
"""
from __future__ import annotations

import pytest

from spacr.qt import theme


@pytest.fixture
def dressed():
    """Run the test in the spaceout dressing and take it off afterwards.

    The mode is process state and this suite is randomly ordered, so a
    leaked dressing would re-colour every test that ran after it.
    """
    was = theme.spaceout_enabled()
    theme.enable_spaceout()
    yield
    if not was:
        theme.disable_spaceout()


# ---------------------------------------------------------------------------
# Readability — the check the colour-blind work used
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", theme.THEMES)
def test_every_dressed_palette_clears_every_contrast_rule(name, dressed):
    """The requirement, measured rather than asserted about.

    :data:`spacr.qt.theme.CONTRAST_RULES` is 33 rules over five page
    surfaces plus the accent and button pairs, and an image theme's surfaces
    are judged through their scrim over the brightest thing that theme's
    wallpaper can put behind them. All of it, in the rainbow.
    """
    assert theme.contrast_failures(name) == []


@pytest.mark.parametrize("name", theme.THEMES)
def test_the_panels_still_separate_from_the_page(name, dressed):
    """The other legibility question: not "can you read it" but "can you see
    it". Judged in CIE L*, at full opacity and at 60 %."""
    assert theme.page_separation_failures(name) == []


@pytest.mark.parametrize("name", theme.IMAGE_THEMES)
def test_the_wallpaper_still_reads_through_the_panels(name, dressed):
    assert theme.scrim_failures(name) == []


def test_the_scrims_are_re_solved_and_that_is_what_makes_the_last_one_pass():
    """Keeping the undressed alphas is not close enough, and this is why.

    Contrast survives a re-hue by construction — the ratio is a function of
    relative luminance, and the luminances do not move. A *translucent*
    surface does not: an image theme composites its panels over the
    wallpaper channel by channel in sRGB, and two colours of equal luminance
    and different hue do not composite to equal luminance. So the alphas
    have to be solved again for the new colours.

    Put the undressed alphas back under the dressing and the wallpaper stops
    reading through the panels — 3 roles on Cell, 4 on Glass, down to
    1.39:1 against a 1.50:1 rule. That is the bug this re-solve prevents,
    reproduced here rather than described.
    """
    was = theme.spaceout_enabled()
    theme.disable_spaceout()
    undressed = {name: dict(alphas)
                 for name, alphas in theme.SCRIM_ALPHA.items()}
    theme.enable_spaceout()
    try:
        assert all(theme.scrim_failures(name) == []
                   for name in theme.IMAGE_THEMES)
        solved = {name: dict(alphas)
                  for name, alphas in theme.SCRIM_ALPHA.items()}
        assert solved != undressed, \
            "the dressing did not re-solve the scrims at all"
        theme.SCRIM_ALPHA.clear()
        theme.SCRIM_ALPHA.update(undressed)
        stale = [row for name in theme.IMAGE_THEMES
                 for row in theme.scrim_failures(name)]
        theme.SCRIM_ALPHA.clear()
        theme.SCRIM_ALPHA.update(solved)
    finally:
        if not was:
            theme.disable_spaceout()
    assert stale, ("the undressed alphas pass under the dressing, so this "
                   "re-solve is not doing anything and the claim above is "
                   "wrong")


# ---------------------------------------------------------------------------
# Why it passes: hue moves, luminance does not
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", theme.THEMES)
def test_every_surface_keeps_the_luminance_it_had(name):
    """The identity the whole design rests on.

    Exact in the reals; on screen it is bounded by 8-bit rounding, which
    :func:`spacr.qt.theme._hue_shift` minimises by choosing the closest
    representable colour on the hue line. 0.006 is a fifth of the smallest
    step the darkest palette makes between two surfaces.

    :data:`spacr.qt.theme.SPACEOUT_INK_ROLES` are excluded and that is not a
    weakening of the rule — it is the rule being applied where it belongs.
    Preserving luminance is how a SURFACE stays as readable as it was; for
    the three ink roles it is what stopped the body text from carrying a hue
    at all, because a role already at the top of the scale has no room to
    move in. Those three are solved against the same contrast rules instead
    of carried, and ``test_spaceout_looks_alive.py`` asserts both halves of
    that: they moved, and they still clear every rule at every point on the
    drift.
    """
    was = theme.spaceout_enabled()
    theme.disable_spaceout()
    plain = theme.palette_for(name)
    theme.enable_spaceout()
    try:
        rainbow = theme.palette_for(name)
        assert set(rainbow) == set(plain), "the dressing changed the keys"
        drift = {
            role: abs(theme.relative_luminance(rainbow[role])
                      - theme.relative_luminance(plain[role]))
            for role in plain if role not in theme.SPACEOUT_INK_ROLES
        }
    finally:
        if not was:
            theme.disable_spaceout()
    worst = max(drift, key=drift.get)
    assert drift[worst] < 0.006, f"{worst} moved by {drift[worst]:.4f}"


@pytest.mark.parametrize("name", theme.THEMES)
def test_the_hue_table_covers_every_role_the_palette_carries(name):
    """A role nobody re-hued is passed through unchanged, which is safe and
    is not the intention. This is what makes a new palette role fail loudly
    here instead of showing up as one grey box in a rainbow."""
    missing = sorted(set(theme.palette_for(name))
                     - set(theme.SPACEOUT_HUES)
                     - set(theme._splash_roles(theme.palette_for(name))))
    assert missing == [], f"{name} has roles with no spaceout hue: {missing}"


def test_it_really_is_a_rainbow(dressed):
    """The other half of the ask, and it needs asserting as much as the
    readability does: a palette that passed every contrast rule by not
    changing anything would pass every test above."""
    surfaces = ("bg", "page", "surface", "surface_alt", "surface_hi",
                "accent", "success", "warning", "error")
    hues = set()
    for name in theme.THEMES:
        palette = theme.palette_for(name)
        for role in surfaces:
            red, green, blue = theme._channels(palette[role], (0, 0, 0))
            if max(red, green, blue) - min(red, green, blue) < 12:
                # A role at the very top or bottom of the luminance scale
                # cannot carry a hue at all — white is white.
                continue
            hues.add(round(_hue_of(red, green, blue) / 30.0))
    assert len(hues) >= 8, \
        f"only {len(hues)} distinct hue families across the palettes: {hues}"


def _hue_of(red: int, green: int, blue: int) -> float:
    """Hue in degrees, from 8-bit sRGB."""
    import colorsys
    return colorsys.rgb_to_hsv(red / 255.0, green / 255.0,
                               blue / 255.0)[0] * 360.0


# ---------------------------------------------------------------------------
# The theme contract does not change
# ---------------------------------------------------------------------------

def test_the_four_themes_are_still_the_four_themes(dressed):
    """spaceout re-hues whichever theme was resolved. It does not become a
    fifth one, which is what would break every screen that reads the
    resolver."""
    assert theme.THEMES == ("dark", "light", "cell", "glass")
    from spacr.qt.preferences import PALETTE_THEMES, resolve_effective_theme
    assert resolve_effective_theme() in PALETTE_THEMES


def test_light_is_still_light_and_dark_is_still_dark(dressed):
    """The light/dark handling every screen reads goes on working: the
    light page is still far lighter than the dark one, and the ink on each
    is still on the right side of it."""
    light = theme.palette_for("light")
    dark = theme.palette_for("dark")
    assert (theme.relative_luminance(light["page"])
            > theme.relative_luminance(dark["page"]) + 0.4)
    assert (theme.relative_luminance(light["fg"])
            < theme.relative_luminance(light["page"]))
    assert (theme.relative_luminance(dark["fg"])
            > theme.relative_luminance(dark["page"]))


def test_taking_the_dressing_off_restores_the_palettes_exactly():
    """Because the suite is randomly ordered and because ``spaceout`` is a
    launcher, not a mode the application can end up half in."""
    was = theme.spaceout_enabled()
    theme.disable_spaceout()
    before = {name: theme.palette_for(name) for name in theme.THEMES}
    alphas = {name: dict(rows) for name, rows in theme.SCRIM_ALPHA.items()}
    theme.enable_spaceout()
    assert theme.palette_for("dark") != before["dark"]
    theme.disable_spaceout()
    try:
        assert {name: theme.palette_for(name)
                for name in theme.THEMES} == before
        assert {name: dict(rows)
                for name, rows in theme.SCRIM_ALPHA.items()} == alphas
    finally:
        if was:
            theme.enable_spaceout()
