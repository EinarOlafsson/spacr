"""The loading screen painted its own colours instead of naming them.

Instruction 74 asked for the splash green to go into the palette so it lives
where every other colour lives. It had stayed a literal in the widget --
`INSTALLER_GREEN = "#003737"` plus four `QColor(255, 255, 255, N)` calls
inline in `paintEvent` -- which meant the one colour that must match the
installer icon was the one colour nobody would find when looking for it.

The colour changed with the move (instruction 78). It was `#003737`, sampled
from the installer icon, and it read as teal because it IS teal -- a very
dark cyan-green at hue 180 -- which made the first thing the application
shows the one full-window surface with a colour cast. It now takes the
THEME'S OWN window background: black on the dark theme, as asked, and
identical to the window that replaces it, so the handover has nothing to
flash. The ink follows for the reason it must -- white on the light theme's
near-white surface would be invisible.

Every lookup carries the literal it replaced. The splash is the first thing
painted, sometimes before the theme is resolved, and a palette lookup that
raised would replace it with a traceback.
"""

import pytest

from spacr.qt.theme import THEMES, palette_for
from spacr.qt.widgets import loading_screen as module

TEAL = "#003737"          # what it used to be
SPLASH_ROLES = ("splash_bg", "splash_ink", "splash_ink_dim",
                "splash_track", "splash_fill")


# ---------------------------------------------------------------------------
# the roles exist and do not vary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role", SPLASH_ROLES)
def test_the_role_is_in_every_palette(role):
    for theme in THEMES:
        assert role in palette_for(theme), (theme, role)


def test_the_teal_is_gone_from_every_theme():
    """The reported complaint: "the background is teal"."""
    for theme in THEMES:
        assert palette_for(theme)["splash_bg"].lower() != TEAL


def test_the_dark_theme_splash_is_black():
    """"i want it to be black or dark gray"."""
    assert palette_for("dark")["splash_bg"].lower() == "#000000"


def test_the_splash_is_the_colour_of_the_window_behind_it():
    """A splash one shade off the window makes the handover flash.

    The loading screen exists so the transition is invisible, so matching
    the window is not a tidiness point -- it is the requirement.
    """
    for theme in THEMES:
        palette = palette_for(theme)
        assert palette["splash_bg"] == palette["bg"], theme
        assert palette["splash_ink"] == palette["fg"], theme


def test_the_ink_is_readable_on_every_background():
    """White ink on the light theme's near-white surface is the failure."""
    from spacr.qt.theme import _composite, _contrast

    for theme in THEMES:
        palette = palette_for(theme)
        lit = _contrast(_composite(palette["splash_ink"],
                                   palette["splash_bg"], 255),
                        palette["splash_bg"])
        assert lit >= 7.0, (theme, lit)


def test_the_dim_weight_is_solved_not_fixed():
    """A fixed alpha is not a fixed contrast.

    Alpha 110 read at 3.04:1 on the dark theme and 2.31:1 on the light one:
    the same number, one side of the floor each.
    """
    from spacr.qt.theme import _contrast, splash_dim_alpha

    for theme in THEMES:
        palette = palette_for(theme)
        ratio = _contrast(palette["splash_ink_dim"], palette["splash_bg"])
        assert ratio >= 3.0, (theme, ratio)
        # Dim enough to still read as unreached.
        assert palette["splash_ink_dim"] != palette["splash_ink"], theme
        assert splash_dim_alpha(palette["splash_ink"],
                                palette["splash_bg"]) < 255, theme


def test_every_palette_value_is_still_hex():
    """The palette contract: #rrggbb, no rgba() and no integers.

    The alpha roles are flattened against the splash background instead --
    exact, because that is the only surface they are ever painted on.
    """
    for theme in THEMES:
        for key, value in palette_for(theme).items():
            assert isinstance(value, str), (theme, key, value)
            assert value.startswith("#") and len(value) == 7, (theme, key,
                                                               value)


# ---------------------------------------------------------------------------
# the widget reads them
# ---------------------------------------------------------------------------

def test_the_module_no_longer_holds_the_colour_as_a_literal():
    import inspect

    source = inspect.getsource(module.LoadingScreen.paintEvent)
    assert "QColor(255, 255, 255" not in source, (
        "a white is inlined in paintEvent again; take it from the palette")
    assert TEAL not in source


def test_the_constant_is_renamed_and_the_old_name_still_works():
    """`INSTALLER_GREEN` described neither the old colour nor the new one."""
    assert module.SPLASH_BACKGROUND.lower() == "#000000"
    assert module.INSTALLER_GREEN == module.SPLASH_BACKGROUND


def test_the_painted_colours_come_from_the_palette():
    dark = palette_for("dark")
    assert module._role_color("splash_bg").name().lower() == "#000000"
    assert module._role_color("splash_ink").name().lower() == "#ffffff"
    for role in ("splash_ink_dim", "splash_track", "splash_fill"):
        assert module._role_color(role).name().lower() == dark[role].lower()


@pytest.mark.parametrize("alpha,expected", [
    (-10, 0), (0, 0), (128, 128), (255, 255), (900, 255)])
def test_the_ink_alpha_is_clamped_rather_than_wrapping(alpha, expected):
    assert module._ink(alpha).alpha() == expected


# ---------------------------------------------------------------------------
# the splash must never be the thing that fails
# ---------------------------------------------------------------------------

def test_an_unreachable_palette_falls_back_to_the_literal(monkeypatch):
    """The splash paints before the theme is necessarily resolved."""
    import spacr.qt.theme as theme

    def explode(*args, **kwargs):
        raise RuntimeError("no palette yet")

    monkeypatch.setattr(theme, "palette_for", explode)
    assert module._role("splash_bg", "#000000") == "#000000"
    assert module._role_color("splash_bg").name().lower() == "#000000"
    assert module._ink(200).getRgb() == (255, 255, 255, 200)


def test_a_missing_role_falls_back_rather_than_painting_nothing(monkeypatch):
    import spacr.qt.theme as theme

    monkeypatch.setattr(theme, "palette_for", lambda *a, **k: {})
    assert module._role("splash_bg", "#000000") == "#000000"


@pytest.mark.parametrize("spec,expected", [
    ("rgba(1, 2, 3, 4)", (1, 2, 3, 4)),
    ("rgba(1, 2, 3)", (1, 2, 3, 255)),
    ("#003737", (0, 55, 55, 255)),
])
def test_both_colour_spellings_parse(spec, expected):
    from PySide6.QtGui import QColor

    assert module._rgba(spec, QColor("#000000")).getRgb() == expected


@pytest.mark.parametrize("spec", ["rgba(nonsense)", "rgba(1, 2)", "not a colour"])
def test_an_unparseable_colour_gives_the_fallback(spec):
    from PySide6.QtGui import QColor

    assert module._rgba(spec, QColor("#000000")).name().lower() == "#000000"


# ---------------------------------------------------------------------------
# it still paints
# ---------------------------------------------------------------------------

def test_the_screen_paints_without_raising(qtbot):
    screen = module.LoadingScreen(total=7)
    qtbot.addWidget(screen)
    screen.resize(640, 400)
    screen.advance(3)
    shot = screen.grab()          # forces a real paintEvent

    assert not shot.isNull()
    assert (shot.width(), shot.height()) == (640, 400)
    # The background it painted is the role's colour, read back off the
    # pixels rather than off the palette that was asked for.
    corner = shot.toImage().pixelColor(2, 2)
    assert corner.name().lower() == palette_for()["splash_bg"].lower()
