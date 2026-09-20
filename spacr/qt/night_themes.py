"""The ten night themes, each a palette, a backdrop and a sound set.

A night theme is one choice that moves three things at once: the colours
the interface is painted in (:data:`spacr.qt.theme.THEMES` grows by ten
because of this module), which ambient animation runs behind it and in
which colours (:mod:`spacr.qt.widgets.ambient`), and which set of sounds
spaCR plays if the user has ever switched sound on
(:mod:`spacr.qt.sound_synth`). Choosing one in Preferences therefore
changes how spaCR looks *and* how it sounds, which is the whole point of
the family.

This module is Qt-free on purpose, the same way
:mod:`spacr.qt.sound_synth` is: :mod:`spacr.qt.preferences` imports it to
validate a stored theme name and must not pull QtGui or QtWidgets in
while doing so. It holds data and three lookups and imports nothing from
the rest of spaCR, so :mod:`spacr.qt.theme`,
:mod:`spacr.qt.widgets.ambient` and :mod:`spacr.qt.sound_synth` can all
read from it without a cycle.

THE PALETTES ARE LITERALS RATHER THAN A RECIPE. Each one was generated
from a hue and a saturation over a shared lightness ramp, then checked
against :func:`spacr.qt.theme.contrast_failures` and
:func:`spacr.qt.theme.page_separation_failures` and written down. Keeping
the numbers here rather than the recipe means the published colours are
the reviewed colours, no solver runs at import, and
``tests/qt/test_ten_night_themes.py`` judges what actually ships.

THREE COLOURS WERE MOVED OFF THE RECIPE BY MEASUREMENT, and they are the
only three. Nocturne's and Aphelion's ``accent_lo`` failed ``bg`` against
``accent_lo`` at 4.5:1 undressed — a deep blue and a deep violet are simply
too dark at the recipe's lightness — and Vesper's failed it at 4.49:1 at
one of the sixty hue offsets the ``spaceout`` dressing can reach. All three
were raised until they passed, and nothing else was touched. See
``docs/notes/spacr/qt/night_themes.md``.

THREE ROLES ARE THE SAME IN ALL TEN, and deliberately: ``success``,
``warning`` and ``error``. A red that means *this run failed* must not
become a theme decision, so the status hues do not follow the palette
around the colour wheel. ``fg`` is white in all ten for the same reason a
night palette wants it: every surface here is dark.

:data:`NIGHT_THEMES` is ordered by hue, warm through cool and back round
to rose, so the Theme menu reads as one family rather than ten unrelated
entries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

__all__ = [
    "NIGHT_THEMES",
    "NIGHT_THEME_KEYS",
    "NightTheme",
    "ambient_for",
    "is_night_theme",
    "palettes",
    "sound_for",
    "theme_for",
]


@dataclass(frozen=True)
class NightTheme:
    """One night theme: what it is called, and the three things it moves.

    :param key: stable identifier. It is the persisted ``prefs/theme``
        value, the key in :data:`spacr.qt.theme.THEMES`, and the key of
        the matching :class:`spacr.qt.sound_synth.SoundTheme`, so the
        three stay in step by construction rather than by a table.
    :param label: name shown in Preferences.
    :param description: one sentence shown as the choice's explanation.
    :param palette: the full interface palette, every role
        :func:`spacr.qt.theme.palette_for` promises.
    :param ambient: the ambient animation this theme runs behind it, one
        of :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES`.
    :param ambient_palette: the colour set that animation is painted in,
        a key of :data:`spacr.qt.widgets.ambient.PALETTE_SETS`.
    """

    key: str
    label: str
    description: str
    palette: Mapping[str, str]
    ambient: str
    ambient_palette: str

    @property
    def sound(self) -> str:
        """Key of this theme's sound set. It is the theme's own key."""
        return self.key


LANTERN_PALETTE = {
    "bg":          "#0e0904",
    "page":        "#3e2f20",
    "surface":     "#181008",
    "surface_alt": "#25190d",
    "surface_hi":  "#382817",
    "border":      "#5c4733",
    "border_soft": "#3b2d1e",
    "fg":          "#ffffff",
    "fg_muted":    "#d7cfc9",
    "fg_dim":      "#afa39a",
    "accent":      "#fba96a",
    "accent_hi":   "#fbcca8",
    "accent_lo":   "#f0791e",
    "accent_soft": "#4b2b13",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#fba96a",
}

HALCYON_PALETTE = {
    "bg":          "#0d0b05",
    "page":        "#3b3423",
    "surface":     "#16130a",
    "surface_alt": "#231d0f",
    "surface_hi":  "#352d1a",
    "border":      "#584e37",
    "border_soft": "#383121",
    "fg":          "#ffffff",
    "fg_muted":    "#d5d1ca",
    "fg_dim":      "#ada79c",
    "accent":      "#f6d46f",
    "accent_hi":   "#f7e4ab",
    "accent_lo":   "#e8b826",
    "accent_soft": "#493c15",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#f6d46f",
}

SOLSTICE_PALETTE = {
    "bg":          "#0b0c06",
    "page":        "#343925",
    "surface":     "#13150b",
    "surface_alt": "#1d2111",
    "surface_hi":  "#2d321d",
    "border":      "#4f553a",
    "border_soft": "#323623",
    "fg":          "#ffffff",
    "fg_muted":    "#d2d4cb",
    "fg_dim":      "#a8ab9e",
    "accent":      "#e2d583",
    "accent_hi":   "#ece5b6",
    "accent_lo":   "#ccb943",
    "accent_soft": "#413c1d",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#e2d583",
}

UNDERTOW_PALETTE = {
    "bg":          "#050d0a",
    "page":        "#223c33",
    "surface":     "#091612",
    "surface_alt": "#0f231c",
    "surface_hi":  "#1a352b",
    "border":      "#36594c",
    "border_soft": "#203930",
    "fg":          "#ffffff",
    "fg_muted":    "#cad6d1",
    "fg_dim":      "#9cada7",
    "accent":      "#7ee7bd",
    "accent_hi":   "#b3efd7",
    "accent_lo":   "#3cd296",
    "accent_soft": "#1b4333",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#7ee7bd",
}

MERIDIAN_PALETTE = {
    "bg":          "#040c0d",
    "page":        "#213a3e",
    "surface":     "#081617",
    "surface_alt": "#0d2224",
    "surface_hi":  "#183437",
    "border":      "#34565b",
    "border_soft": "#1f373a",
    "fg":          "#ffffff",
    "fg_muted":    "#c9d5d6",
    "fg_dim":      "#9bacae",
    "accent":      "#7bdbea",
    "accent_hi":   "#b2e8f0",
    "accent_lo":   "#38c1d7",
    "accent_soft": "#1a3f44",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#7bdbea",
}

CIRRUS_PALETTE = {
    "bg":          "#07090b",
    "page":        "#283037",
    "surface":     "#0c1014",
    "surface_alt": "#131a1f",
    "surface_hi":  "#1f2930",
    "border":      "#3d4952",
    "border_soft": "#252e34",
    "fg":          "#ffffff",
    "fg_muted":    "#ccd0d3",
    "fg_dim":      "#9fa5aa",
    "accent":      "#8bbcda",
    "accent_hi":   "#bad6e8",
    "accent_lo":   "#4e95c0",
    "accent_soft": "#20333e",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#8bbcda",
}

NOCTURNE_PALETTE = {
    "bg":          "#05060d",
    "page":        "#22253c",
    "surface":     "#090b16",
    "surface_alt": "#0f1123",
    "surface_hi":  "#1a1d35",
    "border":      "#363a59",
    "border_soft": "#202339",
    "fg":          "#ffffff",
    "fg_muted":    "#cacbd6",
    "fg_dim":      "#9c9ead",
    "accent":      "#7185f4",
    "accent_hi":   "#acb7f6",
    "accent_lo":   "#596feb",
    "accent_soft": "#161e48",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#7185f4",
}

APHELION_PALETTE = {
    "bg":          "#09050c",
    "page":        "#2e233b",
    "surface":     "#0f0a16",
    "surface_alt": "#181022",
    "surface_hi":  "#271b34",
    "border":      "#463857",
    "border_soft": "#2c2238",
    "fg":          "#ffffff",
    "fg_muted":    "#cfcbd5",
    "fg_dim":      "#a49dac",
    "accent":      "#b775f0",
    "accent_hi":   "#d3aef4",
    "accent_lo":   "#a255e5",
    "accent_soft": "#311847",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#b775f0",
}

PULSAR_PALETTE = {
    "bg":          "#0c060c",
    "page":        "#3a2439",
    "surface":     "#160a15",
    "surface_alt": "#221021",
    "surface_hi":  "#341c32",
    "border":      "#563855",
    "border_soft": "#372236",
    "fg":          "#ffffff",
    "fg_muted":    "#d5cbd4",
    "fg_dim":      "#ac9dab",
    "accent":      "#e87dda",
    "accent_hi":   "#f0b3e7",
    "accent_lo":   "#d43ac0",
    "accent_soft": "#441b3e",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#e87dda",
}

VESPER_PALETTE = {
    "bg":          "#0c0608",
    "page":        "#3a252c",
    "surface":     "#150a0e",
    "surface_alt": "#211117",
    "surface_hi":  "#331c24",
    "border":      "#563944",
    "border_soft": "#37232a",
    "fg":          "#ffffff",
    "fg_muted":    "#d5cbcf",
    "fg_dim":      "#ac9da3",
    "accent":      "#ee779f",
    "accent_hi":   "#f3afc6",
    "accent_lo":   "#dd336c",
    "accent_soft": "#461828",
    "success":     "#66d68f",
    "chip_class":  "#6ed8d1",
    "chip_value":  "#66d68f",
    "warning":     "#eec14f",
    "error":       "#ff7a70",
    "info":        "#ee779f",
}


#: The ten, in menu order: the hue wheel from a lit window through gold,
#: green, teal and blue to violet and back round to rose.
#:
#: NO PAIR REPEATS A BACKDROP. Ten themes over seven animations means an
#: animation is reused; what is not reused is the pair, so no two themes
#: put the same picture on the screen. ``blobs`` carries three of them
#: because it is the one producer that reads as *nebula* at every
#: saturation, and the three colour sets it carries are as far apart as
#: the family goes.
NIGHT_THEMES: Dict[str, NightTheme] = {
    theme.key: theme for theme in (
        NightTheme(
            key="lantern",
            label="Lantern",
            description=("A lit window in the dark: warm orange on near-"
                         "black, out-of-focus lights drifting behind it, "
                         "and a sound set in G minor at 118 BPM."),
            palette=LANTERN_PALETTE,
            ambient="bokeh",
            ambient_palette="ember"),
        NightTheme(
            key="halcyon",
            label="Halcyon",
            description=("Gold haze over a brown-black room, nebula fields "
                         "moving slowly through it, and a sound set in F "
                         "Dorian at 120 BPM."),
            palette=HALCYON_PALETTE,
            ambient="blobs",
            ambient_palette="lowsun"),
        NightTheme(
            key="solstice",
            label="Solstice",
            description=("The long low light of a year's turn: olive and "
                         "pale gold, cells drifting through it, and a "
                         "sound set in G Lydian at 112 BPM."),
            palette=SOLSTICE_PALETTE,
            ambient="cells",
            ambient_palette="lowsun"),
        NightTheme(
            key="undertow",
            label="Undertow",
            description=("Deep water, green-black and slow: rings spreading "
                         "and fading, and a sound set in E minor at 116 BPM "
                         "with the heaviest sub of the ten."),
            palette=UNDERTOW_PALETTE,
            ambient="ripple",
            ambient_palette="deepwater"),
        NightTheme(
            key="meridian",
            label="Meridian",
            description=("A teal horizon line with curtains of light "
                         "folding above it, and a sound set in A Dorian at "
                         "124 BPM."),
            palette=MERIDIAN_PALETTE,
            ambient="aurora",
            ambient_palette="ocean"),
        NightTheme(
            key="cirrus",
            label="Cirrus",
            description=("Thin high cloud: pale ice blue, barely coloured, "
                         "a starfield behind it, and the most open sound "
                         "set of the ten, in B minor at 126 BPM."),
            palette=CIRRUS_PALETTE,
            ambient="drift",
            ambient_palette="ocean"),
        NightTheme(
            key="nocturne",
            label="Nocturne",
            description=("Indigo, a starfield and the slowest sound set of "
                         "the ten — C sharp minor at 108 BPM, long decays "
                         "and almost no drum."),
            palette=NOCTURNE_PALETTE,
            ambient="drift",
            ambient_palette="midnight"),
        NightTheme(
            key="aphelion",
            label="Aphelion",
            description=("Furthest from the sun: cold violet, sand settling "
                         "on a vibrating plate, and a tense sound set in D "
                         "harmonic minor at 128 BPM."),
            palette=APHELION_PALETTE,
            ambient="resonance",
            ambient_palette="midnight"),
        NightTheme(
            key="pulsar",
            label="Pulsar",
            description=("Magenta on black with nebula fields turning "
                         "through it, and the hardest-pumping sound set of "
                         "the ten, in F sharp minor at 128 BPM."),
            palette=PULSAR_PALETTE,
            ambient="blobs",
            ambient_palette="dusk"),
        NightTheme(
            key="vesper",
            label="Vesper",
            description=("The last colour in the sky: plum and dusty rose, "
                         "curtains of light folding overhead, and a soft "
                         "sound set in E flat minor at 114 BPM."),
            palette=VESPER_PALETTE,
            ambient="aurora",
            ambient_palette="dusk"),
    )
}

#: The ten keys, in menu order.
NIGHT_THEME_KEYS: Tuple[str, ...] = tuple(NIGHT_THEMES)


def is_night_theme(name) -> bool:
    """True when ``name`` is one of the ten.

    :param name: a theme key, or anything at all.
    :returns: whether :data:`NIGHT_THEMES` has it.
    """
    return isinstance(name, str) and name in NIGHT_THEMES


def theme_for(name: str) -> NightTheme:
    """The :class:`NightTheme` called ``name``.

    :param name: one of :data:`NIGHT_THEME_KEYS`.
    :returns: the theme.
    :raises KeyError: if ``name`` is not one of the ten.
    """
    return NIGHT_THEMES[name]


def palettes() -> Dict[str, Dict[str, str]]:
    """Every night palette, by theme key.

    :returns: a fresh mapping of key to a copy of that theme's palette,
        shaped for :data:`spacr.qt.theme._PALETTES`.
    """
    return {key: dict(theme.palette) for key, theme in NIGHT_THEMES.items()}


def ambient_for(name: str) -> Tuple[str, str]:
    """The backdrop ``name`` asks for.

    :param name: one of :data:`NIGHT_THEME_KEYS`.
    :returns: ``(animation, palette)`` — an
        :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES` name and a
        :data:`spacr.qt.widgets.ambient.PALETTE_SETS` key.
    :raises KeyError: if ``name`` is not one of the ten.
    """
    theme = NIGHT_THEMES[name]
    return theme.ambient, theme.ambient_palette


def sound_for(name: str) -> str:
    """The sound set ``name`` asks for.

    :param name: one of :data:`NIGHT_THEME_KEYS`.
    :returns: a key of :data:`spacr.qt.sound_synth.SOUND_THEMES`.
    :raises KeyError: if ``name`` is not one of the ten.
    """
    return NIGHT_THEMES[name].sound
