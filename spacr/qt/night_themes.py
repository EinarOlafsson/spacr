"""Ten night palettes and seven separate data-art presets.

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

The data-art catalog appends six independent palette presets. Its keys
select distinct ambient producers and existing sound sets without changing
the ten original night identities. Its explicit colour overrides retain the
night palettes' stable status hues and are checked by the same contrast and
page-separation rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

__all__ = [
    "DATA_ART_THEMES",
    "DATA_ART_THEME_KEYS",
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
    """One palette preset: its label, colours, backdrop and sound set.

    :param key: stable identifier in the theme preference and palette map.
    :param label: name shown in Preferences.
    :param description: one sentence shown as the choice's explanation.
    :param palette: the full interface palette, every role
        :func:`spacr.qt.theme.palette_for` promises.
    :param ambient: the ambient animation this theme runs behind it, one
        of :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES`.
    :param ambient_palette: the colour set that animation is painted in,
        a key of :data:`spacr.qt.widgets.ambient.PALETTE_SETS`.
    :param sound_key: existing sound-set key for data-art presets; ``None``
        retains the original night theme's matching sound key.
    """

    key: str
    label: str
    description: str
    palette: Mapping[str, str]
    ambient: str
    ambient_palette: str
    sound_key: str | None = None

    @property
    def sound(self) -> str:
        """Key of an existing sound set; night presets keep their own."""
        return self.sound_key or self.key


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


NIGHT_THEMES: Dict[str, NightTheme] = {
    theme.key: theme for theme in (
        NightTheme(
            key="lantern",
            label="Lantern",
            description=("A lit window in the dark: warm orange on near-"
                         "black, fine paper facets behind it, "
                         "and a sound set in G minor at 118 BPM."),
            palette=LANTERN_PALETTE,
            ambient="data_art_tissue_facets",
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
                         "pale gold, fine paper facets through it, and a "
                         "sound set in G Lydian at 112 BPM."),
            palette=SOLSTICE_PALETTE,
            ambient="data_art_tissue_facets",
            ambient_palette="lowsun"),
        NightTheme(
            key="undertow",
            label="Undertow",
            description=("Deep water, green-black and slow: waves travelling "
                         "through points, and a sound set in E minor at 116 BPM "
                         "with the heaviest sub of the ten."),
            palette=UNDERTOW_PALETTE,
            ambient="data_art_point_atlas",
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
            description=("Furthest from the sun: cold violet, point waves travelling "
                         "through the dark, and a tense sound set in D "
                         "harmonic minor at 128 BPM."),
            palette=APHELION_PALETTE,
            ambient="data_art_point_atlas",
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


DATA_ART_PALETTES: Dict[str, Dict[str, str]] = {
    "data_art_point_atlas": {
        **NOCTURNE_PALETTE, "bg": "#090d1c", "page": "#262c48",
        "surface": "#0f1429", "surface_alt": "#171e35",
        "surface_hi": "#222b43", "border": "#434e70",
        "border_soft": "#2b3553", "fg_muted": "#d4dbea",
        "fg_dim": "#a9b6cb", "accent": "#a6c7ff",
        "accent_hi": "#d2e2ff", "accent_lo": "#739eea",
        "accent_soft": "#273a5d", "info": "#a6c7ff",
    },
    "data_art_tissue_facets": {
        **HALCYON_PALETTE, "bg": "#12100d", "page": "#46382a",
        "surface": "#1d1812", "surface_alt": "#2a2118",
        "surface_hi": "#3d3022", "border": "#66523a",
        "border_soft": "#433522", "fg_muted": "#dfd5c6",
        "fg_dim": "#b7a994", "accent": "#f0c88b",
        "accent_hi": "#f6dbb1", "accent_lo": "#e5ae68",
        "accent_soft": "#543921", "info": "#f0c88b",
    },
    "data_art_chromatin_ribbon": {
        **VESPER_PALETTE, "bg": "#150c13", "page": "#492b3a",
        "surface": "#22121e", "surface_alt": "#301b2a",
        "surface_hi": "#432637", "border": "#6c4659",
        "border_soft": "#4c3040", "fg_muted": "#e5d1d9",
        "fg_dim": "#c0aab4", "accent": "#f4a5bf",
        "accent_hi": "#ffd1dc", "accent_lo": "#e59fb7",
        "accent_soft": "#603048", "info": "#f4a5bf",
    },
    "data_art_genetic_advection": {
        **MERIDIAN_PALETTE, "bg": "#071218", "page": "#253f48",
        "surface": "#0d1d25", "surface_alt": "#162d35",
        "surface_hi": "#213b43", "border": "#41636c",
        "border_soft": "#2d4d55", "fg_muted": "#d0dfe2",
        "fg_dim": "#a3bac0", "accent": "#7cd7d3",
        "accent_hi": "#adebe5", "accent_lo": "#67ccd1",
        "accent_soft": "#245059", "info": "#7cd7d3",
    },
    "data_art_impulse_lens": {
        **CIRRUS_PALETTE, "bg": "#0b0d0f", "page": "#303338",
        "surface": "#141619", "surface_alt": "#1e2226",
        "surface_hi": "#2b3035", "border": "#50575d",
        "border_soft": "#363d43", "fg_muted": "#dbdedb",
        "fg_dim": "#b0b8b2", "accent": "#e5e6df",
        "accent_hi": "#faf9f2", "accent_lo": "#c6cac5",
        "accent_soft": "#343b40", "info": "#e5e6df",
    },
    "data_art_fungal_growth": {
        **UNDERTOW_PALETTE, "bg": "#08130e", "page": "#254236",
        "surface": "#10231a", "surface_alt": "#193128",
        "surface_hi": "#264538", "border": "#456b56",
        "border_soft": "#315442", "fg_muted": "#d0e3d6",
        "fg_dim": "#a7c0af", "accent": "#92dfa9",
        "accent_hi": "#c0f0c9", "accent_lo": "#69c992",
        "accent_soft": "#28513a", "info": "#92dfa9",
    },
}


DATA_ART_PALETTES["data_art_thore"] = {
    **NOCTURNE_PALETTE, "bg": "#080e17", "page": "#263549",
    "surface": "#101b2c", "surface_alt": "#19283b",
    "surface_hi": "#263c51", "border": "#465f78",
    "border_soft": "#2e435b", "fg_muted": "#d4e0eb",
    "fg_dim": "#acbed0", "accent": "#b8d7eb",
    "accent_hi": "#deeff9", "accent_lo": "#8aafcf",
    "accent_soft": "#304963", "info": "#b8d7eb",
}


DATA_ART_THEMES: Dict[str, NightTheme] = {
    theme.key: theme for theme in (
        NightTheme(
            key="data_art_impulse_lens", label="spaCR field",
            description='A crisp gravitational dot field with optional local mouse influence and expanding ripples.',
            palette=DATA_ART_PALETTES["data_art_impulse_lens"],
            ambient="data_art_impulse_lens", ambient_palette="mono",
            sound_key="cirrus"),
        NightTheme(
            key="data_art_genetic_advection", label="spaCR advection",
            description='Fine particles form evolving vortices and branching currents, with optional mouse gravity.',
            palette=DATA_ART_PALETTES["data_art_genetic_advection"],
            ambient="data_art_genetic_advection", ambient_palette="ocean",
            sound_key="meridian"),
        NightTheme(
            key="data_art_fungal_growth", label="spaCR growth",
            description='A single branching front advances continuously while its trail fades, occupying at most 25% of the backdrop.',
            palette=DATA_ART_PALETTES["data_art_fungal_growth"],
            ambient="data_art_fungal_growth", ambient_palette="deepwater",
            sound_key="undertow"),
        NightTheme(
            key="data_art_thore", label="spaCR Thore",
            description='Fine background rain and branching lightning briefly illuminate the scene.',
            palette=DATA_ART_PALETTES["data_art_thore"],
            ambient="data_art_thore", ambient_palette="midnight",
            sound_key="nocturne"),
        NightTheme(
            key="data_art_point_atlas", label="spaCR waves",
            description='An edge-free landscape of round points carries wide travelling waves.',
            palette=DATA_ART_PALETTES["data_art_point_atlas"],
            ambient="data_art_point_atlas", ambient_palette="midnight",
            sound_key="nocturne"),
        NightTheme(
            key="data_art_tissue_facets", label="Tissue facets",
            description='Fine paper facets move gently and respond locally to the mouse.',
            palette=DATA_ART_PALETTES["data_art_tissue_facets"],
            ambient="data_art_tissue_facets", ambient_palette="lowsun",
            sound_key="halcyon"),
        NightTheme(
            key="data_art_chromatin_ribbon", label="Chromatin satin",
            description='Fine chromatin fibres undulate in travelling waves across folded ribbons.',
            palette=DATA_ART_PALETTES["data_art_chromatin_ribbon"],
            ambient="data_art_chromatin_ribbon", ambient_palette="dusk",
            sound_key="vesper"),
    )
}

DATA_ART_THEME_KEYS: Tuple[str, ...] = tuple(DATA_ART_THEMES)


def is_night_theme(name) -> bool:
    """True when ``name`` is one of the ten.

    :param name: a theme key, or anything at all.
    :returns: whether :data:`NIGHT_THEMES` has it.
    """
    return isinstance(name, str) and name in NIGHT_THEMES


def theme_for(name: str) -> NightTheme:
    """The night or data-art palette preset called ``name``.

    :param name: one of :data:`NIGHT_THEME_KEYS` or :data:`DATA_ART_THEME_KEYS`.
    :returns: the theme.
    :raises KeyError: if ``name`` is unknown.
    """
    if name in NIGHT_THEMES:
        return NIGHT_THEMES[name]
    return DATA_ART_THEMES[name]


def palettes() -> Dict[str, Dict[str, str]]:
    """Every night palette, by theme key.

    :returns: a fresh mapping of key to a copy of that theme's palette,
        shaped for :data:`spacr.qt.theme._PALETTES`.
    """
    return {key: dict(theme.palette) for key, theme in NIGHT_THEMES.items()}


def ambient_for(name: str) -> Tuple[str, str]:
    """The backdrop ``name`` asks for.

    :param name: one of :data:`NIGHT_THEME_KEYS` or :data:`DATA_ART_THEME_KEYS`.
    :returns: ``(animation, palette)`` — an
        :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES` name and a
        :data:`spacr.qt.widgets.ambient.PALETTE_SETS` key.
    :raises KeyError: if ``name`` is unknown.
    """
    theme = theme_for(name)
    return theme.ambient, theme.ambient_palette


def sound_for(name: str) -> str:
    """The sound set ``name`` asks for.

    :param name: one of :data:`NIGHT_THEME_KEYS` or :data:`DATA_ART_THEME_KEYS`.
    :returns: a key of :data:`spacr.qt.sound_synth.SOUND_THEMES`.
    :raises KeyError: if ``name`` is unknown.
    """
    return theme_for(name).sound
