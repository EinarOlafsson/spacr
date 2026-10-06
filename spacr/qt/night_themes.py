"""Ten night palettes and twelve separate data-art presets.

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

The data-art catalog appends twelve independent palette presets. Its keys
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


#: The ten, in menu order: the hue wheel from a lit window through gold,
#: green, teal and blue to violet and back round to rose.
#:
#: NO PAIR REPEATS A BACKDROP. Ten themes over seven animations means an
#: animation is reused; what is not reused is the pair, so no two themes
#: put the same picture on the screen. Three producers carry two themes
#: each and four carry one, and where a producer is reused its two
#: themes are painted in different colour sets: ``blobs`` in low sun and
#: dusk, ``aurora`` in ocean and dusk, ``drift`` in ocean and midnight.
#: What stays unique is the pair, not the producer.
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
    "data_art_spatial_strata": {
        **CIRRUS_PALETTE, "bg": "#0d1011", "page": "#343b3b",
        "surface": "#151a1b", "surface_alt": "#202829",
        "surface_hi": "#2e3839", "border": "#536061",
        "border_soft": "#384445", "fg_muted": "#d8ddda",
        "fg_dim": "#abb7b2", "accent": "#d7dad4",
        "accent_hi": "#f0f1e9", "accent_lo": "#b0beb8",
        "accent_soft": "#344240", "info": "#d7dad4",
    },
    "data_art_molecular_helix": {
        **MERIDIAN_PALETTE, "bg": "#07141a", "page": "#23434a",
        "surface": "#0c2024", "surface_alt": "#143037",
        "surface_hi": "#1e3d44", "border": "#3e6770",
        "border_soft": "#285057", "fg_muted": "#cfe2e2",
        "fg_dim": "#a3bcbd", "accent": "#9ae7e9",
        "accent_hi": "#c9f3ef", "accent_lo": "#76d8db",
        "accent_soft": "#245257", "info": "#9ae7e9",
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
    "data_art_sequence_matrix": {
        **SOLSTICE_PALETTE, "bg": "#0b150f", "page": "#2c4937",
        "surface": "#122119", "surface_alt": "#1b3023",
        "surface_hi": "#284231", "border": "#45684f",
        "border_soft": "#31523b", "fg_muted": "#d4e1d2",
        "fg_dim": "#a7bda7", "accent": "#c7e3a4",
        "accent_hi": "#e2f1c8", "accent_lo": "#aad484",
        "accent_soft": "#35533a", "info": "#c7e3a4",
    },
    "data_art_transcript_rain": {
        **UNDERTOW_PALETTE, "bg": "#071511", "page": "#244336",
        "surface": "#0e2219", "surface_alt": "#173128",
        "surface_hi": "#224135", "border": "#3e6653",
        "border_soft": "#2b4e3d", "fg_muted": "#d0e2d7",
        "fg_dim": "#a5bbad", "accent": "#8ee5bd",
        "accent_hi": "#bff3d8", "accent_lo": "#62d1a1",
        "accent_soft": "#25503a", "info": "#8ee5bd",
    },
    "data_art_regulatory_circuit": {
        **LANTERN_PALETTE, "bg": "#120c09", "page": "#463024",
        "surface": "#1e140e", "surface_alt": "#2c1f16",
        "surface_hi": "#3e2b1e", "border": "#69503c",
        "border_soft": "#493322", "fg_muted": "#e0d3c8",
        "fg_dim": "#bba899", "accent": "#e7ad79",
        "accent_hi": "#f7d2ae", "accent_lo": "#dc9c66",
        "accent_soft": "#54341e", "info": "#e7ad79",
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
    "data_art_interference": {
        **APHELION_PALETTE, "bg": "#160f19", "page": "#403044",
        "surface": "#211625", "surface_alt": "#302138",
        "surface_hi": "#3c2c45", "border": "#614e68",
        "border_soft": "#44334b", "fg_muted": "#e0d7e2",
        "fg_dim": "#b8acbd", "accent": "#dbc8e6",
        "accent_hi": "#f2e4f4", "accent_lo": "#c7a5d7",
        "accent_soft": "#4c3555", "info": "#dbc8e6",
    },
    "data_art_morphogenesis": {
        **SOLSTICE_PALETTE, "bg": "#151410", "page": "#414631",
        "surface": "#201f17", "surface_alt": "#2c3021",
        "surface_hi": "#3b402e", "border": "#606847",
        "border_soft": "#414a33", "fg_muted": "#e0dfce",
        "fg_dim": "#b8baa1", "accent": "#d6dea0",
        "accent_hi": "#edf0c7", "accent_lo": "#c3d39a",
        "accent_soft": "#4b5033", "info": "#d6dea0",
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
}


DATA_ART_THEMES: Dict[str, NightTheme] = {
    theme.key: theme for theme in (
        NightTheme(
            key="data_art_point_atlas", label="Spatial point atlas",
            description="A finely sampled three-dimensional point landscape with depth and cursor-driven parallax.",
            palette=DATA_ART_PALETTES["data_art_point_atlas"],
            ambient="data_art_point_atlas", ambient_palette="midnight",
            sound_key="nocturne"),
        NightTheme(
            key="data_art_tissue_facets", label="Tissue facets",
            description="A crystalline tissue mosaic of shaded geometric facets, with slowly changing local relief.",
            palette=DATA_ART_PALETTES["data_art_tissue_facets"],
            ambient="data_art_tissue_facets", ambient_palette="lowsun",
            sound_key="halcyon"),
        NightTheme(
            key="data_art_spatial_strata", label="Spatial strata",
            description="Fine stacked topographic layers form a moving spatial relief with precise depth and contour detail.",
            palette=DATA_ART_PALETTES["data_art_spatial_strata"],
            ambient="data_art_spatial_strata", ambient_palette="mono",
            sound_key="cirrus"),
        NightTheme(
            key="data_art_molecular_helix", label="Molecular helix",
            description="A rotating molecular helix of shaded beads and paired bases, with perspective and depth.",
            palette=DATA_ART_PALETTES["data_art_molecular_helix"],
            ambient="data_art_molecular_helix", ambient_palette="ocean",
            sound_key="meridian"),
        NightTheme(
            key="data_art_chromatin_ribbon", label="Chromatin satin",
            description="Folded satin-like chromatin ribbons carry fine fibres through soft, interwoven surfaces.",
            palette=DATA_ART_PALETTES["data_art_chromatin_ribbon"],
            ambient="data_art_chromatin_ribbon", ambient_palette="dusk",
            sound_key="vesper"),
        NightTheme(
            key="data_art_sequence_matrix", label="Genome mosaic",
            description="A layered genome mosaic of tiny encoded tiles shifts through an architectural sequence field.",
            palette=DATA_ART_PALETTES["data_art_sequence_matrix"],
            ambient="data_art_sequence_matrix", ambient_palette="fluor",
            sound_key="solstice"),
        NightTheme(
            key="data_art_transcript_rain", label="Transcript rain",
            description="Fine falling transcription marks stream through a layered field of genetic information.",
            palette=DATA_ART_PALETTES["data_art_transcript_rain"],
            ambient="data_art_transcript_rain", ambient_palette="deepwater",
            sound_key="undertow"),
        NightTheme(
            key="data_art_regulatory_circuit", label="Regulatory circuit",
            description="An etched regulatory circuit routes pulses through precise orthogonal paths and small control nodes.",
            palette=DATA_ART_PALETTES["data_art_regulatory_circuit"],
            ambient="data_art_regulatory_circuit", ambient_palette="ember",
            sound_key="lantern"),
        NightTheme(
            key="data_art_genetic_advection", label="Genetic advection",
            description="Thousands of fine genetic-flow particles move through a continuous wind-like field that bends near the cursor.",
            palette=DATA_ART_PALETTES["data_art_genetic_advection"],
            ambient="data_art_genetic_advection", ambient_palette="ocean",
            sound_key="meridian"),
        NightTheme(
            key="data_art_interference", label="Perturbation interference",
            description="Smooth interference waves form a changing pearlescent field, distorted locally by the cursor.",
            palette=DATA_ART_PALETTES["data_art_interference"],
            ambient="data_art_interference", ambient_palette="pastel",
            sound_key="aphelion"),
        NightTheme(
            key="data_art_morphogenesis", label="Morphogenesis",
            description="A fine organic pattern of changing spots and labyrinths evokes the emergence of biological structure.",
            palette=DATA_ART_PALETTES["data_art_morphogenesis"],
            ambient="data_art_morphogenesis", ambient_palette="borealis",
            sound_key="solstice"),
        NightTheme(
            key="data_art_impulse_lens", label="Perturbation lens",
            description="A precision dot lattice bends around moving impulses and the cursor, revealing local perturbation.",
            palette=DATA_ART_PALETTES["data_art_impulse_lens"],
            ambient="data_art_impulse_lens", ambient_palette="mono",
            sound_key="cirrus"),
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
    :raises KeyError: if ``name`` is not one of the ten.
    """
    theme = theme_for(name)
    return theme.ambient, theme.ambient_palette


def sound_for(name: str) -> str:
    """The sound set ``name`` asks for.

    :param name: one of :data:`NIGHT_THEME_KEYS` or :data:`DATA_ART_THEME_KEYS`.
    :returns: a key of :data:`spacr.qt.sound_synth.SOUND_THEMES`.
    :raises KeyError: if ``name`` is not one of the ten.
    """
    return theme_for(name).sound
