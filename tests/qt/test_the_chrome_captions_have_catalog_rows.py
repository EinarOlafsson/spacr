"""Chrome captions outside the settings fields must not stay English.

A user who picks Swedish gets Swedish setting labels, and the chrome
around them has to follow: preference tabs, theme names, resource buttons,
figure-form labels, the annotate keyboard legend.  Worse than plain
English is the half-translated caption a word-level term match produces —
"Intensity Handling (all Objekt)" is neither language.  Only an exact
catalog row beats that fallback, so these checks assert the exact row is
what the runtime returns.
"""
from __future__ import annotations

import pytest

LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")

#: One caption per surface that carries chrome text, so a regression on any
#: one of them is named rather than hidden inside a coverage count.
CHROME_CAPTIONS = (
    # Settings category headers.
    "Intensity Handling (all objects)",
    "Plate Sources & Workflow",
    "Labels & Classes",
    "Evaluation & Results",
    "Classifier",
    # Preferences: tabs, theme names, resource buttons, row labels, footer.
    "Performance",
    "Modules",
    "Logging",
    "Follow system",
    "Dark",
    "Light",
    "Glass",
    "Tubules",
    "Cytoskeleton",
    "Clear RAM",
    "Clear VRAM",
    "Clear CPU",
    "Check disk space",
    "Quit spaCR…",
    "Memory",
    "GPU memory",
    "Threads",
    "Application",
    "Extra Performance",
    "Balanced",
    "Reset to defaults",
    "Log file",
    "Deuteranopia (red-green)",
    "Protanopia (red-green)",
    "Tritanopia (blue-yellow)",
    # The hint the glass installer puts on every button-less popup.
    "press Escape to close",
    # Live preview compartment fields and common controls.
    "Min area (px²)",
    "Max area (px²)",
    "Min object area",
    "Min distance",
    "Area multiplier",
    "Perimeter fraction",
    "Min intensity pct",
    "Max intensity pct",
    "Intensity percentile",
    "Intensity threshold",
    "Intensity merge",
    "Intensity split",
    "Remove border objects",
    "Signal to noise",
    "Background",
    "Remove background",
    "Outline colour",
    "Upper percentile",
    # Figure settings form labels and its window title.
    "All text size",
    "Correct across pairs",
    "Figure title",
    "Grid axis",
    "Grid colour",
    "Grid width",
    "Height (in)",
    "Width (in)",
    "Hide top/right",
    "Legend columns",
    "Legend frame",
    "Legend position",
    "Legend text size",
    "Line style",
    "Marker size",
    "Opacity (all)",
    "Outline width (all)",
    "Palette",
    "Point size (all)",
    "Spine width",
    "Tick label size",
    "Title",
    "Unit of replication",
    "Figure settings",
    # Measure runtime card, merged-array picker, annotate empty state.
    "Crop preview",
    "Choose merged array…",
    "Open an experiment to start annotating",
    # Regression input panel drop hints.
    "Drop files or folders here, or use Add files…",
    "Drop one file here, or use Choose file…",
    # Measurement scan panel section headers.
    "Attached databases",
    # Home totals row and the list-editor chip placeholders.
    "Meas.",
    "add value",
    "add value…",
    "add text",
    "add number",
    # Channel glosses on the fixed-alphabet chips.
    "Red",
    "Green",
    "Blue",
    # Preferences captions the word map could take apart on its own.
    "Cells per montage row",
    "Annotate cells",
    "Plate heatmap",
    # The app registry name that had no row at all.
    "Cellpose Workbench",
)

#: Captions whose translation is legitimately spelled like the English, with
#: the language it is true in. Every one is a loanword or a word the two
#: languages share; anything else that comes out identical is an untranslated
#: caption hiding behind a coverage count.
SPELLED_THE_SAME = {
    "Application": {"fr"},
    "Modules": {"fr"},
    "Palette": {"de", "fr"},
    "Threads": {"de", "pt"},
    "Tubules": {"fr"},
}

#: Section headers the term fallback can decompose. They are the trap this
#: file exists for: the word map knows "objects" and "Workflow" but not the
#: whole heading, so without an exact row the user gets half of each.
DECOMPOSABLE_HEADERS = (
    "Intensity Handling (all objects)",
    "Plate Sources & Workflow",
    "Labels & Classes",
    "Evaluation & Results",
    "Min object area",
    "Remove border objects",
    "Cells per montage row",
    "Annotate cells",
    "Plate heatmap",
)


def _catalog(language):
    from spacr.qt.i18n_catalogs import _module

    return _module(language)


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_chrome_caption_resolves_to_its_catalog_row(language):
    """tr() must return the catalog row, not the English source.

    These captions sit outside the settings fields — tabs, buttons, group
    headers, placeholders — which is exactly the text the language pass had
    left behind.
    """
    from spacr.qt.i18n import tr

    catalog = _catalog(language).UI
    english = []
    for source in CHROME_CAPTIONS:
        assert source in catalog, f"{language}: {source!r} has no catalog row"
        translated = tr(source, language)
        assert translated == catalog[source], (
            f"{language}: {source!r} did not resolve to its row"
        )
        if translated == source and language not in SPELLED_THE_SAME.get(
                source, ()):
            english.append(source)
    assert not english, f"{language} still English: {english}"


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_section_header_is_never_half_translated(language):
    """An exact row must win over the word-by-word term fallback.

    ``_term_translation`` translates the words it knows and leaves the rest,
    which turned "Intensity Handling (all objects)" into "Intensity Handling
    (all Objekt)" and "Plate Sources & Workflow" into "Platta Sources &
    Arbetsflöde" — one English word in the middle of a Swedish heading.
    """
    from spacr.qt.i18n import _term_translation, tr

    mangled = []
    for source in DECOMPOSABLE_HEADERS:
        fallback = _term_translation(source, language)
        assert fallback is not None, (
            f"{language}: {source!r} no longer decomposes, so this check "
            f"has stopped proving anything"
        )
        translated = tr(source, language)
        assert translated != source, f"{language}: {source!r} is still English"
        if translated == fallback:
            mangled.append(f"{source!r} -> {translated!r}")
    assert not mangled, (
        f"{language} took the word-by-word fallback instead of an exact "
        f"row:\n  " + "\n  ".join(mangled)
    )
    assert tr("Intensity Handling (all objects)", "sv") != (
        "Intensity Handling (all Objekt)"
    )


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_maturity_hover_speaks_the_language_of_its_label(language):
    """Home shows a translated stage label over its stage sentence.

    The label ("Alpha", "Beta", "Stable") was already translated while the
    sentence the colour stands in for was not, so the hover text contradicted
    the badge above it.
    """
    from spacr.qt.i18n import tr
    from spacr.qt.theme import STAGE_NOTE

    for stage, note in STAGE_NOTE.items():
        translated = tr(note, language)
        assert translated != note, f"{language}/{stage}: still English"
        assert translated == _catalog(language).UI[note]


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_demo_entry_names_the_modules_as_the_menus_do(language):
    """The end-to-end demo entry must reuse the module names the app shows.

    Its Swedish row read "End-to-end (Task → Mät → Annotate)": one module
    renamed to a word spaCR does not use, one left in English, and a sibling
    row that had leaked the string "demoName" from the generator.
    """
    from spacr.qt.i18n import tr

    entry = tr("End-to-end (Mask → Measure → Annotate) real dataset…", language)
    for module in ("Mask", "Measure", "Annotate"):
        name = tr(module, language)
        assert name in entry, f"{language}: {name!r} missing from {entry!r}"
    assert "Task" not in entry
    assert "demoName" not in tr("End-to-end demo", language)


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_keyboard_legend_keeps_its_markup_and_its_key_names(language):
    """The annotate cheat strip is HTML, and the keys in it are not words.

    A translation that drops a ``<b>`` pair or renames ``Backspace`` stops
    describing the keyboard it documents.
    """
    from spacr.qt.i18n import tr
    from spacr.qt.screens.annotate import AnnotateScreen

    for source in (AnnotateScreen.LEGEND_COMPACT, AnnotateScreen.LEGEND_FULL):
        translated = tr(source, language)
        assert translated != source, f"{language}: legend is still English"
        assert translated.count("<b>") == source.count("<b>")
        assert translated.count("</b>") == source.count("</b>")
        assert translated.count("&nbsp;·&nbsp;") == source.count("&nbsp;·&nbsp;")
        for key in ("<b>0</b>", "<b>Space</b>", "<b>Backspace</b>",
                    "<b>u</b>", "<b>Enter</b>"):
            assert key in translated, f"{language}: {key} lost"


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_translated_caption_keeps_the_tokens_that_are_not_prose(language):
    """Paths, product names and units are data, not words to translate.

    ``measurements/measurements.db`` is where the annotator looks for its
    database and ``intercept_value`` is a setting key; a locale that
    translated either would be describing something that does not exist.
    """
    from spacr.qt.i18n import tr
    from spacr.qt.i18n_catalogs import setting_tooltip
    from spacr.qt.i18n_catalogs.en import SETTING_TOOLTIPS
    from spacr.qt.screens.annotate import AnnotateScreen

    subtitle = next(
        source for source in _catalog(language).UI
        if source.startswith("Pick a folder that contains")
    )
    assert "`measurements/measurements.db`" in tr(subtitle, language)
    assert "spaCR" in tr("Quit spaCR…", language)
    assert "Cellpose" in tr("Cellpose Workbench", language)
    assert "px²" in tr("Min area (px²)", language)

    for key in ("intercept", "intercept_value"):
        source = SETTING_TOOLTIPS[key]
        translated = setting_tooltip(key, source, language)
        assert translated and translated != source
        assert "intercept_value" in translated
    assert "patsy" in setting_tooltip(
        "intercept", SETTING_TOOLTIPS["intercept"], language)
    assert "&nbsp;·&nbsp;" in tr(AnnotateScreen.LEGEND_FULL, language)


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_picture_channel_settings_have_a_label_and_a_tooltip(language):
    """Every declared setting needs a catalog row in every language.

    The red/green/blue channel choices and the regression intercept were
    declared by the application while the catalogs had never heard of them,
    so their panels drew English labels under a translated heading.
    """
    from spacr.qt.i18n_catalogs import setting_label, setting_tooltip
    from spacr.qt.i18n_catalogs.en import SETTING_LABELS, SETTING_TOOLTIPS

    for key in ("red_channel", "green_channel", "blue_channel",
                "intercept", "intercept_value"):
        label_source = SETTING_LABELS[key]
        label = setting_label(key, label_source, language)
        assert label, f"{language}/{key}: no label row"
        # Swedish writes the statistical term exactly as English does; every
        # other pairing here has to differ or the row is not a translation.
        if (key, language) != ("intercept", "sv"):
            assert label != label_source, f"{language}/{key}: label"
        tooltip_source = SETTING_TOOLTIPS[key]
        tooltip = setting_tooltip(key, tooltip_source, language)
        assert tooltip and tooltip != tooltip_source, f"{language}/{key}: tip"


def test_a_stale_source_still_falls_back_to_english():
    """A caption whose English wording changes must not show an old row.

    The rows are bound to a hash of their English source, so an edited
    caption loses its translation instead of showing the sentence it used
    to mean.
    """
    from spacr.qt.i18n_catalogs import ui_text

    assert ui_text("press Escape to close", "sv") == "tryck på Escape för att stänga"
    assert ui_text("press Escape to close!", "sv") is None


def test_no_swedish_caption_calls_a_setting_a_chairman():
    """"Ordförande" is Swedish for chairman and belongs to no setting.

    A drafting model appended it to short technical tokens — "Gpu
    Ordförande", "btrack Ordförande" — and in five places replaced the token
    with it outright. Either way the caption names something that does not
    exist in spaCR.
    """
    from spacr.qt.i18n_catalogs import sv

    offenders = [
        f"{table}/{key!r}: {value!r}"
        for table in ("SETTING_LABELS", "SETTING_TOOLTIPS", "CATEGORY_HELP",
                      "UI", "MODULE_SUMMARIES")
        for key, value in getattr(sv, table).items()
        if "Ordförande" in value
    ]
    assert not offenders, "\n  ".join(offenders)


def test_distinct_settings_keep_distinct_swedish_labels():
    """Two settings that mean different things may not share one caption.

    A translation that collapses several short keys onto the same word
    leaves the panel showing the same row label twice, which is worse than
    English because the user cannot tell which control is which.
    """
    from spacr.qt.i18n_catalogs import sv

    labels = {key: sv.SETTING_LABELS[key]
              for key in ("eps", "pos", "neg", "gpu", "min_n")}
    assert len(set(labels.values())) == len(labels), labels
    captions = {key: sv.UI[key]
                for key in ("Radius", "organelle", "tsne", "umap", "png")}
    assert len(set(captions.values())) == len(captions), captions
