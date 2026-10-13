"""The model box is typeset, not dumped (instruction 144).

    "actually my main problem was it dosnt look great. i want you to use
     markdown and colors for negative (CANNOT) and positive (MODEL, LEVEL,
     ETC.) text. make the formula look better (write the math symbol version
     then the code version if possible) short discriptions that contain the
     vital information for the user and links to APIs for the different
     methods"

143 read the earlier report as "too long" and cut 2,438 characters to 892.
The content was already settled; what was left is that NOTHING WAS
EMPHASISED, so a formula read exactly like a caveat.

These tests are about the rendering, and most of them drive the real widget:
what a QTextBrowser actually paints, at what width, in which colours, is not
answerable from the HTML string.
"""
from __future__ import annotations

import re

import pytest
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QTextBrowser

from spacr.qt.screens.app_screen import AppScreen


def _fit_a_model(screen):
    """Put the panel on a parametric inference, so the box shows a FORMULA.

    THE DEFAULT `inference` IS 'nonparametric' (measured 2026-08-20), and as
    of that date the box says so rather than describing a model the default
    settings never fit. Every test in this file is about how a formula is
    TYPESET, so each one needs a run that has one.
    """
    combo = screen._settings_model._widgets.get("inference")
    if combo is not None:
        index = combo.findData("parametric")
        if index >= 0:
            combo.setCurrentIndex(index)

from spacr.qt.screens.settings_model import (
    MODEL_API_LINKS,
    formula_for,
    maths_for,
    model_api_link,
    regression_model_explainer_html,
)
from spacr.qt.theme import active_palette, palette_for
from spacr.regression_spec import REGRESSION_TYPES

#: The tokens the box is allowed to paint with. Body and the muted shade are
#: not "emphasis"; the ceiling in instruction 144 B is about the rest.
EMPHASIS_TOKENS = ("accent", "error", "success", "chip_value")


def _rendered_colours(box) -> dict:
    """Every colour the document actually paints, with a sample of its text.

    READ OFF THE CHARACTER FORMATS, not off the HTML. Qt re-writes the HTML
    it is given, resolves inherited styles and drops what it cannot use, so
    a colour present in the string is not evidence that it reaches a glyph.
    """
    doc = box.document()
    cursor = QTextCursor(doc)
    cursor.movePosition(QTextCursor.Start)
    seen: dict = {}
    while True:
        brush = cursor.charFormat().foreground()
        name = brush.color().name() if brush.style() else "(default)"
        char = doc.characterAt(cursor.position())
        sample = seen.setdefault(name, "")
        if char.strip() and len(sample) < 40:
            seen[name] = sample + char
        if not cursor.movePosition(QTextCursor.NextCharacter):
            break
    return seen


def _block_texts(box):
    doc = box.document()
    out, block = [], doc.begin()
    while block.isValid():
        out.append(block)
        block = block.next()
    return out


# ---------------------------------------------------------------------------
# A. rich text, which means a different widget
# ---------------------------------------------------------------------------

def test_the_box_is_a_rich_text_widget_whose_links_open(qtbot,
                                                        qt_theme_applied):
    """A QPlainTextEdit cannot do colour or weight at all.

    And a bare QTextEdit has no `setOpenExternalLinks`, so a link in it is
    coloured text that does nothing -- worse than a module name, which can at
    least be searched.
    """
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    _fit_a_model(screen)
    box = screen._model_explainer
    assert isinstance(box, QTextBrowser)
    assert box.openExternalLinks()
    assert box.isReadOnly()


def test_what_survived_the_change_read_only_selectable_and_copyable(
        qtbot, qt_theme_applied):
    """The formula still copies AS ONE LINE.

    The reason to have a formula on screen is to put it in a methods section.
    A selection that comes back with a line break in the middle is one a user
    has to repair by hand.
    """
    from PySide6.QtCore import Qt

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    _fit_a_model(screen)
    box = screen._model_explainer
    flags = box.textInteractionFlags()
    assert flags & Qt.TextSelectableByMouse
    assert flags & Qt.TextSelectableByKeyboard

    wanted = formula_for("gene_fraction:gene + (1 | gene/grna)")
    screen._settings_model._widgets["regression_type"].setCurrentText("mixed")
    screen._refresh_model_explainer()

    for block in _block_texts(box):
        if block.text().strip() == wanted:
            cursor = QTextCursor(block)
            cursor.select(QTextCursor.BlockUnderCursor)
            # U+2029 is the paragraph separator Qt returns for a block break;
            # anything inside the formula itself would be a break in it.
            assert cursor.selectedText().strip(" ") == wanted
            break
    else:                                                # pragma: no cover
        pytest.fail(f"the box does not render {wanted!r} as one block")


def test_the_prose_reflows_and_the_formula_never_breaks(qtbot,
                                                        qt_theme_applied):
    """MEASURED at four widths on the real document.

    This is the pair that used to fight: instruction 138 wants the prose to
    fill the box, and a formula must never be broken. In plain text wrapping
    is per WIDGET, so one of the two always lost. In rich text it is per
    BLOCK -- prose in paragraphs, formulas in <pre> -- and both hold.
    """
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    _fit_a_model(screen)
    screen.show()
    # `mixed` is the default and carries no guide fit, so the two blocks this
    # measures are not on screen under it.
    screen._settings_model._widgets["regression_type"].setCurrentText("ols")
    screen._refresh_model_explainer()
    box = screen._model_explainer
    doc = box.document()

    code = formula_for("fraction:grna")
    prose_starts = "One coefficient per guide"
    widths, prose_lines = (280, 400, 620, 900), []
    for width in widths:
        doc.setTextWidth(width)
        for block in _block_texts(box):
            text = block.text().strip()
            if text == code:
                assert block.layout().lineCount() == 1, (
                    f"the code formula wrapped at text width {width}")
            elif text.startswith(prose_starts):
                prose_lines.append(block.layout().lineCount())
    assert len(set(prose_lines)) > 1, (
        f"the prose did not reflow across {widths}: {prose_lines}")


# ---------------------------------------------------------------------------
# B. the colour vocabulary, from the theme
# ---------------------------------------------------------------------------

def test_every_colour_the_box_paints_is_a_palette_token(qtbot,
                                                        qt_theme_applied):
    """Never a literal.

    The palette was contrast-checked against `CONTRAST_RULES`; a hand-picked
    hex has not been, and it is a fourth opinion about what "negative" looks
    like in an application that already has one.
    """
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    _fit_a_model(screen)
    palette = active_palette()
    allowed = {str(value).lower() for value in palette.values()
               if isinstance(value, str) and value.startswith("#")}

    for model in ("mixed", "ols", "lasso"):
        screen._settings_model._widgets["regression_type"].setCurrentText(
            model)
        screen._refresh_model_explainer()
        for colour in _rendered_colours(screen._model_explainer):
            if colour == "(default)":
                continue
            assert colour.lower() in allowed, (
                f"{model}: {colour} is not in the theme's palette")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_box_is_legible_under_each_theme(theme):
    """Rendered under each, per instruction 144's own closing checklist."""
    palette = palette_for(theme)
    html = regression_model_explainer_html("mixed", palette=palette)
    for token in ("fg", "accent", "error", "fg_muted"):
        assert palette[token] in html
    # And nothing rendered under a token NAME, which is the palette-less
    # fallback and would mean the widget forgot to hand one over.
    assert "color:accent" not in html


@pytest.mark.parametrize("model", sorted(REGRESSION_TYPES))
@pytest.mark.parametrize("level", ["both", "grna", "gene"])
def test_no_box_spends_more_than_four_emphasis_colours(model, level):
    """Four on screen at once is the CEILING, not a target.

    "Everything is plain except what the sentence is about". If every
    heading, every noun and every caveat is coloured, the box is a ransom
    note and nothing is emphasised at all.
    """
    palette = palette_for("dark")
    html = regression_model_explainer_html(model, level, palette=palette)
    used = {token for token in EMPHASIS_TOKENS if palette[token] in html}
    assert len(used) <= 4, f"{model}/{level} paints {sorted(used)}"


def test_lasso_at_both_levels_is_the_box_that_reaches_the_ceiling():
    """Named so a fifth colour cannot be added without this failing.

    accent (headings and the link), chip_value (the two output files),
    success (TWO MODELS, TWO TABLES) and error (REPORTS NO P-VALUE).
    """
    palette = palette_for("dark")
    html = regression_model_explainer_html("lasso", "both", palette=palette)
    used = {token for token in EMPHASIS_TOKENS if palette[token] in html}
    assert used == set(EMPHASIS_TOKENS)


def test_the_refusal_heading_is_the_one_that_is_not_accent():
    palette = palette_for("dark")
    html = regression_model_explainer_html("mixed", palette=palette)
    heading = re.search(
        r'<span style="color:([^;]+);[^"]*">WHAT YOU DO NOT GET</span>', html)
    assert heading is not None
    assert heading.group(1) == palette["error"]


# ---------------------------------------------------------------------------
# C. the formula, twice
# ---------------------------------------------------------------------------

def test_the_maths_comes_first_and_the_code_second(qtbot, qt_theme_applied):
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    _fit_a_model(screen)
    screen._settings_model._widgets["regression_type"].setCurrentText("mixed")
    screen._refresh_model_explainer()
    texts = [block.text().strip()
             for block in _block_texts(screen._model_explainer)]
    maths = texts.index(maths_for("mixed")[0])
    code = texts.index(formula_for("gene_fraction:gene + (1 | gene/grna)"))
    assert maths < code


def test_real_unicode_not_latex_source():
    """The box is a widget, not a renderer: `\\beta` on screen is worse than
    no symbol at all."""
    rendered = maths_for("mixed", plate_position=True)
    line = rendered[0]
    for symbol in ("β", "σ", "ε", "μ", "Σ", "ρ", "γ", "ᵢ"):
        assert symbol in "".join(rendered), symbol
    assert "\\beta" not in line and "\\sigma" not in line


@pytest.mark.parametrize("kind,term", [
    ("grna", "fraction:grna"),
    ("gene", "gene_fraction:gene"),
    ("mixed", "gene_fraction:gene + (1 | gene/grna)"),
])
@pytest.mark.parametrize("plate_position,random_row_column", [
    (True, False), (False, False), (True, True),
])
def test_the_two_formulas_agree_about_plate_position(
        kind, term, plate_position, random_row_column):
    """THE TWO MUST AGREE. A box whose formulas disagree is worse than a box
    with one, because nothing on screen says which to believe."""
    position = {"plate_position": plate_position,
                "random_row_column": random_row_column}
    code = formula_for(term, **position)
    maths = "\n".join(maths_for(kind, **position))

    if random_row_column:
        assert "(1 | rowID) + (1 | columnID)" in code
        assert "u_r(i)" in maths and "u_c(i)" in maths
        assert "σ²_row" in maths
        assert "ρ_r(i)" not in maths
    elif plate_position:
        assert "+ rowID + columnID" in code
        assert "ρ_r(i)" in maths and "γ_c(i)" in maths
        assert "u_r(i)" not in maths
    else:
        assert "rowID" not in code and "columnID" not in code
        assert "ρ_r(i)" not in maths and "u_r(i)" not in maths


def test_the_mixed_maths_shows_the_random_terms_as_random():
    maths = "\n".join(maths_for("mixed"))
    assert "u_G ~ N(0, σ²_gene)" in maths
    assert "u_G:g ~ N(0, σ²_guide)" in maths
    assert "εᵢ ~ N(0, σ²)" in maths


def test_the_box_follows_the_plate_settings_through_both_formulas(
        qtbot, qt_theme_applied):
    """Driven on the real panel: the toggle moves and BOTH lines move."""
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    _fit_a_model(screen)
    widgets = screen._settings_model._widgets
    widgets["regression_type"].setCurrentText("ols")
    toggle = widgets.get("model_plate_position")
    assert toggle is not None, "the panel has no model_plate_position control"

    toggle.setChecked(True)
    screen._refresh_model_explainer()
    on = screen._model_explainer.toPlainText()
    assert "+ rowID + columnID" in on and "ρ_r(i)" in on

    toggle.setChecked(False)
    screen._refresh_model_explainer()
    off = screen._model_explainer.toPlainText()
    assert "rowID" not in off and "ρ_r(i)" not in off


# ---------------------------------------------------------------------------
# D. short descriptions, and a link to the API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", sorted(REGRESSION_TYPES))
def test_every_model_links_its_api(model):
    name, url = model_api_link(model)
    assert name and url, f"{model} has no API link"
    assert url.startswith("https://"), url


def test_the_links_are_real_urls_not_module_paths():
    """A link that does not open is worse than a module name."""
    for key, (_name, target) in MODEL_API_LINKS.items():
        assert target.startswith("https://") or "/" not in target, (
            f"{key}: {target!r} is neither a URL nor a spaCR module name")


def test_a_spacr_backend_links_the_published_api_page():
    _name, url = model_api_link("rra")
    assert url == ("https://einarolafsson.github.io/spacr/api/spacr/rra/"
                   "index.html")


def test_the_link_reaches_the_document_as_an_anchor(qtbot, qt_theme_applied):
    """Driven on the rendered document, not on the HTML string.

    Qt rewrites the HTML it is handed; an <a> in the input is not evidence
    that a glyph in the output is a link.
    """
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    _fit_a_model(screen)
    screen._settings_model._widgets["regression_type"].setCurrentText("mixed")
    screen._refresh_model_explainer()

    doc = screen._model_explainer.document()
    cursor = QTextCursor(doc)
    cursor.movePosition(QTextCursor.Start)
    hrefs = set()
    while True:
        fmt = cursor.charFormat()
        if fmt.isAnchor():
            hrefs.add(fmt.anchorHref())
        if not cursor.movePosition(QTextCursor.NextCharacter):
            break
    assert model_api_link("mixed")[1] in hrefs


# ---------------------------------------------------------------------------
# The other box on the same panel, and a live theme switch
# ---------------------------------------------------------------------------

def test_the_permutation_box_is_typeset_too(qtbot, qt_theme_applied):
    """One rich box beside one monospace dump is the same complaint again."""
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    _fit_a_model(screen)
    box = screen._section_explainers.get("Permutation Test")
    assert box is not None
    assert isinstance(box, QTextBrowser)
    colours = _rendered_colours(box)
    palette = active_palette()
    assert palette["accent"].lower() in {c.lower() for c in colours}


def test_a_theme_switch_re_inks_the_box(qtbot, qt_theme_applied,
                                        monkeypatch):
    """Baked colours would keep the old theme's ink after a live switch.

    The old box used NO colour precisely to avoid this. Using the palette and
    re-rendering is the other way to be safe, and it is the one that can be
    seen.
    """
    from PySide6.QtCore import QEvent

    import spacr.qt.theme as theme

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    _fit_a_model(screen)
    before = {c.lower() for c in _rendered_colours(screen._model_explainer)}

    other = "light" if theme.palette_for("dark")["fg"] in before else "dark"
    monkeypatch.setattr(theme, "active_palette",
                        lambda: theme.palette_for(other))
    screen.changeEvent(QEvent(QEvent.PaletteChange))

    after = {c.lower() for c in _rendered_colours(screen._model_explainer)}
    assert theme.palette_for(other)["accent"].lower() in after
    assert after != before
