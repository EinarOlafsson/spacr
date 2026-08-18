"""The Model & Inference explainer box (instruction 132).

Two halves:

* the CLAIM the box makes about the old formula is measured against the real
  code, so the box cannot go on asserting something that stopped being true;
* the BOX ITSELF is driven through the real regression screen.

Instruction 132's first sentence is the requirement -- "it is important for
the user to know all of this" -- so these tests are about what the box SAYS,
not merely that a widget exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import (
    COLLINEAR_FORMULA,
    GENE_FORMULA,
    GRNA_FORMULA,
    MIXED_FORMULA,
    REGRESSION_LEVELS,
    explainer_width,
    normalise_regression_level,
    regression_model_explainer,
)
from spacr.regression_spec import NO_P_VALUE_TYPES, REGRESSION_TYPES


def _flat(text: str) -> str:
    """The box's text with its hard wrapping collapsed.

    The prose is wrapped to a fixed column, so a phrase the box states can be
    split across two lines. Asserting on the wrapped text would be asserting
    on the wrap.
    """
    return " ".join(str(text).split())


def _pooled_screen_frame(seed: int = 0):
    """A pooled-screen frame shaped like the real thing.

    ONE ROW PER (well, guide), and each well holds only a few of the library's
    guides -- which is what a pooled screen is. A fully-crossed frame, where
    every well holds every guide, is NOT rank deficient under the old formula,
    so a test built on one would have quietly proved the opposite of the
    finding. The maintainer's TSG101 screen is 1945 rows over 610 wells, 823
    guides and 389 genes: about 3.2 guides per well.
    """
    rng = np.random.default_rng(seed)
    genes = [f"g{i:03d}" for i in range(40)]
    guides = {g: [f"{g}_{k}" for k in range(3)] for g in genes}
    library = [(g, s) for g in genes for s in guides[g]]

    rows = []
    for w in range(200):
        prc = f"plate1_r{w // 20 + 1}_c{w % 20 + 1}"
        picks = rng.choice(len(library), size=4, replace=False)
        fractions = rng.dirichlet(np.ones(4))
        for pick, frac in zip(picks, fractions):
            gene, grna = library[pick]
            rows.append(dict(
                prc=prc, plateID="plate1",
                rowID=f"r{w // 20 + 1}", columnID=f"c{w % 20 + 1}",
                gene=gene, grna=grna, fraction=float(frac),
                cell_count=int(rng.integers(30, 400)),
                pred=float(rng.normal()),
            ))
    return pd.DataFrame(rows)


def _rank(formula, frame):
    from patsy import dmatrices
    _, design = dmatrices(formula, data=frame, return_type="dataframe")
    matrix = design.to_numpy(dtype=float)
    return matrix.shape[1], int(np.linalg.matrix_rank(matrix))


# ---------------------------------------------------------------------------
# The claim the box makes, measured
# ---------------------------------------------------------------------------

def test_gene_fraction_is_the_sum_of_the_genes_guide_fractions():
    """The box says so, and every other claim in it follows from this one."""
    from spacr.ml import check_and_clean_data

    clean = check_and_clean_data(_pooled_screen_frame(), "pred")
    per_guide = clean[["prc", "gene", "grna", "fraction"]].drop_duplicates()
    summed = per_guide.groupby(["prc", "gene"], observed=True)["fraction"].sum()
    stored = clean.groupby(["prc", "gene"], observed=True)["gene_fraction"].first()

    assert np.allclose(summed.reindex(stored.index).to_numpy(),
                       stored.to_numpy())


def test_the_old_joint_formula_is_rank_deficient_and_one_level_at_a_time_is_not():
    """THE FINDING THE WHOLE INSTRUCTION RESTS ON.

    `gene_fraction` is the sum of the gene's guide fractions, so in the old
    joint design each gene column is a linear combination of that gene's own
    guide columns. The design is therefore rank deficient BY CONSTRUCTION --
    statsmodels does not refuse, it pseudo-inverts and returns one arbitrary
    solution out of infinitely many.

    Measured on the maintainer's TSG101 frame (610 wells, 823 guides, 389
    genes) the old design is 1248 parameters at rank 862 -- 386 short -- and
    all 386 are accounted for: 102 single-guide genes whose gene column is
    IDENTICAL to their guide column, plus 284 multi-guide genes whose gene
    column lies exactly in the span of their own guide columns. The three
    remaining multi-guide genes are exactly the three with a well holding two
    of their guides. Its shipped results.csv has 244480 and 244480_3 both at
    coefficient 3.389291, p = 2.873149e-13.
    """
    from spacr.ml import check_and_clean_data

    clean = check_and_clean_data(_pooled_screen_frame(), "pred")

    joint_params, joint_rank = _rank(COLLINEAR_FORMULA.replace("y ~", "pred ~"),
                                     clean)
    assert joint_rank < joint_params, (
        "the old joint design came back full rank, which would mean the "
        "premise of instruction 132 no longer holds")

    # Each level ALONE is identifiable on the very same wells. This is the
    # box's promise, and the reason the two-fit split is a fix rather than a
    # trade.
    for formula in (GRNA_FORMULA, GENE_FORMULA):
        params, rank = _rank(formula.replace("y ~", "pred ~"), clean)
        assert rank == params, f"{formula} was not full rank"


def test_a_mixed_fit_gives_no_guide_level_p_values():
    """The box's costliest claim: guides come back as BLUPs, not coefficients.

    A user who takes the new default and goes looking for guide p-values is
    who the "WHAT YOU DO NOT GET" section is for, so the claim is measured
    rather than asserted.
    """
    statsmodels_api = pytest.importorskip("statsmodels.formula.api")

    rng = np.random.default_rng(1)
    rows = []
    for gene in [f"G{i}" for i in range(6)]:
        for guide in range(3):
            for _ in range(12):
                rows.append(dict(gene=gene, grna=f"{gene}_{guide}",
                                 plateID="p1",
                                 gene_fraction=float(rng.uniform(0.05, 0.4)),
                                 y=float(rng.normal())))
    frame = pd.DataFrame(rows)

    fit = statsmodels_api.mixedlm(
        "y ~ gene_fraction:gene", frame, groups=frame["plateID"],
        re_formula="1",
        vc_formula={"gene": "0 + C(gene)", "grna": "0 + C(grna)"},
    ).fit()

    # No guide appears among the parameters that carry a p-value...
    tested = [str(name) for name in fit.pvalues.index]
    assert not any(name.startswith("G") and "_" in name for name in tested)
    # ...and the random effects come back as bare predictions, with no
    # standard error and no p-value anywhere on the fit.
    blups = fit.random_effects
    assert blups and all(isinstance(v, pd.Series) for v in blups.values())
    assert not any(hasattr(fit, attr)
                   for attr in ("re_pvalues", "re_bse", "random_effects_pvalues"))


# ---------------------------------------------------------------------------
# What the box says
# ---------------------------------------------------------------------------

def test_the_mixed_box_states_the_formula_and_what_it_costs():
    text = _flat(regression_model_explainer("mixed"))

    assert MIXED_FORMULA in text
    # The gene is fixed, the guide is random and nested. That is the whole
    # point of choosing it.
    assert "FIXED effect" in text and "RANDOM effect" in text
    assert "nested inside its gene" in text

    # WHAT IT COSTS -- the paragraph the box exists for.
    assert "BLUP" in text
    assert "NOT coefficients with standard errors and p-values" in text
    assert "NO GUIDE-LEVEL HIT LIST" in text
    # ...and where to go instead, by name.
    assert "level='grna'" in text


def test_the_mixed_box_does_not_promise_a_guide_level_correction():
    """It corrects the gene table; there is no guide family to correct."""
    text = _flat(regression_model_explainer("mixed"))
    correction = text.split("MULTIPLE TESTING")[1].split("WHY THE FORMULA")[0]
    assert "no second family" in correction
    assert "guide effects are not tested" in correction


@pytest.mark.parametrize("level,expected,absent", [
    ("grna", GRNA_FORMULA, GENE_FORMULA),
    ("gene", GENE_FORMULA, GRNA_FORMULA),
])
def test_a_single_level_box_states_only_that_levels_formula(level, expected,
                                                            absent):
    text = _flat(regression_model_explainer("ols", level))
    assert expected in text
    assert absent not in text


def test_level_both_states_two_fits_two_tables_and_not_one_design():
    text = _flat(regression_model_explainer("ols", "both"))
    assert GRNA_FORMULA in text and GENE_FORMULA in text
    assert "results_grna.csv" in text and "results_gene.csv" in text
    assert "TWO MODELS, TWO TABLES" in text
    # The distinction that matters: two fits, NOT one design holding both.
    assert "NOT one design containing both" in text
    # Each corrected on its own, and why pooling would be wrong.
    assert "OWN multiple-testing family" in text
    assert "sum of the guide regressors" in text


def test_every_backend_the_pipeline_can_fit_has_its_own_paragraph():
    """A backend added to spacr.ml must not land here without a description.

    Instruction 132: "for each of the other models textbox descriptions also
    describe what each mode does." This fails for any type that reaches the
    dropdown without one.
    """
    generic = "spaCR has no description for this model"
    seen = {}
    for name in ("auto",) + tuple(REGRESSION_TYPES):
        text = _flat(regression_model_explainer(name))
        assert generic not in text, f"{name} has no paragraph of its own"
        assert f"MODEL: {name}" in text
        if name == "mixed":
            body = text.split("WHAT IS MODELLED")[1]
        else:
            body = text.split(f"WHAT {name.upper()} DOES")[1]
        body = body.split("MULTIPLE TESTING")[0].split("WHY THE FORMULA")[0]
        assert len(body.split()) > 25, f"{name}'s paragraph is a stub"
        seen[name] = body

    # rlm and huber are the same estimator and share a paragraph; nothing
    # else may be a copy of another backend's description.
    shared = {"rlm", "huber"}
    bodies = {n: b for n, b in seen.items() if n not in shared}
    assert len(set(bodies.values())) == len(bodies), (
        "two backends are described with identical text")


@pytest.mark.parametrize("name", sorted(NO_P_VALUE_TYPES))
def test_a_backend_with_no_p_value_is_not_described_as_bh_corrected(name):
    """The box must not contradict itself one paragraph later.

    lasso and elasticnet rank by bootstrap selection frequency and report no
    p-value, so "BH-corrected" under them would be false -- and it was, until
    this was written: the MULTIPLE TESTING line said "the single fit is
    BH-corrected as one family" directly beneath a paragraph stating the
    backend reports no p-value at all.
    """
    for level in REGRESSION_LEVELS:
        text = _flat(regression_model_explainer(name, level))
        correction = text.split("MULTIPLE TESTING")[1].split("WHY THE")[0]
        assert "BH-corrected" not in correction
        assert "NOTHING TO BH-CORRECT" in correction
        assert "selection frequency is not a false-discovery rate" in correction


def test_every_box_carries_the_reason_the_formula_changed():
    """The collinearity finding is in the box, for every selection."""
    for name in ("auto", "mixed", "ols", "ridge", "hinge"):
        text = _flat(regression_model_explainer(name))
        assert COLLINEAR_FORMULA in text
        assert "perfectly collinear BY CONSTRUCTION" in text
        assert "SUM of that gene's gRNA fractions" in text
        # It names what statsmodels does instead of refusing, which is why
        # the bug was silent.
        assert "pseudo-inverse" in text
        # ...and the evidence from the real screen.
        assert "244480" in text and "3.389291" in text


def test_the_box_never_offers_the_collinear_formula_as_something_it_fits():
    """It appears ONLY under "WHY THE FORMULA CHANGED", never as a FORMULA."""
    for name in ("auto",) + tuple(REGRESSION_TYPES):
        for level in REGRESSION_LEVELS:
            rendered = regression_model_explainer(name, level)
            before_history = rendered.split("WHY THE FORMULA CHANGED")[0]
            assert COLLINEAR_FORMULA not in before_history
            assert "fraction:grna + gene_fraction:gene" not in _flat(
                before_history)


def test_unknown_levels_fall_back_to_both_rather_than_raising():
    """`level` is new: an older settings CSV has no value for it."""
    for junk in (None, "", "GRNA ", "nonsense", 7):
        assert normalise_regression_level(junk) in REGRESSION_LEVELS
    assert normalise_regression_level("GRNA ") == "grna"
    assert normalise_regression_level("nonsense") == "both"


def test_an_unfittable_model_name_is_not_given_an_invented_formula():
    text = _flat(regression_model_explainer("gls"))
    assert "no description for this model" in text
    assert "y ~" not in text


def test_only_the_formulas_are_wider_than_the_wrap_column():
    """The box does not soft-wrap, so an over-long line needs scrolling.

    Prose must therefore fit the column outright. The formulas are the
    deliberate exception: the indented mixed formula is 62 characters, it is
    one line, and breaking it to fit would defeat the point of the box.
    """
    width = explainer_width()
    for name in ("auto",) + tuple(REGRESSION_TYPES):
        for level in REGRESSION_LEVELS:
            for line in regression_model_explainer(name, level).splitlines():
                if line.strip().startswith("y ~"):
                    continue
                assert len(line) <= width, (
                    f"{name}/{level} has a {len(line)}-character line that "
                    f"will need horizontal scrolling: {line!r}")


# ---------------------------------------------------------------------------
# The box on the real screen
# ---------------------------------------------------------------------------

def test_the_regression_screen_shows_a_readonly_selectable_monospace_box(
        qtbot, qt_theme_applied):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFontInfo

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen.show()

    box = screen._model_explainer
    assert box is not None, "the regression panel built no explainer"
    assert box.isReadOnly()
    # Selectable: a formula you cannot copy into a methods section is a
    # formula you retype.
    flags = box.textInteractionFlags()
    assert flags & Qt.TextSelectableByMouse
    assert flags & Qt.TextSelectableByKeyboard

    # THE FONT AS RENDERED, not as requested. The theme's global
    # `QWidget { font-family: "Open Sans", ... }` rule beats setFont(), so a
    # box that only calls setFont(systemFont(FixedFont)) comes out
    # PROPORTIONAL -- measured 'Open Sans', fixedPitch False -- and the
    # formulas do not align. Asserting on box.font() would have passed
    # against exactly that bug, because font() returns what was requested.
    assert QFontInfo(box.font()).fixedPitch(), (
        f"the explainer renders in {QFontInfo(box.font()).family()!r}, "
        f"which is not a fixed-pitch font")
    assert box.toPlainText().startswith("MODEL: ")


def test_the_box_sits_directly_under_the_model_and_inference_section(
        qtbot, qt_theme_applied):
    """Under the controls it explains, not adrift at the foot of the form."""
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    box = screen._model_explainer

    layout = box.parentWidget().layout()
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    position = order.index(box)
    previous = order[position - 1]
    # Section.title() upper-cases the header text it was given.
    assert previous.title().lower() == "model & inference"


def test_the_box_text_is_legible_against_its_own_background(qtbot,
                                                            qt_theme_applied):
    """The glyphs must actually be readable where they are painted.

    MEASURED ON THE COMPOSITED SCREEN, not on a widget grab. `box.grab()`
    records the box's transparent background AS transparent pixels, which an
    image viewer then shows as white; against light text that looks exactly
    like white-on-white and is purely an artefact of the grab. Compositing the
    whole screen puts the themed page behind the text, which is what the user
    actually sees.

    The comparison blanks the text by replacing every non-space character with
    a space, so the line count and the longest line -- and therefore the
    scrollbar -- are identical between the two renders. Clearing the text
    instead removes the scrollbar, and that alone moves enough dark pixels to
    satisfy any contrast threshold while proving nothing about the glyphs.
    """
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen.resize(1400, 1000)
    for section in screen._settings_sections:
        section.set_expanded(section.title().lower() == "model & inference")
    screen.show()

    box = screen._model_explainer
    box.setMinimumHeight(500)
    qtbot.waitUntil(lambda: box.height() >= 500 and box.width() > 200)

    written = box.toPlainText()
    with_text = screen.grab().toImage()
    box.setPlainText("\n".join(
        "".join(ch if ch.isspace() else " " for ch in line)
        for line in written.splitlines()))
    blank = screen.grab().toImage()
    box.setPlainText(written)

    assert with_text.size() == blank.size()

    def luminance(pixel):
        return (0.2126 * ((pixel >> 16) & 0xFF)
                + 0.7152 * ((pixel >> 8) & 0xFF)
                + 0.0722 * (pixel & 0xFF))

    # Only the box's own rectangle can differ, so measure inside it.
    origin = box.mapTo(screen, box.rect().topLeft())
    changed = 0
    sampled = 0
    strongest = 0.0
    for y in range(origin.y(), origin.y() + box.height(), 2):
        for x in range(origin.x(), origin.x() + box.width(), 2):
            sampled += 1
            inked, empty = with_text.pixel(x, y), blank.pixel(x, y)
            if inked != empty:
                changed += 1
                strongest = max(strongest,
                                abs(luminance(inked) - luminance(empty)))

    assert changed / sampled > 0.03, (
        f"only {changed / sampled:.4%} of the box changes when its glyphs are "
        f"blanked; the text is not painting")
    assert strongest > 60, (
        "the text is painting in a colour too close to its background to read")


def test_the_box_collapses_with_the_section_it_explains(qtbot,
                                                        qt_theme_applied):
    """It must not dangle under a section the user has collapsed."""
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    box = screen._model_explainer
    section = next(s for s in screen._settings_sections
                   if s.title().lower() == "model & inference")
    screen.show()

    section.set_expanded(True)
    assert box.isVisibleTo(box.parentWidget())
    section.header().setChecked(False)
    assert not box.isVisibleTo(box.parentWidget())
    section.header().setChecked(True)
    assert box.isVisibleTo(box.parentWidget())


def test_other_modules_do_not_get_the_regression_explainer(qtbot,
                                                           qt_theme_applied):
    for app_key in ("mask", "measure", "umap"):
        screen = AppScreen(app_key)
        qtbot.addWidget(screen)
        assert screen._model_explainer is None


def test_the_box_follows_the_regression_type_dropdown(qtbot, qt_theme_applied):
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    box = screen._model_explainer
    combo = screen._settings_model._widgets["regression_type"]

    combo.setCurrentText("ols")
    assert GRNA_FORMULA in _flat(box.toPlainText())
    assert MIXED_FORMULA not in _flat(box.toPlainText())

    combo.setCurrentText("mixed")
    mixed_text = _flat(box.toPlainText())
    assert MIXED_FORMULA in mixed_text
    assert "NO GUIDE-LEVEL HIT LIST" in mixed_text

    combo.setCurrentText("quantile")
    quantile_text = _flat(box.toPlainText())
    assert "CONDITIONAL QUANTILE" in quantile_text
    assert "NOT the mean" in quantile_text


def test_the_box_follows_the_panels_own_level_dropdown(qtbot, qt_theme_applied):
    """The real `level` control on the real panel, not an injected stand-in."""
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    box = screen._model_explainer
    screen._settings_model._widgets["regression_type"].setCurrentText("ols")
    level = screen._settings_model._widgets["level"]
    assert [level.itemText(i) for i in range(level.count())] == list(
        REGRESSION_LEVELS)

    level.setCurrentText("grna")
    text = _flat(box.toPlainText())
    assert GRNA_FORMULA in text and GENE_FORMULA not in text

    level.setCurrentText("gene")
    text = _flat(box.toPlainText())
    assert GENE_FORMULA in text and GRNA_FORMULA not in text

    level.setCurrentText("both")
    text = _flat(box.toPlainText())
    assert GRNA_FORMULA in text and GENE_FORMULA in text


def test_choosing_mixed_makes_the_box_ignore_whatever_level_holds(
        qtbot, qt_theme_applied):
    """`level` is greyed out under mixed, so the box must not quote it.

    A stale "LEVEL: grna" line beside a mixed formula would say the run was
    about to fit something it is not.
    """
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    settings = screen._settings_model

    settings._widgets["regression_type"].setCurrentText("ols")
    settings._widgets["level"].setCurrentText("grna")
    assert "LEVEL: grna" in screen._model_explainer.toPlainText()

    settings._widgets["regression_type"].setCurrentText("mixed")
    text = screen._model_explainer.toPlainText()
    assert "LEVEL: not applicable" in text
    assert "LEVEL: grna" not in text
    assert not settings._widgets["level"].isEnabled()


def test_the_box_still_renders_when_the_panel_has_no_level_control(
        qtbot, qt_theme_applied):
    """Older settings, and any panel built without the new control.

    `level` reached the regression panel during this instruction; the box
    must not depend on its presence, so the read path is driven with the
    control removed.
    """
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._settings_model._widgets["regression_type"].setCurrentText("ols")
    screen._settings_model._widgets.pop("level")

    screen._refresh_model_explainer()
    text = _flat(screen._model_explainer.toPlainText())
    assert "LEVEL: both" in text
    assert GRNA_FORMULA in text and GENE_FORMULA in text
