"""The Model & Inference box is a box a reader finishes (instruction 143 B).

    "the text box has too much text it should read more like this:"
    -- and what followed was the box's OWN CURRENT TEXT, pasted back, which
    is the clearest possible statement that the SHAPE is right and the VOLUME
    is wrong.

MEASURED before the cut: `ols`/`both` was 2,438 characters over 29 lines, and
880 of those characters were one block -- WHY THE FORMULA CHANGED -- describing
a design no shipped version fits. Instruction 138 is why it became visible:
the prose was hard-wrapped to 54 columns until the day before and now fills
the width, so the same words read as a denser block.

Two properties, and they pull against each other, which is why they are
tested together:

* the box is SHORT -- what a reader needs on every visit, and nothing they
  have already read once;
* the removed history is still READABLE, with every figure intact, and the
  box says where it went. Deleting the measurement is the one outcome to
  avoid: it is the evidence for why this module fits one level at a time.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: What `ols`/`both` may cost. The instruction's target, held as a hard
#: number: 892 characters when this was written, from 2,438.
OLS_BOTH_CEILING = 900

#: What the WORST selection may cost.
#:
#: RAISED FROM 1,400 ON 2026-08-18 (instruction 140), and the reason is worth
#: keeping because raising a ceiling is normally how one stops meaning
#: anything. `mixed` gained a WHAT IT COSTS section -- the measured 54x/67x
#: against ols, that the optimiser is single-threaded, and which model to
#: reach for instead -- and it took the longest box from `group_lasso`/`both`
#: at 1,303 to `mixed`/anything at 1,795.
#:
#: That is not the volume 143 cut. 143 removed a block describing a design no
#: shipped version fits; this one describes the cost of the DEFAULT, which
#: every user pays and which two people in a row could only discover by
#: waiting. The ceiling still bites: at 1,900 the next paragraph has to
#: justify itself the same way.
WORST_CEILING = 1900

#: And `mixed` specifically, so the section that cost the 400 characters
#: cannot quietly grow past what it bought.
MIXED_CEILING = 1850


def _every_explainer():
    from spacr.qt.screens.settings_model import (REGRESSION_LEVELS,
                                                 regression_model_explainer)
    from spacr.regression_spec import REGRESSION_TYPES

    for family in ("auto",) + tuple(REGRESSION_TYPES):
        for level in REGRESSION_LEVELS:
            yield family, level, regression_model_explainer(family, level)


def _flat(text: str) -> str:
    return " ".join(str(text).split())


def test_the_box_the_maintainer_measured_is_under_the_target():
    from spacr.qt.screens.settings_model import regression_model_explainer

    text = regression_model_explainer("ols", "both")
    assert len(text) < OLS_BOTH_CEILING, (
        f"ols/both is {len(text)} characters; it was 2,438 and the target is "
        f"under about {OLS_BOTH_CEILING}")
    # LINES, not only characters: 29 was the count that made it a wall.
    assert len(text.splitlines()) <= 24


def test_no_selection_anywhere_goes_back_to_being_a_wall():
    """Nineteen backends and three levels; the box renders for all of them."""
    worst = max(_every_explainer(), key=lambda item: len(item[2]))
    family, level, text = worst
    assert len(text) < WORST_CEILING, (
        f"{family}/{level} is the longest box at {len(text)} characters")


def test_the_cost_section_is_what_made_mixed_the_longest_box():
    """The one box that grew, and by how much, held as a number.

    `mixed` is the DEFAULT (132), so this is the box everybody reads. The
    section it gained is the measured cost of taking that default; the
    ceiling is set just above it so a fourth paragraph has to argue for
    itself rather than arriving.
    """
    from spacr.qt.screens.settings_model import regression_model_explainer

    text = regression_model_explainer("mixed")
    assert "WHAT IT COSTS" in text
    assert len(text) < MIXED_CEILING, (
        f"the mixed box is {len(text)} characters")
    # And every other box is still under what 143 left them at.
    for family, level, other in _every_explainer():
        if family == "mixed":
            continue
        assert len(other) < 1400, f"{family}/{level} is {len(other)}"


def test_what_a_reader_needs_every_time_is_still_there():
    """The SHAPE the maintainer approved, kept whole while the volume fell."""
    from spacr.qt.screens.settings_model import (GENE_FORMULA, GRNA_FORMULA,
                                                 regression_model_explainer)

    text = regression_model_explainer("ols", "both")
    flat = _flat(text)

    assert "MODEL: ols -- ordinary least squares" in flat
    assert "LEVEL: both" in flat
    # Each formula, with the file it writes.
    assert (f"FORMULA (guide fit)  ->  results_grna.csv\n"
            f"    {GRNA_FORMULA}") in text
    assert (f"FORMULA (gene fit)   ->  results_gene.csv\n"
            f"    {GENE_FORMULA}") in text
    # ONE sentence per formula, saying what a coefficient IS.
    assert "One coefficient per guide, the unit the screen measures." in flat
    assert "One coefficient per gene, from the summed guide fraction." in flat
    assert "TWO MODELS, TWO TABLES" in flat
    assert "WHAT OLS DOES" in flat


@pytest.mark.parametrize("family,level", [("ols", "both"), ("mixed", "both"),
                                          ("rra", "grna"), ("lasso", "gene")])
def test_a_description_is_one_or_two_sentences(family, level):
    """"WHAT <MODEL> DOES, at one or two sentences" -- `rra` was six.

    Sentences are counted on the terminal full stop, so the abbreviations and
    decimals inside these paragraphs (`0.5`, `1.345`, `1e-6`, `alpha='auto'`,
    `group_lasso.max_lambda`) are not miscounted as sentence ends.
    """
    from spacr.qt.screens.settings_model import regression_model_explainer

    heading, ends = (("WHAT IS MODELLED", "WHAT YOU DO NOT GET")
                     if family == "mixed"
                     else (f"WHAT {family.upper()} DOES", "MULTIPLE TESTING"))
    flat = _flat(regression_model_explainer(family, level))
    body = flat.split(heading)[1].split(ends)[0].strip()
    sentences = [part for part in body.split(". ") if part.strip()]
    assert len(sentences) <= 2, f"{family}: {len(sentences)} sentences"


def test_multiple_testing_is_one_sentence_where_it_used_to_be_five():
    """The four that followed were read-once, and one belonged elsewhere.

    "a gene called by the gene fit AND its guides called by the guide fit is
    two tests of one hypothesis" is a caution for the moment somebody makes
    the claim -- beside the hit list -- not for every time the panel opens.
    """
    from spacr.qt.screens.settings_model import regression_model_explainer

    flat = _flat(regression_model_explainer("ols", "both"))
    correction = flat.split("MULTIPLE TESTING")[1].split("WHY THE")[0].strip()
    assert correction == ("Each fit is its OWN multiple-testing family and is "
                          "BH-corrected within itself.")
    assert "two tests of one hypothesis" not in flat


def test_the_history_is_out_of_every_box():
    """Not one selection still carries the retired design or its numbers."""
    from spacr.qt.screens.settings_model import COLLINEAR_FORMULA

    for family, level, text in _every_explainer():
        flat = _flat(text)
        assert COLLINEAR_FORMULA not in flat, f"{family}/{level}"
        assert "1248 parameters" not in flat, f"{family}/{level}"
        assert "pseudo-inverse" not in flat, f"{family}/{level}"
        assert "3.389291" not in flat, f"{family}/{level}"


def test_every_box_says_where_the_history_went():
    from spacr.qt.screens.settings_model import _HISTORY_POINTER

    for family, level, text in _every_explainer():
        assert text.rstrip().endswith(_HISTORY_POINTER), f"{family}/{level}"
    assert "regression_model_explainer.__doc__" in _HISTORY_POINTER


def test_the_pointer_cannot_be_the_line_that_wraps():
    """It is not a formula, so nothing else protects it.

    :func:`explainer_width` sets the box's minimum width from the longest
    UNBREAKABLE line. It measures every plate-position state, including the
    longer random-effects formula that appears after the user opts in.
    """
    from spacr.qt.screens.settings_model import (MIXED_TERM, _HISTORY_POINTER,
                                                 explainer_width, formula_for)

    longest = formula_for(MIXED_TERM, plate_position=True,
                          random_row_column=True)
    assert explainer_width() == len("    " + longest) == 75
    assert len(_HISTORY_POINTER) <= explainer_width()


def test_the_measurement_survives_where_the_box_points():
    """THE ONE OUTCOME TO AVOID IS DELETING IT.

    Every figure the removed block carried is asserted here, at its own
    precision: a number rounded on the way into the docstring is the claim
    stopping being checkable, which is exactly what the block was for.
    """
    from spacr.qt.screens.settings_model import (COLLINEAR_FORMULA,
                                                 regression_model_explainer)

    doc = " ".join((regression_model_explainer.__doc__ or "").split())

    assert COLLINEAR_FORMULA in doc
    assert "perfectly collinear BY CONSTRUCTION" in doc
    assert "SUM of that gene's gRNA fractions" in doc
    assert "pseudo-inverse" in doc
    for figure in ("610 wells", "823 guides", "389 genes", "1945 rows",
                   "1248 parameters at rank 862", "386", "244480_3",
                   "3.389291", "p = 2.873149e-13",
                   "859 parameters at rank 859", "425 at 425"):
        assert figure in doc, f"{figure} did not survive the move"


def test_the_docstring_is_reachable_from_the_names_the_box_uses():
    """A pointer to a docstring that says nothing about the retired design
    would be a dead end; these are the two other places it is written down."""
    from spacr.qt.screens.settings_model import regression_model_explainer

    doc = " ".join((regression_model_explainer.__doc__ or "").split())
    assert "COLLINEAR_FORMULA" in doc
    assert "spacr.ml.COLLINEAR_FORMULA_FRAGMENT" in doc

    from spacr.ml import COLLINEAR_FORMULA_FRAGMENT

    assert COLLINEAR_FORMULA_FRAGMENT in doc
