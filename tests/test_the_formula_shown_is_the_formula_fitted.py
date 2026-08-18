"""The box shows the formula the run will actually fit.

Reported 2026-08-18: "if i choose mixed this is the formula: y ~
gene_fraction:gene + (1 | gene/grna) + rowID + columnID ... the state of the
model plate possition and random row and column effects should influence this
equation".

It did not. `regression_model_explainer` took only `(regression_type, level)`,
so the plate terms were printed however the two settings were set. A user who
turned plate position OFF still read `+ rowID + columnID` in a formula the run
would not fit -- the same class of failure as an axis that relabels itself
without moving its dots.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.screens.settings_model import (  # noqa: E402
    GENE_TERM, GRNA_TERM, MIXED_TERM, formula_for,
    regression_model_explainer)

pytestmark = pytest.mark.qt


def _formulas(text):
    return [line.strip() for line in text.splitlines()
            if line.strip().startswith("y ~")]


# --------------------------------------------------------------------------- #
#  The three states
# --------------------------------------------------------------------------- #

def test_plate_position_on_is_a_fixed_effect():
    assert formula_for(MIXED_TERM) == (
        "y ~ gene_fraction:gene + (1 | gene/grna) + rowID + columnID")


def test_plate_position_off_removes_the_terms_entirely():
    assert formula_for(MIXED_TERM, plate_position=False) == (
        "y ~ gene_fraction:gene + (1 | gene/grna)")
    assert formula_for(GRNA_TERM, plate_position=False) == (
        "y ~ fraction:grna")


def test_random_row_column_makes_them_variance_components():
    """They are still IN the model -- that is the point of the setting. It
    chooses fixed versus random, not in versus out."""
    assert formula_for(MIXED_TERM, random_row_column=True) == (
        "y ~ gene_fraction:gene + (1 | gene/grna) "
        "+ (1 | rowID) + (1 | columnID)")


def test_random_wins_over_off_rather_than_inventing_a_fourth_state():
    """Off-plus-random is a contradiction and is refused upstream. Rendering
    it as "the terms are random" shows what the refusal is about; rendering it
    as "no terms" would show a formula nothing can produce."""
    assert "(1 | rowID)" in formula_for(
        MIXED_TERM, plate_position=False, random_row_column=True)


# --------------------------------------------------------------------------- #
#  Through the explainer, which is what the user reads
# --------------------------------------------------------------------------- #

def test_the_mixed_box_follows_plate_position():
    with_it = _formulas(regression_model_explainer("mixed", "both"))
    without = _formulas(regression_model_explainer("mixed", "both",
                                                   plate_position=False))
    assert with_it != without
    assert all("rowID" in line for line in with_it)
    assert not any("rowID" in line for line in without)


def test_every_formula_in_the_box_follows_it_not_just_the_first():
    """`both` prints TWO formulas. One tracking the setting and one not is
    worse than neither tracking it."""
    lines = _formulas(regression_model_explainer("ols", "both",
                                                 plate_position=False))
    assert len(lines) == 2
    assert not any("rowID" in line for line in lines)


def test_the_default_is_unchanged():
    """Plate position defaults ON -- by measurement, instruction 143 -- so the
    box a user opens without touching anything must read as it always did."""
    assert _formulas(regression_model_explainer("mixed", "both")) == [
        "y ~ gene_fraction:gene + (1 | gene/grna) + rowID + columnID"]


# --------------------------------------------------------------------------- #
#  And on the real panel, which is where it was reported
# --------------------------------------------------------------------------- #

def test_the_panel_updates_when_the_toggles_move(qtbot):
    """THE CONNECTION WAS THE BUG, not just the missing parameter.

    Both settings are `Toggle`, which is a QCheckBox and has none of the
    combo/text signals the explainer's connect loop looked for -- so nothing
    was connected and the formula never moved. `toggled` is in that list now.
    """
    pytest.importorskip("pyqtgraph")
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    box = screen._model_explainer
    widgets = screen._settings_model._widgets

    before = _formulas(box.toPlainText())
    assert before and all("rowID" in line for line in before)

    widgets["model_plate_position"].setChecked(False)
    qtbot.wait(1)
    assert not any("rowID" in line for line in _formulas(box.toPlainText()))

    widgets["model_plate_position"].setChecked(True)
    widgets["random_row_column_effects"].setChecked(True)
    qtbot.wait(1)
    assert any("(1 | rowID)" in line for line in _formulas(box.toPlainText()))


def test_the_box_matches_what_ml_would_build():
    """The two must not drift. `prepare_formula` is the run's own builder and
    lives in spacr.ml; this asserts the box agrees with it for every state.

    Imported inside the test rather than at module scope: spacr.ml pulls in
    torch through spacr.plot, and the settings panel must never do that --
    there is a separate guard asserting panel-building stays light.
    """
    from spacr.ml import prepare_formula

    for level, term in (("grna", GRNA_TERM), ("gene", GENE_TERM)):
        assert prepare_formula("y", level=level) == formula_for(term)
        assert prepare_formula(
            "y", random_row_column_effects=True, level=level
        ).startswith("y ~ " + term)
