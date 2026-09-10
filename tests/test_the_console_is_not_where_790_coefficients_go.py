"""Instruction 183 — a run's console is for the lines a user has to read.

A screen's fit has hundreds of guides. Printing all of them puts about 80 000
characters into the console in one burst and scrolls away the lines above it:
the not-identifiable warning, the fraction filter's retained fraction, the
pairing counts, the family sentence. Those are the ones that decide whether
the run means anything.

Two properties, and the second is the one that keeps this honest: what is left
out is COUNTED and pointed at, never sampled. "The first twenty coefficients"
is a sample of a table whose interesting rows are wherever they happen to be,
and a reader who sees twenty has no way to know which twenty.

Instruction 182 B rides along: a negative McFadden R² is a headline, not a
number among numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

statsmodels = pytest.importorskip("statsmodels.api")

from spacr.ml import (                                    # noqa: E402
    CONSOLE_COEFFICIENT_LIMIT, mcfadden_note, summary_for_console,
)


def _fit(n_predictors: int, n_rows: int = 200, family=None):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(n_rows, n_predictors)),
                         columns=[f"g{i}" for i in range(n_predictors)])
    design = statsmodels.add_constant(frame)
    if family is None:
        return statsmodels.OLS(rng.normal(size=n_rows), design).fit()
    response = np.clip(rng.random(n_rows), 0.01, 0.99)
    return statsmodels.GLM(response, design, family=family).fit()


def _rows_of(text: str) -> list:
    """The coefficient rows a summary prints, however it is laid out."""
    return [line for line in text.splitlines()
            if line.startswith(("g", "const")) and "  " in line]


def _without_render_clock(text: str) -> str:
    """Remove only statsmodels' wall-clock fields from a rendered summary."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith(("Date:", "Time:"))
    )


# -- what is printed --------------------------------------------------------

def test_a_screen_sized_fit_does_not_put_its_coefficients_on_the_console():
    model = _fit(400)
    trimmed = summary_for_console(model)

    assert len(_rows_of(trimmed)) == 0
    assert len(trimmed.splitlines()) < 30


def test_what_was_left_out_is_counted_never_sampled():
    model = _fit(400)
    trimmed = summary_for_console(model)

    # 400 predictors plus the intercept.
    assert "401 coefficients" in trimmed
    assert "model_summary.txt" in trimmed
    assert "Coefficients tab" in trimmed


def test_the_header_survives_whole_because_that_is_what_a_reader_needs():
    model = _fit(400)
    trimmed = summary_for_console(model)

    for wanted in ("No. Observations", "Df Residuals", "Df Model", "R-squared"):
        assert wanted in trimmed


def test_the_notes_survive_because_the_condition_number_is_the_collinearity():
    """A screen's design is collinear, and Cond. No. is where that shows."""
    model = _fit(400)
    assert "Cond. No." in summary_for_console(model)


def test_a_fit_small_enough_to_read_is_printed_unchanged():
    model = _fit(3)
    assert _without_render_clock(summary_for_console(model)) == (
        _without_render_clock(str(model.summary()))
    )


def test_the_limit_is_the_boundary_it_says_it_is():
    small = _fit(CONSOLE_COEFFICIENT_LIMIT - 1)     # + the intercept
    assert _without_render_clock(summary_for_console(small)) == (
        _without_render_clock(str(small.summary()))
    )
    big = _fit(CONSOLE_COEFFICIENT_LIMIT + 4)
    assert "coefficients — not printed here" in summary_for_console(big)


def test_verbose_prints_every_row_because_that_is_what_verbose_is():
    model = _fit(400)
    assert _without_render_clock(summary_for_console(model, verbose=True)) == (
        _without_render_clock(str(model.summary()))
    )


def test_a_glm_is_cut_in_the_right_place_too():
    """GLM prints more header tables than OLS; the cut is found, not counted."""
    model = _fit(400, family=statsmodels.families.Binomial())
    trimmed = summary_for_console(model)

    assert "401 coefficients" in trimmed
    assert "Model Family" in trimmed and "Link Function" in trimmed
    assert len(_rows_of(trimmed)) == 0


def test_a_model_that_cannot_render_says_so_rather_than_raising():
    class Broken:
        def summary(self):
            raise RuntimeError("no design matrix")

    text = summary_for_console(Broken())
    assert "could not render" in text and "no design matrix" in text


# -- 182 B: a negative pseudo-R² is a headline ------------------------------

def test_a_negative_mcfadden_says_the_fit_is_worse_than_its_own_intercept():
    note = mcfadden_note(-20.2752)
    assert "-20.2752" in note
    assert "NEGATIVE" in note
    assert "WORSE than its own intercept" in note
    # The usual cause, named, because that is what makes the flag actionable.
    assert "transformed twice" in note


def test_an_ordinary_mcfadden_is_just_a_number():
    assert mcfadden_note(0.31) == "McFadden's R²: 0.3100"


def test_an_unavailable_mcfadden_is_said_rather_than_crashed():
    assert "not available" in mcfadden_note(None)
    assert "not available" in mcfadden_note("nonsense")
