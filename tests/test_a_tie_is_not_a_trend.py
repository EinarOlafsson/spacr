"""The residual panel's headline number was an artefact of tied fitted values.

Found by the adversarial verify pass of the QC restyle, 2026-08-16, and it is
a misleading result rather than a cosmetic one.

A well-level regression routinely has most of its fitted values IDENTICAL --
on the tsg101 screen 451 of 610 wells share one value to seven decimal places,
because most wells carry the same guide mixture. LOWESS fits a local
regression in a neighbourhood; where that neighbourhood is one repeated x it
is fitting a line through a vertical stack and the smoothed value there is
unconstrained. Two adjacent points 2e-6 apart came back 0.12 apart.

Taking the maximum over that included the artefact:

    reported |trend| max   0.109
    the real curve away from the tie block   0.030      -- a 3.6x inflation

on the number the panel exists to report, drawn as the ONLY coloured artist
on the panel. A reader would have concluded the residuals had structure.

The curve is still drawn in full. Hiding the spike would hide the tie, and
the tie is worth seeing.
"""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spacr.regression_qc import _trend_off_the_ties


def _tied_curve():
    """The real shape: a large tied block, then a genuine spread."""
    sx = np.concatenate([np.full(451, 0.1406899),
                         np.linspace(0.1407, 0.30, 159)])
    sy = np.concatenate([np.linspace(-0.030, -0.1086, 451),
                         np.linspace(0.010, 0.030, 159)])
    return sx, sy


def test_the_tied_block_does_not_set_the_trend():
    sx, sy = _tied_curve()

    assert float(np.nanmax(np.abs(sy))) == pytest.approx(0.1086, abs=1e-3), (
        "the fixture no longer contains the artefact it exists to catch")
    assert _trend_off_the_ties(sx, sy) == pytest.approx(0.030, abs=1e-3)


def test_a_curve_with_no_ties_is_untouched():
    """The guard must not shrink an honest trend."""
    sx = np.linspace(0, 1, 200)
    sy = np.sin(sx * 3) * 0.2

    assert _trend_off_the_ties(sx, sy) == pytest.approx(
        float(np.nanmax(np.abs(sy))))


def test_a_real_curved_trend_still_reports_its_peak():
    """Structure in the residuals is what this number is FOR."""
    sx = np.linspace(0, 1, 300)
    sy = (sx - 0.5) ** 2 * 2.0

    assert _trend_off_the_ties(sx, sy) == pytest.approx(0.5, abs=1e-2)


def test_everything_tied_is_not_a_trend_at_all():
    """A fit whose every well shares one fitted value has NO trend to
    measure -- there is no x to have a trend against.

    NaN, not zero and not the plain maximum. Zero would claim the residuals
    are flat and the plain maximum would report the vertical spread of a
    single stack as if it were structure; both are assertions the data cannot
    support. NaN says "not measurable here", which is the true answer and the
    one the panel can print.
    """
    sx = np.full(50, 0.25)
    sy = np.linspace(-0.4, 0.4, 50)

    assert np.isnan(_trend_off_the_ties(sx, sy))


def test_it_survives_the_degenerate_shapes():
    assert np.isnan(_trend_off_the_ties(np.array([]), np.array([])))
    assert _trend_off_the_ties(np.array([1.0]), np.array([2.0])) == 2.0
    assert np.isnan(_trend_off_the_ties(np.zeros(9), np.zeros(9))) or \
        _trend_off_the_ties(np.zeros(9), np.zeros(9)) == 0.0


def test_the_panel_reports_the_corrected_number():
    """Through the real panel, not the helper: the number in the annotation
    and the number in the returned report have to be the same one."""
    import matplotlib.pyplot as plt

    from spacr.regression_qc import _panel_residuals_vs_fitted

    rng = np.random.default_rng(0)
    n = 600
    fitted = np.concatenate([np.full(450, 0.1406899),
                             rng.uniform(0.141, 0.30, n - 450)])
    resid = rng.normal(0, 0.02, n)

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.fitted, ctx.resid, ctx.n = fitted, resid, n
    ctx.family, ctx.prediction_note = "gaussian", ""

    figure = plt.figure()
    try:
        report = _panel_residuals_vs_fitted(ctx, figure.add_subplot(111))
        assert np.isfinite(report["max_abs_trend"])
        assert report["max_abs_trend"] < 0.05, (
            f"a tie block inflated the reported trend to "
            f"{report['max_abs_trend']:.3g} on residuals with no structure")
    finally:
        plt.close(figure)
