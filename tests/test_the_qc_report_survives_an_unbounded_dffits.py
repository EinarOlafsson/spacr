"""A saturated row makes DFFITS infinite; the report must still be writable.

A fit with a singleton dummy column fits that row exactly, so its hat diagonal
is 1 and its DFFITS is ``+inf``. That is a real state of a real screen -- the
tsg101 OLS fit of ``pred ~ fraction:grna`` has 186 such rows -- and it must not
take the whole QC report down with it.

The failure mode these tests pin is indirect and worth spelling out, because
nothing about it is visible in the panel's own numbers:

``np.nan_to_num(magnitude, nan=0.0)`` leaves ``+inf`` alone except to swap it
for ``1.797e308``. Autoscaling a stem plot over ``0 .. 1.8e308`` overflows, and
matplotlib answers with a DEGENERATE view: ``ylim == (-1e-12, 1e-12)``. Every
artist positioned in DATA coordinates then lands about ``1e12`` axes-heights
outside the axes, so ``Figure.get_tightbbox`` reports a figure two trillion
inches tall and ``savefig(bbox_inches="tight")`` raises ``TypeError`` from the
Agg renderer's constructor.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from spacr import regression_qc as rq


@pytest.fixture()
def saturated_context():
    """A fit with one perfectly-fitted row, so its DFFITS is ``+inf``."""
    rng = np.random.default_rng(0)
    n = 24
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    frame["solo"] = 0.0
    frame.loc[0, "solo"] = 1.0
    design = sm.add_constant(frame)
    response = rng.normal(size=n)
    model = sm.OLS(response, design).fit()
    return rq.build_context(model, design, response, regression_type="ols")


def test_the_fixture_really_does_produce_an_infinite_dffits(saturated_context):
    """Guard the guard: without a ``+inf`` these tests prove nothing."""
    values, _ = rq.dffits(saturated_context.std_resid,
                          saturated_context.leverage,
                          saturated_context.n, saturated_context.p)
    assert np.isposinf(np.abs(values)).sum() >= 1, (
        "the fixture no longer saturates a row, so it cannot exercise the "
        "unbounded-DFFITS path")


def test_the_dffits_panel_keeps_a_finite_view(saturated_context):
    """The y-limits must come from the finite wells, not from ``1.8e308``."""
    fig = Figure(figsize=(5.6, 4.4), dpi=140)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    rq._panel_dffits(saturated_context, ax)

    low, high = ax.get_ylim()
    assert np.isfinite(low) and np.isfinite(high)
    assert high > 1e-6, (
        f"y-axis collapsed to {(low, high)}: the autoscale overflowed and "
        "matplotlib fell back to a degenerate view, so the panel is blank")


def test_every_dffits_artist_sits_inside_a_sane_figure(saturated_context):
    """No artist may be placed a trillion axes-heights off the page."""
    fig = Figure(figsize=(5.6, 4.4), dpi=140)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    rq._panel_dffits(saturated_context, ax)

    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    assert np.isfinite(bbox.height) and np.isfinite(bbox.width)
    assert bbox.height < 100, (
        f"tight bbox is {bbox.height:.4g} inches tall; an artist is drawn in "
        "data coordinates outside a degenerate y-view")


def test_the_dffits_panel_can_actually_be_saved(saturated_context, tmp_path):
    """The user-facing failure: the panel would not go to disk at all."""
    fig = Figure(figsize=(5.6, 4.4), dpi=140)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    rq._panel_dffits(saturated_context, ax)

    out = tmp_path / "dffits.png"
    fig.savefig(out, bbox_inches="tight")
    assert out.exists() and out.stat().st_size > 0


def test_the_unbounded_well_is_still_drawn_and_still_flagged(saturated_context):
    """Clamping the DRAWING must not hide the well or soften the report."""
    fig = Figure(figsize=(5.6, 4.4), dpi=140)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    stats = rq._panel_dffits(saturated_context, ax)

    values, threshold = rq.dffits(saturated_context.std_resid,
                                  saturated_context.leverage,
                                  saturated_context.n, saturated_context.p)
    unbounded = int(np.isposinf(np.abs(values))[0])
    assert unbounded == 1, "row 0 is the saturated one"
    assert 0 in [int(i) for i in np.where(np.abs(values) > threshold)[0]]
    assert stats["n_above"] >= 1
    assert str(saturated_context.labels[0]) in stats["flagged"], (
        "the infinitely influential well dropped out of the report")

    top = max(seg[1][1] for coll in ax.collections
              for seg in coll.get_segments())
    assert np.isfinite(top) and top > 0, (
        "the unbounded well is drawn at zero height, so the panel shows "
        "nothing where the most influential well is")


def test_the_reported_maximum_is_still_infinite(saturated_context):
    """The STATISTIC is unchanged: only the drawing is clamped."""
    fig = Figure(figsize=(5.6, 4.4), dpi=140)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    stats = rq._panel_dffits(saturated_context, ax)
    assert np.isposinf(stats["max_abs_dffits"]), (
        "max_abs_dffits was quietly clamped; that is a re-analysis, not a "
        "drawing fix")
