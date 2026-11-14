"""Regression QC on the degenerate fits the panels still have to survive.

Every branch here is a path the QC suite takes when the data it is handed is
flat, singular, unlabelled or mis-typed -- the cases that arrive from a real
screen and that a happy-path fit never reaches. Each one matters because the
alternative to a panel that degrades correctly is a panel that either crashes
(and takes an hour-long fit's diagnostics with it) or, worse, draws a curve
that is not what its legend says it is.

Covered here: residuals with zero spread, where a KDE cannot be estimated;
an all-zero design, whose singular values
give a log axis no headroom to compute; a ``p_value`` column that is not
p-values; the plain-text report with no notes, no verdict counts and a verdict
with no explanation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from spacr import regression_qc as rq

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _axes():
    """A bare axes on an object-oriented figure, as the report driver makes."""
    return Figure(figsize=(4.4, 3.4)).add_subplot(1, 1, 1)


class _Fit:
    """A minimal stand-in for a fitted model: whatever attributes are given."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


def _context(fitted, y, X=None, **kwargs):
    """A context over an explicit fitted/observed pair, built the way the report does."""
    response = np.asarray(y, dtype=float)
    n = response.size
    if X is None:
        X = pd.DataFrame({"intercept": np.ones(n),
                          "x": np.linspace(0.0, 1.0, n)})
    return rq.build_context(_Fit(fittedvalues=np.asarray(fitted, dtype=float)),
                            X, response, **kwargs)


# ---------------------------------------------------------------------------
# axis labelling
# ---------------------------------------------------------------------------

def test_a_panel_title_carries_a_well_count_only_when_it_was_told_one():
    """A count of "None wells" on a panel is worse than no count at all.

    ``_finish`` is the shared labeller for the calibration, ROC, PR and
    count-fit panels. Three of them know their n and one does not, so the
    default has to leave the title alone rather than format a placeholder into
    the one line a reader uses to identify the panel.
    """
    with_n = _axes()
    without_n = _axes()

    rq._finish(with_n, "ROC", "false positive rate", "true positive rate",
               n=1234)
    rq._finish(without_n, "Calibration", "mean predicted value in bin",
               "mean observed value in bin")

    assert with_n.get_title() == "ROC\n(n = 1,234 wells)"
    assert without_n.get_title() == "Calibration"
    assert without_n.get_xlabel() == "mean predicted value in bin"
    assert not without_n.spines["top"].get_visible()


# ---------------------------------------------------------------------------
# residual distribution with no spread
# ---------------------------------------------------------------------------

def test_residuals_with_no_spread_get_no_kde_and_claim_none():
    """A legend entry for a curve that was never drawn is a lie on the figure.

    A fit that reproduces every well with the same constant offset has zero
    residual spread, and ``scipy.stats.gaussian_kde`` raises on it (singular
    covariance). The panel skips the KDE -- and it must also drop the KDE from
    the legend, because a reader who sees "KDE" and one grey dashed curve will
    read the normal reference as the empirical density and conclude the
    residuals are perfectly normal.
    """
    n = 8
    flat = _context(np.zeros(n), np.full(n, 2.0))
    spread = _context(np.zeros(n),
                      np.array([2.0, -1.0, 0.5, 3.0, -2.5, 1.0, 0.0, -0.5]))

    flat_ax, spread_ax = _axes(), _axes()
    flat_stats = rq._panel_residual_distribution(flat, flat_ax)
    spread_stats = rq._panel_residual_distribution(spread, spread_ax)

    flat_labels = [t.get_text() for t in flat_ax.texts]
    spread_labels = [t.get_text() for t in spread_ax.texts]

    assert np.ptp(flat.resid) == 0.0
    assert len(flat_ax.lines) == 1                  # the normal reference only
    assert "KDE" not in flat_labels
    # The same panel over residuals that do vary draws both curves and says so.
    assert len(spread_ax.lines) == 2
    assert "KDE" in spread_labels
    assert "normal fit" in flat_labels and "normal fit" in spread_labels
    assert flat_stats["n_points"] == n == spread_stats["n_points"]
    assert np.isnan(flat_stats["skew"])
    assert np.isfinite(spread_stats["skew"])


# ---------------------------------------------------------------------------
# a design with nothing in it
# ---------------------------------------------------------------------------

def test_an_all_zero_design_draws_a_spectrum_with_no_positive_bar():
    """The singular design has to reach the report, not crash before it.

    A design matrix that came out of a filter that dropped every row's
    predictors is entirely zero. Its singular values are all zero, so there is
    no largest/smallest ratio to scale a log axis by -- and this is exactly the
    fit whose conditioning verdict a user most needs to see, because every
    coefficient it produced is one of infinitely many solutions.
    """
    n = 6
    zeros = pd.DataFrame({"a": np.zeros(n), "b": np.zeros(n)})
    healthy = pd.DataFrame({"a": np.ones(n), "b": np.linspace(-1.0, 1.0, n)})
    response = np.arange(n, dtype=float)

    zero_ax, healthy_ax = _axes(), _axes()
    zero_stats = rq._panel_condition_number(
        _context(np.zeros(n), response, X=zeros), zero_ax)
    healthy_stats = rq._panel_condition_number(
        _context(np.zeros(n), response, X=healthy), healthy_ax)

    assert zero_stats["rank"] == 0
    assert not np.isfinite(zero_stats["condition_number"])
    assert "singular" in zero_stats["verdict"]
    assert all(np.isnan(bar.get_height()) for bar in zero_ax.containers[0])
    # A design that does have positive singular values gets real bars and a
    # finite number, so the nan spectrum above is this design's own answer.
    assert healthy_stats["rank"] == 2
    assert healthy_stats["condition_number"] == pytest.approx(1.0)
    assert all(bar.get_height() > 0 for bar in healthy_ax.containers[0])


# ---------------------------------------------------------------------------
# a p_value column that is not p-values
# ---------------------------------------------------------------------------

def test_a_coefficient_table_outside_zero_to_one_draws_no_uniform_line():
    """A malformed p-value column is refused before anything is drawn.

    A coefficient table whose ``p_value`` column holds test statistics or
    -log10 p is not a sparse histogram: it is the wrong quantity. A16 made
    that distinction loud with ``PanelUnavailable`` rather than returning a
    plausible ``too-few`` diagnosis. Genuine p-values remain the positive
    counterpart and still draw the uniform expectation.
    """
    n = 6
    response = np.arange(n, dtype=float)
    bogus = _context(np.zeros(n), response,
                     coef_df=pd.DataFrame({"p_value": [3.0, 7.0, 12.0]}))
    genuine = _context(np.zeros(n), response,
                       coef_df=pd.DataFrame(
                           {"p_value": np.linspace(0.01, 0.99, 40)}))

    bogus_ax, genuine_ax = _axes(), _axes()
    with pytest.raises(
            rq.PanelUnavailable,
            match=r"3 finite p-value\(s\) outside \[0, 1\].*\[3, 12\]",
    ):
        rq._panel_p_value_histogram(bogus, bogus_ax)
    genuine_stats = rq._panel_p_value_histogram(genuine, genuine_ax)

    assert len(bogus_ax.containers) == 0
    assert len(bogus_ax.lines) == 0
    assert genuine_stats["n"] == 40
    assert genuine_stats["source"] == "coefficient table"
    assert len(genuine_ax.lines) == 1


# ---------------------------------------------------------------------------
# the plain-text report
# ---------------------------------------------------------------------------

def _panel(name, level, detail, *, path=None, stats=None):
    """One panel result carrying a verdict, as the driver builds it."""
    return rq.QCPanelResult(
        name=name, title=name.replace("_", " "), group="fit",
        status="written", path=path, stats=dict(stats or {}),
        verdict=rq.PanelVerdict(level, f"{name} says so", detail, 3.25,
                                "statistic"))


def test_a_report_prints_a_notes_block_and_a_verdict_tally_only_when_it_has_them():
    """Empty section headings train a reader to stop reading the report.

    ``format_qc_report`` is what a reviewer greps, and it is assembled from a
    manifest that may or may not carry notes and verdict counts -- a manifest
    hand-built by a caller replaying old panels has neither. A blank "note:"
    line or a "verdicts:" heading over nothing is noise in the one artifact
    that is supposed to be scannable.
    """
    bare = {"panels": [_panel("qq_residuals", "check", "")]}
    full = {"notes": ["leverage computed from the design matrix"],
            "verdict_counts": {"pass": 0, "check": 1, "fail": 0, "unknown": 0},
            "panels": [_panel("qq_residuals", "check", "")]}

    bare_text = rq.format_qc_report(bare)
    full_text = rq.format_qc_report(full)

    assert "note:" not in bare_text
    assert "verdicts:" not in bare_text
    # Both reports still carry the panel and its verdict; only the two optional
    # blocks differ, which is what makes their absence above meaningful.
    assert "verdict: CHECK — qq_residuals says so" in bare_text
    assert "1 panel(s) drawn, 0 skipped, 0 failed" in bare_text
    assert "note: leverage computed from the design matrix" in full_text
    assert "verdicts: 1 CHECK" in full_text


def test_a_verdict_with_no_explanation_still_reaches_the_report(tmp_path):
    """A scored panel must never be dropped for having no sentence attached.

    ``PanelVerdict.detail`` defaults to an empty string, and several scorers
    leave it that way for a categorical result. The report renders the
    ``means:`` line from it, so an empty detail has to skip that line and go
    straight on to the reason and the statistics -- not skip the panel.
    """
    written = tmp_path / "qq_residuals.png"
    written.write_bytes(b"png")
    manifest = {"panels": [
        _panel("qq_residuals", "fail", "", path=str(written),
               stats={"n_points": 96}),
        _panel("cooks_distance", "check", "one well carries 40% of the fit.",
               stats={"max_cooks": 0.8})]}

    text = rq.format_qc_report(manifest)

    lines = [line.strip() for line in text.splitlines()]
    assert "verdict: FAIL — qq_residuals says so" in lines
    assert "means: one well carries 40% of the fit." in lines
    # The panel with no detail printed everything else about itself.
    assert "statistic: 3.25" in lines
    assert "n_points: 96" in lines
    assert f"file: {written.name}" in lines
    assert sum(1 for line in lines if line.startswith("means:")) == 1
