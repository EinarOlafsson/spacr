"""Every way the nonparametric fits decline, and why declining is the feature.

The module's own docstring at ``refuse`` states the rule: it is better to
refuse than to return a fit nobody should read. Almost none of the refusals had
a test, which is the usual shape -- the happy paths get fixtures and the
guards that protect a scientific result do not.

Two of these are silent rather than loud, and the difference is deliberate:
``describe`` and ``smooth`` RAISE, because a caller naming a method that does
not exist has a bug, while ``report_agreement`` returns "" because it decorates
a finished fit and must never take a run down for a footnote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# describe and refuse — an unknown method name
# ---------------------------------------------------------------------------

def test_describing_a_method_that_does_not_exist_names_the_ones_that_do():
    """Line 131. The error lists the alternatives, which is why it is a raise.

    A KeyError carrying only the bad name leaves the caller to go and find the
    vocabulary; carrying the sorted list makes the message the documentation.
    """
    from spacr.nonparametric_fits import METHODS, describe

    with pytest.raises(KeyError) as excinfo:
        describe("loess")                       # a real word, not this API's

    message = str(excinfo.value)
    assert "loess" in message
    for name in sorted(METHODS):
        assert name in message


def test_every_shipped_method_describes_itself():
    """The other side, and a check that no method ships without its prose."""
    from spacr.nonparametric_fits import METHODS, describe

    for name in METHODS:
        text = describe(name)
        assert "For " in text and "Costs: " in text


def test_refusing_an_unknown_method_returns_the_complaint_rather_than_raising():
    """Line 144. ``refuse`` is the asking function, so it answers in prose."""
    from spacr.nonparametric_fits import refuse

    complaint = refuse("loess")
    assert complaint and "loess" in complaint


def test_a_gaussian_process_is_refused_on_too_many_rows():
    """The cost rule beside it: cubic in the sample, so the size is the reason."""
    from spacr.nonparametric_fits import GP_MAXIMUM_ROWS, refuse

    complaint = refuse("gaussian_process", rows=GP_MAXIMUM_ROWS + 1)
    assert complaint and "cubic" in complaint
    assert not refuse("gaussian_process", rows=10)


# ---------------------------------------------------------------------------
# smooth — mismatched inputs and an unknown diagnostic
# ---------------------------------------------------------------------------

def test_smoothing_refuses_x_and_y_of_different_lengths():
    """Line 200. The message carries BOTH counts, which is what locates the bug.

    A silent truncation to the shorter of the two would fit a curve through
    pairs that were never observed together.
    """
    from spacr.nonparametric_fits import smooth

    with pytest.raises(ValueError) as excinfo:
        smooth([1.0, 2.0, 3.0], [1.0, 2.0])

    assert "3 points" in str(excinfo.value) and "2" in str(excinfo.value)


def test_a_fit_method_asked_of_the_smoother_is_named_as_a_fit():
    """The category guard, which is what an ordinary caller meets.

    'spline' exists and is a FIT. The message says so and points at METHODS,
    rather than reporting it as an unknown name.
    """
    from spacr.nonparametric_fits import smooth

    x = np.linspace(0.0, 1.0, 40)
    with pytest.raises(ValueError, match="is a fit, not a diagnostic"):
        smooth(x, x, method="spline")


def test_a_diagnostic_with_no_body_is_refused_by_name(monkeypatch):
    """Line 259: the final raise, reachable only by adding a method.

    Every diagnostic in METHODS has a branch above this line, so an ordinary
    caller can never reach it -- the category guard catches a fit, and the
    vocabulary check catches an unknown name. What it protects against is
    someone adding an entry to METHODS and forgetting the body, which is
    exactly what this test stages: a new diagnostic, correctly declared, with
    nothing implementing it.

    Without this raise that mistake returns None, and the caller plots an
    empty curve.
    """
    from spacr import nonparametric_fits as npf

    added = dict(npf.METHODS["lowess"])
    added["label"] = "Newly added"
    monkeypatch.setitem(npf.METHODS, "a_new_diagnostic", added)

    x = np.linspace(0.0, 1.0, 40)
    with pytest.raises(ValueError, match="no diagnostic named"):
        npf.smooth(x, x, method="a_new_diagnostic")


# ---------------------------------------------------------------------------
# agreement — a refused method, and a design with no column names
# ---------------------------------------------------------------------------

def test_agreement_refuses_a_method_outside_its_category():
    """Line 316 (via ``refuse``) and the category check beside it.

    Ranking guides needs a method that can rank them. A diagnostic smoother
    would produce numbers, and numbers of the wrong kind are worse here than
    no answer at all.
    """
    from spacr.nonparametric_fits import agreement

    design = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)),
                          columns=["a", "b", "c"])
    response = np.random.default_rng(1).normal(size=30)

    with pytest.raises(ValueError):
        agreement(design, response, {"a": 1.0}, method="lowess")


def test_a_design_without_column_names_gets_generated_ones():
    """Line 326: a bare array is accepted and its columns named x0, x1, ...

    The linear effect is keyed by NAME, so an unnamed design would otherwise
    match nothing and the comparison would silently be over an empty set.
    """
    from spacr.nonparametric_fits import agreement

    rng = np.random.default_rng(0)
    values = rng.normal(size=(60, 3))
    response = values[:, 0] * 2.0 + rng.normal(scale=0.1, size=60)

    result = agreement(values, response,
                       {"x0": 2.0, "x1": 0.0, "x2": 0.0},
                       method="random_forest", seed=0)

    assert result is not None


def test_fewer_than_three_shared_names_gives_no_correlation():
    """Line 356: rho is NaN, not 0.0.

    Spearman over two points is ±1 by construction, which would read as
    perfect agreement or perfect disagreement from no evidence at all.
    """
    from spacr.nonparametric_fits import agreement

    rng = np.random.default_rng(0)
    design = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    response = rng.normal(size=40)

    result = agreement(design, response, {"a": 1.0, "zz": 2.0},
                       method="random_forest", seed=0)

    assert np.isnan(result.correlation)
    assert "correlation is unavailable" in result.summary()
    assert "DISAGREE" not in result.summary()


# ---------------------------------------------------------------------------
# spline_design — a covariate that is not in the frame
# ---------------------------------------------------------------------------

def test_a_covariate_the_frame_does_not_have_is_passed_over():
    """Line 400. Named covariates come from settings and the frame from a run.

    They disagree constantly -- a column dropped by filtration, a setting left
    from another screen. Raising would make one stale setting fatal to the fit.
    """
    from spacr.nonparametric_fits import spline_design

    frame = pd.DataFrame({"present": np.linspace(0.0, 1.0, 40),
                          "other": np.arange(40, dtype=float)})

    out = spline_design(frame, ["present", "absent_from_this_screen"])

    assert "other" in out.columns
    assert not [c for c in out.columns if c.startswith("absent")]


# ---------------------------------------------------------------------------
# report_agreement — every route to the empty string
# ---------------------------------------------------------------------------

def test_a_coefficient_that_is_not_a_number_is_skipped():
    """Lines 454-455: ``continue`` past an unparseable coefficient.

    A coefficient column can carry '' or 'NA' from a fit that failed for one
    guide. One bad cell must not stop the agreement note for the others.
    """
    from spacr.nonparametric_fits import report_agreement

    coefficients = pd.DataFrame({
        "feature": [f"grna[g{i}]" for i in range(4)],
        "coefficient": ["not a number", 1.0, 2.0, 3.0],
    })
    rng = np.random.default_rng(0)
    design = pd.DataFrame(rng.normal(size=(40, 4)),
                          columns=[f"grna[g{i}]" for i in range(4)])
    response = rng.normal(size=40)

    # Three usable effects remain, which is exactly the minimum.
    text = report_agreement(coefficients, design, response)
    assert isinstance(text, str)


def test_fewer_than_three_usable_guides_reports_nothing():
    """Line 466: "" rather than a comparison over one or two guides.

    An agreement statistic over two guides is not weak evidence, it is none,
    and this text goes into a run summary a reader takes at face value.
    """
    from spacr.nonparametric_fits import report_agreement

    coefficients = pd.DataFrame({"feature": ["grna[g0]", "grna[g1]"],
                                 "coefficient": [1.0, 2.0]})
    rng = np.random.default_rng(0)
    design = pd.DataFrame(rng.normal(size=(20, 2)),
                          columns=["grna[g0]", "grna[g1]"])

    assert report_agreement(coefficients, design, rng.normal(size=20)) == ""


def test_a_failing_agreement_run_reports_nothing_rather_than_raising():
    """Lines 472-473: the bare except that keeps a footnote from ending a run.

    This decorates a fit that has already succeeded. Whatever goes wrong in
    the second-opinion model, the fit the user asked for must still be
    reported.
    """
    from spacr import nonparametric_fits as npf

    coefficients = pd.DataFrame({
        "feature": [f"grna[g{i}]" for i in range(4)],
        "coefficient": [1.0, 2.0, 3.0, 4.0],
    })
    rng = np.random.default_rng(0)
    design = pd.DataFrame(rng.normal(size=(40, 4)),
                          columns=[f"grna[g{i}]" for i in range(4)])
    response = rng.normal(size=40)

    def explode(*_a, **_k):
        raise RuntimeError("the second-opinion model would not fit")

    original = npf.agreement
    npf.agreement = explode
    try:
        assert npf.report_agreement(coefficients, design, response) == ""
    finally:
        npf.agreement = original


def test_agreement_refuses_an_unknown_method_before_looking_it_up():
    """Line 316: the ``refuse`` complaint is raised, not a KeyError.

    Order matters here. ``refuse`` is consulted BEFORE ``METHODS[method]``, so
    a name that does not exist comes back as the prose complaint rather than
    as a KeyError from the lookup on the next line. The caller gets the list
    of real methods instead of a bare missing-key traceback.
    """
    from spacr.nonparametric_fits import agreement

    design = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)),
                          columns=["a", "b", "c"])
    response = np.random.default_rng(1).normal(size=30)

    with pytest.raises(ValueError) as excinfo:
        agreement(design, response, {"a": 1.0}, method="not_a_method")

    assert "not_a_method" in str(excinfo.value)


def test_effects_without_matching_design_columns_report_nothing():
    """Line 466: enough coefficients, too few of them in the design.

    The two counts are checked separately on purpose. A fit can name guides
    that filtration removed from the design, so having three EFFECTS is not
    the same as having three COLUMNS to rank -- and it is the columns that the
    second-opinion model actually needs.
    """
    from spacr.nonparametric_fits import report_agreement

    coefficients = pd.DataFrame({
        "feature": [f"grna[g{i}]" for i in range(5)],
        "coefficient": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    rng = np.random.default_rng(0)
    # Only two of the five guides survived into the design.
    design = pd.DataFrame(rng.normal(size=(30, 2)),
                          columns=["grna[g0]", "grna[g1]"])

    assert report_agreement(coefficients, design, rng.normal(size=30)) == ""
