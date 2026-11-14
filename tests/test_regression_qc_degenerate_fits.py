"""Table-driven tests of regression QC on the degenerate fits its panels must survive.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: the scaled/unscaled condition number, the
scale-location panel's Brown-Forsythe quartiles, the positional residual
partition, the classifier decision score's sign, and the renderer fallbacks
the report driver records.  Every case is a fit a real screen produces --
singular, flat, tied, unlabelled -- where the alternative to a statistic that
degrades correctly is one quoted off data that cannot support it.
"""
from __future__ import annotations

import os
import types
import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from spacr import regression_qc as rq
from spacr import schema


def _axes():
    """A bare axes on an object-oriented figure, as the report driver makes."""
    return Figure(figsize=(4.6, 3.6)).add_subplot(1, 1, 1)


class _Fit:
    """A minimal stand-in for a fitted model: whatever attributes are given."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


def _context(fitted, y, X=None, index=None, **kwargs):
    """A context over an explicit fitted/observed pair, built the way the report does."""
    response = np.asarray(y, dtype=float)
    n = response.size
    if X is None:
        X = pd.DataFrame({"intercept": np.ones(n), "x": np.linspace(0.0, 1.0, n)},
                         index=index if index is not None else range(n))
    return rq.build_context(_Fit(fittedvalues=np.asarray(fitted, dtype=float)),
                            X, response, **kwargs)


# ---------------------------------------------------------------------------
# condition_number
# ---------------------------------------------------------------------------

def test_condition_number_is_unit_free_singular_aware_and_refuses_empty_designs(subtests):
    """Exact scaled/raw ratios and spectra; singular designs are inf; empty or 1-D designs raise."""
    ramp20 = np.linspace(0.0, 1.0, 20)
    ramp10 = np.linspace(0.0, 1.0, 10)
    column = np.linspace(1.0, 5.0, 8).reshape(-1, 1)
    cases = [
        ("orthogonal 3x3", np.eye(3), 1.0, 1.0, [1.0, 1.0, 1.0]),
        ("orthogonal 4x4", np.eye(4), 1.0, 1.0, [1.0, 1.0, 1.0, 1.0]),
        ("column units only move the raw number", np.diag([1.0, 10.0]), 1.0, 10.0,
         [1.0, 1.0]),
        ("one by one", np.ones((1, 1)), 1.0, 1.0, [1.0]),
        ("duplicated predictor", np.column_stack([np.ones(20), ramp20, ramp20]),
         np.inf, np.inf, None),
        ("two identical columns", np.hstack([column, column]), np.inf, np.inf, None),
        ("all-zero column", np.column_stack([np.ones(10), np.zeros(10), ramp10]),
         np.inf, np.inf, None),
    ]
    for case_id, X, scaled, raw, spectrum in cases:
        with subtests.test(msg=case_id):
            got_scaled, got_raw, singular = rq.condition_number(X)
            assert got_scaled == pytest.approx(scaled, rel=1e-12)
            assert got_raw == pytest.approx(raw, rel=1e-12)
            assert singular.size == min(X.shape)
            assert np.all(np.diff(singular) <= 0), "largest first"
            if spectrum is not None:
                np.testing.assert_allclose(singular, spectrum, rtol=1e-12)

    with subtests.test(msg="an all-zero column is not rescued by the scaling"):
        _s, _r, singular = rq.condition_number(
            np.column_stack([np.ones(10), np.zeros(10), ramp10]))
        assert float(singular[-1]) == 0.0

    with subtests.test(msg="rescaling one column moves only the raw number"):
        X = np.random.default_rng(0).normal(size=(30, 3))
        Y = X.copy()
        Y[:, 0] *= 1000.0
        scaled_a, raw_a, spectrum_a = rq.condition_number(X)
        scaled_b, raw_b, spectrum_b = rq.condition_number(Y)
        assert scaled_a == pytest.approx(1.2346233130206925, rel=1e-9)
        assert scaled_b == pytest.approx(scaled_a, rel=1e-12)
        np.testing.assert_allclose(spectrum_b, spectrum_a, rtol=1e-12)
        assert raw_a == pytest.approx(1.4979053125924162, rel=1e-9)
        assert raw_b == pytest.approx(1442.080006519232, rel=1e-9)

    for case_id, bad, shape in (("no rows", np.empty((0, 3)), "(0, 3)"),
                                ("no columns", np.empty((5, 0)), "(5, 0)"),
                                ("nothing", np.empty((0, 0)), "(0, 0)"),
                                ("one-dimensional", np.arange(5.0), "(5,)")):
        with subtests.test(msg=f"refused: {case_id}"):
            with pytest.raises(ValueError) as caught:
                rq.condition_number(bad)
            assert str(caught.value) == (
                f"design matrix must be a non-empty 2-D array, got {shape}")


# ---------------------------------------------------------------------------
# _panel_scale_location
# ---------------------------------------------------------------------------

def _wobble(n):
    """Deterministic, bounded noise so the statistics below are exact."""
    return np.sin(np.arange(n) * 1.7)


def test_scale_location_quotes_brown_forsythe_only_over_two_real_groups(subtests):
    """A real spread gets both statistics; collapsed, constant or short fits fall back to Spearman."""
    nan = float("nan")
    cases = [
        ("a real spread of predictions", np.linspace(0.0, 10.0, 20),
         dict(spearman_rho=0.09473684210526315, spearman_p=0.6911525963694356,
              levene_p=0.9960209763902056, quartile_sd_ratio=1.1083918110432325,
              verdict="no detectable trend in spread", n_points=20)),
        ("tied quartiles leave one usable group",
         np.array([0.0] + [5.0] * 18 + [10.0]),
         dict(spearman_rho=0.3290450982642491, levene_p=nan, quartile_sd_ratio=nan,
              verdict="variance grows with the fit", n_points=20)),
        ("only two distinct quartile edges", np.array([0.0] * 23 + [10.0]),
         dict(spearman_rho=0.34641016151377546, levene_p=nan, quartile_sd_ratio=nan,
              verdict="variance grows with the fit", n_points=24)),
        ("one predicted value", np.full(24, 4.0),
         dict(spearman_rho=nan, levene_p=nan, quartile_sd_ratio=nan,
              verdict="no detectable trend in spread", n_points=24)),
        ("fewer than twenty points", np.linspace(0.0, 10.0, 10),
         dict(spearman_rho=0.13939393939393938, levene_p=nan, quartile_sd_ratio=nan,
              verdict="no detectable trend in spread", n_points=10)),
    ]
    for case_id, fitted, expected in cases:
        with subtests.test(msg=case_id):
            ctx = _context(fitted, fitted + _wobble(fitted.size))
            ax = _axes()
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats = rq._panel_scale_location(ctx, ax)
            for key, value in expected.items():
                if isinstance(value, float) and np.isnan(value):
                    assert np.isnan(stats[key]), key
                elif isinstance(value, float):
                    assert stats[key] == pytest.approx(value, rel=1e-9), key
                else:
                    assert stats[key] == value, key
            text = "\n".join(t.get_text() for t in ax.texts)
            assert expected["verdict"] in text
            assert "Brown-Forsythe p = " in text


# ---------------------------------------------------------------------------
# _grouped_residuals
# ---------------------------------------------------------------------------

def test_grouped_residuals_partition_the_fitted_rows_in_natural_order(subtests):
    """Groups are the distinct stringified labels, never empty, and cover each row once."""
    with subtests.test(msg="singleton, unbalanced and numeric-looking labels"):
        ctx = types.SimpleNamespace(
            resid=np.arange(9, dtype=float),
            metadata=pd.DataFrame({
                "rowID": ["r1", "r1", "r1", "r2", "r2", "r10", "r3", "r3", "r3"]}))
        groups, values = rq._grouped_residuals(ctx, "rowID")
        assert groups == ["r1", "r2", "r3", "r10"], "natural, not lexical, order"
        assert [v.tolist() for v in values] == [[0.0, 1.0, 2.0], [3.0, 4.0],
                                                [6.0, 7.0, 8.0], [5.0]]

    resid = np.arange(12.0)
    fitted_index = [i for i in range(24) if i % 2 == 0]
    whole_map = pd.DataFrame(
        {schema.ROW_KEY: [1, 1, 2, 2, 3, 3] * 4,
         schema.COLUMN_KEY: np.repeat([1.5, 2.5, 3.5], 8),
         schema.PLATE_KEY: ["p1"] * 12 + [None] * 12},
        index=range(24))
    cut_map = whole_map.loc[fitted_index].reset_index(drop=True)
    routes = {
        "aligned by index from a longer plate map":
            _context(np.zeros(12), resid, index=fitted_index, metadata=whole_map),
        "aligned by length from a cut plate map":
            _context(np.zeros(12), resid, metadata=cut_map),
    }
    expected = {
        schema.ROW_KEY: (["1", "2", "3"], [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]),
        schema.COLUMN_KEY: (["1.5", "2.5", "3.5"],
                            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
        schema.PLATE_KEY: (["p1", "<missing>"], [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]),
    }
    for route, ctx in routes.items():
        for column, (want_groups, want_values) in expected.items():
            with subtests.test(msg=f"{route}: {column}"):
                groups, values = rq._grouped_residuals(ctx, column)
                assert groups == want_groups
                assert [v.tolist() for v in values] == want_values

    refusals = [
        ("one distinct value",
         types.SimpleNamespace(resid=np.zeros(3),
                               metadata=pd.DataFrame({"rowID": ["r1", "r1", "r1"]})),
         "rowID", "every well has rowID = r1; there is no between-rowID effect to see"),
        ("no metadata", types.SimpleNamespace(resid=np.zeros(3), metadata=None),
         "rowID", "no per-well metadata was supplied, so plate position is unknown"),
        ("absent column",
         types.SimpleNamespace(resid=np.zeros(2),
                               metadata=pd.DataFrame({"plateID": ["a", "b"]})),
         "rowID", "metadata has no 'rowID' column (present: plateID)"),
    ]
    for case_id, ctx, column, message in refusals:
        with subtests.test(msg=f"refused: {case_id}"):
            with pytest.raises(rq.PanelUnavailable) as caught:
                rq._grouped_residuals(ctx, column)
            assert str(caught.value) == message


# ---------------------------------------------------------------------------
# build_context: the decision score's sign
# ---------------------------------------------------------------------------

class _UnlabelledClassifier:
    """A wrapped classifier that ranks wells but never published ``classes_``."""

    def __init__(self, n):
        self.fittedvalues = np.zeros(n)

    def decision_function(self, X):
        return np.linspace(-2.0, 2.0, len(X))


class _InvertedClassifier(_UnlabelledClassifier):
    """The same scores, but declaring that the LARGER score is the 0 class."""

    classes_ = np.array([1, 0])


class _AscendingClassifier(_UnlabelledClassifier):
    """sklearn's own convention, stated."""

    classes_ = np.array([0, 1])


class _TextLabelledClassifier(_UnlabelledClassifier):
    """Labels that are not numbers say nothing about which side is larger."""

    classes_ = np.array(["yes", "no"])


def test_build_context_orients_the_decision_score_only_from_declared_classes(subtests):
    """Silence and unorderable labels keep the given sign; a descending classes_ negates it."""
    n = 6
    X = pd.DataFrame({"intercept": np.ones(n), "x": np.linspace(0.0, 1.0, n)})
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ramp = np.linspace(-2.0, 2.0, n)
    cases = [
        ("no classes_", _UnlabelledClassifier, ramp),
        ("ascending classes_", _AscendingClassifier, ramp),
        ("text classes_", _TextLabelledClassifier, ramp),
        ("descending classes_", _InvertedClassifier, -ramp),
    ]
    for case_id, cls, expected in cases:
        with subtests.test(msg=case_id):
            ctx = rq.build_context(cls(n), X, y)
            np.testing.assert_allclose(ctx.decision_score, expected)
            np.testing.assert_allclose(ctx.ranking_score, expected)

    with subtests.test(msg="a model without decision_function has no decision score"):
        ctx = rq.build_context(_Fit(fittedvalues=np.zeros(n)), X, y)
        assert ctx.decision_score is None


# ---------------------------------------------------------------------------
# regression_qc_report: renderer fallbacks
# ---------------------------------------------------------------------------

def test_regression_qc_report_names_every_page_matplotlib_drew_instead(subtests, tmp_path,
                                                                    capsys, monkeypatch):
    """Under pyqtgraph each matplotlib fallback is recorded and printed; native draws name none."""
    import spacr.figures.scene as scene

    drew = {"by": "matplotlib", "why": ""}

    def _wrote(fig, path, *, fmt=None, dpi=None, renderer=None, announce=True,
               title=None, **savefig):
        target = path if os.path.splitext(path)[1] else f"{path}.png"
        with open(target, "wb") as handle:
            handle.write(b"png")
        return target, drew["by"], drew["why"]

    monkeypatch.setattr(scene, "scene_renderer", lambda force=None: ("pyqtgraph", ""))
    monkeypatch.setattr(scene, "write_figure", _wrote)

    n = 30
    X = pd.DataFrame({"intercept": np.ones(n), "x": np.linspace(-1.0, 1.0, n)})
    y = 1.0 + 2.0 * X["x"].to_numpy() + 0.3 * _wobble(n)
    model = _Fit(fittedvalues=1.0 + 2.0 * X["x"].to_numpy())

    bar = "no pyqtgraph translation for a bar"
    page = "combined pages need matplotlib"
    cases = [
        # case id, panels, regression type, what drew it and why, fallbacks expected
        ("one panel falls back", ("condition_number",), None, ("matplotlib", bar), True),
        ("one panel drawn natively", ("condition_number",), None, ("pyqtgraph", ""), False),
        ("a matplotlib draw with no reason is not a fallback", ("condition_number",),
         None, ("matplotlib", ""), False),
        ("the OLS assumption page falls back", rq.OLS_ASSUMPTION_PANELS, "ols",
         ("matplotlib", page), True),
        ("the OLS assumption page drawn natively", rq.OLS_ASSUMPTION_PANELS, "ols",
         ("pyqtgraph", ""), False),
    ]
    for index, (case_id, panels, kind, (by, why), falls_back) in enumerate(cases):
        with subtests.test(msg=case_id):
            drew.update(by=by, why=why)
            capsys.readouterr()
            manifest = rq.regression_qc_report(
                model, X, y, str(tmp_path / f"run{index}"), regression_type=kind,
                panels=panels, combined=True, verbose=True)
            printed = capsys.readouterr().out
            written = [r.name for r in manifest["panels"] if r.path]
            pages = ["regression_qc_report"] + (
                ["ols_assumption_diagnostics"] if kind == "ols" else [])
            assert manifest["renderer"] == "pyqtgraph"
            assert manifest["renderer_counts"] == {by: len(written) + len(pages)}
            if kind == "ols":
                assert manifest["assumptions"] == os.path.join(
                    str(tmp_path / f"run{index}"), rq.QC_DIRNAME,
                    "ols_assumption_diagnostics.png")
            else:
                assert written == ["condition_number"]
                assert manifest["assumptions"] is None
            expected = [(name, why) for name in written + pages] if falls_back else []
            assert manifest["renderer_fallbacks"] == expected
            for name, reason in expected:
                assert f"{name} fell back to matplotlib: {reason}" in printed
            if not falls_back:
                assert "fell back to matplotlib" not in printed

    with subtests.test(msg="an exact fit scores unknown, stamps nothing and prints nothing quietly"):
        drew.update(by="matplotlib", why=page)
        exact = 1.0 + 2.0 * X["x"].to_numpy()
        capsys.readouterr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manifest = rq.regression_qc_report(
                _Fit(fittedvalues=exact), X, exact, str(tmp_path / "exact"),
                regression_type="ols", panels=rq.OLS_ASSUMPTION_PANELS,
                combined=True, verbose=False)
        assert capsys.readouterr().out == ""
        verdicts = {r.name: (r.status, r.verdict) for r in manifest["panels"]}
        unknown = {
            "qq_residuals": "the Q-Q could not be scored",
            "residuals_vs_fitted": "no trend could be measured",
            "scale_location": "variance homogeneity was not tested",
        }
        for name, headline in unknown.items():
            status, verdict = verdicts[name]
            assert status == "written"
            assert (verdict.level, verdict.headline, verdict.detail) == (
                "unknown", headline, "")
        assert verdicts["cooks_distance"][1].level == "pass"
        assert verdicts["influence"][1].level == "pass"
        for name in ("plate_effects", "row_effects", "column_effects"):
            assert verdicts[name] == ("skipped", None)
        assert manifest["verdict_counts"] == {"pass": 2, "check": 0, "fail": 0,
                                              "unknown": 3}

    with subtests.test(msg="an unknown or missing verdict leaves the panel unstamped"):
        for verdict in (None, rq.PanelVerdict("unknown", "no trend could be measured")):
            ax = _axes()
            rq.draw_verdict(ax, verdict)
            assert len(ax.texts) == 0
        ax = _axes()
        rq.draw_verdict(ax, rq.PanelVerdict("fail", "the residuals bend"))
        assert [t.get_text() for t in ax.texts] == ["FAIL  the residuals bend"]
