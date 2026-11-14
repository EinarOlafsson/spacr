"""Table-driven tests of what :mod:`spacr.power_model` refuses, and what it records instead.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: preparing the well-level design, the horseshoe prior's
global scale, fitting, gathering the per-gene estimate, scoring against
ground truth, and the resumable parameter sweep.  The interesting part is the
failures: a design that cannot identify anything, a sweep point that crashed,
a screen with no hits in it.  Each has a wrong answer that still plots (0.5,
a backfilled zero) and the tests pin what the module does instead.

Fits run the real torch ADVI backend on a four-gene, six-well design with
20 optimisation steps, so every sweep row lands as ``not_converged``; the
optional NUTS backends are not reached from here.
"""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from spacr import power_model as pm

#: Small fits: a few hundred milliseconds each once torch is warm.
_FIT = {"n_steps": 20, "n_draws": 8}


def tiny_screen(n_genes: int = 4, n_wells: int = 6, seed: int = 0,
                **_ignored) -> pd.DataFrame:
    """A small tidy screen in the shape ``prepare_model_data`` reads; gene G0 is the hit."""
    rng = np.random.default_rng(int(seed))
    rows = []
    for well in range(n_wells):
        for gene in range(n_genes):
            rows.append({
                "gene": f"G{gene}", "well": f"w{well}",
                "positive": int(rng.integers(0, 5)),
                "n_reads_per_gene_per_well": int(rng.integers(10, 100)),
                "imaging_n_cells_per_gene_per_well": 25,
                "hit": gene == 0,
            })
    return pd.DataFrame(rows)


def _with(frame, column, row, value, dtype=None):
    """A copy of ``frame`` with one cell replaced (optionally after a dtype change)."""
    out = frame.copy()
    if dtype is not None:
        out[column] = out[column].astype(dtype)
    if row is None:
        out[column] = value
    else:
        out.loc[row, column] = value
    return out


@pytest.fixture(scope="module")
def small_fit():
    """One real torch fit of the tiny screen, shared by the fit and gather tests."""
    data = pm.prepare_model_data(tiny_screen())
    return data, pm.fit_model(data, backend="torch", n_steps=20, n_draws=12,
                              seed=0)


# ---------------------------------------------------------------------------
# prepare_model_data
# ---------------------------------------------------------------------------

def test_prepare_model_data_collapses_a_screen_or_refuses_it_by_name(subtests):
    """Mixed-type wells keep first-seen order; non-tables, bad counts and nothing imaged are refused."""
    mixed = pd.DataFrame({
        "gene": ["A", "B", "A", "B"],
        "well": ["w1", "w1", 2, 2],
        "positive": [1, 2, 3, 4],
        "n_reads_per_gene_per_well": [10, 20, 30, 40],
        "imaging_n_cells_per_gene_per_well": [50, 50, 50, 50],
    })
    with subtests.test(msg="unsortable well labels keep first-seen order"):
        data = pm.prepare_model_data(mixed)
        assert list(data.wells) == ["w1", 2]
        assert list(data.genes) == ["A", "B"]
        assert data.Npositive.tolist() == [3, 7]
        assert data.Ntotal.tolist() == [100, 100]
        np.testing.assert_allclose(
            data.log10expression,
            np.log10(np.array([[10 / 30, 20 / 30], [30 / 70, 40 / 70]])
                     + 1e-4))

    refusals = [
        ("not a table", {"gene": ["A"]},
         "prepare_model_data needs a pandas DataFrame, got dict."),
        ("non-numeric count",
         _with(tiny_screen(), "positive", 0, "a lot", dtype=object),
         "column 'positive' contains non-numeric entries, e.g. ['a lot']"),
        ("negative count", _with(tiny_screen(), "positive", 0, -3),
         "column 'positive' contains negative values (min -3). These are counts."),
        ("nothing imaged",
         _with(tiny_screen(), "imaging_n_cells_per_gene_per_well", None, 0),
         "every one of the 6 wells has Ntotal == 0"),
    ]
    for case_id, frame, fragment in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(pm.PowerFitError) as caught:
                pm.prepare_model_data(frame)
            assert fragment in str(caught.value)


# ---------------------------------------------------------------------------
# _horseshoe_global_scale
# ---------------------------------------------------------------------------

def test_horseshoe_global_scale_follows_piironen_vehtari_or_refuses(subtests):
    """An explicit scale wins; the default p0 is 5 % of genes; impossible inputs are refused."""
    accepted = [
        ("explicit scale ignores the rest", (100, 100, 5, 0.25), 0.25),
        ("expected hits", (100, 100, 5, None), 5 / 95 / 10),
        ("default p0 is 5 percent", (100, 100, None, None), 5 / 95 / 10),
        ("default p0 floored at one", (10, 4, None, None), 1 / 9 / 2),
        ("zero wells count as one", (10, 0, 1, None), 1 / 9),
    ]
    for case_id, args, expected in accepted:
        with subtests.test(msg=case_id):
            assert pm._horseshoe_global_scale(*args) == pytest.approx(
                expected, rel=1e-12)

    refused = [
        ("zero scale", (100, 100, None, 0.0), "scale_global must be a positive finite number, got 0.0."),
        ("negative scale", (100, 100, None, -1.0), "got -1.0."),
        ("infinite scale", (100, 100, None, math.inf), "got inf."),
        ("nan scale", (100, 100, None, math.nan), "got nan."),
        ("more hits than genes", (10, 100, 20, None),
         "expected_hits must be in (0, n_genes=10), got 20.0."),
        ("as many hits as genes", (10, 100, 10, None), "got 10.0."),
        ("no hits", (10, 100, 0, None), "got 0.0."),
    ]
    for case_id, args, fragment in refused:
        with subtests.test(msg=case_id):
            with pytest.raises(pm.PowerFitError) as caught:
                pm._horseshoe_global_scale(*args)
            assert fragment in str(caught.value)


# ---------------------------------------------------------------------------
# fit_model
# ---------------------------------------------------------------------------

def test_fit_model_returns_a_labelled_posterior_or_refuses_to_fit(subtests, small_fit):
    """A fit holds n_draws x n_genes draws; non-data, empty, single-well and diverged fits raise."""
    data, fit = small_fit
    with subtests.test(msg="shape and labelling of a fit"):
        assert (fit.backend, fit.requested_backend, fit.method) == ("torch", "torch", "advi")
        assert fit.n_draws == 12
        assert fit.draws.shape == (12, 4)
        assert list(fit.genes) == ["G0", "G1", "G2", "G3"]
        assert fit.data is data and fit.seed == 0

    empty = pm.ModelData(
        wells=np.array([]), genes=np.array([]),
        Npositive=np.array([]), Ntotal=np.array([]),
        log10expression=np.zeros((0, 0)))
    refusals = [
        ("not model data", pd.DataFrame(), {},
         "fit_model needs the ModelData returned by prepare_model_data, got DataFrame."),
        ("empty design", empty, {"backend": "torch"},
         "nothing to fit: the design is 0 wells x 0 genes."),
        ("single well", pm.prepare_model_data(tiny_screen(n_wells=1)),
         {"backend": "torch"}, "a single well cannot identify any gene effect"),
        ("diverged objective", data,
         {"backend": "torch", "n_steps": 200, "learning_rate": 1e12, "seed": 0},
         "ADVI objective went non-finite"),
    ]
    for case_id, model_data, kwargs, fragment in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(pm.PowerFitError) as caught:
                pm.fit_model(model_data, **kwargs)
            assert fragment in str(caught.value)


# ---------------------------------------------------------------------------
# gather_model_estimate
# ---------------------------------------------------------------------------

def _refit(fit, draws):
    """``fit`` with its draws replaced, everything else kept."""
    return pm.PowerFit(
        backend=fit.backend, requested_backend=fit.requested_backend,
        method=fit.method, draws=draws, intercept_draws=fit.intercept_draws,
        genes=fit.genes, converged=fit.converged)


def test_gather_model_estimate_summarises_draws_or_refuses_mislabelled_ones(subtests, small_fit):
    """Exact per-gene summaries from hand draws; non-fits and misaligned draws are refused."""
    _data, fit = small_fit
    hand = pm.PowerFit(
        backend="torch", requested_backend="torch", method="advi",
        draws=np.array([[1.0, 0.0, np.nan], [3.0, -2.0, np.nan]]),
        intercept_draws=np.array([-4.0, -4.0]),
        genes=np.array(["A", "B", "C"]), converged=True)
    with subtests.test(msg="hand draws, one unidentified gene"):
        out = pm.gather_model_estimate(hand)
        assert out["gene"].tolist() == ["A", "B", "C"]
        assert out["variable"].tolist() == [
            "b_log10expressionA", "b_log10expressionB", "b_log10expressionC"]
        np.testing.assert_allclose(out["mean"], [2.0, -1.0, np.nan])
        np.testing.assert_allclose(out["sd"], [math.sqrt(2.0), math.sqrt(2.0), np.nan])
        np.testing.assert_allclose(out["q5"], [1.1, -1.9, np.nan])
        np.testing.assert_allclose(out["q95"], [2.9, -0.1, np.nan])
        np.testing.assert_allclose(out["prob_positive"], [1.0, 0.0, np.nan])
        assert out["identified"].tolist() == [True, True, False]

    with subtests.test(msg="a real fit gives one identified row per gene"):
        out = pm.gather_model_estimate(fit)
        assert out["gene"].tolist() == ["G0", "G1", "G2", "G3"]
        np.testing.assert_allclose(out["mean"], fit.draws.mean(axis=0))

    refusals = [
        ("not a fit", pd.DataFrame(), "gather_model_estimate needs a PowerFit, got DataFrame."),
        ("one column short", _refit(fit, fit.draws[:, :-1]),
         "fit.draws has 3 columns but there are 4 genes; the coefficient labelling"),
        ("flattened draws", _refit(fit, fit.draws.ravel()),
         "fit.draws must be (n_draws, n_genes), got shape (48,)."),
    ]
    for case_id, broken, fragment in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(pm.PowerFitError) as caught:
                pm.gather_model_estimate(broken)
            assert fragment in str(caught.value)


# ---------------------------------------------------------------------------
# evaluate_model_fit
# ---------------------------------------------------------------------------

def _truth(genes, hits):
    return pd.DataFrame({"gene": list(genes), "hit": list(hits)})


def _estimate(genes, means):
    return pd.DataFrame({"gene": list(genes), "mean": list(means)})


def test_evaluate_model_fit_scores_or_says_why_it_cannot(subtests, caplog):
    """Defined screens get AUROC/AP; degenerate ones get NaN and a reason, never 0.5."""
    nan = float("nan")
    cases = [
        ("well separated", _truth("ABCD", [True, False, False, False]),
         _estimate("ABCD", [5.0, 0.1, 0.0, -1.0]),
         dict(model_ap=1.0, model_auroc=1.0, ap_baseline=0.25, n_genes_scored=4,
              n_hits=1, n_non_hits=3, n_unidentified_dropped=0, n_missing_truth=0),
         "", None),
        ("inverted ranking", _truth("ABCD", [True, False, False, False]),
         _estimate("ABCD", [-5.0, 0.1, 0.0, 1.0]),
         dict(model_ap=0.25, model_auroc=0.0, ap_baseline=0.25, n_genes_scored=4,
              n_hits=1, n_non_hits=3),
         "", None),
        ("no hits", _truth("AB", [False, False]), _estimate("AB", [1.0, 0.0]),
         dict(model_ap=nan, model_auroc=nan, ap_baseline=nan, n_hits=0, n_non_hits=2),
         "there is no positive class to rank). This is NOT chance", None),
        ("every gene a hit", _truth("AB", [True, True]), _estimate("AB", [1.0, 0.0]),
         dict(model_auroc=nan, model_ap=nan, n_hits=2, n_non_hits=0),
         "there is no negative class to rank against", None),
        ("non-finite estimate", _truth("ABC", [True, False, False]),
         _estimate("ABC", [np.inf, 0.0, 1.0]),
         dict(model_ap=nan, model_auroc=nan, n_genes_scored=3),
         "the coefficient estimates contain non-finite values", None),
        ("unestimable coefficient", _truth("ABC", [True, False, False]),
         _estimate("ABC", [1.0, np.nan, 0.0]),
         dict(model_ap=1.0, model_auroc=1.0, ap_baseline=0.5, n_genes_scored=2,
              n_unidentified_dropped=1, n_missing_truth=0),
         "", "They are not 'not hits' -- they are untested."),
        ("no overlap", _truth("A", [True]), _estimate("Z", [1.0]),
         dict(model_ap=nan, n_genes_scored=0, n_missing_truth=1),
         "no gene had both an estimate and a ground-truth label, so there is nothing to score",
         "1 of 1 estimated genes have no ground-truth hit status"),
    ]
    for case_id, truth, estimate, expected, reason, logged in cases:
        with subtests.test(msg=case_id):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="spacr.power_model"):
                row = pm.evaluate_model_fit(truth, estimate).iloc[0]
            for key, value in expected.items():
                if isinstance(value, float) and math.isnan(value):
                    assert math.isnan(row[key]), key
                else:
                    assert row[key] == value, key
            if reason:
                assert reason in row["reason"]
            else:
                assert row["reason"] == ""
            if logged:
                assert logged in caplog.text

    refusals = [
        ("truth not a table", {"gene": ["A"], "hit": [True]}, _estimate("A", [1.0]),
         "evaluate_model_fit needs data as a DataFrame, got dict."),
        ("estimate not a table", _truth("A", [True]), [1.0],
         "evaluate_model_fit needs model_estimate as a DataFrame, got list."),
        ("truth without hit", pd.DataFrame({"gene": ["A"]}), _estimate("A", [1.0]),
         "evaluate_model_fit: data is missing column(s) ['hit']. Present: ['gene']"),
        ("estimate without mean", _truth("A", [True]), pd.DataFrame({"gene": ["A"]}),
         "evaluate_model_fit: model_estimate is missing column(s) ['mean']."),
    ]
    for case_id, truth, estimate, fragment in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(pm.PowerFitError) as caught:
                pm.evaluate_model_fit(truth, estimate)
            assert fragment in str(caught.value)


# ---------------------------------------------------------------------------
# scan_parameters
# ---------------------------------------------------------------------------

def _scan(**kwargs):
    """``scan_parameters`` on the tiny screen with its unconverged-point warning absorbed."""
    options = {"simulate_fn": tiny_screen, "backend": "torch", "fit_kwargs": _FIT}
    options.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pm.scan_parameters(**options)


def _explodes(**_kwargs):
    raise RuntimeError("the simulator fell over")


def test_scan_parameters_records_cancels_and_fails_points_honestly(subtests, caplog):
    """One row per point, failures as 'failed' with NaN, exact-False cancels, raise on request."""
    with subtests.test(msg="one row per point and replicate"):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="spacr.power_model"):
            frame = _scan(seed=1, verbose=True, n_genes=[3, 4])
        assert list(frame["n_genes"]) == [3, 4]
        assert list(frame["param_index"]) == [1, 2]
        assert list(frame["replicate"]) == [0, 0]
        assert list(frame["status"]) == ["not_converged", "not_converged"]
        assert list(frame["method"]) == ["advi", "advi"]
        assert list(frame["seed_channel"]) == ["seed", "seed"]
        assert list(frame["n_wells_fit"]) == [6, 6]
        assert list(frame["n_hits"]) == [1, 1]
        assert list(frame["ap_baseline"]) == pytest.approx([1 / 3, 1 / 4])
        assert frame["model_ap"].isna().all() and frame["model_auroc"].isna().all()
        assert frame.attrs == {"n_planned": 2, "cancelled": False}
        assert "point 1/2 replicate 0: n_genes=3" in caplog.text
        assert "point 2 replicate 0 -> status=not_converged" in caplog.text

    with subtests.test(msg="a crashing point is recorded as failed"):
        with pytest.warns(RuntimeWarning, match="1 of 1 point"):
            frame = pm.scan_parameters(simulate_fn=_explodes, backend="torch",
                                       fit_kwargs=_FIT, n_genes=[3])
        row = frame.iloc[0]
        assert row["status"] == "failed"
        assert row["error"] == "RuntimeError: the simulator fell over"
        assert math.isnan(row["model_ap"]) and math.isnan(row["model_auroc"])
        assert row["n_genes_scored"] == 0 and row["method"] == ""

    with subtests.test(msg="on_error='raise' re-raises the first failure"):
        with pytest.raises(RuntimeError, match="the simulator fell over"):
            pm.scan_parameters(simulate_fn=_explodes, backend="torch",
                               on_error="raise", fit_kwargs=_FIT, n_genes=[3])

    with subtests.test(msg="a callback returning exactly False stops the sweep"):
        seen = []

        def _stop(event):
            seen.append((event["index"], event["total"], event["point_index"],
                         event["n_points"], event["resumed"]))
            return False

        frame = _scan(on_point=_stop, n_genes=[3, 4, 5])
        assert seen == [(1, 3, 1, 3, False)]
        assert len(frame) == 1
        assert frame.attrs == {"n_planned": 3, "cancelled": True}

    for case_id, verdict in (("empty list", []), ("None", None), ("zero", 0)):
        with subtests.test(msg=f"a falsy {case_id} does not stop the sweep"):
            frame = _scan(on_point=lambda event, v=verdict: v, n_genes=[3, 4])
            assert len(frame) == 2
            assert frame.attrs["cancelled"] is False


def test_scan_parameters_resumes_from_its_progress_file(subtests, tmp_path):
    """Restored rows are not recomputed, still reach on_point, can be cancelled, and never append twice."""
    progress = tmp_path / "runs" / "sweep.tsv"
    first = _scan(progress_file=str(progress), seed=3, n_genes=[3, 4])
    with subtests.test(msg="the first run writes its rows and creates the folder"):
        written = pd.read_csv(progress, sep="\t")
        assert len(first) == 2
        assert written["run_key"].tolist() == first["run_key"].tolist()
        assert written["status"].tolist() == ["not_converged", "not_converged"]

    with subtests.test(msg="a resume skips finished points and reports them"):
        ran, events = [], []

        def _watch(**kwargs):
            ran.append(kwargs)
            return tiny_screen(**kwargs)

        again = _scan(progress_file=str(progress), simulate_fn=_watch, seed=3,
                      on_point=events.append, n_genes=[3, 4])
        assert ran == []
        assert list(again["n_genes"]) == [3, 4]
        assert again["run_key"].tolist() == first["run_key"].tolist()
        assert [e["resumed"] for e in events] == [True, True]
        assert [e["index"] for e in events] == [1, 2]
        assert again.attrs == {"n_planned": 2, "cancelled": False}

    with subtests.test(msg="a restored row can cancel the sweep"):
        frame = _scan(progress_file=str(progress), seed=3,
                      on_point=lambda event: False, n_genes=[3, 4])
        assert len(frame) == 1
        assert frame.attrs == {"n_planned": 2, "cancelled": True}

    with subtests.test(msg="resume=False onto a non-empty file is refused"):
        with pytest.raises(pm.PowerFitError) as caught:
            _scan(progress_file=str(progress), seed=3, resume=False, n_genes=[3, 4])
        assert "already holds 2 row(s) and resume=False" in str(caught.value)
        assert len(pd.read_csv(progress, sep="\t")) == 2
