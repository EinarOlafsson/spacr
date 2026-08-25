"""A power model that refuses to put a number on a fit it does not have.

The failures here are the interesting part: a design that cannot identify
anything, a sweep point that crashed, a screen with no hits in it. Each of
those has a wrong answer that still plots -- 0.5, or a backfilled zero --
and the tests say what the module does instead.

Everything runs the real torch ADVI fit on a four-gene, six-well design,
which takes well under a second; the optional NUTS backends are not
installed in this environment and are not reached from here.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from spacr import power_model as pm


def tiny_screen(n_genes: int = 4, n_wells: int = 6, seed: int = 0,
                **_ignored) -> pd.DataFrame:
    """A small tidy screen in the shape ``prepare_model_data`` reads."""
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


@pytest.fixture
def model_data():
    return pm.prepare_model_data(tiny_screen())


# --------------------------------------------------------------------------
# preparing the design
# --------------------------------------------------------------------------

def test_something_that_is_not_a_table_is_refused_by_name():
    with pytest.raises(pm.PowerFitError, match="needs a pandas DataFrame"):
        pm.prepare_model_data({"gene": ["A"]})


def test_wells_named_in_types_that_do_not_compare_keep_their_first_seen_order():
    """Sorting is only there to make two runs agree; a design whose labels
    cannot be sorted must still fit rather than raise from inside pandas."""
    frame = pd.DataFrame({
        "gene": ["A", "B", "A", "B"],
        "well": ["w1", "w1", 2, 2],
        "positive": [1, 2, 3, 4],
        "n_reads_per_gene_per_well": [10, 20, 30, 40],
        "imaging_n_cells_per_gene_per_well": [50, 50, 50, 50],
    })
    data = pm.prepare_model_data(frame)
    assert list(data.wells) == ["w1", 2]


def test_a_count_column_that_does_not_parse_names_the_offending_values():
    """Coercing them to 0 would understate the well."""
    frame = tiny_screen()
    frame["positive"] = frame["positive"].astype(object)
    frame.loc[0, "positive"] = "a lot"
    with pytest.raises(pm.PowerFitError, match="non-numeric entries"):
        pm.prepare_model_data(frame)


def test_a_negative_count_is_refused_as_not_a_count():
    frame = tiny_screen()
    frame.loc[0, "positive"] = -3
    with pytest.raises(pm.PowerFitError, match="These are counts"):
        pm.prepare_model_data(frame)


def test_a_screen_where_nothing_was_imaged_has_nothing_to_fit():
    """log(Ntotal) is -inf everywhere, and a fit on that is not a fit."""
    frame = tiny_screen()
    frame["imaging_n_cells_per_gene_per_well"] = 0
    with pytest.raises(pm.PowerFitError, match="nothing to fit"):
        pm.prepare_model_data(frame)


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

def test_a_shadowed_or_half_removed_package_counts_as_not_installed(
        monkeypatch):
    """``find_spec`` itself can raise on a namespace-package shadow; that is
    'not usable', not a crash on start-up."""
    import importlib.util

    def _confused(name):
        raise ValueError("__spec__ is not set")

    monkeypatch.setattr(importlib.util, "find_spec", _confused)
    assert pm._module_installed("numpyro") is False


def test_the_backends_report_says_what_this_interpreter_can_run():
    available = pm.available_backends()
    assert available["torch"] is True
    assert set(available) >= {"torch", "numpyro", "pymc"}


# --------------------------------------------------------------------------
# the horseshoe prior's global scale
# --------------------------------------------------------------------------

def test_a_scale_given_outright_is_used_and_the_rest_ignored():
    assert pm._horseshoe_global_scale(100, 100, 5, 0.25) == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_a_scale_that_is_not_a_positive_finite_number_is_refused(bad):
    with pytest.raises(pm.PowerFitError, match="positive finite number"):
        pm._horseshoe_global_scale(100, 100, None, bad)


def test_expecting_more_hits_than_there_are_genes_is_refused():
    with pytest.raises(pm.PowerFitError):
        pm._horseshoe_global_scale(10, 100, 20, None)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def test_a_fit_reports_how_many_draws_it_holds(model_data):
    fit = pm.fit_model(model_data, backend="torch", n_steps=20, n_draws=12,
                       seed=0)
    assert fit.n_draws == 12 == fit.draws.shape[0]
    assert fit.draws.shape[1] == model_data.n_genes


def test_something_that_is_not_model_data_is_refused_by_name():
    with pytest.raises(pm.PowerFitError, match="needs the ModelData"):
        pm.fit_model(pd.DataFrame())


def test_a_single_well_cannot_identify_any_gene_effect():
    frame = tiny_screen(n_wells=1)
    with pytest.raises(pm.PowerFitError, match="single well"):
        pm.fit_model(pm.prepare_model_data(frame), backend="torch")


def test_an_empty_design_has_nothing_to_fit():
    data = pm.ModelData(
        wells=np.array([]), genes=np.array([]),
        Npositive=np.array([]), Ntotal=np.array([]),
        log10expression=np.zeros((0, 0)))
    with pytest.raises(pm.PowerFitError, match="nothing to fit"):
        pm.fit_model(data, backend="torch")


def test_an_objective_that_goes_non_finite_stops_rather_than_reporting_noise(
        model_data):
    """Every estimate downstream of a diverged optimisation would be noise
    wearing a number."""
    with pytest.raises(pm.PowerFitError, match="ADVI objective went "
                                               "non-finite"):
        pm.fit_model(model_data, backend="torch", n_steps=200,
                     learning_rate=1e12, seed=0)


# --------------------------------------------------------------------------
# gathering the estimate
# --------------------------------------------------------------------------

def test_something_that_is_not_a_fit_is_refused_by_name():
    with pytest.raises(pm.PowerFitError, match="needs a PowerFit"):
        pm.gather_model_estimate(pd.DataFrame())


def test_draws_that_do_not_line_up_with_the_genes_are_refused(model_data):
    """The coefficient labelling would be wrong, and a mislabelled hit list
    is worse than no hit list."""
    fit = pm.fit_model(model_data, backend="torch", n_steps=20, n_draws=8,
                       seed=0)
    broken = pm.PowerFit(
        backend=fit.backend, requested_backend=fit.requested_backend,
        method=fit.method, draws=fit.draws[:, :-1],
        intercept_draws=fit.intercept_draws, genes=fit.genes,
        converged=fit.converged)
    with pytest.raises(pm.PowerFitError, match="coefficient labelling"):
        pm.gather_model_estimate(broken)


def test_draws_of_the_wrong_shape_entirely_are_refused(model_data):
    fit = pm.fit_model(model_data, backend="torch", n_steps=20, n_draws=8,
                       seed=0)
    broken = pm.PowerFit(
        backend=fit.backend, requested_backend=fit.requested_backend,
        method=fit.method, draws=fit.draws.ravel(),
        intercept_draws=fit.intercept_draws, genes=fit.genes,
        converged=fit.converged)
    with pytest.raises(pm.PowerFitError, match=r"must be \(n_draws"):
        pm.gather_model_estimate(broken)


# --------------------------------------------------------------------------
# scoring, and refusing to score
# --------------------------------------------------------------------------

def _estimate(genes, means):
    return pd.DataFrame({"gene": list(genes), "mean": list(means)})


def test_a_truth_table_without_the_columns_it_needs_is_refused():
    with pytest.raises(pm.PowerFitError):
        pm.evaluate_model_fit(pd.DataFrame({"gene": ["A"]}),
                              _estimate(["A"], [1.0]))


def test_something_that_is_not_a_table_is_refused_by_name_here_too():
    with pytest.raises(pm.PowerFitError, match="as a DataFrame"):
        pm.evaluate_model_fit({"gene": ["A"], "hit": [True]},
                              _estimate(["A"], [1.0]))


def test_a_screen_with_no_hits_is_not_reported_as_chance_performance():
    """Reporting 0.5 here would be a fabrication."""
    truth = pd.DataFrame({"gene": ["A", "B"], "hit": [False, False]})
    out = pm.evaluate_model_fit(truth, _estimate(["A", "B"], [1.0, 0.0]))
    row = out.iloc[0]
    assert np.isnan(row["model_auroc"]) and np.isnan(row["model_ap"])
    assert "no positive class" in row["reason"]


def test_a_screen_where_every_gene_is_a_hit_is_undefined_too():
    truth = pd.DataFrame({"gene": ["A", "B"], "hit": [True, True]})
    out = pm.evaluate_model_fit(truth, _estimate(["A", "B"], [1.0, 0.0]))
    assert "no negative class" in out.iloc[0]["reason"]
    assert np.isnan(out.iloc[0]["model_auroc"])


def test_non_finite_estimates_imply_no_ranking_at_all():
    truth = pd.DataFrame({"gene": ["A", "B", "C"],
                          "hit": [True, False, False]})
    out = pm.evaluate_model_fit(
        truth, _estimate(["A", "B", "C"], [np.inf, 0.0, 1.0]))
    assert "non-finite" in out.iloc[0]["reason"]
    assert np.isnan(out.iloc[0]["model_ap"])


def test_a_gene_with_no_estimable_coefficient_is_untested_not_a_non_hit(
        caplog):
    truth = pd.DataFrame({"gene": ["A", "B", "C"],
                          "hit": [True, False, False]})
    estimate = _estimate(["A", "B", "C"], [1.0, np.nan, 0.0])
    with caplog.at_level(logging.WARNING):
        out = pm.evaluate_model_fit(truth, estimate)
    assert int(out.iloc[0]["n_unidentified_dropped"]) == 1
    assert "they are not" in caplog.text.lower()


def test_no_gene_with_both_an_estimate_and_a_label_scores_nothing():
    truth = pd.DataFrame({"gene": ["A"], "hit": [True]})
    out = pm.evaluate_model_fit(truth, _estimate(["Z"], [1.0]))
    assert "nothing to score" in out.iloc[0]["reason"]


def test_a_well_separated_screen_scores_above_its_baseline():
    truth = pd.DataFrame({"gene": list("ABCD"),
                          "hit": [True, False, False, False]})
    out = pm.evaluate_model_fit(truth, _estimate(list("ABCD"),
                                                 [5.0, 0.1, 0.0, -1.0]))
    row = out.iloc[0]
    assert row["model_auroc"] == 1.0
    assert row["model_ap"] > row["ap_baseline"]
    assert row["reason"] == ""


# --------------------------------------------------------------------------
# sweep bookkeeping
# --------------------------------------------------------------------------

def test_a_string_is_one_value_and_not_a_sweep_over_its_characters():
    """The classic version of this bug: 'cellpose' as a four-way sweep."""
    assert pm._is_scalar("cellpose") is True
    assert pm._is_scalar(b"raw") is True
    assert pm._is_scalar(np.float64(3.0)) is True
    assert pm._is_scalar(np.array(3.0)) is True
    assert pm._is_scalar(np.array([1, 2])) is False
    assert pm._is_scalar([1, 2]) is False
    assert pm._is_scalar(range(3)) is False


def test_a_numpy_scalar_hashes_the_same_as_the_python_number_it_equals():
    """A grid built with ``np.linspace`` must resume a grid written by hand."""
    assert pm._jsonable(np.float64(2.5)) == 2.5
    assert pm._jsonable(np.int64(3)) == 3
    assert pm._jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert pm._jsonable([np.int64(1), 2]) == [1, 2]
    assert pm._jsonable(None) is None
    assert pm._jsonable({"a": 1}) == repr({"a": 1})


def test_the_same_point_keys_the_same_however_the_numbers_were_written():
    left = pm._run_key({"n_genes": np.int64(4)}, 0, 7, "torch")
    right = pm._run_key({"n_genes": 4}, 0, 7, "torch")
    assert left == right
    assert left != pm._run_key({"n_genes": 5}, 0, 7, "torch")


def test_a_missing_simulator_says_what_to_pass_instead(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if fromlist and "simulate_screen" in fromlist:
            raise ImportError("no simulator installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(pm.PowerFitError, match="Pass simulate_fn"):
        pm._default_simulator()


def test_the_seed_reaches_a_simulator_by_whichever_name_it_accepts():
    def by_seed(seed=0, **kw):
        return pd.DataFrame({"seed": [seed]})

    def by_rng(n_genes=4, rng=None):
        return pd.DataFrame({"kind": [type(rng).__name__]})

    def by_random_state(n_genes=4, random_state=None):
        return pd.DataFrame({"random_state": [random_state]})

    def by_nothing():
        return pd.DataFrame({"x": [1]})

    frame, channel = pm._call_simulator(by_seed, {}, 11)
    assert channel == "seed" and frame.iloc[0]["seed"] == 11

    frame, channel = pm._call_simulator(by_rng, {}, 11)
    assert channel == "rng" and frame.iloc[0]["kind"] == "Generator"

    frame, channel = pm._call_simulator(by_random_state, {}, 11)
    assert channel == "random_state" and frame.iloc[0]["random_state"] == 11

    _frame, channel = pm._call_simulator(by_nothing, {}, 11)
    assert channel == "numpy-global"


def test_a_simulator_with_no_readable_signature_is_given_the_seed_anyway():
    """Builtins and C callables have no signature; assuming they take
    nothing would silently unseed the sweep."""
    calls = {}

    class Opaque:
        def __call__(self, **kwargs):
            calls.update(kwargs)
            return pd.DataFrame({"x": [1]})

        @property
        def __signature__(self):
            raise ValueError("no signature for this object")

    _frame, channel = pm._call_simulator(Opaque(), {"n_genes": 3}, 5)
    assert channel == "seed"
    assert calls == {"n_genes": 3, "seed": 5}


# --------------------------------------------------------------------------
# the sweep itself
# --------------------------------------------------------------------------

_FIT = {"n_steps": 20, "n_draws": 8}


def test_a_sweep_records_one_row_per_point_and_replicate(tmp_path):
    frame = pm.scan_parameters(
        simulate_fn=tiny_screen, backend="torch", seed=1,
        fit_kwargs=_FIT, n_genes=[3, 4])
    assert len(frame) == 2
    assert list(frame["n_genes"]) == [3, 4]
    assert frame.attrs["n_planned"] == 2
    assert frame.attrs["cancelled"] is False


def test_a_verbose_sweep_says_what_it_is_running_and_what_it_got(caplog):
    with caplog.at_level(logging.INFO, logger="spacr.power_model"):
        pm.scan_parameters(simulate_fn=tiny_screen, backend="torch", seed=2,
                           fit_kwargs=_FIT, verbose=True, n_genes=[3])
    assert "point 1/1 replicate 0" in caplog.text
    assert "status=" in caplog.text


def test_a_point_that_crashes_is_recorded_as_failed_not_as_chance():
    """A sweep that backfilled 0.5 would plot as a design sitting at
    chance, and 'cannot find its hits' and 'crashed' have opposite
    consequences."""
    def _explodes(**kwargs):
        raise RuntimeError("the simulator fell over")

    frame = pm.scan_parameters(simulate_fn=_explodes, backend="torch",
                               fit_kwargs=_FIT, n_genes=[3])
    row = frame.iloc[0]
    assert row["status"] == "failed"
    assert "RuntimeError" in row["error"]
    assert np.isnan(row["model_ap"]) and np.isnan(row["model_auroc"])


def test_a_sweep_can_be_told_to_raise_on_the_first_failure_instead():
    def _explodes(**kwargs):
        raise RuntimeError("the simulator fell over")

    with pytest.raises(RuntimeError, match="fell over"):
        pm.scan_parameters(simulate_fn=_explodes, backend="torch",
                           on_error="raise", fit_kwargs=_FIT, n_genes=[3])


def test_a_sweep_stops_when_the_callback_says_exactly_false():
    seen = []

    def _stop_after_one(event):
        seen.append(event["point_index"])
        return False

    frame = pm.scan_parameters(simulate_fn=tiny_screen, backend="torch",
                               fit_kwargs=_FIT, on_point=_stop_after_one,
                               n_genes=[3, 4, 5])
    assert seen == [1]
    assert len(frame) == 1
    assert frame.attrs["cancelled"] is True


def test_a_callback_returning_something_incidental_does_not_stop_the_sweep():
    """`is False`, not falsy: stopping a five-minute sweep on an empty list
    is not a decision to infer."""
    frame = pm.scan_parameters(simulate_fn=tiny_screen, backend="torch",
                               fit_kwargs=_FIT, on_point=lambda event: [],
                               n_genes=[3, 4])
    assert len(frame) == 2
    assert frame.attrs["cancelled"] is False


def test_a_resumed_sweep_skips_the_points_it_already_did(tmp_path):
    progress = tmp_path / "runs" / "sweep.tsv"
    first = pm.scan_parameters(progress_file=str(progress),
                               simulate_fn=tiny_screen, backend="torch",
                               seed=3, fit_kwargs=_FIT, n_genes=[3, 4])
    assert progress.exists() and len(first) == 2

    ran = []

    def _watch(**kwargs):
        ran.append(kwargs)
        return tiny_screen(**kwargs)

    again = pm.scan_parameters(progress_file=str(progress),
                               simulate_fn=_watch, backend="torch",
                               seed=3, fit_kwargs=_FIT, n_genes=[3, 4])
    assert ran == []                       # nothing was recomputed
    assert len(again) == 2
    assert list(again["n_genes"]) == [3, 4]


def test_a_resumed_sweep_still_reports_the_rows_it_restored(tmp_path):
    """A progress bar that skips the restored rows runs backwards on the
    second attempt."""
    progress = tmp_path / "sweep.tsv"
    pm.scan_parameters(progress_file=str(progress), simulate_fn=tiny_screen,
                       backend="torch", seed=4, fit_kwargs=_FIT,
                       n_genes=[3, 4])
    events = []
    pm.scan_parameters(progress_file=str(progress), simulate_fn=tiny_screen,
                       backend="torch", seed=4, fit_kwargs=_FIT,
                       on_point=events.append, n_genes=[3, 4])
    assert [e["resumed"] for e in events] == [True, True]
    assert [e["index"] for e in events] == [1, 2]


def test_a_resumed_sweep_can_be_cancelled_on_a_restored_row(tmp_path):
    progress = tmp_path / "sweep.tsv"
    pm.scan_parameters(progress_file=str(progress), simulate_fn=tiny_screen,
                       backend="torch", seed=5, fit_kwargs=_FIT,
                       n_genes=[3, 4])
    frame = pm.scan_parameters(progress_file=str(progress),
                               simulate_fn=tiny_screen, backend="torch",
                               seed=5, fit_kwargs=_FIT,
                               on_point=lambda event: False, n_genes=[3, 4])
    assert len(frame) == 1
    assert frame.attrs["cancelled"] is True


def test_refusing_to_resume_onto_a_file_that_already_has_rows(tmp_path):
    """Appending would put two rows with the same run_key in one file, and
    every later mean over it would double-count them."""
    progress = tmp_path / "sweep.tsv"
    pm.scan_parameters(progress_file=str(progress), simulate_fn=tiny_screen,
                       backend="torch", seed=6, fit_kwargs=_FIT, n_genes=[3])
    with pytest.raises(pm.PowerFitError, match="resume=False"):
        pm.scan_parameters(progress_file=str(progress),
                           simulate_fn=tiny_screen, backend="torch", seed=6,
                           resume=False, fit_kwargs=_FIT, n_genes=[3])


def test_the_default_simulator_is_the_one_the_port_ships_with():
    from spacr.power_simulate import simulate_screen

    assert pm._default_simulator() is simulate_screen
