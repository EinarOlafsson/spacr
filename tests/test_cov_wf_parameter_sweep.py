"""The sweep must survive trials whose output is not the shape it hoped for.

Every path here is a place where the sweep meets a partial answer: a
coefficient table with no q column, a fit that reports only its significant
rows, a runner that hands back something that is not a mapping at all, a
results table where the positive control was never recovered, and a row a user
clicked without ever choosing a folder for the rerun. Each of those is a real
condition -- different regression families and corrections emit different
keys -- and each one used to be reachable only through a real fit, so nothing
exercised it. If any of them regressed the user would see a crash in the
middle of a sweep, or a summary claiming a control rank nothing measured,
rather than a row that says what it knows.

The regressions are never actually fitted: runners and
``spacr.ml.perform_regression`` are injected, and the parallel pool runs its
jobs in this process.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from spacr import parameter_sweep as ps


@pytest.fixture
def coefficients():
    """A coefficient table with the positive control in it, effects and p."""
    return pd.DataFrame({
        "grna": ["gra14_1", "other_1", "other_2"],
        "coefficient": [0.9, 0.1, -0.05],
        "p_value": [0.001, 0.4, 0.9],
    })


@pytest.fixture
def quiet_worker(monkeypatch):
    """Keep ``_execute_trial``'s politeness out of the pytest process.

    ``be_polite`` calls ``os.nice(19)`` and writes ``oom_score_adj``, which
    are irreversible for the test session; ``_pin_threads`` resizes live BLAS
    pools. Neither is what these tests are about.
    """
    monkeypatch.setattr(ps, "be_polite", lambda: None)
    monkeypatch.setattr(ps, "_pin_threads", lambda *a, **k: None)


def _fake_regression(monkeypatch, answers):
    """Make ``spacr.ml.perform_regression`` return the next canned answer."""
    import spacr.ml as ml

    seen = []

    def fake(settings):
        seen.append(dict(settings))
        return answers[len(seen) - 1]

    monkeypatch.setattr(ml, "perform_regression", fake)
    return seen


# ---------------------------------------------------------------------------
# a partial coefficient table
# ---------------------------------------------------------------------------

def test_a_control_without_a_q_column_still_reports_its_raw_p(coefficients):
    """Not every correction writes a q column, and the row must not invent one.

    ``quantile`` and the permutation families report a raw or permutation p
    and no adjusted value at all. Writing a q of NaN, or crashing on the
    missing column, would make the control row for those families unusable --
    and the control row is the only thing that says whether the setting is
    worth keeping.
    """
    without_q = ps._named_control_rows(coefficients, {"positive": "gra14"})

    with_q = ps._named_control_rows(
        coefficients.assign(q_value=[0.01, 0.6, 0.95]),
        {"positive": "gra14"})

    assert without_q == {"positive_present": True, "positive_effect": 0.9,
                         "positive_rank": 1, "positive_p": 0.001}
    assert "positive_q" not in without_q
    assert with_q["positive_q"] == 0.01
    assert with_q["positive_p"] == 0.001


def test_a_trial_that_only_reports_significant_rows_has_no_hit_count():
    """The hit count comes from the full table, not from the surviving rows.

    ``n_below_alpha`` is what ``summarise_sweep`` compares across settings, and
    it is only meaningful when the trial handed back every coefficient with its
    q value. A fit that reports just its significant rows must therefore
    contribute the count it does have and stay silent about the one it does
    not, rather than reporting "3 hits below alpha" from a table that was
    already filtered.
    """
    significant_only = ps._count_hits(
        {"significant": pd.DataFrame({"grna": ["a", "b"]})})

    full_table = ps._count_hits({"results": pd.DataFrame(
        {"grna": ["a", "b", "c"], "q_value": [0.01, 0.2, 0.9]})})

    assert significant_only == {"n_significant": 2}
    assert "n_below_alpha" not in significant_only
    assert full_table == {"n_results": 3, "n_below_alpha": 1}


# ---------------------------------------------------------------------------
# the pool worker meeting an unexpected output
# ---------------------------------------------------------------------------

def test_a_trial_whose_fit_returns_nothing_is_still_a_timed_row(
        tmp_path, monkeypatch, quiet_worker):
    """A pool worker must never die on the shape of what came back.

    ``_execute_trial`` runs in a spawned process, so an exception here is a
    BrokenProcessPool and the loss of every row that worker had not yet
    returned. A fit that hands back ``None`` instead of a result mapping has to
    become a plain row -- trial id, folder, seconds -- exactly like the one a
    normal fit produces, only without the metrics nobody could compute.
    """
    seen = _fake_regression(monkeypatch, [
        None,
        {"results": pd.DataFrame({"grna": ["a", "b"], "q_value": [0.01, 0.9]})},
    ])

    empty = ps._execute_trial(
        ({}, {"trial_id": 1, "regression_type": "ols"}, str(tmp_path), {},
         False, False))
    full = ps._execute_trial(
        ({}, {"trial_id": 2, "regression_type": "ridge"}, str(tmp_path), {},
         False, False))

    assert empty["status"] == "ok"
    assert empty["trial_id"] == 1
    assert empty["regression_type"] == "ols"
    assert "n_results" not in empty
    assert isinstance(empty["seconds"], float)
    assert full["n_results"] == 2
    assert full["n_below_alpha"] == 1
    assert [s["src"] for s in seen] == [empty["folder"], full["folder"]]


def test_a_fit_with_no_results_table_reports_no_control_at_all(
        tmp_path, monkeypatch, quiet_worker):
    """An absent control column and a lost control must not look the same.

    ``{alias}_present`` is written as ``False`` for every alias whenever a
    results table was read, so ``positive_present=False`` means "this setting
    lost the control". A trial that reported no coefficient table at all can
    make no such claim, and writing the column anyway would put a fabricated
    failure into the sweep summary's control-recovery count.
    """
    controls = {"positive": "gra14"}
    _fake_regression(monkeypatch, [
        {"significant": pd.DataFrame({"grna": ["gra14_1"]})},
        {"results": pd.DataFrame({"grna": ["gra14_1", "x"],
                                  "coefficient": [0.9, 0.1],
                                  "q_value": [0.01, 0.9]})},
    ])

    no_table = ps._execute_trial(
        ({}, {"trial_id": 1}, str(tmp_path), controls, False, False))
    with_table = ps._execute_trial(
        ({}, {"trial_id": 2}, str(tmp_path), controls, False, False))

    assert no_table["status"] == "ok"
    assert no_table["n_significant"] == 1
    assert "positive_present" not in no_table
    assert with_table["positive_present"] is True
    assert with_table["positive_rank"] == 1


# ---------------------------------------------------------------------------
# the parallel sweep keeps the caller's own filters
# ---------------------------------------------------------------------------

class _InlineExecutor:
    """A ProcessPoolExecutor that runs each job here, in this interpreter.

    The submission policy is what these tests read; a second interpreter would
    add seconds and a spawned import of torch to prove nothing extra.
    """

    def __init__(self, max_workers=1, mp_context=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, payload):
        from concurrent.futures import Future

        future = Future()
        try:
            future.set_result(fn(payload))
        except BaseException as error:                    # noqa: BLE001
            future.set_exception(error)
        return future


@pytest.fixture
def inline_pool(monkeypatch):
    """Run the parallel sweep's jobs in this process."""
    import concurrent.futures as cf

    monkeypatch.setattr(cf, "ProcessPoolExecutor", _InlineExecutor)


def test_the_parallel_sweep_obeys_the_filters_it_was_given(
        tmp_path, inline_pool, monkeypatch):
    """A caller's own filters must not be swapped for the built-in ones.

    The sweep screen builds a space whose filters encode what THIS dataset
    cannot fit. Replacing them with the defaults whenever they happen to be
    supplied would run the combination the caller explicitly excluded -- which,
    for the cell-level permutation, is a measured 57 GB fit and a dead desktop.
    """
    rejected = []

    def no_ridge(trial):
        if trial.get("regression_type") == "ridge":
            rejected.append(trial["regression_type"])
            return "ridge is excluded for this dataset"
        return None

    space = ps.SweepSpace(axes={"regression_type": ["ols", "ridge"]},
                          fixed={"alpha": 1}, filters=[no_ridge])
    monkeypatch.setattr(ps, "memory_is_low", lambda *a, **k: False)
    monkeypatch.setattr(ps, "_execute_trial",
                        lambda payload: {"trial_id": payload[1]["trial_id"],
                                         "regression_type":
                                             payload[1]["regression_type"],
                                         "status": "ok", "seconds": 0.1})

    results = ps.run_sweep_parallel({"src": str(tmp_path)}, tmp_path / "out",
                                    space, n_jobs=1, progress_every=0)

    assert space.filters == [no_ridge], "the caller's filters were replaced"
    assert rejected == ["ridge"]
    assert list(results["regression_type"]) == ["ols"]
    trials = json.loads((tmp_path / "out" / "sweep_trials.json").read_text())
    assert [t["regression_type"] for t in trials] == ["ols"]


# ---------------------------------------------------------------------------
# the in-process sweep meeting an unexpected output
# ---------------------------------------------------------------------------

def test_an_in_process_trial_that_returns_no_mapping_still_lands_in_the_table(
        tmp_path):
    """One odd return value must not cost the sweep every later trial.

    ``run_sweep`` writes the CSV as it goes precisely so an interrupted sweep
    is still worth something. A runner that hands back a list rather than a
    result mapping has to be recorded as a finished trial with no metrics, so
    the trials after it still run.
    """
    answers = {"ols": ["not", "a", "mapping"],
               "ridge": {"results": pd.DataFrame(
                   {"grna": ["a", "b"], "q_value": [0.01, 0.4]})}}
    space = ps.SweepSpace(axes={"regression_type": ["ols", "ridge"]},
                          fixed={"alpha": 1})

    results = ps.run_sweep({}, tmp_path, space, progress_every=0,
                           runner=lambda settings:
                               answers[settings["regression_type"]])

    by_family = results.set_index("regression_type")
    assert list(results["status"]) == ["ok", "ok"]
    assert "n_results" not in results.columns or \
        pd.isna(by_family.loc["ols", "n_results"])
    assert by_family.loc["ridge", "n_results"] == 2
    assert by_family.loc["ridge", "n_below_alpha"] == 1
    assert (tmp_path / "sweep_results.csv").is_file()


def test_an_in_process_trial_with_no_results_frame_claims_no_control(tmp_path):
    """Control recovery is only reported by a trial that had a table to read.

    ``summarise_sweep`` counts ``positive_present`` across trials to answer
    "did this screen recover its control". A trial that produced no coefficient
    table must not contribute a False to that count, or a sweep of settings
    that mostly report summaries alone would read as a screen that keeps losing
    its positive control.
    """
    frame = pd.DataFrame({"grna": ["gra14_1", "x"], "coefficient": [0.9, 0.1],
                          "q_value": [0.01, 0.9]})
    answers = {"ols": {"significant": frame.head(1)}, "ridge": {"results": frame}}
    space = ps.SweepSpace(axes={"regression_type": ["ols", "ridge"]},
                          fixed={"alpha": 1})

    results = ps.run_sweep({}, tmp_path, space, progress_every=0,
                           controls={"positive": "gra14"},
                           runner=lambda settings:
                               answers[settings["regression_type"]])

    by_family = results.set_index("regression_type")
    assert by_family.loc["ols", "n_significant"] == 1
    assert pd.isna(by_family.loc["ols", "positive_present"])
    assert by_family.loc["ridge", "positive_present"] is True
    assert by_family.loc["ridge", "positive_rank"] == 1


# ---------------------------------------------------------------------------
# summarising a sweep that never found the control
# ---------------------------------------------------------------------------

def test_a_sweep_that_never_recovered_the_control_reports_no_best_rank():
    """"Best rank" must come from a trial, never from an empty selection.

    ``positive_control_best_rank`` is the headline of the summary: it names the
    trial a user should open first. When no trial recovered the control there
    is no such trial, and the summary has to say ``0/N`` and stop -- reporting a
    rank taken from an empty table, or crashing on it, would send the user to a
    setting that never found anything.
    """
    columns = {"trial_id": [1, 2], "status": ["ok", "ok"],
               "seconds": [1.0, 2.0], "n_below_alpha": [3, 9]}
    never = pd.DataFrame({**columns,
                          "positive_control_rank": [float("nan")] * 2})
    once = pd.DataFrame({**columns, "positive_control_rank": [float("nan"), 4]})

    lost = ps.summarise_sweep(never, controls=())
    found = ps.summarise_sweep(once, controls=())

    assert lost["positive_control_recovered_in"] == "0/2 trials"
    assert "positive_control_best_rank" not in lost
    assert "positive_control_median_rank" not in lost
    assert lost["hits_range"] == [3, 9]
    assert found["positive_control_recovered_in"] == "1/2 trials"
    assert found["positive_control_best_rank"] == 4
    assert found["positive_control_best_trial"] == 2


# ---------------------------------------------------------------------------
# reopening a row that names no folder
# ---------------------------------------------------------------------------

def test_reopening_a_row_with_no_folder_creates_no_directory(tmp_path,
                                                             monkeypatch):
    """Clicking a row is a look, and a look must not litter the disk.

    A row read back from a CSV that was moved, or one built in memory before
    the sweep chose a destination, names no folder. ``rerun_trial`` has to fit
    it and hand back the figures anyway rather than calling ``makedirs`` on
    ``None``; and when a destination IS given, that folder has to appear,
    because the rerun writes into it.
    """
    import matplotlib.pyplot as plt

    made = []

    def fake(settings):
        made.append(plt.figure())
        return {"results": pd.DataFrame({"grna": ["a"], "q_value": [0.01]})}

    import spacr.ml as ml
    monkeypatch.setattr(ml, "perform_regression", fake)
    row = {"trial_id": 7, "regression_type": "ridge", "status": "ok"}

    try:
        homeless = ps.rerun_trial({}, row)
        assert "src" not in homeless["settings"]
        assert list(tmp_path.iterdir()) == [], \
            "a rerun with no folder still wrote to disk"

        housed = ps.rerun_trial({}, row, destination=str(tmp_path / "reopened"))
    finally:
        for figure in made:
            plt.close(figure)

    assert homeless["settings"]["regression_type"] == "ridge"
    assert homeless["settings"]["verbose"] is True
    assert len(homeless["figures"]) == 1
    assert homeless["output"]["results"].iloc[0]["grna"] == "a"
    assert housed["settings"]["src"] == str(tmp_path / "reopened")
    assert (tmp_path / "reopened").is_dir()
