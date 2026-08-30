"""A sweep is judged by what its rows say, not by how many trials ran.

The value of a sweep row is that it carries the design it fitted, where the
named controls landed, and why a trial failed — so a trial that dies still
produces a row. These tests drive the orchestration with injected runners so
no regression is actually fitted, and read the rows and the CSV the sweep
writes as it goes.
"""
from __future__ import annotations

import builtins
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import parameter_sweep as ps


@pytest.fixture
def small_space():
    """Two trials, so a sweep finishes in milliseconds."""
    return ps.SweepSpace(axes={"regression_type": ["ols", "ridge"]},
                         fixed={"alpha": 1})


@pytest.fixture
def coefficients():
    """A coefficient table with a positive and a negative control in it."""
    return pd.DataFrame({
        "grna": ["gra14_1", "eaf1_1", "other_1", "other_2"],
        "coefficient": [0.9, -0.7, 0.1, -0.05],
        "p_value": [0.001, 0.004, 0.4, 0.9],
        "q_value": [0.01, 0.02, 0.6, 0.95],
    })


# ---------------------------------------------------------------------------
# enumerating trials
# ---------------------------------------------------------------------------

def test_a_mode_that_is_neither_grid_nor_random_is_refused():
    """A typo must not silently enumerate in some third order."""
    with pytest.raises(ValueError, match="'grid' or 'random'"):
        ps.build_trials(ps.SweepSpace(), mode="exhaustive")


# ---------------------------------------------------------------------------
# where the controls landed
# ---------------------------------------------------------------------------

def test_an_empty_result_table_says_nothing_about_the_controls():
    """No coefficients is no evidence, not an absent control."""
    assert ps._named_control_rows(None, {"positive": "gra14"}) == {}
    assert ps._named_control_rows(pd.DataFrame(), {"positive": "gra14"}) == {}


def test_a_table_with_no_label_column_says_nothing():
    """Without a name column there is nothing to look a control up by."""
    frame = pd.DataFrame({"coefficient": [0.5], "p_value": [0.01]})

    assert ps._named_control_rows(frame, {"positive": "gra14"}) == {}


def test_a_control_that_is_not_there_is_recorded_as_absent(coefficients):
    """A setting that loses the positive control is not a setting to use."""
    out = ps._named_control_rows(coefficients, {"positive": "never_seen"})

    assert out == {"positive_present": False}


def test_a_control_carries_its_effect_rank_and_both_p_values(coefficients):
    """The row answers "did this setting recover the control, and how well"."""
    out = ps._named_control_rows(coefficients,
                                 {"positive": "gra14", "negative": "eaf1"})

    assert out["positive_present"] is True
    assert out["positive_effect"] == pytest.approx(0.9)
    assert out["positive_rank"] == 1
    assert out["positive_q"] == pytest.approx(0.01)
    assert out["positive_p"] == pytest.approx(0.001)
    assert out["negative_rank"] == 2


def test_without_an_effect_column_there_is_no_rank(coefficients):
    """Rank is by effect size; with no effect there is nothing to rank by."""
    out = ps._named_control_rows(coefficients.drop(columns=["coefficient"]),
                                 {"positive": "gra14"})

    assert out["positive_present"] is True
    assert "positive_rank" not in out
    assert "positive_effect" not in out
    assert out["positive_q"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# how much data reached the fit
# ---------------------------------------------------------------------------

def test_a_design_summary_is_empty_for_something_that_is_not_a_mapping():
    """A runner that returned a frame has no design to summarise."""
    assert ps._design_summary(None) == {}
    assert ps._design_summary(pd.DataFrame()) == {}


def test_the_design_summary_counts_wells_guides_and_rows():
    """A hit count means little without the size of the fit it came from."""
    frame = pd.DataFrame({"prc": ["p1_A_1", "p1_A_1", "p1_A_2"],
                          "grna": ["g1", "g2", "g1"]})

    summary = ps._design_summary({"model_data": frame, "n_cells": 900})

    assert summary["n_wells"] == 2
    assert summary["n_guides"] == 2
    assert summary["n_cells"] == 900
    assert summary["n_rows_fitted"] == 3


def test_a_count_that_is_not_a_number_is_left_out(caplog):
    """A malformed count is dropped rather than crashing the summary."""
    summary = ps._design_summary({"n_wells": "lots", "n_cells": None})

    assert "n_wells" not in summary
    assert "n_cells" not in summary


# ---------------------------------------------------------------------------
# corrections from one fit
# ---------------------------------------------------------------------------

def test_without_p_values_each_correction_is_named_and_left_blank():
    """Thirteen methods still produce thirteen rows, with nothing in them."""
    rows = ps.correction_rows({}, ["fdr_bh", "bonferroni"])

    assert rows == [{"multiple_testing_method": "fdr_bh"},
                    {"multiple_testing_method": "bonferroni"}]
    assert ps.correction_rows({"results": pd.DataFrame({"a": [1]})},
                              ["holm"]) == [{"multiple_testing_method": "holm"}]


def test_each_correction_reports_the_methods_own_verdict(coefficients):
    """Step-down methods do not reduce to "adjusted <= alpha" afterwards."""
    rows = ps.correction_rows({"results": coefficients},
                              ["fdr_bh", "bonferroni", "holm"], alpha=0.05)

    assert [row["multiple_testing_method"] for row in rows] == \
        ["fdr_bh", "bonferroni", "holm"]
    for row in rows:
        assert row["n_tests"] == 4
        assert 0 <= row["n_below_alpha"] <= 4
        assert 0.0 <= row["smallest_adjusted_p"] <= 1.0
    assert "correction_error" not in rows[0]


def test_one_broken_correction_does_not_sink_the_rest(coefficients):
    """A method that raises leaves its own row carrying the reason."""
    rows = ps.correction_rows({"results": coefficients},
                              ["fdr_bh", "not_a_method"])

    assert "correction_error" in rows[1]
    assert "n_below_alpha" not in rows[1]
    assert rows[0]["n_tests"] == 4


def test_p_values_that_are_all_missing_leave_the_smallest_adjusted_as_nan():
    """No usable p-value is NaN, not zero."""
    frame = pd.DataFrame({"grna": ["a", "b"], "p_value": ["x", "y"]})

    rows = ps.correction_rows({"results": frame}, ["fdr_bh"])

    assert rows[0]["n_tests"] == 0
    assert np.isnan(rows[0]["smallest_adjusted_p"])


# ---------------------------------------------------------------------------
# what the machine can afford
# ---------------------------------------------------------------------------

def test_without_systemd_run_there_is_no_containment(monkeypatch):
    """Containment is claimed only when the tool is actually there."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    ps.containment_available.cache_clear()
    try:
        assert ps.containment_available() is False
    finally:
        ps.containment_available.cache_clear()


def test_a_meminfo_without_memavailable_reports_no_limit(monkeypatch):
    """An unreadable memory figure must not stop a sweep starting."""
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return io.StringIO("MemTotal:  1000 kB\nSwapFree:  0 kB\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    assert ps.free_memory_gb() == float("inf")


# ---------------------------------------------------------------------------
# the sequential sweep
# ---------------------------------------------------------------------------

def test_an_in_process_sweep_writes_one_row_per_trial(tmp_path, small_space,
                                                      coefficients):
    """Each trial's row carries its settings, its status and its timing."""
    def runner(settings):
        return {"results": coefficients, "model_data": pd.DataFrame(
            {"prc": ["p1_A_1", "p1_A_2"], "grna": ["g1", "g2"]})}

    results = ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out",
                           small_space, contained=False, runner=runner,
                           controls={"positive": "gra14"}, progress_every=0)

    assert list(results["status"]) == ["ok", "ok"]
    assert set(results["regression_type"]) == {"ols", "ridge"}
    assert (results["positive_present"]).all()
    assert results["n_wells"].tolist() == [2, 2]
    assert (tmp_path / "out" / "sweep_trials.json").is_file()


def test_a_correction_axis_becomes_one_row_per_method_from_one_fit(
        tmp_path, coefficients):
    """Thirteen fits for thirteen numbers that all come from the first one."""
    space = ps.SweepSpace(axes={"regression_type": ["ols"]}, fixed={"alpha": 1})

    def runner(settings):
        return {"results": coefficients}

    results = ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out", space,
                           contained=False, runner=runner,
                           corrections=["fdr_bh", "bonferroni"],
                           progress_every=0)

    assert len(results) == 2
    assert set(results["multiple_testing_method"]) == {"fdr_bh", "bonferroni"}
    assert results["trial_id"].nunique() == 1
    assert results["n_tests"].tolist() == [4, 4]


def test_a_failed_trial_is_a_row_with_a_reason(tmp_path, small_space):
    """A sweep that stopped at the first exception would waste the rest."""
    def runner(settings):
        raise MemoryError("not enough memory for this design")

    results = ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out",
                           small_space, contained=False, runner=runner,
                           corrections=["fdr_bh"], learn_from_failures=0,
                           progress_every=0)

    assert list(results["status"]) == ["failed", "failed"]
    assert set(results["error_type"]) == {"MemoryError"}
    assert set(results["multiple_testing_method"]) == {"fdr_bh"}
    folder = results.iloc[0]["folder"]
    assert os.path.isfile(os.path.join(folder, "error.txt"))


def test_progress_writes_the_partial_csv_as_it_goes(tmp_path, small_space,
                                                    coefficients, capsys):
    """A sweep left running unattended is the one whose partials matter."""
    def runner(settings):
        return {"results": coefficients}

    ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out", small_space,
                 contained=False, runner=runner, progress_every=1)

    out = capsys.readouterr().out
    assert "[sweep] 1/2 trials" in out
    assert "min left" in out
    written = pd.read_csv(tmp_path / "out" / "sweep_results.csv")
    assert len(written) == 2


def test_a_contained_sweep_takes_the_childs_row_whole(tmp_path, small_space,
                                                      monkeypatch, capsys):
    """The child returns a finished row; the parent must not re-decorate it."""
    seen = []

    def fake_child(settings, *, trial_id=None, controls=None, **kwargs):
        seen.append(trial_id)
        return {"trial_id": trial_id, "status": "ok", "seconds": 1.5,
                "n_hits": 7}

    monkeypatch.setattr(ps, "run_trial_contained", fake_child)

    results = ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out",
                           small_space, contained=True, memory_floor_gb=0.0,
                           progress_every=1)

    assert len(seen) == 2
    assert list(results["status"]) == ["ok", "ok"]
    assert results["n_hits"].tolist() == [7, 7]
    assert results["seconds"].tolist() == [1.5, 1.5]
    assert "[sweep] 1/2 trials (ok)" in capsys.readouterr().out
    assert pd.read_csv(tmp_path / "out" / "sweep_results.csv").shape[0] == 2


def test_a_contained_sweep_stops_repeating_a_failing_signature(
        tmp_path, monkeypatch):
    """Two failures of one model family is enough; the rest are skipped."""
    space = ps.SweepSpace(
        axes={"regression_type": ["ols"], "fraction_threshold": [0.1, 0.2, 0.3]},
        fixed={"alpha": 1})
    calls = []

    def fake_child(settings, *, trial_id=None, controls=None, **kwargs):
        calls.append(trial_id)
        return {"trial_id": trial_id, "status": "failed",
                "error_type": "MemoryError", "seconds": 0.1}

    monkeypatch.setattr(ps, "run_trial_contained", fake_child)

    results = ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out", space,
                           contained=True, learn_from_failures=2,
                           memory_floor_gb=0.0, progress_every=0)

    assert len(calls) == 2, "the third trial of the same signature is skipped"
    assert list(results["status"])[-1] == "skipped"


def test_a_sweep_stops_at_the_free_memory_floor(tmp_path, small_space,
                                                monkeypatch, capsys):
    """A sweep yields rather than competing with the rest of the machine."""
    monkeypatch.setattr(ps, "free_memory_gb", lambda: 0.5)
    monkeypatch.setattr(ps, "run_trial_contained",
                        lambda *a, **k: {"status": "ok"})

    results = ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out",
                           small_space, contained=True, memory_floor_gb=4.0,
                           progress_every=0)

    assert list(results["status"]) == ["skipped"]
    assert "stopped at the free-memory floor" in results.iloc[0]["error"]
    assert "below the" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the parallel sweep
# ---------------------------------------------------------------------------

class _InlineExecutor:
    """A pool that runs each job where it was submitted.

    The point of the test is the SUBMISSION policy — how many jobs are in
    flight, when the memory floor holds one back, what a dead worker turns
    into — and none of that needs a second interpreter.
    """

    def __init__(self, max_workers=1, mp_context=None):
        self.max_workers = max_workers
        self.submitted = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, payload):
        from concurrent.futures import Future

        future = Future()
        self.submitted.append(payload)
        try:
            future.set_result(fn(payload))
        except BaseException as error:                    # noqa: BLE001
            future.set_exception(error)
        return future


@pytest.fixture
def inline_pool(monkeypatch):
    """Run the parallel sweep's jobs in this process."""
    import concurrent.futures as cf

    made = []

    def factory(max_workers=1, mp_context=None):
        pool = _InlineExecutor(max_workers=max_workers)
        made.append(pool)
        return pool

    monkeypatch.setattr(cf, "ProcessPoolExecutor", factory)
    return made


def test_the_parallel_sweep_writes_a_row_per_trial(tmp_path, small_space,
                                                   inline_pool, monkeypatch,
                                                   capsys):
    """Every trial's row reaches the CSV as it finishes, not at the end."""
    monkeypatch.setattr(ps, "_execute_trial",
                        lambda payload: {"trial_id": payload[1]["trial_id"],
                                         "status": "ok", "seconds": 0.2})

    results = ps.run_sweep_parallel({"src": str(tmp_path)}, tmp_path / "out",
                                    small_space, n_jobs=2, progress_every=1)

    assert list(results["status"]) == ["ok", "ok"]
    assert (tmp_path / "out" / "sweep_results.csv").is_file()
    trials = json.loads((tmp_path / "out" / "sweep_trials.json").read_text())
    assert len(trials) == 2
    out = capsys.readouterr().out
    assert "2 trials across" in out
    assert "1/2 done" in out


def test_a_worker_that_dies_becomes_a_failed_row(tmp_path, small_space,
                                                 inline_pool, monkeypatch):
    """A BrokenProcessPool is a result, not the end of the sweep."""
    def die(payload):
        raise RuntimeError("A child process terminated abruptly")

    monkeypatch.setattr(ps, "_execute_trial", die)

    results = ps.run_sweep_parallel({"src": str(tmp_path)}, tmp_path / "out",
                                    small_space, n_jobs=1, progress_every=0)

    assert list(results["status"]) == ["failed", "failed"]
    assert set(results["error_type"]) == {"RuntimeError"}
    assert "terminated abruptly" in results.iloc[0]["error"]


def test_the_pool_holds_back_when_memory_is_low(tmp_path, inline_pool,
                                                monkeypatch, capsys):
    """Keeping n_jobs in flight is what lets the memory floor bite."""
    space = ps.SweepSpace(
        axes={"regression_type": ["ols"], "fraction_threshold": [0.1, 0.2]},
        fixed={"alpha": 1})
    answers = iter([False, True, False, False, False, False])
    monkeypatch.setattr(ps, "memory_is_low",
                        lambda *a, **k: next(answers, False))
    monkeypatch.setattr(ps, "recommended_workers",
                        lambda *a, **k: (2, "test capacity"))
    monkeypatch.setattr(ps, "_execute_trial",
                        lambda payload: {"trial_id": payload[1]["trial_id"],
                                         "status": "ok", "seconds": 0.1})

    results = ps.run_sweep_parallel({"src": str(tmp_path)}, tmp_path / "out",
                                    space, n_jobs=2, progress_every=0)

    assert len(results) == 2
    assert "held back" in capsys.readouterr().out


def test_the_parallel_sweep_refuses_to_run_inside_a_worker(tmp_path,
                                                           small_space,
                                                           monkeypatch):
    """A script with no main guard fork-bombs itself; say so instead."""
    import multiprocessing

    class NotMain:
        name = "SpawnPoolWorker-1"

    monkeypatch.setattr(multiprocessing, "current_process", lambda: NotMain())

    with pytest.raises(RuntimeError) as caught:
        ps.run_sweep_parallel({"src": str(tmp_path)}, tmp_path / "out",
                              small_space, n_jobs=1)

    assert '__main__' in str(caught.value)
    assert "guard" in str(caught.value)


# ---------------------------------------------------------------------------
# reading the sweep back
# ---------------------------------------------------------------------------

def test_ranking_by_a_control_leaves_a_table_that_has_no_such_column():
    """A sweep that never named a control cannot be ranked by one."""
    frame = pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "ok"]})

    assert ps.rank_trials(frame, role="positive") is frame
    assert ps.rank_trials(None, role="positive") is None


def test_ranking_leaves_a_table_whose_percentiles_are_all_missing():
    """With nothing to sort by, the order is left exactly as it was."""
    frame = pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "ok"],
                          "positive_control_percentile": [np.nan, np.nan]})

    assert ps.rank_trials(frame, role="positive") is frame


def test_a_failed_trial_never_outranks_one_that_ran():
    """NaN last, and a failure is not a good control recovery."""
    frame = pd.DataFrame({
        "trial_id": [1, 2, 3],
        "status": ["failed", "ok", "ok"],
        "positive_control_percentile": [0.01, 0.9, 0.2],
    })

    ordered = ps.rank_trials(frame, role="positive")

    assert ordered["trial_id"].tolist() == [3, 2, 1]


def test_an_empty_sweep_summarises_as_no_trials():
    """Zero trials is a stated zero, not an exception."""
    assert ps.summarise_sweep(pd.DataFrame()) == {"trials": 0}


def test_the_summary_counts_failures_by_reason():
    """The reasons are what tell a user which axis to narrow."""
    frame = pd.DataFrame({
        "status": ["ok", "failed", "failed"],
        "seconds": [60.0, 30.0, 30.0],
        "error_type": [None, "MemoryError", "MemoryError"],
    })

    summary = ps.summarise_sweep(frame)

    assert summary["trials"] == 3
    assert summary["succeeded"] == 1
    assert summary["failed"] == 2
    assert summary["total_minutes"] == 2.0
    assert summary["failure_reasons"] == {"MemoryError": 2}


def test_a_row_reproduces_its_own_regression_settings():
    """A sweep row carries every setting, so a trial can be re-run from it."""
    row = {"trial_id": 3, "status": "ok", "seconds": 12.0,
           "regression_type": "ridge", "alpha": "1",
           "positive_present": True, "positive_rank": 1,
           "folder": "/tmp/trial_3"}

    settings = ps.settings_for_trial({"src": "/data"}, row)

    assert settings["src"] == "/tmp/trial_3", "the trial's own folder wins"
    assert settings["regression_type"] == "ridge"
    assert settings["alpha"] == 1, "a stored literal is read back as one"
    assert "positive_present" not in settings
    assert "status" not in settings


# ---------------------------------------------------------------------------
# containment, and what happens without it
# ---------------------------------------------------------------------------

def test_the_containment_note_states_the_actual_limits(monkeypatch):
    """The user is told which caps are on, and what happens when one bites."""
    monkeypatch.setattr(ps, "containment_available", lambda: True)

    note = ps.containment_note()

    assert ps.TRIAL_MEMORY_MAX in note
    assert ps.TRIAL_CPU_QUOTA in note
    assert "only that trial is stopped" in note


def test_without_containment_the_note_says_what_is_left(monkeypatch):
    """An uncapped sweep is a decision the user should make knowingly."""
    monkeypatch.setattr(ps, "containment_available", lambda: False)

    note = ps.containment_note()

    assert "unavailable" in note
    assert "cannot prevent a single trial from exhausting system memory" in note
    assert "Reduce the worker count" in note


def test_a_contained_trial_returns_the_row_its_child_wrote(tmp_path,
                                                           monkeypatch):
    """The child's JSON is the row; the parent does not rebuild it."""
    import subprocess

    monkeypatch.setattr(ps, "containment_available", lambda: False)

    def fake_run(command, **kwargs):
        out_path = command[-1]
        with open(out_path, "w") as handle:
            json.dump({"trial_id": 7, "status": "ok", "n_hits": 3,
                       "seconds": 2.5}, handle)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    row = ps.run_trial_contained({"src": str(tmp_path)}, trial_id=7)

    assert row == {"trial_id": 7, "status": "ok", "n_hits": 3, "seconds": 2.5}


def test_a_child_that_wrote_nothing_is_reported_as_killed(tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """"Killed" and "crashed" want different responses from the user."""
    import subprocess

    monkeypatch.setattr(ps, "containment_available", lambda: False)
    monkeypatch.setattr(
        subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 137, "", "Killed"))

    row = ps.run_trial_contained({"src": str(tmp_path)}, trial_id=9)

    assert row["status"] == "killed"
    assert row["error_type"] == "MemoryMax"
    assert "WITHOUT a memory cap" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# one trial, in a pool worker
# ---------------------------------------------------------------------------

def test_an_executed_trial_that_raises_leaves_a_row_and_a_traceback(
        tmp_path, monkeypatch):
    """A failed trial is a result, written where the trial's folder is."""
    from spacr import ml

    def explode(settings):
        raise MemoryError("the design matrix did not fit")

    monkeypatch.setattr(ml, "perform_regression", explode)
    trial = {"trial_id": 1, "regression_type": "ols"}

    row = ps._execute_trial(({"src": str(tmp_path)}, trial,
                             str(tmp_path / "out"), {}, False))

    assert row["status"] == "failed"
    assert row["error_type"] == "MemoryError"
    assert "design matrix" in row["error"]
    assert os.path.isfile(os.path.join(row["folder"], "error.txt"))
    assert row["seconds"] >= 0


def test_an_executed_trial_records_what_the_fit_produced(tmp_path,
                                                         monkeypatch,
                                                         coefficients):
    """The same columns whichever way the trial ran."""
    from spacr import ml

    monkeypatch.setattr(ml, "perform_regression", lambda settings: {
        "results": coefficients,
        "model_data": pd.DataFrame({"prc": ["p1_A_1"], "grna": ["g1"]}),
    })
    trial = {"trial_id": 2, "regression_type": "ols"}

    row = ps._execute_trial(({"src": str(tmp_path)}, trial,
                             str(tmp_path / "out"), {"positive": "gra14"},
                             False))

    assert row["status"] == "ok"
    assert row["n_wells"] == 1
    assert row["positive_present"] is True


# ---------------------------------------------------------------------------
# the sweep's own bookkeeping
# ---------------------------------------------------------------------------

def test_the_default_runner_is_the_regression(tmp_path, small_space,
                                              monkeypatch, coefficients):
    """``contained=False`` with no runner uses spaCR's own regression."""
    from spacr import ml

    called = []

    def runner(settings):
        called.append(settings["regression_type"])
        return {"results": coefficients}

    monkeypatch.setattr(ml, "perform_regression", runner)

    results = ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out",
                           small_space, contained=False, progress_every=0)

    assert called == ["ols", "ridge"]
    assert list(results["status"]) == ["ok", "ok"]


def test_a_csv_that_cannot_be_written_does_not_stop_the_sweep(
        tmp_path, small_space, monkeypatch, coefficients):
    """The results write is best-effort; losing it must not lose the sweep.

    The point of writing after every trial is that "a sweep killed halfway
    still leaves a usable table rather than nothing" — so a write that cannot
    happen has to leave the sweep's own return value intact, exactly as the
    guarded progress write ten lines above it already does.
    """
    def refuse(self, *args, **kwargs):
        raise OSError("the results disk is full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", refuse)

    results = ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out",
                           small_space, contained=False,
                           runner=lambda settings: {"results": coefficients},
                           progress_every=1)

    assert list(results["status"]) == ["ok", "ok"]


def test_a_contained_csv_that_cannot_be_written_does_not_stop_the_sweep(
        tmp_path, small_space, monkeypatch):
    """The contained path writes its partials the same best-effort way."""
    monkeypatch.setattr(ps, "run_trial_contained",
                        lambda *a, **k: {"status": "ok", "seconds": 0.1})

    def refuse(self, *args, **kwargs):
        raise OSError("the results disk is full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", refuse)

    results = ps.run_sweep({"src": str(tmp_path)}, tmp_path / "out",
                           small_space, contained=True, memory_floor_gb=0.0,
                           progress_every=1)

    assert list(results["status"]) == ["ok", "ok"]


def test_a_missing_value_in_a_row_is_not_a_setting():
    """A NaN in the CSV means "this trial had none", not "set it to NaN"."""
    row = {"trial_id": 1, "status": "ok", "seconds": 1.0,
           "regression_type": "ols", "alpha": float("nan")}

    settings = ps.settings_for_trial({"src": "/data"}, row)

    assert "alpha" not in settings
    assert settings["regression_type"] == "ols"


def test_a_traceback_that_cannot_be_written_still_leaves_the_failed_row(
        tmp_path, monkeypatch):
    """The row is the result; the traceback file beside it is a bonus."""
    from spacr import ml

    def explode(settings):
        raise MemoryError("the design matrix did not fit")

    monkeypatch.setattr(ml, "perform_regression", explode)
    real_open = builtins.open

    def refuse(path, *args, **kwargs):
        if str(path).endswith("error.txt"):
            raise OSError("read-only file system")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", refuse)
    trial = {"trial_id": 1, "regression_type": "ols"}

    row = ps._execute_trial(({"src": str(tmp_path)}, trial,
                             str(tmp_path / "out"), {}, False))

    monkeypatch.undo()
    assert row["status"] == "failed"
    assert row["error_type"] == "MemoryError"
    assert not os.path.exists(os.path.join(row["folder"], "error.txt"))
