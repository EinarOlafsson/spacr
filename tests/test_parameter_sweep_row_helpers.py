"""Table-driven tests of the helpers that build and read parameter-sweep rows.

A sweep row carries where the named controls landed, how much data reached
the fit, one verdict per multiple-testing correction, and why a trial failed;
the sweep table is then ranked and summarised from those rows.  One general
test per unit (``_named_control_rows``, ``_design_summary``,
``correction_rows``, ``rank_trials``, ``summarise_sweep``,
``_register_resource_workers``, ``memory_is_low``, ``run_trial_contained``,
``_execute_trial``), each walking a case table with pytest's ``subtests``
fixture and asserting exact values.  No regression is fitted: runners,
``spacr.ml.perform_regression``, the child process and the run context are
injected doubles.
"""
from __future__ import annotations

import builtins
import json
import os
import subprocess
import types

import numpy as np
import pandas as pd
import pytest

from spacr import parameter_sweep as ps


def _coefficients():
    """A coefficient table with a positive (gra14) and negative (eaf1) control."""
    return pd.DataFrame({
        "grna": ["gra14_1", "eaf1_1", "other_1", "other_2"],
        "coefficient": [0.9, -0.7, 0.1, -0.05],
        "p_value": [0.001, 0.004, 0.4, 0.9],
        "q_value": [0.01, 0.02, 0.6, 0.95],
    })


class _Context:
    """A run context that records the worker stamps registered against it."""

    def __init__(self, sampler=None, refuse=False):
        self.seen = []
        self.refuse = refuse
        if sampler is not None:
            self._resource_sampler = sampler

    def register_worker(self, stamp):
        if self.refuse:
            raise RuntimeError("the resource record is closed")
        self.seen.append(stamp)


# ---------------------------------------------------------------------------
# where the controls landed
# ---------------------------------------------------------------------------

def test_named_control_rows_report_presence_rank_effect_and_probabilities(subtests):
    """Each named control gets present/rank/effect/q/p from its best-ranked guide."""
    base = _coefficients()
    no_q = base.drop(columns=["q_value"]).iloc[[0, 2, 3]].reset_index(drop=True)
    cases = [
        ("no table", None, {"positive": "gra14"}, {}),
        ("empty table", pd.DataFrame(), {"positive": "gra14"}, {}),
        ("no label column",
         pd.DataFrame({"coefficient": [0.5], "p_value": [0.01]}),
         {"positive": "gra14"}, {}),
        ("control absent", base, {"positive": "never_seen"},
         {"positive_present": False}),
        ("two controls", base, {"positive": "gra14", "negative": "eaf1"},
         {"positive_present": True, "positive_rank": 1, "positive_effect": 0.9,
          "positive_q": 0.01, "positive_p": 0.001,
          "negative_present": True, "negative_rank": 2,
          "negative_effect": -0.7, "negative_q": 0.02, "negative_p": 0.004}),
        ("duplicate alias uses the best-ranked guide",
         pd.DataFrame({"grna": ["gra14_weak", "other_1", "gra14_strong"],
                       "coefficient": [0.02, 0.5, 0.95],
                       "p_value": [0.8, 0.03, 0.001],
                       "q_value": [0.99, 0.06, 0.005]}),
         {"positive": "gra14"},
         {"positive_present": True, "positive_rank": 1,
          "positive_effect": 0.95, "positive_q": 0.005, "positive_p": 0.001}),
        ("no effect column means no rank", base.drop(columns=["coefficient"]),
         {"positive": "gra14"},
         {"positive_present": True, "positive_q": 0.01, "positive_p": 0.001}),
        ("no q column reports only the raw p", no_q, {"positive": "gra14"},
         {"positive_present": True, "positive_effect": 0.9,
          "positive_rank": 1, "positive_p": 0.001}),
        ("NaN effects, substring needles and an absent control",
         pd.DataFrame({"grna": ["ctrl_pos_1", "TargetA_2", "ctrl_neg_3",
                                "TargetB_4"],
                       "coefficient": [0.9, np.nan, -0.4, 0.2],
                       "q_value": [0.01, 0.5, 0.2, 0.3]}),
         {"positive": "ctrl_pos", "negative": "ctrl_neg",
          "absent": "ctrl_missing"},
         {"positive_present": True, "positive_rank": 1,
          "positive_effect": 0.9, "positive_q": 0.01,
          "negative_present": True, "negative_rank": 2,
          "negative_effect": -0.4, "negative_q": 0.2,
          "absent_present": False}),
        ("ranked by absolute effect",
         pd.DataFrame({"grna": ["TSC2_1", "AAVS1_1", "TSC2_2"],
                       "coefficient": [0.2, -0.8, 0.4]}),
         {"control": "AAVS1"},
         {"control_present": True, "control_rank": 1,
          "control_effect": -0.8}),
        ("alternative column names, a missing p is left out",
         pd.DataFrame({"gene": ["A", "B"], "effect": [0.1, 0.5],
                       "adjusted_p_value": [0.2, 0.01],
                       "permutation_p_value": [0.3, np.nan]}),
         {"x": "B", "y": "A"},
         {"x_present": True, "x_rank": 1, "x_effect": 0.5, "x_q": 0.01,
          "y_present": True, "y_rank": 2, "y_effect": 0.1, "y_q": 0.2,
          "y_p": 0.3}),
    ]
    for case_id, frame, names, expected in cases:
        with subtests.test(msg=case_id):
            out = ps._named_control_rows(frame, names)
            assert out == expected
            for key, value in out.items():
                if key.endswith("_present"):
                    assert value is expected[key]


# ---------------------------------------------------------------------------
# how much data reached the fit
# ---------------------------------------------------------------------------

def test_design_summary_counts_wells_guides_rows_and_cells(subtests):
    """Only a mapping has a design; malformed counts are dropped, not raised."""
    cases = [
        ("None", None, {}),
        ("a frame is not a mapping", pd.DataFrame(), {}),
        ("model data and a cell count",
         {"model_data": pd.DataFrame({"prc": ["p1_A_1", "p1_A_1", "p1_A_2"],
                                      "grna": ["g1", "g2", "g1"]}),
          "n_cells": 900},
         {"n_rows_prepared": 3, "n_rows_fitted": 3, "n_wells": 2,
          "n_guides": 2, "n_cells": 900}),
        ("counts that are not numbers", {"n_wells": "lots", "n_cells": None},
         {}),
    ]
    for case_id, output, expected in cases:
        with subtests.test(msg=case_id):
            assert ps._design_summary(output) == expected


# ---------------------------------------------------------------------------
# corrections from one fit
# ---------------------------------------------------------------------------

def test_correction_rows_give_one_verdict_per_method_from_one_fit(subtests):
    """One row per method; no p-values leaves it blank, a bad method carries its error."""
    unknown = ("ValueError: Unsupported multiple-testing method "
               "'not_a_method'. Choose one of: none, bonferroni, sidak, holm, "
               "holm_sidak, simes_hochberg, hommel, fdr_bh, fdr_by, fdr_tsbh, "
               "fdr_tsbky, fdr_gbs, storey.")
    cases = [
        ("no output", {}, ["fdr_bh", "bonferroni"], 0.05,
         [{"multiple_testing_method": "fdr_bh"},
          {"multiple_testing_method": "bonferroni"}]),
        ("not a mapping", ["results"], ["holm"], 0.05,
         [{"multiple_testing_method": "holm"}]),
        ("results without p values", {"results": pd.DataFrame({"a": [1]})},
         ["holm"], 0.05, [{"multiple_testing_method": "holm"}]),
        ("three methods, each its own verdict",
         {"results": _coefficients()}, ["fdr_bh", "bonferroni", "holm"], 0.05,
         [{"multiple_testing_method": m, "n_below_alpha": 2, "n_tests": 4,
           "smallest_adjusted_p": 0.004}
          for m in ("fdr_bh", "bonferroni", "holm")]),
        ("a stricter alpha", {"results": _coefficients()}, ["bonferroni"],
         0.005,
         [{"multiple_testing_method": "bonferroni", "n_below_alpha": 1,
           "n_tests": 4, "smallest_adjusted_p": 0.004}]),
        ("one broken method does not sink the rest",
         {"results": _coefficients()}, ["fdr_bh", "not_a_method"], 0.05,
         [{"multiple_testing_method": "fdr_bh", "n_below_alpha": 2,
           "n_tests": 4, "smallest_adjusted_p": 0.004},
          {"multiple_testing_method": "not_a_method",
           "correction_error": unknown}]),
    ]
    for case_id, output, methods, alpha, expected in cases:
        with subtests.test(msg=case_id):
            rows = ps.correction_rows(output, methods, alpha=alpha)
            assert rows == [pytest.approx(row) if "smallest_adjusted_p" in row
                            else row for row in expected]

    with subtests.test(msg="unusable p values leave the smallest as NaN"):
        frame = pd.DataFrame({"grna": ["a", "b"], "p_value": ["x", "y"]})
        (row,) = ps.correction_rows({"results": frame}, ["fdr_bh"])
        assert {k: v for k, v in row.items() if k != "smallest_adjusted_p"} \
            == {"multiple_testing_method": "fdr_bh", "n_below_alpha": 0,
                "n_tests": 0}
        assert np.isnan(row["smallest_adjusted_p"])


# ---------------------------------------------------------------------------
# reading the sweep back
# ---------------------------------------------------------------------------

def test_rank_trials_orders_by_control_percentile_with_failures_last(subtests):
    """Untouched when nothing can rank; otherwise ascending, NaN and failed last."""
    no_column = pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "ok"]})
    all_missing = pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "ok"],
                                "positive_control_percentile": [np.nan, np.nan]})
    for case_id, frame in [("no such column", no_column),
                           ("all percentiles missing", all_missing),
                           ("empty table", pd.DataFrame()),
                           ("no table", None)]:
        with subtests.test(msg=case_id):
            assert ps.rank_trials(frame, role="positive") is frame

    ordered_cases = [
        ("a failed trial never outranks one that ran",
         pd.DataFrame({"trial_id": [1, 2, 3],
                       "status": ["failed", "ok", "ok"],
                       "positive_control_percentile": [0.01, 0.9, 0.2]}),
         "positive", [3, 2, 1]),
        ("the negative role reads its own column",
         pd.DataFrame({"trial_id": [1, 2, 3], "status": ["ok", "ok", "ok"],
                       "negative_control_percentile": [0.5, None, 0.1]}),
         "negative", [3, 1, 2]),
    ]
    for case_id, frame, role, expected in ordered_cases:
        with subtests.test(msg=case_id):
            ordered = ps.rank_trials(frame, role=role)
            assert ordered["trial_id"].tolist() == expected
            assert list(ordered.columns) == list(frame.columns)
            assert ordered.index.tolist() == list(range(len(frame)))


def test_summarise_sweep_counts_outcomes_controls_and_hits(subtests):
    """The summary states counts, failure reasons, control recovery and hit ranges."""
    columns = {"trial_id": [1, 2], "status": ["ok", "ok"],
               "seconds": [1.0, 2.0], "n_below_alpha": [3, 9]}
    cases = [
        ("empty sweep", pd.DataFrame(), {}, {"trials": 0}),
        ("failures by reason",
         pd.DataFrame({"status": ["ok", "failed", "failed"],
                       "seconds": [60.0, 30.0, 30.0],
                       "error_type": [None, "MemoryError", "MemoryError"]}),
         {},
         {"trials": 3, "succeeded": 1, "failed": 2, "total_minutes": 2.0,
          "failure_reasons": {"MemoryError": 2}}),
        ("control never recovered has no best rank",
         pd.DataFrame({**columns, "positive_control_rank": [np.nan] * 2}),
         {"controls": ()},
         {"trials": 2, "succeeded": 2, "failed": 0, "total_minutes": 0.1,
          "positive_control_recovered_in": "0/2 trials", "hits_median": 6.0,
          "hits_range": [3, 9]}),
        ("control recovered once names its trial",
         pd.DataFrame({**columns, "positive_control_rank": [np.nan, 4]}),
         {"controls": ()},
         {"trials": 2, "succeeded": 2, "failed": 0, "total_minutes": 0.1,
          "positive_control_recovered_in": "1/2 trials",
          "positive_control_best_rank": 4, "positive_control_best_trial": 2,
          "positive_control_median_rank": 4.0, "hits_median": 6.0,
          "hits_range": [3, 9]}),
        ("named control columns and hits by axis",
         pd.DataFrame({"trial_id": [1, 2, 3],
                       "status": ["ok", "ok", "failed"],
                       "seconds": [6.0, 6.0, 0.0],
                       "gra14_present": [True, False, np.nan],
                       "gra14_rank": [3.0, np.nan, np.nan],
                       "n_below_alpha": [4, 2, 0],
                       "regression_type": ["ols", "ridge", "ols"]}),
         {},
         {"trials": 3, "succeeded": 2, "failed": 1, "total_minutes": 0.2,
          "gra14_recovered_in": "1/2 trials", "gra14_median_rank": 3.0,
          "hits_median": 3.0, "hits_range": [2, 4],
          "hits_by_regression_type": {"ols": 4.0, "ridge": 2.0}}),
    ]
    for case_id, frame, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert ps.summarise_sweep(frame, **kwargs) == expected


# ---------------------------------------------------------------------------
# worker accounting and the memory guard
# ---------------------------------------------------------------------------

def test_register_resource_workers_moves_stamps_onto_the_run(subtests,
                                                             monkeypatch):
    """Private stamp columns go to the active run and never survive in the row."""
    from spacr import runctx

    cases = [
        ("list and single stamp registered in order",
         {"status": "ok", "_resource_workers": [{"pid": 1}, {"pid": 2}],
          "_resource_worker": {"pid": 3}},
         "context", {"status": "ok"}, [{"pid": 1}, {"pid": 2}, {"pid": 3}]),
        ("non-mapping stamps are ignored",
         {"_resource_workers": [{"pid": 1}, "not a stamp", 7],
          "_resource_worker": "nope"},
         "context", {}, [{"pid": 1}]),
        ("a tuple of stamps counts too",
         {"_resource_workers": ({"pid": 4},)}, "context", {}, [{"pid": 4}]),
        ("a non-list stamp column is dropped",
         {"_resource_workers": "nonsense"}, "context", {}, []),
        ("outside a run the row is still cleaned",
         {"status": "ok", "_resource_workers": [{"pid": 1}]}, None,
         {"status": "ok"}, []),
        ("a context that refuses does not lose the trial",
         {"status": "ok", "_resource_workers": [{"pid": 1}]}, "refuse",
         {"status": "ok"}, []),
    ]
    for case_id, row, kind, expected_row, expected_seen in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            context = None if kind is None else _Context(
                refuse=(kind == "refuse"))
            mp.setattr(runctx, "current_run_context", lambda: context)
            out = ps._register_resource_workers(dict(row))
            assert out == expected_row
            assert (context.seen if context else []) == expected_seen

    with subtests.test(msg="a runctx that will not import is survived"), \
            monkeypatch.context() as mp:
        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "runctx" in name or "current_run_context" in (fromlist or ()):
                raise ImportError("runctx is unavailable")
            return real(name, g, l, fromlist, level)

        mp.setattr(builtins, "__import__", refuse)
        out = ps._register_resource_workers(
            {"status": "ok", "_resource_workers": [{"pid": 1}]})
        assert out == {"status": "ok"}


def test_memory_is_low_trips_on_the_machine_floor_or_spacr_own_tree(
        subtests, monkeypatch):
    """The guard answers from free memory and, given a ceiling, spaCR's tree."""
    import psutil
    from spacr import runctx

    gib = 1024 ** 3

    class _Sampler:
        def __init__(self, tree_bytes):
            self._summary = {"last_tree_memory_bytes": tree_bytes}

    class _ClosedSampler:
        @property
        def _summary(self):
            raise RuntimeError("the sampler is closed")

    def broken_vm():
        raise RuntimeError("psutil is unreadable")

    free_10 = lambda: types.SimpleNamespace(available=10 * gib)  # noqa: E731
    cases = [
        # id, virtual_memory, context, floor, ceiling, expected, state
        ("free above the floor", free_10, None, 8.0, None, False,
         {"available_gib": 10.0, "spacr_tree_gib": None,
          "machine_low": False, "spacr_low": False}),
        ("free below the floor", free_10, None, 12.0, None, True,
         {"available_gib": 10.0, "spacr_tree_gib": None,
          "machine_low": True, "spacr_low": False}),
        ("spaCR's own tree over its ceiling", free_10,
         _Context(sampler=_Sampler(40 * gib)), 0.0, 1.0, True,
         {"available_gib": 10.0, "spacr_tree_gib": 40.0,
          "machine_low": False, "spacr_low": True}),
        ("spaCR's tree under its ceiling", free_10,
         _Context(sampler=_Sampler(40 * gib)), 0.0, 1000.0, False,
         {"available_gib": 10.0, "spacr_tree_gib": 40.0,
          "machine_low": False, "spacr_low": False}),
        ("a tree figure without a ceiling is only recorded", free_10,
         _Context(sampler=_Sampler(40 * gib)), 0.0, None, False,
         {"available_gib": 10.0, "spacr_tree_gib": 40.0,
          "machine_low": False, "spacr_low": False}),
        ("a closed sampler still leaves the machine check", free_10,
         _Context(sampler=_ClosedSampler()), 12.0, 1.0, True,
         {"available_gib": 10.0, "spacr_tree_gib": None,
          "machine_low": True, "spacr_low": False}),
        ("a context without a sampler", free_10, _Context(), 8.0, 1.0, False,
         {"available_gib": 10.0, "spacr_tree_gib": None,
          "machine_low": False, "spacr_low": False}),
        ("unreadable free memory is never low by itself", broken_vm, None,
         10 ** 9, None, False,
         {"available_gib": None, "spacr_tree_gib": None,
          "machine_low": False, "spacr_low": False}),
    ]
    for case_id, vm, context, floor, ceiling, expected, state in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            mp.setattr(psutil, "virtual_memory", vm)
            mp.setattr(runctx, "current_run_context", lambda: context)
            assert ps.memory_is_low(floor_gib=floor,
                                    spacr_ceiling_gib=ceiling) is expected
            assert ps._LAST_MEMORY_STATE == state

    with subtests.test(msg="this machine: impossible and zero floors"):
        assert ps.memory_is_low(floor_gib=10 ** 9) is True
        assert ps.memory_is_low(floor_gib=0.0) is False


# ---------------------------------------------------------------------------
# one trial in a contained child
# ---------------------------------------------------------------------------

def test_run_trial_contained_returns_the_childs_row_or_says_why_not(
        subtests, monkeypatch, tmp_path, capsys):
    """The child's JSON is the row; no JSON is a killed or timed-out row."""
    import multiprocessing

    child_row = {"trial_id": 7, "status": "ok", "n_hits": 3, "seconds": 2.5,
                 "_resource_worker": {"pid": 123}}
    killed_error = ("the trial was killed; it may have exceeded "
                    "MemoryMax=24G. Killed")
    cases = [
        # id, containment, process, child behaviour, expected row
        ("main process registers and strips the stamp", False,
         "MainProcess", ("write", child_row),
         {"trial_id": 7, "status": "ok", "n_hits": 3, "seconds": 2.5}),
        ("a pool worker passes the stamp on untouched", False,
         "SpawnPoolWorker-1", ("write", child_row), child_row),
        ("no result file after exit 137 is a kill", False, "MainProcess",
         ("exit", 137, "Killed"),
         {"status": "killed", "trial_id": 9, "error_type": "MemoryMax",
          "error": killed_error}),
        ("an unreadable result file is a kill too", False, "MainProcess",
         ("garbage", 137, "Killed"),
         {"status": "killed", "trial_id": 9, "error_type": "MemoryMax",
          "error": killed_error}),
        ("a timeout", True, "MainProcess", ("timeout",),
         {"status": "timeout", "trial_id": 9, "error_type": "Timeout",
          "error": ("the trial was killed; it may have exceeded "
                    "MemoryMax=24G. timed out")}),
    ]
    for case_id, contained, process, behaviour, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            folder = tmp_path / case_id.replace(" ", "_")
            folder.mkdir()
            out_path = folder / "_trial_result.json"
            # A pool worker's folder starts clean: no stale result to remove.
            if process == "MainProcess":
                out_path.write_text(json.dumps({"status": "stale"}),
                                    encoding="utf-8")
            commands = []
            registered = []

            def fake_run(command, **kwargs):
                commands.append(command)
                assert not out_path.exists(), "a stale result was reused"
                assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
                kind = behaviour[0]
                if kind == "timeout":
                    raise subprocess.TimeoutExpired(command, 1)
                if kind == "write":
                    out_path.write_text(json.dumps(behaviour[1]),
                                        encoding="utf-8")
                    return subprocess.CompletedProcess(command, 0, "", "")
                if kind == "garbage":
                    out_path.write_text("{not json", encoding="utf-8")
                return subprocess.CompletedProcess(command, behaviour[1], "",
                                                   behaviour[2])

            real_register = ps._register_resource_workers

            def register(row):
                registered.append(dict(row))
                return real_register(row)

            mp.setattr(ps, "containment_available", lambda: contained)
            mp.setattr(subprocess, "run", fake_run)
            mp.setattr(ps, "_register_resource_workers", register)
            mp.setattr(multiprocessing, "current_process",
                       lambda: types.SimpleNamespace(name=process))
            trial_id = 7 if behaviour[0] == "write" else 9
            capsys.readouterr()
            row = ps.run_trial_contained({"src": str(folder)},
                                         trial_id=trial_id,
                                         controls={"positive": "gra14"})
            printed = capsys.readouterr().out

            assert row == expected
            payload = json.loads((folder / "_trial_settings.json").read_text(
                encoding="utf-8"))
            assert payload == {"settings": {"src": str(folder)},
                               "trial_id": trial_id,
                               "controls": {"positive": "gra14"}}
            (command,) = commands
            assert command[-4:] == ["-m", "spacr.sweep_child",
                                    str(folder / "_trial_settings.json"),
                                    str(out_path)]
            if contained:
                assert command[:4] == ["systemd-run", "--user", "--scope",
                                       "--quiet"]
                assert "MemoryMax=24G" in command
                assert "CPUQuota=400%" in command
                assert "WITHOUT a memory cap" not in printed
            else:
                assert command[:3] == ["nice", "-n", "19"]
                assert "WITHOUT a memory cap" in printed
            expect_registered = (behaviour[0] == "write"
                                 and process == "MainProcess")
            assert registered == ([child_row] if expect_registered else [])


# ---------------------------------------------------------------------------
# one trial in a pool worker
# ---------------------------------------------------------------------------

def test_execute_trial_always_returns_a_timed_row(subtests, monkeypatch,
                                                  tmp_path):
    """Whatever the fit did -- raise, return nothing, a partial table -- a row comes back."""
    from spacr import ml

    full = {"results": pd.DataFrame({"grna": ["gra14_1", "x"],
                                     "coefficient": [0.9, 0.1],
                                     "q_value": [0.01, 0.9]}),
            "model_data": pd.DataFrame({"prc": ["p1_A_1"], "grna": ["g1"]})}

    def explode(settings):
        raise MemoryError("the design matrix did not fit\nsecond line")

    cases = [
        # id, fit, controls, refuse traceback, expected subset, absent keys
        ("a fit that raises leaves a failed row and a traceback", explode, {},
         False,
         {"status": "failed", "_overload": True, "error_type": "MemoryError",
          "error": "the design matrix did not fit"}, ()),
        ("an unwritable traceback still leaves the row", explode, {}, True,
         {"status": "failed", "error_type": "MemoryError"}, ()),
        ("a full fit records hits, design and controls",
         lambda settings: full, {"positive": "gra14"}, False,
         {"status": "ok", "n_results": 2, "n_below_alpha": 1, "n_wells": 1,
          "n_guides": 1, "positive_present": True, "positive_rank": 1,
          "positive_effect": 0.9, "positive_q": 0.01}, ("error",)),
        ("a fit that returns nothing is a plain row", lambda settings: None,
         {"positive": "gra14"}, False, {"status": "ok"},
         ("n_results", "positive_present")),
        ("no results table claims no control",
         lambda settings: {"significant": full["results"].head(1)},
         {"positive": "gra14"}, False, {"status": "ok", "n_significant": 1},
         ("positive_present", "n_results")),
    ]
    for index, (case_id, fit, controls, refuse, expected, absent) in \
            enumerate(cases, start=1):
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            seen = []

            def fake_fit(settings, fit=fit):
                seen.append(dict(settings))
                return fit(settings)

            mp.setattr(ml, "perform_regression", fake_fit)
            if index > 1:
                # be_polite renices the process and _pin_threads resizes BLAS
                # pools; the first case runs them for real, the rest skip it.
                mp.setattr(ps, "be_polite", lambda: None)
                mp.setattr(ps, "_pin_threads", lambda *a, **k: None)
            else:
                mp.setattr(os, "nice", lambda increment: 0)
                for name in ps._THREAD_VARS:
                    mp.setenv(name, os.environ.get(name, "4"))
            if refuse:
                real_open = builtins.open

                def refusing(path, *args, **kwargs):
                    if str(path).endswith("error.txt"):
                        raise OSError("read-only file system")
                    return real_open(path, *args, **kwargs)

                mp.setattr(builtins, "open", refusing)
            out = tmp_path / f"out{index}"
            row = ps._execute_trial(
                ({"src": "/data"}, {"trial_id": index, "regression_type": "ols"},
                 str(out), controls, False, False))
            mp.undo()

            folder = str(out / f"trial_{index:04d}")
            assert row["trial_id"] == index
            assert row["folder"] == folder
            assert row["regression_type"] == "ols"
            assert {k: row[k] for k in expected} == expected
            for key in absent:
                assert key not in row
            assert isinstance(row["seconds"], float) and row["seconds"] >= 0
            assert [s["worker_kind"] for s in row["_resource_workers"]] == \
                ["parameter_sweep_pool"]
            assert [s["src"] for s in seen] == [folder]
            assert seen[0]["regression_qc"] is False
            error_file = os.path.join(folder, "error.txt")
            if row["status"] == "failed" and not refuse:
                with open(error_file, encoding="utf-8") as handle:
                    assert "MemoryError: the design matrix did not fit" in \
                        handle.read()
            else:
                assert not os.path.exists(error_file)

    with subtests.test(msg="a contained trial takes the child's row"), \
            monkeypatch.context() as mp:
        mp.setattr(ps, "be_polite", lambda: None)
        mp.setattr(ps, "_pin_threads", lambda *a, **k: None)
        calls = []

        def child(settings, *, trial_id=None, controls=None):
            calls.append((trial_id, controls))
            return {"trial_id": 99, "status": "ok", "n_hits": 4,
                    "_resource_worker": {"pid": 5}}

        mp.setattr(ps, "run_trial_contained", child)
        row = ps._execute_trial(({}, {"trial_id": 3, "alpha": 1},
                                 str(tmp_path / "contained"),
                                 {"positive": "gra14"}, True))
        assert calls == [(3, {"positive": "gra14"})]
        assert row["trial_id"] == 3
        assert row["status"] == "ok"
        assert row["n_hits"] == 4
        assert row["alpha"] == 1
        assert [s.get("pid") for s in row["_resource_workers"]][1:] == [5]
        assert isinstance(row["seconds"], float)
