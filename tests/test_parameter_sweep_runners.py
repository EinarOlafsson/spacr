"""Table-driven tests of the sequential and parallel parameter-sweep runners.

A sweep is judged by its rows: one per trial, carrying the settings it ran,
its status and timing, and -- for a trial that died -- the reason.  The CSV is
written as the sweep goes so an interrupted sweep still leaves a table.  One
general test per runner path (``run_sweep`` in-process, ``run_sweep`` with
contained children, ``run_sweep_parallel``), each walking a case table with
pytest's ``subtests`` fixture.  No regression is fitted: runners,
``run_trial_contained`` and ``_execute_trial`` are doubles, and the parallel
pool runs its jobs in this process.
"""
from __future__ import annotations

import builtins
import json
import os

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


def _two_trials(**extra):
    """Two trials (ols, ridge), so a sweep finishes in milliseconds."""
    return ps.SweepSpace(axes={"regression_type": ["ols", "ridge"]},
                         fixed={"alpha": 1}, **extra)


def _refuse_error_txt(mp):
    """Make every ``error.txt`` open fail as on a read-only disk."""
    real_open = builtins.open

    def refusing(path, *args, **kwargs):
        if str(path).endswith("error.txt"):
            raise OSError("read-only file system")
        return real_open(path, *args, **kwargs)

    mp.setattr(builtins, "open", refusing)


def _refuse_csv(mp):
    """Make every ``DataFrame.to_csv`` fail as on a full disk."""
    def refuse(self, *args, **kwargs):
        raise OSError("the results disk is full")

    mp.setattr(pd.DataFrame, "to_csv", refuse)


# ---------------------------------------------------------------------------
# the sequential sweep, in this process
# ---------------------------------------------------------------------------

def test_run_sweep_in_process_writes_one_row_per_trial(subtests, monkeypatch,
                                                       tmp_path, capsys):
    """Every in-process trial is a row, whatever its runner returned or raised."""
    from spacr import ml

    with_design = {"results": _coefficients(), "model_data": pd.DataFrame(
        {"prc": ["p1_A_1", "p1_A_2"], "grna": ["g1", "g2"]})}
    small = pd.DataFrame({"grna": ["a", "b"], "q_value": [0.01, 0.4]})
    frame = pd.DataFrame({"grna": ["gra14_1", "x"], "coefficient": [0.9, 0.1],
                          "q_value": [0.01, 0.9]})

    def raise_memory(settings):
        raise MemoryError("not enough memory for this design")

    def by_family(answers):
        return lambda settings: answers[settings["regression_type"]]

    cases = [
        # id, runner (or "default"), kwargs, patches, check
        ("settings, status, design and controls",
         lambda settings: with_design,
         {"controls": {"positive": "gra14"}}, (), "design"),
        ("a correction axis becomes one row per method from one fit",
         lambda settings: {"results": _coefficients()},
         {"corrections": ["fdr_bh", "bonferroni"],
          "space": ps.SweepSpace(axes={"regression_type": ["ols"]},
                                 fixed={"alpha": 1})}, (), "corrections"),
        ("a failed trial is a row with a reason and a traceback",
         raise_memory, {"corrections": ["fdr_bh"], "learn_from_failures": 0},
         (), "failed"),
        ("an unwritable traceback does not stop later trials",
         raise_memory, {"learn_from_failures": 0}, ("error_txt",),
         "failed_no_traceback"),
        ("progress prints and writes the partial CSV",
         lambda settings: {"results": _coefficients()},
         {"progress_every": 1}, (), "progress"),
        ("a CSV that cannot be written does not stop the sweep",
         lambda settings: {"results": _coefficients()},
         {"progress_every": 1}, ("csv",), "no_csv"),
        ("the default runner is the regression", "default", {}, (),
         "default"),
        ("a return that is not a mapping still lands in the table",
         by_family({"ols": ["not", "a", "mapping"],
                    "ridge": {"results": small}}),
         {}, (), "no_mapping"),
        ("a trial with no results frame claims no control",
         by_family({"ols": {"significant": frame.head(1)},
                    "ridge": {"results": frame}}),
         {"controls": {"positive": "gra14"}}, (), "no_results"),
    ]
    for index, (case_id, runner, kwargs, patches, check) in enumerate(cases):
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            kwargs = dict(kwargs)
            space = kwargs.pop("space", _two_trials())
            kwargs.setdefault("progress_every", 0)
            called = []
            if runner == "default":
                def regression(settings):
                    called.append(settings["regression_type"])
                    return {"results": _coefficients()}

                mp.setattr(ml, "perform_regression", regression)
                runner_kwargs = {}
            else:
                def recording(settings, runner=runner):
                    called.append(settings["regression_type"])
                    return runner(settings)

                runner_kwargs = {"runner": recording}
            if "error_txt" in patches:
                _refuse_error_txt(mp)
            if "csv" in patches:
                _refuse_csv(mp)
            out = tmp_path / f"out{index}"
            if index == 0:
                # A rerun into a folder that already exists is not refused.
                out.mkdir()
            capsys.readouterr()
            results = ps.run_sweep({"src": "/data"}, out, space,
                                   contained=False, **runner_kwargs, **kwargs)
            printed = capsys.readouterr().out
            mp.undo()

            trials = json.loads((out / "sweep_trials.json").read_text(
                encoding="utf-8"))
            assert [t["trial_id"] for t in trials] == \
                sorted(results["trial_id"].unique().tolist())
            folders = [str(out / f"trial_{t['trial_id']:04d}") for t in trials]
            assert called == [t["regression_type"] for t in trials]
            assert (results["seconds"] >= 0).all()

            if check == "design":
                assert results["trial_id"].tolist() == [1, 2]
                assert results["regression_type"].tolist() == ["ols", "ridge"]
                assert results["alpha"].tolist() == [1, 1]
                assert results["status"].tolist() == ["ok", "ok"]
                assert results["folder"].tolist() == folders
                assert results["positive_present"].tolist() == [True, True]
                assert results["positive_rank"].tolist() == [1, 1]
                assert results["n_wells"].tolist() == [2, 2]
                assert results["n_results"].tolist() == [4, 4]
                assert results["n_below_alpha"].tolist() == [2, 2]
                assert pd.read_csv(out / "sweep_results.csv").shape[0] == 2
            elif check == "corrections":
                assert results["multiple_testing_method"].tolist() == \
                    ["fdr_bh", "bonferroni"]
                assert results["trial_id"].tolist() == [1, 1]
                assert results["n_tests"].tolist() == [4, 4]
                assert results["n_below_alpha"].tolist() == [2, 2]
            elif check in ("failed", "failed_no_traceback"):
                assert results["status"].tolist() == ["failed", "failed"]
                assert results["error_type"].tolist() == ["MemoryError"] * 2
                assert results["error"].tolist() == \
                    ["not enough memory for this design"] * 2
                if check == "failed":
                    assert results["multiple_testing_method"].tolist() == \
                        ["fdr_bh", "fdr_bh"]
                    for folder in folders:
                        assert os.path.isfile(os.path.join(folder, "error.txt"))
                else:
                    assert list(out.glob("**/error.txt")) == []
            elif check == "progress":
                assert "[sweep] 1/2 trials (ok), " in printed
                assert "[sweep] 2/2 trials (ok), " in printed
                assert "min left" in printed
                written = pd.read_csv(out / "sweep_results.csv")
                assert written["regression_type"].tolist() == ["ols", "ridge"]
            elif check == "no_csv":
                assert results["status"].tolist() == ["ok", "ok"]
                assert not (out / "sweep_results.csv").exists()
            elif check == "default":
                assert results["status"].tolist() == ["ok", "ok"]
            elif check == "no_mapping":
                rows = results.set_index("regression_type")
                assert results["status"].tolist() == ["ok", "ok"]
                assert pd.isna(rows.loc["ols", "n_results"])
                assert rows.loc["ridge", "n_results"] == 2
                assert rows.loc["ridge", "n_below_alpha"] == 1
                assert (out / "sweep_results.csv").is_file()
            elif check == "no_results":
                rows = results.set_index("regression_type")
                assert rows.loc["ols", "n_significant"] == 1
                assert pd.isna(rows.loc["ols", "positive_present"])
                assert rows.loc["ridge", "positive_present"] is True
                assert rows.loc["ridge", "positive_rank"] == 1


# ---------------------------------------------------------------------------
# the sequential sweep, one contained child per trial
# ---------------------------------------------------------------------------

def test_run_sweep_contained_takes_each_childs_row(subtests, monkeypatch,
                                                   tmp_path, capsys):
    """Child rows are taken whole; repeated failures and low memory stop trials."""
    three = ps.SweepSpace(
        axes={"regression_type": ["ols"],
              "fraction_threshold": [0.1, 0.2, 0.3]},
        fixed={"alpha": 1})
    ok_row = {"status": "ok", "seconds": 1.5, "n_hits": 7,
              "_resource_workers": [{"pid": 1}]}
    failing = {"status": "failed", "error_type": "MemoryError",
               "seconds": 0.1}
    cases = [
        # id, space, child row, kwargs, patches
        ("the child's row is taken whole", _two_trials(), ok_row,
         {"progress_every": 1, "memory_floor_gb": 0.0}, ()),
        ("a repeated failure signature is skipped", three, failing,
         {"learn_from_failures": 2, "memory_floor_gb": 0.0}, ()),
        ("the free-memory floor stops the sweep", _two_trials(), ok_row,
         {"memory_floor_gb": 4.0}, ("low_memory",)),
        ("an unwritable partial CSV does not stop the sweep", _two_trials(),
         {"status": "ok"}, {"progress_every": 1, "memory_floor_gb": 0.0},
         ("csv",)),
    ]
    for index, (case_id, space, child_row, kwargs, patches) in \
            enumerate(cases):
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            calls = []

            def child(settings, *, trial_id=None, controls=None, **extra):
                calls.append((trial_id, controls, settings["src"]))
                return {"trial_id": trial_id, **child_row}

            mp.setattr(ps, "run_trial_contained", child)
            if "low_memory" in patches:
                mp.setattr(ps, "free_memory_gb", lambda: 0.5)
            if "csv" in patches:
                _refuse_csv(mp)
            kwargs = {"progress_every": 0, **kwargs}
            out = tmp_path / f"out{index}"
            capsys.readouterr()
            results = ps.run_sweep({"src": "/data"}, out, space,
                                   contained=True,
                                   controls={"positive": "gra14"}, **kwargs)
            printed = capsys.readouterr().out
            mp.undo()

            if index == 0:
                assert calls == [
                    (1, {"positive": "gra14"}, str(out / "trial_0001")),
                    (2, {"positive": "gra14"}, str(out / "trial_0002"))]
                assert results["status"].tolist() == ["ok", "ok"]
                assert results["trial_id"].tolist() == [1, 2]
                assert results["n_hits"].tolist() == [7, 7]
                assert results["seconds"].tolist() == [1.5, 1.5]
                assert "_resource_workers" not in results.columns
                assert "[sweep] 1/2 trials (ok)" in printed
                assert "[sweep] 2/2 trials (ok)" in printed
                assert pd.read_csv(out / "sweep_results.csv").shape[0] == 2
            elif index == 1:
                assert [c[0] for c in calls] == [1, 2]
                assert results["status"].tolist() == \
                    ["failed", "failed", "skipped"]
                skipped = results.iloc[-1]
                assert skipped["error_type"] == "MemoryError"
                assert skipped["error"] == "same failure as trial 1"
                assert skipped["seconds"] == 0.0
                assert skipped["fraction_threshold"] == 0.3
                assert skipped["folder"] is None
            elif index == 2:
                assert calls == []
                assert results["status"].tolist() == ["skipped"]
                assert results.iloc[0]["error"] == \
                    "stopped at the free-memory floor"
                assert "[sweep] stopping: 0 GB free is below the 4 GB floor" \
                    in printed
            else:
                assert results["status"].tolist() == ["ok", "ok"]
                assert not (out / "sweep_results.csv").exists()


# ---------------------------------------------------------------------------
# the parallel sweep
# ---------------------------------------------------------------------------

class _InlinePool:
    """A ProcessPoolExecutor double that runs each job when its result is read.

    The submission policy -- how many jobs are in flight, when the memory
    floor holds one back, retries, cancellation -- is what is under test;
    none of it needs a second interpreter.
    """

    made: list = []

    def __init__(self, max_workers, mp_context):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.submitted = []
        _InlinePool.made.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, function, payload):
        assert function is ps._execute_trial
        trial_id = payload[1]["trial_id"]
        attempt = self.submitted.count(trial_id) + 1
        self.submitted.append(trial_id)
        outcome = self.outcome

        class _Future:
            def result(self):
                return outcome(payload, attempt)

        return _Future()


def test_run_sweep_parallel_keeps_n_jobs_in_flight_and_retries_overloads(
        subtests, monkeypatch, tmp_path, capsys):
    """Each trial becomes a row; dead workers fail, overloads retry, cancels stop."""
    import concurrent.futures as cf
    import multiprocessing

    from spacr.cancellation import PipelineCancelled
    from spacr.resource_log import _StaggeredContext

    def ok(payload, attempt):
        trial = payload[1]
        return {"trial_id": trial["trial_id"], "status": "ok", "seconds": 0.2,
                "_resource_workers": [{"pid": 1}],
                **{k: v for k, v in trial.items() if k != "trial_id"}}

    def die(payload, attempt):
        raise RuntimeError("A child process terminated abruptly")

    def cancel_first(payload, attempt):
        if payload[1]["trial_id"] == 1:
            raise PipelineCancelled("user pressed stop")
        return ok(payload, attempt)

    def overload_once(payload, attempt):
        trial_id = payload[1]["trial_id"]
        folder = os.path.join(payload[2], "trial_0001")
        if trial_id == 1 and attempt == 1:
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "error.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write("MemoryError: first try")
            return {"trial_id": 1, "status": "failed", "_overload": True,
                    "error_type": "MemoryError", "error": "first try",
                    "seconds": 4.5}
        if trial_id == 1:
            with open(os.path.join(folder, "error.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write("retry")
            return {"trial_id": 1, "status": "ok", "_overload": False,
                    "seconds": 2.0}
        return {"trial_id": trial_id, "status": "ok"}

    def overload_then_cancel(payload, attempt):
        if attempt == 1:
            return {"trial_id": 1, "status": "failed", "_overload": True,
                    "error_type": "MemoryError", "error": "busy",
                    "seconds": 1.0}
        raise PipelineCancelled("stopped during retry")

    def overload_then_die(payload, attempt):
        if attempt == 1:
            raise MemoryError("worker ran out")
        raise RuntimeError("died on retry")

    def no_ridge(trial):
        if trial.get("regression_type") == "ridge":
            return "ridge is excluded for this dataset"
        return None

    trials3 = [{"trial_id": i, "regression_type": "ols"} for i in (1, 2, 3)]
    cases = [
        # id, outcome, space, trials override, n_jobs, memory answers,
        # tree GiB, real recommended_workers, expected
        ("a row per trial, written as each finishes", ok, _two_trials(), None,
         2, None, None, True,
         {"submitted": [1, 2], "status": ["ok", "ok"]}),
        ("a worker that dies becomes a failed row", die, _two_trials(), None,
         1, None, None, False,
         {"submitted": [1, 2], "status": ["failed", "failed"]}),
        ("the pool refills after each completed future", ok, None, trials3,
         2, None, None, False,
         {"submitted": [1, 2, 3], "status": ["ok", "ok", "ok"]}),
        ("low memory holds a submission back", ok,
         ps.SweepSpace(axes={"regression_type": ["ols"],
                             "fraction_threshold": [0.1, 0.2]},
                       fixed={"alpha": 1}),
         None, 2, [False, True], 3.5, False,
         {"submitted": [1, 2], "status": ["ok", "ok"]}),
        ("held back without a spaCR tree figure", ok, None, trials3[:2], 2,
         [False, True], None, False,
         {"submitted": [1, 2], "status": ["ok", "ok"]}),
        ("the caller's filters are obeyed", ok,
         _two_trials(filters=[no_ridge]), None, 1, None, None, False,
         {"submitted": [1], "status": ["ok"]}),
        ("cancelling a primary trial stops the sweep", cancel_first, None,
         trials3, 2, None, None, False,
         {"submitted": [1, 2], "raises": "user pressed stop"}),
        ("an overloaded trial is retried and keeps its primary files",
         overload_once, None, trials3[:2], 2, None, None, False,
         {"submitted": [1, 2, 1], "status": ["ok", "ok"]}),
        ("cancelling an overload retry stops the sweep", overload_then_cancel,
         None, trials3[:1], 2, None, None, False,
         {"submitted": [1, 1], "raises": "stopped during retry"}),
        ("a retry whose worker dies is a failed row", overload_then_die,
         None, trials3[:1], 2, None, None, False,
         {"submitted": [1, 1], "status": ["failed"]}),
    ]
    for index, (case_id, outcome, space, trials, n_jobs, answers, tree,
                real_workers, expected) in enumerate(cases):
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            _InlinePool.made = []
            mp.setattr(_InlinePool, "outcome", staticmethod(outcome),
                       raising=False)
            mp.setattr(cf, "ProcessPoolExecutor", _InlinePool)
            mp.setattr(cf, "as_completed", lambda snapshot: iter(snapshot))
            for name in ps._THREAD_VARS:
                mp.setenv(name, os.environ.get(name, "4"))
            if not real_workers:
                mp.setattr(ps, "recommended_workers",
                           lambda **kwargs: (n_jobs, "test capacity"))
                mp.setattr(ps, "_pin_threads", lambda *a, **k: None)
            if trials is not None:
                mp.setattr(ps, "build_trials",
                           lambda *a, trials=trials, **k: trials)
            state = {"spacr_tree_gib": tree}
            mp.setattr(ps, "_LAST_MEMORY_STATE", state)
            sequence = iter(answers or [])
            mp.setattr(ps, "memory_is_low",
                       lambda *a, **k: next(sequence, False))
            out = tmp_path / f"out{index}"
            # The refill case reports every second trial, so the third
            # (last) completion prints no progress line.
            progress = 2 if index == 2 else 1
            capsys.readouterr()
            # The real worker sizing also runs the start-up RAM guard; the
            # sized cases turn it off so the pool is the size they asked for.
            base = ({"src": "/data"} if real_workers
                    else {"src": "/data", "ram_guard": False})
            call = lambda: ps.run_sweep_parallel(  # noqa: E731
                base, out, space,
                n_jobs=n_jobs, progress_every=progress)
            if "raises" in expected:
                with pytest.raises(PipelineCancelled,
                                   match=expected["raises"]):
                    call()
                results = None
            else:
                results = call()
            printed = capsys.readouterr().out
            mp.undo()

            (pool,) = _InlinePool.made
            assert pool.submitted == expected["submitted"]
            assert isinstance(pool.mp_context, _StaggeredContext)
            assert pool.mp_context.get_context().get_start_method() == "spawn"
            if not real_workers:
                assert pool.max_workers == n_jobs
                assert f"trials across {n_jobs} workers (test capacity)" \
                    in printed
            if results is not None:
                assert results["status"].tolist() == expected["status"]
                assert results["trial_id"].tolist() == \
                    sorted(results["trial_id"].tolist())
                assert "_resource_workers" not in results.columns
                assert "_overload" not in results.columns
                saved = pd.read_csv(out / "sweep_results.csv")
                assert saved["status"].tolist() == expected["status"]
                total = len(results)
                if progress == 1:
                    assert f"[sweep] {total}/{total} done, " in printed

            if index == 0:
                listed = json.loads((out / "sweep_trials.json").read_text(
                    encoding="utf-8"))
                assert [t["regression_type"] for t in listed] == \
                    ["ols", "ridge"]
                assert "[sweep] 2 trials across " in printed
                assert "[sweep] 1/2 done, 1 ok, " in printed
            elif index == 1:
                assert results["error_type"].tolist() == ["RuntimeError"] * 2
                assert results["error"].tolist() == \
                    ["A child process terminated abruptly"] * 2
                assert results["seconds"].tolist() == [0.0, 0.0]
                assert "[sweep] 1/2 done, 0 ok, " in printed
            elif index == 2:
                assert "[sweep] 2/3 done, 2 ok, " in printed
                assert "[sweep] 1/3 done" not in printed
                assert "[sweep] 3/3 done" not in printed
            elif index == 3:
                assert "[sweep] 1/2 done, 1 ok, " in printed
                assert ", paused 1x for memory" in printed
                assert ("[sweep] held back 1 times because free memory fell "
                        "below the floor; the sweep yielded rather than "
                        "competing with the rest of the machine; spaCR's "
                        "process tree was 3.5 GiB.") in printed
            elif index == 4:
                assert ("competing with the rest of the machine.\n") in printed
            elif index == 5:
                assert space.filters == [no_ridge]
                assert results["regression_type"].tolist() == ["ols"]
                listed = json.loads((out / "sweep_trials.json").read_text(
                    encoding="utf-8"))
                assert [t["regression_type"] for t in listed] == ["ols"]
            elif index == 6:
                assert not (out / "sweep_results.csv").exists()
            elif index == 7:
                first = results.set_index("trial_id").loc[1]
                assert bool(first["overload_retry"]) is True
                assert first["primary_error_type"] == "MemoryError"
                assert first["primary_error"] == "first try"
                assert first["primary_seconds"] == pytest.approx(4.5)
                folder = out / "trial_0001"
                assert (folder / "error.txt.primary").read_text(
                    encoding="utf-8") == "MemoryError: first try"
                assert (folder / "error.txt").read_text(
                    encoding="utf-8") == "retry"
                assert not (folder / "_trial_result.json.primary").exists()
            elif index == 8:
                saved = pd.read_csv(out / "sweep_results.csv")
                assert saved["status"].tolist() == ["failed"]
                assert "overload_retry" not in saved.columns
            elif index == 9:
                row = results.iloc[0]
                assert row["error_type"] == "RuntimeError"
                assert row["error"] == "died on retry"
                assert bool(row["overload_retry"]) is True
                assert row["primary_error_type"] == "MemoryError"
                assert row["primary_error"] == "worker ran out"
                assert row["primary_seconds"] == 0.0

    with subtests.test(msg="a worker process refuses to start a sweep"), \
            monkeypatch.context() as mp:
        class _NotMain:
            name = "SpawnPoolWorker-1"

        mp.setattr(multiprocessing, "current_process", lambda: _NotMain())
        with pytest.raises(RuntimeError) as caught:
            ps.run_sweep_parallel({"src": "/data"}, tmp_path / "refused",
                                  _two_trials(), n_jobs=1)
        assert str(caught.value) == (
            "run_sweep_parallel was called from a worker process. The script "
            "that calls it needs an `if __name__ == \"__main__\":` guard -- "
            "without one, each spawned worker re-runs the script and starts "
            "its own sweep.")
        assert not (tmp_path / "refused").exists()
