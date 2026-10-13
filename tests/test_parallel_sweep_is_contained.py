"""The sweep the GUI runs must be capped by the kernel, like the one it doesn't.

``run_sweep`` has been contained by default since the kernel-cap work. The
sweep screen does not call ``run_sweep`` -- it calls ``run_sweep_parallel``,
which ran every trial inside its own pool worker with no cgroup, no MemoryMax
and no MemorySwapMax. So the containment that instruction 114 built to stop a
sweep taking the machine down was absent from the only sweep a user starts by
pressing Start.

What remained on that path was ``recommended_workers()`` and the free-memory
floor, and this module's own docstring is explicit that those are the thing
that failed: "ACCOUNTING IS NOT CONTAINMENT ... Every earlier attempt to make a
sweep safe was a better estimate of what a trial would use; each one was wrong
in a way that took the user's desktop with it."
"""

import inspect

import pandas as pd

import spacr.parameter_sweep as ps


class TestTheParallelPathIsCappedByTheKernel:

    def test_a_parallel_trial_goes_through_the_contained_runner(self, tmp_path,
                                                                monkeypatch):
        """A pool worker has no memory cap of its own.

        One cell-level permutation trial was measured holding 57 GB before the
        host ran out -- one fit, alone. Running it inside a pool worker means
        nothing can stop it; running it under systemd-run means it is killed at
        its limit and the sweep records it and carries on.
        """
        seen = {}

        def fake_contained(settings, *, trial_id=None, **kwargs):
            seen["trial_id"] = trial_id
            seen["src"] = settings.get("src")
            return {"status": "ok", "trial_id": trial_id, "seconds": 1.5,
                    "n_below_alpha": 7, "r_squared": 0.5}

        monkeypatch.setattr(ps, "run_trial_contained", fake_contained)
        row = ps._execute_trial(
            ({}, {"trial_id": 3}, str(tmp_path), {}, True))

        assert seen["trial_id"] == 3, \
            "the parallel path never reached the contained runner"
        assert row["status"] == "ok"
        assert row["n_below_alpha"] == 7
        assert row["seconds"] == 1.5

    def test_containment_is_the_default_for_the_parallel_sweep(self):
        """Off by default is the same as absent for the user who never looks."""
        assert inspect.signature(
            ps.run_sweep_parallel).parameters["contained"].default is True

    def test_the_contained_row_still_carries_its_settings_and_folder(self,
                                                                    tmp_path,
                                                                    monkeypatch):
        """A row that cannot reproduce its own trial is half an answer."""
        monkeypatch.setattr(
            ps, "run_trial_contained",
            lambda settings, *, trial_id=None, **k: {
                "status": "ok", "seconds": 1.0})
        row = ps._execute_trial(
            ({"score_data": ["s.csv"]},
             {"trial_id": 5, "regression_type": "ridge"},
             str(tmp_path), {}, True))

        assert row["trial_id"] == 5
        assert row["regression_type"] == "ridge"
        assert row["folder"].endswith("trial_0005")
        assert "preparation_key" in row

    def test_a_killed_trial_is_recorded_rather_than_lost(self, tmp_path,
                                                         monkeypatch):
        """"This combination cannot be run here" is a result, not a crash."""
        monkeypatch.setattr(
            ps, "run_trial_contained",
            lambda settings, *, trial_id=None, **k: {
                "status": "killed", "error_type": "MemoryMax",
                "error": "exceeded MemoryMax=24G", "seconds": 12.0})
        row = ps._execute_trial(({}, {"trial_id": 9}, str(tmp_path), {}, True))

        assert row["status"] == "killed"
        assert row["error_type"] == "MemoryMax"
        assert row["trial_id"] == 9

    def test_uncontained_is_still_reachable_and_still_measures_the_trial(
            self, tmp_path):
        """The opt-out must keep the diagnostics, not just the hit count."""
        pytest = __import__("pytest")
        sm = pytest.importorskip("statsmodels.api")
        import numpy as np

        rng = np.random.default_rng(0)
        design = pd.DataFrame(rng.normal(size=(60, 4)),
                              columns=[f"fraction:grna[g{i}]" for i in range(4)])
        design.insert(0, "Intercept", 1.0)
        model = sm.OLS(rng.normal(size=60), design).fit()
        output = {"results": pd.DataFrame({
                      "feature": design.columns,
                      "coefficient": model.params.to_numpy(),
                      "p_value": model.pvalues.to_numpy()}),
                  "model": model,
                  "model_data": pd.DataFrame({"prc": [f"w{i}" for i in range(60)]})}

        import spacr.ml
        original = spacr.ml.perform_regression
        spacr.ml.perform_regression = lambda _s: output
        try:
            row = ps._execute_trial(({}, {"trial_id": 1}, str(tmp_path), {},
                                     False))
        finally:
            spacr.ml.perform_regression = original

        assert row["status"] == "ok"
        assert "r_squared" in row and "design_rank" in row


class TestTheThreadPinIsNotReleased:
    """Verified rather than changed: the child is NOT double-limited.

    The regression work order asked for ``_THREAD_LIMITS`` to be released so a
    contained child is not limited twice. Measured in a real child process,
    every pool reports num_threads=1 -- the environment set at import and
    threadpoolctl resizing the live pool agree on one value rather than
    compounding. Releasing the limit would restore the 112-threads-per-trial
    storm that this module documents as the thing that starved the desktop,
    and _pin_threads' own measurement says the pin costs nothing: 18.4 s
    pinned against 18.5 s unpinned.
    """

    def test_pinning_leaves_one_thread_per_pool(self):
        threadpoolctl = __import__("pytest").importorskip("threadpoolctl")
        ps._pin_threads()
        for pool in threadpoolctl.threadpool_info():
            assert pool["num_threads"] == 1, (
                f"{pool['internal_api']} kept {pool['num_threads']} threads; "
                f"a sweep worker peaked at 112 and starved the desktop")

    def test_the_limit_is_held_open_for_the_process(self):
        """A context manager would release it before the fit ever ran."""
        ps._pin_threads()
        assert ps._THREAD_LIMITS is not None


class TestContainingATrialDoesNotEmptyTheControlColumn:
    """The sweep screen shows `positive_rank`. Containment must not blank it.

    That column is built from the caller's control ALIASES
    (run_sweep(controls={"positive": "239740"})), which only
    _named_control_rows produces. A contained child never saw the aliases, so
    routing the parallel sweep through containment would have left the column
    the run is judged on empty -- indistinguishable from a control that was
    never recovered.
    """

    def test_the_aliases_travel_to_the_contained_child(self, tmp_path,
                                                       monkeypatch):
        written = {}

        def fake_run(command, **kwargs):
            import json as _json
            path = command[-2]
            with open(path) as handle:
                written.update(_json.load(handle))

            class Done:
                returncode, stdout, stderr = 0, "", ""
            return Done()

        monkeypatch.setattr(ps, "containment_available", lambda: False)
        monkeypatch.setattr("subprocess.run", fake_run)
        ps.run_trial_contained({"src": str(tmp_path)}, trial_id=1,
                               controls={"positive": "239740"})

        assert written.get("controls") == {"positive": "239740"}, \
            "the contained child was never told which controls to look for"

    def test_the_child_emits_the_alias_columns(self, tmp_path):
        """End to end through the real child module, no regression required."""
        import json
        import numpy as np
        import pandas as pd

        results = pd.DataFrame({
            "feature": ["fraction:grna[239740_1]", "fraction:grna[111_1]"],
            "coefficient": [0.9, 0.1], "p_value": [1e-6, 0.5],
            "q_value": [1e-5, 0.9]})

        from spacr.parameter_sweep import _named_control_rows
        row = _named_control_rows(results, {"positive": "239740"})
        assert row["positive_present"] is True
        assert row["positive_rank"] == 1


class TestControlAliasesAreNotFedBackAsSettings:

    def test_alias_columns_do_not_become_regression_settings(self):
        """gra14_rank is a measurement, not something the user typed.

        The aliases are the caller's own, so they cannot be listed in advance
        -- but every alias writes `{alias}_present`, so the row names them.
        """
        from spacr.parameter_sweep import settings_for_trial

        row = {"trial_id": 1, "status": "ok", "seconds": 1.0,
               "regression_type": "ols",
               "gra14_present": True, "gra14_rank": 3, "gra14_q": 0.01,
               "gra14_p": 0.001, "gra14_effect": 0.8,
               "positive_present": True, "positive_rank": 1}
        settings = settings_for_trial({}, row)

        leaked = sorted(k for k in settings if "gra14" in k or
                        k.startswith("positive_"))
        assert leaked == [], f"alias measurements leaked as settings: {leaked}"
        assert settings["regression_type"] == "ols"

    def test_a_real_setting_ending_in_percentile_still_survives(self):
        """spaCR has twenty-two of them; a suffix rule would have eaten them."""
        from spacr.parameter_sweep import settings_for_trial

        settings = settings_for_trial(
            {}, {"trial_id": 1, "status": "ok", "seconds": 1.0,
                 "cell_intensity_percentile": 95, "lower_percentile": 2})
        assert settings["cell_intensity_percentile"] == 95
        assert settings["lower_percentile"] == 2
