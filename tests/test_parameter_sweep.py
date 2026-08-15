"""Parameter Sweep: the space, the safety limits, and the screen.

The safety tests are not decoration. A sweep sized from the core count
exhausted memory and killed the user's editor twice, so "how many workers"
is a correctness question here, not a tuning one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.parameter_sweep import (
    DEFAULT_SWEEP_SPACE, MAX_WORKERS, SweepSpace, build_trials,
    memory_is_low, recommended_workers, run_sweep, summarise_sweep,
)

pytestmark = pytest.mark.qt


# ------------------------------------------------------------------- safety


def test_workers_are_sized_from_memory_not_cores():
    """The failure this whole guard exists for.

    Fourteen workers were chosen because the box had 32 cores. Each trial
    needs several GiB for its own copy of the tables, so the machine ran out
    of memory and the OOM killer took the editor.
    """
    workers, reason = recommended_workers(measured_gib=1000.0)
    assert workers == 1, "a trial that needs 1 TiB must not get 14 workers"
    assert "GiB" in reason and "per trial" in reason


def test_a_request_is_clamped_never_honoured_blindly():
    workers, reason = recommended_workers(measured_gib=1000.0, requested=14)
    assert workers == 1
    assert "asked for 14" in reason


def test_the_ceiling_holds_however_much_memory_is_free():
    workers, _reason = recommended_workers(measured_gib=0.001, requested=999)
    assert workers <= MAX_WORKERS


def test_unmeasurable_memory_is_treated_as_scarce(monkeypatch):
    """Not knowing must be conservative, not optimistic."""
    import builtins

    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    workers, reason = recommended_workers(requested=16)
    assert workers <= 2
    assert "could not be measured" in reason


def test_memory_is_low_answers_without_raising():
    assert isinstance(memory_is_low(), bool)
    # A floor above any real machine's free memory must read as low.
    assert memory_is_low(floor_gib=10 ** 6) is True


# -------------------------------------------------------------------- space


def test_the_default_space_covers_every_axis_the_screen_offers():
    space = SweepSpace()
    assert set(space.axes) == set(DEFAULT_SWEEP_SPACE)
    # Every correction spaCR implements is sweepable, or the comparison the
    # sweep exists to make has a hole in it.
    from spacr.multiple_testing import METHODS

    assert set(space.axes["multiple_testing_method"]) == set(METHODS)


def test_illegal_combinations_never_reach_a_trial():
    trials = build_trials(SweepSpace(), mode="grid", max_trials=4000)
    for trial in trials:
        if trial.get("random_row_column_effects"):
            assert trial["regression_type"] in (None, "ols", "mixed")
        if trial.get("inference") == "nonparametric":
            assert not trial.get("random_row_column_effects")
        if trial.get("alpha") not in (None, 1):
            assert trial["regression_type"] in ("lasso", "ridge",
                                                "elasticnet", "hinge")


def test_the_cap_counts_trials_run_not_tuples_considered():
    trials = build_trials(SweepSpace(), mode="random", max_trials=250, seed=3)
    assert len(trials) == 250


def test_an_unticked_axis_is_pinned_not_dropped():
    """A sweep must record every setting it ran under."""
    space = SweepSpace(axes={"regression_type": ["ols", "ridge"]},
                       fixed={"fraction_threshold": 0.02})
    for trial in build_trials(space, max_trials=10):
        assert trial["fraction_threshold"] == 0.02


# ------------------------------------------------------------------ running


def _fake_runner(settings):
    frame = pd.DataFrame({
        "grna": ["225160_2", "239740_3", "000000_1"],
        "coefficient": [0.4, 0.3, 0.01],
        "q_value": [0.001, 0.002, 0.9],
    })
    return {"results": frame, "significant": frame.head(2)}


def test_a_sweep_records_one_row_per_trial_with_the_controls(tmp_path):
    space = SweepSpace(axes={"regression_type": ["ols", "ridge"],
                             "multiple_testing_method": ["fdr_bh", "none"]})
    results = run_sweep({}, tmp_path, space, mode="grid", max_trials=4,
                        controls={"eaf1": "225160", "gra14": "239740"},
                        progress_every=0, runner=_fake_runner)
    assert len(results) == 4
    assert (results["status"] == "ok").all()
    assert results["eaf1_present"].all() and results["gra14_present"].all()
    assert (results["eaf1_rank"] == 1).all()


def test_the_results_table_survives_a_sweep_that_is_killed(tmp_path):
    """Written after every trial, so a stopped sweep keeps what it learned."""
    space = SweepSpace(axes={"regression_type": ["ols", "ridge", "glm"]})
    run_sweep({}, tmp_path, space, mode="grid", max_trials=3,
              progress_every=0, runner=_fake_runner)
    written = pd.read_csv(tmp_path / "sweep_results.csv")
    assert len(written) == 3


def test_a_failing_trial_is_a_row_not_a_crash(tmp_path):
    def explode(settings):
        raise ValueError("Poisson regression requires integer count data")

    space = SweepSpace(axes={"regression_type": ["poisson", "ols"]})
    results = run_sweep({}, tmp_path, space, mode="grid", max_trials=2,
                        progress_every=0, runner=explode)
    assert (results["status"] == "failed").all()
    assert results["error_type"].eq("ValueError").all()


def test_a_family_that_always_fails_is_learned_not_rediscovered(tmp_path):
    calls = {"n": 0}

    def explode(settings):
        calls["n"] += 1
        raise ValueError("requires integer count data")

    space = SweepSpace(axes={
        "regression_type": ["poisson"],
        "multiple_testing_method": list(DEFAULT_SWEEP_SPACE[
            "multiple_testing_method"]),
    })
    results = run_sweep({}, tmp_path, space, mode="grid", max_trials=13,
                        progress_every=0, learn_from_failures=2,
                        runner=explode)
    # Two real attempts, the rest recorded as skipped rather than repeated.
    assert calls["n"] == 2
    assert (results["status"] == "skipped").sum() == 11


def test_the_summary_reports_the_spread_not_a_winner(tmp_path):
    """A sweep answers 'what survives my choices', not 'which gave most hits'."""
    space = SweepSpace(axes={"multiple_testing_method": ["fdr_bh", "none"]})
    results = run_sweep({}, tmp_path, space, mode="grid", max_trials=2,
                        controls={"eaf1": "225160"}, progress_every=0,
                        runner=_fake_runner)
    summary = summarise_sweep(results, controls=("eaf1",))
    assert summary["trials"] == 2 and summary["succeeded"] == 2
    assert summary["eaf1_recovered_in"] == "2/2 trials"


# ------------------------------------------------------------------- screen


def test_the_screen_registers_as_a_spacr_module(qtbot):
    import spacr.qt.screens  # noqa: F401 - triggers registration
    from spacr.qt.app import APPS

    assert any(row[0] == "parameter_sweep" for row in APPS)


def test_the_screen_builds_a_space_from_its_ticked_axes(qtbot):
    from spacr.qt.screens.parameter_sweep import _make_screen

    screen = _make_screen()
    qtbot.addWidget(screen)
    # Untick everything except the model family.
    for key, (include, _editor) in screen._axis_rows.items():
        include.setChecked(key == "regression_type")
    space = screen.space()
    assert set(space.axes) == {"regression_type"}
    # Everything else is pinned, not lost.
    assert "multiple_testing_method" in space.fixed


def test_the_screen_refuses_to_start_without_inputs(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt.screens.parameter_sweep import _make_screen

    screen = _make_screen()
    qtbot.addWidget(screen)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: warned.append(args))
    screen.start()
    assert warned, "starting with no score/count CSVs must warn, not run"


def test_the_screen_estimates_without_running_anything(qtbot):
    from spacr.qt.screens.parameter_sweep import _make_screen

    screen = _make_screen()
    qtbot.addWidget(screen)
    screen.max_trials.setValue(120)
    assert screen.estimate() == 120
    assert "combinations" in screen.status.text()
