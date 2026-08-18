"""A sweep does not pay for nineteen figures it will never open.

`regression()` grew a `qc` parameter so a parameter sweep could decline the
diagnostic suite. Nothing ever passed one. `perform_regression` called
`regression()` without it and no settings key mapped to it, so the parameter
was unreachable from every caller that exists and EVERY sweep trial paid the
full suite:

    ~5.8 s and ~19 figures plus a combined PDF, per trial
    -> ~10 minutes and ~2,000 files per hundred trials

with no way to say no. On a single analysis that suite is exactly what you
want, which is why it stays on by default; across a sweep it is the dominant
cost of the run and almost none of it is ever looked at.

What a sweep still gets is the SCALAR diagnostics -- trial_metrics computes
them in ~150 ms -- so sorting trials by control rank, genomic inflation or
R-squared is unaffected. The pictures are what is declined, not the numbers.
"""
from __future__ import annotations

import inspect
import re

import pytest


def test_regression_still_offers_the_switch():
    from spacr.ml import regression

    assert "qc" in inspect.signature(regression).parameters
    assert inspect.signature(regression).parameters["qc"].default is True, (
        "one analysis should get its diagnostics without being asked")


def test_perform_regression_actually_passes_it():
    """The defect: the parameter existed and nothing reached it."""
    from spacr import ml

    # RENAMED BY INSTRUCTION 132 (maintainer, 2026-08-17): the call is
    # `regression_levels(...)` now, because level='both' fits the guide model
    # and the gene model SEPARATELY rather than putting both in one collinear
    # design. `regression_levels` forwards **kwargs to `regression`, so `qc=`
    # still has to be on the call for the switch to reach the suite.
    source = inspect.getsource(ml.perform_regression)
    call = re.search(r"regression_levels\(\s*merged_df.*?\n\s*\)", source, re.S)
    assert call, "could not find the regression_levels() call to check"
    assert "qc=" in call.group(0), (
        "perform_regression calls regression_levels() without qc=, so the "
        "switch is unreachable and every sweep trial pays the full suite")


def test_the_settings_key_survives_normalisation():
    """A key the defaults pass swallows is a key the sweep cannot set."""
    from spacr.settings import get_perform_regression_default_settings

    out = get_perform_regression_default_settings(
        {"src": "/tmp/x", "regression_qc": False})
    assert out.get("regression_qc") is False


def test_it_defaults_to_on_when_nobody_says_otherwise():
    from spacr.settings import get_perform_regression_default_settings

    out = get_perform_regression_default_settings({"src": "/tmp/x"})
    # Absent is fine -- ml.py reads it with a default of True. What must NOT
    # happen is it arriving as False without anyone asking.
    assert out.get("regression_qc", True) is True


def test_a_sweep_trial_declines_it(tmp_path):
    from spacr.parameter_sweep import _trial_settings

    settings, _folder = _trial_settings(
        {"src": str(tmp_path), "regression_type": "ols"},
        {"trial_id": 1, "alpha": 0.5}, str(tmp_path))

    assert settings["regression_qc"] is False, (
        "a sweep trial is writing the full diagnostic suite; at ~5.8 s and "
        "~19 files each that is the dominant cost of the sweep")


def test_a_sweep_that_asks_for_the_pictures_gets_them(tmp_path):
    """setdefault, not a hard override: the choice stays the caller's."""
    from spacr.parameter_sweep import _trial_settings

    settings, _folder = _trial_settings(
        {"src": str(tmp_path), "regression_type": "ols",
         "regression_qc": True},
        {"trial_id": 2}, str(tmp_path))

    assert settings["regression_qc"] is True


def test_reopening_one_trial_fits_it_with_the_diagnostics(tmp_path):
    """Choosing a trial to look at again is exactly when the pictures are
    worth 5.8 seconds. That path does not go through _trial_settings."""
    from spacr.parameter_sweep import settings_for_trial

    settings = settings_for_trial(
        {"src": str(tmp_path), "regression_type": "ols"},
        {"trial_id": 3, "alpha": 0.5, "hits": 4, "seconds": 1.0},
        destination=str(tmp_path))

    assert settings.get("regression_qc", True) is True, (
        "reopening a trial should draw its diagnostics")
