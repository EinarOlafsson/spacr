"""The DEFAULT inference hands back the keys a run is registered by.

`inference` defaults to 'nonparametric' since 2026-08-18, which routes
`perform_regression` through `analysis_mode='guide_permutation'` and returns
from a branch of its own. That branch returned the permutation's results and
nothing else -- no `res_folder`.

`app_screen._on_regression_done` and the Measurements queue both take the
run's folder from `res_folder`, so the default produced a complete results
folder on disk and then registered it with `folder=""`. That is the reported
"No summary: this panel was opened from a results table on disk": the panel
had no folder to read the summary from, because the run never handed one
over.
"""
import numpy as np
import pandas as pd
import pytest

from tests.test_regression_entry_points import APP_KEY, _write_screen


@pytest.fixture()
def default_run(tmp_path):
    from spacr.cli import MODULES, resolve_settings
    from spacr.ml import perform_regression

    score_csv, count_csv, _cdir = _write_screen(tmp_path)
    settings_csv = tmp_path / "regression.csv"
    pd.DataFrame(
        [("score_data", repr([score_csv])), ("count_data", repr([count_csv])),
         ("toxo", "False"), ("metadata_files", "[]")],
        columns=["Key", "Value"],
    ).to_csv(settings_csv, index=False)
    settings = resolve_settings(MODULES[APP_KEY], str(settings_csv))
    settings["min_cell_count"] = None
    np.random.seed(0)
    return settings, perform_regression(settings)


def test_the_default_is_the_permutation_path(default_run):
    settings, out = default_run

    assert settings["inference"] == "nonparametric"
    assert out["analysis_mode"] == "guide_permutation"


def test_it_hands_back_the_folder_it_wrote_into(default_run):
    import os

    _settings, out = default_run

    folder = out.get("res_folder")
    assert folder, "the GUI registers a run BY THIS KEY; empty means no run"
    assert os.path.isdir(folder)


def test_it_hands_back_its_own_settings_not_the_callers_dict(default_run):
    settings, out = default_run

    assert out["settings"]["inference"] == "nonparametric"
    assert out["settings"] is not settings, (
        "the run handed back the caller's own dict, so mutating the copy "
        "would reach back into the settings the caller still holds")


def test_the_results_are_still_there(default_run):
    _settings, out = default_run

    assert len(out["results"]) > 0
    assert "significant" in out
