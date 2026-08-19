"""A run that COMPLETES records what it cost, in its own folder.

Instruction 160 -- "i ran one ols then a mixed regression model and this hung
my computer twice so i had to restart it" -- asks for peak RSS and GPU memory
in the run folder so the next such report arrives with a number instead of a
description.

The per-stage readings were already being taken and `regression_failure`
already printed them, but ONLY for a run that failed -- which is the one case
where the machine did not hang. The run whose cost the next report has to be
compared against is the one that finished.
"""
import os

import pytest

from spacr.fit_resources import RESOURCE_KEY
from spacr.ml import FIT_RESOURCES_FILENAME, _write_fit_resources


@pytest.fixture()
def recorded(tmp_path):
    """Settings carrying two stage readings, and a run folder to write into."""
    folder = tmp_path / "run"
    folder.mkdir()
    settings = {RESOURCE_KEY: [
        {"stage": "reading data", "rss": 1_000_000_000, "gpu": None},
        {"stage": "fitting", "rss": 4_000_000_000, "gpu": 2_200_000_000},
    ]}
    return settings, str(folder)


def test_the_record_lands_beside_the_results(recorded):
    settings, folder = recorded

    path = _write_fit_resources({"res_folder": folder}, settings)

    assert path == os.path.join(folder, FIT_RESOURCES_FILENAME)
    assert os.path.exists(path)


def test_it_names_the_peak_and_the_stage_it_happened_in(recorded):
    settings, folder = recorded

    _write_fit_resources({"res_folder": folder}, settings)
    text = open(os.path.join(folder, FIT_RESOURCES_FILENAME)).read()

    assert "PEAK resident" in text
    assert "fitting" in text, "a cost without its stage cannot say where it grew"
    assert "PEAK GPU" in text


def test_not_measured_is_not_reported_as_zero(recorded):
    # The rule fit_resources already follows: psutil absent or no CUDA tensor
    # allocated is UNKNOWN, and calling it 0 would read as "this fit was free".
    settings, folder = recorded

    _write_fit_resources({"res_folder": folder}, settings)
    text = open(os.path.join(folder, FIT_RESOURCES_FILENAME)).read()

    assert "not measured" in text
    assert "0 B" not in text


def test_the_folder_can_come_from_the_settings_when_the_result_has_none(recorded):
    settings, folder = recorded
    settings["_regression_folder"] = folder

    assert _write_fit_resources({}, settings)


def test_a_run_with_no_readings_writes_nothing(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()

    assert _write_fit_resources({"res_folder": str(folder)}, {}) == ""
    assert not os.path.exists(os.path.join(folder, FIT_RESOURCES_FILENAME))


def test_it_never_raises_whatever_it_is_handed():
    # A measurement that can fail the run it measures is worse than no
    # measurement -- the rule `_stage` already follows.
    assert _write_fit_resources(None, None) == ""
    assert _write_fit_resources({"res_folder": "/does/not/exist"},
                                {RESOURCE_KEY: [{"stage": "x", "rss": 1}]}) == ""


def test_the_record_covers_the_FIT_and_not_only_the_setup(tmp_path):
    """The stages recorded must include one taken after the fit returned.

    Only two stage points existed -- "placing the results folder" and
    "reading the counts" -- both BEFORE any fitting. So the peak was always
    the peak of the setup, the GPU column read 0.0 B for a fit that ran on
    the GPU, and instruction 160's "log RSS and GPU memory per fit stage" was
    not what the file recorded.
    """
    import numpy as np
    import pandas as pd

    from spacr.cli import MODULES, resolve_settings
    from spacr.fit_resources import RESOURCE_KEY
    from spacr.ml import perform_regression
    from tests.test_regression_entry_points import APP_KEY, _write_screen

    score_csv, count_csv, _ = _write_screen(tmp_path)
    settings_csv = tmp_path / "regression.csv"
    pd.DataFrame(
        [("score_data", repr([score_csv])), ("count_data", repr([count_csv])),
         ("toxo", "False"), ("metadata_files", "[]")],
        columns=["Key", "Value"]).to_csv(settings_csv, index=False)
    settings = resolve_settings(MODULES[APP_KEY], str(settings_csv))
    settings["min_cell_count"] = None
    np.random.seed(0)

    out = perform_regression(settings)

    stages = [str(r.get("stage", "")) for r in settings.get(RESOURCE_KEY) or []]
    assert stages, "nothing was recorded at all"
    assert any("returned" in stage for stage in stages), (
        f"no reading was taken after the fit; stages were {stages}")
    folder = out.get("res_folder")
    assert os.path.exists(os.path.join(folder, FIT_RESOURCES_FILENAME))
