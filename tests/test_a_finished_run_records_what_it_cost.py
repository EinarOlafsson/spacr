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
