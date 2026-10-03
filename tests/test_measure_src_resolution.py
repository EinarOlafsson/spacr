"""Measure resolves a plate's output subfolder to its merged folder (issue 136).

The report: ``src`` pointed at the plate's measure output folder, measure
appended ``merged`` to it, and the run died in ``_listdir_visible`` with a
raw FileNotFoundError that was then filed automatically as a bug.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr.errors import ConfigurationError
from spacr.measure import _measure_merged_folder
from spacr.validate import ERROR, WARNING, _resolve_measure_src, validate_settings


@pytest.fixture
def plate(tmp_path):
    """A plate whose merged folder holds one array."""
    root = tmp_path / "plate1"
    (root / "merged").mkdir(parents=True)
    np.save(root / "merged" / "a.npy", np.zeros((4, 4, 2), dtype=np.uint16))
    return root


@pytest.mark.parametrize("sub", ["measure", "measurements", "masks"])
@pytest.mark.parametrize("exists", [True, False])
def test_an_output_subfolder_resolves_to_the_merged_folder(plate, sub, exists, capsys):
    src = plate / sub
    if exists:
        src.mkdir()

    merged, note = _resolve_measure_src(str(src))

    assert merged == str(plate / "merged")
    assert sub in note and str(plate / "merged") in note
    assert _measure_merged_folder(str(src)) == str(plate / "merged")
    assert "[measure]" in capsys.readouterr().out


@pytest.mark.parametrize("which", ["root", "merged"])
def test_the_plate_root_and_merged_folder_are_read_as_before(plate, which):
    src = plate if which == "root" else plate / "merged"

    merged, note = _resolve_measure_src(str(src))

    assert merged == str(plate / "merged")
    assert note is None


def test_a_missing_src_is_a_configuration_error_with_a_fix(tmp_path):
    missing = tmp_path / "nowhere" / "plate9"

    with pytest.raises(ConfigurationError) as caught:
        _measure_merged_folder(str(missing))

    assert "src does not exist" in str(caught.value)
    assert "plate folder" in str(caught.value)


def test_an_existing_folder_without_merged_says_what_to_set(tmp_path):
    (tmp_path / "other").mkdir()

    with pytest.raises(ConfigurationError) as caught:
        _measure_merged_folder(str(tmp_path / "other"))

    assert "no merged folder" in str(caught.value)


def test_preflight_warns_about_the_resolution_instead_of_erroring(plate):
    problems = validate_settings({"src": str(plate / "measure")}, "measure")

    src_problems = [p for p in problems if p.setting == "src"]
    assert any(p.severity == WARNING and "merged" in p.message for p in src_problems)
    assert not any(p.severity == ERROR for p in src_problems)


def test_preflight_still_errors_on_a_path_that_resolves_nowhere(tmp_path):
    problems = validate_settings({"src": str(tmp_path / "gone")}, "measure")

    assert any(p.severity == ERROR and p.setting == "src"
               and "does not exist" in p.message for p in problems)


def test_a_settings_path_error_is_not_filed_automatically():
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    check = AppScreen._is_a_settings_path_error
    src = os.path.join(os.sep, "data", "plate1", "measure")
    missing = (
        "Traceback (most recent call last):\n  File \"x\", line 1\n"
        f"FileNotFoundError: [Errno 2] No such file or directory: "
        f"'{os.path.join(src, 'merged')}'\n")

    assert check(missing, {"src": src}) is True
    assert check(missing, {"src": os.path.join(os.sep, "elsewhere")}) is False
    assert check("spacr.errors.ConfigurationError: Measure cannot start", {}) is True
    assert check("ValueError: boom", {"src": src}) is False
