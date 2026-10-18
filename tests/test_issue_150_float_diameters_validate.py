"""GitHub #150: float diameters pass every settings entry point.

The GUI validator reported ``cell_diameter=60.0 is a float, but int is
expected`` for all three object diameters and told the user to "Set
cell_diameter to a int". Item 696 stores diameters canonically (a whole
float becomes an int, a fraction stays a float); these tests pin that for
the reporter's values through the settings file, the command line, the
defaults, the GUI run wrapper and :func:`spacr.validate.validate_settings`
for every app, and the fix text now takes the right article.
"""
from __future__ import annotations

import csv

import pytest

DIAMETERS = ("cell_diameter", "nucleus_diameter", "pathogen_diameter")
VALUES = [32.5, "32.5", 60.0, "60.0", 30.0, 25.0, None, ""]


def _diameter_problems(problems):
    return [p for p in problems if p.setting in DIAMETERS]


@pytest.mark.parametrize("value", VALUES)
def test_every_app_accepts_the_reported_diameters(value):
    from spacr.validate import APP_FUNCTIONS, validate_settings

    settings = {key: value for key in DIAMETERS}
    for app in sorted(APP_FUNCTIONS):
        assert _diameter_problems(validate_settings(dict(settings), app)) == [], app


@pytest.mark.parametrize("value, stored", [
    (32.5, 32.5), ("32.5", 32.5),
    (60.0, 60), ("60.0", 60), (None, None)])
def test_the_mask_defaults_store_one_canonical_diameter(value, stored):
    from spacr.settings import set_default_settings_preprocess_generate_masks
    from spacr.validate import validate_settings

    settings = set_default_settings_preprocess_generate_masks(
        {key: value for key in DIAMETERS})
    assert [settings[key] for key in DIAMETERS] == [stored] * 3
    assert _diameter_problems(validate_settings(dict(settings), "mask")) == []


def test_a_settings_csv_with_zero_and_floats_loads_and_validates(tmp_path):
    from spacr.cli import _load_settings_csv
    from spacr.validate import validate_settings

    path = tmp_path / "gen_mask_settings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerow(["pathogen_diameter", "25.0"])
        writer.writerow(["cell_diameter", "60.0"])
        writer.writerow(["nucleus_diameter", "32.5"])
    loaded = _load_settings_csv(str(path))
    assert _diameter_problems(validate_settings(dict(loaded), "mask")) == []


@pytest.mark.parametrize("text, stored", [("32.5", 32.5), ("60.0", 60)])
def test_the_command_line_set_takes_floats(text, stored):
    from spacr.cli import coerce_value
    from spacr.settings import expected_types

    assert coerce_value("pathogen_diameter", text, None, expected_types) == stored


def test_the_gui_run_wrapper_coerces_without_a_type_error():
    from spacr.validate import coerce_expected_types, validate_settings

    settings = coerce_expected_types(
        {"cell_diameter": 60.0, "nucleus_diameter": "30.0",
         "pathogen_diameter": 25.0}, "mask")
    assert settings == {"cell_diameter": 60, "nucleus_diameter": 30,
                        "pathogen_diameter": 25}
    assert _diameter_problems(validate_settings(dict(settings), "mask")) == []


def test_the_object_grid_reads_floats():
    from spacr.object_settings_table import _diameter_setting

    assert _diameter_setting(25.0) == 25
    assert _diameter_setting(32.5) == 32.5


def test_a_type_fix_uses_the_right_article():
    from spacr.validate import _check_types

    problems = _check_types({"batch_size": 2.5, "pathogen_channel": 1.5}, "mask")
    fixes = [p.fix for p in problems]
    assert fixes and all(" to a int" not in fix for fix in fixes)
    assert any(" to an int" in fix for fix in fixes)
