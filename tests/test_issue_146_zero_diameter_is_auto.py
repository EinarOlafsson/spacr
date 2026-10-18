"""GitHub #146: ``pathogen_diameter=0`` means "auto" at every entry point.

A settings CSV with ``pathogen_diameter`` 0 stopped ``spacr-run mask`` with
"must be greater than zero". Item 696 made 0 and "auto" one stored value;
these tests pin that for the reporter's route (a CSV through the command
line) and for the defaults, the ``--set`` override, the GUI run wrapper and
:func:`spacr.validate.validate_settings` for every app.
"""
from __future__ import annotations

import csv

import pytest

DIAMETERS = ("cell_diameter", "nucleus_diameter", "pathogen_diameter")
ZEROS = [0, 0.0, "0", "0.0", " 0 ", "auto", "AUTO"]


def _diameter_problems(problems):
    return [p for p in problems if p.setting in DIAMETERS]


@pytest.mark.parametrize("value", ZEROS)
def test_every_app_accepts_zero_as_auto(value):
    from spacr.validate import APP_FUNCTIONS, validate_settings

    settings = {key: value for key in DIAMETERS}
    for app in sorted(APP_FUNCTIONS):
        assert _diameter_problems(validate_settings(dict(settings), app)) == [], app


@pytest.mark.parametrize("value", ZEROS)
def test_the_mask_defaults_store_zero_as_auto(value):
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = set_default_settings_preprocess_generate_masks(
        {key: value for key in DIAMETERS})
    assert [settings[key] for key in DIAMETERS] == ["auto"] * 3


def test_the_reported_settings_csv_dry_run_has_no_diameter_error(tmp_path, capsys):
    from spacr.cli import main

    src = tmp_path / "src"
    src.mkdir()
    path = tmp_path / "gen_mask_settings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerow(["src", str(src)])
        writer.writerow(["pathogen_diameter", "0"])
    main(["mask", "--settings", str(path), "--dry-run"])
    report = capsys.readouterr()
    text = report.out + report.err
    assert "pathogen_diameter=0 must be greater than zero" not in text
    assert "pathogen_diameter" in text and "'auto'" in text


@pytest.mark.parametrize("text", ["0", "0.0", "auto"])
def test_the_command_line_set_takes_zero(text):
    from spacr.cli import coerce_value
    from spacr.settings import expected_types

    assert coerce_value("pathogen_diameter", text, None, expected_types) == "auto"


def test_the_gui_run_wrapper_turns_zero_into_auto():
    from spacr.object_settings_table import _diameter_setting
    from spacr.validate import coerce_expected_types

    assert coerce_expected_types({"pathogen_diameter": 0}, "mask") == {
        "pathogen_diameter": "auto"}
    assert _diameter_setting(0) == "auto"
