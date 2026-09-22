"""A regression settings file that names `Toxoplasma` or `toxo` still loads.

Instruction 364 retired the `Toxoplasma` switch on 2026-09-19. The maintainer
decided it that day, verbatim: "Retire both" (with `barcodes`), old settings
files to keep loading through the existing migration. `annotation_source`
says everything the switch did except one thing -- false, which meant no
annotation at all -- so the switch is FOLDED onto it rather than renamed:

* a name already in `annotation_source` wins, as it did before;
* otherwise true means ``'toxoplasma'`` and false means ``''``, no annotation.

Every test here starts from a FILE, written the way spaCR writes one and read
back by the reader a run uses, because a file saved before today is where the
old key lives. The shipped example `spaCR_settings/6_Regression_settings.csv`
still says `toxo,True` and is loaded as it is.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytest

from spacr.cli import load_settings_file
from spacr.settings import (SEMANTIC_FOLDS, expected_types,
                            get_perform_regression_default_settings, tooltips)
from spacr.validate import RETIRED_SETTINGS, WARNING, _check_retired_keys

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_EXAMPLE = ROOT / "spaCR_settings" / "6_Regression_settings.csv"


def _old_file(tmp_path, rows):
    """A regression settings CSV as a run before 2026-09-19 saved it."""
    path = tmp_path / "regression_settings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows([("src", str(tmp_path))] + list(rows))
    return path


def _loaded(tmp_path, rows):
    """The file, read back and handed to the regression defaults."""
    return get_perform_regression_default_settings(
        load_settings_file(str(_old_file(tmp_path, rows))))


@pytest.mark.parametrize("rows, expected", [
    ([("Toxoplasma", "False")], ""),
    ([("Toxoplasma", "True")], "toxoplasma"),
    ([("toxo", "False")], ""),
    ([("toxo", "True")], "toxoplasma"),
    ([("Toxoplasma", "False"), ("toxo", "True")], ""),
    ([("Toxoplasma", "True"), ("annotation_source", "")], "toxoplasma"),
    ([("Toxoplasma", "False"), ("annotation_source", "")], ""),
    ([("Toxoplasma", "True"), ("annotation_source", "human")], "human"),
    ([("Toxoplasma", "False"), ("annotation_source", "Plasmodium falciparum")],
     "Plasmodium falciparum"),
])
def test_the_old_switch_lands_on_annotation_source(tmp_path, rows, expected):
    """Every shape an old file can take, and what it meant before today."""
    settings = _loaded(tmp_path, rows)
    assert settings["annotation_source"] == expected
    assert "Toxoplasma" not in settings
    assert "toxo" not in settings


def test_off_stays_off_even_as_the_word_false(tmp_path):
    """A value that reaches the fold as TEXT is still read as off.

    `bool('False')` is True, and the code this replaced used `bool()`.
    """
    settings = get_perform_regression_default_settings(
        {"src": str(tmp_path), "Toxoplasma": "False"})
    assert settings["annotation_source"] == ""


def test_the_shipped_example_file_still_loads(tmp_path):
    """`spaCR_settings/6_Regression_settings.csv` says `toxo,True`."""
    raw = load_settings_file(str(SHIPPED_EXAMPLE))
    assert raw.get("toxo") is True, "the example no longer carries the key"
    settings = get_perform_regression_default_settings(raw)
    assert settings["annotation_source"] == "toxoplasma"
    assert "toxo" not in settings and "Toxoplasma" not in settings


def test_the_migration_says_what_it_did(tmp_path, caplog):
    """A file whose meaning is carried across gets a line saying so."""
    with caplog.at_level(logging.INFO, logger="spacr.settings"):
        _loaded(tmp_path, [("Toxoplasma", "False")])
    said = " ".join(record.getMessage() for record in caplog.records)
    assert "Toxoplasma=False" in said and "annotation_source=''" in said, said


def test_the_doctor_names_the_replacement_and_what_the_old_value_means():
    """Not "renamed", and not "the value is ignored": both would be false."""
    for key in ("Toxoplasma", "toxo"):
        problems = _check_retired_keys({key: False})
        assert len(problems) == 1, key
        problem = problems[0]
        assert problem.severity == WARNING
        assert "folded into 'annotation_source'" in problem.message
        assert "false means no annotation" in problem.fix
        assert "ignored" not in problem.fix


def test_the_switch_is_retired_everywhere_a_setting_is_declared():
    """No type, no tooltip, no default, and both spellings are folds."""
    for key in ("Toxoplasma", "toxo"):
        assert key not in expected_types
        assert key not in tooltips
        assert RETIRED_SETTINGS[key] == "annotation_source"
        assert key in SEMANTIC_FOLDS
    assert "Toxoplasma" not in get_perform_regression_default_settings({})


def test_a_raw_dict_handed_to_ml_gets_the_same_answer():
    """ml.py reads a dict that may never have met the defaults."""
    from spacr.ml import _annotation_source, _toxoplasma_is_on

    assert _annotation_source({"Toxoplasma": False}) == ""
    assert _annotation_source({"toxo": True}) == "toxoplasma"
    assert _annotation_source({"Toxoplasma": False,
                               "annotation_source": "human"}) == "human"
    assert _toxoplasma_is_on({"toxo": True}) is True
    assert _toxoplasma_is_on({"Toxoplasma": False}) is False


def test_the_toxoplasma_figures_follow_the_name_not_the_switch():
    """The hyperLOPIT volcano and the GT1/ME49 reports are Toxoplasma's.

    Before the retirement they were gated on the switch, which defaulted on,
    so a run annotated with 'human' still drew them.
    """
    from spacr.ml import _toxoplasma_is_on

    assert _toxoplasma_is_on({"annotation_source": "toxoplasma"}) is True
    assert _toxoplasma_is_on({"annotation_source": "Toxoplasma gondii"})
    assert _toxoplasma_is_on({"annotation_source": "human"}) is False
    assert _toxoplasma_is_on({"annotation_source": ""}) is False
    assert _toxoplasma_is_on({"Toxoplasma": True,
                              "annotation_source": "human"}) is False


def test_a_parameter_sweep_trial_still_leaves_annotation_off():
    """The sweep turned the switch off for its trials; it says so by name now."""
    from spacr.parameter_sweep import settings_for_trial

    settings = settings_for_trial({"src": "/tmp"}, {"alpha": 1.0})
    assert settings["annotation_source"] == ""
    assert "Toxoplasma" not in settings
