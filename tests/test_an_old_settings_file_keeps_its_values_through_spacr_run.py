"""`spacr-run <module> --settings old.csv` keeps what the old file said.

FOUND WHILE RETIRING THE LAST OF 364's SETTINGS, 2026-09-19, and it is a
defect in the migration itself rather than in any one setting.
`resolve_settings` layers the module's DEFAULTS first and the file on top,
and `settings._fold_renamed_settings` resolves a collision with "the NEW name
wins where both are present" -- which is right for one file and wrong here,
because the defaults have already filled in the new name. So the file's value
lost to the default, every time, on every rename the doctor reports:

    min_cell_count,50   ->  min_cells_per_well = 100   (the default)

with the fold saying "this settings file names both" about a file that named
one. The GUI never had this: it translates a dict before it meets a form.

The file is migrated before it meets the defaults now. These tests are
written against the CLI's own entry point, because that is where the bug was
-- the fold and the tables were correct throughout.
"""
from __future__ import annotations

import csv
import logging

import pytest

from spacr.cli import MODULES, resolve_settings


def _file(tmp_path, rows, name="settings.csv"):
    path = tmp_path / name
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows([("Key", "Value")] + list(rows))
    return str(path)


def test_a_renamed_value_is_not_replaced_by_the_default(tmp_path):
    """The measurement that found it, as an assertion."""
    module = MODULES["regression"]
    assert resolve_settings(module, None)["min_cells_per_well"] == 100, (
        "pick a probe unlike the default or this test proves nothing")

    resolved = resolve_settings(
        module, _file(tmp_path, [("src", str(tmp_path)),
                                 ("min_cell_count", "50")]))

    assert resolved["min_cells_per_well"] == 50
    assert "min_cell_count" not in resolved


@pytest.mark.parametrize("old, new, value", [
    ("min_n", "min_observations_per_hit", 7),
    ("positive_control", "positive_control_id", "c9"),
    ("negative_control", "negative_control_id", "c8"),
])
def test_every_rename_survives_the_trip(tmp_path, old, new, value):
    resolved = resolve_settings(
        MODULES["regression"],
        _file(tmp_path, [("src", str(tmp_path)), (old, str(value))]))
    assert resolved[new] == value
    assert old not in resolved


def test_the_retired_toxoplasma_switch_still_turns_annotation_off(tmp_path):
    """364's decision of 2026-09-19, through the path a cluster job takes."""
    from spacr.ml import _annotation_source
    from spacr.settings import get_perform_regression_default_settings

    resolved = resolve_settings(
        MODULES["regression"],
        _file(tmp_path, [("src", str(tmp_path)), ("Toxoplasma", "False")]))

    assert resolved["annotation_source"] == ""
    assert "Toxoplasma" not in resolved
    run = get_perform_regression_default_settings(dict(resolved))
    assert _annotation_source(run) == "", "the run would annotate anyway"


def test_a_split_reaches_both_of_its_names(tmp_path):
    resolved = resolve_settings(
        MODULES["regression"],
        _file(tmp_path, [("src", str(tmp_path)),
                         ("control_wells", "['c12']")]))
    assert resolved["stain_baseline_wells"] == ["c12"]
    assert resolved["analysis_excluded_wells"] == ["c12"]


def test_the_semantic_fold_collapses_the_step_count(tmp_path):
    """`gradient_accumulation: false` means one batch per step, not zero."""
    resolved = resolve_settings(
        MODULES["train"] if "train" in MODULES else MODULES["classify"],
        _file(tmp_path, [("src", str(tmp_path)),
                         ("gradient_accumulation", "False")]))
    assert resolved.get("gradient_accumulation_steps") == 1
    assert "gradient_accumulation" not in resolved


def test_a_current_file_is_left_alone_and_says_nothing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="spacr.cli"):
        resolved = resolve_settings(
            MODULES["regression"],
            _file(tmp_path, [("src", str(tmp_path)),
                             ("min_cells_per_well", "50")]))
    assert resolved["min_cells_per_well"] == 50
    assert [r.getMessage() for r in caplog.records] == []


def test_each_migrated_key_is_named_once_with_the_doctors_sentence(
        tmp_path, caplog):
    """The pre-flight runs next and can only see the new name."""
    with caplog.at_level(logging.WARNING, logger="spacr.cli"):
        resolve_settings(
            MODULES["regression"],
            _file(tmp_path, [("src", str(tmp_path)), ("min_n", "3"),
                             ("toxo", "False")]))
    said = [record.getMessage() for record in caplog.records]
    assert len(said) == 2, said
    assert any("'min_n' was renamed to 'min_observations_per_hit'" in line
               for line in said), said
    assert any("'toxo' was folded into 'annotation_source'" in line
               for line in said), said


def test_a_withdrawn_key_is_kept_for_the_pre_flight_to_name(tmp_path):
    """A key with no successor stays, so `validate` can still report it."""
    resolved = resolve_settings(
        MODULES["map_barcodes"],
        _file(tmp_path, [("src", str(tmp_path)), ("barcodes", "/x.csv")]))
    assert resolved["barcodes"] == "/x.csv"


def test_a_queued_job_is_layered_the_same_way(tmp_path):
    """`spacr.batch` says it layers "exactly as spacr-run layers them"."""
    from spacr.batch import Job, resolve_job_settings

    path = _file(tmp_path, [("src", str(tmp_path)), ("min_cell_count", "50")])
    assert resolve_job_settings(
        Job(module="regression", settings=path))["min_cells_per_well"] == 50
    assert resolve_job_settings(
        Job(module="regression",
            settings={"src": str(tmp_path), "min_cell_count": 50}),
    )["min_cells_per_well"] == 50
