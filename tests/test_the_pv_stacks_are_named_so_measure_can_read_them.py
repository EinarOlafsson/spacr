"""Item 449: the merged stacks must carry names spaCR's Measure can parse.

A trial Measure run over the first stacks failed every field --
`cell.prcf disagrees with its component identity columns`, with every
identity column reading `error` -- because the dataset's own stems have
the experiment and the acquisition in front of the plate, and
`schema.parse_field_stem` splits on underscores.

These tests hold the naming, and the one thing that makes it more than a
string edit: two different experiments in this dataset both have a
`plate1`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_pv_count_dataset", ROOT / "tools" / "build_pv_count_dataset.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

from spacr import schema

CSA = "CSA_screen__screen_20250124_133156__plate1_A02_1_1"
MTOC = "MTOC_Screen__mtocScreen_20250530_111302__plate1_G13_9_1"


def test_the_datasets_own_stem_cannot_be_parsed_at_all():
    """The defect, stated as the thing it broke."""
    with pytest.raises(schema.SchemaError):
        schema.parse_field_stem(CSA)


def test_the_renamed_field_parses():
    field = schema.parse_field_stem(builder.field_name(CSA))
    assert field.plateID == "CSAscreen-plate1"
    assert (field.rowID, field.columnID, field.fieldID) == ("r1", "c2", "f1")


def test_two_experiments_with_a_plate1_do_not_collide():
    """The reason the experiment is folded in rather than dropped. Without
    it both of these are `plate1_A02_1`-shaped and Measure would key two
    different fields to one identity."""
    csa = schema.parse_field_stem(builder.field_name(CSA))
    mtoc = schema.parse_field_stem(builder.field_name(MTOC))
    assert csa.plateID != mtoc.plateID
    assert csa.plateID == "CSAscreen-plate1"
    assert mtoc.plateID == "MTOCScreen-plate1"


def test_the_well_and_field_survive_the_rename():
    field = schema.parse_field_stem(builder.field_name(MTOC))
    assert (field.rowID, field.columnID, field.fieldID) == ("r7", "c13", "f9")


def test_the_folder_is_the_experiment_and_the_plate():
    """One folder per plate was not enough: a folder holding both
    experiments' plate1 hands Measure two fields with one identity."""
    assert builder.plate_of(CSA) == "CSAscreen-plate1"
    assert builder.plate_of(MTOC) == "MTOCScreen-plate1"


def test_a_stem_of_another_shape_is_left_alone():
    """Refusing to guess is better than inventing an identity."""
    assert builder.field_name("something_else") == "something_else"
    assert builder.plate_of("something_else") == "unplated"


def test_no_underscore_is_smuggled_in_through_the_experiment():
    """`CSA_screen` has one, and an underscore is what the parser splits
    on -- so it must not survive into the name."""
    name = builder.field_name(CSA)
    assert name.count("_") == 2, name
