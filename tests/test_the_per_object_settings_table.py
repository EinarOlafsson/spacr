"""The per-object table: one row per question, one column per object.

Instruction 364, approved by the maintainer over tabs and over leaving the
names flat. Instruction 326 is why it matters: the organelle count is
arbitrary up to 26, and a flat vocabulary grows by twenty settings per
organelle while a table grows by one column.

The property that makes it safe to edit through is that it is LOSSLESS. What
the user sees is the settings file rearranged, not transformed, so no saved
settings file, notebook or tutorial has to migrate.
"""
from __future__ import annotations

import pytest

from spacr.object_settings_table import (OBJECT_ORDER, column_label, families,
                                         from_table, questions, to_table, widen)


@pytest.fixture(scope="module")
def mask_settings():
    from spacr import settings as S
    return S.get_timelapse_settings({})


def test_the_table_round_trips_every_mask_setting(mask_settings):
    """Rearranged, not transformed. This is the whole safety argument."""
    assert from_table(to_table(mask_settings), mask_settings) == mask_settings


def test_the_table_round_trips_measure_too(mask_settings):
    from spacr import settings as S

    measure = S.get_measure_crop_settings({})
    assert from_table(to_table(measure), measure) == measure


def test_it_collapses_the_repetition_it_was_built_for(mask_settings):
    """113 per-object keys, far fewer questions. The saving is the point."""
    table = to_table(mask_settings)
    keys = sum(len(row) for row in table.values())
    assert keys > len(table), (
        f"{keys} keys became {len(table)} rows -- no repetition was collapsed")
    assert len(table) < keys / 1.5, (
        "the table barely collapses anything; check the object prefixes")


def test_an_organelle_slot_is_split_at_its_own_prefix():
    """`organelleb_min_area` is organelle B, not organelle A.

    The prefixes are matched longest-first. Backwards, every organelle slot
    files under `organelle` and the table silently shows one column where
    there are twenty-six -- which no assertion about totals would catch.
    """
    table = to_table({"organelle_min_area": 1, "organelleb_min_area": 2,
                      "organellec_min_area": 3})
    assert table["min_area"] == {"organelle": 1, "organelleb": 2,
                                 "organellec": 3}


def test_a_question_an_object_does_not_ask_is_absent_not_none(mask_settings):
    """A blank cell says "not applicable"; a None reads as "not set yet".

    cytoplasm is DERIVED -- cell minus the rest -- so it has no channel and
    no diameter, and that is a different fact from a diameter nobody has
    chosen yet.
    """
    table = to_table(mask_settings)
    for question in ("channel", "diameter"):
        if question in table:
            assert "cytoplasm" not in table[question], (
                f"cytoplasm was given a {question}")


def test_adding_an_object_is_one_column_not_twenty_settings(mask_settings):
    """The operation instruction 326 needs.

    In the flat vocabulary a new organelle is twenty new settings that every
    consumer, tooltip and translation catalog has to learn. Here it is a
    column, and the number of QUESTIONS does not move at all.
    """
    table = to_table(mask_settings)
    before_rows = len(table)
    before_cells = sum(len(row) for row in table.values())

    wider = widen(table, "organellec")

    assert len(wider) == before_rows, "adding an object added a question"
    added = sum(len(row) for row in wider.values()) - before_cells
    assert added > 0, "the new column is empty"
    # It starts where the first organelle is, not at a global default.
    for question, row in wider.items():
        if "organellec" in row and "organelle" in table.get(question, {}):
            assert row["organellec"] == table[question]["organelle"]


def test_widening_is_still_lossless(mask_settings):
    """A widened table must still flatten to real settings keys."""
    wider = widen(to_table(mask_settings), "organellec")
    flat = from_table(wider, mask_settings)
    assert set(mask_settings) <= set(flat)
    assert any(k.startswith("organellec_") for k in flat)


def test_a_column_header_is_something_a_user_recognises():
    """`organelleb` is a storage spelling, not a name for a person.

    The letter suffixes exist because object types are embedded in
    underscore-separated object keys and `organelle2` is ambiguous with label
    2 -- an implementation constraint with no business in a column header.
    """
    assert column_label("cell") == "Cell"
    label = column_label("organelleb")
    assert "organelleb" not in label, label
    assert "2" in label, label


def test_the_object_order_puts_the_derived_kind_last():
    """cytoplasm answers far fewer questions, so its column is sparse.

    Sparse columns belong at the edge; in the middle they break up the dense
    ones a reader is comparing.
    """
    assert OBJECT_ORDER[-1] == "cytoplasm"
    assert OBJECT_ORDER[:3] == ("cell", "nucleus", "pathogen")


def test_questions_and_families_agree(mask_settings):
    """`questions` lists rows; `families` says who is in each."""
    table = to_table(mask_settings)
    assert set(questions(mask_settings)) == set(table)
    fam = families(mask_settings)
    for question, objects in fam.items():
        assert set(objects) == set(table[question])
