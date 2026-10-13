"""``migrate_unescaped_plate_names`` is idempotent, and why.

A plate folder whose name holds an underscore -- ``exp_1`` -- used to
produce ``exp_1_A01_1_1.npy``: five separator-delimited components for a
four-component grammar, which made every field of that plate unmeasurable.
The migration escapes the plate component so the plate can be measured
without re-segmenting it.

RUNNING IT TWICE MUST NOT CORRUPT ANYTHING, and that is the whole risk:
escaping is not idempotent, because a literal percent is escaped first, so
``exp%5F1_A01_1_1`` would become ``exp%255F1_A01_1_1``. Safety comes from
the component COUNT -- a migrated stem has four components and is skipped
before escaping is ever attempted.

This file also pins the premise a deletion rests on. The function used to
carry a second guard, ``if safe == stem: continue``, which could not fire:
more than four components means the plate holds a separator, and escaping
a separator always changes the string. It was counted by instruction 288
as an uncoverable item and is gone; the property that makes its absence
safe is asserted below rather than left implied.
"""
from __future__ import annotations

import os

import pytest

from spacr.io import migrate_unescaped_plate_names


def _plate(root, folder, name):
    """Write one array file under ``<root>/<folder>/``."""
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"")
    return path


def test_a_raw_separator_in_the_plate_is_escaped(tmp_path):
    """The case the migration exists for."""
    _plate(tmp_path, "merged", "exp_1_A01_1_1.npy")
    planned = migrate_unescaped_plate_names(str(tmp_path), dry_run=True)
    assert len(planned) == 1
    _old, new = planned[0]
    assert os.path.basename(new) == "exp%5F1_A01_1_1.npy"


def test_a_plate_that_needs_nothing_is_left_alone(tmp_path):
    """Four components, so it never reaches the escaping at all."""
    _plate(tmp_path, "merged", "exp1_A01_1_1.npy")
    assert migrate_unescaped_plate_names(str(tmp_path), dry_run=True) == []


def test_running_it_twice_changes_nothing_the_second_time(tmp_path):
    """THE PROPERTY THE DELETED GUARD LOOKED LIKE IT PROTECTED.

    It does not come from comparing the escaped name to the original; it
    comes from the component count. An already-migrated stem has four
    components and is skipped before escaping is attempted -- which is
    what stops ``exp%5F1`` becoming ``exp%255F1``.
    """
    _plate(tmp_path, "merged", "exp_1_A01_1_1.npy")
    migrate_unescaped_plate_names(str(tmp_path))
    assert (tmp_path / "merged" / "exp%5F1_A01_1_1.npy").is_file()
    assert migrate_unescaped_plate_names(str(tmp_path), dry_run=True) == []


@pytest.mark.parametrize("plate", [
    "exp_1", "a_b_c", "_leading", "trailing_", "with%_percent", "a b_c",
])
def test_escaping_always_changes_an_over_long_stem(plate):
    """The premise the deletion rests on, driven rather than assumed.

    If any plate name here escaped to itself, the removed guard would
    have been reachable and removing it would have changed behaviour.
    """
    from spacr.schema import KeyParseError, escape_field_stem_plate

    stem = f"{plate}_A01_1_1"
    assert len(stem.split("_")) > 4, "this case does not exercise the premise"
    try:
        escaped = escape_field_stem_plate(stem, timelapse=True)
    except KeyParseError:
        pytest.skip(f"{stem!r} is not a field stem")
    assert escaped != stem, (
        f"{stem!r} escapes to itself, so the deleted `safe == stem` guard "
        "was reachable after all")
