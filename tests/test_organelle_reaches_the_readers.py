"""Both database readers dropped the organelle table silently.

`io._read_and_join_tables` looped over the literal ``['nucleus', 'pathogen']``
for its roll-up, and `io._read_and_merge_data` had one hardcoded block per
table -- cytoplasm, nucleus, pathogen -- and no organelle block at all.

So asking for organelle measurements read the table, dropped it, and returned
a frame with no organelle columns and no message. `spacr.schema` declares
organelle a cell_id-linked child, `filters.py` and `merge_tables.py` both
handle it; only the two io readers did not.

This blocks instruction 76 outright: a SECOND organelle cannot be added until
the first one arrives.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.object_roles import ALL_ROLES, CHILD_ROLES, SEGMENTED_ROLES


def test_the_child_roles_are_the_segmented_ones_minus_the_cell():
    """The cell is the anchor; everything else belongs to one."""
    assert set(CHILD_ROLES) | {"cell"} == set(SEGMENTED_ROLES)
    assert "cell" not in CHILD_ROLES
    assert "organelle" in CHILD_ROLES


def test_organelle_is_in_the_child_roles_at_all():
    """The literal it replaced was ['nucleus', 'pathogen']."""
    assert "organelle" in CHILD_ROLES, (
        "organelle is missing from the roll-up again, so asking for it "
        "returns a frame with no organelle columns and no message")


def test_both_readers_take_the_roles_from_the_registry():
    """A literal in either reader means a new role reaches only one of them."""
    import inspect

    from spacr import io

    source = inspect.getsource(io)
    assert "for entity in ['nucleus', 'pathogen']" not in source, (
        "a reader spells the child roles as a literal again")
    assert "for entity in CHILD_ROLES" in source


def test_the_merge_reader_handles_every_organelle_role():
    """Organelle is a LOOP now, and that is the point of instruction 76.

    This used to require a literal ``if 'organelle' in data_dict:`` block,
    which was right when there was exactly one organelle and wrong the moment
    there were organelle_1 and organelle_2 -- a per-table block cannot cover
    a list that grows. The single-object tables keep their blocks; the
    organelles are iterated. Asserting the literal would now be asserting the
    limitation, and this file's own first test forbids spelling roles out.
    """
    import inspect

    from spacr import io
    from spacr.object_roles import ORGANELLE_ROLES

    source = inspect.getsource(io._read_and_merge_data)
    for role in ("cytoplasm", "nucleus", "pathogen"):
        assert f"if '{role}' in data_dict:" in source, (
            f"_read_and_merge_data has no block for {role!r}, so that table "
            f"is read and silently dropped")
    assert "for organelle_role in ORGANELLE_ROLES:" in source, (
        "the organelle tables are not iterated, so only the first slot "
        "would be read and the rest silently dropped")
    assert len(ORGANELLE_ROLES) >= 1


def test_the_organelle_block_keys_on_cell_id_like_its_siblings():
    """It is a cell_id-linked child; keying it any other way joins on a
    coincidence."""
    import inspect

    from spacr import io

    source = inspect.getsource(io._read_and_merge_data)
    block = source[source.index("for organelle_role in ORGANELLE_ROLES:"):]
    block = block[:block.index("if 'png_list' in data_dict:")]

    assert "dropna(subset=['cell_id'])" in block
    assert "_split_object_data(\n                organelles, 'prcfo', 'cell_id')" in block \
        or "'prcfo', 'cell_id'" in block
    # The ROLE, not the literal 'organelle'. With organelle_1 and
    # organelle_2 each slot has to report against its own name, or a row
    # dropped from the second is blamed on the first -- which is the same
    # class of mistake the literal was guarding against, one slot later.
    # Whitespace-normalised: the call is wrapped across three lines, and a
    # test that broke when someone reflowed an argument list would be
    # pinning the formatter rather than the behaviour.
    flat = " ".join(block.split())
    assert "_merge_grouped( merged_df, organelles_g_df, organelle_role)" in flat \
        or "_merge_grouped(merged_df, organelles_g_df, organelle_role)" in flat, (
        "the organelle roll-up is not named by its own role in the merge, so "
        "a dropped row would be reported against the wrong table")
