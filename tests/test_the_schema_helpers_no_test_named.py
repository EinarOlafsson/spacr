"""Schema helpers no test had ever named.

Instruction 60, on the module that decides what a column is CALLED. Ten
public callables in ``spacr.schema`` had never appeared in a test, and two of
them carry rules whose failure mode is a plate whose database will not open
and a warning the user has been taught to ignore.

``canonical_rename_plan`` folds case, because SQLite compares identifiers
case-insensitively and these frames go through ``to_sql``: a frame holding
``row`` and ``rowid`` already HAS the canonical column, and renaming ``row``
to ``rowID`` produces a pair pandas is happy with and ``to_sql`` refuses.

``comparable_key_value`` is the opposite rule. ``1``, ``1.0``, ``'01'`` and
``' 1 '`` are one well, and a naive equality warns on every file that stored
one copy as text and the other as a number -- a warning that fires every time
is a warning nobody reads when the real one arrives.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# comparable_key_value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [1, 1.0, "1", "01", " 1 ", "1.0"])
def test_every_spelling_of_one_well_compares_equal(value):
    from spacr.schema import comparable_key_value

    assert comparable_key_value(value) == comparable_key_value(1)


@pytest.mark.parametrize("blank", [None, float("nan"), "", "   "])
def test_missing_is_its_own_value_and_they_all_agree(blank):
    """Two columns that are both blank on a row AGREE there. Treating each
    kind of blank as different would report a disagreement on every row a
    field was left empty."""
    from spacr.schema import comparable_key_value

    assert comparable_key_value(blank) == comparable_key_value(None)


def test_two_different_wells_do_not_compare_equal():
    from spacr.schema import comparable_key_value

    assert comparable_key_value("A01") != comparable_key_value("A02")


def test_a_column_is_reduced_element_by_element():
    from spacr.schema import comparable_key_values

    assert comparable_key_values([1, "01", None]) == \
        comparable_key_values(["1", 1.0, ""])


def test_the_result_is_hashable_because_it_is_used_as_a_key():
    from spacr.schema import comparable_key_values

    assert isinstance(comparable_key_values([1, 2]), tuple)
    assert len(set(comparable_key_values([1, 2]))) == 2


# ---------------------------------------------------------------------------
# canonical_rename_plan
# ---------------------------------------------------------------------------

def test_a_legacy_spelling_is_planned_for_rename():
    from spacr.schema import canonical_rename_plan

    plan = canonical_rename_plan(["row", "col", "value"])
    assert plan, "nothing was planned for a frame full of legacy spellings"
    assert all(old != new for old, new in plan.items())


def test_a_frame_that_already_has_the_canonical_column_is_left_alone():
    """Renaming `row` to `rowID` beside an existing `rowID` gives pandas a
    duplicate it is happy with and to_sql is not."""
    from spacr.schema import canonical_rename_plan

    plan = canonical_rename_plan(["row", "rowID"])
    assert "row" not in plan


def test_the_collision_test_folds_case():
    """SQLite compares identifiers case-insensitively, so `rowid` IS `rowID`
    as far as to_sql is concerned -- and a plan that ignored case produced a
    database that could not be opened at all."""
    from spacr.schema import canonical_rename_plan

    plan = canonical_rename_plan(["row", "rowid"])
    assert "row" not in plan, plan


def test_a_canonical_frame_needs_no_plan():
    from spacr.schema import canonical_rename_plan

    assert canonical_rename_plan(["rowID", "columnID", "plateID"]) == {}


def test_an_empty_frame_gives_an_empty_plan():
    from spacr.schema import canonical_rename_plan

    assert canonical_rename_plan([]) == {}


def test_an_explicit_target_overrules_the_derived_one():
    """`requested` is ``{source: target}``, not a filter -- it is how a
    caller that has already decided where a column goes says so, and the
    plan has to honour it rather than re-deriving a different answer."""
    from spacr.schema import canonical_rename_plan

    plan = canonical_rename_plan(["row"], requested={"row": "plateID"})
    assert plan.get("row") == "plateID"


def test_an_explicit_target_that_already_exists_is_still_refused():
    """The keep-both rule is about the FRAME, so naming the target by hand
    does not make a duplicate column safe to create."""
    from spacr.schema import canonical_rename_plan

    plan = canonical_rename_plan(["row", "plateID"],
                                 requested={"row": "plateID"})
    assert "row" not in plan, plan
