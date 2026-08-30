"""Reading an organelle summary column name back into a described feature.

spaCR can carry four organelle roles, and each writes its own column prefix --
``organelle_summary_organelleb_area`` and so on. They must all resolve to the
SAME described property, because the description is what a report prints beside
the number: two roles described differently would read as two different
measurements when they are one measurement of two objects.

The uncovered arc is the loop finding no role prefix at all, which is what
every non-organelle column name does.
"""
from __future__ import annotations

import pytest


def test_every_organelle_role_resolves_to_the_same_property():
    """The canonicalising branch, across all four roles.

    Each role's prefix is rewritten to the canonical one before the lookup, so
    the description comes from a single entry. The object_type still records
    WHICH organelle it was.
    """
    from spacr.feature_dict import ORGANELLE_ROLES, _parse_organelle_summary

    descriptions = set()
    for role in ORGANELLE_ROLES:
        entry = _parse_organelle_summary(f"organelle_summary_{role}_count")
        assert entry is not None
        descriptions.add(entry.description)

    assert len(descriptions) == 1, (
        "the roles must share one description; they are one measurement of "
        f"different objects, got {descriptions}")


def test_a_name_with_no_role_prefix_is_not_an_organelle_summary():
    """The loop running out without matching, then the plain lookup failing.

    Every name that is not an organelle summary takes this route, so it is the
    common path rather than an edge case. None is the answer, and it is what
    lets the caller go on to the other parsers -- an entry invented here would
    claim a column this parser does not understand.
    """
    from spacr.feature_dict import _parse_organelle_summary

    assert _parse_organelle_summary("not_a_summary_name") is None
    assert _parse_organelle_summary("cell_area") is None


def test_the_longest_role_prefix_wins():
    """The ``key=len, reverse=True`` sort, which is load-bearing.

    'organelle' is a prefix of 'organelleb'. Sorted shortest-first, the
    'organelle' branch would claim every organelleb column and describe it as
    the wrong object -- the roles would silently collapse into one.
    """
    from spacr.feature_dict import _parse_organelle_summary

    entry = _parse_organelle_summary("organelle_summary_organelleb_count")

    assert entry.object_type == "organelleb"


def test_an_unknown_property_under_a_known_role_loses_which_organelle_it_was():
    """The ``break`` after a role matched but the property did not.

    Breaking rather than continuing is right -- the role IS matched, so trying
    shorter prefixes could only mis-attribute it. But the fall-through then
    hard-codes ``object_type="organelle"``, so an unrecognised property under
    ``organelleb`` comes back attributed to ``organelle``.

    Pinned as CURRENT BEHAVIOUR, not endorsed. The column name is preserved in
    full, so nothing is lost that cannot be recovered -- but a report grouping
    unknown features by object_type will file this one under the wrong
    organelle. It matters only for properties spaCR does not know, which is why
    it has gone unnoticed, and it is recorded in instruction 310.
    """
    from spacr.feature_dict import _parse_organelle_summary

    entry = _parse_organelle_summary(
        "organelle_summary_organelleb_a_property_that_does_not_exist")

    assert entry is not None
    assert entry.column == (
        "organelle_summary_organelleb_a_property_that_does_not_exist")
    assert entry.object_type == "organelle"      # NOT organelleb -- see above


def test_a_summary_name_with_an_unrecognised_role_runs_the_loop_out():
    """The loop completing with no prefix matched at all.

    ``organelle_summary_zzz_count`` looks like a summary and names no role
    spaCR has, so every iteration continues and the plain lookup below decides.
    """
    from spacr.feature_dict import _parse_organelle_summary

    entry = _parse_organelle_summary("organelle_summary_zzz_count")

    assert entry is not None
    assert entry.column == "organelle_summary_zzz_count"
