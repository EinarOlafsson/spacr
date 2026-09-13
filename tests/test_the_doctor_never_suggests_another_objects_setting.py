"""A typo suggestion must never cross from one object role to another.

WHY THIS IS WORSE THAN NO SUGGESTION AT ALL. `difflib` scores on characters,
and `cell_`, `nucleus_`, `pathogen_` and `organelle_` keys are identical apart
from that one word -- so a key one role does not have matches another role's at
well over the 0.85 cutoff, and the doctor says "did you mean ...?" in a tone
that invites the reader to do it.

Following it changes a DIFFERENT CHANNEL's preprocessing. The run then works,
produces masks, and is wrong in a way nothing reports. A wrong suggestion gets
FOLLOWED; silence gets investigated.

THE OBSERVED CASE, which is what this file exists for. Before 364 declared the
organelle preprocessing settings, a user who worked out
`remove_background_organelle` was told:

    'remove_background_organelle' is not a spaCR setting;
        did you mean 'remove_background_cell'?

AND NOTE THE SHAPE. The role is the LAST word, not the first. A guard that
looked only at `<role>_suffix` -- which is the common shape and the one
`object_roles.split_role_setting` handles -- would have missed the only case
anybody has actually hit. `validate._object_role_in` looks at both ends for
exactly that reason.
"""
from __future__ import annotations

import difflib

import pytest

from spacr.validate import _check_unknown_keys, _object_role_in


def _suggestion(key):
    """What the doctor offers for ``key``, or None."""
    problems = _check_unknown_keys({key: 1})
    if not problems:
        return None
    text = str(problems[0].message)
    if "did you mean" not in text:
        return None
    return text.split("did you mean ")[-1].rstrip("?").strip("'\"")


def test_the_role_is_found_at_either_end_of_the_key():
    """`<role>_suffix` AND `prefix_<role>`, because the bug used the second."""
    assert _object_role_in("cell_min_area") == "cell"
    assert _object_role_in("organelle_diameter") == "organelle"
    assert _object_role_in("remove_background_organelle") == "organelle"
    assert _object_role_in("remove_background_cell") == "cell"
    assert _object_role_in("verbose") is None
    assert _object_role_in("adjust_cells") is None


def test_the_historical_cross_role_suggestion_cannot_come_back():
    """The exact case, reproduced by hiding the key the way it used to be.

    `remove_background_organelle` was undeclared, so the nearest live name was
    another role's. This asserts that even then the doctor stays silent rather
    than sending the reader at the wrong channel.
    """
    from spacr.settings import expected_types

    probe = "remove_background_organelle"
    without = sorted(k for k in expected_types if k != probe)
    raw = difflib.get_close_matches(probe, without, n=5, cutoff=0.85)
    assert raw, "the historical mismatch no longer reproduces; rewrite this test"
    assert _object_role_in(raw[0]) != _object_role_in(probe), (
        "difflib's best answer is no longer a different role, so this test is "
        "not exercising what it claims")

    kept = [c for c in raw if _object_role_in(c) in (None, _object_role_in(probe))]
    assert kept == [], (
        f"a cross-role name survived the guard and would be suggested: {kept}")


@pytest.mark.parametrize("typo,expected", [
    ("cell_min_are", "cell_min_area"),
    ("nucleus_diamter", "nucleus_diameter"),
    ("organelle_diamter", "organelle_diameter"),
    # the role-at-the-end shape must still be helped WITHIN its own role
    ("remove_background_organell", "remove_background_organelle"),
])
def test_a_typo_within_one_role_is_still_helped(typo, expected):
    """The guard must not buy silence by refusing to suggest anything."""
    assert _suggestion(typo) == expected


@pytest.mark.parametrize("typo,expected", [("verbos", "verbose"),
                                           ("figuresiz", "figuresize")])
def test_keys_with_no_object_role_are_unaffected(typo, expected):
    """Most settings name no role, and nothing about them should change."""
    assert _suggestion(typo) == expected


def test_no_live_setting_is_ever_offered_a_foreign_role():
    """Swept across the roles rather than argued from one example."""
    from spacr.settings import expected_types

    known = sorted(expected_types)
    crossed = []
    for role in ("cell", "nucleus", "pathogen", "organelle"):
        for suffix in ("min_area", "diameter", "background", "signal_to_noise",
                       "channel", "cellprob_threshold"):
            probe = f"{role}_{suffix}"
            if probe in expected_types:
                continue                      # a live key needs no suggestion
            got = _suggestion(probe)
            if got and _object_role_in(got) not in (None, role):
                crossed.append((probe, got))
    assert crossed == [], f"the doctor crossed object roles for: {crossed}"
