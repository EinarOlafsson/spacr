"""Every organelle slot's three per-slot keys reach the settings dialog.

`settings.py` builds the category map at import time by walking the
extra organelle roles and appending each slot's ``channel``, ``mask_dim``
and ``chann_dim`` key to the General category. The append is guarded:

    if _key in expected_types and _key not in categories['General']:

Both halves of that guard are always true today -- all 75 keys are
declared, and none is pre-listed -- so the false arc cannot run. That is
worth pinning rather than shrugging at, because the guard is silent in
the direction that matters: a slot key that stopped being declared would
simply not appear in the dialog, with no error anywhere, and the setting
would be unreachable from the interface while still being read by the
pipeline.
"""
from __future__ import annotations

import inspect

import pytest

from spacr import settings as S

SUFFIXES = ("channel", "mask_dim", "chann_dim")


def _extra_roles():
    """The slots the loop walks: every role but the first.

    The first is the base ``organelle``, whose keys are in the category
    map already -- which is why the loop starts at index 1.
    """
    return S.ORGANELLE_SLOT_ROLES[1:]


def test_there_are_extra_slots_to_walk():
    """Otherwise every assertion below is vacuous over an empty loop."""
    assert len(_extra_roles()) >= 2
    assert S.ORGANELLE_SLOT_ROLES[0] == "organelle"


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_every_slot_declares_this_key(suffix):
    """THE PIN, first half: ``_key in expected_types``.

    A key absent here is a setting the dialog never offers. It is the
    quiet failure -- no exception, no missing widget error, just a
    control that is not there.
    """
    undeclared = [f"{role}_{suffix}" for role in _extra_roles()
                  if f"{role}_{suffix}" not in S.expected_types]

    assert undeclared == [], (
        f"{len(undeclared)} organelle slot keys are not in expected_types, "
        f"so the settings dialog silently omits them: {undeclared[:5]}")


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_no_slot_key_is_listed_in_general_before_the_loop(suffix):
    """THE PIN, second half: ``_key not in categories['General']``.

    The guard would swallow a duplicate rather than showing one. A key
    listed twice puts the same control on the screen twice, and the
    second one writes over the first -- so this being unreachable is the
    statement that the base list and the generated one do not overlap.
    """
    general = list(S.categories["General"])
    generated = [f"{role}_{suffix}" for role in _extra_roles()]

    assert len(general) == len(set(general)), (
        "the General category has duplicates, so a control is built twice")
    for key in generated:
        assert general.count(key) <= 1, (
            f"{key} appears in General more than once")


def test_every_generated_key_actually_landed_in_general():
    """The outcome, rather than the guard: the loop's whole purpose.

    Asserted over the WHOLE set rather than a sample, because the failure
    mode is one slot missing out of twenty-five, which is exactly what a
    sample misses.
    """
    general = set(S.categories["General"])
    missing = [f"{role}_{suffix}"
               for role in _extra_roles() for suffix in SUFFIXES
               if f"{role}_{suffix}" not in general]

    assert missing == [], (
        f"{len(missing)} generated slot keys never reached the General "
        f"category: {missing[:5]}")


def test_the_three_suffixes_are_the_ones_the_loop_walks():
    """A fourth per-slot suffix added to the code and not here would be
    untested; this fails rather than passing quietly over two of three."""
    source = inspect.getsource(S)
    marker = source.index("for _suffix in (")
    clause = source[marker:source.index("\n", marker)]

    for suffix in SUFFIXES:
        assert repr(suffix).strip("'\"") in clause, (
            f"the loop no longer walks {suffix!r}")
    assert clause.count(",") == len(SUFFIXES) - 1, (
        f"the loop walks a different number of suffixes than the {len(SUFFIXES)} "
        f"this file checks: {clause}")


def test_the_advanced_and_basic_organelle_keys_are_carried_too():
    """The two `extend` calls above the suffix loop, and the reason the
    prefix filter is there: only keys that are per-organelle are given a
    per-slot copy."""
    advanced = set(S.categories["Organelle advanced"])
    basic = set(S.categories["Organelle"])

    for role in _extra_roles()[:3]:
        assert any(key.startswith(role) for key in basic | advanced), (
            f"slot {role} contributed no organelle settings at all")
