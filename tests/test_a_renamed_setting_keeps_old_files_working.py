"""A renamed setting does not break the settings files already written.

Instruction 364 renames seven settings whose names said the wrong thing.
Every one of them appears in settings CSVs people have saved, and A
MISSING KEY IS A DEFAULT RATHER THAN AN ERROR -- so a rename without a
migration is a silent behaviour change on somebody else's machine, which
is the exact failure this instruction exists to remove rather than to
create.

TWO MECHANISMS, AND THEY ANSWER DIFFERENT QUESTIONS.
`spacr.settings._fold_renamed_settings` makes the RUN work: it moves an
old key onto the new name, carrying its value, before any default is
filled in. `spacr.validate.RETIRED_SETTINGS` makes the run SAY SO: the
pre-flight names the old key and what replaced it. Neither is sufficient
alone -- a fold with no message is a silent rename, and a message with no
fold is a broken settings file with an explanation.

THE ORDER INSIDE THE FACTORY IS PART OF THE CONTRACT. The fold runs
BEFORE the `setdefault`s. Run it after and the default is already sitting
under the new name, `setdefault` declines, and the user's own value is
dropped on the floor -- the run works, uses the wrong number, and says
nothing.
"""

from __future__ import annotations

import pytest

from spacr.settings import (RENAMED_SETTINGS, _fold_renamed_settings,
                            expected_types)
from spacr.validate import RETIRED_SETTINGS

#: The factory each renamed setting is declared by, so the fold can be
#: asked of the function a user's settings file actually goes through.
#:
#: IT HELD ONE ENTRY AND THAT WAS THE HOLE. `test_the_fold_runs_before_the
#: _defaults_are_filled_in` walks this table, so a rename missing from it
#: was never asked the one question that matters -- and three of the four
#: renames were missing. The fold had exactly ONE call site in
#: `settings.py`, so `min_observations_per_hit`, `min_cells_per_well` and
#: both control identifiers reached their factories after `setdefault` had
#: already put the default in place: an old settings file naming the old
#: key kept running and silently lost its VALUE, which is the one failure
#: a rename is supposed to make impossible.
#:
#: `test_every_rename_names_the_factory_that_declares_it` now fails when a
#: new rename is added without an entry here, so the table cannot go stale
#: again in the direction that hides the bug.
FACTORIES = {
    "window_length": "set_default_generate_barecode_mapping",
    "min_observations_per_hit": "get_perform_regression_default_settings",
    "min_cells_per_well": "get_perform_regression_default_settings",
    "positive_control_id": "set_default_analyze_screen",
    "negative_control_id": "set_default_analyze_screen",
    "analysis_excluded_wells": "get_perform_regression_default_settings",
    "stain_baseline_wells": "set_analyze_invasion_defaults",
}


def test_every_rename_names_the_factory_that_declares_it():
    """A rename with no factory entry is a rename nothing checks.

    The ordering test below walks `FACTORIES`, so an omission here is not
    a gap in a table -- it is a rename whose fold has never been asked to
    run before its defaults. That is exactly how three of the four got
    into the tree with only one call site between them.
    """
    from spacr.settings import RENAMED_SETTINGS

    for _old, new in RENAMED_SETTINGS.items():
        for name in _targets(new):
            assert name in FACTORIES, (
                f"{name} is a renamed setting with no factory named here, "
                "so nothing asserts its fold runs before its defaults")


def _targets(new):
    """The new name, or all of them when the entry is a SPLIT."""
    return (new,) if isinstance(new, str) else tuple(new)


def test_every_rename_is_declared_in_all_three_places():
    """A rename is a fold, a retirement message and a live new name."""
    for old, new in RENAMED_SETTINGS.items():
        assert RETIRED_SETTINGS.get(old) == new, (
            f"{old} folds to {new} but the pre-flight does not say so")
        for name in _targets(new):
            assert name in expected_types, (
                f"{name} is a new name for {old} and has no declared type")
        assert old not in expected_types, (
            f"{old} was renamed and is still declared, so it is live and "
            "withdrawn at once")


def test_an_old_settings_file_still_runs_and_keeps_its_value():
    """The value moves with the name. This is the whole point."""
    for old, new in RENAMED_SETTINGS.items():
        settings = {old: "the user's own value"}
        _fold_renamed_settings(settings)
        for name in _targets(new):
            # A SPLIT SENDS THE VALUE TO BOTH. The old key did both jobs
            # at once, so sending it to one half would change what the
            # other does on a file nobody has edited.
            assert settings[name] == "the user's own value"
        assert old not in settings, (
            "the old key survived the fold, so a later validation pass "
            "will report it as unknown on a file that has been migrated")


def test_the_new_name_wins_when_both_are_present():
    """An explicit new spelling is the later decision of the two."""
    for old, new in RENAMED_SETTINGS.items():
        for name in _targets(new):
            settings = {old: "old", name: "new"}
            _fold_renamed_settings(settings)
            assert settings[name] == "new"
            assert old not in settings


def test_the_fold_runs_before_the_defaults_are_filled_in():
    """Otherwise the user's value is dropped and nothing says so.

    Asked of the real factory rather than of the helper, because the
    ordering is a property of the factory and the helper cannot enforce
    it.
    """
    import spacr.settings as module

    for new, factory_name in FACTORIES.items():
        # A SPLIT REACHES ITS TARGETS FROM ONE OLD KEY, so the lookup is
        # "which old name leads here" rather than "which maps exactly to
        # this string" -- `control_wells` maps to a TUPLE, and asking for
        # an exact match raised StopIteration on both of its halves.
        old = next(o for o, n in RENAMED_SETTINGS.items()
                   if new in _targets(n))
        factory = getattr(module, factory_name)
        default = factory({}).get(new)
        # A PROBE THE DEFAULT CANNOT BE. Comparing "not the default" was
        # not enough: a setting whose default is None -- and
        # `stain_baseline_wells` is one -- has nothing to differ from, and
        # the assertion could only say "no default to compare against"
        # about the very case it exists to check.
        probe = default + 1 if isinstance(default, int) else "mine"
        assert probe != default
        mine = factory({old: probe}).get(new)
        assert mine == probe, (
            f"{factory_name} filled in the default before folding {old}, so "
            "a settings file naming the old key silently loses its value")


def test_a_settings_file_naming_the_old_key_is_told_what_replaced_it():
    """The message, which is the half a fold cannot provide."""
    from spacr.validate import _check_retired_keys

    for old, new in RENAMED_SETTINGS.items():
        problems = _check_retired_keys({old: 1})
        assert problems, f"{old} is retired and the pre-flight is silent"
        text = " ".join(str(one) for one in problems)
        assert old in text
        for name in _targets(new):
            assert name in text, (
                f"the message for {old} does not name {name}: {text}")


def test_no_retirement_points_at_a_name_that_was_renamed_again():
    """A chain of renames sends the reader to a second dead end.

    `minimum_cell_count` was retired into `min_cell_count`, and
    `min_cell_count` was later renamed to `min_cells_per_well` -- so the
    first entry named a setting that no longer existed. The user follows
    it, finds nothing, and has no reason to think the trail continues.
    Asserted over every entry rather than that one, because the trap is
    the SECOND rename and there will be more of those.
    """
    live = set(expected_types)
    broken = {}
    for old, new in RETIRED_SETTINGS.items():
        if not new:
            continue
        for name in _targets(new):
            if name not in live:
                broken[old] = new
    assert broken == {}, (
        "these retirements name a setting that no longer exists, which is "
        f"a chain rather than a replacement: {broken}")


def test_nothing_is_renamed_onto_a_name_that_is_itself_retired():
    """A chain of renames sends the reader in a circle."""
    for _old, new in RENAMED_SETTINGS.items():
        for name in _targets(new):
            assert name not in RETIRED_SETTINGS, (
                f"{name} is a rename target and is itself retired")
