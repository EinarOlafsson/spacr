"""The doctor's table and the run's table must say the same thing.

WHY THIS FILE EXISTS. A rename is recorded in TWO places that nothing kept in
step: `spacr.validate.RETIRED_SETTINGS` decides what the doctor SAYS, and
`spacr.settings.RENAMED_SETTINGS` decides what the run DOES. Four renames were
recorded and never performed -- `minimum_cell_count`, `organelle_min_size`,
`organelle_max_size`, `redunction_method` -- so `spacr-doctor` printed
"renamed to X" about a file whose value the run then replaced with a default.

A SECOND, WORSE CLASS was recorded in NEITHER. The four families renamed by
b7ae412af (`<role>_FT`, `<role>_CP_prob`, `<role>_Signal_to_noise`,
`<role>_min_object_area`) were absent from both tables, so an old file got no
migration AND no warning: `cell_FT=0.42` came out of the mask factory as
`cell_flow_threshold=100`, the default, silently.

AND A THIRD: `set_default_settings_preprocess_generate_masks` never called the
fold at all, so for the mask pipeline both tables could have been perfect and
every value would still have been lost.

Each test below is written as a thing that must be IMPOSSIBLE, because the
defect was never a wrong value in a table -- it was the absence of anything
that would notice.
"""
from __future__ import annotations

import pytest

from spacr.object_roles import (ALL_ROLES, RENAMED_SETTING_SUFFIXES,
                                split_role_setting)
from spacr.settings import (RENAMED_SETTINGS, SEMANTIC_FOLDS, expected_types,
                            _fold_renamed_settings, surviving_setting_name)
from spacr.validate import RETIRED_SETTINGS

PROBE = "PROBE-VALUE-THAT-NO-DEFAULT-USES"

#: A handful of roles is enough and the full 705 is not: the suffix rules are
#: role-agnostic by construction, so a rule that works for one generated slot
#: works for all of them. These cover the segmented roles, the derived one,
#: the first organelle slot and two generated ones.
#:
#: `organellea` IS NOT A ROLE and is deliberately not listed. The slots run
#: `organelle`, `organelleaa`, `organelleab`, ... -- there is no single-letter
#: spelling -- so parametrising on it produced tests that skipped themselves
#: and proved nothing. `test_every_sample_role_is_real` keeps that from
#: happening again silently.
SAMPLE_ROLES = ("cell", "nucleus", "pathogen", "cytoplasm",
                "organelle", "organelleq", "organelleaa")


def test_every_sample_role_is_real():
    """A misspelt role makes every test parametrised on it skip, not fail.

    Found the hard way: `organellea` looked like a generated slot, is not
    one, and turned twelve assertions into twelve skips that read as passes.
    """
    unreal = [role for role in SAMPLE_ROLES if role not in ALL_ROLES]
    assert unreal == [], (
        f"these are not roles, so every case using them skips: {unreal}")


def _targets(replacement):
    """The names a RETIRED_SETTINGS replacement points at."""
    if isinstance(replacement, str):
        return (replacement,)
    return tuple(replacement)


def test_every_rename_the_doctor_reports_is_performed_by_the_run():
    """IMPOSSIBLE: the doctor names a successor the run never moves to.

    This is the exact defect. All four of the missing renames fail here.
    """
    unperformed = {}
    for old, replacement in RETIRED_SETTINGS.items():
        if not replacement or old in SEMANTIC_FOLDS:
            continue
        settings = {old: PROBE}
        _fold_renamed_settings(settings)
        for name in _targets(replacement):
            if settings.get(name) != PROBE:
                unperformed[old] = replacement
    assert unperformed == {}, (
        "the doctor reports these as renamed and the run does not perform "
        f"them, so the value is replaced by a default: {unperformed}")


def test_a_semantic_fold_is_declared_rather_than_merely_absent():
    """IMPOSSIBLE: a semantic migration quietly treated as a plain rename.

    `gradient_accumulation` is a boolean and its successor is a step COUNT, so
    moving the value puts `False` where `int()` is waiting -- and `int(False)`
    is 0, which is not a state the code has. `steps = 1` is the off state.

    It is exempt from the agreement test above, so the exemption has to be a
    declaration rather than an accident of omission, or the next reader adds
    it to RENAMED_SETTINGS to make that test pass and breaks the run.
    """
    assert SEMANTIC_FOLDS, "the exemption set must not be empty"
    for key in SEMANTIC_FOLDS:
        assert key in RETIRED_SETTINGS, (
            f"{key} claims a semantic fold but the doctor does not know it")
        assert key not in RENAMED_SETTINGS, (
            f"{key} is a SEMANTIC migration and must not be a plain move")
        assert surviving_setting_name(key) == (), (
            f"{key} must not resolve as a rename")
        settings = {key: False, "gradient_accumulation_steps": 8}
        _fold_renamed_settings(settings)
        assert settings[key] is False, "the fold must leave it alone"


@pytest.mark.parametrize("role", SAMPLE_ROLES)
@pytest.mark.parametrize("old_suffix", sorted(RENAMED_SETTING_SUFFIXES))
def test_a_suffix_rename_works_for_every_role_or_none(role, old_suffix):
    """IMPOSSIBLE: a rename that works on one slot and not the one beside it.

    `organelle_min_size` was in the table and `organelleq_min_size` was not,
    so the same setting migrated in slot one and vanished in slot seventeen.
    """
    new = f"{role}_{RENAMED_SETTING_SUFFIXES[old_suffix]}"
    old = f"{role}_{old_suffix}"
    # TWO DIFFERENT REASONS TO SKIP, and saying which matters: "the new name
    # is not live here" means the family does not exist for this role, while
    # "the old name is STILL live" means the rule must deliberately not fire
    # -- the seven `_size` cases. Reporting both as "not a live setting" hid
    # the second, which is the one with teeth.
    if new not in expected_types:
        pytest.skip(f"{new} is not a setting, so {role} has no {old_suffix} "
                    "family to rename")
    if old in expected_types:
        pytest.skip(f"{old} is STILL LIVE and must not be migrated; "
                    "test_no_suffix_rename_retires_a_setting_that_is_still_"
                    "live asserts that")
    settings = {old: PROBE}
    _fold_renamed_settings(settings)
    assert settings.get(new) == PROBE, (
        f"{old} did not reach {new}; a role-family rename must not depend "
        "on which slot the user happened to use")
    assert old not in settings, f"{old} was left behind as a dead key"


def test_no_suffix_rename_retires_a_setting_that_is_still_live():
    """IMPOSSIBLE: a rename rule eating a working setting.

    Seven `_size` keys are still live and mean something DIFFERENT from their
    `_area` neighbours -- `cell_min_size` filters at measurement time,
    `cell_min_area` at segmentation time. A blanket `_size -> _area` would
    move a measurement filter onto a segmentation control.
    """
    eaten = []
    for role in ALL_ROLES:
        for old_suffix in RENAMED_SETTING_SUFFIXES:
            old = f"{role}_{old_suffix}"
            if old not in expected_types:
                continue
            settings = {old: PROBE}
            _fold_renamed_settings(settings)
            if settings != {old: PROBE}:
                eaten.append((old, settings))
    assert eaten == [], (
        f"these are LIVE settings and a suffix rule migrated them: {eaten}")


def test_the_seven_live_size_settings_are_the_ones_we_measured():
    """The liveness gate is load-bearing, so pin what it is protecting.

    If this list changes, the `_size -> _area` rule's blast radius changed
    with it and somebody should look rather than update the number.
    """
    live = sorted(k for k in expected_types
                  if k.endswith("_min_size") or k.endswith("_max_size"))
    assert live == [
        "cell_max_size", "cell_min_size", "cytoplasm_min_size",
        "nucleus_max_size", "nucleus_min_size",
        "pathogen_max_size", "pathogen_min_size",
    ], live
    assert not [k for k in live if k.startswith("organelle")], (
        "organelle's _size pair was retired by b7ae412af; a live one means "
        "the retirement was undone and the rename rule now eats it")


def test_the_doctor_speaks_for_every_key_the_run_migrates():
    """IMPOSSIBLE: a value silently moved, or a warning with no move.

    The two must agree in BOTH directions. `organelleq_min_size` used to be
    migrated by the Qt panel, ignored by the run and unmentioned by the
    doctor -- three answers to one question.
    """
    from spacr.validate import _check_retired_keys

    disagreed = []
    corpus = list(RETIRED_SETTINGS)
    corpus += [f"{role}_{suffix}" for role in SAMPLE_ROLES
               for suffix in RENAMED_SETTING_SUFFIXES]
    for key in corpus:
        if key in SEMANTIC_FOLDS:
            continue
        migrated = bool(surviving_setting_name(key))
        spoke = bool(_check_retired_keys({key: PROBE}))
        withdrawn = RETIRED_SETTINGS.get(key) == ""
        if withdrawn:
            # A withdrawal is warned about and never moved, by design.
            assert spoke and not migrated, key
            continue
        if migrated != spoke:
            disagreed.append((key, {"run_migrates": migrated,
                                    "doctor_speaks": spoke}))
    assert disagreed == [], (
        f"the run and the doctor disagree about these keys: {disagreed}")


def test_a_withdrawn_setting_keeps_its_key():
    """IMPOSSIBLE: dropping a value the user set, even a withdrawn one.

    The instruction is explicit -- "do NOT silently drop an unrecognised key.
    A settings file that quietly loses a value the user set is worse than one
    that refuses to load." A withdrawn key has no consumer, but it keeps its
    place so the doctor can name it.
    """
    withdrawn = [k for k, v in RETIRED_SETTINGS.items() if not v]
    assert withdrawn, "expected withdrawals in the table"
    for key in withdrawn:
        settings = {key: PROBE}
        _fold_renamed_settings(settings)
        assert settings == {key: PROBE}, (
            f"{key} is withdrawn, not renamed; the fold must leave it")


def test_a_chain_of_renames_resolves_to_the_end(monkeypatch):
    """IMPOSSIBLE: a two-step chain stopping after one step.

    391 creates exactly this -- `<role>_min_object_area` becomes
    `_min_split_area` becomes `_minimum_area_to_split` -- so the oldest files
    need two hops and a resolver that takes one leaves the value on a key
    nothing reads.
    """
    monkeypatch.setitem(RENAMED_SETTING_SUFFIXES, "aa", "bb")
    monkeypatch.setitem(RENAMED_SETTING_SUFFIXES, "bb", "min_area")
    assert surviving_setting_name("organelle_aa") == ("organelle_min_area",), (
        "the chain stopped early; the intermediate name is not a live "
        "setting and a value left there is a value lost")


def test_a_cycle_leaves_the_key_alone_instead_of_hanging(monkeypatch):
    """IMPOSSIBLE: a malformed table hanging the run.

    A cycle must not loop, and must not guess: the key keeps its own name so
    the unknown-key check reports it.
    """
    monkeypatch.setitem(RENAMED_SETTING_SUFFIXES, "xx", "yy")
    monkeypatch.setitem(RENAMED_SETTING_SUFFIXES, "yy", "xx")
    assert surviving_setting_name("organelle_xx") == ()
    settings = {"organelle_xx": PROBE}
    _fold_renamed_settings(settings)
    assert settings == {"organelle_xx": PROBE}


def test_two_old_spellings_for_one_name_do_not_depend_on_column_order():
    """IMPOSSIBLE: the winner decided by which column the CSV listed first.

    `min_cell_count` and `minimum_cell_count` both become
    `min_cells_per_well`. Before this, adding the second made the winner
    whichever the dict happened to yield first.
    """
    first = {"min_cell_count": 50, "minimum_cell_count": 30}
    second = {"minimum_cell_count": 30, "min_cell_count": 50}
    _fold_renamed_settings(first)
    _fold_renamed_settings(second)
    assert first == second, (
        f"insertion order changed the answer: {first} vs {second}")


def test_the_mask_factory_keeps_the_values_an_old_file_set():
    """IMPOSSIBLE: the regression that started all of this.

    Asserted on the real factory rather than the helper, because the helper
    was never the problem -- the factory not calling it was.
    """
    from spacr.settings import set_default_settings_preprocess_generate_masks

    defaults = set_default_settings_preprocess_generate_masks({"src": "path"})
    assert defaults["cell_flow_threshold"] != 0.42, (
        "pick a probe unlike the default or this test proves nothing")

    old_file = set_default_settings_preprocess_generate_masks({
        "src": "path",
        "cell_FT": 0.42,
        "cell_CP_prob": -0.7,
        "cell_min_object_area": 123,
        "nucleus_Signal_to_noise": 7.5,
        "organellez_min_size": 44,
    })
    assert old_file["cell_flow_threshold"] == 0.42
    assert old_file["cell_cellprob_threshold"] == -0.7
    assert old_file["nucleus_signal_to_noise"] == 7.5
    assert old_file["organellez_min_area"] == 44
    # THE TWO-HOP CHAIN, and it is the reason this assertion changed once
    # already. `cell_min_object_area` became `cell_min_split_area` in
    # b7ae412af and `cell_minimum_area_to_split` in 391, so the oldest files
    # need BOTH hops. This asserted the INTERMEDIATE name until 391 landed and
    # failed the moment it did -- which is the guard working: a resolver that
    # stopped after one step would leave the value on a key nothing reads.
    assert old_file["cell_minimum_area_to_split"] == 123
    assert "cell_min_split_area" not in old_file, (
        "the value stopped on the intermediate name, which nothing reads")
    for dead in ("cell_FT", "cell_CP_prob", "cell_min_object_area",
                 "nucleus_Signal_to_noise", "organellez_min_size"):
        assert dead not in old_file, f"{dead} was left behind"


def test_the_run_says_what_it_migrated(caplog):
    """IMPOSSIBLE: a file's behaviour changing with nothing said.

    This is a behaviour change on historical data: a file that read
    `cell_FT=0.42` as 100 yesterday reads it as 0.42 today, and the masks
    change. That is the fix working, and it is also exactly what makes
    somebody think their data changed.
    """
    import logging

    with caplog.at_level(logging.INFO, logger="spacr.settings"):
        _fold_renamed_settings({"cell_FT": 0.42})
    said = " ".join(record.getMessage() for record in caplog.records)
    assert "cell_FT" in said and "cell_flow_threshold" in said, said
    assert "0.42" in said, "the value has to be in the line, not just the names"


def test_a_current_settings_file_says_nothing(caplog):
    """A file with no old keys must not print migration noise."""
    import logging

    with caplog.at_level(logging.INFO, logger="spacr.settings"):
        _fold_renamed_settings({"cell_flow_threshold": 0.42,
                                "cell_min_size": 9, "src": "path"})
    assert caplog.records == [], (
        f"a current file printed migration lines: "
        f"{[r.getMessage() for r in caplog.records]}")


def test_the_qt_load_agrees_with_the_run():
    """IMPOSSIBLE: the panel and the run disagreeing about a key.

    The panel had its own regex and its own answer. It now delegates, and
    these are the two cases where its old answer was actually wrong: it
    stored a SPLIT under a tuple key, and it folded `gradient_accumulation`
    onto the step count where `int()` makes it zero.
    """
    from spacr.qt.screens.app_screen import _translate_legacy_setting_keys

    split = _translate_legacy_setting_keys({"control_wells": ["c12"]})
    assert split == {"stain_baseline_wells": ["c12"],
                     "analysis_excluded_wells": ["c12"]}, split
    assert not any(isinstance(k, tuple) for k in split), (
        "a split was stored under a tuple key, which no widget reads")

    gradient = _translate_legacy_setting_keys({"gradient_accumulation": False})
    assert gradient == {"gradient_accumulation": False}, gradient
    assert "gradient_accumulation_steps" not in gradient, (
        "int(False) is 0, and a step count of zero is not an off state")

    for old, new in (("cell_FT", "cell_flow_threshold"),
                     ("organelleq_min_size", "organelleq_min_area")):
        assert _translate_legacy_setting_keys({old: PROBE}) == {new: PROBE}


def test_split_role_setting_never_matches_a_bare_key():
    """IMPOSSIBLE: a suffix rule retiring a standalone setting.

    `FT`, `CP_prob`, `Signal_to_noise` and `flow_threshold` are all live
    settings in their own right in the apply/test-model submodules. A key
    with no role cannot match a role-family rule.
    """
    for bare in RENAMED_SETTING_SUFFIXES:
        assert split_role_setting(bare) is None, bare
        if bare in expected_types:
            settings = {bare: PROBE}
            _fold_renamed_settings(settings)
            assert settings == {bare: PROBE}, (
                f"{bare} is a live standalone setting and was migrated")


def test_a_settings_change_has_not_stranded_reviewed_translations():
    """IMPOSSIBLE: landing a settings change that kills the catalog build.

    THIS TEST EXISTS BECAUSE THE GUARD WAS ALREADY THERE AND I DID NOT RUN IT.
    `tests/test_the_reviewed_runtime_reporter_tells_moved_from_stale.py`
    asserts the same thing, and instruction 391's settings batch was pushed
    without it in the selected set -- so 32 reviewed records were stranded,
    every phase of a forty-minute catalog rebuild exited 1 on the first
    language, and another session paid for it.

    THE FAILURE IS NOT IN THE CATALOGS, WHICH IS WHY IT IS EASY TO MISS.
    Reviewed runtime evidence is validated against the LIVE setting surface
    and the API builder loads it before it starts, so removing a setting or
    rewriting a tooltip breaks the LOADER -- and nothing in the settings
    suite notices.

    It lives HERE, beside the rename and withdrawal guards, because this is
    the file anyone changing a setting already runs. A guard in a file that
    has to be remembered is a guard that will be forgotten again.

    Costs about twenty seconds. A dead rebuild costs forty minutes and
    somebody else's afternoon.
    """
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    tool = root / "tools" / "check_reviewed_runtime_evidence.py"
    if not tool.exists():                      # pragma: no cover
        pytest.skip("the reviewed-runtime reporter is not in this tree")

    done = subprocess.run([sys.executable, str(tool)], cwd=str(root),
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, (
        "a settings change in this tree has stranded reviewed translations, "
        "and the catalog build will fail on the first language rather than "
        "on the catalogs. Repair them before pushing -- the report below says "
        "which are re-pointable and which must be dropped:\n\n"
        + done.stdout[-4000:])
