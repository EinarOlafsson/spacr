"""The alpha shelf, after the modules on it were actually assessed.

Twenty-six of forty-three modules shipped labelled alpha. A label most of an
application wears predicts nothing, and a user who sees it on Database
Browser, Report and Plaque Assay alike learns to ignore it — including on the
one module where it would have been worth heeding. What is pinned here is
that the label means something again, and that the assessment behind each
change is written down next to it.
"""
from __future__ import annotations

import pytest

from spacr.qt import maturity


def test_the_assessment_covers_every_module_it_claims_to():
    """A promotion for a module that does not exist is a stale opinion.

    Registration is forced first: `feature_dict` joins the registry from
    `spacr.qt.__init__` at ``run()`` time rather than from ``app.py``, so
    importing the registry alone does not see it.
    """
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.qt.app import APPS
    known = {row[0] for row in APPS}
    missing = sorted(set(maturity.PROMOTIONS) - known)
    assert not missing, f"promotions for modules that are not registered: {missing}"


def test_every_promotion_is_to_a_real_stage():
    from spacr.qt.app import STAGES
    for app_key, (stage, _reason) in maturity.PROMOTIONS.items():
        assert stage in STAGES, f"{app_key} is promoted to {stage!r}"
        assert stage != "alpha", (
            f"{app_key} is in the promotion table but stays alpha; leave it "
            "out instead")


def test_every_decision_names_its_evidence():
    """"Well tested" is not a reason. "769 assertions across four files, a
    spacr-run module and lesson 32" is one, because the next person to
    disagree can go and look."""
    import re
    for app_key, (_stage, reason) in maturity.PROMOTIONS.items():
        assert len(reason.split()) >= 12, f"{app_key}: {reason!r} is too thin"
        assert re.search(r"\d", reason), (
            f"{app_key}: {reason!r} cites no countable evidence")


def test_the_promotions_are_applied_at_launch():
    """The table is inert unless something writes it into the registry.

    ``keys=()`` runs phase 1 alone. Phase 2 writes the alpha default into
    every *registered* app nobody assessed, and the table handed in here
    holds only the promoted keys — without it the answer would be dominated
    by twenty-six unrelated modules having their default filled in.
    """
    stages = {key: "alpha" for key in maturity.PROMOTIONS}
    changed = maturity.apply(stages, keys=())
    assert set(changed) == set(maturity.PROMOTIONS)
    for app_key, (stage, _reason) in maturity.PROMOTIONS.items():
        if stage == "stable":
            assert app_key not in stages
        else:
            assert stages[app_key] == stage


def test_signing_an_app_off_deletes_its_line_rather_than_rewriting_it():
    """``APP_STAGE`` records what is NOT signed off.

    "stable" is the absence of an entry, so promoting to stable has to
    remove the key. Writing the word in gives the table a second way to say
    the same thing, and the home suite fails on exactly that.
    """
    stable = [k for k, (s, _r) in maturity.PROMOTIONS.items() if s == "stable"]
    assert stable, "no module was promoted to stable at all"
    stages = {key: "alpha" for key in maturity.PROMOTIONS}
    maturity.apply(stages)
    assert not (set(stable) & set(stages))
    assert set(stages.values()) <= {"alpha", "beta"}


def test_applying_twice_changes_nothing_the_second_time():
    stages = {key: "alpha" for key in maturity.PROMOTIONS}
    maturity.apply(stages)
    assert maturity.apply(stages) == []


def test_a_module_promoted_further_elsewhere_is_not_demoted():
    """The table is a snapshot of one assessment; the next one should not be
    silently undone by re-importing this module."""
    stages = {"run_compare": "stable"}
    assert maturity.apply(stages, keys=()) == []
    assert stages["run_compare"] == "stable"


def test_no_assessed_module_is_still_alpha_after_launch():
    import spacr.qt
    from spacr.qt.app import app_stage
    spacr.qt.register_self_registering_modules()
    still_alpha = [key for key in maturity.PROMOTIONS
                   if app_stage(key) == "alpha"]
    assert not still_alpha, (
        f"assessed but still labelled alpha: {still_alpha}")


def test_the_shelf_is_no_longer_most_of_the_application():
    """Twenty-six of forty-three was the problem, and it was not that the
    modules were unfinished — none of the twenty-six screen modules holds a
    single NotImplementedError. It was that nobody had gone back to relabel
    finished work.

    A majority, because that is the claim: a label MOST of an application
    wears predicts nothing. A tighter bound would be a different claim, and
    a false one — new work is supposed to arrive at alpha, so the shelf
    refilling is health rather than rot. What stops it rotting is the test
    below, which says nothing already assessed may be sitting on it.
    """
    import spacr.qt
    from spacr.qt.app import APPS, app_stage
    spacr.qt.register_self_registering_modules()
    stages = [app_stage(row[0]) for row in APPS]
    alpha = stages.count("alpha")
    assert alpha < len(stages) / 2, (
        f"{alpha} of {len(stages)} modules are alpha; a label most of the "
        "application wears predicts nothing")


def test_anything_still_alpha_was_simply_not_assessed():
    """Alpha is allowed to mean "arrived after the last assessment". It is
    not allowed to mean "assessed and left there anyway"."""
    import spacr.qt
    from spacr.qt.app import APPS, app_stage
    spacr.qt.register_self_registering_modules()
    for key, name, _desc, _section in APPS:
        if app_stage(key) == "alpha":
            assert key not in maturity.PROMOTIONS, (
                f"{name} was assessed and is still alpha")


def test_a_reason_can_be_looked_up_for_any_decision():
    for app_key in maturity.PROMOTIONS:
        assert maturity.reason_for(app_key)
    for app_key in maturity.AFFIRMED:
        assert maturity.reason_for(app_key)
    assert maturity.reason_for("not-a-module") == ""


# ---------------------------------------------------------------------------
# The default an unassessed module gets
# ---------------------------------------------------------------------------
# `stable` is the ABSENCE of a line in APP_STAGE, which makes it the one
# label a module can acquire by saying nothing. Everything below is about
# that: what a module nobody has looked at reads as, and that it can never
# be the top one.

def test_no_module_reads_stable_without_somebody_having_said_so():
    """The bug, stated directly.

    Eight apps — the seven core-pipeline modules and Recruitment — were
    reading "stable" because ``APP_STAGE.get(key, STAGE_STABLE)`` says so
    for a key nobody ever wrote down, not because anybody signed them off.
    After ``apply()`` every app reading stable is one this file names.
    """
    import spacr.qt
    from spacr.qt.app import APPS, app_stage
    spacr.qt.register_self_registering_modules()
    silent = [key for key, *_rest in APPS
              if app_stage(key) == "stable"
              and key not in maturity.assessed_keys()]
    assert not silent, (
        f"these modules read as stable and nobody assessed them: {silent}")


def test_an_unassessed_module_never_reads_stable():
    """The same rule from the other side, over the live registry."""
    import spacr.qt
    from spacr.qt.app import app_stage
    spacr.qt.register_self_registering_modules()
    for key in maturity.unassessed_apps():
        assert app_stage(key) in ("alpha", "beta"), (
            f"{key} is unassessed and reads {app_stage(key)!r}")


def test_a_module_that_declares_nothing_lands_on_alpha():
    """An app registered without a ``stage=`` — the case the default is for.

    It is registered for real rather than faked into ``APP_STAGE``: the
    silent inheritance happens in ``register_app``, which simply writes no
    line when no stage is given, and a test that skipped it would be
    asserting about a dict rather than about the registry.
    """
    import spacr.qt
    from spacr.qt.app import (APP_STAGE, SECTION_DATA, app_stage,
                              register_app, unregister_app)
    spacr.qt.register_self_registering_modules()
    key = "maturity_default_probe"
    unregister_app(key)
    try:
        register_app(key, "Default Probe",
                     "An app whose author said nothing about how finished "
                     "it is", SECTION_DATA)
        # This is the inheritance being guarded against.
        assert key not in APP_STAGE
        assert app_stage(key) == "stable"

        changed = maturity.apply()
        assert key in changed
        assert APP_STAGE[key] == maturity.UNASSESSED_STAGE
        assert app_stage(key) == "alpha"
        # And it stays put: the default is written once, not re-decided.
        assert maturity.apply() == []
    finally:
        unregister_app(key)


def test_the_default_never_overrules_a_module_that_did_declare_one():
    """It fills a gap; it does not have an opinion.

    A module that registered itself as beta is unassessed *and* labelled,
    and those are different states — nine of the modules on the shelf are in
    exactly that position. Demoting them would make this function an
    assessment rather than a default.
    """
    stages = {"declared_beta": "beta", "declared_alpha": "alpha"}
    changed = maturity.apply(stages, keys=("declared_beta", "declared_alpha",
                                           "declared_nothing"))
    assert changed == ["declared_nothing"]
    assert stages == {"declared_beta": "beta", "declared_alpha": "alpha",
                      "declared_nothing": "alpha"}


def test_an_assessed_module_is_left_alone_by_the_default():
    """AFFIRMED is what keeps "signed off" apart from "never looked at"."""
    stages = {}
    maturity.apply(stages, keys=tuple(maturity.AFFIRMED))
    assert stages == {}, (
        "an affirmed module was defaulted back onto the alpha shelf")
    for app_key, (stage, _reason) in maturity.AFFIRMED.items():
        assert stage in ("stable", "beta"), (
            f"{app_key} is 'affirmed' at {stage!r}; that is a promotion or a "
            "no-op, not an affirmation")


def test_every_affirmation_names_its_evidence_too():
    """Same bar as a promotion: countable, and checkable by the next reader."""
    import re
    for app_key, (_stage, reason) in maturity.AFFIRMED.items():
        assert len(reason.split()) >= 12, f"{app_key}: {reason!r} is too thin"
        assert re.search(r"\d", reason), (
            f"{app_key}: {reason!r} cites no countable evidence")


def test_no_module_is_assessed_twice_over():
    assert not (set(maturity.PROMOTIONS) & set(maturity.AFFIRMED))
    assert not (set(maturity.RETIREMENTS) & set(maturity.AFFIRMED))


def test_the_affirmations_are_about_modules_that_exist():
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.qt.app import APPS
    known = {row[0] for row in APPS}
    missing = sorted(set(maturity.AFFIRMED) - known)
    assert not missing, f"affirmations for modules that do not exist: {missing}"


def test_the_default_survives_a_registry_it_cannot_import(monkeypatch):
    """A headless caller must not be the reason a stage table is mangled."""
    monkeypatch.setattr(maturity, "_registered_keys", lambda: ())
    stages = {}
    assert maturity.apply(stages) == []
    assert stages == {}


def test_nothing_was_retired_and_that_is_recorded_as_a_finding():
    """Zero retirements is the result, not an omission: every one of the
    twenty-six has an implementation, a dedicated test file and either a CLI
    module or a written reason it is GUI-only. The structure stays so that
    retiring something later is a one-line change with a reason attached."""
    assert maturity.RETIREMENTS == {}
    assert isinstance(maturity.RETIREMENTS, dict)


def test_registering_is_idempotent():
    assert maturity.register() is True
    assert maturity.register() is True


def test_the_module_is_in_the_launch_registration_list():
    """And last in it — it can only reassess modules already registered."""
    from spacr.qt import SELF_REGISTERING_MODULES
    assert SELF_REGISTERING_MODULES[-1] == "spacr.qt.maturity"


@pytest.mark.parametrize("app_key", sorted(maturity.PROMOTIONS))
def test_every_promoted_module_still_builds_its_screen(app_key, qtbot):
    """A promotion is a promise the module works. The cheapest check that it
    is not a promise about a screen that raises on construction."""
    from spacr.qt.app import MainWindow
    window = getattr(test_every_promoted_module_still_builds_its_screen,
                     "_window", None)
    if window is None:
        window = MainWindow()
        qtbot.addWidget(window)
        test_every_promoted_module_still_builds_its_screen._window = window
    window._on_nav_selected(app_key)
    assert window._screens.get(app_key) is not None
