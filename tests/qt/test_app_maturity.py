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
    """The table is inert unless something writes it into the registry."""
    stages = {key: "alpha" for key in maturity.PROMOTIONS}
    changed = maturity.apply(stages)
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
    assert maturity.apply(stages) == []
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
    assert maturity.reason_for("not-a-module") == ""


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
