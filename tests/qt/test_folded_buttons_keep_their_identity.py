"""A folded module's button must survive the loss of its registry row.

Folding a module ends with dropping its row from the app registry, and the
registry then answers that key exactly as it answers a typo: no name, no
sentence, and "stable" for the maturity. The host's ``FOLD_FALLBACK`` is the
only record left of what the tile said, and every field has to come out of
it -- a button that read the name from the fallback but the stage from the
registry lit up in the colour of finished code for a module assessed alpha.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app as app_module                          # noqa: E402
from spacr.qt.widgets.fold_strip import (                       # noqa: E402
    FOLD_HOST_MODULES, FoldButton, folded_fallback, folded_modules,
)

#: The keys that are buttons on some host's masthead rather than tiles.
FOLDED = ("timelapse", "motility", "illumination", "anndata_export",
          "explain_cv", "classifier_evaluation", "agreement",
          "image_scatter", "pca", "barcode_qc", "volcano_explorer")


def test_every_folded_key_has_a_record(qapp):
    """No folded module may be left with nothing but its key."""
    missing = [k for k in FOLDED if not folded_fallback(k)[0]]
    assert missing == [], f"folded with no fallback record: {missing}"


def test_the_rows_really_are_gone(qapp):
    """The premise: these keys are folded, so the registry cannot answer."""
    registered = {row[0] for row in app_module.APPS}
    still_there = [k for k in FOLDED if k in registered]
    assert still_there == [], f"not actually folded: {still_there}"


@pytest.mark.parametrize("key", FOLDED)
def test_the_button_wears_the_stage_the_record_kept(qapp, key):
    """An alpha module must not light up as stable once its row is gone."""
    kept_stage = folded_fallback(key)[2]
    assert kept_stage, f"{key} kept no stage"
    assert FoldButton(key).property("stage") == kept_stage


@pytest.mark.parametrize("key", FOLDED)
def test_the_button_is_not_the_key_title_cased(qapp, key):
    """"Explain Cv" is what the key gives; "Explain CV Model" is the name."""
    name = folded_fallback(key)[0]
    button = FoldButton(key)
    first_line = button.toolTip().splitlines()[0]
    assert first_line == name
    assert button.accessibleName() == name


@pytest.mark.parametrize("key", FOLDED)
def test_the_sentence_the_tile_carried_is_still_there(qapp, key):
    """The tooltip is name then description, not the name alone."""
    description = folded_fallback(key)[1]
    assert description, f"{key} kept no description"
    assert description in FoldButton(key).toolTip()


def test_a_registered_key_still_answers_from_the_registry(qapp):
    """The fallback is for folded keys only; a live row must win."""
    row = next(r for r in app_module.APPS if r[1] and r[2])
    button = FoldButton(row[0])
    assert button.toolTip().splitlines()[0] == row[1]
    assert button.property("stage") == app_module.app_stage(row[0])


def test_an_unknown_key_is_still_drawn_rather_than_crashing(qapp):
    """A typo must degrade to a stable-looking button, not an exception."""
    button = FoldButton("no_such_module_anywhere")
    assert button.property("stage") == "stable"
    assert "No Such Module Anywhere" in button.toolTip()


def test_every_host_that_folds_something_is_walked(qapp):
    """A host whose table is not reachable contributes nothing, silently.

    The failure mode this guards is a new fold host defining FOLD_FALLBACK
    and never being added to FOLD_HOST_MODULES: nothing raises, the buttons
    just quietly go back to being title-cased keys.
    """
    reachable = folded_modules()
    for key in FOLDED:
        assert key in reachable, f"{key} is not reachable from any host"


def test_the_inventory_reports_the_host_that_draws_each_button(qapp):
    """Shared fallback copy must not be mistaken for fold ownership."""
    from importlib import import_module

    expected = {}
    duplicates = {}
    for module_name in FOLD_HOST_MODULES:
        module = import_module(module_name)
        members = getattr(module, "FOLDED_APPS", None)
        if members is None:
            members = getattr(module, "FOLD_ORDER", ())
        for key in members or ():
            if key in expected:
                duplicates.setdefault(key, [expected[key]]).append(module_name)
            else:
                expected[key] = module_name

    assert not duplicates, f"folded modules assigned to two hosts: {duplicates}"
    actual = {key: entry[3] for key, entry in folded_modules().items()}
    assert actual == expected
