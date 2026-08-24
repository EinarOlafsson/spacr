"""What Classify (CV) and Classify (ML) carried, the merged screen carries.

Merging two registry rows into one moved the settings across -- the
merged screen's defaults are a strict superset of either half's -- but
four behaviours hang off the LITERAL KEY of the screen they replaced, and
none of them followed:

* the cross-validated hyperparameter search, whose panel is built from a
  table keyed on the app; the merged screen had none, and there is no
  other door to that search;
* the drop policy, without which a bundle of plates falls through to the
  generic handler that replaces `src` with ONE path -- and the merged
  screen's `src` is a list, so four plates dropped together became one;
* the pre-flight refusals, which is how "this folder has no
  measurements.db" and "scoring needs a model_path" are said before a run
  starts rather than during it;
* the chaining ports, which is how a module is IN the pipeline graph at
  all. Without them the Core chain read measure -> (nothing) -> regression.

Each is asserted against the merged key rather than against a count, so
this fails if a future merge drops one the same way.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

MERGED = "classify_merged"


def test_the_merged_screen_is_the_one_the_registry_offers(qapp):
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.qt.app import APPS

    keys = {row[0] for row in APPS}
    assert MERGED in keys
    # And the two it replaced are gone from the GUI registry.
    assert "classify" not in keys
    assert "ml_analyze" not in keys


def test_it_is_in_the_pipeline_graph(qapp):
    """`chained_app_keys` is the registry intersected with the modules
    that declare ports, so a screen with no ports is not chainable."""
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.qt import chaining

    chained = set(chaining.chained_app_keys())
    assert MERGED in chained, "Classify is not in the chain at all"
    # The pipeline it sits in the middle of.
    assert {"measure", "regression"} <= chained


def test_its_ports_are_the_ones_the_classifier_had(qapp):
    from spacr import ports

    declared = ports.ports_for(MERGED) if hasattr(ports, "ports_for") \
        else ports.PORTS[MERGED]
    consumes = {port.role for port in declared.consumes}
    produces = {port.role for port in declared.produces}

    assert "db" in consumes, "it no longer reads the measurements database"
    assert {"model", "scores"} <= produces


def test_dropping_plates_on_it_uses_the_classifier_policy(qapp):
    """The generic handler would collapse a multi-plate drop to one path."""
    from spacr.qt import dnd_handlers

    table = getattr(dnd_handlers, "DROP_HANDLERS", None)
    if table is None:                       # the table has been renamed
        table = next(value for name, value in vars(dnd_handlers).items()
                     if isinstance(value, dict) and "mask" in value)
    assert table.get(MERGED) is table.get("classify")
    assert table.get(MERGED) is not None


def test_the_preflight_refusals_still_fire(qapp):
    from spacr import validate

    assert MERGED in validate.DB_APPS

    problems = validate.validate_settings(
        {"src": "/definitely/not/a/plate/folder",
         "apply_model_to_dataset": True, "train": False}, MERGED)
    said = " ".join(str(getattr(p, "message", p)) for p in problems)
    assert "model" in said.lower(), (
        "scoring a dataset with no model_path is no longer refused")


def test_the_hyperparameter_search_has_a_panel_again(qapp):
    """There is no other door to the cross-validated search."""
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.hyperparam import APP_CRITERIA, DEFAULT_SPACES
    from spacr.qt.screens.hyperparam import APP_PARAMS

    assert MERGED in APP_PARAMS and APP_PARAMS[MERGED]
    assert MERGED in APP_CRITERIA and APP_CRITERIA[MERGED]
    assert MERGED in DEFAULT_SPACES

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(MERGED)
    assert screen._hyperparam is not None, "the search panel is missing"
