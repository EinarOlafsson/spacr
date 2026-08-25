"""The maturity table applied without a registry, and with a retirement in it.

``tests/qt/test_app_maturity.py`` holds the assessment itself to account --
every promotion carries evidence, no app inherits ``stable`` from an absent
line. What is driven here is what happens when the registry this module
writes into is not there at all, and what a RETIREMENT does: the table ships
empty, so the loop that acts on one has never run.
"""
from __future__ import annotations

import logging
import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt import maturity


def test_an_unimportable_registry_is_no_keys_rather_than_a_crash(monkeypatch):
    """This module is imported by the launch sequence and by tests alike."""
    monkeypatch.setitem(sys.modules, "spacr.qt.app", None)
    assert maturity._registered_keys() == ()
    assert maturity.unassessed_apps() == []


def test_apply_without_a_stage_table_changes_nothing(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "spacr.qt.app", None)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.maturity"):
        assert maturity.apply() == []
    assert "not importable" in caplog.text


def test_a_retired_app_is_unregistered_and_named(monkeypatch):
    """The loop that retires an app; the shipped table is deliberately empty."""
    from spacr.qt import app as app_mod

    retired = []
    monkeypatch.setattr(maturity, "RETIREMENTS", {"gone": "superseded"})
    monkeypatch.setattr(app_mod, "unregister_app", retired.append)

    stages = {}
    maturity.apply(stages, keys=())
    assert retired == ["gone"]
    assert maturity.reason_for("gone") == "superseded"
    assert "gone" in maturity.assessed_keys()


def test_an_app_that_will_not_retire_does_not_stop_the_rest(monkeypatch,
                                                            caplog):
    from spacr.qt import app as app_mod

    monkeypatch.setattr(maturity, "RETIREMENTS", {"gone": "superseded"})
    monkeypatch.setattr(app_mod, "unregister_app",
                        lambda key: (_ for _ in ()).throw(
                            KeyError("never registered")))
    stages = {}
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.maturity"):
        changed = maturity.apply(stages, keys=("brand_new",))
    assert "could not retire" in caplog.text
    assert changed == ["brand_new"], \
        "a failed retirement swallowed the unassessed default"
    assert stages["brand_new"] == maturity.UNASSESSED_STAGE


def test_a_stage_somebody_else_set_is_never_demoted():
    """The table is one assessment; a later one must not be undone by it."""
    promoted = next(iter(maturity.PROMOTIONS))
    stages = {promoted: "stable"}
    assert maturity.apply(stages, keys=()) == []
    assert stages[promoted] == "stable"


def test_an_empty_string_is_not_a_stage_anybody_stated():
    stages = {"half_written": ""}
    assert maturity.apply(stages, keys=("half_written",)) == ["half_written"]
    assert stages["half_written"] == maturity.UNASSESSED_STAGE
