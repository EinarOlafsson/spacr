"""The maturity table applied without a registry, and with a retirement in it.

``tests/qt/test_app_maturity.py`` holds the assessment itself to account --
every promotion carries evidence, no app inherits ``stable`` from an absent
line. What is driven here is what happens when the registry this module
writes into is not there at all, and what a RETIREMENT does: the table ships
empty, so the loop that acts on one has never run.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt import maturity


def test_an_unimportable_registry_is_no_keys_rather_than_a_crash(monkeypatch):
    """This module is imported by the launch sequence and by tests alike."""
    monkeypatch.setitem(sys.modules, "spacr.qt.app", None)
    assert maturity._registered_keys() == ()
    assert maturity.unassessed_apps() == []


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
