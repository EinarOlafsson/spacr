"""Paths that earlier tests reached only when the timing happened to fall right.

The coverage ratchet failed on CI run 41790895465 with the code unchanged
from the run before it: ``collect_once`` reclaiming a deleted frozen window
on an explicit request, a warming slice declining for a hidden window, and
the sheet check without a recorded signature were each covered in one run
and not the next. Each is driven directly here.
"""
from __future__ import annotations

import threading

from PySide6.QtWidgets import QWidget

from spacr.qt import gc_policy, preferences


def test_a_requested_collection_reclaims_a_deleted_frozen_window_first(qapp, monkeypatch):
    requested = threading.Event()
    requested.set()
    reclaimed, collected = [], []
    monkeypatch.setattr(gc_policy, "_requested", requested)
    monkeypatch.setattr(gc_policy, "_INPUT",
                        {"where": None, "moved_at": float("-inf"), "retry_armed": False})
    monkeypatch.setattr(gc_policy, "_a_frozen_window_was_deleted", lambda: True)
    monkeypatch.setattr(gc_policy, "_reclaim_frozen", lambda: reclaimed.append(True))
    monkeypatch.setattr(gc_policy, "_collect_and_freeze",
                        lambda *a, **k: collected.append("freeze"))
    monkeypatch.setattr(gc_policy.gc, "collect", lambda *a: collected.append("collect"))

    assert gc_policy.collect_once() == 2
    assert reclaimed == [True] and collected == []
    assert not requested.is_set()


def test_a_hidden_window_counts_as_busy_for_a_warming_slice(qapp):
    window = QWidget()
    try:
        assert not window.isVisible()
        assert preferences._the_user_is_busy(window)
    finally:
        window.deleteLater()


def test_without_a_recorded_sheet_signature_the_sheet_is_not_behind(qapp, monkeypatch):
    monkeypatch.setattr(qapp, "_spacr_preferences_style_signature", None, raising=False)

    assert preferences._the_application_sheet_is_behind() is False
