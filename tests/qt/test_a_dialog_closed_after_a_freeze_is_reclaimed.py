"""A window that was alive at a freeze and closed afterwards is still collected.

``spacr.qt.app._freeze_what_survived`` and the background importer call
``gc.freeze()`` after a launch, a screen build or an import. A freeze moves
EVERY tracked object into the permanent generation, including a dialog that
happens to be open, and the collector never looks at the permanent generation
again. Before 2026-10-31 such a dialog, closed and deleted after the freeze,
kept its Python wrapper and every object in its reference cycles for the rest
of the session: with the main window and six module screens open, all fifteen
Preferences dialogs opened and closed between two screen builds stayed alive.

A freeze is now made on the GUI-thread collector's next tick, after that tick
has collected what is not frozen yet, so a dialog closed just before it is
collected rather than frozen. The tick also remembers which windows were alive
at the freeze; once one of them is deleted, a later tick gives the permanent
generation back to the collector, collects it and freezes what survives.
"""
from __future__ import annotations

import gc
import weakref

import pytest

pytestmark = pytest.mark.qt


class _Owner:
    """A plain object in a reference cycle with the dialog that holds it."""

    def __init__(self, dialog):
        self.dialog = dialog


def _cyclic_dialog():
    from PySide6.QtWidgets import QDialog, QLabel

    dialog = QDialog()
    QLabel("frozen while open", dialog)
    dialog.owner = _Owner(dialog)
    dialog.show()
    return dialog


@pytest.fixture(autouse=True)
def _no_other_work_defers_the_tick(monkeypatch):
    """Keep another test's background import or busy worker from deferring these ticks."""
    from spacr.qt import gc_policy

    monkeypatch.setattr(gc_policy, "_a_heavy_import_is_running", lambda: False)
    monkeypatch.setattr(gc_policy, "_defer_for_busy_workers", lambda: False)


def _delete(dialog, qapp):
    from PySide6.QtCore import QEvent

    dialog.close()
    dialog.deleteLater()
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)


def test_a_dialog_closed_after_a_later_freeze_is_reclaimed(qapp):
    """A screen build freezes while a dialog is open; closing it frees its cycle."""
    from spacr.qt import gc_policy
    from spacr.qt.app import _freeze_what_survived

    dialog = _cyclic_dialog()
    owner = weakref.ref(dialog.owner)
    try:
        _freeze_what_survived()
        assert gc_policy.collect_once() == 2
        assert gc.get_freeze_count() > 0
        assert owner() is not None
        _delete(dialog, qapp)
        del dialog
        assert owner() is not None
        assert gc_policy.collect_once() == 2
        assert owner() is None
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()


def test_a_worker_thread_freeze_is_reclaimed_on_the_gui_thread(qapp):
    """A background import freezes off the GUI thread; the reclaim stays on it."""
    import threading

    from spacr.qt import gc_policy

    gc.unfreeze()
    dialog = _cyclic_dialog()
    owner = weakref.ref(dialog.owner)
    try:
        worker = threading.Thread(target=gc_policy._freeze_survivors)
        worker.start()
        worker.join()
        assert gc.get_freeze_count() == 0
        assert gc_policy.collect_once() == 2
        assert gc.get_freeze_count() > 0
        _delete(dialog, qapp)
        del dialog
        assert owner() is not None
        assert gc_policy.collect_once() == 2
        assert owner() is None
        assert not gc_policy._a_frozen_window_was_deleted()
    finally:
        gc.unfreeze()


def test_a_freeze_with_no_window_deleted_since_costs_no_full_sweep(qapp, monkeypatch):
    """Hiding a frozen window, or freezing again, is not a reason to sweep everything."""
    from spacr.qt import gc_policy
    from spacr.qt.app import _freeze_what_survived

    dialog = _cyclic_dialog()
    try:
        _freeze_what_survived()
        gc_policy.collect_once()
        dialog.hide()
        _freeze_what_survived()
        sweeps = []
        monkeypatch.setattr(gc_policy, "_reclaim_frozen", lambda: sweeps.append(1))
        gc_policy.collect_once()
        assert sweeps == []
    finally:
        _delete(dialog, qapp)
        gc.unfreeze()


def test_a_dialog_closed_just_before_a_freeze_is_collected_not_frozen(qapp):
    """Its cycle is garbage that no tick has collected yet when the next screen is built."""
    from spacr.qt import gc_policy
    from spacr.qt.app import _freeze_what_survived

    dialog = _cyclic_dialog()
    owner = weakref.ref(dialog.owner)
    try:
        _delete(dialog, qapp)
        del dialog
        assert owner() is not None
        _freeze_what_survived()
        assert gc_policy.collect_once() == 2
        assert owner() is None
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()
