"""The motility preview's guards: stale results, empty plates and dead threads.

The preview reads merged arrays off the GUI thread, and every path below is
one where the answer arriving is not the answer that was asked for -- a scan
from a plate the user has already navigated away from, a plate with no time
series in it, a worker whose C++ half has gone while its result was in
flight. Each has to leave the panel usable and say what happened.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import motility_preview as module
from spacr.qt.widgets.motility_preview import (
    MotilityPreviewPanel, MotilityRequest, _MotilityWorker, run_motility_pass)

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    widget = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


# ---------------------------------------------------------------------------
# The worker-thread body
# ---------------------------------------------------------------------------

def test_a_pass_forwards_every_field_of_its_request(monkeypatch):
    """The request is the whole argument list, in the order the reader takes it.

    Two of the fields are plane indices into the merged array; swapping them
    would track the pathogen plane and report it as the cell one, which is a
    preview that looks entirely normal and describes the wrong object.
    """
    seen = {}

    def _record(merged_dir, metas, n_channels, tracked, pathogen,
                max_frames=12):
        seen.update(merged_dir=merged_dir, metas=metas, n_channels=n_channels,
                    tracked=tracked, pathogen=pathogen, max_frames=max_frames)
        return "the point table"

    monkeypatch.setattr(module, "build_point_table", _record)
    request = MotilityRequest(merged_dir="/plate/merged", metas=[{"t": 0}],
                              n_channels=3, tracked_plane=4,
                              pathogen_plane=5, max_frames=7)

    assert run_motility_pass(request) == "the point table"
    assert seen == {"merged_dir": "/plate/merged", "metas": [{"t": 0}],
                    "n_channels": 3, "tracked": 4, "pathogen": 5,
                    "max_frames": 7}


def test_the_worker_body_delivers_the_table_and_an_empty_error(monkeypatch):
    """What runs off the GUI thread emits ``(table, "")`` on success.

    The panel branches on the error string, so a body that emitted the table
    with a non-empty error would throw away a result it had already paid for.
    """
    monkeypatch.setattr(module, "run_motility_pass",
                        lambda request: "the point table")
    worker = _MotilityWorker(MotilityRequest())
    delivered = []
    worker.finished_result.connect(lambda table, error: delivered.append(
        (table, error)))

    worker.run()

    assert delivered == [("the point table", "")]


def test_the_worker_body_delivers_the_reason_it_failed(monkeypatch):
    """A failed read emits ``(None, reason)`` rather than raising into Qt.

    An exception escaping a QThread's run body is not caught anywhere; the
    panel would sit on "Scanning…" for ever with nothing to show.
    """
    def _explode(request):
        raise MemoryError("the merged array will not fit")

    monkeypatch.setattr(module, "run_motility_pass", _explode)
    worker = _MotilityWorker(MotilityRequest())
    delivered = []
    worker.finished_result.connect(lambda table, error: delivered.append(
        (table, error)))

    worker.run()

    assert delivered[0][0] is None
    assert "the merged array will not fit" in delivered[0][1]


# ---------------------------------------------------------------------------
# Opening a plate
# ---------------------------------------------------------------------------

def test_an_empty_path_starts_no_scan(panel):
    """An empty drop or a cancelled dialog is not a plate to open.

    Submitting the job anyway would put "Scanning …" on screen and then fail
    against the current working directory.
    """
    assert panel.load_folder_async("") is False
    assert panel.load_folder_async(None) is False


def test_a_scan_from_a_plate_the_user_left_is_discarded(panel):
    """Only the newest request may change the panel.

    Two plates opened quickly finish in whatever order the disk allows;
    installing the older answer would show the previous plate's wells under
    the new plate's name.
    """
    panel._load_token = 5
    before = panel._status.text()

    panel._on_plate_scanned(4, {"merged": "/somewhere", "groups": {"g": [1, 2]}})

    assert panel._status.text() == before


def test_a_plate_that_could_not_be_read_says_why(panel):
    """The scan's own message reaches the status line unchanged.

    It names the folder and the reason; replacing it with "load failed"
    would drop the only actionable half.
    """
    panel._on_plate_scanned(panel._load_token,
                            {"error": "Load failed: no merged folder"})

    assert panel._status.text() == "Load failed: no merged folder"


def test_a_plate_with_no_time_series_says_what_a_preview_needs(panel):
    """A plate of single time points cannot show motility at all.

    "No groups" is not a failure of the plate; it is a statement about what
    this preview measures, and the message has to say so or the user will
    look for a broken folder.
    """
    panel._on_plate_scanned(panel._load_token,
                            {"merged": "/plate/merged", "groups": {}})

    assert "two or more time points" in panel._status.text()


# ---------------------------------------------------------------------------
# The sample cap
# ---------------------------------------------------------------------------

def test_a_cap_that_changes_nothing_redraws_nothing(panel, monkeypatch):
    """Re-sampling costs a full re-read, so an unchanged cap must not trigger it.

    The sampler answers whether the cap actually moved; ignoring that answer
    would re-read the merged arrays on every spinbox repeat.
    """
    monkeypatch.setattr(panel._sampler, "set_max", lambda value: False)
    calls = []
    monkeypatch.setattr(panel, "_populate_group_box",
                        lambda: calls.append("repopulated"))

    panel._on_max_sets_changed(9)

    assert calls == []


# ---------------------------------------------------------------------------
# Threads that are already gone
# ---------------------------------------------------------------------------

class _DeletedWorker:
    """A worker whose C++ half PySide6 has already taken away."""

    def __init__(self, log=None):
        self.log = log if log is not None else []

    def wait(self, *args):
        self.log.append("wait")
        raise RuntimeError("Internal C++ object already deleted.")

    def setParent(self, _parent):
        self.log.append("setParent")
        raise RuntimeError("Internal C++ object already deleted.")


def test_retiring_a_worker_that_is_already_gone_is_not_an_error(panel):
    """A deleted QThread has certainly finished, so there is nothing to wait for.

    Letting the RuntimeError escape would make the panel impossible to reuse
    after its worker was collected.
    """
    panel._retired_worker = _DeletedWorker()

    panel._release_worker()

    assert panel._retired_worker is None


def test_closing_over_a_deleted_worker_still_closes(panel, qtbot):
    """The close path waits on both workers and must survive either being gone.

    A close handler that raises leaves the window on screen with its panel
    already shut down -- a card that looks alive and answers nothing.
    """
    from PySide6.QtGui import QCloseEvent

    attempts = []
    panel._worker = _DeletedWorker(attempts)
    panel._retired_worker = _DeletedWorker(attempts)

    panel.closeEvent(QCloseEvent())

    assert attempts.count("wait") >= 2, "both workers were waited on"


# ---------------------------------------------------------------------------
# Keeping the plane spinner and the plane dropdown in step
# ---------------------------------------------------------------------------

def test_a_dropdown_already_showing_the_spinners_plane_changes_nothing(panel):
    """The two controls name one plane, so agreeing is a no-op.

    Writing the value back regardless would bounce a signal between them on
    every change.
    """
    before = int(panel._tracked_plane.value())
    panel._channel_box.clear()
    panel._channel_box.addItem(f"Ch {before}", before)
    panel._channel_box.setCurrentIndex(0)

    panel._sync_plane_spin_from_combo()

    assert int(panel._tracked_plane.value()) == before


def test_the_plane_controls_do_nothing_before_they_exist(panel):
    """Both directions are wired during construction and can fire early.

    A signal that arrives before the other control has been built must be
    dropped, not answered against a half-built panel.
    """
    box = panel._channel_box
    spin = panel._tracked_plane
    panel._points = "the cached point table"
    try:
        del panel._channel_box
        panel._sync_plane_combo_from_spin()
        del panel._tracked_plane
        panel._on_display_channel_changed()
    finally:
        panel._channel_box = box
        panel._tracked_plane = spin

    assert panel._points == "the cached point table", (
        "an early signal did not throw the cache away")
