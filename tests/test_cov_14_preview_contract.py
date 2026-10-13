"""The live-preview contract never fails while explaining itself.

Everything here runs on the GUI thread while a panel is repainting, so each
question has an answer even when the panel behind it is half-built or half
torn down:

* a panel that has not said why it is blocked is not blocked;
* a panel whose blocked-reason check raises does not take the status line
  down with it -- the run guard falls back to "not blocked" and the button
  press is what reports the real failure;
* a worker whose C++ object has already been freed is not running, so the
  panel returns to idle instead of refusing every further pass;
* a result carrying no token, or a token that is not a number, is not stale
  -- discarding it would drop the only answer a pass produced.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _panel():
    from spacr.qt.widgets.preview_contract import LivePreviewContract

    class _Bare(LivePreviewContract):
        pass

    return _Bare()


def test_a_panel_that_says_nothing_is_not_blocked():
    """The default hook returns no reason, so the run guard lets a pass go."""
    panel = _panel()

    assert panel._preview_blocked_reason() == ""
    assert panel.preview_blocked_reason() == ""
    assert panel.can_preview() is True


def test_a_panel_that_fails_while_explaining_is_not_blocked():
    """A raising hook falls back to "not blocked" rather than propagating.

    The status line is repainted from this call; letting the failure out
    would leave the panel with no status at all.
    """
    from spacr.qt.widgets.preview_contract import LivePreviewContract

    class _Broken(LivePreviewContract):
        def _preview_blocked_reason(self):
            raise RuntimeError("the panel is half-built")

    panel = _Broken()

    assert panel.preview_blocked_reason() == ""
    assert panel.can_preview() is True


def test_a_panel_that_names_a_reason_is_blocked():
    """A hook that answers does block the run guard."""
    from spacr.qt.widgets.preview_contract import LivePreviewContract

    class _Blocked(LivePreviewContract):
        def _preview_blocked_reason(self):
            return "Load an image first."

    panel = _Blocked()

    assert panel.preview_blocked_reason() == "Load an image first."
    assert panel.can_preview() is False


def test_a_deleted_worker_is_not_running():
    """A worker whose C++ object is gone leaves the panel idle, not busy."""
    panel = _panel()

    class _Deleted:
        def isRunning(self):  # noqa: N802 - Qt name
            raise RuntimeError("Internal C++ object already deleted.")

    panel._worker = _Deleted()

    assert panel.preview_running() is False


def test_no_worker_at_all_is_not_running():
    """A panel that has never started a pass is idle."""
    assert _panel().preview_running() is False


def test_a_panel_with_no_status_label_swallows_the_status(qtbot):
    """Setting a status on a panel with no label is a no-op, not an error."""
    panel = _panel()

    assert panel.set_preview_status("scanning") is None
    assert panel.preview_status() == ""


def test_a_result_with_no_token_is_not_stale():
    """A pass that carried no token is the only answer there is."""
    assert _panel().preview_stale(None) is False


def test_a_result_with_an_unreadable_token_is_not_stale():
    """A token that is not a number is kept rather than discarded."""
    panel = _panel()

    assert panel.preview_stale("not-a-token") is False
    assert panel.preview_stale(object()) is False


def test_a_result_from_a_superseded_pass_is_stale():
    """A token behind the current generation is dropped on arrival."""
    panel = _panel()
    panel._run_token = 3

    assert panel.preview_stale(2) is True
    assert panel.preview_stale(3) is False
