"""A re-fit dialog that cannot build a runnable settings dict says so first.

This dialog is the one plot context-menu entry that STARTS A FIT, so what it
refuses matters as much as what it offers. Results opened from disk without
their settings CSV, or settings that name no count data, cannot be re-fitted
at all -- and the dialog has to say that in its own notice line with the
Re-fit button disabled, rather than let the user press it and fail minutes
later inside the regression.

:func:`spacr.qt.widgets.refit_dialog.ask_refit` is the caller's whole view of
this: an accepted dialog hands back ``(settings, notes)`` and a cancelled one
hands back nothing at all.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog, QDialogButtonBox   # noqa: E402

from spacr.qt.widgets import refit_dialog as rd           # noqa: E402

pytestmark = pytest.mark.qt


_RUNNABLE = {
    "count_data": "/data/screen/counts.csv",
    "score_data": "/data/screen/scores.csv",
    "regression_type": "ols",
    "level": "both",
}


def _ok_button(dialog):
    box = dialog.findChild(QDialogButtonBox)
    return box.button(QDialogButtonBox.Ok)


def test_settings_with_no_count_data_disable_the_refit_button(qtbot):
    """Nothing to re-fit from is refused up front, in the dialog's own words."""
    dialog = rd.RefitDialog({"regression_type": "ols"})
    qtbot.addWidget(dialog)

    assert _ok_button(dialog).isEnabled() is False
    assert "count data" in dialog._notice.text()


def test_runnable_settings_leave_the_refit_button_available(qtbot):
    """The refusal above is a refusal, not the dialog's normal state."""
    dialog = rd.RefitDialog(dict(_RUNNABLE))
    qtbot.addWidget(dialog)

    assert _ok_button(dialog).isEnabled() is True
    assert dialog._notice.text()


def test_accepting_the_dialog_hands_back_the_new_settings(qtbot, monkeypatch):
    """``ask_refit`` returns what the run needs, not the dialog."""
    monkeypatch.setattr(rd.RefitDialog, "exec",
                        lambda self: QDialog.Accepted)

    answer = rd.ask_refit(dict(_RUNNABLE))

    assert answer is not None
    settings, notes = answer
    assert settings["count_data"] == _RUNNABLE["count_data"]
    assert isinstance(notes, list)


def test_cancelling_the_dialog_starts_no_fit(qtbot, monkeypatch):
    """Cancel has to be distinguishable from "re-fit with no changes"."""
    monkeypatch.setattr(rd.RefitDialog, "exec",
                        lambda self: QDialog.Rejected)

    assert rd.ask_refit(dict(_RUNNABLE)) is None
