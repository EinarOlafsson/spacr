"""What the re-fit dialog promises before a fit, and what ``ask_refit`` returns.

The re-fit dialog is the one plot context-menu entry that STARTS A FIT, so its
notice line and its Re-fit button are promises about the next run: settings
that name no usable count table grey the button and say so, a later usable
count path names the destination it derives, a dropped penalty weight is
announced, and the notice is still written when a host has emptied the button
box.  ``ask_refit`` hands back ``(settings, notes)`` only for an accepted
dialog; a cancel starts nothing.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture; every case builds its own dialog.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog, QDialogButtonBox  # noqa: E402

from spacr.qt.widgets import refit_dialog as rd  # noqa: E402

pytestmark = pytest.mark.qt

#: A settings dict a re-fit can actually run from.
_RUNNABLE = {
    "count_data": "/data/screen/counts.csv",
    "score_data": "/data/screen/scores.csv",
    "regression_type": "ols",
    "level": "both",
}

_SCREEN_DEST = f"Writes to {os.path.normpath('/data/screen/results/ols')}."


def _ok(dialog):
    return dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)


def test_the_refit_notice_and_button_follow_the_count_data(qtbot, subtests):
    """No usable counts: refused and greyed; usable counts: destination named."""
    cases = [
        # id, settings, notice exactly (str) or fragments (list), Re-fit enabled
        ("runnable settings name their destination", dict(_RUNNABLE),
         _SCREEN_DEST, True),
        ("no count data at all", {"regression_type": "ols"},
         ["count data"], False),
        ("only a blank count path", dict(_RUNNABLE, count_data=[""]),
         ["no usable count data path"], False),
        ("a later usable count path, dropping alpha",
         dict(_RUNNABLE, count_data=["", "/data/second/counts.csv"],
              alpha=0.3),
         ["does not read alpha=0.3", "reset to default",
          f"Writes to {os.path.normpath('/data/second/results/ols')}."],
         True),
        ("a named count path, dropping alpha", dict(_RUNNABLE, alpha=0.3),
         ["does not read alpha=0.3", _SCREEN_DEST], True),
    ]
    for case_id, settings, notice, enabled in cases:
        with subtests.test(msg=case_id):
            dialog = rd.RefitDialog(settings)
            qtbot.addWidget(dialog)
            text = dialog._notice.text()
            if isinstance(notice, str):
                assert text == notice
            else:
                for fragment in notice:
                    assert fragment in text
                if not enabled:
                    assert "Writes to" not in text
            assert _ok(dialog).text() == "Re-fit"
            assert _ok(dialog).isEnabled() is enabled
            box = dialog.findChild(QDialogButtonBox)
            assert box.button(QDialogButtonBox.Cancel).isEnabled() is True

    with subtests.test(msg="the notice survives an emptied button box"):
        dialog = rd.RefitDialog(dict(_RUNNABLE))
        qtbot.addWidget(dialog)
        box = dialog.findChild(QDialogButtonBox)
        box.clear()
        assert box.button(QDialogButtonBox.Ok) is None
        dialog._fdr_alpha.setValue(0.01)
        assert "significance level 0.05 -> 0.01" in dialog._notice.text()
        assert _SCREEN_DEST in dialog._notice.text()


def _pick_ridge(dialog):
    index = dialog._type.findData("ridge")
    assert index >= 0, "ridge is not offered by the dialog"
    dialog._type.setCurrentIndex(index)


def test_ask_refit_returns_the_run_only_when_accepted(qtbot, monkeypatch,
                                                      subtests):
    """Cancel is ``None``; accept is the new settings and what changed."""
    cases = [
        # id, exec stand-in, expected regression type (None: cancelled), notes
        ("cancelling starts no fit", lambda self: QDialog.Rejected, None,
         None),
        ("accepting unchanged hands back the same run",
         lambda self: QDialog.Accepted, "ols", []),
        ("accepting a new model hands back the change",
         lambda self: (_pick_ridge(self), QDialog.Accepted)[1], "ridge",
         ["'ols' -> 'ridge'"]),
    ]
    for case_id, run, regression, notes in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as patch:
                patch.setattr(rd.RefitDialog, "exec", run)
                answer = rd.ask_refit(dict(_RUNNABLE))
            if regression is None:
                assert answer is None
                continue
            settings, said = answer
            assert settings["regression_type"] == regression
            assert settings["count_data"] == _RUNNABLE["count_data"]
            assert isinstance(said, list)
            for fragment in notes:
                assert any(fragment in note for note in said), said
            if not notes:
                assert not any("->" in note for note in said), said
