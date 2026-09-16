"""Guards for the sweep-runs half of instruction 100's last four repairs.

A failed delete has to leave the row on the table AND say which folder would
not go. The xfail that came off asserted only the first; these pin the second,
and the shape change underneath both.

Kept apart from ``test_xfail_backlog_item_100_repairs`` so the Qt skip cannot
take the other three modules' guards down with it.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spacr.qt.widgets import sweep_runs as sr                    # noqa: E402


def test_a_delete_failure_carries_the_folder_and_the_reason_apart(tmp_path):
    """`_delete_folders` reports a pair, so the folder can still be matched."""
    missing = str(tmp_path / "not-there")

    deleted, failed = sr._delete_folders([missing])

    assert deleted == []
    assert len(failed) == 1
    folder, why = failed[0]
    assert folder == missing
    assert why and missing not in why


def test_the_failure_sentence_still_names_the_folder_and_the_reason():
    """Splitting the pair must not cost the words the user reads."""
    rendered = sr._format_failures([("/runs/ols_1", "Permission denied")])

    assert "/runs/ols_1" in rendered
    assert "Permission denied" in rendered


def test_the_folder_is_recoverable_from_either_failure_shape():
    """A pre-formatted entry from an older build still yields a path."""
    assert sr._failed_folder(("/runs/ols_1", "denied")) == "/runs/ols_1"
    assert sr._failed_folder("/runs/ols_1") == "/runs/ols_1"


def test_a_failed_delete_says_which_folder_would_not_go(qtbot, tmp_path,
                                                        monkeypatch):
    """The row survives AND the status line names the folder and the reason."""
    import shutil

    panel = sr.SweepRunsPanel()
    qtbot.addWidget(panel)

    folder = str(tmp_path / "results" / "ols_1")
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"feature": ["a"], "coefficient": [0.5],
                  "p_value": [0.1]}).to_csv(
        os.path.join(folder, "results.csv"), index=False)
    panel.load_run_from_disk(folder)

    def refuse(path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(shutil, "rmtree", refuse)

    assert panel.delete_runs_from_disk([panel.loaded_run()],
                                       confirm=lambda *a: True) == 0

    said = panel._status.text()
    assert folder in said
    assert "Permission denied" in said
    assert os.path.isdir(folder)
    assert len(panel._recorded) == 1
