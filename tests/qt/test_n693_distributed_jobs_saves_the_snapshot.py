"""Item 693: Distributed Jobs saves the Submit remote… snapshot as a CSV.

"Submit remote…" hands the form to Distributed Jobs as a snapshot and writes
no file. "Save settings CSV…" there writes that snapshot with the same writer
as Export settings…, so it can be copied to a cluster; Submit is unchanged.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog

from spacr.cli import load_settings_file
from spacr.qt.screens import app_screen as app_screen_module
from spacr.qt.screens.distributed_jobs import DistributedJobsScreen
from spacr.remote_execution import JobStore, ProfileStore, RemoteJobManager

pytestmark = pytest.mark.qt

SNAPSHOT = {"src": "/mnt/share/plate 1", "channels": [0, 1, 2],
            "cell_min_size": 800, "save_png": False, "pathogen_channel": None}


def _screen(qtbot, tmp_path):
    """A Distributed Jobs screen with empty stores and no polling."""
    manager = RemoteJobManager(ProfileStore(tmp_path / "profiles.json"),
                               JobStore(tmp_path / "jobs.json"),
                               lambda *a, **k: None)
    screen = DistributedJobsScreen(manager=manager, threaded=False,
                                   auto_poll=False)
    qtbot.addWidget(screen)
    return screen


def _save_to(monkeypatch, path):
    """Answer the save dialog with ``path`` and record what it offered."""
    offered = []

    def answer(_parent, _caption, default, _filter):
        offered.append(default)
        return str(path), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(answer))
    return offered


def test_the_button_waits_for_a_snapshot(qtbot, tmp_path):
    """Disabled until Submit remote… sends one, and again once edited."""
    screen = _screen(qtbot, tmp_path)
    assert screen._save_snapshot.text() == "Save settings CSV…"
    assert not screen._save_snapshot.isEnabled()
    screen.configure_submission("measure", SNAPSHOT)
    assert screen._save_snapshot.isEnabled()
    screen._settings_path.textEdited.emit("/other.csv")
    assert screen._settings_snapshot is None
    assert not screen._save_snapshot.isEnabled()


def test_saving_writes_the_snapshot_and_keeps_submit_unchanged(
        qtbot, monkeypatch, tmp_path):
    """The CSV holds the snapshot; the snapshot is still what Submit sends."""
    monkeypatch.setitem(app_screen_module._SETTINGS_CSV_FOLDER, "last", "")
    screen = _screen(qtbot, tmp_path)
    screen.configure_submission("measure", SNAPSHOT)
    target = tmp_path / "cluster" / "measure_settings.csv"
    offered = _save_to(monkeypatch, target)
    screen._save_snapshot_csv()
    assert offered == [os.path.join("/mnt/share/plate 1", "measure_settings.csv")]
    assert load_settings_file(str(target)) == SNAPSHOT
    assert screen._settings_snapshot == SNAPSHOT
    assert ("spacr-run measure --settings measure_settings.csv"
            in screen._status.text())
    assert str(target) in screen._status.text()


def test_cancel_and_write_failures(qtbot, monkeypatch, tmp_path):
    """Cancel leaves the status; a failed write is said in the status."""
    screen = _screen(qtbot, tmp_path)
    screen._save_snapshot_csv()
    screen.configure_submission("mask", SNAPSHOT)
    before = screen._status.text()
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen._save_snapshot_csv()
    assert screen._status.text() == before

    _save_to(monkeypatch, tmp_path / "out.csv")

    def broken(_settings, _path):
        raise OSError("read-only share")

    monkeypatch.setattr("spacr.cli.write_settings_file", broken)
    screen._save_snapshot_csv()
    assert "read-only share" in screen._status.text()
