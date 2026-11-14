"""Batch Runner: editing refusals, the queue table and the file pickers.

The screen is left alone overnight, so every refusal has to land in the
inline status line with its exact sentence, the table has to survive rows and
cells it did not draw itself, and a cancelled file dialog must never change
the queue or write a file.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from PySide6.QtWidgets import QFileDialog

from spacr import batch as bt
from spacr.qt.screens.batch import COLUMNS, BatchScreen

pytestmark = pytest.mark.qt

TIME = COLUMNS.index("Time")


def _plate(tmp_path, name):
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    (src / f"{name}_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"")
    return str(src)


def _settings(tmp_path, name, plate):
    path = tmp_path / name
    path.write_text(f"Key,Value\nsrc,{_plate(tmp_path, plate)}\ncell_channel,0\n",
                    encoding="utf-8")
    return str(path)


class _Runner:
    """Stands in for ``spacr-run``; nothing real is ever executed."""

    def __call__(self, job, settings_path, log_path):
        return 0


@pytest.fixture()
def screen(qtbot):
    widget = BatchScreen(threaded=False, runner=_Runner())
    qtbot.addWidget(widget)
    return widget


@pytest.fixture()
def settings(tmp_path):
    return _settings(tmp_path, "mask.csv", "plate1")


def _dialog(monkeypatch, name, answer):
    """Replace one QFileDialog static with an answer; record the captions."""
    asked = []

    def _answer(*args, **kwargs):
        asked.append(args[1:])
        return answer, ""

    monkeypatch.setattr(QFileDialog, name, staticmethod(_answer))
    return asked


def _select_nothing(screen):
    screen._table.setCurrentCell(-1, -1)


def test_editing_refusals_keep_the_queue_and_say_why(screen, tmp_path,
                                                    subtests):
    """Duplicate, remove and move refuse with their own exact sentence."""
    first = _settings(tmp_path, "mask.csv", "plate1")
    second = _settings(tmp_path, "mask2.csv", "plate2")
    assert screen.add_job("mask", first, label="first") is True
    assert screen.add_job("mask", second, label="second") is True

    def running_first():
        screen.select_job("mask-1")
        screen.queue().jobs[0].status = bt.STATUS_RUNNING
        screen._busy = True

    def busy_second():
        screen.select_job("mask-2")
        screen._busy = True

    def vanished_settings():
        screen.select_job("mask-1")
        os.remove(first)

    cases = [
        ("duplicate with nothing selected", lambda: _select_nothing(screen),
         screen.duplicate_selected, "Select a job to duplicate."),
        ("remove with nothing selected", lambda: _select_nothing(screen),
         screen.remove_selected, "Select a job to remove."),
        ("move with nothing selected", lambda: _select_nothing(screen),
         lambda: screen.move_selected(-1), "Select a job to reorder."),
        ("remove the running job", running_first, screen.remove_selected,
         "That job is running — stop the queue first."),
        ("reorder a running queue", busy_second,
         lambda: screen.move_selected(-1),
         "The queue is running — stop it before reordering."),
        ("duplicate a job whose settings vanished", vanished_settings,
         screen.duplicate_selected,
         "Could not duplicate — see the problems above."),
    ]
    for case_id, prepare, act, message in cases:
        with subtests.test(msg=case_id):
            prepare()
            try:
                assert act() is False
                assert screen.queue().ids == ["mask-1", "mask-2"]
                assert screen.last_error == message
                assert screen.status_text() == message
            finally:
                screen._busy = False
                screen.queue().jobs[0].status = bt.STATUS_PENDING
    assert "not found" in screen.problems_text()
    assert first in screen.problems_text()


def test_the_table_drops_stray_rows_and_redraws_a_lost_time_cell(
        screen, tmp_path, subtests):
    """Rows below the queue go; the tick redraws only the running job."""
    first = _settings(tmp_path, "mask.csv", "plate1")
    second = _settings(tmp_path, "mask2.csv", "plate2")
    screen.add_job("mask", first, label="first")
    screen.add_job("mask", second, label="second")

    with subtests.test(msg="empty rows below the queue are dropped"):
        screen._table.setRowCount(5)
        assert screen._table.item(3, 0) is None
        screen._refresh_table()
        assert screen._table.rowCount() == 2
        assert [screen.row_values(r)[:4] for r in range(2)] == [
            ["1", "mask-1", "mask", "first"],
            ["2", "mask-2", "mask", "second"]]
        assert [screen.row_values(r)[TIME] for r in range(2)] == ["—", "—"]

    now = datetime.now(timezone.utc)
    running = screen.queue().jobs[1]
    running.status = bt.STATUS_RUNNING
    running.started = (now - timedelta(seconds=65)).isoformat()
    running.finished = now.isoformat()
    log = tmp_path / "mask-2.log"
    log.write_text("segmenting field 3\n", encoding="utf-8")
    running.log_path = str(log)
    screen._refresh_table()

    with subtests.test(msg="a cell the table lost is drawn again"):
        assert screen._table.takeItem(1, TIME) is not None
        assert screen._table.item(1, TIME) is None
        _select_nothing(screen)
        screen._refresh_running_row()
        assert screen._table.item(1, TIME).text() == "1m 05s"
        assert screen.log_text() == ""

    with subtests.test(msg="an existing cell ticks and the selected log loads"):
        screen._table.item(1, TIME).setText("stale")
        screen.select_job("mask-2")
        screen._refresh_running_row()
        assert screen.row_values(1)[TIME] == "1m 05s"
        assert screen.row_values(0)[TIME] == "—"
        assert screen.log_text() == "segmenting field 3\n"

    with subtests.test(msg="a running job with no row is skipped"):
        screen._table.setRowCount(1)
        screen._refresh_running_row()
        assert screen._table.rowCount() == 1
        assert screen.row_values(0)[TIME] == "—"


def test_the_buttons_beside_the_editor_fill_add_open_and_save(
        screen, settings, tmp_path, monkeypatch, subtests):
    """Pickers only act on a chosen path; cancel changes nothing."""
    with subtests.test(msg="add reads the boxes, splitting the lists"):
        screen._module_combo.setCurrentIndex(
            screen._module_combo.findData("mask"))
        screen._settings_edit.setText(settings)
        screen._label_edit.setText("typed in")
        screen._overrides_edit.setText("cell_channel=0, ,nucleus_channel=0")
        screen._on_add_clicked()
        job = screen.queue().jobs[0]
        assert (job.module, job.label, job.override_args, job.depends_on) == (
            "mask", "typed in", ["cell_channel=0", "nucleus_channel=0"], [])
        assert screen.queue().ids == ["mask-1"]

    with subtests.test(msg="a chosen settings file only fills the box"):
        screen._settings_edit.setText("")
        asked = _dialog(monkeypatch, "getOpenFileName", settings)
        screen._pick_settings_file()
        assert screen._settings_edit.text() == settings
        assert asked == [("Choose a settings file", "",
                          "Settings (*.csv *.json);;All files (*)")]
        assert screen.queue().ids == ["mask-1"]

    with subtests.test(msg="a cancelled settings picker keeps the box"):
        screen._settings_edit.setText("keep me")
        _dialog(monkeypatch, "getOpenFileName", "")
        screen._pick_settings_file()
        assert screen._settings_edit.text() == "keep me"

    with subtests.test(msg="a cancelled queue open keeps the queue"):
        asked = _dialog(monkeypatch, "getOpenFileName", "")
        screen._pick_queue_to_load()
        assert asked == [("Open a queue", "",
                          "Queue files (*.json);;All files (*)")]
        assert screen.queue().ids == ["mask-1"]

    with subtests.test(msg="a cancelled save writes nothing"):
        asked = _dialog(monkeypatch, "getSaveFileName", "")
        screen._pick_queue_to_save()
        assert asked == [("Save the queue", "queue.json",
                          "Queue files (*.json);;All files (*)")]
        assert sorted(n for n in os.listdir(tmp_path)
                      if n.endswith(".json")) == []
        assert screen.queue_path() == ""

    saved = str(tmp_path / "saved.json")
    with subtests.test(msg="a chosen save path is written and remembered"):
        _dialog(monkeypatch, "getSaveFileName", saved)
        screen._pick_queue_to_save()
        assert screen.queue_path() == saved
        assert screen.status_text() == (
            f"Saved 1 job(s) to {saved}. The run keeps this file up to date, "
            f"so it can be resumed.")
        with open(saved, encoding="utf-8") as handle:
            assert [j["label"] for j in json.load(handle)["jobs"]] == [
                "typed in"]

    with subtests.test(msg="the next save starts at the remembered path"):
        asked = _dialog(monkeypatch, "getSaveFileName", "")
        screen._pick_queue_to_save()
        assert asked[0][1] == saved
