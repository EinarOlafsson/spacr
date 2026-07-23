"""Tests for the plate queue.

Split into:

* Pure-Python queue mechanics (add/remove/save/load, CSV import,
  ``run_queue`` with an injectable runner).
* Qt QueueScreen wiring (construction, table rows, wire_add_current,
  runner start/stop with a stubbed pipeline entry).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from spacr.qt import plate_queue as pq
from spacr.qt.plate_queue import (
    PlateQueue, QueueItem, Status, import_plates_from_csv, run_queue,
)


# ---------------------------------------------------------------------------
# Isolation — never touch ~/.spacr/queue.json
# ---------------------------------------------------------------------------

@pytest.fixture
def q(tmp_path: Path) -> PlateQueue:
    return PlateQueue(path=tmp_path / "queue.json")


# ---------------------------------------------------------------------------
# Queue mechanics
# ---------------------------------------------------------------------------

class TestPlateQueue:
    def test_new_queue_is_empty(self, q):
        assert len(q) == 0
        assert q.next_queued() is None
        assert q.is_all_done() is True

    def test_add_persists_to_disk(self, tmp_path):
        path = tmp_path / "q.json"
        q = PlateQueue(path=path)
        q.add(QueueItem.build("mask", {"src": "/tmp/a"}))
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["items"][0]["app_key"] == "mask"

    def test_load_roundtrips(self, tmp_path):
        path = tmp_path / "q.json"
        q1 = PlateQueue(path=path)
        it = QueueItem.build("mask", {"src": "/tmp/x"})
        q1.add(it)
        q2 = PlateQueue(path=path)
        assert len(q2) == 1
        assert q2.items()[0].label == it.label

    def test_remove(self, q):
        it = QueueItem.build("mask", {"src": "/tmp/a"})
        q.add(it)
        assert q.remove(it.id) is True
        assert q.remove("nonexistent") is False
        assert len(q) == 0

    def test_clear_finished(self, q):
        it1 = QueueItem.build("mask", {"src": "/tmp/a"})
        it2 = QueueItem.build("mask", {"src": "/tmp/b"})
        it3 = QueueItem.build("mask", {"src": "/tmp/c"})
        q.add(it1); q.add(it2); q.add(it3)
        q.update(it1.id, status=Status.SUCCESS)
        q.update(it2.id, status=Status.FAILED)
        n = q.clear_finished()
        assert n == 2
        assert len(q) == 1
        assert q.items()[0].id == it3.id

    def test_update_missing_id_is_noop(self, q):
        q.update("nope", status=Status.SUCCESS)
        assert len(q) == 0

    def test_bad_queue_file_starts_empty(self, tmp_path):
        p = tmp_path / "corrupt.json"
        p.write_text("{not valid json")
        q = PlateQueue(path=p)
        assert len(q) == 0

    def test_status_transitions(self, q):
        it = QueueItem.build("mask", {"src": "/tmp/a"})
        q.add(it)
        assert q.next_queued() is it
        q.update(it.id, status=Status.RUNNING)
        assert q.next_queued() is None
        q.update(it.id, status=Status.SUCCESS)
        assert q.is_all_done() is True


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------

class TestCsvImport:
    def test_import_uses_src_and_overrides(self, tmp_path):
        csv_path = tmp_path / "plates.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["src", "diameter", "flow_threshold"])
            w.writerow(["/tmp/plate_a", "30", "0.4"])
            w.writerow(["/tmp/plate_b", "45", "0.6"])
        items = import_plates_from_csv(
            csv_path, base_settings={"model_name": "cyto3"},
            app_key="mask")
        assert len(items) == 2
        assert items[0].settings["src"] == "/tmp/plate_a"
        assert items[0].settings["diameter"] == 30
        assert items[0].settings["flow_threshold"] == pytest.approx(0.4)
        assert items[0].settings["model_name"] == "cyto3"

    def test_import_skips_rows_without_src(self, tmp_path):
        csv_path = tmp_path / "plates.csv"
        with csv_path.open("w", newline="") as fh:
            fh.write("src,diameter\n\n/tmp/x,30\n")
        items = import_plates_from_csv(csv_path, {}, "mask")
        assert len(items) == 1

    def test_import_bool_and_none_coercion(self, tmp_path):
        csv_path = tmp_path / "plates.csv"
        with csv_path.open("w", newline="") as fh:
            fh.write("src,plot,percentiles\n/tmp/x,true,none\n")
        items = import_plates_from_csv(csv_path, {}, "mask")
        assert items[0].settings["plot"] is True
        assert items[0].settings["percentiles"] is None

    def test_import_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_plates_from_csv(tmp_path / "gone.csv", {}, "mask")


# ---------------------------------------------------------------------------
# run_queue with an injectable runner
# ---------------------------------------------------------------------------

class TestRunQueue:
    def test_run_queue_processes_all(self, q):
        for src in ("a", "b", "c"):
            q.add(QueueItem.build("mask", {"src": src}))
        seen = []
        def runner(item):
            seen.append(item.settings["src"])
        run_queue(q, runner=runner)
        assert seen == ["a", "b", "c"]
        assert all(i.status == Status.SUCCESS for i in q.items())

    def test_run_queue_marks_failure(self, q):
        q.add(QueueItem.build("mask", {"src": "bad"}))
        q.add(QueueItem.build("mask", {"src": "ok"}))
        def runner(item):
            if item.settings["src"] == "bad":
                raise RuntimeError("oops")
        run_queue(q, runner=runner)
        statuses = [i.status for i in q.items()]
        assert Status.FAILED in statuses
        assert Status.SUCCESS in statuses

    def test_run_queue_stop_on_error(self, q):
        q.add(QueueItem.build("mask", {"src": "bad"}))
        q.add(QueueItem.build("mask", {"src": "next"}))
        def runner(item):
            raise RuntimeError("nope")
        run_queue(q, runner=runner, stop_on_error=True)
        assert q.items()[0].status == Status.FAILED
        # Second item stays queued
        assert q.items()[1].status == Status.QUEUED


# ---------------------------------------------------------------------------
# Qt screen wiring
# ---------------------------------------------------------------------------

class TestQueueScreen:
    def test_screen_constructs(self, qtbot, tmp_path):
        from spacr.qt.screens.queue import QueueScreen
        q = PlateQueue(path=tmp_path / "q.json")
        scr = QueueScreen(queue=q)
        qtbot.addWidget(scr)
        assert scr.queue() is q

    def test_add_item_populates_table(self, qtbot, tmp_path):
        from spacr.qt.screens.queue import QueueScreen
        q = PlateQueue(path=tmp_path / "q.json")
        scr = QueueScreen(queue=q)
        qtbot.addWidget(scr)
        scr.add_item("mask", {"src": "/tmp/plate1"})
        assert scr._table.rowCount() == 1
        assert scr._table.item(0, 1).text() == "mask"

    def test_wire_add_current_uses_callback(self, qtbot, tmp_path):
        from spacr.qt.screens.queue import QueueScreen
        q = PlateQueue(path=tmp_path / "q.json")
        scr = QueueScreen(queue=q)
        qtbot.addWidget(scr)
        scr.wire_add_current(lambda: ("mask", {"src": "/tmp/foo"}))
        # Click programmatically
        scr._btn_add.click()
        assert len(q) == 1
        assert q.items()[0].settings["src"] == "/tmp/foo"

    def test_wire_add_current_ignores_empty_src(self, qtbot, tmp_path,
                                                    monkeypatch):
        from spacr.qt.screens.queue import QueueScreen
        from PySide6.QtWidgets import QMessageBox
        # Neutralise message-box dialogs so tests don't block
        monkeypatch.setattr(QMessageBox, "information",
                             lambda *a, **k: QMessageBox.Ok)
        q = PlateQueue(path=tmp_path / "q.json")
        scr = QueueScreen(queue=q)
        qtbot.addWidget(scr)
        scr.wire_add_current(lambda: ("mask", {"src": ""}))
        scr._btn_add.click()
        assert len(q) == 0

    def test_remove_row_shrinks_table(self, qtbot, tmp_path):
        from spacr.qt.screens.queue import QueueScreen
        q = PlateQueue(path=tmp_path / "q.json")
        scr = QueueScreen(queue=q)
        qtbot.addWidget(scr)
        it = scr.add_item("mask", {"src": "/tmp/x"})
        scr._on_remove(it.id)
        assert scr._table.rowCount() == 0

    def test_runner_processes_stubbed_pipeline(self, qtbot, tmp_path,
                                                    monkeypatch):
        """Stub resolve_pipeline_entry so the runner exercises the QThread
        path without loading spacr.core."""
        from spacr.qt.screens import queue as qm
        called = []
        def _fake_entry(app_key):
            def _fn(settings):
                called.append((app_key, dict(settings)))
            return _fn
        monkeypatch.setattr(
            "spacr.qt.bridge.resolve_pipeline_entry", _fake_entry)

        q = PlateQueue(path=tmp_path / "q.json")
        scr = qm.QueueScreen(queue=q)
        qtbot.addWidget(scr)
        scr.add_item("mask", {"src": "/tmp/a"})
        scr.add_item("mask", {"src": "/tmp/b"})

        finished = []
        scr._btn_run.click()
        # Wait for the runner to finish naturally
        qtbot.waitUntil(lambda: q.is_all_done() and
                         (scr._runner is None or not scr._runner.isRunning()),
                         timeout=5000)
        srcs = [c[1]["src"] for c in called]
        assert srcs == ["/tmp/a", "/tmp/b"]
        assert all(i.status == Status.SUCCESS for i in q.items())
