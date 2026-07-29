"""Regression tests for Annotate's native worker shutdown contract."""
from __future__ import annotations

from copy import deepcopy
import sqlite3
import threading
import time

import numpy as np
from PIL import Image


def _request(screen, generation, rows):
    return (
        generation,
        list(rows),
        None,
        deepcopy(screen._settings),
    )


def test_page_load_worker_processes_native_image_calls_sequentially(qtbot):
    """One page QThread must not create a second pool of native workers."""
    from spacr.qt.screens.annotate import _PageLoadWorker

    lock = threading.Lock()
    active = 0
    max_active = 0
    seen = []

    def load(row):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
            seen.append(row)
        return Image.new("RGB", (4, 4)), None

    worker = _PageLoadWorker(1, list(range(12)), load)
    with qtbot.waitSignal(worker.finished, timeout=5000):
        worker.start()

    assert seen == list(range(12))
    assert max_active == 1
    worker.deleteLater()


def test_page_loader_serializes_requests_and_keeps_only_latest(
    qtbot, qt_theme_applied, monkeypatch,
):
    """Rapid page/resize changes never overlap QThreads or replay stale work."""
    from spacr.qt.screens import annotate as annotate_mod

    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0
    seen = []

    def load(row, src, settings):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            seen.append(row)
        if row == "first":
            entered.set()
            assert release.wait(5)
        time.sleep(0.01)
        with lock:
            active -= 1
        return Image.new("RGB", settings.image_size), None

    monkeypatch.setattr(annotate_mod, "_load_thumb_image_worker", load)
    screen._queue_page_load(_request(screen, 1, ["first"]))
    assert entered.wait(5)

    # The middle request is obsolete before the first worker can finish.
    screen._queue_page_load(_request(screen, 2, ["stale"]))
    screen._queue_page_load(_request(screen, 3, ["latest"]))
    assert screen._pending_page_load[0] == 3
    release.set()

    qtbot.waitUntil(
        lambda: (
            screen._page_worker is None
            and screen._pending_page_load is None
            and seen == ["first", "latest"]
        ),
        timeout=5000,
    )
    assert max_active == 1
    screen.close()


def test_close_waits_for_active_page_worker_and_drops_pending_request(
    qtbot, qt_theme_applied, monkeypatch,
):
    """Closing cannot destroy the screen while its QThread still runs."""
    from spacr.qt.screens import annotate as annotate_mod

    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    entered = threading.Event()
    release = threading.Event()
    seen = []

    def load(row, src, settings):
        seen.append(row)
        entered.set()
        assert release.wait(5)
        return Image.new("RGB", settings.image_size), None

    monkeypatch.setattr(annotate_mod, "_load_thumb_image_worker", load)
    screen._queue_page_load(_request(screen, 1, ["active"]))
    assert entered.wait(5)
    screen._queue_page_load(_request(screen, 2, ["must-not-run"]))

    timer = threading.Timer(0.1, release.set)
    timer.start()
    started = time.monotonic()
    screen.close()
    elapsed = time.monotonic() - started
    timer.join()

    assert elapsed >= 0.08
    assert seen == ["active"]
    assert screen._closing is True
    assert screen._page_worker is None
    assert screen._pending_page_load is None


def test_cellpose_outline_model_is_never_evaluated_concurrently(monkeypatch):
    """The shared PyTorch/Cellpose model is protected across Python threads."""
    from spacr.qt import annotate_engine as engine

    lock = threading.Lock()
    active = 0
    max_active = 0

    class FakeModel:
        def eval(self, image, **_kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return (np.ones_like(image, dtype=np.uint8),)

    monkeypatch.setattr(engine, "_cellpose_outline_model", FakeModel())
    errors = []

    def run():
        try:
            engine._cellpose_foreground(np.ones((8, 8), dtype=np.float32))
        except Exception as exc:  # pragma: no cover - assertion reports detail
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert max_active == 1


def test_save_worker_stop_drains_sqlite_and_joins_thread(tmp_path):
    """A blocking stop leaves no daemon thread or live SQLite connection."""
    from spacr.qt.annotate_engine import SaveWorker, ensure_annotation_column

    db = tmp_path / "measurements.db"
    paths = [f"crop-{i}.png" for i in range(20)]
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany(
            'INSERT INTO "png_list" (png_path) VALUES (?)',
            [(path,) for path in paths],
        )
    ensure_annotation_column(str(db), "annotate")

    worker = SaveWorker(str(db), "annotate")
    worker.start()
    worker.submit({path: 1 for path in paths})
    worker.stop(wait=True)

    assert worker.is_alive is False
    with sqlite3.connect(db) as conn:
        saved = conn.execute(
            'SELECT COUNT(*) FROM "png_list" WHERE annotate = 1'
        ).fetchone()[0]
    assert saved == len(paths)
