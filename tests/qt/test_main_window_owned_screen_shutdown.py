"""Main-window close drains cached screens and respects their close vetoes."""

from copy import deepcopy
import sqlite3
import threading

from PIL import Image
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QWidget

from spacr.qt.app import MainWindow


def test_cached_annotate_close_drains_page_and_saves_pending_labels_once(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    from spacr.qt.screens import annotate

    source = tmp_path / "experiment"
    (source / "measurements").mkdir(parents=True)
    (source / "data").mkdir()
    crop = source / "data" / "crop.png"
    Image.new("RGB", (24, 24), "blue").save(crop)
    crop_bytes = crop.read_bytes()
    db = source / "measurements" / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        connection.execute('INSERT INTO "png_list" VALUES (?)', (str(crop),))

    entered, release = threading.Event(), threading.Event()
    original_load = annotate._load_thumb_image_worker
    loaded, interrupted = [], []

    def blocked_load(row, src, settings, should_stop=None):
        loaded.append(row)
        entered.set()
        assert release.wait(5)
        interrupted.append(should_stop())
        return original_load(row, src, settings, should_stop=should_stop)

    monkeypatch.setattr(annotate, "_load_thumb_image_worker", blocked_load)
    window = MainWindow()
    qtbot.addWidget(window)
    screen = annotate.AnnotateScreen()
    window._stack.addWidget(screen)
    window._stack.setCurrentWidget(screen)
    window._screens["annotate"] = screen
    window._screens["annotate-alias"] = screen
    screen._settings.grid_rows = screen._settings.grid_cols = 1
    screen._settings.image_size = (24, 24)
    screen._compute_grid_dims = lambda: None
    close_calls = []
    original_close = screen.close

    def close_screen():
        close_calls.append(True)
        return original_close()

    monkeypatch.setattr(screen, "close", close_screen)
    try:
        screen._open_source(str(source))
        qtbot.waitUntil(entered.is_set, timeout=5000)
        page_worker = screen._page_worker
        original_interrupt = page_worker.requestInterruption

        def interrupt_page():
            original_interrupt()
            release.set()

        monkeypatch.setattr(page_worker, "requestInterruption", interrupt_page)
        writer = screen._worker
        result_signal = QSignalSpy(page_worker.done)
        assert page_worker.isRunning()
        assert writer.is_alive
        assert screen._set_annotation(0, 1)
        assert screen._pending_updates == {str(crop): 1}
        with sqlite3.connect(db) as connection:
            assert connection.execute('SELECT annotate FROM png_list').fetchone() == (None,)
        screen._queue_page_load((screen._page_gen + 1, list(screen._page_paths),
                                 None, deepcopy(screen._settings)))

        assert window.close()

        assert close_calls == [True]
        assert screen._closing
        assert screen._page_worker is None
        assert screen._pending_page_load is None
        assert not page_worker.isRunning()
        assert interrupted == [True]
        assert screen._worker is None
        assert not writer.is_alive
        assert screen._pending_updates == {}
        with sqlite3.connect(db) as connection:
            assert connection.execute('SELECT annotate FROM png_list').fetchone() == (1,)
        qtbot.wait(10)
        assert result_signal.count() == 0
        assert len(loaded) == 1
        assert screen._thumb_pixmaps == [None]
        assert crop.read_bytes() == crop_bytes
    finally:
        release.set()
        original_close()


def test_cached_widget_close_is_unique_and_a_veto_keeps_window_open(qtbot):
    closed = []

    class Screen(QWidget):
        def __init__(self, name, reject=False):
            super().__init__()
            self.name = name
            self.reject = reject

        def closeEvent(self, event):
            closed.append(self.name)
            if self.reject:
                event.ignore()
            else:
                super().closeEvent(event)

    window = MainWindow()
    qtbot.addWidget(window)
    first, veto, later = Screen("first"), Screen("veto", True), Screen("later")
    for screen in (first, veto, later):
        window._stack.addWidget(screen)
    window._screens.update(first=first, alias=first, veto=veto, later=later)
    try:
        assert window.close() is False
        assert closed == ["first", "veto"]
        assert window._closing is False
        veto.reject = False
        closed.clear()
        assert window.close()
        assert closed == ["first", "veto", "later"]
    finally:
        veto.reject = False
        window.close()
