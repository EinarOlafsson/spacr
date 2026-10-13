"""Load test data, then Load or Stream, ends with crops on the screen.

Reported 2026-09-21: pressing Load test data, then Load, then opening the
settings and pressing OK did nothing. The test data arrived and the settings
were half-filled, but the plate was never OPENED: no grid, no save worker, and
``src`` named the database file, so OK re-derived ``db_path`` as
``measurements.db/measurements/measurements.db`` and found nothing to page.

Every download here is simulated. The slow one finishes on a worker thread
after a real delay and reports back through a queued signal, which is the
shape of the real downloader, so nothing in the screen may rely on a timer
that happens to fire after the files land.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from spacr.qt.screens import annotate as annotate_mod

N_CROPS = 6
ROWS, COLS = 2, 3


def _write_annotate_half(plate: Path) -> None:
    """What the Load archive unpacks: crops, the database, the settings."""
    (plate / "measurements").mkdir(parents=True, exist_ok=True)
    (plate / "data").mkdir(parents=True, exist_ok=True)
    (plate / "settings").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    paths = []
    for i in range(N_CROPS):
        p = plate / "data" / f"crop_{i}.png"
        Image.fromarray(rng.integers(0, 255, (24, 24, 3),
                                     dtype=np.uint8)).save(p)
        paths.append(str(p))
    with sqlite3.connect(plate / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    (plate / "settings" / "annotate_settings.csv").write_text(
        "Key,Value\nannotation_column,infected\nimg_size,24\n"
        f"src,{plate}/measurements.db\n")


def _write_stream_half(plate: Path) -> None:
    """What the Stream archive unpacks: the merged arrays."""
    (plate / "merged").mkdir(parents=True, exist_ok=True)
    np.save(plate / "merged" / "field.npy", np.zeros((4, 4, 3)))


class _Landed(QObject):
    """Carries a finished simulated download back to the GUI thread."""

    done = Signal(object)


class FakeDownloads:
    """Stands in for the two example downloaders.

    Each call writes its half of the plate on a worker thread, after
    ``delay`` seconds, and hands the callback back to the GUI thread through
    a queued signal -- the way `_HFDownloadUI.on_finished` does.
    """

    def __init__(self, plate: Path, delay: float = 0.0):
        self.plate = plate
        self.delay = delay
        self.calls = []
        self._bridge = _Landed()
        self._bridge.done.connect(lambda job: job(), Qt.QueuedConnection)
        self._threads = []

    def _run(self, name, writer, on_done):
        self.calls.append(name)

        def work():
            time.sleep(self.delay)
            writer(self.plate)
            result = type("R", (), {"dataset_path": self.plate})()
            self._bridge.done.emit(lambda: on_done(result, ""))

        thread = threading.Thread(target=work, daemon=True)
        self._threads.append(thread)
        thread.start()

    def annotate(self, parent, dest, on_done):
        self._run("annotate", _write_annotate_half, on_done)

    def measure(self, parent, dest, on_done):
        self._run("measure", _write_stream_half, on_done)


class Chose:
    """The chooser, already answered."""

    def __init__(self, route):
        self.chosen = route

    def exec(self):
        return 1


@pytest.fixture
def plate(tmp_path, monkeypatch):
    import spacr.qt.hf_download as hf

    folder = tmp_path / "plate1"
    monkeypatch.setattr(hf, "example_plate_folder", lambda: folder)
    return folder


@pytest.fixture
def screen(qtbot):
    s = annotate_mod.AnnotateScreen()
    qtbot.addWidget(s)
    s._settings.grid_rows = ROWS
    s._settings.grid_cols = COLS
    s._compute_grid_dims = lambda: None
    s._rebuild_grid()
    yield s
    if s._worker is not None:
        s._worker.stop(wait=True)


@pytest.fixture
def downloads(plate, monkeypatch):
    import spacr.qt.hf_download as hf

    fake = FakeDownloads(plate)
    monkeypatch.setattr(hf, "download_annotate_example", fake.annotate)
    monkeypatch.setattr(hf, "download_measure_example", fake.measure)
    yield fake
    for thread in fake._threads:
        thread.join(timeout=10)


def _loaded(screen):
    """The state the user asked to land in: crops drawn, settings filled."""
    return (screen._content_stack.currentWidget() is screen._grid_scroll
            and len(screen._page_paths) == N_CROPS
            and all(screen._thumb_pixmaps[i] is not None
                    for i in range(N_CROPS)))


def _assert_filled_in(screen, plate):
    s = screen._settings
    assert s.src == str(plate), s.src
    assert s.db_path == str(plate / "measurements" / "measurements.db")
    assert s.annotation_column == "infected"
    assert screen._worker is not None, "no save worker: nothing would persist"


def test_load_with_the_data_absent_downloads_then_opens(qtbot, screen, plate,
                                                         downloads):
    assert screen._choose_the_test_data(chooser=Chose("load")) == ""
    assert downloads.calls == ["annotate"]
    qtbot.waitUntil(lambda: _loaded(screen), timeout=10000)
    _assert_filled_in(screen, plate)
    assert screen._settings.crop_source == "png"
    assert screen._btn_test_data.isEnabled()


def test_load_with_the_data_present_opens_straight_away(qtbot, screen, plate,
                                                         downloads):
    _write_annotate_half(plate)
    screen._choose_the_test_data(chooser=Chose("load"))
    assert downloads.calls == []
    qtbot.waitUntil(lambda: _loaded(screen), timeout=10000)
    _assert_filled_in(screen, plate)


def test_stream_with_nothing_present_fetches_both_halves(qtbot, screen, plate,
                                                          downloads):
    """Streaming cuts crops out of merged/ at the rows the database lists."""
    screen._choose_the_test_data(chooser=Chose("stream"))
    qtbot.waitUntil(lambda: _loaded(screen), timeout=10000)
    assert sorted(downloads.calls) == ["annotate", "measure"]
    _assert_filled_in(screen, plate)
    assert screen._settings.crop_source == "merged"


def test_stream_with_the_data_present_opens_straight_away(qtbot, screen, plate,
                                                           downloads):
    _write_annotate_half(plate)
    _write_stream_half(plate)
    screen._choose_the_test_data(chooser=Chose("stream"))
    assert downloads.calls == []
    qtbot.waitUntil(lambda: _loaded(screen), timeout=10000)
    assert screen._settings.crop_source == "merged"


def test_a_slow_download_still_ends_loaded(qtbot, screen, plate, downloads):
    downloads.delay = 1.5
    screen._choose_the_test_data(chooser=Chose("load"))
    assert not screen._btn_test_data.isEnabled()
    qtbot.wait(200)
    assert not _loaded(screen)
    qtbot.waitUntil(lambda: _loaded(screen), timeout=15000)
    _assert_filled_in(screen, plate)


def test_settings_ok_works_after_the_test_data(qtbot, screen, plate, downloads,
                                               monkeypatch):
    """The reported dead button."""
    screen._choose_the_test_data(chooser=Chose("load"))
    qtbot.waitUntil(lambda: _loaded(screen), timeout=10000)

    seen = {}

    def press_ok(dialog):
        seen["src"] = dialog._src_edit.text()
        box = dialog.findChild(QDialogButtonBox)
        box.button(QDialogButtonBox.Ok).click()
        return dialog.result()

    monkeypatch.setattr(annotate_mod._SettingsDialog, "exec", press_ok)
    before = screen._page_gen
    screen._on_open_settings()

    assert seen["src"] == str(plate)
    assert screen._settings.db_path == str(
        plate / "measurements" / "measurements.db")
    qtbot.waitUntil(lambda: screen._page_gen > before and _loaded(screen),
                    timeout=10000)


def test_settings_ok_opens_a_source_that_was_only_typed(qtbot, screen, plate,
                                                         monkeypatch):
    """OK with a source filled in and nothing open yet must open it."""
    _write_annotate_half(plate)
    screen._settings.src = str(plate)

    def press_ok(dialog):
        dialog.findChild(QDialogButtonBox).button(
            QDialogButtonBox.Ok).click()
        return dialog.result()

    monkeypatch.setattr(annotate_mod._SettingsDialog, "exec", press_ok)
    screen._on_open_settings()
    qtbot.waitUntil(lambda: _loaded(screen), timeout=10000)
    assert screen._worker is not None


def test_a_database_path_typed_as_the_source_is_read_as_its_plate(
        qtbot, screen, plate, monkeypatch):
    """The shipped settings file names the database; the form must cope."""
    _write_annotate_half(plate)
    screen._settings.src = str(plate / "measurements" / "measurements.db")

    def press_ok(dialog):
        dialog.findChild(QDialogButtonBox).button(
            QDialogButtonBox.Ok).click()
        return dialog.result()

    monkeypatch.setattr(annotate_mod._SettingsDialog, "exec", press_ok)
    screen._on_open_settings()
    assert screen._settings.src == str(plate)
    qtbot.waitUntil(lambda: _loaded(screen), timeout=10000)
