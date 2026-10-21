"""The Annotate grid draws real crop pixels, and says so when it cannot.

Reported 2026-10-20 on Linux with 1.5.1.5: after Load test data the class
rings showed but every tile was black. The cached example folder held a
measurements database from another plate (50,009 rows naming plate4 crops),
so nothing was downloaded, every row missed its file, and each tile got a
near-black placeholder that read as a rendering failure. The Mac had the
published database and drew the cells.

These tests render the grid offscreen and read the tiles' pixels back.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QObject, Qt, Signal

from spacr.qt.screens import annotate as annotate_mod

N_CROPS = 6
ROWS, COLS = 2, 3


def _write_crops(plate: Path) -> list:
    """Bright, textured crops under ``data/``: easy to tell from blank."""
    folder = plate / "data" / "single_nucleus" / "plate1_A01" / "cell_png"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    paths = []
    for i in range(N_CROPS):
        p = folder / f"plate1_A01_1_{i}.png"
        Image.fromarray(rng.integers(90, 255, (32, 32, 3),
                                     dtype=np.uint8)).save(p)
        paths.append(str(p))
    return paths


def _write_database(plate: Path, paths) -> Path:
    database = plate / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    conn.close()
    return database


def _write_settings(plate: Path) -> None:
    (plate / "settings").mkdir(parents=True, exist_ok=True)
    (plate / "settings" / "annotate_settings.csv").write_text(
        "Key,Value\nannotation_column,infected\nimg_size,32\n",
        encoding="utf-8")


def _other_plates_paths(plate: Path) -> list:
    """Rows of a database from another plate: none of these files exist."""
    return [str(plate / "data" / "single_nucleus" / "plate4_F23" / "cell_png"
                / f"plate4_F23_13_{i}.png") for i in range(N_CROPS)]


class _Landed(QObject):
    done = Signal(object)


class _PublishedDownload:
    """Stands in for the annotate download: writes the published half."""

    def __init__(self, plate: Path):
        self.plate = plate
        self.calls = []
        self._bridge = _Landed()
        self._bridge.done.connect(lambda job: job(), Qt.QueuedConnection)
        self.threads = []

    def __call__(self, parent, dest, on_done):
        self.calls.append(Path(dest))

        def work():
            _write_database(self.plate, _write_crops(self.plate))
            _write_settings(self.plate)
            result = type("R", (), {"dataset_path": self.plate})()
            self._bridge.done.emit(lambda: on_done(result, ""))

        thread = threading.Thread(target=work, daemon=True)
        self.threads.append(thread)
        thread.start()


class _Chose:
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
def download(plate, monkeypatch):
    import spacr.qt.hf_download as hf

    fake = _PublishedDownload(plate)
    monkeypatch.setattr(hf, "download_annotate_example", fake)
    yield fake
    for thread in fake.threads:
        thread.join(timeout=10)


@pytest.fixture
def screen(qtbot):
    s = annotate_mod.AnnotateScreen()
    qtbot.addWidget(s)
    s.resize(900, 700)
    s._settings.grid_rows = ROWS
    s._settings.grid_cols = COLS
    s._compute_grid_dims = lambda: None
    s._rebuild_grid()
    s.show()
    notices = []
    s._console.append_notice = (
        lambda source, **values: notices.append((source, values)))
    s.notices = notices
    yield s
    if s._worker is not None:
        s._worker.stop(wait=True)


def _drawn(screen):
    return (len(screen._page_paths) == N_CROPS
            and all(screen._thumb_pixmaps[i] is not None
                    for i in range(N_CROPS)))


def _tile_levels(screen, slot):
    """Max-channel values sampled from the middle of a rendered tile."""
    thumb = screen._thumbs[slot]
    image = thumb.grab().toImage()
    w, h = image.width(), image.height()
    assert w > 8 and h > 8, (w, h)
    values = []
    for y in range(h // 4, 3 * h // 4, 2):
        for x in range(w // 4, 3 * w // 4, 2):
            c = image.pixelColor(x, y)
            values.append(max(c.red(), c.green(), c.blue()))
    return np.asarray(values)


def test_published_test_data_renders_visible_crop_pixels(qtbot, screen, plate,
                                                          download):
    _write_database(plate, _write_crops(plate))
    _write_settings(plate)
    screen._choose_the_test_data(chooser=_Chose("load"))
    assert download.calls == []
    qtbot.waitUntil(lambda: _drawn(screen), timeout=10000)
    for slot in range(N_CROPS):
        levels = _tile_levels(screen, slot)
        assert (levels > 60).mean() > 0.5, (slot, levels.mean())
        assert levels.std() > 10, (slot, levels.std())
    assert not [n for n in screen.notices if "missing on disk" in n[0]]


def test_a_database_from_another_plate_is_set_aside_and_refetched(
        qtbot, screen, plate, download):
    """The reported case: rings over black tiles, fixed without user action."""
    _write_crops(plate)
    stale = _write_database(plate, _other_plates_paths(plate))
    screen._choose_the_test_data(chooser=_Chose("load"))
    assert download.calls == [plate]
    kept = list(stale.parent.glob("measurements.db.unmatched-*"))
    assert len(kept) == 1, "the stale database, and any labels in it, is kept"
    qtbot.waitUntil(lambda: _drawn(screen), timeout=10000)
    for slot in range(N_CROPS):
        assert (_tile_levels(screen, slot) > 60).mean() > 0.5, slot


def test_missing_crop_files_draw_hatched_tiles_and_say_where(
        qtbot, screen, plate, caplog):
    """Moved data: a tile must never look like a dark cell."""
    _write_database(plate, _other_plates_paths(plate))
    _write_settings(plate)
    with caplog.at_level("WARNING", logger=annotate_mod.LOG.name):
        screen._use_the_test_data(plate, "load")
        qtbot.waitUntil(lambda: _drawn(screen), timeout=10000)
    for slot in range(N_CROPS):
        levels = _tile_levels(screen, slot)
        assert (levels > 60).mean() > 0.3, "a missing crop is not drawn dark"
        assert (levels < 120).mean() > 0.2 and (levels > 130).mean() > 0.2, \
            "a missing crop is hatched, not a flat tile"
    said = [values for source, values in screen.notices
            if "missing on disk" in source]
    assert len(said) == 1, screen.notices
    assert said[0]["missing"] == N_CROPS and said[0]["total"] == N_CROPS
    assert said[0]["folder"] == str(Path(_other_plates_paths(plate)[0]).parent)
    assert "missing on disk" in screen._status_label.text()
    logged = [r for r in caplog.records if "crop file missing" in r.getMessage()]
    assert len(logged) == 1


def test_an_unreadable_database_is_left_alone(tmp_path):
    folder = tmp_path / "plate1" / "measurements"
    folder.mkdir(parents=True)
    database = folder / "measurements.db"
    database.write_bytes(b"not a database at all")
    assert annotate_mod._example_database_lists_its_crops(database)
    assert annotate_mod._usable_example_database(tmp_path / "plate1")
    assert database.is_file()


def test_a_database_that_lists_nothing_is_left_alone(tmp_path):
    plate = tmp_path / "plate1"
    database = _write_database(plate, [])
    assert annotate_mod._usable_example_database(plate)
    assert database.is_file()
