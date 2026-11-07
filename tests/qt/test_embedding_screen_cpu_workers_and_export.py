"""The Embeddings screen's CPU workers x threads controls.

The controls default to Auto, which resolves from the core count. The
encoder is a stand-in, the run is unthreaded and on the CPU.
"""

import os
import sqlite3

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


class _Mean:
    in_channels = 3

    def __call__(self, stack):
        return stack.mean(axis=(1, 2)).astype(np.float32)


def _database(tmp_path):
    from PIL import Image

    root = tmp_path / "screen"
    data = root / "data" / "cell_png"
    measurements = root / "measurements"
    data.mkdir(parents=True)
    measurements.mkdir(parents=True)
    rows = []
    for column in ("01", "02"):
        for label in (1, 2, 3):
            name = f"plate1_A{column}_1_{label}.png"
            Image.fromarray(np.full((16, 16, 3), 30 + label, dtype=np.uint8)
                            ).save(str(data / name))
            rows.append((str(data / name), name, "plate1", "A", column, "1",
                         f"o{label}"))
    db = measurements / "measurements.db"
    conn = sqlite3.connect(str(db), timeout=30)
    try:
        conn.execute("CREATE TABLE png_list (png_path TEXT, file_name TEXT,"
                     " plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT,"
                     " cell_id TEXT)")
        conn.executemany("INSERT INTO png_list VALUES (?,?,?,?,?,?,?)", rows)
        conn.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT,"
                     " columnID TEXT, fieldID TEXT, path_name TEXT,"
                     " file_name TEXT)")
        conn.commit()
    finally:
        conn.close()
    return str(db)


def _screen(qtbot):
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    screen = EmbeddingsScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


def test_auto_workers_and_threads_resolve_from_the_core_count(qtbot):
    from spacr.embeddings import _cpu_parallelism

    screen = _screen(qtbot)
    assert screen._workers.value() == 0 and screen._threads.value() == 0
    assert screen._workers.specialValueText() == "Auto"
    assert screen._cpu_choice() == _cpu_parallelism()
    screen._workers.setValue(3)
    screen._threads.setValue(5)
    assert screen._cpu_choice() == (3, 5)


def test_embed_passes_the_workers_and_threads_to_the_plate(qtbot, monkeypatch):
    import spacr.embeddings as emb

    asked = []

    def encode(crops, spec, *, record, workers, threads):
        asked.append((workers, threads))
        values = crops.mean(axis=(1, 2)).astype(np.float32)
        return emb.EmbeddingResult(values, ("emb_0", "emb_1", "emb_2"),
                                   spec, 3, 3)

    monkeypatch.setattr(emb, "_embed_plate", encode)
    screen = _screen(qtbot)
    screen.set_crops(np.random.default_rng(0).random((4, 8, 8, 3)))
    screen._workers.setValue(2)
    screen._threads.setValue(1)
    screen.embed()
    assert asked == [(2, 1)]
