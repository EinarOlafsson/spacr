"""The PCA export confirmation appears without freezing the window.

Lesson 60 caught a 0.7 s stall after the three-file export: the scores frame
carries every measurement column of the full cell table, and the three
``to_csv`` calls ran on the GUI thread between the save dialog closing and the
confirmation being shown. The writes now run on a worker; these tests bound the
time the GUI thread spends in ``export_csv`` and the longest gap between event
loop ticks until the confirmation is on the label.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer                                # noqa: E402
from PySide6.QtWidgets import QFileDialog                        # noqa: E402

from spacr.qt.linked_selection import LinkedSelection            # noqa: E402
from spacr.qt.screens import pca as pca_screen                   # noqa: E402

ROWS = 2341
COLUMNS = 700
CALL_BUDGET_S = 0.15
GAP_BUDGET_S = 0.25


@pytest.fixture
def wide_table():
    """A cell table as wide as the lesson's: 2341 objects, 700 measurements."""
    rng = np.random.default_rng(60)
    data = {f"m{i}": rng.normal(size=ROWS) for i in range(COLUMNS)}
    data["plateID"] = ["p1"] * ROWS
    data["rowID"] = [f"r{i % 8 + 1}" for i in range(ROWS)]
    data["columnID"] = [f"c{i % 12 + 1}" for i in range(ROWS)]
    data["fieldID"] = ["f1"] * ROWS
    data["object_label"] = list(range(ROWS))
    return pd.DataFrame(data)


@pytest.fixture
def threaded_screen(qtbot):
    """A PCA screen whose reads, fits and writes run on worker threads."""
    made = pca_screen.PCAScreen(link=LinkedSelection(), threaded=True)
    qtbot.addWidget(made)
    try:
        yield made
    finally:
        made.close()


def _fitted(qtbot, screen, table):
    screen.set_frame(table)
    qtbot.waitUntil(lambda: not screen.is_busy(), timeout=60000)
    screen.pca.features.set_selected(["m0", "m1", "m2"])
    screen.pca.recompute()
    qtbot.waitUntil(lambda: screen.pca.result is not None
                    and screen.pca.result.n_features == 3
                    and not screen.is_busy(), timeout=60000)


def test_the_three_file_export_does_not_stall_the_gui_thread(
        qtbot, threaded_screen, wide_table, tmp_path, monkeypatch):
    """export_csv returns at once and the event loop keeps ticking."""
    _fitted(qtbot, threaded_screen, wide_table)
    target = tmp_path / "pca_example.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    ticks = []
    pulse = QTimer()
    pulse.setInterval(10)
    pulse.timeout.connect(lambda: ticks.append(time.perf_counter()))
    pulse.start()
    qtbot.wait(50)

    began = time.perf_counter()
    threaded_screen.export_csv()
    returned = time.perf_counter() - began
    qtbot.waitUntil(
        lambda: threaded_screen._source.text().startswith("wrote "),
        timeout=60000)
    ticks.append(time.perf_counter())
    pulse.stop()

    marks = [began] + [t for t in ticks if t >= began]
    worst = max(b - a for a, b in zip(marks, marks[1:]))
    assert returned < CALL_BUDGET_S, returned
    assert worst < GAP_BUDGET_S, worst
    assert threaded_screen._source.text() == \
        "wrote pca_example_scores / _loadings / _variance .csv"
    scores = pd.read_csv(tmp_path / "pca_example_scores.csv")
    assert len(scores) == ROWS
    assert {"PC1", "m699"} <= set(scores.columns)
    assert len(pd.read_csv(tmp_path / "pca_example_loadings.csv")) == 3


def test_the_screen_stays_busy_until_the_export_is_written(
        qtbot, threaded_screen, wide_table, tmp_path, monkeypatch):
    """Callers that wait for is_busy() find all three files on disk."""
    _fitted(qtbot, threaded_screen, wide_table)
    target = tmp_path / "busy.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))

    threaded_screen.export_csv()
    assert threaded_screen.is_busy()
    qtbot.waitUntil(lambda: not threaded_screen.is_busy(), timeout=60000)

    for part in ("scores", "loadings", "variance"):
        assert (tmp_path / f"busy_{part}.csv").is_file()
    assert threaded_screen._source.text().startswith("wrote busy_scores")


def test_a_threaded_export_to_a_missing_folder_is_reported(
        qtbot, threaded_screen, wide_table, tmp_path, monkeypatch):
    """The worker's OSError reaches the label, not a dialog."""
    _fitted(qtbot, threaded_screen, wide_table)
    nowhere = tmp_path / "no" / "such" / "pca.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(nowhere), "")))

    threaded_screen.export_csv()
    qtbot.waitUntil(lambda: not threaded_screen.is_busy(), timeout=60000)

    assert "could not write those files" in threaded_screen._source.text()
