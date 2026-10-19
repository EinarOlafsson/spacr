"""Identifying a clicked curve when the run or the best/last epoch is unknown."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from spacr.qt.screens import train_compare as tcs

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = tcs.TrainCompareScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _series(**kwargs):
    base = dict(label="run-a train", run_id="gone", epoch_range=lambda: (1, 1),
                best=lambda metric: None, last=lambda metric: None)
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_a_curve_whose_run_is_no_longer_listed_is_described_without_a_path(screen):
    screen._figure.spacr_series_by_label = {"run-a train": _series()}
    clicked = []
    screen.series_clicked.connect(clicked.append)
    text = screen.identify_series("run-a train")
    assert text == "run-a train · epoch 1"
    assert screen._picked.text() == text
    assert clicked == ["run-a train"]


def test_a_curve_with_a_best_and_last_epoch_names_both_and_its_run_path(screen):
    run = SimpleNamespace(run_id="r1", path="/runs/r1")
    screen._runs = [run]
    metric = screen.selected_metric()
    screen._figure.spacr_series_by_label = {"s": _series(
        label="s", run_id="r1", epoch_range=lambda: (1, 9),
        best=lambda m: {"epoch": 4, "value": 0.5},
        last=lambda m: {"epoch": 9, "value": 0.25})}
    text = screen.identify_series("s")
    assert f"epochs 1–9" in text
    assert f"last {metric} 0.2500 @ 9" in text
    assert f"best {metric} 0.5000 @ 4" in text
    assert text.endswith("/runs/r1")


def test_an_unknown_label_clears_the_caption(screen):
    screen._picked.setText("stale")
    assert screen.identify_series("nope") == ""
    assert screen._picked.text() == ""


def test_the_read_only_cell_cannot_be_edited(qtbot):
    item = tcs._cell("loss")
    assert item.text() == "loss"
    assert not item.flags() & Qt.ItemIsEditable
