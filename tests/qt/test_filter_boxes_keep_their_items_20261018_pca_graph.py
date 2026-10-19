"""Typing in the PCA feature and graph column search boxes keeps the items."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt

from spacr.qt.widgets import graph_builder as gb
from spacr.qt.widgets.pca_view import FeaturePicker

TERMS = ("per", "AREA", "", "zzz", " in ", "a", "")


def _frame(n=30):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "area": rng.normal(size=n), "perimeter": rng.normal(size=n),
        "intensity": rng.normal(size=n), "plateID": rng.integers(0, 3, n),
        "circularity": rng.normal(size=n)})


def _shown(widget, names):
    return [widget._list.item(i).data(Qt.UserRole)
            for i in range(widget._list.count())
            if not widget._list.item(i).isHidden()]


@pytest.mark.parametrize("kind", ["pca", "graph"])
def test_typing_never_replaces_items(qtbot, kind):
    widget = FeaturePicker() if kind == "pca" else gb.ColumnWell()
    qtbot.addWidget(widget)
    widget.set_frame(_frame())
    names = list(widget.available() if kind == "pca" else widget.columns())
    items = [widget._list.item(i) for i in range(widget._list.count())]
    count = widget._list.count()
    assert count == len(names)
    for term in TERMS:
        widget._search.setText(term)
        needle = term.strip().lower()
        expected = [n for n in names if not needle or needle in n.lower()]
        assert widget._list.count() == count
        assert all(widget._list.item(i) is items[i] for i in range(count))
        assert _shown(widget, names) == expected
    if kind == "graph":
        widget._search.setText("per")
        assert widget.visible_columns() == ["perimeter"]


def test_ticks_survive_a_filter_round_trip(qtbot):
    picker = FeaturePicker()
    qtbot.addWidget(picker)
    picker.set_frame(_frame())
    picker.set_selected(["area", "intensity"])
    picker._search.setText("per")
    picker._search.setText("")
    assert picker.selected() == ("area", "intensity")
    states = {picker._list.item(i).data(Qt.UserRole):
              picker._list.item(i).checkState()
              for i in range(picker._list.count())}
    assert states["area"] == Qt.Checked
    assert states["perimeter"] == Qt.Unchecked
    picker._search.setText("per")
    picker.select_all()
    picker._search.setText("")
    assert set(picker.selected()) == {"area", "intensity", "perimeter"}
    picker._list.item(0).setCheckState(Qt.Unchecked)
    picker._search.setText("a")
    picker._search.setText("")
    assert "area" not in picker.selected()
