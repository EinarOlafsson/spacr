"""Construction + basic interactions for reusable Qt widgets."""
from __future__ import annotations

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt.widgets.card import Card
from spacr.qt.widgets.divider import Divider
from spacr.qt.widgets.section import Section
from spacr.qt.widgets.tile import Tile
from spacr.qt.widgets.toggle import Toggle
from spacr.qt.widgets.usage_bar import UsageBar


def test_card_title_subtitle_and_body(qtbot):
    card = Card(title="Console", subtitle="output")
    qtbot.addWidget(card)
    labels = card.findChildren(QLabel)
    texts = [lbl.text() for lbl in labels]
    assert "Console" in texts
    assert "output" in texts
    # Body layout accepts widgets.
    added = QLabel("hello")
    card.body_layout.addWidget(added)
    assert added.parent() is not None


def test_card_no_title_no_labels(qtbot):
    card = Card()
    qtbot.addWidget(card)
    assert not card.findChildren(QLabel)


def test_divider_default_horizontal(qtbot):
    d = Divider()
    qtbot.addWidget(d)
    assert d.objectName() == "Divider"


def test_section_add_row_and_widget(qtbot):
    sec = Section("General")
    qtbot.addWidget(sec)
    assert sec.title() == "GENERAL"
    w1 = QLabel("value")
    sec.add_row("Label", w1)
    w2 = QLabel("full-width")
    sec.add_widget(w2)
    # Both children are inside the section.
    assert w1.parent() is not None
    assert w2.parent() is not None


def test_tile_emits_clicked(qtbot):
    tile = Tile(text="Mask", caption="Mask")
    qtbot.addWidget(tile)
    with qtbot.waitSignal(tile.clicked, timeout=1000):
        tile._button.click()


def test_toggle_toggling(qtbot):
    t = Toggle()
    qtbot.addWidget(t)
    assert not t.isChecked()
    t.setChecked(True)
    assert t.isChecked()
    t.setChecked(False)
    assert not t.isChecked()


@pytest.mark.parametrize("pct,expected", [
    (10.0, "UsageBar"),
    (80.0, "UsageBarWarn"),
    (95.0, "UsageBarError"),
])
def test_usage_bar_thresholds(qtbot, pct, expected):
    bar = UsageBar("RAM")
    qtbot.addWidget(bar)
    bar.set_value(pct)
    assert bar._bar.objectName() == expected
    assert bar._pct.text().endswith("%")
