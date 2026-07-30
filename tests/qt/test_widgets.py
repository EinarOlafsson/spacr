"""Construction + basic interactions for reusable Qt widgets."""
from __future__ import annotations

from pathlib import Path

import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt.widgets.card import Card
from spacr.qt.widgets.divider import Divider
from spacr.qt.widgets.info_link import InfoLink
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


def test_info_link_is_icon_only_and_opens_its_url(qtbot, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    opened = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        lambda url: opened.append(url.toString()) or True,
    )
    link = InfoLink("https://example.test/docs")
    qtbot.addWidget(link)
    assert link.text() == ""
    assert link.icon().isNull()
    assert link._dot_diameter == 7.0
    assert link.size().width() == 14
    assert link.accessibleName() == "Open API reference"
    link.click()
    assert opened == ["https://example.test/docs"]


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


def test_section_places_an_information_icon_beside_the_label(qtbot):
    sec = Section("General")
    qtbot.addWidget(sec)
    label = QLabel("Channels")
    field = QLabel("0, 1, 2")
    info = InfoLink("https://example.test/channels")
    sec.add_row(label, field, info_widget=info)
    assert info.parentWidget().objectName() == "SettingLabelWithInfo"
    assert sec._row_widgets == [(label, field)]


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


def test_toggle_is_three_quarters_of_the_original_switch_size(qtbot):
    t = Toggle()
    qtbot.addWidget(t)
    assert (t._track_w, t._track_h, t._knob_d) == (30, 17, 12)
    assert t._track_x == 2
    assert t._minimum_knob_x() > t._track_x


def test_toggle_knob_can_be_clicked_in_both_states(qtbot):
    t = Toggle()
    qtbot.addWidget(t)
    t.resize(50, 24)
    t.show()

    QTest.mouseClick(
        t, Qt.LeftButton,
        pos=QPoint(t._minimum_knob_x() + t._knob_d // 2, t.height() // 2),
    )
    assert t.isChecked()

    QTest.mouseClick(
        t, Qt.LeftButton,
        pos=QPoint(t._maximum_knob_x() + t._knob_d // 2, t.height() // 2),
    )
    assert not t.isChecked()


def test_toggle_knob_can_be_dragged_between_states(qtbot):
    t = Toggle()
    qtbot.addWidget(t)
    t.resize(50, 24)
    t.show()
    y = t.height() // 2

    QTest.mousePress(
        t, Qt.LeftButton,
        pos=QPoint(t._minimum_knob_x() + t._knob_d // 2, y),
    )
    QTest.mouseMove(
        t,
        pos=QPoint(t._maximum_knob_x() + t._knob_d // 2, y),
    )
    QTest.mouseRelease(
        t, Qt.LeftButton,
        pos=QPoint(t._maximum_knob_x() + t._knob_d // 2, y),
    )
    assert t.isChecked()

    QTest.mousePress(
        t, Qt.LeftButton,
        pos=QPoint(t._maximum_knob_x() + t._knob_d // 2, y),
    )
    QTest.mouseMove(
        t,
        pos=QPoint(t._minimum_knob_x() + t._knob_d // 2, y),
    )
    QTest.mouseRelease(
        t, Qt.LeftButton,
        pos=QPoint(t._minimum_knob_x() + t._knob_d // 2, y),
    )
    assert not t.isChecked()


def test_all_settings_booleans_use_switches(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets
    model = SettingsWidgets("measure")
    model.build_sections()
    boolean_widgets = [
        widget for key, widget in model._widgets.items()
        if isinstance(model._defaults.get(key), bool)
    ]
    assert boolean_widgets
    assert all(isinstance(widget, Toggle) for widget in boolean_widgets)


def test_qt_boolean_controls_do_not_construct_plain_checkboxes():
    qt_root = Path(__file__).resolve().parents[2] / "spacr" / "qt"
    offenders = []
    for path in qt_root.rglob("*.py"):
        if path.name == "toggle.py":
            continue
        if "QCheckBox(" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(qt_root).as_posix())
    assert not offenders, f"plain checkbox controls remain: {offenders}"


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
