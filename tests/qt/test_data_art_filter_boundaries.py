"""Manual gravity-filter and custom palette boundaries for data art."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QWidget

from spacr.qt.widgets import ambient


def test_manual_stop_releases_gravity_click_filter_until_restart(qtbot):
    """Pausing a visible lens stops collecting clicks without blocking controls."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(360, 240)
    backdrop = ambient.install_ambient(host, theme="data_art_impulse_lens", seed=7)
    button = QPushButton("Run", host)
    button.setGeometry(140, 90, 80, 40)
    clicks = []
    button.clicked.connect(lambda: clicks.append(True))
    host.show()
    qtbot.waitExposed(host)
    assert backdrop._interaction_app is not None

    backdrop.stop()
    assert not backdrop.is_running()
    assert backdrop._interaction_app is None
    qtbot.mouseClick(button, Qt.LeftButton)
    assert clicks == [True]
    assert not backdrop._pending_art_impulses

    backdrop.start()
    assert backdrop.is_running()
    assert backdrop._interaction_app is not None
    before = len(backdrop.engine._gravity_impulses)
    qtbot.mouseClick(button, Qt.LeftButton)
    assert clicks == [True, True]
    qtbot.waitUntil(lambda: any(
        strength == 1.0 for _stamp, _origin, strength
        in backdrop.engine._gravity_impulses[before:]))

    backdrop.stop()
    assert backdrop._interaction_app is None
    assert not backdrop._pending_art_impulses
    host.close()


def test_gravity_filter_ignores_outside_clicks_and_deduplicates_one_point(qtbot):
    """Only a hit in the backdrop queues one impulse before the next tick."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(360, 240)
    backdrop = ambient.install_ambient(host, theme="data_art_impulse_lens", seed=7)
    inside = QPushButton("Inside", host)
    inside.setGeometry(20, 20, 60, 30)
    outside = QPushButton("Outside", host)
    outside.setGeometry(180, 90, 80, 30)
    host.show()
    qtbot.waitExposed(host)
    backdrop.setGeometry(0, 0, 100, 100)
    backdrop._timer.setInterval(100_000)
    backdrop._pending_art_impulses.clear()
    assert backdrop._interaction_app is not None

    qtbot.mouseClick(outside, Qt.LeftButton)
    assert not backdrop._pending_art_impulses
    qtbot.mouseClick(inside, Qt.LeftButton)
    qtbot.mouseClick(inside, Qt.LeftButton)
    assert len(backdrop._pending_art_impulses) == 1
    backdrop.stop()
    host.close()
