"""The flow backdrops keep the existing one-worker, zero-work-when-hidden contract."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QDialog, QWidget

from spacr.qt.widgets import ambient

FLOW_NAMES = ("cytoplasm", "synapse", "wind", "atlas", "helix",
              "chromatin", "nebula", "silk")
FLOW_THEMES = tuple(f"flow_{name}" for name in FLOW_NAMES) + tuple(
    f"flow_{name}_mouse" for name in FLOW_NAMES)


@pytest.mark.parametrize("theme", FLOW_THEMES)
def test_flow_4k_shading_stays_within_the_existing_pixel_ceiling(theme):
    engine = ambient.make_engine(theme, "spacr", "#101418", seed=7)
    assert isinstance(engine, ambient._BufferedEngine)
    image = engine.shade(3840, 2160)
    assert image is not None
    assert image.width() * image.height() <= ambient.BUFFER_MAX_PIXELS
    assert image.width() < 3840 and image.height() < 2160
    assert engine._buffer is not None
    assert engine._buffer.bytesPerLine() * engine._buffer.height() <= \
        ambient.BUFFER_MAX_PIXELS * 4


@pytest.mark.parametrize("theme", FLOW_THEMES)
def test_each_flow_retires_its_worker_when_hidden_or_paused(qtbot, theme):
    widget = ambient.AmbientWidget(theme=theme, palette="spacr", seed=7)
    qtbot.addWidget(widget)
    widget.resize(320, 200)
    widget.show()
    qtbot.waitExposed(widget)
    assert widget.is_running()
    assert widget.shading_thread_alive()
    first = widget._producer_box[0]

    widget.hide()
    assert not widget.is_running()
    assert not widget.shading_thread_alive()
    assert not first.is_alive()
    assert widget._last_frame is None

    widget.show()
    qtbot.waitExposed(widget)
    assert widget.shading_thread_alive()
    second = widget._producer_box[0]
    assert second is not first
    widget.set_animating(False)
    assert not widget.is_running()
    assert not second.is_alive()
    assert widget._last_frame is None

    widget.set_animating(True)
    assert widget.shading_thread_alive()
    third = widget._producer_box[0]
    widget.close()
    assert not widget.is_running()
    assert not third.is_alive()


def test_switching_all_flows_keeps_one_timer_and_no_old_workers(qtbot):
    widget = ambient.AmbientWidget(theme="blobs", palette="spacr", seed=7)
    qtbot.addWidget(widget)
    widget.resize(320, 200)
    widget.show()
    qtbot.waitExposed(widget)
    timer = widget._timer
    old = widget._producer_box[0]
    for theme in FLOW_THEMES:
        widget.set_theme(theme)
        assert not old.is_alive(), theme
        assert widget.theme() == theme
        assert widget._timer is timer
        assert widget.is_running()
        assert widget.shading_thread_alive()
        old = widget._producer_box[0]
    widget.close()
    assert not old.is_alive()


def test_mouse_flow_quiets_when_minimized_and_moves_behind_a_dialog(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(400, 260)
    widget = ambient.AmbientWidget(host, theme="flow_wind_mouse",
                                   palette="spacr", seed=7)
    widget.follow_parent()
    host.show()
    qtbot.waitExposed(host)
    assert widget.shading_thread_alive()

    dialog = QDialog(host)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    before = widget.time()
    qtbot.waitUntil(lambda: widget.time() > before, timeout=2000)
    assert widget.is_running()
    dialog.close()

    host.setWindowState(Qt.WindowMinimized)
    widget.eventFilter(host, QEvent(QEvent.WindowStateChange))
    assert not widget.shading_thread_alive()
    host.setWindowState(Qt.WindowNoState)
    widget.eventFilter(host, QEvent(QEvent.WindowStateChange))
    assert widget.shading_thread_alive()
    host.close()
    assert not widget.shading_thread_alive()
