"""The offered data-art backdrops respect the bounded worker lifecycle."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QDialog, QWidget

from spacr.qt.widgets import ambient

ART_KEYS = (
    "data_art_point_atlas",
    "data_art_tissue_facets",
    "data_art_genetic_advection",
    "data_art_impulse_lens",
    "data_art_fungal_growth",
)


def _retirement_failure(widget, producer, phase):
    """Describe an actual failed retirement without inspecting worker locals."""
    import json
    import os
    import sys
    import traceback
    from pathlib import Path

    thread = producer._thread
    frame = sys._current_frames().get(None if thread is None else thread.ident)
    stack = [] if frame is None else [
        {"file": Path(entry.filename).name, "function": entry.name, "line": entry.lineno}
        for entry in traceback.extract_stack(frame)]
    return json.dumps({
        "phase": phase,
        "join_budget_s": ambient.PRODUCER_JOIN_S,
        "widget_hidden": widget.isHidden(),
        "widget_visible": widget.isVisible(),
        "timer_running": widget.is_running(),
        "producer_current": widget._producer_box[0] is producer,
        "thread_alive": producer.is_alive(),
        "stop_requested": producer._stop.is_set(),
        "wake_requested": producer._wake.is_set(),
        "frame_size": producer.size,
        "frames_shaded": producer.frames_shaded,
        "engine_family": getattr(producer._engine, "family", None),
        "configured_backend": getattr(producer._engine, "_graphics_backend", None),
        "spacr_no_gl": bool(os.environ.get("SPACR_NO_GL")),
        "graphics_probe": ambient._FLOW_GRAPHICS_PROBE,
        "packed_kernel_available": ambient._PACKED_SCATTER is not None,
        "packed_compiler_started": ambient._PACKED_SCATTER_STARTED,
        "packed_compiler_failed": ambient._PACKED_SCATTER_FAILED,
        "python_stack": stack,
    }, sort_keys=True)


@pytest.mark.parametrize("key", ART_KEYS)
def test_4k_data_art_shades_inside_the_existing_pixel_ceiling(key):
    engine = ambient.make_engine(key, "spacr", "#101418", seed=7)
    assert isinstance(engine, ambient._BufferedEngine)
    assert engine.buffer_size(1920, 1080) == (1920, 1080)
    image = engine.shade(3840, 2160)
    assert image is not None
    assert (image.width(), image.height()) == (1920, 1080)
    assert image.width() * image.height() <= ambient.BUFFER_MAX_PIXELS
    assert image.width() < 3840 and image.height() < 2160
    assert image.bytesPerLine() * image.height() <= ambient.BUFFER_MAX_PIXELS * 4
    if engine._buffer is not None:
        assert engine._buffer.bytesPerLine() * engine._buffer.height() <= \
            ambient.BUFFER_MAX_PIXELS * 4
    else:
        assert key == "data_art_fungal_growth"


@pytest.mark.parametrize("key", ART_KEYS)
def test_each_data_art_worker_retires_when_hidden_paused_and_closed(qtbot, key):
    widget = ambient.AmbientWidget(theme=key, palette="spacr", seed=7)
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
    assert not first.is_alive(), _retirement_failure(widget, first, "hide")
    assert widget._last_frame is None

    widget.show()
    qtbot.waitExposed(widget)
    second = widget._producer_box[0]
    assert second is not first
    widget.set_animating(False)
    assert not widget.is_running()
    assert not second.is_alive(), _retirement_failure(widget, second, "pause")
    assert widget._last_frame is None

    widget.set_animating(True)
    third = widget._producer_box[0]
    assert third is not second
    widget.close()
    assert not widget.is_running()
    assert not third.is_alive(), _retirement_failure(widget, third, "close")


def test_switching_every_data_art_keeps_one_timer_and_retires_old_workers(qtbot):
    widget = ambient.AmbientWidget(theme="blobs", palette="spacr", seed=7)
    qtbot.addWidget(widget)
    widget.resize(320, 200)
    widget.show()
    qtbot.waitExposed(widget)
    timer = widget._timer
    old = widget._producer_box[0]
    for key in ART_KEYS:
        widget.set_theme(key)
        assert not old.is_alive(), key
        assert widget.theme() == key
        assert widget._timer is timer
        assert widget.is_running()
        assert widget.shading_thread_alive()
        old = widget._producer_box[0]
    widget.close()
    assert not old.is_alive()


def test_data_art_quiets_when_minimized_and_runs_behind_a_dialog(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(400, 260)
    widget = ambient.AmbientWidget(host, theme="data_art_point_atlas",
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
