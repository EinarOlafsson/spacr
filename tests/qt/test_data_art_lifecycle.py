"""The twelve data-art backdrops respect the existing bounded worker lifecycle."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QDialog, QWidget

from spacr.qt.widgets import ambient

ART_KEYS = (
    "data_art_point_atlas",
    "data_art_tissue_facets",
    "data_art_spatial_strata",
    "data_art_molecular_helix",
    "data_art_chromatin_ribbon",
    "data_art_sequence_matrix",
    "data_art_transcript_rain",
    "data_art_regulatory_circuit",
    "data_art_genetic_advection",
    "data_art_interference",
    "data_art_morphogenesis",
    "data_art_impulse_lens",
)


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
    assert engine._buffer is not None
    assert engine._buffer.bytesPerLine() * engine._buffer.height() <= \
        ambient.BUFFER_MAX_PIXELS * 4


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
    assert not first.is_alive()
    assert widget._last_frame is None

    widget.show()
    qtbot.waitExposed(widget)
    second = widget._producer_box[0]
    assert second is not first
    widget.set_animating(False)
    assert not widget.is_running()
    assert not second.is_alive()
    assert widget._last_frame is None

    widget.set_animating(True)
    third = widget._producer_box[0]
    assert third is not second
    widget.close()
    assert not widget.is_running()
    assert not third.is_alive()


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
