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


def _delayed_failed_probe(monkeypatch, tmp_path):
    """Route only the real graphics probe to one observable CPU-only child.

    The capability cache is per process and an earlier backdrop in this run
    may already have filled it, so it starts empty here. moderngl is an
    optional dependency: it is reported present for the probe's own lookup
    only, so the real child is started (and the child never imports it).
    """
    import importlib.util
    import subprocess
    import sys

    monkeypatch.setattr(ambient, '_FLOW_GRAPHICS_PROBE', None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name, *args, **kwargs: object() if name == 'moderngl'
                        else real_find_spec(name, *args, **kwargs))
    assert ambient._FLOW_GRAPHICS_PROBE is None
    command = (
        "import moderngl; "
        f"c=moderngl.create_context(**{ambient._graphics_context_options()!r}); "
        "r=str(c.info['GL_RENDERER']).lower(); "
        f"assert not any(x in r for x in {ambient._SOFTWARE_RENDERERS!r}); "
        "c.release(); print('hardware-gl-ok')")
    actual_popen = subprocess.Popen
    sentinel = tmp_path / 'probe-started.pid'
    child_script = (
        "import os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(3); sys.exit(1)")
    children = []

    def cpu_probe_only(args, *popen_args, **kwargs):
        """Replace only the exact graphics probe with a real CPU-only child."""
        if args == [sys.executable, '-c', command]:
            assert not children, 'only one actual hardware-probe child is expected'
            child = actual_popen([sys.executable, '-c', child_script, str(sentinel)],
                                 *popen_args, **kwargs)
            children.append(child)
            return child
        assert not any('moderngl' in str(arg) for arg in args), 'unexpected GL command was blocked'
        return actual_popen(args, *popen_args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', cpu_probe_only)
    monkeypatch.delenv('SPACR_NO_GL', raising=False)
    return children, sentinel


def test_slow_failed_hardware_probe_retires_with_the_hidden_point_atlas(qtbot, monkeypatch, tmp_path):
    """A real pending failed probe must not outlive the existing hide budget."""
    children, sentinel = _delayed_failed_probe(monkeypatch, tmp_path)
    widget = ambient.AmbientWidget(theme='data_art_point_atlas', palette='spacr', seed=7)
    qtbot.addWidget(widget)
    widget.resize(320, 200)
    producer = None
    try:
        assert widget.engine._graphics_backend == 'gpu'
        widget.show()
        qtbot.waitUntil(sentinel.exists, timeout=2000)
        assert len(children) == 1 and int(sentinel.read_text()) == children[0].pid
        assert children[0].poll() is None
        producer = widget._producer_box[0]
        assert producer.is_alive() and widget.is_running()
        widget.hide()
        assert not widget.is_running()
        assert not widget.shading_thread_alive()
        assert not producer.is_alive(), _retirement_failure(widget, producer, 'hide during real delayed failed probe')
        assert children[0].poll() is not None, 'retirement must also reap the actual probe child'
        assert ambient._FLOW_GRAPHICS_PROBE is None, 'cancellation is not a cached capability refusal'
        assert widget._last_frame is None
    finally:
        for child in children:
            child.wait(timeout=5)
        if producer is not None:
            producer.stop()
            assert not producer.is_alive()
        widget.close()



def test_waiting_probe_worker_retires_without_cancelling_the_owning_widget(qtbot, monkeypatch, tmp_path):
    """Cancelling a real lock waiter leaves the other widget's probe owned."""
    import sys
    import traceback

    children, sentinel = _delayed_failed_probe(monkeypatch, tmp_path)
    owner = ambient.AmbientWidget(theme='data_art_point_atlas', palette='spacr', seed=7)
    waiter = ambient.AmbientWidget(theme='data_art_point_atlas', palette='spacr', seed=8)
    for widget in (owner, waiter):
        qtbot.addWidget(widget)
        widget.resize(320, 200)
    first = second = None
    try:
        assert owner.engine._graphics_backend == waiter.engine._graphics_backend == 'gpu'
        owner.show()
        qtbot.waitUntil(sentinel.exists, timeout=2000)
        assert len(children) == 1 and int(sentinel.read_text()) == children[0].pid
        assert children[0].poll() is None
        first = owner._producer_box[0]
        assert first.is_alive() and ambient._FLOW_GRAPHICS_PROBE_LOCK.locked()
        waiter.show()
        second = waiter._producer_box[0]

        def waiting_at_the_real_probe_lock():
            """Observe the actual worker frame at the production lock acquisition."""
            frame = sys._current_frames().get(second._thread.ident)
            return frame is not None and any(
                entry.name == '_flow_graphics_preflight'
                and '_FLOW_GRAPHICS_PROBE_LOCK.acquire' in (entry.line or '')
                for entry in traceback.extract_stack(frame))

        qtbot.waitUntil(waiting_at_the_real_probe_lock, timeout=2000)
        assert len(children) == 1 and children[0].poll() is None
        waiter.hide()
        assert not waiter.is_running() and not waiter.shading_thread_alive()
        assert not second.is_alive(), _retirement_failure(waiter, second, 'hide while waiting for the owned probe lock')
        assert waiter._last_frame is None
        assert first.is_alive() and children[0].poll() is None
        assert ambient._FLOW_GRAPHICS_PROBE is None
        owner.hide()
        assert not owner.is_running() and not owner.shading_thread_alive()
        assert not first.is_alive(), _retirement_failure(owner, first, 'hide owning the delayed probe after waiter retirement')
        assert children[0].poll() is not None
        assert ambient._FLOW_GRAPHICS_PROBE is None
        assert owner._last_frame is None
    finally:
        for child in children:
            child.wait(timeout=5)
        for producer in (first, second):
            if producer is not None:
                producer.stop()
                assert not producer.is_alive()
        owner.close()
        waiter.close()
