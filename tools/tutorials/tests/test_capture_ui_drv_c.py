"""Scenes: host clips contain their actions; the Xvfb path stays a plain capture."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import capture_ui_drv_c as drv


def _capture(host):
    calls = []

    def capture(name, **kwargs):
        calls.append(('capture', name, kwargs))
    capture.host = host
    capture.begin = lambda name: calls.append(('begin', name)) or True
    capture.end = lambda: calls.append(('end',))
    capture.calls = calls
    return capture


def test_xvfb_path_is_a_plain_capture():
    capture = _capture(None)
    scenes = drv.Scenes(capture, lambda s=0: None)
    assert scenes.begin('a') is False
    scenes.end('a')
    scenes.end('b', desktop=True)
    assert capture.calls == [('capture', 'a', {}), ('capture', 'b', {'desktop': True})]


def test_host_clip_wraps_actions_and_begin_is_idempotent(monkeypatch):
    monkeypatch.setattr(drv.capture_ui, 'wait', lambda *a, **k: None)
    monkeypatch.setattr(drv.capture_ui, 'point_at', lambda *a, **k: [])
    capture = _capture(SimpleNamespace())
    scenes = drv.Scenes(capture, lambda s=0: None)
    assert scenes.begin('a') and scenes.begin('a')
    scenes.end('a')
    scenes.begin('b')
    scenes.begin('c')            # a forgotten end closes the open clip first
    scenes.end('c')
    assert capture.calls == [('begin', 'a'), ('end',), ('begin', 'b'), ('end',), ('begin', 'c'), ('end',)]


def test_host_end_without_begin_records_the_old_clip(monkeypatch):
    capture = _capture(SimpleNamespace())
    scenes = drv.Scenes(capture, lambda s=0: None)
    scenes.end('x')
    assert capture.calls == [('capture', 'x', {})]
