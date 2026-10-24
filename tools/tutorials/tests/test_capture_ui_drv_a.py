"""Rec: a clip that spans actions, in the capture host and in the Xvfb recorder."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture_ui_drv_a as drv


class Xvfb:
    host = None

    def __init__(self):
        self.stills = []

    def __call__(self, name, **kwargs):
        self.stills.append((name, kwargs))


class Host(Xvfb):
    def __init__(self):
        super().__init__()
        self.host = SimpleNamespace(pump=SimpleNamespace(run=lambda seconds: None))
        self.events = []
        self.begin = lambda name: self.events.append(('begin', name)) or True
        self.end = lambda: self.events.append(('end',))


def settle(seconds=0):
    return None


def test_xvfb_takes_the_still_after_the_actions():
    capture, order = Xvfb(), []
    with drv.Rec(capture, 'scene', settle, desktop=True):
        order.append('actions')
        assert capture.stills == []
    assert capture.stills == [('scene', {'desktop': True})]


def test_xvfb_failure_takes_no_still():
    capture = Xvfb()
    with pytest.raises(RuntimeError):
        with drv.Rec(capture, 'scene', settle):
            raise RuntimeError('x')
    assert capture.stills == []


def test_host_clip_begins_before_and_ends_after_the_actions():
    capture = Host()
    with drv.Rec(capture, 'scene', settle):
        assert capture.events == [('begin', 'scene')]
    assert capture.events == [('begin', 'scene'), ('end',)]
    assert capture.stills == []


def test_host_clip_spans_a_callback_and_closes_once():
    capture = Host()
    rec = drv.Rec(capture, 'scene', settle).start()
    rec.stop()
    rec.stop()
    rec.abort()
    assert capture.events == [('begin', 'scene'), ('end',)]


def test_host_failure_still_ends_the_clip():
    capture = Host()
    with pytest.raises(RuntimeError):
        with drv.Rec(capture, 'scene', settle):
            raise RuntimeError('x')
    assert capture.events == [('begin', 'scene'), ('end',)]
