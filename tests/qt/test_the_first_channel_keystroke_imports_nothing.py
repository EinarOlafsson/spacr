"""The first keystroke in a Mask channel does not import the filter parser.

On one core shared with other work, the second keystroke into a Mask channel
took 100-170 ms, of which 85-120 ms was importing :mod:`spacr.qt.mask_engine`
(and imageio with it) on the GUI thread to compare ``object_filters``. The
binding now starts that import on a worker as it is built.
"""
from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QObject, Signal

from spacr.qt.widgets import object_grid_binding

pytestmark = pytest.mark.qt


class _Grid(QObject):
    settings_changed = Signal()


class _Panel:
    _widgets: dict = {}


def test_building_the_binding_warms_the_filter_parser_off_the_gui_thread(
        qapp, monkeypatch):
    started = []

    class Recorded:
        def __init__(self, target, name, daemon):
            started.append((target, name, daemon))

        def start(self):
            pass

    monkeypatch.delitem(sys.modules, "spacr.qt.mask_engine", raising=False)
    monkeypatch.setattr(object_grid_binding.threading, "Thread", Recorded)
    object_grid_binding.ObjectGridBinding(_Grid(), _Panel())
    assert started == [(object_grid_binding._warm_the_filter_parser,
                        "spacr-warm-filter-parser", True)]
    monkeypatch.undo()
    monkeypatch.delitem(sys.modules, "spacr.qt.mask_engine", raising=False)
    object_grid_binding._warm_the_filter_parser()
    assert "spacr.qt.mask_engine" in sys.modules


def test_a_warm_parser_starts_no_thread(qapp, monkeypatch):
    import spacr.qt.mask_engine  # noqa: F401

    started = []
    monkeypatch.setattr(object_grid_binding.threading, "Thread",
                        lambda *a, **k: started.append(k))
    object_grid_binding.ObjectGridBinding(_Grid(), _Panel())
    assert started == []
