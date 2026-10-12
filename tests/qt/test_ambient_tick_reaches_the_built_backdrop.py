"""The backdrop's frame timer calls the widget that was built, or stops.

Reported 2026-10-11 from Make Masks, once per frame::

    File "spacr/qt/widgets/ambient.py", line 10147, in _on_tick
        self._follow_the_run()
    File "spacr/qt/widgets/ambient.py", line 9600, in _follow_the_run
        if paced != self._run_paced:
    AttributeError: 'AmbientWidget' object has no attribute '_run_paced'

``_run_paced`` is set in ``__init__`` before the timer exists, so the
widget that ran ``__init__`` always has it. The tick reached a Python object
with none of that state. The timer was connected to the declared slot
``AmbientWidget._on_tick``; PySide calls such a slot on the wrapper it finds
for the receiver's C++ address at call time, not on the object that made the
connection. spaCR is known to leave more than one wrapper at one address
(``spacr.qt.gui_scale._child_layouts``), and the same lookup missing the
wrapper's ``__dict__`` was the 2026-10-10 "Slot 'AmbientWidget::' not found"
report (``test_ambient_tick_is_a_declared_slot``). Declaring the slot turned
that message into this AttributeError: the call now succeeds, on the wrong
object, and fails at its first attribute, which is ``_run_paced``.

The timer now calls a closure holding a weak reference to the built widget,
so no receiver lookup happens. If that widget's state is gone, the timer
stops instead of raising once per frame.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

THEME = "data_art_impulse_lens"


def _pump(qapp, ms):
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)


def _backdrop(qtbot, monkeypatch):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.ambient import AmbientWidget

    monkeypatch.setattr(AmbientWidget, "_start_producer", lambda self: None)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 120)
    return host, AmbientWidget(host, theme=THEME, gravity_radius=0)


def test_ticks_reach_the_widget_that_ran_init(qapp, qtbot, monkeypatch):
    from spacr.qt.widgets.ambient import AmbientWidget

    host, widget = _backdrop(qtbot, monkeypatch)
    seen = []
    real = AmbientWidget._follow_the_run

    def counted(self):
        seen.append(self)
        return real(self)

    monkeypatch.setattr(AmbientWidget, "_follow_the_run", counted)
    vars(widget).pop("__METAOBJECT__", None)
    widget._timer.start(5)
    _pump(qapp, 120)
    widget._timer.stop()
    assert seen
    assert all(caller is widget for caller in seen)


def test_a_tick_on_a_backdrop_without_its_state_stops_instead_of_raising(
        qapp, qtbot, monkeypatch):
    host, widget = _backdrop(qtbot, monkeypatch)
    timer = widget._timer
    timer.start(5)
    _pump(qapp, 40)
    assert timer.isActive()
    state = dict(vars(widget))
    vars(widget).clear()
    try:
        _pump(qapp, 80)
        assert not timer.isActive()
    finally:
        vars(widget).update(state)


def test_a_collected_backdrop_leaves_no_ticking_timer(qapp, qtbot, monkeypatch):
    import gc

    from spacr.qt.widgets.ambient import AmbientWidget

    monkeypatch.setattr(AmbientWidget, "_start_producer", lambda self: None)
    widget = AmbientWidget(theme=THEME, gravity_radius=0)
    widget._timer.start(5)
    del widget
    gc.collect()
    _pump(qapp, 60)
