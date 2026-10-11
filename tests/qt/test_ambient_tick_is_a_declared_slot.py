"""The backdrop's frame timer reaches a slot declared on the class.

Reported 2026-10-10 from the Gate Editor: the terminal repeated
``AttributeError: Slot 'AmbientWidget::' not found.`` once per frame.

``AmbientWidget._on_tick`` was a plain method. PySide registers such a
method as a dynamic slot in a per-object meta-object kept in the Python
wrapper's ``__dict__`` (``__METAOBJECT__``), and connects the timer to its
index there (33, right after QWidget's own methods). Every tick then asks
``object->metaObject()`` again. Whenever that lookup lands on a wrapper
without the per-object meta-object, PySide falls back to the class
meta-object, which ends at 33, so index 33 has no signature and PySide
prints the empty slot name above, once per tick. spaCR is known to hand
PySide more than one wrapper for one address (see
``test_app_wide_filters_tolerate_foreign_wrappers`` and
``spacr.qt.gui_scale._child_layouts``), which is how that lookup misses.

Declaring the slot with ``@Slot()`` puts it on the class meta-object, so
there is no per-object meta-object to lose. The tests drop that per-object
meta-object deliberately: before the fix that alone reproduced the report.
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


def _lose_the_per_object_meta_object(widget):
    """What PySide sees when the metaObject() lookup misses the wrapper."""
    vars(widget).pop("__METAOBJECT__", None)


def test_the_tick_slot_lives_on_the_class(qtbot):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.ambient import AmbientWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 120)
    widget = AmbientWidget(host, theme=THEME)
    assert "__METAOBJECT__" not in vars(widget)
    _lose_the_per_object_meta_object(widget)
    meta = widget.metaObject()
    assert meta.className() == "AmbientWidget"
    assert meta.indexOfSlot("_on_tick()") >= meta.methodOffset()


def test_ticks_without_the_per_object_meta_object_print_nothing(
        qapp, qtbot, capfd, monkeypatch):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.ambient import AmbientWidget

    ticks = []
    real = AmbientWidget._follow_the_run

    def counted(self):
        ticks.append(1)
        return real(self)

    monkeypatch.setattr(AmbientWidget, "_follow_the_run", counted)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 120)
    widget = AmbientWidget(host, theme=THEME)
    _lose_the_per_object_meta_object(widget)
    capfd.readouterr()
    widget._timer.start(5)
    _pump(qapp, 150)
    widget._timer.stop()
    err = capfd.readouterr().err
    assert "not found" not in err, err
    assert ticks


def test_the_gate_editor_window_prints_no_missing_slot(qapp, qtbot, capfd):
    from spacr.qt import preferences as prefs
    from spacr.qt.app import MainWindow
    from spacr.qt.widgets.ambient import AmbientWidget

    prefs.set_ambient_theme(THEME)
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    capfd.readouterr()
    assert window.open_module("gate_editor") == "gate_editor"
    _pump(qapp, 300)
    backdrops = [w for w in qapp.allWidgets() if isinstance(w, AmbientWidget)]
    assert backdrops
    for backdrop in backdrops:
        assert "__METAOBJECT__" not in vars(backdrop)
        _lose_the_per_object_meta_object(backdrop)
        backdrop._timer.start(5)
    window.resize(1100, 760)
    _pump(qapp, 200)
    window.resize(1500, 950)
    _pump(qapp, 200)
    for backdrop in backdrops:
        backdrop._timer.stop()
    err = capfd.readouterr().err
    assert "Slot '" not in err, err
