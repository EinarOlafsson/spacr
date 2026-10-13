"""Undo and redo in the settings panel, the gate editor and Annotate.

Each editor records a finished edit on its undo stack and answers Ctrl+Z,
Ctrl+Shift+Z and Ctrl+Y. The mask editor keeps its own snapshot history in
``mask_engine.MaskHistory`` and is covered by its own tests.
"""
from __future__ import annotations

import sqlite3

import numpy as np
from PIL import Image
from PySide6.QtGui import QKeySequence, QUndoStack
from PySide6.QtWidgets import QSpinBox, QWidget

from spacr.qt.shortcuts import _bind_undo_keys, _record_edit


def test_a_recorded_edit_undoes_and_redoes_through_its_setter(qtbot):
    box = {"v": 2}
    stack = QUndoStack()
    assert not _record_edit(stack, "same", box.update, 1, 1)
    box["v"] = 2
    assert _record_edit(stack, "edit", lambda v: box.update(v=v), 1, 2)
    assert box["v"] == 2 and stack.count() == 1
    stack.undo()
    assert box["v"] == 1
    stack.redo()
    assert box["v"] == 2


def test_the_keys_are_bound_on_the_editor_and_its_children(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    made = _bind_undo_keys(widget, QUndoStack(widget))
    keys = {s.key().toString() for s in made}
    assert keys == {QKeySequence("Ctrl+Z").toString(),
                    QKeySequence("Ctrl+Shift+Z").toString(),
                    QKeySequence("Ctrl+Y").toString()}


def _spin_setting(model):
    for key, widget in model._built_controls():
        if isinstance(widget, QSpinBox) and widget.maximum() > widget.value():
            return key, widget
    raise AssertionError("no spin box setting on the form")


def test_every_setting_change_is_undoable(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    parent = QWidget()
    qtbot.addWidget(parent)
    model = SettingsWidgets("measure", parent=parent)
    model.build_sections()
    seen = []
    model._enable_commit_observation(seen.append)
    key, spin = _spin_setting(model)
    before = spin.value()
    model.set_value_for_key(key, before + 1)
    assert spin.value() == before + 1
    assert model.undo_stack.count() == 1
    model.undo_stack.undo()
    assert spin.value() == before
    model.undo_stack.redo()
    assert spin.value() == before + 1
    assert model.undo_stack.count() == 1
    assert set(seen) == {key}


def test_a_bulk_load_moves_the_baseline_without_a_step(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    parent = QWidget()
    qtbot.addWidget(parent)
    model = SettingsWidgets("measure", parent=parent)
    model.build_sections()
    model._enable_commit_observation(lambda _key: None)
    key, spin = _spin_setting(model)
    loaded = spin.value() + 1
    model._applying_settings = True
    model.set_value_for_key(key, loaded)
    model._applying_settings = False
    assert model.undo_stack.count() == 0
    model.set_value_for_key(key, loaded + 1)
    model.undo_stack.undo()
    assert spin.value() == loaded


def test_gate_edits_are_undoable(qtbot):
    from spacr.qt.widgets.gate_editor import GateEditorPanel
    from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    panel.set_gates(GateSet([ThresholdGate(name="g", column="a", low=1)]))
    assert len(panel.gates) == 1 and panel.undo_stack.count() == 1
    panel.undo_stack.undo()
    assert len(panel.gates) == 0
    panel.undo_stack.redo()
    assert "g" in panel.gates.to_dict().__repr__()
    assert panel.undo_stack.count() == 1


def test_an_undone_annotation_can_be_redone(qtbot, tmp_path):
    from spacr.qt.screens import annotate as annotate_mod

    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)
    paths = []
    for i in range(6):
        p = src / "data" / f"crop_{i}.png"
        Image.fromarray(np.full((24, 24, 3), i * 20, np.uint8)).save(p)
        paths.append(str(p))
    with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" VALUES (?)',
                         [(p,) for p in paths])
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows, screen._settings.grid_cols = 2, 3
    screen._settings.image_size = (24, 24)
    screen._compute_grid_dims = lambda: None
    screen._rebuild_grid()
    screen._open_source(str(src))
    qtbot.waitUntil(lambda: len(screen._page_paths) == 6, timeout=5000)
    try:
        screen._set_focus_slot(0)
        screen.handle_key("1")
        assert screen._current_value(0) == 1
        screen._kbd_undo()
        assert screen._current_value(0) is None
        screen._kbd_redo()
        assert screen._current_value(0) == 1
        screen._kbd_redo()
        assert screen._current_value(0) == 1
    finally:
        if screen._worker is not None:
            screen._worker.stop(wait=True)
