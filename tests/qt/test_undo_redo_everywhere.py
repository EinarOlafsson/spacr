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



def test_committed_organelle_edit_undoes_without_losing_another_slots_hidden_values(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    loaded = {'number_of_organelles': 2, 'organelle_mask_dim': 0,
              'organelleb_mask_dim': 1, 'organelle_diameter': 12,
              'organelleb_diameter': 23}
    screen.apply_settings_dict(loaded)
    model = screen._settings_model
    assert model.collect()['organelleb_mask_dim'] == 1
    assert model.collect()['organelleb_diameter'] == 23
    stack = model.undo_stack
    stack.clear()
    assert model.set_value_for_key('organelle_diameter', 17)
    assert model.collect()['organelle_diameter'] == 17
    assert stack.count() == 1
    stack.undo()
    assert model.collect()['organelle_diameter'] == 12
    stack.redo()
    values = model.collect()
    assert values['organelle_diameter'] == 17
    assert values['organelleb_diameter'] == 23 and values['organelleb_mask_dim'] == 1
    assert screen._settings_model is model and values['number_of_organelles'] == 2
    screen.close()


def test_app_bulk_load_rebases_the_next_real_edit_without_replacing_the_model(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    model = screen._settings_model
    screen.apply_settings_dict({'number_of_organelles': 1, 'organelle_mask_dim': 0,
                                'organelle_diameter': 31})
    assert screen._settings_model is model
    assert model.collect()['organelle_diameter'] == 31
    assert model.undo_stack.count() == 0
    assert model.set_value_for_key('organelle_diameter', 37)
    assert model.undo_stack.count() == 1
    model.undo_stack.undo()
    values = model.collect()
    assert values['organelle_diameter'] == 31 and values['organelle_mask_dim'] == 0
    assert values['number_of_organelles'] == 1
    screen.close()


def test_progressive_slot_classifier_keeps_shared_keys_and_marks_only_slots_above_the_count():
    from spacr.qt.screens.settings_model import SettingsWidgets

    settings = {'number_of_organelles': 2, 'organelle_channel': 0,
                'organelleb_channel': 1, 'organellez_diameter': 9,
                'cell_diameter': 15, 'organelle_custom_metadata': 'preserve'}
    assert SettingsWidgets._organelle_keys_beyond(0, settings) == {
        'organelle_channel', 'organelleb_channel', 'organellez_diameter',
        'organelle_custom_metadata'}
    assert SettingsWidgets._organelle_keys_beyond(1, settings) == {
        'organelleb_channel', 'organellez_diameter'}
    assert SettingsWidgets._organelle_keys_beyond(26, settings) == set()
    assert settings['number_of_organelles'] == 2 and settings['cell_diameter'] == 15


def test_incomplete_count_is_refused_for_commit_but_remains_undoable_without_losing_other_slots(qtbot):
    import pytest
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    screen.apply_settings_dict({'number_of_organelles': 2, 'organelle_mask_dim': 0,
                                'organelleb_mask_dim': 1, 'organelleb_diameter': 23})
    model = screen._settings_model
    field = model._widgets['number_of_organelles']
    stack = model.undo_stack
    stack.clear()
    existing = dict(model._widgets)
    assert field.text() == '2' and model.collect()['number_of_organelles'] == 2
    for invalid in ('-', '1.5', 'no', '703'):
        field.setText(invalid)
        with pytest.raises(ValueError):
            model._valid_committed_value('number_of_organelles')
        field.editingFinished.emit()
        assert field.text() == invalid and stack.canUndo()
        with pytest.raises(ValueError):
            model.collect()
        assert model._valid_committed_value('organelleb_diameter') == 23
        stack.undo()
        restored = model.collect()
        assert field.text() == '2' and restored['number_of_organelles'] == 2
        assert restored['organelleb_diameter'] == 23
    field.setText('2')
    field.editingFinished.emit()
    assert model.collect()['number_of_organelles'] == 2
    assert all(model._widgets[key] is value for key, value in existing.items())
    assert model.collect()['organelleb_mask_dim'] == 1
    screen.close()


def test_canceled_real_model_picker_keeps_the_field_and_model_settings(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QLineEdit
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.model_zoo_picker import ModelZooPicker

    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    field = QLineEdit('existing-checkpoint', screen)
    before = screen._settings_model.collect()
    inspected = []

    def cancel(dialog):
        assert isinstance(dialog, ModelZooPicker)
        inspected.append(dialog)
        return QDialog.Rejected

    monkeypatch.setattr(ModelZooPicker, 'exec', cancel)
    screen._choose_a_model_for('custom_model', field)
    assert len(inspected) == 1
    assert field.text() == 'existing-checkpoint'
    assert screen._settings_model.collect() == before
    screen.close()


def test_ram_guard_choices_change_only_the_requested_policy_and_recheck_after_freeing_memory(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    for choice in ('use', 'keep', 'free', 'cancel'):
        settings = {'src': 'unchanged-input', 'n_jobs': 8, 'ram_guard': True}
        calls, freed = [], []

        def plan_for(source, requested):
            calls.append((source, requested))
            return {'exceeds': not freed, 'max_safe': 2}

        accepted = screen._confirm_ram_guard(
            settings, plan_for=plan_for, ask=lambda plan: choice,
            free_ram=lambda: freed.append(True))
        assert accepted is (choice != 'cancel')
        assert settings['src'] == 'unchanged-input'
        assert settings['n_jobs'] == (2 if choice == 'use' else 8)
        assert settings['ram_guard'] is (choice != 'keep')
        assert calls == [('unchanged-input', 8)] * (2 if choice == 'free' else 1)
        assert freed == ([True] if choice == 'free' else [])
    screen.close()
