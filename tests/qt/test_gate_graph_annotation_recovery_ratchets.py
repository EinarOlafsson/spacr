"""Real table, context-action and queued-render recovery preserve source rows."""
import sqlite3
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QInputDialog

from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate
from spacr.qt.widgets.graph_spec import GraphSpec


@pytest.fixture(autouse=True)
def calm_preferences():
    from spacr.qt import preferences
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)


def _frame():
    return pd.DataFrame({'plateID': ['p1'] * 40, 'rowID': ['r1'] * 40,
                         'columnID': ['c1'] * 40, 'fieldID': ['f1'] * 40,
                         'object_label': np.arange(40), 'area': np.arange(40),
                         'intensity': np.arange(40) * 2})


def _menu(screen, monkeypatch):
    from spacr.qt.widgets import figure_settings
    monkeypatch.setattr(figure_settings, '_exec_menu', lambda *args: None)
    canvas = screen.gates.canvas if isinstance(screen, GateEditorScreen) else screen.builder.canvas
    return figure_settings._show_canvas_menu(canvas.figure().canvas, None)


def _action(menu, text):
    return next(a for a in menu.actions() if a.text() == text)


def _accept_label(dialog):
    dialog.output_column.setEditText('phenotype')
    dialog.label.setEditText('chosen')
    dialog.apply_button.click()
    return dialog.result()


def test_gate_retained_actions_follow_current_database_csv_and_gate_state(qtbot, tmp_path, monkeypatch):
    from spacr.qt.widgets import condition_annotation_dialog as dialogs
    frame = _frame()
    db = tmp_path / 'measurements.db'
    with sqlite3.connect(db) as connection:
        frame.to_sql('cell', connection, index=False)
    csv = tmp_path / 'source.csv'
    frame.to_csv(csv, index=False)
    original_csv = csv.read_bytes()
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(str(db), table='cell')
    screen.gates.set_gates(GateSet().add(ThresholdGate(name='all', column='area', low=0)))
    screen.gates.tree.select('all')
    menu = _menu(screen, monkeypatch)
    annotate = _action(menu, 'Annotate…')
    assert annotate.isEnabled()
    monkeypatch.setattr(dialogs._SelectionAnnotationDialog, 'exec', _accept_label)
    annotate.trigger()
    assert screen._frame.phenotype.eq('chosen').all()
    # Re-open a genuine context menu now that the saved-table action is enabled.
    labeled_menu = _menu(screen, monkeypatch)
    save = _action(labeled_menu, 'Save annotated table')
    export = _action(labeled_menu, 'Export table')
    assert save.isEnabled()
    prompts = []
    monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **k: (prompts.append(True) or 'unused', False))
    save.trigger()
    assert prompts == [True] and screen._table == 'cell'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: ('', ''))
    export.trigger()
    blocked_export = tmp_path / 'directory.csv'
    blocked_export.mkdir()
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(blocked_export), ''))
    export.trigger()
    assert blocked_export.is_dir() and not list(blocked_export.iterdir())
    assert screen._source.text() and screen._frame.phenotype.eq('chosen').all()
    # An earlier enabled action must recheck the current source after a normal reload.
    screen.load_path(str(csv))
    save.trigger()
    assert prompts == [True] and screen._condition_definition is None
    screen.gates.set_gates(GateSet().add(ThresholdGate(name='all', column='area', low=0)))
    screen.gates.tree.select('all')
    annotate.trigger()
    assert screen._frame.phenotype.eq('chosen').all()
    save.trigger()
    assert prompts == [True]  # CSV has labels, but cannot become a database table.
    screen.gates.set_gates(GateSet())
    annotate.trigger()
    assert 'Select a gate' in screen._source.text()
    assert csv.read_bytes() == original_csv
    with sqlite3.connect(db) as connection:
        pd.testing.assert_frame_equal(pd.read_sql_query('SELECT * FROM cell', connection), frame)
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [('cell',)]


def test_native_gate_empty_settings_and_public_panel_detachment_recover(qtbot, monkeypatch):
    frame = _frame()
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._point_limit.setValue(3)
    assert screen.formulas.computed_frame() is None
    screen.set_frame(frame)
    menu = _menu(screen, monkeypatch)
    action = _action(menu, 'Annotate…')
    assert action.isEnabled()
    screen.gates.set_frame(None)  # documented public Optional[DataFrame] detachment
    action.trigger()
    assert screen.gates.canvas.population() is None
    assert 'Select a gate' in screen._source.text()
    pd.testing.assert_frame_equal(screen._frame, frame)
    screen.set_frame(frame)
    assert screen.gates.canvas.population() is not None
    pd.testing.assert_frame_equal(frame, _frame())


def test_gate_legacy_sampling_reads_real_physical_and_saved_derived_tables(tmp_path):
    from spacr.derived_tables import default_definition, save_definition
    frame = _frame()
    db = tmp_path / 'measurements.db'
    with sqlite3.connect(db) as connection:
        frame.to_sql('cell', connection, index=False)
    recipe = default_definition(str(db), ['cell'], base='cell', name='Derived cells')
    save_definition(str(db), recipe)
    whole = GateEditorScreen._read(str(db), 'Derived cells', 1.0, 100)
    assert whole.object_label.tolist() == list(range(40))
    sampled = GateEditorScreen._read(str(db), 'Derived cells', .5, None)
    assert sampled.object_label.tolist() == list(range(0, 40, 2))
    physical = GateEditorScreen._read(str(db), 'cell', .5, 10)
    assert 0 < len(physical) <= 10 and physical.object_label.is_unique
    assert set(physical.object_label) <= set(frame.object_label)
    with sqlite3.connect(db) as connection:
        pd.testing.assert_frame_equal(pd.read_sql_query('SELECT * FROM cell', connection), frame)


def test_graph_seed_roundtrip_retained_action_and_actual_queued_render_clear(qtbot, monkeypatch):
    from spacr.qt.widgets import condition_annotation_dialog as dialogs
    screen = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(screen)
    frame = _frame()
    screen.set_frame(frame)
    canvas = screen.builder.canvas
    spec = GraphSpec(x='area', y='intensity', point_budget=12, seed=17)
    canvas.set_spec(spec)
    first = canvas.render_data.frame.index.to_numpy(copy=True)
    canvas.set_spec(replace(spec, seed=71))
    assert not np.array_equal(first, canvas.render_data.frame.index.to_numpy())
    canvas.set_spec(spec)
    assert np.array_equal(first, canvas.render_data.frame.index.to_numpy())
    canvas.brush(-1, -1, 100, 100)
    menu = _menu(screen, monkeypatch)
    action = _action(menu, 'Annotate selection')
    assert action.isEnabled()
    monkeypatch.setattr(dialogs._SelectionAnnotationDialog, 'exec', lambda d: d.reject() or d.result())
    action.trigger()
    assert screen._condition_definition is None and 'condition' not in screen._frame
    # Use normal Qt queued delivery of the actual owned receiver; no synthetic signal.
    canvas.rendered.disconnect(screen._on_rendered)
    canvas.rendered.connect(screen._on_rendered, Qt.QueuedConnection)
    canvas.render_now()
    canvas.set_frame(None)
    qtbot.wait(1)
    action.trigger()
    assert canvas.render_data is None and canvas.selected_count() == 0
    assert not screen._to_annotate.isEnabled()
    pd.testing.assert_frame_equal(screen._annotation_base_frame, frame)
    screen.set_frame(frame)
    canvas.set_spec(spec)
    qtbot.waitUntil(lambda: canvas.render_data is not None)
    assert canvas.render_data.n_total == len(frame)
    pd.testing.assert_frame_equal(frame, _frame())


@pytest.mark.parametrize('kind', ['gate', 'graph'])
def test_caller_owned_table_change_during_label_dialog_refuses_stale_recipe(
        qtbot, monkeypatch, kind):
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt.widgets import condition_annotation_dialog as dialogs
    frame = _frame()
    expected = frame.copy(deep=True)
    screen = GateEditorScreen(threaded=False) if kind == 'gate' else GraphBuilderScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(frame)
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda parent, title, text: warnings.append((title, text)))
    if kind == 'gate':
        screen.gates.set_gates(GateSet().add(ThresholdGate(name='all', column='area', low=0)))
        screen.gates.tree.select('all')
        monkeypatch.setattr(dialogs._SelectionAnnotationDialog, 'exec', _accept_label)
        _action(_menu(screen, monkeypatch), 'Annotate…').trigger()
        assert screen._condition_definition is not None
        # Optional public formula detachment must leave the current gate data intact.
        screen.formulas.set_frame(None)
        screen._point_limit.setValue(2)
        assert screen.formulas.computed_frame() is None
        assert screen.gates.canvas.population() is not None
        frame.loc[0, 'area'] = 777
        expected.loc[0, 'area'] = 777
        screen.set_frame(frame)
        assert screen._condition_definition is None and 'phenotype' not in screen._frame
        screen.gates.set_gates(GateSet().add(ThresholdGate(name='all', column='area', low=0)))
        screen.gates.tree.select('all')
        action = _action(_menu(screen, monkeypatch), 'Annotate…')
    else:
        canvas = screen.builder.canvas
        canvas.set_spec(GraphSpec(x='area', y='intensity', point_budget=12))
        canvas.brush(-1, -1, 100, 100)
        action = _action(_menu(screen, monkeypatch), 'Annotate selection')

    def accept_then_live_source_edit(dialog):
        result = _accept_label(dialog)
        # This is the caller's real public set_frame input, not an internal field.
        frame.loc[0, 'area'] = 888
        expected.loc[0, 'area'] = 888
        return result

    monkeypatch.setattr(dialogs._SelectionAnnotationDialog, 'exec', accept_then_live_source_edit)
    assert action.isEnabled()
    action.trigger()
    assert screen._condition_definition is None
    assert 'phenotype' not in screen._frame
    pd.testing.assert_frame_equal(screen._annotation_base_frame, expected)
    pd.testing.assert_frame_equal(frame, expected)
    if kind == 'gate':
        assert 'changed' in screen._source.text().lower() and warnings == []
    else:
        assert len(warnings) == 1 and 'changed' in warnings[0][1].lower()


def test_graph_condition_editor_cancel_preserves_the_live_table(qtbot, monkeypatch):
    from spacr.qt.widgets.condition_annotation_dialog import ConditionAnnotationDialog
    screen = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(screen)
    frame = _frame()
    screen.set_frame(frame)
    screen.builder.canvas.set_spec(GraphSpec(x='area', y='intensity', point_budget=12))
    before = screen.builder.canvas.render_data.frame.copy(deep=True)
    def cancel(dialog):
        dialog.reject()
        return dialog.result()
    monkeypatch.setattr(ConditionAnnotationDialog, 'exec', cancel)
    screen.open_condition_dialog()
    assert screen._condition_definition is None
    pd.testing.assert_frame_equal(screen._annotation_base_frame, frame)
    pd.testing.assert_frame_equal(screen.builder.canvas.render_data.frame, before)


def test_controlled_missing_renderer_provider_preserves_the_live_gate_workspace(qtbot):
    """Controlled unit boundary, not a claimed native missing-canvas producer."""
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    frame = _frame()
    screen.set_frame(frame)
    panel = screen.gates
    canvas = panel.canvas
    figure = canvas.figure()
    axes = [(ax, ax.get_xlim(), ax.get_ylim()) for ax in figure.axes]
    population = canvas.population().copy(deep=True)
    source_text = screen._source.text()
    point_text = screen._point_count.text()
    context_actions = figure.canvas._spacr_context_actions
    queried = []

    class MissingRendererProvider:
        """Delegate the real owned panel while reporting its renderer unavailable."""
        @property
        def canvas(self):
            queried.append('canvas')
            return None

        def __getattr__(self, name):
            queried.append(name)
            return getattr(panel, name)

    try:
        screen.gates = MissingRendererProvider()
        screen._narrow_to_cutoffs()
    finally:
        screen.gates = panel
    assert queried == ['canvas']
    assert panel.canvas is canvas and canvas.figure() is figure
    assert [(ax, ax.get_xlim(), ax.get_ylim()) for ax in figure.axes] == axes
    assert figure.canvas._spacr_context_actions == context_actions
    pd.testing.assert_frame_equal(canvas.population(), population)
    pd.testing.assert_frame_equal(screen._frame, frame)
    pd.testing.assert_frame_equal(frame, _frame())
    assert screen._source.text() == source_text and screen._point_count.text() == point_text


def test_live_gate_row_append_rebuilds_positional_identity_when_formulas_refresh(qtbot):
    """A caller's updated table remains correctly indexed on a public formula refresh."""
    frame = _frame()
    original = frame.copy(deep=True)
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(frame)
    frame.loc[len(frame)] = ['p1', 'r1', 'c1', 'f1', 400, 99, 198]
    screen.formulas.set_formulas([])
    assert len(screen._display_order) == 41
    assert set(screen._display_order) == set(range(41))
    displayed = screen.gates.canvas.population()
    assert displayed.index.is_unique and set(displayed.index) <= set(range(41))
    pd.testing.assert_frame_equal(displayed, frame.iloc[displayed.index])
    pd.testing.assert_frame_equal(frame.iloc[:40], original)
    assert frame.iloc[40].to_dict() == {'plateID': 'p1', 'rowID': 'r1', 'columnID': 'c1',
                                     'fieldID': 'f1', 'object_label': 400, 'area': 99,
                                     'intensity': 198}
    assert 'of 41 points' in screen._point_count.text()
    pd.testing.assert_frame_equal(screen._frame, frame)
