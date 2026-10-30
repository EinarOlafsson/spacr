"""Display caps and selected labels retain original row identity."""
import copy

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QPushButton

from spacr.condition_annotations import new_definition, table_identity
from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.widgets.graph_spec import GraphSpec


def _screen(qtbot):
    screen = GraphBuilderScreen(threaded=False, link=LinkedSelection())
    qtbot.addWidget(screen)
    frame = pd.DataFrame({'x': np.arange(100), 'y': np.arange(100)}, index=[4] * 100)
    screen.set_frame(frame)
    screen.builder.set_spec(GraphSpec(x='x', y='y', point_budget=12))
    return screen, frame


def test_random_display_limits_are_nested_and_source_is_untouched(qtbot):
    screen, frame = _screen(qtbot)
    before = frame.copy(deep=True)
    canvas = screen.builder.canvas
    first = canvas.render_data.frame.index.to_numpy()
    assert len(first) == 12
    assert not np.array_equal(first, np.arange(12))
    screen._point_limit.setValue(25)
    canvas.render_now()
    second = canvas.render_data.frame.index.to_numpy()
    assert len(second) == len(set(second)) == 25
    assert np.array_equal(first, second[:12])
    screen._point_limit.setValue(7)
    canvas.render_now()
    assert np.array_equal(canvas.render_data.frame.index.to_numpy(), first[:7])
    pd.testing.assert_frame_equal(frame, before)
    assert len(screen._frame) == 100
    assert '100' in screen._point_count.text()


def test_csv_brush_annotations_use_original_positions_and_survive_chart_roundtrip(qtbot, monkeypatch, tmp_path):
    screen, frame = _screen(qtbot)
    canvas = screen.builder.canvas
    canvas.brush(-1, -1, 101, 101)
    assert len(canvas._local_selection) == 12 and canvas.selected_count() == 12
    screen._point_limit.setValue(7)
    canvas.render_now()
    selected = canvas.render_data.frame.index.to_numpy(copy=True)
    assert canvas.selected_count() == 7
    from spacr.qt.widgets import condition_annotation_dialog as dialogs

    def choose(parent, base, source, definition, positions):
        assert set(positions) == set(selected)
        result = new_definition(base, source)
        tokens = table_identity(base)[2]
        result['conditions'] = [dict(name='chosen', metadata_column='x', include='', exclude='',
                                     manual_rows=[tokens[int(i)] for i in positions])]
        return result

    monkeypatch.setattr(dialogs, '_annotate_selection', choose)
    screen._annotate_selected()
    labels = screen._frame['condition'].fillna('')
    assert set(np.flatnonzero(labels.eq('chosen'))) == set(selected)
    assert 'condition' not in frame
    csv_path = tmp_path / 'source.csv'
    frame.to_csv(csv_path, index=False)
    screen.load_path(str(csv_path))
    screen.builder.set_spec(GraphSpec(x='x', y='y', point_budget=12))
    screen.builder.canvas.brush(-1, -1, 101, 101)
    selected = screen.builder.canvas.render_data.frame.index.to_numpy(copy=True)
    screen._annotate_selected()
    chart = tmp_path / 'chart.json'
    screen.save_chart(chart)
    expected = copy.deepcopy(screen._condition_definition)
    screen.load_chart(chart)
    assert screen._condition_definition == expected
    assert screen.builder.canvas.render_data.n_shown == 12


def test_annotations_are_context_actions_not_buttons(qtbot, monkeypatch):
    screen, _frame = _screen(qtbot)
    from spacr.qt.widgets import figure_settings
    monkeypatch.setattr(figure_settings, '_exec_menu', lambda menu, position: None)
    menu = figure_settings._show_canvas_menu(screen.builder.canvas._canvas, None)
    assert not next(action for action in menu.actions() if action.text() == 'Annotate selection').isEnabled()
    canvas = screen.builder.canvas
    canvas.brush(-1, -1, 101, 101)
    menu = figure_settings._show_canvas_menu(canvas._canvas, None)
    assert next(action for action in menu.actions() if action.text() == 'Annotate selection').isEnabled()
    captions = {button.text() for button in screen.findChildren(QPushButton)}
    assert 'Annotate conditions' not in captions
    assert 'Open selection in Annotate' not in captions


def test_nonfinite_positions_do_not_count_as_displayed_points(qtbot):
    screen = GraphBuilderScreen(threaded=False, link=LinkedSelection())
    qtbot.addWidget(screen)
    frame = pd.DataFrame({'x': [1., 2., float('nan'), 4.], 'y': [1., float('inf'), 3., 4.]})
    screen.set_frame(frame)
    screen.builder.set_spec(GraphSpec(x='x', y='y', point_budget=4, roles={'x': 'continuous', 'y': 'continuous'}))
    data = screen.builder.canvas.render_data
    assert data.n_total == 4 and data.n_shown == 2
    assert set(data.frame.index) == {0, 3}
    assert len(screen._frame) == 4


@pytest.mark.parametrize('kind', ['scatter', 'histogram'])
def test_selection_retains_manual_axis_limits_and_scale(qtbot, kind):
    screen, _frame = _screen(qtbot)
    canvas = screen.builder.canvas
    screen.builder.set_spec(GraphSpec(x='x', y='y' if kind == 'scatter' else None, kind=kind, point_budget=12))
    ax = canvas.axes_at()
    ax.set_xscale('log')
    ax.set_xlim(2, 30)
    ax.set_ylim(1, 40)
    canvas.brush(3, 2, 20, 30)
    ax = canvas.axes_at()
    assert ax.get_xscale() == 'log'
    assert ax.get_xlim() == (2, 30)
    assert ax.get_ylim() == (1, 40)
