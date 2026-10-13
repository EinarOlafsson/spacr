"""Saved planner inputs are validated before any working controls change."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from spacr.qt.screens import power
from spacr.sp_stats import _nested_variance_components, _plan_arrayed_design


@pytest.fixture(scope='module')
def pilot_and_plan():
    rng = np.random.default_rng(585)
    rows = []
    for replicate in range(4):
        for condition in ('control', 'treated'):
            for well in range(3):
                for field in range(3):
                    for _ in range(8):
                        rows.append({'plateID': f'p{replicate}',
                                     'prc': f'p{replicate}_{condition}_{well}',
                                     'fieldID': field, 'condition': condition,
                                     'area': 10 + rng.normal(),
                                     'count': int(rng.poisson(2)),
                                     'infected': int(rng.random() < .3)})
    pilot = pd.DataFrame(rows)
    comps = _nested_variance_components(pilot, 'area', replicate='plateID', condition='condition')
    inputs = {'effect': 2., 'power': .8, 'alpha': .05, 'paired': False,
              'costs': [20., 1., .1], 'max_replicates': 4, 'max_wells': 3,
              'max_fields': 3, 'readout': 'continuous', 'cells': None}
    designs = _plan_arrayed_design(comps, **inputs)
    plan = {'schema': 'spacr-arrayed-plan-v1',
            'pilot': {'path': 'fixture.csv', 'table': 'cell', 'columns': {
                'value': 'area', 'well': 'prc', 'field': 'fieldID',
                'replicate': 'plateID', 'condition': 'condition'}},
            'design_inputs': inputs, 'variance_components': comps,
            'designs': designs.to_dict(orient='records'), 'recommendation_index': 0,
            'simulation': {'design_index': 0, 'n_sim': 500, 'seed': 0,
                           'cells_per_field': 8, 'power': .95}, 'summary': 'Valid prior plan'}
    # Native JSON scalar types, as the actual Save action writes them.
    return pilot, json.loads(json.dumps(plan, allow_nan=False))


@pytest.fixture
def planner(qtbot, monkeypatch, tmp_path, pilot_and_plan):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: True)
    screen = power.PowerScreen(threaded=False)
    qtbot.addWidget(screen)
    source = tmp_path / 'good.json'
    source.write_text(json.dumps(pilot_and_plan[1]))
    assert screen._load_arrayed_plan(str(source))
    return screen


def _state(screen):
    return {'pilot': screen._pilot_path.text(), 'table': screen._pilot_table.text(),
            'columns': {k: c.currentText() for k, c in screen._pilot_columns.items()},
            'readout': screen._plan_readout.currentData(), 'effect': screen._plan_effect.value(),
            'power': screen._plan_power.value(), 'alpha': screen._plan_alpha.value(),
            'paired': screen._plan_paired.isChecked(),
            'costs': [c.value() for c in screen._plan_costs.values()],
            'limits': [c.value() for c in screen._plan_limits.values()],
            'rows': [[screen._plan_table.item(r, c).text() for c in range(5)]
                     for r in range(screen._plan_table.rowCount())],
            'save': screen._save_plan.isEnabled(),
            'plan': json.loads(json.dumps(screen._arrayed_plan, allow_nan=False))}


@pytest.mark.parametrize('path,value', [
    ((), []), ((), True), (('pilot',), []), (('pilot', 'path'), False),
    (('pilot', 'columns'), []), (('pilot', 'columns', 'value'), []),
    (('design_inputs',), []), (('design_inputs', 'paired'), 'false'),
    (('design_inputs', 'effect'), True), (('design_inputs', 'effect'), float('nan')),
    (('design_inputs', 'effect'), float('inf')), (('design_inputs', 'effect'), 1e13),
    (('design_inputs', 'effect'), .12345), (('design_inputs', 'power'), .4),
    (('design_inputs', 'alpha'), 0), (('design_inputs', 'costs'), [1, 2]),
    (('design_inputs', 'costs'), [1, 2, 3, 4]), (('design_inputs', 'costs'), '123'),
    (('design_inputs', 'costs'), [1, True, 3]), (('design_inputs', 'costs'), [1, -1, 3]),
    (('design_inputs', 'max_fields'), 0), (('design_inputs', 'max_fields'), 51),
    (('design_inputs', 'max_fields'), 2.5), (('design_inputs', 'max_fields'), True),
    (('design_inputs', 'readout'), []), (('design_inputs', 'cells'), 0),
    (('variance_components',), []), (('variance_components', 'estimated', 'cell'), 1),
    (('variance_components', 'cell'), -1), (('variance_components', 'cell'), None),
    (('variance_components', 'mean'), True), (('summary',), {}),
    (('designs',), {}), (('designs',), []), (('designs', 0, 'power'), 'not numeric'),
    (('designs', 0, 'power'), True), (('designs', 0, 'power'), 1.1),
    (('designs', 0, 'cost'), -1), (('designs', 0, 'replicates'), 1),
    (('designs', 0, 'replicates'), 1.5), (('designs', 0, 'fields'), 4),
    (('recommendation_index',), True), (('recommendation_index',), 999),
    (('simulation',), None), (('simulation', 'design_index'), 999),
    (('simulation', 'power'), float('-inf')), (('simulation', 'n_sim'), 0),
    (('simulation', 'seed'), -1), (('simulation', 'cells_per_field'), False),
])
def test_actual_load_button_refuses_malformed_fields_without_partial_changes(
        planner, tmp_path, monkeypatch, qtbot, pilot_and_plan, path, value):
    before = _state(planner)
    bad = copy.deepcopy(pilot_and_plan[1])
    if not path:
        bad = value
    else:
        parent = bad
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
    # An early form change would reveal validation performed too late.
    if isinstance(bad, dict) and isinstance(bad.get('pilot'), dict):
        bad['pilot']['table'] = 'should never reach the working form'
    source = tmp_path / 'malformed.json'
    source.write_text(json.dumps(bad))
    monkeypatch.setattr(power.QFileDialog, 'getOpenFileName', lambda *_a, **_k: (str(source), ''))
    qtbot.mouseClick(planner._load_plan, Qt.LeftButton)
    assert _state(planner) == before
    assert 'Could not load the plan:' in planner._plan_summary.text()


@pytest.mark.parametrize('text', ['{"schema":"spacr-arrayed-plan-v1","schema":"other"}',
                                  '{"deep":' + '[' * 2000 + '0' + ']' * 2000 + '}'])
def test_malformed_json_keeps_prior_snapshot(planner, tmp_path, text):
    before = _state(planner)
    path = tmp_path / 'malformed.json'
    path.write_text(text)
    assert planner._load_arrayed_plan(str(path)) is False
    assert _state(planner) == before


@pytest.mark.parametrize('readout,baseline,effect', [
    ('count', 0, 1), ('count', 1, -2),
    ('proportion', 0, .2), ('proportion', 1, -.2),
    ('proportion', .5, .6), ('proportion', .5, -.6),
])
def test_impossible_readout_means_preserve_working_state(
        planner, pilot_and_plan, tmp_path, readout, baseline, effect):
    before = _state(planner)
    plan = copy.deepcopy(pilot_and_plan[1])
    plan['design_inputs'].update(readout=readout, baseline=baseline, effect=effect)
    path = tmp_path / 'invalid-readout.json'
    path.write_text(json.dumps(plan))
    assert not planner._load_arrayed_plan(str(path))
    assert _state(planner) == before


@pytest.mark.parametrize('readout,column,effect', [('continuous', 'area', 2.),
                                                 ('count', 'count', 2.),
                                                 ('proportion', 'infected', .2)])
def test_actual_plan_save_load_roundtrip_and_legacy_defaults(
        planner, pilot_and_plan, tmp_path, qtbot, monkeypatch, readout, column, effect):
    csv = tmp_path / 'pilot.csv'
    pilot_and_plan[0].to_csv(csv, index=False)
    planner._pilot_path.setText(str(csv))
    planner._pilot_columns['value'].setEditText(column)
    planner._pilot_columns['condition'].setEditText('condition')
    planner._plan_readout.setCurrentIndex(planner._plan_readout.findData(readout))
    planner._plan_effect.setValue(effect)
    planner._plan_paired.setChecked(True)
    buttons = [button for button in planner._arrayed_planner.findChildren(QPushButton)
               if button.text() == 'Plan the design']
    qtbot.mouseClick(buttons[0], Qt.LeftButton)
    assert planner._arrayed_plan is not None
    assert planner._arrayed_plan['variance_components']['estimated']['replicate_condition']
    assert abs(planner._arrayed_plan['designs'][0]['power'] -
               planner._arrayed_plan['simulation']['power']) < .08
    target = tmp_path / 'plan.json'
    monkeypatch.setattr(power.QFileDialog, 'getSaveFileName', lambda *_a, **_k: (str(target), ''))
    qtbot.mouseClick(planner._save_plan, Qt.LeftButton)
    saved = json.loads(target.read_text())
    expected = _state(planner)
    planner._plan_effect.setValue(-3)
    planner._pilot_path.setText('not this pilot')
    csv.unlink()  # Loading a saved result must not need the pilot again.
    monkeypatch.setattr(power.QFileDialog, 'getOpenFileName', lambda *_a, **_k: (str(target), ''))
    qtbot.mouseClick(planner._load_plan, Qt.LeftButton)
    assert _state(planner) == expected
    qtbot.mouseClick(planner._save_plan, Qt.LeftButton)
    assert json.loads(target.read_text()) == saved
    if readout == 'continuous':
        saved['design_inputs'].pop('readout')
        saved['pilot']['columns'].pop('condition')
        saved['variance_components'].pop('replicate_condition')
        saved['variance_components']['estimated'].pop('replicate_condition')
        target.write_text(json.dumps(saved))
        assert planner._load_arrayed_plan(str(target))
        assert planner._plan_readout.currentData() == 'continuous'
        assert planner._pilot_columns['condition'].currentText() == ''


def test_actual_plan_and_simulation_use_same_whole_cell_design(
        planner, pilot_and_plan, tmp_path, qtbot):
    # Unequal field sizes yield a non-integer harmonic mean. A plan using
    # that mean and a simulation using its rounded value are different
    # experiments, even when both print the same rounded summary.
    pilot = pilot_and_plan[0]
    within_field = pilot.groupby(['prc', 'fieldID']).cumcount()
    pilot = pilot[within_field < np.where(pilot.fieldID == 0, 3, 8)]
    source = tmp_path / 'uneven.csv'
    pilot.to_csv(source, index=False)
    planner._pilot_path.setText(str(source))
    button = next(b for b in planner._arrayed_planner.findChildren(QPushButton)
                  if b.text() == 'Plan the design')
    qtbot.mouseClick(button, Qt.LeftButton)
    plan = planner._arrayed_plan
    assert plan is not None
    effective = plan['variance_components']['cells_per_field_effective']
    assert effective != round(effective)
    expected = max(1, round(effective))
    assert plan['design_inputs']['cells'] == expected
    assert plan['simulation']['cells_per_field'] == expected
    assert all(row['cells_per_field'] == expected for row in plan['designs'])
