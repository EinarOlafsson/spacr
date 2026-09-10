"""Bounded independent layout/axis checks, with positive-before-corruption tests."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trellis_evidence import layout, verify_trellis, verify_rendered_axes
from spacr.qt.widgets.graph_spec import GraphSpec
from spacr.qt.widgets.trellis_spec import TrellisSpec, trellis


@pytest.fixture
def example():
    rows = [dict(plateID='p', rowID=r, columnID=c, fieldID='f1', object_label=i,
                 x=i+100*j+.25, y=.1*i+20*j+.125)
            for j, (r,c) in enumerate([('r5','c1'),('r5','c2'),('r12','c1')]) for i in range(13)]
    args = dict(x='x', y='y', facet_row='rowID', facet_col='columnID')
    result = trellis(pd.DataFrame(rows), TrellisSpec(GraphSpec(**args)))
    return rows, args, result


def test_actual_small_trellis_matches_independent_membership_and_extrema(example):
    rows, args, result = example
    p = verify_trellis(result, rows, **args)
    assert p['empty_real_panels'] == 1
    assert p['shown_points'] == 39
    assert p['panels'][0]['x_limits'] == pytest.approx([-10.35, 222.85])
    assert p['max_axis_limit_error'] < 1e-12


@pytest.mark.parametrize('mode', ['shared','free','row','col'])
def test_each_actual_scale_mode_against_its_own_source_rows(example, mode):
    rows, args, _ = example
    r = trellis(pd.DataFrame(rows), TrellisSpec(GraphSpec(**args), scale_x=mode, scale_y=mode))
    assert verify_trellis(r, rows, **args, scale_x=mode, scale_y=mode)['shown_points'] == 39


@pytest.mark.parametrize('kind', ['population','identity','channel','role'])
def test_corrupt_population_after_positive(example, kind):
    rows, args, r = example
    assert verify_trellis(r, rows, **args)['rows'] == 39
    if kind == 'population': r = replace(r, data=replace(r.data, n_total=38))
    elif kind == 'identity': r.frame.loc[0, 'object_label'] = 999
    elif kind == 'channel': r = replace(r, spec=replace(r.spec, graph=replace(r.spec.graph, x='y')))
    else: r.kinds['x'] = 'categorical'
    with pytest.raises(ValueError, match='population'):
        verify_trellis(r, rows, **args)


@pytest.mark.parametrize('kind', ['shape','label','occupied','membership','limits'])
def test_corrupt_panel_after_positive(example, kind):
    rows, args, r = example
    assert verify_trellis(r, rows, **args)['shown_points'] == 39
    if kind == 'shape': r = replace(r, shape=(1,4)); message='shape'
    else:
        p = r.panels[0]
        if kind == 'label': p=replace(p,row_level='wrong'); message='group position'
        elif kind == 'occupied': p=replace(p,occupied=False); message='group position'
        elif kind == 'membership': p=replace(p,index=p.index[::-1]); message='membership'
        else: p=replace(p,scales=replace(p.scales,x_limits=(0,1))); message='axis limits'
        r = replace(r,panels=(p,)+r.panels[1:])
    with pytest.raises(ValueError, match=message): verify_trellis(r, rows, **args)


def test_wrapping_padding_is_not_an_empty_real_group():
    rows = [dict(field=f'f{i}') for i in range(1,14)]
    shape, seats = layout(rows, None, 'field', 5)
    assert shape == (3,5)
    assert sum(p['occupied'] for p in seats) == 12
    assert len([p for p in seats if not p['occupied']]) == 3
    assert sum(len(p['index']) for p in seats) == 12
    assert seats[11]['col_level'] == 'f12'


def axes_from_proof(proof, result, rows):
    axes = {}
    for p, group in zip(proof['panels'], result.panels):
        title = ' · '.join(v for v in (p['row_level'],p['col_level']) if v is not None)
        label = (title+'  ·  ' if title else '')+f"n = {p['n']:,}"
        points = [(rows[i]['x'],rows[i]['y']) for i in group.index]
        axes[p['row'],p['col']] = SimpleNamespace(
            get_visible=lambda p=p:p['occupied'], get_xlim=lambda p=p:p['x_limits'],
            get_ylim=lambda p=p:p['y_limits'], get_title=lambda label=label:label,
            collections=[SimpleNamespace(get_offsets=lambda points=points:points)])
    return SimpleNamespace(_axes=axes)


def test_no_invented_point_in_actual_empty_group(example):
    rows, args, r = example
    proof = verify_trellis(r, rows, **args)
    canvas = axes_from_proof(proof, r, rows)
    assert verify_rendered_axes(canvas, r, rows, proof)['rendered_points_independently_checked'] == 39
    assert r.panels[-1].occupied and r.panels[-1].n == 0
    canvas._axes[1,1].collections[0].get_offsets = lambda: [(0,0)]
    with pytest.raises(ValueError, match='scatter points'):
        verify_rendered_axes(canvas, r, rows, proof)


def test_actual_wrapped_scatter_cap_and_hidden_padding():
    rows = [dict(plateID='p', rowID='r1', columnID='c1', fieldID=f'f{j}',
                 object_label=i, x=i+100*j+.25, y=.1*i+20*j+.125)
            for j in range(1,14) for i in range(13)]
    args = dict(x='x', y='y', facet_col='fieldID', wrap=5)
    r = trellis(pd.DataFrame(rows), TrellisSpec(GraphSpec(x='x', y='y', facet_col='fieldID'), wrap=5))
    proof = verify_trellis(r, rows, **args)
    assert (proof['rows'], proof['shown_points'], proof['unused_wrap_slots']) == (169, 156, 3)
    canvas = axes_from_proof(proof, r, rows)
    assert verify_rendered_axes(canvas, r, rows, proof)['rendered_points_independently_checked'] == 156
    canvas._axes[2,4].get_visible = lambda: True
    with pytest.raises(ValueError, match='visibility'):
        verify_rendered_axes(canvas, r, rows, proof)


@pytest.mark.parametrize('kind', ['visibility','limits','points','title','panel'])
def test_rendered_discrepancy_after_positive(example, kind):
    rows, args, r = example
    proof = verify_trellis(r, rows, **args)
    canvas = axes_from_proof(proof,r,rows)
    assert verify_rendered_axes(canvas,r,rows,proof)['rendered_points_independently_checked'] == 39
    ax = canvas._axes[0,0]
    if kind == 'visibility': ax.get_visible=lambda:False
    elif kind == 'limits': ax.get_xlim=lambda:(0,1)
    elif kind == 'points': ax.collections[0].get_offsets=lambda:[(0,0)]
    elif kind == 'title': ax.get_title=lambda:'n = 0'
    else: canvas._axes.pop((0,0))
    with pytest.raises(ValueError): verify_rendered_axes(canvas,r,rows,proof)
