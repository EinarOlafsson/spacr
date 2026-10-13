"""Editable gate volumes refuse what they cannot represent and keep what they can.

Item 672 turned analytic and view-drawn three-axis gates into closed
piecewise-linear meshes. These tests pin the refusals that guard that
conversion and the geometry checks behind it.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    BoxGate, EditableVolumeGate, Gate, GateError, RectGate, ViewGate,
    _mesh_triangle_contact, _validate_edited_polygon,
)


def _frame(points):
    return pd.DataFrame(points, columns=['x', 'y', 'z'])


def test_a_concave_outline_is_accepted_and_its_volume_keeps_membership():
    dart = ((-1., -1.), (1., 0.), (-1., 1.), (-.5, 0.))
    _validate_edited_polygon(dart)
    gate = ViewGate(name='dart', x_column='x', y_column='y', z_column='z',
                    projection=tuple(map(tuple, np.eye(4))), vertices=dart)
    mesh = EditableVolumeGate.from_gate(gate, [(-2, 2)] * 3)
    points = np.random.default_rng(672).uniform(-1.5, 1.5, (800, 3))
    witnesses = np.asarray([[-.8, 0, 0], [0, 0, .5], [.5, .1, -1]])
    rows = _frame(np.vstack([points, witnesses]))
    assert np.array_equal(mesh.mask(rows), gate.mask(rows))
    assert mesh.mask(_frame(witnesses)).tolist() == [False, True, True]


def test_a_self_crossing_outline_is_refused():
    with pytest.raises(GateError, match='cross'):
        _validate_edited_polygon(((0, 0), (2, 2), (2, 0), (0, 1)))


def test_coplanar_triangles_whose_edges_cross_are_a_contact():
    up = np.asarray([[0., 0, 0], [4, 0, 0], [2, 4, 0]])
    down = np.asarray([[0., 3, 0], [4, 3, 0], [2, -1, 0]])
    assert _mesh_triangle_contact(up, down, np.empty((0, 3)))
    apart = down + [10, 0, 0]
    assert not _mesh_triangle_contact(up, apart, np.empty((0, 3)))


def test_a_two_axis_gate_cannot_become_an_editable_volume():
    rect = RectGate(name='flat', x_column='x', y_column='y',
                    x_low=0, x_high=1, y_low=0, y_high=1)
    with pytest.raises(GateError, match='three-dimensional'):
        EditableVolumeGate.from_gate(rect)


@dataclass(frozen=True)
class _Sphere(Gate):
    """A three-axis gate type the editable volume does not know."""

    @property
    def columns(self):
        return ('x', 'y', 'z')


def test_an_unknown_three_axis_gate_is_refused_rather_than_guessed():
    with pytest.raises(GateError, match='no supported finite editable volume'):
        EditableVolumeGate.from_gate(_Sphere(name='sphere'), [(0, 1)] * 3)


def test_a_view_boundary_that_falls_behind_the_camera_is_refused():
    projection = np.eye(4)
    projection[3, 0] = 1.
    gate = ViewGate(name='view', x_column='x', y_column='y', z_column='z',
                    projection=tuple(map(tuple, projection)),
                    vertices=((0, 0), (2, 0), (0, 1)))
    original = gate.to_dict()
    with pytest.raises(GateError, match='in front of its camera'):
        EditableVolumeGate.from_gate(gate, [(0, 1)] * 3)
    assert gate.to_dict() == original
    inside = ViewGate(name='view', x_column='x', y_column='y', z_column='z',
                      projection=tuple(map(tuple, projection)),
                      vertices=((0, 0), (.5, 0), (0, .5)))
    mesh = EditableVolumeGate.from_gate(inside, [(0, 1)] * 3)
    points = np.random.default_rng(5).uniform(0, 1, (400, 3))
    assert np.array_equal(mesh.mask(_frame(points)), inside.mask(_frame(points)))


def test_box_sampling_limits_are_enforced():
    box = BoxGate.from_limits('volume', ('x', 'y', 'z'), [(0, 1)] * 3)
    with pytest.raises(GateError, match='circumference'):
        EditableVolumeGate.from_gate(box, segments=4)
