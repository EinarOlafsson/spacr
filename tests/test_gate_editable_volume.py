"""Edited three-axis boundaries must define the same scientific volume they show."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    BoxGate, CylinderGate, EditableVolumeGate, GateError, GateSet, PrismGate,
    ViewGate, _EllipsoidGate, gate_from_dict,
)


def _box():
    return EditableVolumeGate.from_gate(BoxGate.from_limits('volume', ('x', 'y', 'z'), [(0, 1)] * 3))


def test_volume_contact_validation_observes_cancellation_between_pairs(monkeypatch):
    from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
    from spacr.qt.widgets import gate_spec

    gate = _box()
    token = CancellationToken()
    original = gate_spec._mesh_triangle_contact
    calls = []

    def contact(*args, **kwargs):
        calls.append(True)
        result = original(*args, **kwargs)
        token.cancel()
        return result

    monkeypatch.setattr(gate_spec, '_mesh_triangle_contact', contact)
    with installed_token(token), pytest.raises(PipelineCancelled):
        gate_spec._validate_volume_mesh(np.asarray(gate.vertices), np.asarray(gate.faces))
    assert len(calls) == 1


def _frame(points):
    return pd.DataFrame(points, columns=['x', 'y', 'z'])


@pytest.mark.parametrize('perspective', [False, True])
def test_view_drawn_boundary_preserves_its_projected_region_within_finite_depth(perspective):
    projection = np.eye(4)
    if perspective:
        projection[3, 2] = .15
    gate = ViewGate(name='view', x_column='x', y_column='y', z_column='z',
                    projection=tuple(map(tuple, projection)),
                    vertices=((-1, -1), (1, -1), (1, 1), (-1, 1)))
    original = gate.to_dict()
    mesh = EditableVolumeGate.from_gate(gate, [(-2, 2)] * 3)
    points = np.random.default_rng(93).uniform(-2, 2, (500, 3))
    assert np.array_equal(mesh.mask(_frame(points)), gate.mask(_frame(points)))
    assert gate.to_dict() == original
    top_group = max(mesh.surface_groups)
    edited = mesh.moved_surfaces([top_group], [0, 0, .5])
    assert edited != mesh
    restored = gate_from_dict(edited.to_dict())
    assert np.array_equal(restored.mask(_frame(points)), edited.mask(_frame(points)))


@pytest.mark.parametrize('projection,limits', [
    (np.eye(4), None),
    (np.eye(4), [(-2, 2), (0, 0), (-2, 2)]),
    (np.diag([1, 1, 0, 1]), [(-2, 2)] * 3),
    (np.diag([1, 1, 1, -1]), [(-2, 2)] * 3),
    (np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]), [(-2, 2)] * 3),
])
def test_unresolved_or_camera_crossing_view_volumes_are_refused(projection, limits):
    gate = ViewGate(name='view', x_column='x', y_column='y', z_column='z',
                    projection=tuple(map(tuple, projection)),
                    vertices=((-1, -1), (1, -1), (1, 1), (-1, 1)))
    with pytest.raises(GateError):
        EditableVolumeGate.from_gate(gate, limits)


def test_original_box_and_mesh_agree_on_points_boundaries_and_missing_values():
    box = BoxGate.from_limits('volume', ('x', 'y', 'z'), [(0, 1)] * 3)
    rng = np.random.default_rng(17)
    frame = _frame(np.vstack([rng.uniform(-.2, 1.2, (500, 3)), [[0, .5, .5], [1, 1, 1], [np.nan, .5, .5], [np.inf, .5, .5]]]))
    assert np.array_equal(_box().mask(frame), box.mask(frame))
    assert _box().mask(frame.iloc[:0]).shape == (0,)


@pytest.mark.parametrize('axis', range(3))
def test_vertex_edit_changes_all_three_data_axes_without_projection(axis):
    gate = _box()
    target = np.asarray(gate.vertices[7])
    target[axis] += .25
    edited = gate.with_anchor('anchor:7:1', target)
    assert np.array_equal(np.asarray(edited.vertices)[7], target)
    assert edited.columns == gate.columns
    assert np.array_equal(np.asarray(edited.vertices)[:7], np.asarray(gate.vertices)[:7])


def test_editing_depth_creates_a_nonuniform_volume_with_independent_membership():
    gate = _box()
    edited = gate.with_anchor('anchor:5:1', (1, 0, 2))
    points = _frame([[.9, .1, 1.5], [.1, .9, 1.5], [.5, .5, .5], [.9, .1, 1.9]])
    assert gate.mask(points).tolist() == [False, False, True, False]
    assert edited.mask(points).tolist() == [True, False, True, False]
    assert gate_from_dict(edited.to_dict()) == edited


def test_adjacent_selected_surfaces_share_one_vertex_displacement():
    gate = _box()
    groups = [1, 5, 1]
    moved = gate.moved_surfaces(groups, (.1, .1, .1))
    indices = {i for face, group in zip(gate.faces, gate.surface_groups) if group in groups for i in face}
    for i, (before, after) in enumerate(zip(gate.vertices, moved.vertices)):
        assert np.asarray(after) == pytest.approx(np.asarray(before) + (.1 if i in indices else 0))
    assert np.asarray(moved.vertices)[7] == pytest.approx([1.1, 1.1, 1.1])


def test_density_samples_edges_and_all_surfaces_without_changing_geometry():
    gate = _box()
    anchors = [gate.anchors(density) for density in (1, 2, 3, 4)]
    assert [len(a) for a in anchors] == [8, 26, 56, 98]
    for axis in range(3):
        values = sorted({round(a.point[axis], 12) for a in anchors[3]})
        assert values == [0, .25, .5, .75, 1]
    assert len({a.role for a in anchors[3]}) == len(anchors[3])
    assert gate == _box()


@pytest.mark.parametrize('role', ['anchor:4:1,5:1', 'anchor:4:1,5:1,7:1'])
def test_edge_and_surface_anchor_insertion_keeps_a_closed_shared_mesh(role):
    gate = _box()
    anchor = next(a for a in gate.anchors(6) if a.role == role)
    edited = gate.with_anchor(role, np.asarray(anchor.point) + (0, 0, .1))
    assert len(edited.vertices) == len(gate.vertices) + 1
    assert edited.vertices[-1] == pytest.approx(np.asarray(anchor.point) + (0, 0, .1))
    assert all(len(face) == 3 for face in edited.faces)
    assert edited.mask(_frame([[.5, .5, .5]])).tolist() == [True]


def test_concave_prism_preserves_excluded_notch_and_depth():
    prism = PrismGate(name='volume', u_column='x', v_column='y', axis_column='z',
                      vertices=((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)), axis_low=0, axis_high=1)
    mesh = EditableVolumeGate.from_gate(prism)
    frame = _frame([[.5, .5, .5], [1.5, .5, .5], [.5, 1.5, .5], [1.5, 1.5, .5], [.5, .5, 1.1]])
    assert mesh.mask(frame).tolist() == [True, True, True, False, False]
    assert np.array_equal(mesh.mask(frame), prism.mask(frame))


def test_clockwise_prism_has_the_same_region():
    prism = PrismGate(name='volume', u_column='x', v_column='y', axis_column='z',
                      vertices=((0, 1), (1, 1), (1, 0), (0, 0)), axis_low=0, axis_high=1)
    assert np.array_equal(EditableVolumeGate.from_gate(prism).mask(_frame([[.5, .5, .5], [2, 2, .5]])), [True, False])


def test_cylinder_has_even_depth_anchors_and_can_deform_one_depth_front():
    cylinder = CylinderGate(name='volume', u_column='x', v_column='y', axis_column='z',
                            u_radius=1, v_radius=1, axis_low=0, axis_high=1)
    mesh = EditableVolumeGate.from_gate(cylinder, segments=16, depth_steps=4)
    assert sorted(set(point[2] for point in mesh.vertices)) == [0, .25, .5, .75, 1]
    edited = mesh.with_anchor('anchor:64:1', (1, 0, 2))
    frame = _frame([[.98, 0, 1.2], [-.98, 0, 1.2], [0, 0, .5]])
    assert cylinder.mask(frame).tolist() == [False, False, True]
    assert edited.mask(frame).tolist() == [True, False, True]


def test_ellipsoid_conversion_is_closed_and_respects_column_order():
    ellipse = _EllipsoidGate(name='volume', x_column='z', y_column='x', z_column='y', x_radius=1, y_radius=2, z_radius=3)
    mesh = EditableVolumeGate.from_gate(ellipse, segments=12, depth_steps=2)
    assert mesh.columns == ('z', 'x', 'y')
    assert mesh.mask(_frame([[0, 0, 0], [2.1, 0, 0], [0, 3.1, 0], [0, 0, 1.1]])).tolist() == [True, False, False, False]


def test_edited_volume_hierarchy_and_json_roundtrip_keep_actual_point_counts():
    parent = _box().rename('parent')
    child = parent.with_anchor('anchor:5:1', (1, 0, 2)).rename('child').with_parent('parent')
    gates = GateSet([parent, child])
    frame = _frame([[.5, .5, .5], [.9, .1, 1.5], [2, 2, 2]])
    restored = GateSet.from_dict(gates.to_dict())
    assert restored.to_dict() == gates.to_dict()
    assert restored.mask(frame, 'child').tolist() == [True, False, False]


@pytest.mark.parametrize('mutation', [
    lambda g: replace(g, vertices=tuple((np.nan, *v[1:]) if i == 0 else v for i, v in enumerate(g.vertices))),
    lambda g: replace(g, faces=g.faces[:-1]),
    lambda g: replace(g, faces=g.faces + (g.faces[0],)),
    lambda g: replace(g, faces=((0, 0, 1),) + g.faces[1:]),
    lambda g: replace(g, vertices=(g.vertices[1],) + g.vertices[1:]),
    lambda g: g.with_anchor('anchor:5:1', (-1, 1, -.5)),
    lambda g: g.with_anchor('anchor:99:1', (0, 0, 0)),
    lambda g: g.moved_vertices([1], (0, np.inf, 0)),
    lambda g: g.moved_surfaces([99], (1, 0, 0)),
])
def test_invalid_deformation_is_refused_and_original_remains_usable(mutation):
    gate = _box()
    with pytest.raises(GateError):
        mutation(gate)
    assert gate.mask(_frame([[.5, .5, .5]])).tolist() == [True]


def test_unbounded_volume_requires_explicit_bounds_and_does_not_guess():
    gate = BoxGate(name='volume', x_column='x', y_column='y', z_column='z')
    with pytest.raises(GateError):
        EditableVolumeGate.from_gate(gate)
    assert EditableVolumeGate.from_gate(gate, [(0, 1)] * 3) == _box()


def test_anisotropic_axis_units_do_not_change_membership():
    box = BoxGate.from_limits('volume', ('x', 'y', 'z'), [(1e8, 1e8 + 2), (0, 1e-6), (-1e4, 1e4)])
    mesh = EditableVolumeGate.from_gate(box)
    frame = _frame([[1e8 + 1, .5e-6, 0], [1e8 + 3, .5e-6, 0], [1e8 + 1, .5e-6, 2e4]])
    assert np.array_equal(mesh.mask(frame), box.mask(frame))


@pytest.mark.parametrize('second, common, expected', [
    ([[0, 0, 0], [1, 0, 0], [1, -1, 0]], [[0, 0, 0], [1, 0, 0]], False),
    ([[.5, 0, 0], [.6, -.2, 0], [.4, -.2, 0]], [], True),
    ([[.1, .1, 0], [.2, .1, 0], [.1, .2, 0]], [], True),
    ([[2, 2, 0], [3, 2, 0], [2, 3, 0]], [], False),
    ([[.2, .2, -1], [.2, .2, 1], [.8, .2, 1]], [], True),
    ([[2, 2, -1], [2, 2, 1], [3, 2, 1]], [], False),
])
def test_surface_contact_distinguishes_shared_edges_and_actual_crossings(second, common, expected):
    from spacr.qt.widgets.gate_spec import _mesh_triangle_contact

    first = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    assert _mesh_triangle_contact(first, np.asarray(second, float), np.asarray(common, float).reshape(-1, 3)) is expected


def _legacy_mesh_triangle_contact(first, second, common):
    tolerance = 1e-10
    n1 = np.cross(first[1] - first[0], first[2] - first[0])
    n2 = np.cross(second[1] - second[0], second[2] - second[0])
    coplanar = np.max(np.abs((second - first[0]) @ n1)) <= tolerance * np.linalg.norm(n1)
    if coplanar:
        axis = int(np.argmax(np.abs(n1)))
        a, b = np.delete(first, axis, axis=1), np.delete(second, axis, axis=1)
        def cross(u, v):
            return float(u[0] * v[1] - u[1] * v[0])
        common_flat = np.delete(np.asarray(common).reshape(-1, 3), axis, axis=1)
        for left, right in ((a, b), (b, a)):
            for point in left:
                if any(np.linalg.norm(point - vertex) <= tolerance for vertex in common_flat):
                    continue
                signs = [cross(right[(i + 1) % 3] - right[i], point - right[i]) for i in range(3)]
                if min(signs) >= -tolerance or max(signs) <= tolerance:
                    return True
        for i in range(3):
            for j in range(3):
                p, r = a[i], a[(i + 1) % 3] - a[i]
                q, s = b[j], b[(j + 1) % 3] - b[j]
                denominator = cross(r, s)
                if abs(denominator) > tolerance:
                    t, u = cross(q - p, s) / denominator, cross(q - p, r) / denominator
                    if tolerance < t < 1 - tolerance and tolerance < u < 1 - tolerance:
                        return True
        return False
    for triangle, other, normal in ((first, second, n2), (second, first, n1)):
        for i in range(3):
            start, end = triangle[i], triangle[(i + 1) % 3]
            denominator = float((end - start) @ normal)
            if abs(denominator) <= tolerance * np.linalg.norm(normal):
                continue
            t = float((other[0] - start) @ normal) / denominator
            if t < -tolerance or t > 1 + tolerance:
                continue
            point = start + t * (end - start)
            if any(np.linalg.norm(point - vertex) <= tolerance for vertex in common):
                continue
            u, v, w = other[1] - other[0], other[2] - other[0], point - other[0]
            uu, uv, vv = float(u @ u), float(u @ v), float(v @ v)
            wu, wv = float(w @ u), float(w @ v)
            determinant = float(np.cross(u, v) @ np.cross(u, v))
            s, r = (vv * wu - uv * wv) / determinant, (uu * wv - uv * wu) / determinant
            if s >= -tolerance and r >= -tolerance and s + r <= 1 + tolerance:
                return True
    return False


def _scalar_triangle_basis(triangle):
    u, v = triangle[1] - triangle[0], triangle[2] - triangle[0]
    normal = np.cross(u, v)
    return (normal, np.linalg.norm(normal), u, v,
            float(u @ u), float(u @ v), float(v @ v), float(normal @ normal))


@pytest.mark.parametrize('scale', [1e-3, 1., 1e3])
@pytest.mark.parametrize('cached', [False, True])
def test_surface_contact_arithmetic_matches_the_original_predicate(scale, cached):
    from spacr.qt.widgets.gate_spec import _mesh_triangle_contact

    rng = np.random.default_rng(673)
    cases = []
    for _ in range(48):
        first, second = rng.normal(size=(2, 3, 3))
        cases.append((first, second, np.empty((0, 3))))
        shared = second.copy()
        shared[0] = first[0]
        cases.append((first, shared.copy(), first[:1]))
        shared[1] = first[1]
        cases.append((first, shared.copy(), first[:2]))
    first = np.asarray([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    for delta in (0., -2e-10, -.5e-10, .5e-10, 2e-10):
        cases.append((first, np.asarray([[.2, .2, delta], [.8, .2, delta], [.2, .8, delta]]), np.empty((0, 3))))
        cases.append((first, np.asarray([[1 + delta, 0, -1.], [1 + delta, 0, 1.], [1 + delta, 1, 0.]]), np.empty((0, 3))))
    cases.append((first, np.asarray([[0., 0, 0], [1, 0, 0], [1, -1, 0]]), first[:2]))
    for first, second, common in cases:
        for left, right in ((first, second), (second, first)):
            left, right, shared = left * scale, right * scale, common * scale
            basis = (_scalar_triangle_basis(left), _scalar_triangle_basis(right)) if cached else None
            assert _mesh_triangle_contact(left, right, shared, _basis=basis) is _legacy_mesh_triangle_contact(left, right, shared)


def test_surface_contact_computes_each_triangle_normal_only_once(monkeypatch):
    from spacr.qt.widgets import gate_spec

    first = np.asarray([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    second = np.asarray([[.2, .2, -1.], [.2, .2, 1.], [.8, .2, 1.]])
    original = np.cross
    calls = []
    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)
    monkeypatch.setattr(gate_spec.np, 'cross', counted)
    assert gate_spec._mesh_triangle_contact(first, second, np.empty((0, 3)))
    assert len(calls) == 2


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('coordinates', [
    (0., 0., 0.), (.5e-10, 0., 0.), (1e-10, 0., 0.),
    (.8e-10, .8e-10, 0.), (99e-10, 0., 0.), (100e-10, 0., 0.),
    (101e-10, 0., 0.), (0., 0., 101e-10),
    (np.nan, 0., 0.), (np.inf, 0., 0.), (np.nan, np.inf, 0.),
])
def test_shared_vertex_distance_matches_original_norm(dimension, coordinates):
    from spacr.qt.widgets.gate_spec import _mesh_shared_vertex

    point = np.asarray(coordinates[:dimension])
    vertex = np.zeros(dimension)
    assert bool(_mesh_shared_vertex(point, vertex, 1e-10)) == bool(np.linalg.norm(point - vertex) <= 1e-10)


def test_shared_vertex_generic_distance_includes_every_coordinate():
    from math import dist
    from spacr.qt.widgets.gate_spec import _mesh_shared_vertex

    vertex = np.asarray([1., 2., 3., 4.])
    point = vertex + np.asarray([.25, -.5, .75, -1.])
    distance = dist(point, vertex)
    assert not _mesh_shared_vertex(point, vertex, distance - .01)
    assert _mesh_shared_vertex(point, vertex, distance + .01)
    assert _mesh_shared_vertex(vertex.copy(), vertex, 0.)


@pytest.mark.parametrize('dimension', [2, 3])
def test_shared_vertex_distance_keeps_norm_for_nearby_points_only(monkeypatch, dimension):
    from spacr.qt.widgets import gate_spec

    original = np.linalg.norm
    calls = []
    def norm(delta):
        calls.append(delta.copy())
        return original(delta)
    monkeypatch.setattr(gate_spec.np.linalg, 'norm', norm)
    vertex = np.zeros(dimension)
    assert gate_spec._mesh_shared_vertex(vertex.copy(), vertex, 1e-10)
    point = vertex.copy()
    point[-1] = 101e-10
    assert not gate_spec._mesh_shared_vertex(point, vertex, 1e-10)
    point[-1] = .5e-10
    assert gate_spec._mesh_shared_vertex(point, vertex, 1e-10)
    assert len(calls) == 1 and np.array_equal(calls[0], point - vertex)


def test_bulk_triangle_normals_match_scalar_cross_operands_exactly():
    rng = np.random.default_rng(673)
    triangles = rng.uniform(0, 1, (256, 3, 3))
    triangles[:2] = [[[0., -0., 0.], [1., 0., 0.], [0., 1., 0.]],
                     [[0., 0., -0.], [0., 1., 0.], [0., 0., 1.]]]
    bulk = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    scalar = np.asarray([np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                         for triangle in triangles])
    assert bulk.dtype == scalar.dtype and bulk.tobytes() == scalar.tobytes()


def test_volume_validator_reuses_validated_normals_and_caches_exact_scalar_bases(monkeypatch):
    from spacr.qt.widgets import gate_spec

    gate = _box()
    vertices, faces = np.asarray(gate.vertices), np.asarray(gate.faces)
    original = np.cross
    shapes = []
    def counted(first, second, **kwargs):
        shapes.append(np.asarray(first).shape)
        return original(first, second, **kwargs)
    monkeypatch.setattr(gate_spec.np, 'cross', counted)
    gate_spec._validate_volume_mesh(vertices, faces)
    assert shapes == [(len(faces), 3)]


@pytest.mark.parametrize('jitter', [0., .5e-10, 1e-10, 2e-10])
@pytest.mark.parametrize('reorder', [False, True])
def test_mesh_prefilter_keeps_all_original_candidates_and_their_order(monkeypatch, jitter, reorder):
    from spacr.qt.widgets import gate_spec

    gate = _box()
    vertices, faces = np.asarray(gate.vertices), np.asarray(gate.faces)
    vertices[:, 0] += jitter * np.asarray([1, -1, 2, -2, 3, -3, 4, -4])
    if reorder:
        faces = faces[np.random.default_rng(673).permutation(len(faces))]
    points = (vertices - vertices.min(axis=0)) / np.ptp(vertices, axis=0)
    triangles = points[faces]
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    expected = []
    for i in range(len(triangles)):
        candidates = np.flatnonzero(np.all(high[i] >= low[i + 1:] - 1e-10, axis=1)
                                    & np.all(high[i + 1:] >= low[i] - 1e-10, axis=1)) + i + 1
        expected.extend((i, int(j)) for j in candidates)
    indices = {triangle.tobytes(): index for index, triangle in enumerate(triangles)}
    seen = []
    def record_contact(first, second, common, _basis=None):
        assert _basis is not None
        for triangle, basis in zip((first, second), _basis):
            expected_basis = _scalar_triangle_basis(triangle)
            assert all(np.array_equal(actual, expected) for actual, expected in zip(basis, expected_basis))
        seen.append((indices[first.tobytes()], indices[second.tobytes()]))
        return False
    monkeypatch.setattr(gate_spec, '_mesh_triangle_contact', record_contact)
    gate_spec._validate_volume_mesh(vertices, faces)
    assert seen == expected
    assert seen == sorted(seen)


def test_disconnected_closed_shells_are_not_accepted_as_one_volume():
    gate = _box()
    points = gate.vertices + tuple(tuple(np.asarray(v) + (3, 0, 0)) for v in gate.vertices)
    faces = gate.faces + tuple(tuple(i + 8 for i in face) for face in gate.faces)
    with pytest.raises(GateError):
        replace(gate, vertices=points, faces=faces, surface_groups=())


@pytest.mark.parametrize('density', [0, 9])
def test_invalid_anchor_density_does_not_change_the_volume(density):
    with pytest.raises(GateError):
        _box().anchors(density)


@pytest.mark.parametrize('indices', [[-1], [99], [1.2], ['unavailable']])
def test_invalid_vertex_selection_does_not_silently_move_another_vertex(indices):
    with pytest.raises(GateError):
        _box().moved_vertices(indices, (0, 0, .1))


def test_clockwise_concave_cap_survives_very_different_measurement_units():
    footprint = [(1e6 * x, 1e-8 * y) for x, y in reversed(((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)))]
    prism = PrismGate(name='volume', u_column='x', v_column='y', axis_column='z', vertices=tuple(footprint), axis_low=0, axis_high=1)
    mesh = EditableVolumeGate.from_gate(prism)
    points = _frame([[.5e6, .5e-8, .5], [1.5e6, 1.5e-8, .5]])
    assert mesh.mask(points).tolist() == [True, False]


@pytest.mark.parametrize('vertices', [
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (0, np.nan)),
    ((0, 0), (1, 0), (0, 0)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (1, 1), (0, 1), (1, 0)),
])
def test_invalid_planar_edit_is_refused_before_creating_a_gate(vertices):
    from spacr.qt.widgets.gate_spec import _validate_edited_polygon

    with pytest.raises(GateError):
        _validate_edited_polygon(vertices)


def test_valid_concave_outline_and_triangular_cap_remain_valid():
    from spacr.qt.widgets.gate_spec import _triangulate_volume_cap, _validate_edited_polygon

    outline = ((0, 0), (2, 0), (1, .7), (2, 2), (0, 2))
    _validate_edited_polygon(outline)
    assert len(_triangulate_volume_cap(outline)) == 3
    assert _triangulate_volume_cap(((0, 0), (1, 0), (0, 1))) == [(0, 1, 2)]


@pytest.mark.parametrize('outline', [((0, 0), (1, 0), (2, 0)),
                                   ((0, 0), (0, 1), (1, 1), (1, 0)),
                                   ((0, 0), (1, 0), (0, np.inf))])
def test_untriangulatable_caps_are_refused(outline):
    from spacr.qt.widgets.gate_spec import _triangulate_volume_cap

    with pytest.raises(GateError):
        _triangulate_volume_cap(outline)


@pytest.mark.parametrize('mutation', [
    lambda g: replace(g, x_column=''),
    lambda g: replace(g, z_column='x'),
    lambda g: replace(g, vertices=((0, 0),) * 8),
    lambda g: replace(g, faces=((0.5, 1, 2),)),
    lambda g: replace(g, faces=()),
    lambda g: replace(g, faces=((-1, 1, 2),)),
    lambda g: replace(g, surface_groups=(0,)),
    lambda g: replace(g, surface_groups=(-1,) * len(g.faces)),
    lambda g: replace(g, vertices=g.vertices[:3]),
    lambda g: replace(g, vertices=tuple((i, i * i, 0) for i in range(8))),
    lambda g: replace(g, vertices=tuple((i, i * i, i * i * i) for i in range(8193))),
    lambda g: replace(g, faces=g.faces * 1400),
])
def test_malformed_or_over_budget_meshes_do_not_enter_the_gate_set(mutation):
    original = _box()
    with pytest.raises(GateError):
        mutation(original)
    assert original.mask(_frame([[.5, .5, .5]])).tolist() == [True]


def test_closed_flat_tetrahedron_is_refused_even_when_all_axis_extents_are_positive():
    with pytest.raises(GateError, match='nonzero region'):
        EditableVolumeGate(name='flat', x_column='x', y_column='y', z_column='z',
                           vertices=((0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 2)),
                           faces=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))


def test_whole_volume_translation_and_scaling_preserve_shared_topology():
    gate = _box()
    assert gate.centre() == (.5, .5)
    moved = gate.translated(2, -3)
    assert np.asarray(moved.vertices) == pytest.approx(np.asarray(gate.vertices) + [2, -3, 0])
    scaled = gate.scaled(2)
    assert np.asarray(scaled.vertices) == pytest.approx(.5 + (np.asarray(gate.vertices) - .5) * 2)
    anchored = gate.scaled(2, about=(0, 0))
    assert np.asarray(anchored.vertices) == pytest.approx([0, 0, .5] + (np.asarray(gate.vertices) - [0, 0, .5]) * 2)
    assert moved.faces == scaled.faces == anchored.faces == gate.faces
    assert EditableVolumeGate.from_gate(gate) is gate


@pytest.mark.parametrize('factor', [0, -1, np.inf, np.nan])
def test_invalid_scale_cannot_invert_or_corrupt_a_volume(factor):
    with pytest.raises(GateError):
        _box().scaled(factor)


@pytest.mark.parametrize('role,point', [('vertex:0', (0, 0, 0)), ('anchor:0:1', (0, 0)),
                                     ('anchor:0:1', (0, np.inf, 0)), ('anchor:0:1,7:1', (0, 0, 0))])
def test_stale_or_nonfinite_anchor_cannot_split_unrelated_surfaces(role, point):
    with pytest.raises(GateError):
        _box().with_anchor(role, point)


@pytest.mark.parametrize('gate,arguments', [
    (BoxGate.from_limits('volume', ('x', 'y', 'z'), [(0, 1)] * 3), {'segments': 7}),
    (BoxGate.from_limits('volume', ('x', 'y', 'z'), [(0, 1)] * 3), {'depth_steps': 17}),
    (PrismGate(name='bad', u_column='x', v_column='y', axis_column='z', vertices=((0, 0), (1, 0), (2, 0)), axis_low=0, axis_high=1), {}),
])
def test_unusable_sampling_and_footprints_are_explicitly_refused(gate, arguments):
    with pytest.raises(GateError):
        EditableVolumeGate.from_gate(gate, **arguments)


def test_a_flat_gate_has_no_invented_three_dimensional_extent():
    from spacr.qt.widgets.gate_spec import RectGate

    with pytest.raises(GateError):
        EditableVolumeGate.from_gate(RectGate(name='flat', x_column='x', y_column='y'))


def test_view_boundary_refuses_a_singular_camera_and_normalizes_clockwise_outline():
    projection = np.eye(4)
    gate = ViewGate(name='view', x_column='x', y_column='y', z_column='z', projection=tuple(map(tuple, projection)),
                    vertices=((-1, -1), (-1, 1), (1, 1), (1, -1)))
    assert EditableVolumeGate.from_gate(gate, [(-2, 2)] * 3).mask(_frame([[0, 0, 0]])).tolist() == [True]
    projection[1] = projection[0]
    with pytest.raises(GateError, match='camera cannot resolve'):
        EditableVolumeGate.from_gate(replace(gate, projection=tuple(map(tuple, projection))), [(-2, 2)] * 3)
