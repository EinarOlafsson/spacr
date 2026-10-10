"""Native volume edits preserve the same physical boundary and label voxels."""
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.widgets.gate_spec import BoxGate, EditableVolumeGate, GateError
from spacr.tiff_io import write_tiff


def _box(limits, name='900'):
    return EditableVolumeGate.from_gate(BoxGate.from_limits(name, ('X', 'Y', 'Z'), limits))


def _source(tmp_path, axes='ZYX', spacing=(2., .5, .25)):
    image = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)
    write_tiff(tmp_path / 'source.tif', image, metadata={'axes': axes, 'voxel_spacing': list(spacing), 'unit': 'um'})
    return image


@pytest.mark.parametrize('axes', [''.join(values) for values in permutations('XYZ')])
def test_all_axis_orders_use_independent_physical_voxel_centres(axes):
    shape = tuple({'X': 7, 'Y': 6, 'Z': 5}[axis] for axis in axes)
    spacing = tuple({'X': .25, 'Y': .5, 'Z': 2.}[axis] for axis in axes)
    image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    labels = np.zeros(shape, dtype=np.uint16)
    session = engine.VolumeMaskSession(image, labels, axes=axes, spacing=spacing)
    limits = ((.25, 1.25), (.5, 2.), (2., 8.))
    changed = session.apply_geometry(900, _box(limits))
    coordinates = np.indices(shape)
    expected = np.ones(shape, dtype=bool)
    for axis, (low, high) in zip('XYZ', limits):
        values = (coordinates[axes.index(axis)] + .5) * spacing[axes.index(axis)]
        expected &= (values >= low) & (values <= high)
    assert changed == expected.sum()
    assert np.array_equal(session.mask, expected.astype(np.uint16) * 900)
    assert np.array_equal(session.image, image)
    assert np.array_equal(labels, np.zeros(shape))
    assert session.mask.dtype == labels.dtype
    assert session.undo() and np.array_equal(session.mask, labels)
    assert session.redo() and np.array_equal(session.mask, expected * 900)


def test_shared_surface_motion_and_nonuniform_depth_change_actual_voxels():
    session = engine.VolumeMaskSession(np.zeros((6, 6, 6), dtype=np.uint16), np.zeros((6, 6, 6), dtype=np.uint16))
    gate = _box([(1, 5), (1, 5), (1, 3)])
    session.apply_geometry(900, gate)
    edited = gate.with_anchor('anchor:5:1', (5, 1, 5))
    session.apply_geometry(900, edited)
    assert session.mask[3, 1, 4] == 900
    assert session.mask[3, 4, 1] == 0
    assert session.mask[1, 4, 1] == 900
    assert session.undo() and session.geometries[900] == gate
    assert session.mask[3, 1, 4] == 0
    assert session.redo() and session.geometries[900] == edited


@pytest.mark.parametrize('shape', [(2, 2, 2), (1, 2, 3)])
def test_existing_label_exact_voxel_boundary_preview_does_not_edit(shape):
    labels = np.zeros((5, 6, 7), dtype=np.uint16)
    labels[1:1 + shape[0], 2:2 + shape[1], 3:3 + shape[2]] = 900
    labels[4, 5, 6] = 17
    session = engine.VolumeMaskSession(np.zeros_like(labels), labels, spacing=(2, .5, .25))
    gate = session.geometry_for_label(900)
    assert not session.geometries and not session._undo
    assert session.apply_geometry(900, gate) == 0
    assert np.array_equal(session.mask, labels)
    assert session.geometries[900] == gate
    assert session.mask[4, 5, 6] == 17
    assert session.next_label() == 1


def test_disconnected_existing_label_is_refused_without_bbox_replacement():
    labels = np.zeros((5, 5, 5), dtype=np.uint16)
    labels[0, 0, 0] = labels[4, 4, 4] = 900
    session = engine.VolumeMaskSession(labels, labels)
    with pytest.raises(GateError):
        session.geometry_for_label(900)
    assert np.array_equal(session.mask, labels)
    assert not session.geometries and not session._undo


def test_collision_and_history_budget_refusals_preserve_all_state(monkeypatch):
    labels = np.zeros((5, 5, 5), dtype=np.uint16)
    labels[2, 2, 2] = 17
    session = engine.VolumeMaskSession(labels, labels)
    with pytest.raises(ValueError, match='overlaps'):
        session.apply_geometry(900, _box([(1, 4)] * 3))
    assert np.array_equal(session.mask, labels) and not session._undo
    monkeypatch.setattr(session, 'MAX_HISTORY_BYTES', 1)
    with pytest.raises(ValueError, match='undo memory'):
        session.apply_geometry(900, _box([(0, 1)] * 3))
    assert np.array_equal(session.mask, labels) and not session.geometries


@pytest.mark.parametrize('axis,spacing', [('TYX', (1, 1, 1)), ('ZZX', (1, 1, 1)), ('ZYX', (0, 1, 1)), ('ZYX', (np.nan, 1, 1)), ('ZYX', (1, np.inf, 1)), ('ZYX', (True, 1, 1)), ('ZYX', (1, 1))])
def test_invalid_axes_and_spacing_are_refused(axis, spacing):
    with pytest.raises(ValueError):
        engine.VolumeMaskSession(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), dtype=np.uint16), axes=axis, spacing=spacing)


@pytest.mark.parametrize('value,dtype', [(-1, np.int16), (1.1, np.float32), (70000, np.uint32)])
def test_invalid_labels_are_not_truncated(value, dtype):
    labels = np.full((2, 3, 4), value, dtype=dtype)
    with pytest.raises(ValueError):
        engine.VolumeMaskSession(np.zeros(labels.shape), labels)


@pytest.mark.parametrize('axes,spacing', [('ZYX', (2, .5, .25)), ('XYZ', (.25, .5, 2))])
def test_real_tiff_save_load_preserves_source_axes_spacing_dtype_ids_and_geometry(tmp_path, axes, spacing):
    image = _source(tmp_path, axes, spacing)
    original_source = (tmp_path / 'source.tif').read_bytes()
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    session.apply_geometry(900, _box([(a, b / 2) for a, b in session.limits]))
    path = engine.save_volume_mask(session)
    reopened = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    assert Path(path) == tmp_path / 'masks/source.tif'
    assert np.array_equal(reopened.image, image)
    assert np.array_equal(reopened.mask, session.mask)
    assert reopened.mask.dtype == session.mask.dtype
    assert reopened.axes == axes and reopened.spacing == tuple(spacing) and reopened.unit == 'um'
    assert reopened.geometries == session.geometries
    assert (tmp_path / 'source.tif').read_bytes() == original_source


def test_failed_second_replace_restores_previous_mask_and_geometry_pair(tmp_path, monkeypatch):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    session.apply_geometry(900, _box([(0, 1)] * 3))
    engine.save_volume_mask(session)
    target = Path(session.mask_path)
    sidecar = target.with_suffix('.tif.volume.json')
    before = (target.read_bytes(), sidecar.read_bytes())
    session.apply_geometry(900, _box([(0, 2)] * 3))
    original = engine.os.replace
    def reject(source, destination):
        if Path(source).name == 'geometry.json':
            raise OSError('injected geometry publication failure')
        return original(source, destination)
    monkeypatch.setattr(engine.os, 'replace', reject)
    with pytest.raises(OSError, match='publication'):
        engine.save_volume_mask(session)
    assert before == (target.read_bytes(), sidecar.read_bytes())
    assert not list(target.parent.glob('.spacr-volume-*'))
    assert not list(target.parent.glob('*.lock'))


def test_changed_source_or_mask_sidecar_is_refused(tmp_path):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    engine.save_volume_mask(session)
    write_tiff(session.mask_path, np.ones(session.mask.shape, dtype=np.uint16), metadata={'axes': session.axes})
    with pytest.raises(ValueError, match='does not match'):
        engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    (tmp_path / 'source.tif').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source volume changed'):
        engine.save_volume_mask(session)


def test_rgb_and_undeclared_page_axes_are_not_guessed_as_depth(tmp_path):
    import tifffile
    tifffile.imwrite(tmp_path / 'rgb.tif', np.zeros((5, 6, 3), dtype=np.uint8), photometric='rgb')
    with pytest.raises(ValueError, match='three spatial axes'):
        engine.load_volume_image_and_mask(str(tmp_path), 'rgb.tif')


def test_sparse_history_remains_bounded_and_redo_is_dropped_after_new_edit(monkeypatch):
    session = engine.VolumeMaskSession(np.zeros((5, 5, 5)), np.zeros((5, 5, 5), dtype=np.uint16))
    monkeypatch.setattr(session, 'MAX_HISTORY_BYTES', 150)
    for size in [1, 2, 1, 2, 1]:
        session.apply_geometry(900, _box([(0, size)] * 3))
        assert sum(sum(value.nbytes for value in row[:3]) for row in session._undo + session._redo) <= 150
    assert session.undo()
    session.apply_geometry(900, _box([(0, 1.5)] * 3))
    assert not session.redo()


def test_ome_physical_units_and_imagej_spacing_are_preserved(tmp_path):
    import tifffile
    image = np.zeros((4, 5, 6), dtype=np.uint16)
    tifffile.imwrite(tmp_path / 'ome.ome.tif', image, ome=True, metadata={'axes': 'ZYX', 'PhysicalSizeX': 250., 'PhysicalSizeXUnit': 'nm', 'PhysicalSizeY': .5, 'PhysicalSizeYUnit': 'µm', 'PhysicalSizeZ': 2., 'PhysicalSizeZUnit': 'µm'})
    session = engine.load_volume_image_and_mask(str(tmp_path), 'ome.ome.tif')
    assert session.spacing == (2., .5, .25) and session.unit == 'um'
    tifffile.imwrite(tmp_path / 'imagej.tif', image, imagej=True, resolution=(4., 2.), metadata={'axes': 'ZYX', 'spacing': 2., 'unit': 'um'})
    imagej = engine.load_volume_image_and_mask(str(tmp_path), 'imagej.tif')
    assert imagej.spacing == session.spacing and imagej.unit == session.unit


def test_existing_saved_pixels_changed_by_another_session_are_not_overwritten(tmp_path):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    engine.save_volume_mask(session)
    target = Path(session.mask_path)
    write_tiff(target, np.full(session.mask.shape, 17, dtype=np.uint16), metadata={'axes': session.axes})
    externally_written = target.read_bytes()
    with pytest.raises(ValueError, match='outside this session'):
        engine.save_volume_mask(session)
    assert target.read_bytes() == externally_written
    assert not list(target.parent.glob('*.lock'))


def test_voxel_membership_requests_are_bounded_and_extreme_geometry_does_not_overflow(monkeypatch):
    session = engine.VolumeMaskSession(np.zeros((17, 17, 17)), np.zeros((17, 17, 17), dtype=np.uint16))
    original = EditableVolumeGate.mask
    counts = []
    def measured(gate, frame):
        counts.append(len(frame))
        return original(gate, frame)
    monkeypatch.setattr(EditableVolumeGate, 'mask', measured)
    session.apply_geometry(900, _box([(0, 17)] * 3))
    assert max(counts) <= 4096 and sum(counts) == 17 ** 3
    before = session.mask.copy()
    assert session.apply_geometry(900, _box([(1e20, 2e20)] * 3)) == 17 ** 3
    assert not session.mask.any()
    assert session.undo() and np.array_equal(session.mask, before)


@pytest.mark.parametrize('label', [True, 900.1, '900', -1])
def test_label_identity_cannot_be_coerced_or_silently_truncated(label):
    session = engine.VolumeMaskSession(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), dtype=np.uint16))
    with pytest.raises(ValueError):
        session.geometry_for_label(label)
    with pytest.raises(ValueError):
        session.apply_geometry(label, _box([(0, 1)] * 3))
    assert not session.mask.any() and not session._undo


def test_foreign_mask_columns_and_label_names_are_refused_without_voxel_changes():
    session = engine.VolumeMaskSession(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), dtype=np.uint16))
    with pytest.raises(ValueError):
        session.apply_geometry(17, _box([(0, 1)] * 3))
    assert not session.mask.any() and not session._undo


def test_existing_integer_label_dtype_and_fortran_input_order_are_preserved():
    labels = np.zeros((2, 3, 4), dtype=np.uint8, order='F')
    session = engine.VolumeMaskSession(np.zeros_like(labels), labels)
    session.apply_geometry(17, _box([(0, 1)] * 3, name='17'))
    assert session.mask.dtype == np.uint8 and session.mask[0, 0, 0] == 17
    assert session.undo() and not session.mask.any()
    assert session.redo() and session.mask[0, 0, 0] == 17
    with pytest.raises(ValueError):
        session.apply_geometry(900, _box([(0, 1)] * 3))
    assert not labels.any()


@pytest.mark.parametrize('dtype,shape', [(np.uint16, (2, 3, 4)), (np.float32, (5, 6, 7))])
def test_wrong_saved_volume_is_rejected_before_mask_decoding(tmp_path, monkeypatch, dtype, shape):
    import tifffile
    _source(tmp_path)
    (tmp_path / 'masks').mkdir()
    write_tiff(tmp_path / 'masks/source.tif', np.zeros(shape, dtype=dtype), metadata={'axes': 'ZYX'})
    original = tifffile.TiffPageSeries.asarray
    calls = []
    def checked(series, *args, **kwargs):
        calls.append(series.parent.filename)
        assert series.parent.filename != 'source.tif' or str(series.parent.filehandle.path) == str(tmp_path / 'source.tif')
        return original(series, *args, **kwargs)
    monkeypatch.setattr(tifffile.TiffPageSeries, 'asarray', checked)
    with pytest.raises(ValueError, match='shape, integer dtype or decoded size'):
        engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    assert len(calls) == 1


@pytest.mark.parametrize('changed', ['source', 'mask', 'sidecar'])
def test_external_file_changes_during_staging_are_preserved_without_session_mutation(tmp_path, monkeypatch, changed):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    engine.save_volume_mask(session)
    session.apply_geometry(900, _box([(0, 1)] * 3))
    source, mask = Path(session.source_path), Path(session.mask_path)
    sidecar = mask.with_suffix('.tif.volume.json')
    path = {'source': source, 'mask': mask, 'sidecar': sidecar}[changed]
    original = engine.write_tiff
    def mutate_after_stage(*args, **kwargs):
        original(*args, **kwargs)
        path.write_bytes(b'external concurrent writer')
    monkeypatch.setattr(engine, 'write_tiff', mutate_after_stage)
    before_mask = session.mask.copy()
    before_geometry = dict(session.geometries)
    before_history = len(session._undo)
    with pytest.raises(ValueError, match='changed during saving'):
        engine.save_volume_mask(session)
    assert path.read_bytes() == b'external concurrent writer'
    assert np.array_equal(session.mask, before_mask)
    assert session.geometries == before_geometry and len(session._undo) == before_history
    assert not list(mask.parent.glob('.spacr-volume-*'))
    assert not list(mask.parent.glob('*.lock'))


def test_cancellation_before_voxel_publication_preserves_session_and_history():
    session = engine.VolumeMaskSession(np.zeros((5, 5, 5)), np.zeros((5, 5, 5), dtype=np.uint16))
    session._cancelled = lambda: True
    with pytest.raises(InterruptedError, match='cancelled'):
        session.apply_geometry(900, _box([(0, 2)] * 3))
    assert not session.mask.any() and not session.geometries and not session._undo


def test_saved_boundary_disagreement_is_refused_without_overwriting_outputs(tmp_path):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    session.apply_geometry(900, _box([(0, 1)] * 3))
    path = Path(engine.save_volume_mask(session))
    original = path.read_bytes()
    session.mask[-1, -1, -1] = 900
    with pytest.raises(ValueError, match='boundary does not match'):
        engine.save_volume_mask(session)
    assert path.read_bytes() == original


def test_saved_anchor_controls_and_mesh_roundtrip(tmp_path):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    session._editor_controls = {'anchor_density': 4, 'show_anchors': False}
    engine.save_volume_mask(session)
    reopened = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    assert reopened._editor_controls == session._editor_controls


def test_nonfinite_physical_extents_are_refused_before_geometry():
    with pytest.raises(ValueError, match='physical image bounds'):
        engine.VolumeMaskSession(np.zeros((5, 5, 5)), np.zeros((5, 5, 5), dtype=np.uint16), spacing=(1e308, 1, 1))
