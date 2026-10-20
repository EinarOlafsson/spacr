"""Native volume session, loader, saver and vvvv publisher edge behaviour."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.widgets.gate_spec import BoxGate, EditableVolumeGate, GateSet
from spacr.tiff_io import write_tiff


def _box(limits, name='900'):
    return EditableVolumeGate.from_gate(BoxGate.from_limits(name, ('X', 'Y', 'Z'), limits))


def _empty(shape=(5, 5, 5), dtype=np.uint16):
    return engine.VolumeMaskSession(np.zeros(shape), np.zeros(shape, dtype=dtype))


def _source(tmp_path, name='source.tif'):
    image = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    write_tiff(tmp_path / name, image, metadata={'axes': 'ZYX', 'voxel_spacing': [1., 1., 1.], 'unit': 'um'})
    return image


class _FlakyTabular:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def _publish(self, target, write):
        self.calls += 1
        if self.calls <= self.failures:
            raise PermissionError('held by vvvv')
        return ('published', target)


def test_vvvv_publisher_retries_a_held_target_then_publishes(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, 'sleep', sleeps.append)
    tabular = _FlakyTabular(failures=3)
    assert engine._vvvv_publisher(tabular)('out.csv', None) == ('published', 'out.csv')
    assert tabular.calls == 4 and sleeps == [0.05] * 3


def test_vvvv_publisher_raises_after_twenty_held_attempts(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, 'sleep', sleeps.append)
    tabular = _FlakyTabular(failures=100)
    with pytest.raises(PermissionError, match='held by vvvv'):
        engine._vvvv_publisher(tabular)('out.csv', None)
    assert tabular.calls == 20 and len(sleeps) == 19


@pytest.mark.parametrize('spacing', [None, ('a', 1, 1), 3.0])
def test_unparseable_spacing_is_refused(spacing):
    with pytest.raises(ValueError, match='Voxel spacing') as info:
        engine.VolumeMaskSession(np.zeros((2, 2, 2)), np.zeros((2, 2, 2), dtype=np.uint16), spacing=spacing)
    assert isinstance(info.value.__cause__, (TypeError, ValueError))


def test_next_label_skips_occupied_ids_and_refuses_a_full_dtype():
    labels = np.arange(256, dtype=np.uint8).reshape(4, 8, 8)
    session = engine.VolumeMaskSession(np.zeros(labels.shape), labels)
    with pytest.raises(ValueError, match='No unused object identifier'):
        session.next_label()
    labels = labels.copy()
    labels[labels == 200] = 0
    assert engine.VolumeMaskSession(np.zeros(labels.shape), labels).next_label() == 200


def test_geometry_for_an_absent_positive_label_is_refused():
    labels = np.zeros((3, 3, 3), dtype=np.uint16)
    labels[1, 1, 1] = 5
    session = engine.VolumeMaskSession(np.zeros(labels.shape), labels)
    with pytest.raises(ValueError, match='existing positive object label'):
        session.geometry_for_label(6)
    gate = session.geometry_for_label(5)
    assert gate.name == '5' and len(gate.vertices) == 8 and len(gate.faces) == 12


def test_label_with_too_much_boundary_detail_is_refused():
    labels = np.zeros((24, 24, 24), dtype=np.uint16)
    labels[::2, ::2, ::2] = 7
    session = engine.VolumeMaskSession(np.zeros(labels.shape), labels)
    with pytest.raises(ValueError, match='too much boundary detail'):
        session.geometry_for_label(7)
    assert not session.geometries


def test_boundary_between_voxel_centres_paints_nothing_but_keeps_geometry():
    session = _empty()
    gate = _box([(0.1, 0.4), (0, 5), (0, 5)])
    assert session.apply_geometry(900, gate) == 0
    assert not session.mask.any()
    assert session.geometries == {900: gate}
    assert len(session._undo) == 1
    assert session.undo() and session.geometries == {}


def test_cancellation_after_painting_changes_nothing():
    session = _empty()
    session._cancelled = lambda: True
    with pytest.raises(InterruptedError, match='cancelled'):
        session.apply_geometry(900, _box([(0.1, 0.4), (0, 5), (0, 5)]))
    assert not session.mask.any() and not session.geometries and not session._undo


def test_reapplying_the_same_geometry_records_no_history():
    session = _empty()
    gate = _box([(0, 2)] * 3)
    assert session.apply_geometry(900, gate) == 8
    assert session.apply_geometry(900, gate) == 0
    assert len(session._undo) == 1 and session.mask.sum() == 8 * 900


def test_history_budget_evicts_the_oldest_edit():
    session = _empty()
    session.apply_geometry(900, _box([(0, 1)] * 3))
    one_voxel = session._history_bytes
    session.MAX_HISTORY_BYTES = one_voxel * 1.5
    session.apply_geometry(901, _box([(3, 4)] * 3, name='901'))
    assert len(session._undo) == 1
    assert session._history_bytes == one_voxel
    assert session.undo()
    assert session.mask[0, 0, 0] == 900 and session.mask[3, 3, 3] == 0
    assert not session.undo()
    assert session.mask[0, 0, 0] == 900


def test_undo_stack_keeps_only_twenty_five_edits():
    session = _empty((1, 1, 30))
    for index in range(26):
        session.apply_geometry(900, _box([(index, index + 1), (0, 1), (0, 1)]))
    assert len(session._undo) == 25
    assert session._history_bytes == sum(sum(value.nbytes for value in row[:3]) for row in session._undo)
    while session.undo():
        pass
    assert session.mask[0, 0, 0] == 900 and session.mask.sum() == 900


@pytest.mark.parametrize('geometries', [
    {True: None}, {0: None}, {900: 'not a gate'},
    {900: EditableVolumeGate.from_gate(BoxGate.from_limits('901', ('X', 'Y', 'Z'), [(0, 1)] * 3))},
])
def test_invalid_saved_geometry_is_refused(geometries):
    session = _empty()
    session.geometries = dict(geometries)
    with pytest.raises(ValueError, match='invalid object geometry'):
        session._validate_boundaries()


def test_ome_without_physical_sizes_keeps_pixel_coordinates(tmp_path):
    import tifffile
    tifffile.imwrite(tmp_path / 'plain.ome.tif', np.zeros((3, 4, 5), dtype=np.uint16), ome=True, metadata={'axes': 'ZYX'})
    session = engine.load_volume_image_and_mask(str(tmp_path), 'plain.ome.tif')
    assert session.spacing == (1., 1., 1.) and session.unit == 'pixel'
    assert session.axes == 'ZYX' and session.mask.shape == (3, 4, 5)


@pytest.mark.parametrize('metadata', [
    {'PhysicalSizeX': 1., 'PhysicalSizeY': 1.},
    {'PhysicalSizeX': 1., 'PhysicalSizeXUnit': 'pc', 'PhysicalSizeY': 1., 'PhysicalSizeZ': 1.},
])
def test_ome_with_incomplete_or_unsupported_spacing_is_refused(tmp_path, metadata):
    import tifffile
    tifffile.imwrite(tmp_path / 'bad.ome.tif', np.zeros((3, 4, 5), dtype=np.uint16), ome=True, metadata={'axes': 'ZYX', **metadata})
    with pytest.raises(ValueError, match='OME volumes need complete') as info:
        engine.load_volume_image_and_mask(str(tmp_path), 'bad.ome.tif')
    assert isinstance(info.value.__cause__, KeyError)


def test_imagej_zero_resolution_is_refused(tmp_path):
    import tifffile
    tifffile.imwrite(tmp_path / 'imagej.tif', np.zeros((3, 4, 5), dtype=np.uint16), imagej=True,
                     resolution=((0, 1), (1, 1)), metadata={'axes': 'ZYX', 'spacing': 2.})
    with pytest.raises(ValueError, match='Voxel spacing'):
        engine.load_volume_image_and_mask(str(tmp_path), 'imagej.tif')


def test_tiff_over_the_voxel_budget_is_refused_before_decoding(tmp_path, monkeypatch):
    _source(tmp_path)
    monkeypatch.setattr(engine.VolumeMaskSession, 'MAX_VOXELS', 10)
    with pytest.raises(ValueError, match='exceeds the native volume memory budget'):
        engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')


def test_saved_mask_with_other_axes_is_refused(tmp_path):
    _source(tmp_path)
    (tmp_path / 'masks').mkdir()
    write_tiff(tmp_path / 'masks/source.tif', np.zeros((6, 5, 4), dtype=np.uint16), metadata={'axes': 'XYZ'})
    with pytest.raises(ValueError, match='mask axes do not match'):
        engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')


def test_source_changing_while_opening_is_refused(tmp_path, monkeypatch):
    _source(tmp_path)
    original = engine._yolo_source_hash
    seen = []

    def drifting(path):
        seen.append(Path(path).name)
        digest = original(path)
        return digest if len(seen) == 1 else 'changed' + digest
    monkeypatch.setattr(engine, '_yolo_source_hash', drifting)
    with pytest.raises(ValueError, match='changed while opening'):
        engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    assert seen == ['source.tif', 'source.tif']


def _saved(tmp_path):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    session.apply_geometry(900, _box([(0, 2)] * 3))
    engine.save_volume_mask(session)
    sidecar = Path(session.mask_path + '.volume.json')
    return session, sidecar, json.loads(sidecar.read_text(encoding="utf-8"))


def test_sidecar_with_non_volume_geometry_is_refused(tmp_path):
    session, sidecar, record = _saved(tmp_path)
    record['geometry'] = GateSet([BoxGate.from_limits('900', ('X', 'Y', 'Z'), [(0, 2)] * 3)]).to_dict()
    sidecar.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match='invalid object geometry'):
        engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')


@pytest.mark.parametrize('controls', [
    [], {'anchor_density': 2}, {'anchor_density': 9, 'show_anchors': True},
    {'anchor_density': 2.0, 'show_anchors': True}, {'anchor_density': 2, 'show_anchors': 1},
])
def test_sidecar_with_invalid_anchor_controls_is_refused(tmp_path, controls):
    session, sidecar, record = _saved(tmp_path)
    record['editor_controls'] = controls
    sidecar.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match='anchor controls are invalid'):
        engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')


def test_sidecar_restores_geometry_and_controls(tmp_path):
    session, sidecar, record = _saved(tmp_path)
    reopened = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    assert reopened.geometries == session.geometries
    assert reopened._editor_controls == {'anchor_density': 2, 'show_anchors': True}
    assert int((reopened.mask == 900).sum()) == 8


def test_save_refuses_a_session_without_a_source():
    with pytest.raises(ValueError, match='source volume changed'):
        engine.save_volume_mask(_empty())
    with pytest.raises(ValueError, match='source volume changed'):
        engine.save_volume_mask('not a session')


def test_save_refuses_to_overwrite_the_source(tmp_path):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    before = (tmp_path / 'source.tif').read_bytes()
    session.mask_path = session.source_path
    with pytest.raises(ValueError, match='must not overwrite the source'):
        engine.save_volume_mask(session)
    assert (tmp_path / 'source.tif').read_bytes() == before


def test_save_refuses_while_another_save_holds_the_lock(tmp_path):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    lock = Path(session.mask_path + '.volume.json.lock')
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b'')
    with pytest.raises(ValueError, match='already being saved'):
        engine.save_volume_mask(session)
    assert lock.exists() and not Path(session.mask_path).exists()


def test_failed_first_publication_removes_the_new_mask(tmp_path, monkeypatch):
    _source(tmp_path)
    session = engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')
    session.apply_geometry(900, _box([(0, 1)] * 3))
    original = engine.os.replace

    def reject(source, destination):
        if Path(source).name == 'geometry.json':
            raise OSError('injected geometry publication failure')
        return original(source, destination)
    monkeypatch.setattr(engine.os, 'replace', reject)
    with pytest.raises(OSError, match='publication'):
        engine.save_volume_mask(session)
    target = Path(session.mask_path)
    assert not target.exists() and not Path(session.mask_path + '.volume.json').exists()
    assert not list(target.parent.glob('.spacr-volume-*')) and not list(target.parent.glob('*.lock'))
    assert session._saved_mask_sha256 is None
