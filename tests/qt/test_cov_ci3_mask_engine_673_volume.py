"""Native volume session, loader, saver and vvvv publisher edge behaviour."""
import time

import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.widgets.gate_spec import BoxGate, EditableVolumeGate
from spacr.tiff_io import write_tiff


def _box(limits, name='900'):
    return EditableVolumeGate.from_gate(BoxGate.from_limits(name, ('X', 'Y', 'Z'), limits))


def _empty(shape=(5, 5, 5), dtype=np.uint16):
    return engine.VolumeMaskSession(np.zeros(shape), np.zeros(shape, dtype=dtype))


def test_mesh_boundary_cancellation_preserves_volume_and_edit_hook(monkeypatch):
    from spacr.cancellation import CancellationToken, PipelineCancelled, current_token, installed_token
    from spacr.qt.widgets import gate_spec

    labels = np.ones((2, 2, 2), dtype=np.uint16)
    session = engine.VolumeMaskSession(np.zeros(labels.shape), labels)
    edit_hook = lambda: False
    session._cancelled = edit_hook
    token = CancellationToken()
    original = engine.np.argwhere
    constructor = gate_spec.EditableVolumeGate
    constructed = []

    def construct(*args, **kwargs):
        constructed.append(True)
        return constructor(*args, **kwargs)

    def positions(*args, **kwargs):
        result = original(*args, **kwargs)
        token.cancel()
        return result

    monkeypatch.setattr(engine.np, 'argwhere', positions)
    monkeypatch.setattr(gate_spec, 'EditableVolumeGate', construct)
    outer = CancellationToken()
    with installed_token(outer):
        with installed_token(token), pytest.raises(PipelineCancelled):
            session.geometry_for_label(1)
        assert current_token() is outer
    assert not constructed
    np.testing.assert_array_equal(session.mask, labels)
    assert session._cancelled is edit_hook
    assert not session.geometries and not session._undo and not session._redo
    assert session._saved_mask_sha256 is None and session._saved_geometry_sha256 is None


def test_uncancelled_mesh_is_exact_and_restores_outer_token():
    from spacr.cancellation import CancellationToken, current_token, installed_token

    labels = np.ones((2, 2, 2), dtype=np.uint16)
    session = engine.VolumeMaskSession(np.zeros(labels.shape), labels)
    expected = session.geometry_for_label(1)
    outer, token = CancellationToken(), CancellationToken()
    with installed_token(outer):
        with installed_token(token):
            actual = session.geometry_for_label(1)
        assert current_token() is outer
    assert actual.to_dict() == expected.to_dict()
    np.testing.assert_array_equal(session.mask, labels)
    assert not session.geometries and not session._undo and not session._redo


def test_refused_mesh_restores_outer_token_without_changing_state():
    from spacr.cancellation import CancellationToken, current_token, installed_token

    labels = np.zeros((3, 3, 3), dtype=np.uint16)
    labels[0, 0, 0] = labels[2, 2, 2] = 1
    session = engine.VolumeMaskSession(np.zeros(labels.shape), labels)
    outer, token = CancellationToken(), CancellationToken()
    with installed_token(outer):
        with installed_token(token), pytest.raises(ValueError):
            session.geometry_for_label(1)
        assert current_token() is outer
    np.testing.assert_array_equal(session.mask, labels)
    assert not session.geometries and not session._undo and not session._redo
    assert session._saved_mask_sha256 is None and session._saved_geometry_sha256 is None


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


@pytest.mark.parametrize('geometries', [
    {True: None}, {0: None}, {900: 'not a gate'},
    {900: EditableVolumeGate.from_gate(BoxGate.from_limits('901', ('X', 'Y', 'Z'), [(0, 1)] * 3))},
])
def test_invalid_saved_geometry_is_refused(geometries):
    session = _empty()
    session.geometries = dict(geometries)
    with pytest.raises(ValueError, match='invalid object geometry'):
        session._validate_boundaries()


def test_saved_mask_with_other_axes_is_refused(tmp_path):
    _source(tmp_path)
    (tmp_path / 'masks').mkdir()
    write_tiff(tmp_path / 'masks/source.tif', np.zeros((6, 5, 4), dtype=np.uint16), metadata={'axes': 'XYZ'})
    with pytest.raises(ValueError, match='mask axes do not match'):
        engine.load_volume_image_and_mask(str(tmp_path), 'source.tif')


