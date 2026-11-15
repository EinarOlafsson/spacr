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


