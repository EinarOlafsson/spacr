"""Legacy subblock metadata preserves scene bounds before bounded decoding."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from tools.web_mask_generator import inputs


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    axes = 'STCZYX0'
    shape = (2, 2, 2, 3, 8, 12, 1)
    start = (5, 10, 1, 7, -2, 20, 0)
    array = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    entries = [SimpleNamespace(axes=axes, start=(5, 10, 1, 7, -2, 20, 0),
                               shape=(1, 2, 2, 3, 5, 7, 1)),
               SimpleNamespace(axes=axes, start=(6, 10, 1, 7, 0, 23, 0),
                               shape=(1, 2, 2, 3, 6, 9, 1))]
    reads = []

    class Handle:
        def __init__(self, path):
            self.axes = axes
            self.shape = shape
            self.start = start
            self.dtype = array.dtype
            self.filtered_subblock_directory = entries

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def asarray(self):
            reads.append('decoded')
            return array

    monkeypatch.setitem(sys.modules, 'czifile', SimpleNamespace(CziFile=Handle))
    path = tmp_path / 'legacy.czi'
    path.write_bytes(b'fake header; pixel source is explicit isolated legacy reader')
    return path, array, entries, reads, Handle


def test_legacy_absolute_scene_origins_and_exact_selected_pixels(legacy):
    path, array, _, reads, _ = legacy
    before = array.copy()
    source = path.read_bytes()
    description = inputs.describe_upload(path)
    assert len(description['series']) == 2
    assert reads == []
    assert [row['scene_id'] for row in description['series']] == [5, 6]
    assert [row['shape'] for row in description['series']] == [[2, 2, 3, 5, 7], [2, 2, 3, 6, 9]]
    result = inputs.read_upload_plane(path, series=1, indices={'T': 1, 'Z': 2}, channels=[1, 0])
    np.testing.assert_array_equal(result['image'], array[1, 1, [1, 0], 2, 2:8, 3:12, 0])
    assert result['image'].dtype == np.uint16
    assert result['selection'] == {'format': 'CZI', 'channels': [1, 0], 'series': 1,
        'scene_id': 6, 'source_axes': 'TCZYX', 'indices': {'T': 1, 'Z': 2}}
    assert reads == ['decoded']
    np.testing.assert_array_equal(array, before)
    assert path.read_bytes() == source


@pytest.mark.parametrize('changes,message', [
    ({'max_decoded_bytes': 3000}, 'byte limit'),
    ({'max_plane_pixels': 1}, 'pixel limit'),
    ({'indices': {'T': 2, 'Z': 0}}, 'out of range'),
    ({'indices': {'T': True, 'Z': 0}}, 'integer'),
    ({'indices': {'T': 1.5, 'Z': 0}}, 'integer'),
    ({'indices': {}}, 'explicit index'),
    ({'channels': [2]}, 'declared range'),
    ({'channels': [0, 0]}, 'distinct'),
    ({'series': 2}, 'out of range'),
])
def test_whole_legacy_canvas_and_selection_refuse_before_any_decode(legacy, changes, message):
    path, _, _, reads, _ = legacy
    kwargs = {'series': 0, 'indices': {'T': 0, 'Z': 0}, **changes}
    with pytest.raises(ValueError, match=message):
        inputs.read_upload_plane(path, **kwargs)
    assert reads == []


def test_legacy_subblocks_outside_canvas_refuse_before_decode(legacy):
    path, _, entries, reads, _ = legacy
    entries[1].start = (6, 10, 1, 7, 0, 23, 0)
    entries[1].shape = (1, 2, 2, 3, 6, 99, 1)
    with pytest.raises(ValueError, match='unambiguous spatial axes'):
        inputs.read_upload_plane(path, series=1, indices={'T': 0, 'Z': 0})
    assert reads == []


@pytest.mark.parametrize('attribute,value', [
    ('start', (5., 10, 1, 7, -2, 20, 0)),
    ('shape', (1, 2, 2, 3, 5.5, 7, 1)),
])
def test_subblock_coordinates_must_be_declared_integers(legacy, attribute, value):
    path, _, entries, reads, _ = legacy
    setattr(entries[0], attribute, value)
    with pytest.raises(ValueError, match='unambiguous spatial axes'):
        inputs.read_upload_plane(path, indices={'T': 0, 'Z': 0})
    assert reads == []
