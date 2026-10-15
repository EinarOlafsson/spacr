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


@pytest.mark.parametrize('axis,size', [('S', 10**12), ('T', 99), ('X', 99)])
def test_raw_subblock_extent_refuses_before_desktop_scene_expansion(legacy, monkeypatch, axis, size):
    from spacr import convert

    path, _, entries, reads, _ = legacy
    shape = list(entries[0].shape)
    shape[entries[0].axes.index(axis)] = size
    entries[0].shape = tuple(shape)

    def no_metadata_expansion(*args, **kwargs):
        pytest.fail('Desktop metadata helper reached before raw subblock bounds were validated')

    monkeypatch.setattr(convert, '_legacy_czi_series', no_metadata_expansion)
    with pytest.raises(ValueError, match='unambiguous spatial axes'):
        inputs.describe_upload(path)
    assert reads == []


def test_huge_consistent_legacy_canvas_budget_precedes_scene_expansion(legacy, monkeypatch):
    from spacr import convert

    path, _, entries, reads, handle_type = legacy
    original_init = handle_type.__init__

    def init(self, path):
        original_init(self, path)
        self.shape = (10**12, *self.shape[1:])

    monkeypatch.setattr(handle_type, '__init__', init)
    entries[0].shape = (10**12, *entries[0].shape[1:])
    monkeypatch.setattr(convert, '_legacy_czi_series', lambda *args: pytest.fail(
        'Desktop helper must not expand huge consistent S extent'))
    with pytest.raises(ValueError, match='byte limit'):
        inputs.describe_upload(path)
    assert reads == []


def test_callers_actual_canvas_budget_precedes_scene_expansion(legacy, monkeypatch):
    from spacr import convert

    path, _, _, reads, _ = legacy
    monkeypatch.setattr(convert, '_legacy_czi_series', lambda *args: pytest.fail(
        'Desktop helper ran before the caller-specific canvas byte budget'))
    with pytest.raises(ValueError, match='byte limit'):
        inputs.read_upload_plane(path, indices={'T': 0, 'Z': 0}, max_decoded_bytes=3000)
    assert reads == []


def test_distinct_scene_count_refuses_before_expansion(legacy, monkeypatch):
    from spacr import convert

    path, _, _, reads, _ = legacy
    monkeypatch.setattr(convert, '_legacy_czi_series', lambda *args: pytest.fail(
        'Distinct-scene limit must precede desktop expansion'))
    for action in (lambda: inputs.describe_upload(path, max_scenes=1),
                   lambda: inputs.read_upload_plane(path, indices={'T': 0, 'Z': 0}, max_scenes=1)):
        with pytest.raises(ValueError, match='scene limit'):
            action()
    assert reads == []


def test_large_contained_scene_interval_refuses_before_expansion(legacy, monkeypatch):
    from spacr import convert

    path, _, entries, reads, handle_type = legacy
    original_init = handle_type.__init__

    def init(self, path):
        original_init(self, path)
        self.shape = (5000, *self.shape[1:])

    monkeypatch.setattr(handle_type, '__init__', init)
    entries[0].shape = (5000, *entries[0].shape[1:])
    monkeypatch.setattr(convert, '_legacy_czi_series', lambda *args: pytest.fail(
        'Scene dictionary expansion must not run even when pixels fit the byte budget'))
    with pytest.raises(ValueError, match='scene limit'):
        inputs.describe_upload(path)
    assert reads == []


def test_explicit_scene_cap_and_repeated_scene_subblocks_preserve_pixels(legacy):
    path, array, entries, reads, _ = legacy
    entries.extend([entries[0]] * 5)
    assert len(inputs.describe_upload(path, max_scenes=2)['series']) == 2
    result = inputs.read_upload_plane(path, series=1, indices={'T': 1, 'Z': 2},
                                     channels=[1, 0], max_scenes=2)
    np.testing.assert_array_equal(result['image'], array[1, 1, [1, 0], 2, 2:8, 3:12, 0])
    assert reads == ['decoded']


@pytest.mark.parametrize('value', [True, np.bool_(False), 0, -1, 1.5, '2'])
def test_scene_limit_requires_positive_integral_configuration(legacy, value):
    path, _, _, reads, _ = legacy
    with pytest.raises(ValueError, match='integer'):
        inputs.describe_upload(path, max_scenes=value)
    with pytest.raises(ValueError, match='integer'):
        inputs.read_upload_plane(path, indices={'T': 0, 'Z': 0}, max_scenes=value)
    assert reads == []
