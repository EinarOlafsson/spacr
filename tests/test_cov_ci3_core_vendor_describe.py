"""Edge cases of the F548 vendor watch: stack layouts, container and companion parsing."""

import json
import os
import sys
import types

import numpy as np
import pytest
import tifffile

from spacr import core
from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token


def _cancelled():
    token = CancellationToken('stop now')
    token.cancel()
    return installed_token(token)


def test_observed_field_of_a_vendor_watch_is_the_stack_key():
    assert core._watch_observed_field_of('run/acq_1.ome.tif', {}, {}, 'vendor') == ('acq1', None)


def test_vendor_key_keeps_unknown_suffixes_and_falls_back_to_field():
    assert core._watch_vendor_key('notes.png') == 'notespng'
    assert core._watch_vendor_key('--.tif') == 'field'


class _Series:
    def __init__(self, axes, shape, pages):
        self.axes, self.shape, self.pages = axes, shape, pages


class _Handle:
    is_ome, is_imagej, is_shaped = True, False, False

    def __init__(self, *series):
        self.series = list(series)


class _Page:
    def __init__(self, shape):
        self.shape = shape

    def asarray(self):
        return np.ones(self.shape, np.uint16)


class _BrokenPages:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        raise OSError('truncated IFD chain')


def test_layout_refuses_two_series():
    one = _Series('TYX', (2, 4, 4), [_Page((4, 4))] * 2)
    with pytest.raises(ValueError, match='declares 2 series; one field per file'):
        core._watch_vendor_layout(_Handle(one, one), 'timelapse')


@pytest.mark.parametrize('axes, shape', [('QYX', (2, 4, 4)), ('TTYX', (2, 2, 4, 4)),
                                         ('TYXZ', (2, 4, 4, 2))])
def test_layout_refuses_undeclared_or_misplaced_axes(axes, shape):
    series = _Series(axes, shape, [])
    with pytest.raises(ValueError, match='must be T, Z and C before YX'):
        core._watch_vendor_layout(_Handle(series), 'timelapse')


def test_layout_reports_pages_that_cannot_be_reached_yet():
    series = _Series('TYX', (2, 4, 4), _BrokenPages())
    with pytest.raises(ValueError, match=r'not all in the file yet \(truncated IFD chain\)'):
        core._watch_vendor_layout(_Handle(series), 'timelapse')


def test_unreadable_names_a_plane_that_does_not_read_whole(tmp_path, monkeypatch):
    path = tmp_path / 'acq.tif'
    tifffile.imwrite(path, np.ones((4, 4), np.uint16))
    fake = _Series('TYX', (2, 4, 4), [_Page((4, 4)), _Page((3, 3))])
    monkeypatch.setattr(core, '_watch_vendor_layout',
                        lambda handle, mode: (fake, 'TYX', (2, 4, 4)))
    assert core._watch_vendor_unreadable(str(path), 'timelapse') == (
        'declared plane 2 does not read whole')


def test_unreadable_lets_stop_through(tmp_path):
    path = tmp_path / 'acq.tif'
    tifffile.imwrite(path, np.ones((2, 1, 4, 4), np.uint16), imagej=True,
                     metadata={'axes': 'TCYX'})
    assert core._watch_vendor_unreadable(str(path), 'timelapse') is None
    with _cancelled(), pytest.raises(PipelineCancelled, match='stop now'):
        core._watch_vendor_unreadable(str(path), 'timelapse')


def test_stage_refuses_a_plane_that_no_longer_reads_whole(tmp_path, monkeypatch):
    path = tmp_path / 'acq.tif'
    tifffile.imwrite(path, np.ones((4, 4), np.uint16))
    fake = _Series('TYX', (2, 4, 4), [_Page((4, 4)), _Page((2, 4))])
    monkeypatch.setattr(core, '_watch_vendor_layout',
                        lambda handle, mode: (fake, 'TYX', (2, 4, 4)))
    field = tmp_path / 'field'
    field.mkdir()
    with pytest.raises(ValueError, match=r'declared plane 2 does not read whole\.'):
        core._watch_vendor_stage(str(field), str(path), 'acq', 'timelapse', {'channels': [0]})
    written = os.listdir(field)
    assert len(written) == 1 and written[0].startswith('acq_A01')


@pytest.mark.parametrize('channels', ['[0', '[3]', '[]', 'None', [True], [-1]])
def test_vendor_channels_must_index_declared_channels(channels):
    with pytest.raises(ValueError, match=r"within the stack's 2 declared channel\(s\)"):
        core._watch_vendor_channels({'channels': channels}, 2)


def test_vendor_channels_accept_a_string_list():
    assert core._watch_vendor_channels({'channels': '[0, 1]'}, 2) is None


@pytest.mark.parametrize('roles', [{}, {'nucleus_channel': 4}, {'cell_channel': -1},
                                   {'nucleus_channel': 0, 'cell_channel': 1.0}])
def test_series_emit_needs_object_channels_inside_the_stack(tmp_path, roles):
    planes = [((t, z, 0), 'acq.tif', np.full((4, 4), t + z + 1, np.uint16))
              for t in range(2) for z in range(2)]
    with pytest.raises(ValueError, match="object channels must index the stack's"):
        core._watch_vendor_emit(str(tmp_path), 'series', {'channels': [0], **roles},
                                ('acq', 'A01', 1), {'T': 2, 'Z': 2, 'C': 1}, planes)


def test_series_emit_writes_its_map_and_plan(tmp_path):
    planes = [((t, z, 0), 'acq.tif', np.full((4, 4), t + z + 1, np.uint16))
              for t in range(2) for z in range(2)]
    digests, plan = core._watch_vendor_emit(
        str(tmp_path), 'series', {'channels': [0], 'nucleus_channel': 0},
        ('acq', 'A01', 1), {'T': 2, 'Z': 2, 'C': 1}, planes)
    assert len(digests) == 4 and plan['names'] == sorted(digests)
    assert (tmp_path / 'conversion_map.csv').read_text(encoding="utf-8").startswith('target,source,')


def test_old_czifile_without_czi_image_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, 'czifile', types.ModuleType('czifile'))
    with pytest.raises(ValueError, match=r'needs the optional reader czifile>=2026\.3\.12'):
        core._watch_vendor_reader('.czi')
    modern = types.ModuleType('czifile')
    modern.CziImage = object
    monkeypatch.setitem(sys.modules, 'czifile', modern)
    assert core._watch_vendor_reader('.czi') is modern


def test_vendor_field_flags_planes_outside_its_declared_sizes():
    field = core._watch_vendor_field('nd2', 'a.nd2', 'a', 'A01', 1, {'T': 1, 'Z': 1, 'C': 1},
                                     {(1, 0, 0): ('a.nd2', ('nd2', 0, 0))}, ['a.nd2'])
    assert field['reason'] == 'the metadata lists planes outside its declared dimensions'
    assert field['key'] == 'a_A01_1' and field['members'] == ['a.nd2']


class _Context:
    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        return self.handle

    def __exit__(self, *exc):
        return False


def _reader(monkeypatch, **classes):
    module = types.SimpleNamespace(**{name: (lambda path, h=handle: _Context(h))
                                      for name, handle in classes.items()})
    monkeypatch.setattr(core, '_watch_vendor_reader', lambda extension: module)


def test_nd2_with_an_unsupported_axis_is_refused(tmp_path, monkeypatch):
    _reader(monkeypatch, ND2File=types.SimpleNamespace(sizes={'Q': 2, 'Y': 4, 'X': 4}))
    with pytest.raises(ValueError, match=r"ND2 axes \['Q'\] are not supported"):
        core._watch_container_fields(str(tmp_path / 'a.nd2'), 'a.nd2')


def _czi(metadata, scenes):
    return types.SimpleNamespace(metadata=lambda: metadata, scenes=scenes)


def test_czi_waits_for_declared_scenes(tmp_path, monkeypatch):
    scene = types.SimpleNamespace(sizes={'T': 1, 'Y': 4, 'X': 4})
    _reader(monkeypatch, CziFile=_czi('<SizeS>3</SizeS>', {0: scene}))
    with pytest.raises(ValueError, match='1 of 3 declared scenes are in the file'):
        core._watch_container_fields(str(tmp_path / 'a.czi'), 'a.czi')


def test_czi_with_an_unsupported_axis_is_refused(tmp_path, monkeypatch):
    scene = types.SimpleNamespace(sizes={'A': 2, 'Y': 4, 'X': 4})
    _reader(monkeypatch, CziFile=_czi('', {0: scene}))
    with pytest.raises(ValueError, match=r"CZI axes \['A'\] are not supported"):
        core._watch_container_fields(str(tmp_path / 'a.czi'), 'a.czi')


def test_czi_scene_waits_for_declared_timepoints(tmp_path, monkeypatch):
    scene = types.SimpleNamespace(sizes={'T': 2, 'C': 1, 'Y': 4, 'X': 4})
    _reader(monkeypatch, CziFile=_czi('<SizeT>5</SizeT><SizeT>9</SizeT>', {7: scene}))
    with pytest.raises(ValueError, match='scene 7 holds 2 of 5 declared T positions'):
        core._watch_container_fields(str(tmp_path / 'a.czi'), 'a.czi')


def test_czi_declared_reads_first_size_per_axis():
    assert core._watch_czi_declared('<SizeT> 4 </SizeT><SizeZ>2</SizeZ><SizeT>9</SizeT>'
                                    '<SizeQ>3</SizeQ>') == {'T': 4, 'Z': 2}
    assert core._watch_czi_declared(None) == {}


def test_lif_image_with_unsupported_axes_waits_with_its_reason(tmp_path, monkeypatch):
    images = [types.SimpleNamespace(name='odd', sizes={'Q': 2, 'M': 3, 'Y': 4, 'X': 4}),
              types.SimpleNamespace(name='ok', sizes={'T': 2, 'M': 2, 'Y': 4, 'X': 4})]
    _reader(monkeypatch, LifFile=types.SimpleNamespace(images=images))
    fields = core._watch_container_fields(str(tmp_path / 'my run.lif'), 'my run.lif')
    assert [field['field'] for field in fields] == [1, 2, 3]
    assert fields[0]['reason'] == "LIF image 'odd' has unsupported axes ['Q']"
    assert fields[0]['planes'] == {} and fields[0]['plate'] == 'myrun'
    assert fields[1]['reason'] is None and fields[2]['reason'] is None
    assert fields[2]['planes'][(1, 0, 0)] == ('my run.lif', ('lif', 1, 1, 1, 0, 0))


def test_lif_without_images_is_not_a_field(tmp_path, monkeypatch):
    _reader(monkeypatch, LifFile=types.SimpleNamespace(images=[]))
    with pytest.raises(ValueError, match='the file declares no images'):
        core._watch_container_fields(str(tmp_path / 'a.lif'), 'a.lif')


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize('data', [{'Summary': 5}, ['Summary'], {}])
def test_micromanager_needs_a_summary(tmp_path, data):
    path = _write_json(tmp_path / 'metadata.txt', data)
    with pytest.raises(ValueError, match='metadata.txt has no Summary'):
        core._watch_micromanager_fields(str(path), 'metadata.txt')


def test_micromanager_ignores_entries_that_place_no_plane(tmp_path):
    path = _write_json(tmp_path / 'Pos0' / 'metadata.txt', {
        'Summary': {'Frames': 2, 'Slices': 1, 'Channels': 1, 'ChNames': ['DAPI'],
                    'PositionIndex': 3},
        'Extra': 7,
        'FrameKey-0-x-0': {'FileName': 'bad.tif'},
        'FrameKey-0-0': {'FileName': 'short.tif'},
        'FrameKey-1-4-0': {},
        'Other-0': {'FileName': 'other.tif'}})
    fields = core._watch_micromanager_fields(str(path), 'Pos0/metadata.txt')
    assert len(fields) == 1
    field = fields[0]
    assert field['plate'] == 'Pos0' and field['field'] == 4
    assert field['planes'] == {} and field['members'] == ['metadata.txt']
    assert field['reason'] == '0 of 2 declared planes are listed'


def test_micromanager_names_planes_from_channel_names(tmp_path):
    path = _write_json(tmp_path / 'acq_1' / 'Default' / 'metadata.txt', {
        'Summary': {'IntendedDimensions': {'time': 2, 'z': 1, 'channel': 1},
                    'ChNames': ['DAPI']},
        'FrameKey-0-0-0': {}, 'FrameKey-1-0-0': {'FileName': 'Default/t1.tif'}})
    field, = core._watch_micromanager_fields(str(path), 'acq_1/Default/metadata.txt')
    assert field['plate'] == 'acq1' and field['reason'] is None
    assert field['planes'] == {(0, 0, 0): ('img_000000000_DAPI_000.tif', ('page', 0)),
                               (1, 0, 0): ('t1.tif', ('page', 0))}


OME = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'


def _companion(path, body):
    path.write_text(f'<?xml version="1.0"?><OME xmlns="{OME}">{body}</OME>', encoding="utf-8")
    return str(path)


def test_ome_companion_image_needs_pixels(tmp_path):
    path = _companion(tmp_path / 'a.companion.ome', '<Image ID="Image:0"/>')
    with pytest.raises(ValueError, match='OME image 0 has no Pixels'):
        core._watch_ome_companion_fields(path, 'a.companion.ome')


def test_ome_companion_refuses_a_bad_dimension_order(tmp_path):
    path = _companion(tmp_path / 'a.companion.ome',
                      '<Image ID="Image:0"><Pixels DimensionOrder="XYZZT" SizeZ="2"/></Image>')
    with pytest.raises(ValueError, match="dimension order 'ZZT'"):
        core._watch_ome_companion_fields(path, 'a.companion.ome')


def test_ome_companion_without_images_is_refused(tmp_path):
    path = _companion(tmp_path / 'a.companion.ome', '')
    with pytest.raises(ValueError, match='the companion declares no images'):
        core._watch_ome_companion_fields(path, 'a.companion.ome')


def test_ome_companion_tiffdata_without_a_file_waits(tmp_path):
    path = _companion(tmp_path / 'a.companion.ome',
                      '<Image ID="Image:0"><Pixels DimensionOrder="XYZCT" SizeT="2">'
                      '<TiffData IFD="0"/></Pixels></Image>')
    field, = core._watch_ome_companion_fields(path, 'a.companion.ome')
    assert field['reason'] == 'a TiffData entry names no file'
    assert field['planes'] == {} and field['well'] == 'A01' and field['field'] == 1


def test_ome_companion_tiffdata_past_the_declared_planes_waits(tmp_path):
    path = _companion(tmp_path / 'a.companion.ome',
                      '<Image ID="Image:0"><Pixels DimensionOrder="XYZCT" SizeZ="2">'
                      '<TiffData FirstZ="1" PlaneCount="2"><UUID FileName="a.tif"/>'
                      '</TiffData></Pixels></Image>')
    field, = core._watch_ome_companion_fields(path, 'a.companion.ome')
    assert field['reason'] == 'a TiffData entry runs past the declared planes'
    assert field['sizes'] == {'T': 1, 'Z': 2, 'C': 1}


def _index(path, body):
    path.write_text(f'<EvaluationInputData xmlns="urn:h">{body}</EvaluationInputData>', encoding="utf-8")
    return str(path)


def test_harmony_entry_needs_url_and_position(tmp_path):
    path = _index(tmp_path / 'Index.xml',
                  '<Images><Image><URL>a.tiff</URL><Row>1</Row><Col>x</Col><FieldID>1</FieldID>'
                  '<TimepointID>0</TimepointID><PlaneID>1</PlaneID><ChannelID>1</ChannelID>'
                  '</Image></Images>')
    with pytest.raises(ValueError, match='lacks its URL or position'):
        core._watch_harmony_fields(path, 'Index.xml')


def test_harmony_index_without_images_is_refused(tmp_path):
    path = _index(tmp_path / 'Index.xml', '<Plates><Plate><Name>P1</Name></Plate></Plates>')
    with pytest.raises(ValueError, match='the index lists no images'):
        core._watch_harmony_fields(path, 'Index.xml')


def _pages(folder, shapes):
    planes = {}
    for t, shape in enumerate(shapes):
        name = f'p{t}.tif'
        tifffile.imwrite(folder / name, np.full(shape, t + 1, np.uint16))
        planes[(t, 0, 0)] = (name, ('page', 0))
    return planes


def test_planes_must_share_one_2d_shape(tmp_path):
    planes = _pages(tmp_path, [(4, 4), (5, 4)])
    with pytest.raises(ValueError, match='declared plane T2 Z1 C1 does not read whole'):
        list(core._watch_vendor_planes(str(tmp_path), {'planes': planes}))


def test_planes_reject_a_non_2d_plane(tmp_path):
    tifffile.imwrite(tmp_path / 'rgb.tif', np.zeros((4, 4, 3), np.uint8), photometric='rgb')
    description = {'planes': {(0, 0, 0): ('rgb.tif', ('page', 0))}}
    with pytest.raises(ValueError, match='declared plane T1 Z1 C1 does not read whole'):
        list(core._watch_vendor_planes(str(tmp_path), description))


def test_described_unreadable_reports_reason_mode_and_stop(tmp_path):
    planes = _pages(tmp_path, [(4, 4), (4, 4)])
    good = {'reason': None, 'sizes': {'T': 2, 'Z': 1, 'C': 1}, 'planes': planes}
    assert core._watch_described_unreadable(str(tmp_path), good, 'timelapse') is None
    assert core._watch_described_unreadable(
        str(tmp_path), dict(good, reason='still growing'), 'timelapse') == 'still growing'
    assert core._watch_described_unreadable(str(tmp_path), good, 'volume') == (
        'declared sizes T2 Z1 C1 do not fit the volume watch mode')
    (tmp_path / 'p1.tif').write_bytes(b'not a tiff')
    assert core._watch_described_unreadable(str(tmp_path), good, 'timelapse').startswith(
        'TiffFileError')
    planes = _pages(tmp_path, [(4, 4), (4, 4)])
    with _cancelled(), pytest.raises(PipelineCancelled, match='stop now'):
        core._watch_described_unreadable(str(tmp_path), good, 'timelapse')


def test_described_stage_refuses_a_copy_that_changed(tmp_path):
    snapshot = tmp_path / 'acq' / 'Pos0'
    _write_json(snapshot / 'metadata.txt', {
        'Summary': {'Frames': 2, 'ChNames': ['DAPI']},
        'FrameKey-0-0-0': {'FileName': 'a.tif'}, 'FrameKey-1-0-0': {'FileName': 'b.tif'}})
    field, = core._watch_vendor_describe(str(snapshot / 'metadata.txt'),
                                         'acq/Pos0/metadata.txt')
    description = dict(field, folder='acq/Pos0', sizes={'T': 3, 'Z': 1, 'C': 1})
    out = tmp_path / 'field'
    out.mkdir()
    with pytest.raises(ValueError, match='private copy no longer describes this field'):
        core._watch_described_stage(str(out), str(snapshot), description, 'timelapse',
                                    {'channels': [0]})
    assert os.listdir(out) == []


def test_a_lif_plane_is_read_once_per_file_and_selected_by_its_axes(
        tmp_path, monkeypatch):
    """A LIF locator opens the file once and reads one frame by T, M, Z, C.

    The reader is a stand-in so this runs where liffile is not installed,
    as on the coverage runners.
    """
    import types

    opened, frames = [], []

    class Image:
        sizes = {'T': 2, 'M': 3, 'C': 2}

        def __init__(self, index):
            self.index = index

        def frame(self, **selection):
            frames.append((self.index, selection))
            return [[self.index, selection.get('M', -1)]]

    class LifFile:
        def __init__(self, path):
            opened.append(path)
            self.images = [Image(0), Image(1)]

    monkeypatch.setattr(core, '_watch_vendor_reader',
                        lambda extension: types.SimpleNamespace(LifFile=LifFile)
                        if extension == '.lif' else None)
    handles = {}

    first = core._watch_vendor_plane(handles, str(tmp_path), 'run.lif',
                                     ('lif', 1, 1, 2, 0, 1))
    second = core._watch_vendor_plane(handles, str(tmp_path), 'run.lif',
                                      ('lif', 0, 0, 1, 3, 0))

    assert opened == [os.path.join(str(tmp_path), 'run.lif')]
    assert first.tolist() == [[1, 2]] and second.tolist() == [[0, 1]]
    assert frames == [(1, {'T': 1, 'M': 2, 'C': 1}),
                      (0, {'T': 0, 'M': 1, 'C': 0})]
