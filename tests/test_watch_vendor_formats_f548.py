"""Live watch of companion-described per-plane series and ND2/CZI/LIF containers.

Micro-Manager ``metadata.txt``, OME ``*.companion.ome`` and Opera/Operetta
``Index.xml`` declare every plane of a per-plane series; ND2, CZI and LIF
files declare their own frames, scenes and positions. A field is analysed
only once every declared plane is present and decodes, each position is one
field, and the live result equals a batch run on the finished acquisition.
"""

import csv
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import convert, core
from tests.test_native_tzyx_batch_f548 import _settings as _series_settings
from tests.test_watch_folder_and_analyse import MASK, Recorder, real_pipeline
from tests.test_watch_native_t_series_f548 import _model
from tests.test_watch_native_z_t1_f548 import _mask as _volume_mask
from tests.test_watch_nested_f548 import _fast

pytest_plugins = ['tests.test_cov_object_masks_sam']

DATA = Path(__file__).parent / 'data' / 'watch_vendor_f548'

_OME = 'https://downloads.openmicroscopy.org/images/'
_PUBLIC = {
    'header_test2.nd2': _OME + 'ND2/jonas/header_test2.nd2',
    'MeOh_high_fluo_003.nd2': _OME + 'ND2/aryeh/MeOh_high_fluo_003.nd2',
    'FRAP.lif': (_OME + 'Leica-LIF/seanwarren/150519_FRAP_test_ROIs_chromagreen/'
                 '150519_FRAP_test_ROIs_chromagreen.lif'),
    'omer_Index.idx.xml': (_OME + 'PerkinElmer-Operetta/omer/006P_M3/'
                           '006P__2017-08-19T12_42_59-Measurement%203/Images/Index.idx.xml'),
}
_MM14 = _OME + 'Micro-Manager/1.4.22/thomas/test/test_sep/Pos0/'


def _public(name, url=None):
    """Download one public CC BY 4.0 sample into a cache, or skip when offline."""
    cache = Path(os.environ.get('SPACR_TEST_DOWNLOADS',
                                Path(tempfile.gettempdir()) / 'spacr_test_downloads'))
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.part')
        try:
            urllib.request.urlretrieve(url or _PUBLIC[name], partial)
        except OSError as exc:
            pytest.skip(f'public sample {name} unavailable: {exc}')
        os.replace(partial, path)
    return path


def czi_plane(scene, t, z, c, size=32):
    """The known plane zeiss_czi/write_czi.py wrote into the CZI fixtures."""
    plane = np.full((size, size), 40, np.uint16)
    plane[5:17, 5:17] = 500 + 37 * scene + 11 * t + 7 * z + 3 * c
    plane[18:29, 18:29] = 800 + 37 * scene + 11 * t + 7 * z + 3 * c
    return plane


def _batch(root, stacks, mode):
    """Write the finished acquisition's planes for the batch route of ``mode``.

    :param stacks: ``{(plate, well, field): TZCYX array}`` read independently
        of the watcher.
    """
    batch = root / 'batch'
    batch.mkdir()
    rows = []
    for (plate, well, field), stack in sorted(stacks.items()):
        times, planes, channels = stack.shape[:3]
        if mode == 'volume':
            for c in range(channels):
                tifffile.imwrite(batch / convert.target_name(plate, well, field, c + 1),
                                 stack[0, :, c], metadata={'axes': 'ZYX'},
                                 photometric='minisblack')
            continue
        for t, z, c in np.ndindex(times, planes, channels):
            name = convert.target_name(plate, well, field, channel=c + 1, z=z + 1, t=t + 1)
            tifffile.imwrite(batch / name, stack[t, z, c], metadata={'axes': 'YX'})
            rows.append({'target': name, 'source': 'acquired', 'plate': plate,
                         'well': well, 'field': field, 'channel': c + 1, 'z': z + 1,
                         't': t + 1})
    if mode == 'series':
        with open(batch / convert.MAP_FILENAME, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return batch


def _same_merged(batch, watched, count):
    names = sorted(path.name for path in (batch / 'merged').glob('*.npy'))
    assert len(names) == count
    assert names == sorted(path.name for path in
                           (watched / 'spacr_watch/merged').glob('*.npy'))
    for name in names:
        np.testing.assert_array_equal(np.load(batch / 'merged' / name),
                                      np.load(watched / 'spacr_watch/merged' / name))


def _staged(watched, key):
    return watched / 'spacr_watch' / 'fields' / key


def _timelapse(folder, **changes):
    return {**MASK, **_fast(folder), 'timelapse': True, **changes}


def _series(folder, **changes):
    settings = {**_series_settings(folder, cell_channel=None), **_fast(folder),
                'watch_pipeline': 'mask'}
    settings.update(changes)
    return settings


def _mm_stack(folder):
    """Read the Micro-Manager 2.0 sample by its file names, without metadata."""
    return np.stack([np.stack([tifffile.imread(
        folder / f'img_channel{c:03d}_position000_time{t:09d}_z000.tif')
        for c in range(2)])[None] for t in range(2)])


def test_micromanager_waits_for_its_metadata_then_matches_batch(tmp_path, real_pipeline):
    source = DATA / 'micromanager_2_0' / 'Pos0'
    stack = _mm_stack(source)
    batch = _batch(tmp_path, {('acq', 'A01', 1): stack}, 'timelapse')
    core.preprocess_generate_masks(dict(MASK, timelapse=True, src=str(batch)))
    watched = tmp_path / 'watched'
    folder = watched / 'acq' / 'Pos0'
    folder.mkdir(parents=True)
    for path in sorted(source.glob('*.tif'))[:3]:
        shutil.copy2(path, folder / path.name)
    first = core._watch_folder_and_analyse(_timelapse(watched))
    assert not first['done'] and not (watched / 'spacr_watch/fields').exists()
    text = (source / 'metadata.txt').read_text()
    (folder / 'metadata.txt').write_text(text[:len(text) // 2])
    second = core._watch_folder_and_analyse(_timelapse(watched))
    assert not second['done'] and not (watched / 'spacr_watch/fields').exists()
    (folder / 'metadata.txt').write_text(text)
    third = core._watch_folder_and_analyse(_timelapse(watched))
    assert not third['done'] and third['incomplete'] == ['acq_A01_1']
    for path in sorted(source.glob('*.tif'))[3:]:
        shutil.copy2(path, folder / path.name)
    result = core._watch_folder_and_analyse(_timelapse(watched))
    assert result['done'] == ['acq_A01_1'] and not result['failed']
    assert not result['incomplete']
    staged = _staged(watched, 'acq_A01_1')
    assert sorted(path.name for path in (staged / '.watch_vendor').iterdir()) == sorted(
        path.name for path in source.iterdir())
    _same_merged(batch, watched, 2)
    again = core._watch_folder_and_analyse(_timelapse(watched))
    assert again['done'] == ['acq_A01_1']


def test_micromanager_frame_keys_without_file_names_use_the_naming_rule(tmp_path):
    summary = {'Frames': 2, 'Slices': 1, 'Channels': 2, 'ChNames': ['DAPI', 'GFP'],
               'PositionIndex': 3}
    data = {'Summary': summary}
    for t, c in np.ndindex(2, 2):
        data[f'FrameKey-{t}-{c}-0'] = {'Channel': summary['ChNames'][c]}
    path = tmp_path / 'run_1' / 'Pos3' / 'metadata.txt'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    (field,) = core._watch_micromanager_fields(str(path), 'run_1/Pos3/metadata.txt')
    assert field['key'] == 'run1_A01_4' and field['reason'] is None
    assert field['planes'][(1, 0, 1)] == ('img_000000001_GFP_000.tif', ('page', 0))
    del data['FrameKey-1-1-0']
    path.write_text(json.dumps(data))
    (field,) = core._watch_micromanager_fields(str(path), 'run_1/Pos3/metadata.txt')
    assert field['reason'] == '3 of 4 declared planes are listed'


def test_real_micromanager_1_4_metadata_places_every_plane(tmp_path):
    folder = tmp_path / 'test_sep' / 'Pos0'
    folder.mkdir(parents=True)
    names = ['metadata.txt'] + [f'img_{t:09d}_{channel}_000.tif' for t in range(4)
                                for channel in ('FITC', 'Rhodamine')]
    for name in names:
        shutil.copy2(_public('mm14_' + name, _MM14 + name), folder / name)
    (field,) = core._watch_micromanager_fields(str(folder / 'metadata.txt'),
                                               'test_sep/Pos0/metadata.txt')
    assert field['key'] == 'testsep_A01_1' and field['sizes'] == {'T': 4, 'Z': 1, 'C': 2}
    assert field['reason'] is None and set(field['members']) == set(names)
    assert core._watch_described_unreadable(str(folder), field, 'timelapse') is None
    assert 'do not fit the volume' in core._watch_described_unreadable(
        str(folder), field, 'volume')


def test_ome_companion_volume_waits_for_every_file_then_matches_batch(
        tmp_path, fake_model, monkeypatch):
    seen = _model(monkeypatch)
    source = DATA / 'ome_companion'
    volume = np.stack([tifffile.imread(source / f'multifile-Z{z}.ome.tiff')
                       for z in range(1, 6)])
    settings = {'channels': [0], 'cell_channel': None}
    batch = _batch(tmp_path, {('multifile', 'A01', 1): volume[None, :, None]}, 'volume')
    core.preprocess_generate_masks(dict(_volume_mask(batch), **settings))
    batch_calls = len(seen)
    watched = tmp_path / 'watched'
    watched.mkdir()
    shutil.copy2(source / 'multifile.companion.ome', watched)
    for z in range(1, 5):
        shutil.copy2(source / f'multifile-Z{z}.ome.tiff', watched)
    (watched / 'multifile-Z5.ome.tiff').write_bytes(
        (source / 'multifile-Z5.ome.tiff').read_bytes()[:400])
    watch = {**_volume_mask(watched), **_fast(watched), **settings}
    first = core._watch_folder_and_analyse(watch)
    assert not first['done'] and first['incomplete'] == ['multifile_A01_1']
    assert len(seen) == batch_calls
    os.remove(watched / 'multifile-Z5.ome.tiff')
    shutil.copy2(source / 'multifile-Z5.ome.tiff', watched)
    result = core._watch_folder_and_analyse(watch)
    assert result['done'] == ['multifile_A01_1'] and not result['failed']
    staged = _staged(watched, 'multifile_A01_1')
    assert len(list((staged / '.watch_planar').glob('*.tif'))) == 5
    _same_merged(batch, watched, 1)
    assert len(seen) == 2 * batch_calls


def test_plate_companion_names_real_wells_and_fields(tmp_path):
    fields = core._watch_ome_companion_fields(
        str(DATA / 'ome_plate_companion' / 'hcs.companion.ome'), 'hcs.companion.ome')
    assert [field['key'] for field in fields] == [
        'hcs_A02_1', 'hcs_B01_1', 'hcs_B03_1', 'hcs_C02_1', 'hcs_C02_2']
    assert fields[4]['members'] == ['hcs.companion.ome', 'well-C2-2.ome.tiff']
    shutil.copytree(DATA / 'ome_plate_companion', tmp_path / 'plate')
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_timelapse(tmp_path), recorder)
    assert not recorder.calls and len(result['incomplete']) == 5


def _harmony_index(path, plate, images):
    """Write a Harmony V5 export index listing ``(row, col, field, t, z, c, url)``."""
    entries = ''.join(
        f'<Image Version="1"><id>{row:02d}{col:02d}K1F{field}P{z}R{c}</id><State>Ok</State>'
        f'<URL>{url}</URL><Row>{row}</Row><Col>{col}</Col><FieldID>{field}</FieldID>'
        f'<PlaneID>{z}</PlaneID><TimepointID>{t}</TimepointID><ChannelID>{c}</ChannelID>'
        f'<ChannelName>ch{c}</ChannelName></Image>'
        for row, col, field, t, z, c, url in images)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<EvaluationInputData xmlns:xsd="http://www.w3.org/2001/XMLSchema" Version="1" '
        'xmlns="http://www.perkinelmer.com/PEHH/HarmonyV5"><Plates><Plate>'
        f'<PlateID>{plate}</PlateID><Name>{plate}</Name></Plate></Plates>'
        f'<Images>{entries}</Images></EvaluationInputData>', encoding='utf-8')


def test_harmony_index_takes_complete_fields_and_matches_batch(tmp_path, real_pipeline):
    images, planes = [], {}
    for field in (1, 2):
        for t, c in np.ndindex(2, 2):
            url = f'r02c03f{field:02d}p01-ch{c + 1}sk{t + 1}fk1fl1.tiff'
            images.append((2, 3, field, t, 1, c + 1, url))
            plane = np.full((32, 32), 40, np.uint16)
            plane[5:17, 5:17] = 500 + 50 * field + 10 * t + c
            plane[18:29, 18:29] = 900 + 50 * field + 10 * t + c
            planes[url] = plane
    stacks = {('P1', 'B03', field): np.stack([np.stack([
        planes[f'r02c03f{field:02d}p01-ch{c + 1}sk{t + 1}fk1fl1.tiff']
        for c in range(2)])[None] for t in range(2)]) for field in (1, 2)}
    batch = _batch(tmp_path, stacks, 'timelapse')
    core.preprocess_generate_masks(dict(MASK, timelapse=True, src=str(batch)))
    watched = tmp_path / 'watched' / 'Images'
    watched.mkdir(parents=True)
    _harmony_index(watched / 'Index.xml', 'P-1', images)
    late = 'r02c03f02p01-ch2sk2fk1fl1.tiff'
    for url, plane in planes.items():
        if url != late:
            tifffile.imwrite(watched / url, plane)
    root = watched.parent
    first = core._watch_folder_and_analyse(_timelapse(root))
    assert first['done'] == ['P1_B03_1'] and first['incomplete'] == ['P1_B03_2']
    tifffile.imwrite(watched / late, planes[late])
    result = core._watch_folder_and_analyse(_timelapse(root))
    assert result['done'] == ['P1_B03_1', 'P1_B03_2'] and not result['incomplete']
    _same_merged(batch, root, 4)


def test_real_harmony_index_declares_every_field(tmp_path):
    index = _public('omer_Index.idx.xml')
    fields = core._watch_harmony_fields(str(index), 'Images/Index.idx.xml')
    assert len(fields) == 29
    assert {field['key'] for field in fields} == {f'006P_A02_{n}' for n in range(1, 30)}
    for field in fields:
        assert field['sizes'] == {'T': 1, 'Z': 18, 'C': 6} and field['reason'] is None
        assert len(field['members']) == 109
    assert fields[0]['planes'][(0, 0, 0)] == ('r01c02f01p01-ch1sk1fk1fl1.tiff', ('page', 0))


def test_czi_scenes_are_fields_and_wait_while_the_file_grows(tmp_path, fake_model,
                                                             monkeypatch):
    pytest.importorskip('czifile')
    seen = _model(monkeypatch)
    data = (DATA / 'zeiss_czi' / 'two_scenes_tzc.czi').read_bytes()
    stacks = {('twoscenestzc', 'A01', scene + 1): np.stack([np.stack([np.stack([
        czi_plane(scene, t, z, c) for c in range(2)]) for z in range(3)])
        for t in range(2)]) for scene in range(2)}
    batch = _batch(tmp_path, stacks, 'series')
    core.preprocess_generate_masks(_series_settings(batch, cell_channel=None))
    batch_calls = len(seen)
    watched = tmp_path / 'watched'
    watched.mkdir()
    acquired = watched / 'two_scenes_tzc.czi'
    acquired.write_bytes(data[:len(data) // 2])
    first = core._watch_folder_and_analyse(_series(watched))
    assert not first['done'] and first['incomplete'] == ['twoscenestzc']
    with open(acquired, 'ab') as handle:
        handle.write(data[len(data) // 2:])
    result = core._watch_folder_and_analyse(_series(watched))
    keys = ['twoscenestzc_A01_1', 'twoscenestzc_A01_2']
    assert result['done'] == keys and not result['failed']
    copies = [_staged(watched, key) / '.watch_vendor' / acquired.name for key in keys]
    assert copies[0].stat().st_ino == copies[1].stat().st_ino
    assert copies[0].read_bytes() == data
    for scene, key in enumerate(keys):
        for t, z, c in np.ndindex(2, 3, 2):
            name = convert.target_name('twoscenestzc', 'A01', scene + 1,
                                       channel=c + 1, z=z + 1, t=t + 1)
            np.testing.assert_array_equal(tifffile.imread(_staged(watched, key) / name),
                                          czi_plane(scene, t, z, c))
    _same_merged(batch, watched, 4)
    assert len(seen) == 2 * batch_calls


def test_czi_timelapse_scenes_match_the_projected_batch(tmp_path, real_pipeline):
    pytest.importorskip('czifile')
    stacks = {('twoscenestc', 'A01', scene + 1): np.stack([np.stack([
        czi_plane(scene, t, 0, c) for c in range(2)])[None] for t in range(3)])
        for scene in range(2)}
    batch = _batch(tmp_path, stacks, 'timelapse')
    core.preprocess_generate_masks(dict(MASK, timelapse=True, src=str(batch)))
    watched = tmp_path / 'watched'
    watched.mkdir()
    shutil.copy2(DATA / 'zeiss_czi' / 'two_scenes_tc.czi', watched)
    result = core._watch_folder_and_analyse(_timelapse(watched))
    assert result['done'] == ['twoscenestc_A01_1', 'twoscenestc_A01_2']
    _same_merged(batch, watched, 6)


def test_lif_tile_positions_are_fields_and_match_batch(tmp_path, fake_model, monkeypatch):
    liffile = pytest.importorskip('liffile')
    seen = _model(monkeypatch)
    source = DATA / 'leica_lif' / 'PR2729_frameOrderCombinedScanTypes.lif'
    with liffile.LifFile(source) as handle:
        whole = handle.images[0].asarray()
    assert whole.shape == (2, 4, 3, 2, 64, 64)
    stacks = {('PR2729frameOrderCombinedScanTypes', 'A01', tile + 1): whole[:, tile]
              for tile in range(4)}
    batch = _batch(tmp_path, stacks, 'series')
    core.preprocess_generate_masks(_series_settings(batch, cell_channel=None))
    batch_calls = len(seen)
    watched = tmp_path / 'watched'
    watched.mkdir()
    data = source.read_bytes()
    (watched / source.name).write_bytes(data[:len(data) // 2])
    first = core._watch_folder_and_analyse(_series(watched))
    assert not first['done'] and len(first['incomplete']) == 4
    (watched / source.name).write_bytes(data)
    result = core._watch_folder_and_analyse(_series(watched))
    assert result['done'] == [f'PR2729frameOrderCombinedScanTypes_A01_{n}'
                              for n in range(1, 5)]
    _same_merged(batch, watched, 8)
    assert len(seen) == 2 * batch_calls


def test_real_frap_lif_series_are_timelapse_fields(tmp_path, real_pipeline):
    liffile = pytest.importorskip('liffile')
    source = _public('FRAP.lif')
    with liffile.LifFile(source) as handle:
        stacks = {('FRAP', 'A01', index + 1): image.asarray()[:, None, None]
                  for index, image in enumerate(handle.images)}
    assert len(stacks) == 8
    settings = {'channels': [0], 'nucleus_channel': 0, 'cell_channel': None}
    batch = _batch(tmp_path, stacks, 'timelapse')
    core.preprocess_generate_masks(dict(MASK, timelapse=True, src=str(batch), **settings))
    watched = tmp_path / 'watched'
    watched.mkdir()
    shutil.copy2(source, watched / 'FRAP.lif')
    result = core._watch_folder_and_analyse(_timelapse(watched, **settings))
    assert result['done'] == sorted(f'FRAP_A01_{n}' for n in range(1, 9))
    _same_merged(batch, watched, 40)


def _nd2_reference(path, times, planes):
    """Read a whole ND2 at once with nd2.imread, not frame by frame."""
    import nd2

    stack = np.asarray(nd2.imread(str(path)))
    return stack.reshape(times, planes, 1, *stack.shape[-2:])


def test_real_nd2_t_by_z_waits_while_growing_then_matches_batch(tmp_path, fake_model,
                                                                monkeypatch):
    pytest.importorskip('nd2')
    source = _public('header_test2.nd2')
    stack = _nd2_reference(source, 4, 5)
    seen = _model(monkeypatch)
    batch = _batch(tmp_path, {('headertest2', 'A01', 1): stack}, 'series')
    core.preprocess_generate_masks(_series_settings(batch, cell_channel=None, channels=[0]))
    batch_calls = len(seen)
    watched = tmp_path / 'watched'
    watched.mkdir()
    data = source.read_bytes()
    (watched / source.name).write_bytes(data[:len(data) * 2 // 3])
    first = core._watch_folder_and_analyse(_series(watched, channels=[0]))
    assert not first['done'] and first['incomplete'] == ['headertest2']
    (watched / source.name).write_bytes(data)
    result = core._watch_folder_and_analyse(_series(watched, channels=[0]))
    assert result['done'] == ['headertest2_A01_1'] and not result['failed']
    _same_merged(batch, watched, 4)
    assert len(seen) == 2 * batch_calls


def test_real_nd2_timelapse_matches_projected_batch(tmp_path, real_pipeline):
    pytest.importorskip('nd2')
    source = _public('MeOh_high_fluo_003.nd2')
    stack = _nd2_reference(source, 13, 1)
    settings = {'channels': [0], 'nucleus_channel': 0, 'cell_channel': None}
    batch = _batch(tmp_path, {('MeOhhighfluo003', 'A01', 1): stack}, 'timelapse')
    core.preprocess_generate_masks(dict(MASK, timelapse=True, src=str(batch), **settings))
    watched = tmp_path / 'watched'
    watched.mkdir()
    shutil.copy2(source, watched / source.name)
    result = core._watch_folder_and_analyse(_timelapse(watched, **settings))
    assert result['done'] == ['MeOhhighfluo003_A01_1']
    _same_merged(batch, watched, 13)


class _Loop:
    def __init__(self, count):
        self.count = count


class _FakeND2:
    """A two-position ND2 reader double with P, T and C loops."""

    written = 4

    def __init__(self, _path):
        self.sizes = {'P': 2, 'T': 2, 'C': 2, 'Y': 4, 'X': 5}
        self.experiment = [_Loop(2), _Loop(2)]
        self.loop_indices = tuple({'T': t, 'P': p} for t in range(2) for p in range(2))
        self.attributes = type('A', (), {'sequenceCount': self.written})()

    def read_frame(self, index):
        return np.full((2, 4, 5), index, np.uint16) + np.arange(2)[:, None, None]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def test_nd2_positions_become_fields_and_missing_frames_wait(tmp_path, monkeypatch):
    fake = type('nd2', (), {'ND2File': _FakeND2})
    monkeypatch.setattr(core, '_watch_vendor_reader', lambda extension: fake)
    first, second = core._watch_container_fields(str(tmp_path / 'acq.nd2'), 'run/acq.nd2')
    assert (first['key'], second['key']) == ('acq_A01_1', 'acq_A01_2')
    assert first['sizes'] == {'T': 2, 'Z': 1, 'C': 2} and first['reason'] is None
    assert second['planes'][(1, 0, 1)] == ('acq.nd2', ('nd2', 3, 1))
    plane = core._watch_vendor_plane({}, str(tmp_path), 'acq.nd2', ('nd2', 3, 1))
    np.testing.assert_array_equal(plane, np.full((4, 5), 4, np.uint16))
    monkeypatch.setattr(_FakeND2, 'written', 3)
    with pytest.raises(ValueError, match='3 of 4 declared frames'):
        core._watch_container_fields(str(tmp_path / 'acq.nd2'), 'acq.nd2')


def test_a_missing_reader_names_the_optional_extra(monkeypatch):
    import importlib

    real = importlib.import_module

    def refuse(name, *args):
        if name == 'liffile':
            raise ImportError('not installed')
        return real(name, *args)

    monkeypatch.setattr(importlib, 'import_module', refuse)
    with pytest.raises(ValueError, match=r'spacr\[vendor-watch\]'):
        core._watch_vendor_reader('.lif')


def test_unlisted_tiffs_beside_a_companion_are_not_analysed(tmp_path, capsys):
    shutil.copytree(DATA / 'ome_companion', tmp_path / 'set')
    tifffile.imwrite(tmp_path / 'set' / 'stray.ome.tif',
                     np.ones((2, 1, 8, 8), np.uint16), ome=True,
                     metadata={'axes': 'TCYX'})
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_timelapse(tmp_path), recorder)
    assert not recorder.calls and result['incomplete'] == ['multifile_A01_1']
    assert 'stray.ome.tif sits beside an acquisition companion' in capsys.readouterr().out


def test_a_growing_czi_is_taken_only_after_it_settles(tmp_path):
    import threading
    import time

    pytest.importorskip('czifile')
    data = (DATA / 'zeiss_czi' / 'two_scenes_tc.czi').read_bytes()
    path = tmp_path / 'grow.czi'
    path.write_bytes(data[:4096])
    finished = []

    def writer():
        with open(path, 'ab') as handle:
            for start in range(4096, len(data), 4096):
                time.sleep(0.1)
                handle.write(data[start:start + 4096])
                handle.flush()
        finished.append(time.time())

    thread = threading.Thread(target=writer)
    recorder = Recorder()
    thread.start()
    try:
        result = core._watch_folder_and_analyse(
            {**_timelapse(tmp_path), 'watch_settle_seconds': 0.4,
             'watch_idle_minutes': 0.05}, recorder)
    finally:
        thread.join()
    assert result['done'] == ['grow_A01_1', 'grow_A01_2']
    assert [call[0] for call in recorder.calls] == result['done']
    assert all(call[2] >= finished[0] + 0.4 for call in recorder.calls)
