"""Bounded local hints retain actual source scope and leave microscope files untouched."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

from spacr.qt.widgets import home


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.arange(48, dtype=np.uint16).reshape(6, 8)).save(path)
    return path


def _digest(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_dropped_microscope_image_uses_siblings_and_bounded_local_metadata(tmp_path, monkeypatch):
    folder = tmp_path / 'acquisition'
    first = _image(folder / 'plate_A01_T0001F001L01A01Z01C01.tif')
    second = _image(folder / 'plate_A01_T0001F001L01A01Z01C02.tif')
    metadata = folder / 'metadata.json'
    metadata.write_text(json.dumps({'microscope': 'not proven', 'channels': [1, 2],
                                    'pixel_size_um': .5, 'unrelated': 'not a hint'}))
    before = _digest([first, second, metadata])

    def no_pixels(*args, **kwargs):
        raise AssertionError('inference must not decode source pixels')

    monkeypatch.setattr(TiffImagePlugin.TiffImageFile, 'load', no_pixels)
    result = home._easy_inspect_path(str(first))
    assert result['actions'][0][3] == str(first)
    assert next(action for action in result['actions'] if action[0] == 'mask')[3] == str(folder)
    assert any(hint['convention'] == 'cellvoyager' and hint['matched'] == 2
               for hint in result['context']['filename_matches'])
    sidecar = next(hint for hint in result['context']['metadata_files'] if hint['name'] == metadata.name)
    assert sidecar['fields'] == ['channels', 'microscope', 'pixel_size_um']
    assert 'not verified acquisition metadata' in result['message']
    assert 'not proven' not in result['message']
    assert result['confidence'] == 'filename_hint'
    assert _digest([first, second, metadata]) == before
    json.dumps(result)


def test_master_folder_offers_actual_child_inputs_and_explicit_ambiguity(tmp_path):
    master = tmp_path / 'screen'
    first = _image(master / 'plateA' / 'field.tif')
    second = _image(master / 'plateB' / 'field.tif')
    before = _digest([first, second])
    result = home._easy_inspect_path(str(master))
    masks = [Path(action[3]) for action in result['actions'] if action[0] == 'mask']
    assert first.parent in masks and second.parent in masks
    assert master not in masks
    assert result['actions'][0][3] == str(first.parent)
    assert 'nearby folders' in result['message'] and str(first.parent) in result['message']
    assert _digest([first, second]) == before


def test_child_folder_can_find_master_metadata_and_sibling_image_groups(tmp_path):
    master = tmp_path / 'screen'
    first = _image(master / 'plateA' / 'field.tif')
    second = _image(master / 'plateB' / 'field.tif')
    sidecar = master / 'MeasurementProtocol.xml'
    sidecar.write_text('<protocol />')
    result = home._easy_inspect_path(str(first))
    assert str(second.parent) in result['context']['image_folders']
    assert any(hint['folder'] == str(master) and hint['name'] == sidecar.name
               for hint in result['context']['metadata_files'])
    assert next(action for action in result['actions'] if action[0] == 'mask')[3] == str(first.parent)


def test_global_scan_and_metadata_budgets_do_not_read_large_sidecars(tmp_path, monkeypatch):
    folder = tmp_path / 'many'
    folder.mkdir()
    for index in range(150):
        (folder / f'unknown-{index}.txt').write_text('not a data table')
    sidecar = folder / 'metadata.json'
    sidecar.write_bytes(b'{' + b' ' * (home._EASY_METADATA_LIMIT * 3))
    calls = []
    read = home._easy_read_header

    def bounded(path, limit):
        calls.append((Path(path), limit))
        return read(path, limit)

    monkeypatch.setattr(home, '_easy_read_header', bounded)
    result = home._easy_inspect_path(str(folder))
    assert result['context']['sampled_files'] <= home._EASY_FILE_LIMIT
    assert result['context']['sampled_directories'] <= home._EASY_DIRECTORY_LIMIT
    assert result['context']['truncated']
    assert 'bounded sample' in result['message']
    assert all(limit <= home._EASY_METADATA_LIMIT + 1 for _, limit in calls)
    assert sum(limit > home._EASY_HEADER_LIMIT for _, limit in calls) <= 4
    assert next(hint for hint in result['context']['metadata_files'] if hint['name'] == 'metadata.json')['truncated']


def test_symlinks_and_deep_folders_are_not_followed_or_used_as_source_inputs(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    image = _image(tmp_path / 'outside' / 'field.tif')
    link = root / 'shortcut.tif'
    link.symlink_to(image)
    deep = _image(root / 'child' / 'grandchild' / 'deep.tif')
    result = home._easy_inspect_path(str(link))
    assert not result['actions'] and 'symbolic link' in result['message']
    sample, _, _, _ = home._easy_local_sample(root)
    assert link not in sample and deep not in sample


def test_bad_extension_does_not_silently_substitute_a_nearby_valid_image(tmp_path):
    image = _image(tmp_path / 'field.tif')
    renamed = tmp_path / 'field.bin'
    renamed.write_bytes(image.read_bytes())
    result = home._easy_inspect_path(str(renamed))
    assert not result['actions']
    assert 'correctly named copy' in result['message']
    assert renamed.read_bytes() == image.read_bytes()


def test_explicit_table_keeps_its_own_import_action_among_many_image_children(tmp_path):
    master = tmp_path / 'screen'
    images = [_image(master / name / 'field.tif') for name in ('plateA', 'plateB', 'plateC')]
    table = master / 'objects.csv'
    table.write_text('label,area\n1,24\n2,12\n')
    before = _digest([table, *images])
    result = home._easy_inspect_path(str(table))
    assert len(result['actions']) <= 4
    assert result['actions'][0][0] == 'foreign'
    assert result['actions'][0][3] == str(table)
    assert 'nearby folders' in result['message']
    assert _digest([table, *images]) == before


def test_unknown_home_account_is_ordinary_local_input_guidance():
    from spacr.qt.i18n import tr

    result = home._easy_inspect_path('~spacr_no_such_local_account_733/input.tif')
    assert not result['actions']
    assert result['message'] == tr('Choose an existing local file or folder.')
