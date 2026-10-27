"""Bounded local hints retain actual source scope and leave microscope files untouched."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
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


def test_child_keeps_its_scope_and_explicit_master_exposes_sibling_context(tmp_path):
    master = tmp_path / 'screen'
    first = _image(master / 'plateA' / 'field.tif')
    second = _image(master / 'plateB' / 'field.tif')
    sidecar = master / 'MeasurementProtocol.xml'
    sidecar.write_text('<protocol />')
    result = home._easy_inspect_path(str(first))
    assert result['context']['image_folders'] == [str(first.parent)]
    assert not result['context']['metadata_files']
    assert next(action for action in result['actions'] if action[0] == 'mask')[3] == str(first.parent)
    result = home._easy_inspect_path(str(master))
    assert {str(first.parent), str(second.parent)} <= set(result['context']['image_folders'])
    assert any(hint['folder'] == str(master) and hint['name'] == sidecar.name
               for hint in result['context']['metadata_files'])


def test_global_scan_and_metadata_budgets_do_not_read_large_sidecars(tmp_path, monkeypatch):
    folder = tmp_path / 'many'
    folder.mkdir()
    for index in range(150):
        (folder / f'unknown-{index}.txt').write_text('not a data table')
    sidecar = folder / 'metadata.json'
    sidecar.write_bytes(b'{' + b' ' * (home._EASY_METADATA_LIMIT * 3))
    calls = []
    read = home._easy_read_header

    def bounded(path, limit, offset=0):
        calls.append((Path(path), limit))
        return read(path, limit, offset)

    monkeypatch.setattr(home, '_easy_read_header', bounded)
    result = home._easy_inspect_path(str(folder))
    assert result['context']['sampled_files'] <= home._EASY_FILE_LIMIT
    assert result['context']['sampled_directories'] <= home._EASY_DIRECTORY_LIMIT
    assert result['context']['filename_sample_limit'] == 1000
    assert result['context']['header_files'] <= 16
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
    assert {'foreign', 'graph_builder', 'gate_editor'} <= {action[0] for action in result['actions']}
    assert result['actions'][0][0] == 'foreign'
    assert result['actions'][0][3] == str(table)
    assert {action[0] for action in result['actions']} == {'foreign', 'graph_builder', 'gate_editor'}
    assert result['context']['image_folders'] == []
    assert 'regex' not in result
    assert 'nearby folders' not in result['message']
    assert _digest([table, *images]) == before


def test_unknown_home_account_is_ordinary_local_input_guidance():
    from spacr.qt.i18n import tr

    result = home._easy_inspect_path('~spacr_no_such_local_account_733/input.tif')
    assert not result['actions']
    assert result['message'] == tr('Choose an existing local file or folder.')


def _late_ifd_image(path, big, endian='<'):
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.arange(4096, dtype=endian + 'u2').reshape(64, 64).tobytes()
    start = 16 if big else 8
    offset = start + len(pixels)
    tags = [(256, 4, 64), (257, 4, 64), (258, 3, 16), (259, 3, 1),
            (262, 3, 1), (273, 4 if big else 3, start), (277, 3, 1), (278, 4, 64),
            (279, 4, len(pixels))]
    marker = b'II' if endian == '<' else b'MM'
    header = struct.pack(endian + '2sHHHQ', marker, 43, 8, 0, offset) if big else struct.pack(endian + '2sHI', marker, 42, offset)
    entries = b''.join(struct.pack(endian + ('HHQ' if big else 'HHI'), tag, kind, 1)
                       + struct.pack(endian + ('H' if kind == 3 else 'I'), value)
                       + bytes((8 if big else 4) - (2 if kind == 3 else 4))
                       for tag, kind, value in tags)
    path.write_bytes(header + pixels + struct.pack(endian + ('Q' if big else 'H'), len(tags))
                     + entries + bytes(8 if big else 4))
    return path


@pytest.mark.parametrize('big', [False, True], ids=['classic-tiff', 'bigtiff'])
@pytest.mark.parametrize('endian', ['<', '>'], ids=['little-endian', 'big-endian'])
def test_cq1_late_ifd_image_keeps_source_and_master_outputs_need_explicit_selection(tmp_path, monkeypatch, big, endian):
    import sqlite3
    import tifffile

    from spacr.validate import _resolve_measure_src

    plate = tmp_path / 'experiment' / 'Projection'
    folder = plate / 'orig'
    first = _late_ifd_image(folder / 'W0025F0001T0001Z000C1.tif', big, endian)
    second = _late_ifd_image(folder / 'W0025F0001T0001Z000C2.tif', big, endian)
    with tifffile.TiffFile(first) as image:
        assert image.pages[0].shape == (64, 64)
        assert image.byteorder == endian and image.is_bigtiff is big
    for index in range(140):
        (folder / f'unrelated-{index}.dat').write_bytes(b'unknown')
    merged = plate / 'merged'
    merged.mkdir()
    values = np.zeros((6, 8, 3), dtype=np.uint16)
    values[2:4, 2:4, 2] = 1
    np.save(merged / 'plate1_A01_1.npy', values)
    database = plate / 'measurements' / 'measurements.db'
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE objects (label INTEGER, area REAL)')
        connection.execute('INSERT INTO objects VALUES (1, 4)')
    settings = plate / 'settings' / 'gen_mask_settings.json'
    settings.parent.mkdir()
    settings.write_text(json.dumps({'src': str(folder), 'cell_channel': 1}))
    files = [path for path in plate.rglob('*') if path.is_file()]
    before = _digest(files)
    reads = {}
    read = home._easy_read_header

    def bounded(path, limit, offset=0):
        reads[Path(path)] = reads.get(Path(path), 0) + limit
        return read(path, limit, offset)

    def no_pixels(*args, **kwargs):
        raise AssertionError('local suggestions must not decode source pixels')

    monkeypatch.setattr(home, '_easy_read_header', bounded)
    monkeypatch.setattr(TiffImagePlugin.TiffImageFile, 'load', no_pixels)
    monkeypatch.setattr(tifffile.TiffPage, 'asarray', no_pixels)
    result = home._easy_inspect_path(str(first))
    assert [action[0] for action in result['actions']] == ['make_masks', 'mask']
    assert [action[3] for action in result['actions']] == [str(first), str(folder)]
    assert any(hint['convention'] == 'cq1' and hint['matched'] == 2
               for hint in result['context']['filename_matches'])
    assert result['confidence'] == 'filename_hint'
    assert result['metadata_preset'] == 'cq1'
    assert result['settings_overrides'] == {'mask': {'metadata_type': 'cq1'}}
    assert result['context']['truncated']
    assert result['context']['sampled_files'] <= home._EASY_FILE_LIMIT
    assert result['context']['sampled_directories'] <= home._EASY_DIRECTORY_LIMIT
    assert result['context']['filename_sample_limit'] == 100
    assert result['context']['header_files'] <= 16
    assert all(total <= home._EASY_HEADER_LIMIT for path, total in reads.items()
               if path.suffix != '.json')
    assert _digest(files) == before
    result = home._easy_inspect_path(str(plate))
    measure = next(action for action in result['actions'] if action[0] == 'measure')
    assert measure[3] == str(plate)
    assert _resolve_measure_src(measure[3])[0] == str(merged)
    marker = plate / 'plate_batch.json'
    marker.write_text('{')
    result = home._easy_inspect_path(str(plate))
    assert not any(action[0] == 'measure' for action in result['actions'])
    marker.unlink()
    (merged / 'plate1_A01_1.npy').unlink()
    result = home._easy_inspect_path(str(plate))
    assert not any(action[0] == 'measure' for action in result['actions'])
    remaining = [path for path in files if path.exists()]
    assert _digest(remaining) == {str(path): before[str(path)] for path in remaining}


@pytest.mark.parametrize('defect', ['missing-organization', 'unsafe-dimensions'])
def test_tiff_metadata_refuses_missing_organization_and_unsafe_dimensions(tmp_path, defect):
    import struct

    path = _late_ifd_image(tmp_path / 'broken.tif', False)
    raw = bytearray(path.read_bytes())
    offset = struct.unpack_from('<I', raw, 4)[0]
    for index in range(struct.unpack_from('<H', raw, offset)[0]):
        position = offset + 2 + index * 12
        tag = struct.unpack_from('<H', raw, position)[0]
        if defect == 'missing-organization' and tag == 273:
            struct.pack_into('<H', raw, position, 65000)
        elif defect == 'unsafe-dimensions' and tag in (256, 257):
            struct.pack_into('<I', raw, position + 8, 1000000)
    path.write_bytes(raw)
    before = _digest([path])
    result = home._easy_inspect_path(str(path))
    assert not result['actions']
    assert _digest([path]) == before


def test_import_capabilities_offer_registered_containers_without_opening_readers(tmp_path, monkeypatch):
    import importlib.util

    from spacr import convert

    def no_reader(*args, **kwargs):
        raise AssertionError('general inference must not run unbounded vendor metadata readers')

    monkeypatch.setattr(convert, '_describe', no_reader)
    monkeypatch.setattr(convert, '_import_reader', no_reader)
    find_spec = importlib.util.find_spec
    modules = {value[0] for value in convert._READERS.values()}
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None if name in modules else find_spec(name))
    for extension, (reader, install) in convert._READERS.items():
        path = tmp_path / f'acquisition{extension}'
        path.write_bytes(b'Uninspected local container\x00')
        before = _digest([path])
        result = home._easy_inspect_path(str(path))
        assert result['actions'][0][0] == 'convert'
        assert result['actions'][0][3] == str(tmp_path)
        assert reader in result['actions'][0][2] and install in result['actions'][0][2]
        evidence = next(row for row in result['context']['reader_candidates'] if row['path'] == str(path))
        assert evidence['reader'] == reader and evidence['install_hint'] == install
        assert evidence['metadata_inspected'] is False
        assert evidence['available'] is False
        assert not any(action[0] in ('mask', 'make_masks') for action in result['actions'])
        assert _digest([path]) == before


def test_explicit_regex_validation_preserves_actions_and_seeds_only_real_mask_fields(tmp_path):
    import time

    from spacr.import_plan import for_get_regex
    from spacr.regex_infer import _metadata_parse_report

    images = [_image(tmp_path / name) for name in ('scan_101_C1.tif', 'scan_101_C2.tif')]
    pattern = r'scan_(?P<fieldID>\d+)_C(?P<chanID>\d+)\.tif$'
    actions = [('make_masks', 'view', 'inspect', str(images[0])),
               ('mask', 'segment', 'review', str(tmp_path)),
               ('measure', 'measure', 'review', str(tmp_path))]
    proposal = home._easy_regex_evidence(images, tmp_path, [])
    original = {'path': str(images[0]), 'actions': actions, 'regex': proposal}
    assert proposal['required']
    invalid = home._easy_validate_regex(original, r'(?P<fieldID>missing)')
    assert invalid['regex']['required'] and invalid['regex']['validation_error']
    assert invalid['actions'] == actions and 'settings_overrides' not in invalid
    started = time.monotonic()
    valid = home._easy_validate_regex(invalid, pattern)
    valid_seconds = time.monotonic() - started
    assert 0 < valid_seconds < 3.5
    assert valid['actions'] == actions and not valid['regex']['required']
    assert 'validation_error' not in valid['regex']
    assert valid['settings_overrides'] == {'mask': {
        'custom_regex': for_get_regex(pattern), 'metadata_type': 'custom'}}
    assert _metadata_parse_report([p.name for p in images], 'custom', 'tif',
                                  valid['settings_overrides']['mask']['custom_regex'])[:2] == (2, 2)
    for group in valid['regex']['groups']:
        assert valid['regex']['example_path'][group['start']:group['end']] in ('101', '1')
    rejected = home._easy_validate_regex(valid, '(')
    assert rejected['regex']['required'] and rejected['regex']['validation_error']
    assert 'settings_overrides' not in rejected
    assert original['regex']['required']


def test_image_does_not_borrow_database_actions_and_explicit_database_stays_browsable(tmp_path):
    import sqlite3

    plate = tmp_path / 'Projection'
    image = _image(plate / 'orig' / 'W0025F0001T0001Z000C1.tif')
    artifacts = plate / 'artifacts.db'
    measurements = plate / 'measurements' / 'measurements.db'
    selected = image.parent / 'user.db'
    measurements.parent.mkdir()
    for database in (artifacts, measurements, selected):
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE objects (label INTEGER, area REAL)')
    before = _digest([image, artifacts, measurements, selected])
    result = home._easy_inspect_path(str(image))
    assert not any(action[0] == 'db_browser' for action in result['actions'])
    assert result['context']['database_schemas'] == []
    assert not any(action[3] in (str(measurements), str(artifacts)) for action in result['actions'])
    assert not any(action[0] == 'measure' for action in result['actions'])
    explicit = home._easy_inspect_path(str(selected))
    assert explicit['actions'][0][0] == 'db_browser' and explicit['actions'][0][3] == str(selected)
    assert _digest([image, artifacts, measurements, selected]) == before


@pytest.mark.parametrize('cancel', [False, True], ids=['deadline', 'cancel'])
def test_explicit_pathological_regex_keeps_gui_responsive_and_retires_its_child(
        qtbot, tmp_path, cancel):
    import multiprocessing
    import threading
    import time
    from functools import partial

    from PySide6.QtCore import QTimer
    from spacr.qt.job_runner import JobRunner

    image = _image(tmp_path / ('a' * 120 + 'X.tif'))
    before = _digest([image])
    result = {'actions': [('mask', 'mask', 'review', str(tmp_path))], 'regex': {
        'pattern': '', 'sample_filenames': [image.name], 'example_path': str(image),
        'source_folder': str(tmp_path), 'img_format': 'tif', 'required': True}}
    children = {child.pid for child in multiprocessing.active_children()}
    cancellation = threading.Event()
    received, ticks = [], []
    runner = JobRunner()
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    started = time.monotonic()
    try:
        runner.submit(partial(home._easy_validate_regex, result,
                              r'(?P<fieldID>(a+)+)\.tif$', cancellation.is_set), on_done=received.append)
        qtbot.waitUntil(lambda: any(child.pid not in children for child in multiprocessing.active_children()), timeout=2000)
        if cancel:
            cancellation.set()
        qtbot.waitUntil(lambda: bool(received) and not runner.is_busy(), timeout=6000)
        elapsed = time.monotonic() - started
        assert received[0]['regex']['required'] and 'settings_overrides' not in received[0]
        assert received[0]['actions'] == result['actions']
        assert not [child for child in multiprocessing.active_children() if child.pid not in children]
        if cancel:
            assert elapsed < 2.5
        else:
            assert 2.5 <= elapsed < 5.5
            assert len(ticks) >= 20
            assert received[0]['regex']['validation_error']
        assert _digest([image]) == before
    finally:
        cancellation.set()
        timer.stop()
        runner.shutdown(3000)


def test_regex_process_start_failure_returns_guidance_without_joining_unstarted_child(tmp_path, monkeypatch):
    from multiprocessing.process import BaseProcess

    image = _image(tmp_path / 'scan_1_C1.tif')
    original = {'actions': [('mask', 'mask', 'review', str(tmp_path))], 'regex': {
        'pattern': '', 'sample_filenames': [image.name], 'example_path': str(image),
        'source_folder': str(tmp_path), 'img_format': 'tif', 'required': True}}

    def failed_start(process):
        raise OSError('The operating system refused process creation')

    monkeypatch.setattr(BaseProcess, 'start', failed_start)
    result = home._easy_validate_regex(original, r'scan_(?P<fieldID>\d+)_C(?P<chanID>\d+)\.tif$')
    assert result['regex']['required'] and result['regex']['validation_error']
    assert result['actions'] == original['actions'] and 'settings_overrides' not in result
