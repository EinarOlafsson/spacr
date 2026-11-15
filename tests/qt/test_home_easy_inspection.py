"""What the easy Home screen's input inspector offers for a path it cannot fully trust.

``_easy_inspect_path`` classifies a selected file or folder from bounded header
bytes and offers routes; ``_easy_specialized_inputs`` recognises spatial and
OPS layouts; ``_easy_run_evidence`` finds run manifests and coefficient tables;
``_easy_validate_regex`` checks a filename pattern in an owned child process.
Every case reads real files and asserts the exact routes, hints or refusal, and
that no input byte changed: a corrupt header, a pipe, a permission failure or
a file that vanishes mid-scan never becomes an analysis route, and a validation
child is always retired.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture; every case writes its own folder.
"""
from __future__ import annotations

import copy
import json
import multiprocessing
import os
import signal
import threading
from multiprocessing import spawn
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from spacr.qt.i18n import tr  # noqa: E402
from spacr.qt.widgets import home as home_mod  # noqa: E402

pytestmark = pytest.mark.qt

REGEX = r'scan_(?P<fieldID>\d+)_C(?P<chanID>\d+)\.tif$'


def _folder(tmp_path, name):
    folder = tmp_path / name
    folder.mkdir()
    return folder


def _tiff(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.arange(30, dtype=np.uint16).reshape(5, 6)).save(path)
    return path


def _snapshot(*paths):
    return {path: path.read_bytes() for path in paths}


def _unchanged(before):
    return all(path.read_bytes() == raw for path, raw in before.items())


def _deny_named_scan(mp, denied):
    """Make ``os.scandir`` refuse one directory, as a permission failure would."""
    original = home_mod.os

    class NamedScanBoundary:
        def __getattr__(self, name):
            return getattr(original, name)

        def scandir(self, path):
            if Path(path) == denied:
                raise PermissionError('the named acquisition directory is unavailable')
            return original.scandir(path)

    mp.setattr(home_mod, 'os', NamedScanBoundary())


def _scale(folder, value):
    scale = folder / 'spatial' / 'scalefactors_json.json'
    scale.parent.mkdir(parents=True, exist_ok=True)
    scale.write_text(json.dumps({'microns_per_pixel': value}))
    return scale


def _counts(path):
    with h5py.File(path, 'w') as archive:
        archive.create_dataset('data', data=[1])
    return path


def test_easy_inspect_path_offers_routes_only_for_validated_inputs(tmp_path, monkeypatch, subtests):
    """_easy_inspect_path refuses corrupt, special or unreadable inputs and keeps valid image routes."""
    documents = [(b'<Workspace/>', 'flowjo'), (b'<GraphPadPrismFile/>', 'prism'),
                 (b'<CDXML/>', 'chemdraw')]
    for content, family in documents:
        with subtests.test(msg=f'external_document_{family}'):
            path = _folder(tmp_path, f'doc_{family}') / 'source.dat'
            path.write_bytes(content)
            assert home_mod._easy_file_kind(path) == 'external_' + family
            result = home_mod._easy_inspect_path(str(path))
            assert result['actions'] == []
            assert result['context']['unsupported_documents'] == [dict(kind=family, path=str(path))]
            assert path.read_bytes() == content

    unsupported = [
        ('invalid_bigtiff_offset_width', 'source.tif', b'II+\x00\x04\x00\x00\x00' + b'\x00' * 8),
        ('short_count', 'source.tif', b'II*\x00\x08\x00\x00\x00\x01'),
        ('empty_ifd', 'source.tif', b'II*\x00\x08\x00\x00\x00\x00\x00'),
        ('unbounded_ifd', 'source.tif', b'II*\x00\x08\x00\x00\x00\xff\xff'),
        ('short_entry', 'source.tif', b'II*\x00\x08\x00\x00\x00\x01\x00' + b'\x00' * 5),
        ('corrupt_gzip_sequence', 'source.fastq.gz', b'\x1f\x8b' + b'not compressed sequence bytes'),
    ]
    for case_id, name, raw in unsupported:
        with subtests.test(msg=case_id):
            path = _folder(tmp_path, case_id) / name
            path.write_bytes(raw)
            assert home_mod._easy_file_kind(path) == 'unsupported'
            assert home_mod._easy_inspect_path(str(path))['actions'] == []
            assert path.read_bytes() == raw

    with subtests.test(msg='malformed_named_settings'):
        path = _folder(tmp_path, 'settings') / 'mask_settings.json'
        path.write_text('{"src": broken')
        before = path.read_bytes()
        result = home_mod._easy_inspect_path(str(path))
        assert result['path'] == str(path) and result['actions'] == []
        assert 'saved_settings' not in result.get('context', {})
        assert path.read_bytes() == before

    if hasattr(os, 'mkfifo'):
        with subtests.test(msg='fifo_refused_before_any_header_read'):
            path = _folder(tmp_path, 'fifo') / 'input.pipe'
            os.mkfifo(path)

            def unexpected(*args, **kwargs):
                raise AssertionError('a FIFO must never reach the blocking header reader')

            with monkeypatch.context() as mp:
                mp.setattr(home_mod, '_easy_read_header', unexpected)
                result = home_mod._easy_inspect_path(str(path))
            assert result == dict(path=str(path), actions=[], message=tr(
                'Choose a regular file or folder, rather than a device or pipe.'))

    with subtests.test(msg='corrupt_sqlite_is_not_evidence'):
        folder = _folder(tmp_path, 'sqlite')
        database = folder / 'measurements.db'
        database.write_bytes(b'SQLite format 3\x00' + bytes(112))
        before = _snapshot(database)
        result = home_mod._easy_inspect_path(str(folder))
        assert result['context']['database_schemas'] == []
        assert result['context']['analysis_inputs'] == {}
        assert not any(action[0] in ('db_browser', 'graph_builder', 'regression', 'umap')
                       for action in result['actions'])
        assert _unchanged(before)

    with subtests.test(msg='mixed_images_keep_external_document_guidance'):
        folder = _folder(tmp_path, 'mixed')
        image = _tiff(folder / 'field.tif')
        workspace = folder / 'analysis.wsp'
        workspace.write_bytes(b'<Workspace/>')
        before = _snapshot(image, workspace)
        result = home_mod._easy_inspect_path(str(folder))
        assert [action[0] for action in result['actions']] == ['make_masks', 'mask']
        assert home_mod._easy_external_guidance('flowjo') in result['message']
        assert result['context']['unsupported_documents'] == [dict(kind='flowjo', path=str(workspace))]
        assert _unchanged(before)

    with subtests.test(msg='stat_failure_becomes_local_guidance'):
        path = _folder(tmp_path, 'stat') / 'image.tif'
        path.write_bytes(b'untouched input')
        exists = Path.exists

        def denied(candidate):
            if candidate == path:
                raise PermissionError('the selected path is unavailable')
            return exists(candidate)

        with monkeypatch.context() as mp:
            mp.setattr(Path, 'exists', denied)
            result = home_mod._easy_inspect_path(str(path))
        assert result == dict(path=str(path), actions=[], message=tr(
            'This input could not be inspected. Check permissions and choose another path.'))
        assert path.read_bytes() == b'untouched input'

    with subtests.test(msg='unreadable_representative_keeps_image_routes'):
        folder = _folder(tmp_path, 'representative')
        image = _tiff(folder / 'field.tif')
        refused_path = folder / 'unreadable.dat'
        refused_path.write_bytes(b'private inaccessible acquisition metadata')
        before = _snapshot(image, refused_path)
        original_read = home_mod._easy_read_header
        refused = []

        def read(path, limit, offset=0):
            if path == refused_path:
                refused.append(path)
                raise PermissionError('the selected representative cannot be read')
            return original_read(path, limit, offset)

        with monkeypatch.context() as mp:
            mp.setattr(home_mod, '_easy_read_header', read)
            result = home_mod._easy_inspect_path(str(folder))
        actions = {action[0]: action[3] for action in result['actions']}
        assert refused
        assert actions['make_masks'] == str(folder) and actions['mask'] == str(folder)
        assert 'Some files could not be read' in result['message']
        assert result['context']['database_schemas'] == []
        assert result['context']['image_dimensions'][0]['path'] == str(image)
        assert _unchanged(before)

    with subtests.test(msg='table_removed_after_classification_gains_no_routes'):
        folder = _folder(tmp_path, 'removed_table')
        image = _tiff(folder / 'field.tif')
        table = folder / 'observations.csv'
        table.write_text('field,value\n1,2.5\n2,3.5\n')
        original = table.read_bytes()
        image_bytes = image.read_bytes()
        archived = folder / 'removed-observations.csv'
        original_read = home_mod._easy_read_header
        removed = []

        def read(path, limit, offset=0):
            if path == table and limit == home_mod._EASY_HEADER_LIMIT and not removed:
                table.rename(archived)
                removed.append(path)
            return original_read(path, limit, offset)

        with monkeypatch.context() as mp:
            mp.setattr(home_mod, '_easy_read_header', read)
            result = home_mod._easy_inspect_path(str(folder))
        keys = {action[0] for action in result['actions']}
        assert removed == [table] and not table.exists()
        assert {'make_masks', 'mask'} <= keys
        assert keys.isdisjoint({'graph_builder', 'gate_editor', 'regression'})
        assert result['context']['analysis_inputs'] == {}
        assert archived.read_bytes() == original and image.read_bytes() == image_bytes


def test_easy_specialized_inputs_needs_readable_calibration_and_headers(tmp_path, monkeypatch, subtests):
    """_easy_specialized_inputs offers spatial/OPS hints only from valid calibration and readable files."""
    for case_id, value in [('calibration_without_a_number', {'unit': 'unknown'}),
                           ('calibration_as_text', 'not a physical scale')]:
        with subtests.test(msg=case_id):
            root = _folder(tmp_path, case_id)
            slide = _tiff(root / 'slide.tif')
            before = _snapshot(slide, _scale(root, value))
            assert home_mod._easy_specialized_inputs(root, [slide], [], []) == {}
            assert _unchanged(before)

    with subtests.test(msg='empty_hd_bins_offer_calibrated_morphology_review'):
        root = _folder(tmp_path, 'empty_bins')
        slide = _tiff(root / 'slide.tif')
        (root / 'binned_outputs').mkdir()
        before = _snapshot(slide, _scale(root, .5))
        hints = home_mod._easy_specialized_inputs(root, [slide], [], [])
        assert set(hints) == {'spatial_masks'}
        assert hints['spatial_masks']['settings'] == dict(spatial_source=str(root),
                                                          spatial_platform='visium_hd')
        assert hints['spatial_masks']['image'] == str(slide)
        assert _unchanged(before)

    with subtests.test(msg='unreadable_hd_bins_guess_no_bin'):
        root = _folder(tmp_path, 'unreadable_bins')
        slide = _tiff(root / 'slide.tif')
        bins = root / 'binned_outputs'
        (bins / 'square_002um').mkdir(parents=True)
        before = _snapshot(slide, _scale(root, .5))
        with monkeypatch.context() as mp:
            _deny_named_scan(mp, bins)
            hints = home_mod._easy_specialized_inputs(root, [slide], [], [])
        assert set(hints) == {'spatial_masks'}
        assert hints['spatial_masks']['settings'] == dict(spatial_source=str(root),
                                                          spatial_platform='visium_hd')
        assert hints['spatial_masks']['image'] == str(slide)
        assert _unchanged(before)

    with subtests.test(msg='unreadable_xenium_focus_uses_the_mip'):
        root = _folder(tmp_path, 'xenium')
        marker = root / 'experiment.xenium'
        marker.write_text('{}')
        slide = _tiff(root / 'morphology_mip.ome.tif')
        focus = root / 'morphology_focus'
        focus.mkdir()
        before = _snapshot(marker, slide)
        with monkeypatch.context() as mp:
            _deny_named_scan(mp, focus)
            hints = home_mod._easy_specialized_inputs(root, [slide], [], [])
        assert set(hints) == {'spatial_masks'}
        assert hints['spatial_masks']['image'] == str(slide)
        assert hints['spatial_masks']['settings']['spatial_platform'] == 'xenium'
        assert _unchanged(before)

    with subtests.test(msg='unreadable_count_directory_promotes_no_mapping'):
        root = _folder(tmp_path, 'counts')
        slide = _tiff(root / 'slide.tif')
        scale = _scale(root, .5)
        positions = root / 'spatial' / 'tissue_positions.csv'
        positions.write_text('barcode,in_tissue\na,1\n')
        counts = _counts(root / 'filtered_feature_bc_matrix.h5')
        before = _snapshot(slide, scale, positions, counts)
        with monkeypatch.context() as mp:
            _deny_named_scan(mp, root)
            hints = home_mod._easy_specialized_inputs(root, [slide], [], [])
        assert set(hints) == {'spatial_masks'}
        assert hints['spatial_masks']['image'] == str(slide)
        assert _unchanged(before)

    with subtests.test(msg='root_calibration_does_not_validate_a_malformed_bin_calibration'):
        root = _folder(tmp_path, 'hd_root')
        slide = _tiff(root / 'slide.tif')
        root_scale = _scale(root, .5)
        selected = root / 'binned_outputs' / 'square_002um'
        bin_scale = _scale(selected, ['not a calibration'])
        positions = selected / 'spatial' / 'tissue_positions.csv'
        positions.write_text('barcode,in_tissue\na,1\n')
        counts = _counts(selected / 'filtered_feature_bc_matrix.h5')
        before = _snapshot(slide, root_scale, bin_scale, positions, counts)
        hints = home_mod._easy_specialized_inputs(root, [slide], [], [])
        assert set(hints) == {'spatial_masks'}
        assert hints['spatial_masks']['settings'] == dict(
            spatial_source=str(root), spatial_platform='visium_hd', spatial_bin_um=2)
        assert hints['spatial_masks']['image'] == str(slide)
        assert _unchanged(before)

    with subtests.test(msg='ops_library_needs_local_readable_headers'):
        base = _folder(tmp_path, 'ops')
        root = base / 'selected'
        tile = _tiff(root / 'genotype' / '10X_c2_A1_A594_Site-0.tif')
        malformed = root / 'library.csv'
        malformed.write_bytes(b'\xffinvalid table')
        outside = base / 'outside.csv'
        outside.write_text('barcode,sequence\na,ACGT\n')
        before = _snapshot(tile, malformed, outside)
        hints = home_mod._easy_specialized_inputs(root, [outside, malformed], [tile], [])
        assert set(hints) == {'ops'}
        assert hints['ops']['settings'] == dict(genotype_source=str(tile.parent))
        assert hints['ops']['source'] == str(tile.parent)
        assert _unchanged(before)


def test_easy_run_evidence_reads_manifests_and_tables_within_quotas(tmp_path, subtests):
    """_easy_run_evidence caps runs and tables, keeps failure provenance, skips unreadable evidence."""
    with subtests.test(msg='run_and_coefficient_quotas'):
        root = _folder(tmp_path, 'quotas')
        files, before = [], {}
        for index in range(10):
            folder = root / 'runs' / f'run-{index:02d}'
            folder.mkdir(parents=True)
            manifest = folder / 'manifest.json'
            manifest.write_text(json.dumps(dict(app_key='mask', schema_version=1, status='partial')))
            coefficients = folder / 'results.csv'
            coefficients.write_text('feature,coefficient\narea,1.5\n')
            (folder / 'diagnostics').mkdir()
            report = folder / 'diagnostics' / 'fit.txt'
            report.write_text('owned diagnostic')
            files.extend((manifest, coefficients))
            before.update(_snapshot(manifest, coefficients, report))
        result = home_mod._easy_run_evidence(root, files)
        assert len(result['manifests']) == 8
        assert all(item['missing_outputs'] == ['merged'] for item in result['manifests'])
        assert len(result['results']) == 4
        assert all(item['diagnostics'] for item in result['results'])
        assert _unchanged(before)

    with subtests.test(msg='manifest_finds_its_table_outside_the_sample'):
        root = _folder(tmp_path, 'outside_sample')
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(dict(app_key='regression', schema_version=1,
                                            status='success')))
        table = root / 'results_gene.csv'
        table.write_text('feature,coefficient\narea,1.5\n')
        before = _snapshot(manifest, table)
        result = home_mod._easy_run_evidence(root, [])
        assert result == dict(manifests=[], results=[dict(
            path=str(table), folder=str(root), diagnostics=False, diagnostics_folder=None)])
        assert _unchanged(before)

    if os.name == 'posix' and os.geteuid() != 0:
        with subtests.test(msg='unreadable_and_malformed_evidence'):
            root = _folder(tmp_path, 'permissions')
            runs = root / 'runs'
            (runs / 'broken').mkdir(parents=True)
            (runs / 'broken' / 'manifest.json').write_text('{invalid')
            (runs / 'denied').mkdir()
            inaccessible = runs / 'denied' / 'manifest.json'
            inaccessible.write_text(json.dumps(dict(app_key='mask', schema_version=1, status='failed')))
            valid = root / 'results.csv'
            valid.write_text('feature,coefficient\narea,2\n')
            diagnostics = root / 'diagnostics'
            diagnostics.mkdir()
            (diagnostics / 'fit.txt').write_text('diagnostic')
            empty = root / 'empty' / 'results.csv'
            empty.parent.mkdir()
            empty.write_bytes(b'\xff')
            before = valid.read_bytes()
            inaccessible.chmod(0)
            diagnostics.chmod(0)
            try:
                result = home_mod._easy_run_evidence(root, [empty, valid])
            finally:
                inaccessible.chmod(0o600)
                diagnostics.chmod(0o700)
            assert result['manifests'] == []
            assert result['results'] == [dict(path=str(valid), folder=str(root),
                                              diagnostics=False, diagnostics_folder=None)]
            assert valid.read_bytes() == before


def _regex_protocol_worker_ignoring_termination(payload, pattern, connection, ready):
    import signal as _signal
    import threading as _threading

    _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)
    ready.set()
    connection.send_bytes(b'null')
    try:
        _threading.Event().wait(30)
    finally:
        connection.close()


def _regex_request():
    return {'regex': dict(sample_filenames=['scan_1_C1.tif'], img_format='tif',
                          example_path='/local/scan_1_C1.tif')}


def _live_children():
    return {child.pid for child in multiprocessing.active_children()}


def test_easy_validate_regex_bounds_input_and_retires_its_child(tmp_path, monkeypatch, subtests):
    """_easy_validate_regex refuses bounded inputs without a child and always retires a spawned one."""
    for reason in ('cancelled', 'pattern', 'samples', 'payload'):
        with subtests.test(msg=f'no_child_for_{reason}'):
            result = {'path': '/local/images', 'settings_overrides': {'mask': {'size': 12}},
                      'regex': {'sample_filenames': ['image.tif'],
                                'example_path': '/local/image.tif'}}
            pattern = '(.*)'
            if reason == 'pattern':
                pattern = 'x' * 4097
            elif reason == 'samples':
                result['regex']['sample_filenames'] = []
            elif reason == 'payload':
                result['regex']['example_path'] = 'x' * 1048577
            before = copy.deepcopy(result)

            def unexpected(*args, **kwargs):
                raise AssertionError('bounded refusal must not start a child')

            with monkeypatch.context() as mp:
                mp.setattr(multiprocessing, 'get_context', unexpected)
                actual = home_mod._easy_validate_regex(
                    result, pattern, cancelled=lambda: reason == 'cancelled')
            assert result == before
            assert actual['regex']['pattern'] == pattern and actual['regex']['required']
            assert 'settings_overrides' not in actual
            assert bool(actual['regex'].get('validation_error')) is (reason != 'cancelled')

    with subtests.test(msg='missing_interpreter_is_a_local_refusal'):
        result = _regex_request()
        before = copy.deepcopy(result)
        executable = spawn.get_executable()
        children = _live_children()
        try:
            multiprocessing.set_executable(str(tmp_path / 'missing-python'))
            actual = home_mod._easy_validate_regex(result, REGEX)
        finally:
            multiprocessing.set_executable(executable)
        assert actual['regex']['required'] and actual['regex']['validation_error']
        assert result == before
        assert _live_children() == children

    with subtests.test(msg='cancellation_after_spawn_retires_the_child'):
        context = multiprocessing.get_context('spawn')
        factory = context.Process
        start = factory.start
        cancelled, created = threading.Event(), []

        def started(child):
            start(child)
            created.append(child)
            cancelled.set()

        result = _regex_request()
        before = copy.deepcopy(result)
        children = _live_children()
        with monkeypatch.context() as mp:
            mp.setattr(factory, 'start', started)
            actual = home_mod._easy_validate_regex(result, REGEX, cancelled=cancelled.is_set)
        assert cancelled.is_set() and len(created) == 1 and created[0]._closed
        assert actual['regex']['required'] and 'validation_error' not in actual['regex']
        assert 'settings_overrides' not in actual and result == before
        assert _live_children() == children

    if os.name == 'posix':
        with subtests.test(msg='invalid_reply_escalates_retirement_to_kill'):
            context = multiprocessing.get_context('spawn')
            process_class = context.Process
            ready = context.Event()
            children, pipes, operations, exits = [], [], [], []
            original_start = process_class.start
            original_terminate = process_class.terminate
            original_kill = process_class.kill
            original_close = process_class.close

            class ProtocolWorkerContext:
                def Pipe(self, **kwargs):
                    endpoints = context.Pipe(**kwargs)
                    pipes.extend(endpoints)
                    return endpoints

                def Process(self, *, target, args):
                    assert target is home_mod._easy_regex_worker
                    child = process_class(target=_regex_protocol_worker_ignoring_termination,
                                          args=(*args, ready))
                    children.append(child)
                    return child

            def start_ready(child):
                original_start(child)
                assert ready.wait(10)

            def terminate(child):
                operations.append('terminate')
                original_terminate(child)

            def kill(child):
                operations.append('kill')
                original_kill(child)

            def close(child):
                operations.append('close')
                exits.append(child.exitcode)
                original_close(child)

            result = _regex_request()
            before = copy.deepcopy(result)
            prior_children = _live_children()
            try:
                with monkeypatch.context() as mp:
                    mp.setattr(multiprocessing, 'get_context', lambda method: ProtocolWorkerContext())
                    mp.setattr(process_class, 'start', start_ready)
                    mp.setattr(process_class, 'terminate', terminate)
                    mp.setattr(process_class, 'kill', kill)
                    mp.setattr(process_class, 'close', close)
                    actual = home_mod._easy_validate_regex(result, REGEX)
                assert operations == ['terminate', 'kill', 'close']
                assert exits == [-signal.SIGKILL]
                assert len(children) == 1 and children[0]._closed
                assert all(connection.closed for connection in pipes)
                assert actual['regex']['required'] and actual['regex']['validation_error']
                assert result == before and 'settings_overrides' not in actual
                assert _live_children() == prior_children
            finally:
                for child in children:
                    if not child._closed:
                        if child.is_alive():
                            original_kill(child)
                        child.join()
                        original_close(child)
                for connection in pipes:
                    connection.close()
