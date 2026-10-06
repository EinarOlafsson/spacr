"""A finite mapped acquisition retains exact batch percentile normalization."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import convert, core
from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
from tests.test_watch_folder_and_analyse import MASK, MEASURE, _channels, _rows, real_pipeline
from tests.test_watch_pipeline_resume_f548 import _bytes


def _acquisition(tmp_path, *, uneven=False):
    raw, converted = tmp_path / 'raw', tmp_path / 'converted'
    for index, well in enumerate(('A01', 'A02', 'A03')):
        directory = raw / well
        directory.mkdir(parents=True)
        for channel, image in enumerate(_channels(index * 12), start=1):
            if uneven:
                image = np.pad(image, ((0, 4 * index), (0, 3 * index)))
            tifffile.imwrite(directory / f'field01_C{channel}.tif', image)
    result = convert.convert_folder({'src': str(raw), 'dst': str(converted),
                                     'preview_rows': 0})
    assert result.is_complete
    watched = tmp_path / 'watched'
    watched.mkdir()
    shutil.copy2(converted / convert.MAP_FILENAME, watched / convert.MAP_FILENAME)
    return converted, watched, result.rows()


def _arrive(source, watched, rows):
    for row in rows:
        shutil.copy2(source / row['target'], watched / row['target'])


def _settings(watched, **extra):
    return dict(MASK, src=str(watched), batch_size=2, keep_intermediate=True,
                watch_folder=True, watch_pipeline='mask',
                watch_normalization_pool='fixed_map',
                watch_settle_seconds=0.01, watch_poll_seconds=0.01,
                watch_idle_minutes=0.001, **extra)


class CohortRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, directory, settings):
        names = sorted(path.name for path in Path(directory).glob('*.tif'))
        self.calls.append(names)
        from spacr.io import _escaped_field_stem
        from spacr.utils import _extract_filename_metadata

        parsed = _extract_filename_metadata(
            names, directory, core._watch_pattern(settings, 'tif', {}), 'cellvoyager')
        keys = {_escaped_field_stem(key[0], key[1], key[2], key[4]) for key in parsed}
        merged = Path(directory) / 'merged'
        merged.mkdir()
        for key in keys:
            np.save(merged / (key + '.npy'), np.zeros((4, 4, 2), dtype=np.uint16))


def test_incomplete_pool_waits_for_unseen_members_then_runs_once(tmp_path, capsys):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, [row for row in rows if row['well'] == 'A01'])
    recorder = CohortRecorder()
    first = core._watch_folder_and_analyse(_settings(watched), recorder)
    assert first['done'] == [] and len(first['incomplete']) == 3
    assert recorder.calls == []
    assert not (watched / 'spacr_watch/merged').exists()
    assert not (watched / 'spacr_watch/measurements').exists()
    assert 'all declared fields are required before analysis' in capsys.readouterr().out
    _arrive(source, watched, [row for row in rows if row['well'] != 'A01'])
    second = core._watch_folder_and_analyse(_settings(watched), recorder)
    assert len(second['done']) == 3 and not second['failed'] and not second['incomplete']
    assert len(recorder.calls) == 1 and len(recorder.calls[0]) == 6
    ledger = json.loads(Path(second['ledger']).read_text())
    spec = ledger['normalization_pool']
    assert spec['field_order'] == second['done']
    assert spec['batch_size'] == 2
    unit = ledger['normalization_cohorts'][spec['id']]
    assert unit['status'] == 'done'
    assert len(unit['snapshot_sha256']) == 6
    assert all(entry['normalization_cohort'] == spec['id']
               for entry in ledger['fields'].values())
    assert core._watch_folder_and_analyse(_settings(watched), recorder)['done'] == second['done']
    assert len(recorder.calls) == 1


@pytest.mark.parametrize('uneven', (False, True))
def test_original_pixel_batch_normalization_masks_and_measurements_match(
        tmp_path, real_pipeline, uneven):
    from spacr.measure import measure_crop

    source, watched, rows = _acquisition(tmp_path, uneven=uneven)
    _arrive(source, watched, rows)
    batch = tmp_path / 'batch'
    batch.mkdir()
    _arrive(source, batch, rows)
    recipe = tmp_path / 'measure.json'
    recipe.write_text(json.dumps(MEASURE))
    settings = _settings(watched, watch_measure_settings=str(recipe))
    settings['watch_pipeline'] = 'mask_measure'
    core.preprocess_generate_masks(dict(MASK, src=str(batch), batch_size=2,
                                         keep_intermediate=True))
    measure_crop(dict(MEASURE, src=str(batch / 'merged')))
    result = core._watch_folder_and_analyse(settings)
    assert len(result['done']) == 3 and not result['failed']
    combined = watched / 'spacr_watch'
    ledger = json.loads(Path(result['ledger']).read_text())
    staged = combined / 'fields' / ledger['normalization_pool']['id']
    for name in sorted(path.name for path in (batch / 'merged').glob('*.npy')):
        np.testing.assert_array_equal(np.load(batch / 'merged' / name),
                                      np.load(combined / 'merged' / name))
    for table in ('cell', 'nucleus'):
        assert _rows(batch / 'measurements/measurements.db', table) == _rows(
            combined / 'measurements/measurements.db', table)
    for archive in sorted((batch / 'masks').glob('*_norm.npz')):
        with np.load(archive, allow_pickle=True) as reference, np.load(
                staged / 'masks' / archive.name, allow_pickle=True) as actual:
            np.testing.assert_array_equal(reference['data'], actual['data'])
            np.testing.assert_array_equal(reference['filenames'], actual['filenames'])
    assert len(list((batch / 'masks').glob('*_norm.npz'))) == 2
    assert all((watched / row['target']).read_bytes() == (source / row['target']).read_bytes()
               for row in rows)


@pytest.mark.parametrize('failure', (OSError, PipelineCancelled))
def test_interrupted_collection_resumes_one_cohort_without_reanalysis(
        tmp_path, real_pipeline, monkeypatch, failure):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, rows)
    recipe = tmp_path / 'measure.json'
    recipe.write_text(json.dumps(MEASURE))
    settings = _settings(watched, watch_measure_settings=str(recipe))
    settings['watch_pipeline'] = 'mask_measure'
    analyse, append = core._watch_analyse_field, core._watch_merge_database
    calls = []

    def count(directory, snapshot):
        calls.append(directory)
        analyse(directory, snapshot)

    def interrupt(*args, **kwargs):
        raise failure('interrupted collection')

    monkeypatch.setattr(core, '_watch_merge_database', interrupt)
    if failure is PipelineCancelled:
        with pytest.raises(PipelineCancelled):
            core._watch_folder_and_analyse(settings, count)
    else:
        assert len(core._watch_folder_and_analyse(settings, count)['failed']) == 3
    work = watched / 'spacr_watch'
    ledger = json.loads((work / 'watch_ledger.json').read_text())
    unit = ledger['normalization_cohorts'][ledger['normalization_pool']['id']]
    assert 'collection_checkpoint' in unit
    preserved = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in Path(calls[0]).rglob('*.npy')}
    monkeypatch.setattr(core, '_watch_merge_database', append)
    result = core._watch_folder_and_analyse(settings, count)
    assert len(result['done']) == 3 and not result['failed']
    assert len(calls) == 1
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == saved
               for path, saved in preserved.items())
    assert _rows(work / 'measurements/measurements.db', 'cell') == _rows(
        Path(calls[0]) / 'measurements/measurements.db', 'cell')
    assert len(_rows(work / 'measurements/measurements.db', 'cell')[1]) > 0
    assert core._watch_folder_and_analyse(settings, count)['done'] == result['done']
    assert len(calls) == 1


@pytest.mark.parametrize('change', ('source', 'output', 'map', 'recipe', 'mode', 'missing_checkpoint'))
def test_changed_pool_provenance_refuses_resume_without_writing(tmp_path, change):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, rows)
    settings = _settings(watched)
    result = core._watch_folder_and_analyse(settings, CohortRecorder())
    path = Path(result['ledger'])
    ledger = json.loads(path.read_text())
    if change == 'source':
        tifffile.imwrite(watched / rows[0]['target'], np.ones((64, 64), dtype=np.uint16))
    elif change == 'output':
        output = next((watched / 'spacr_watch/merged').glob('*.npy'))
        output.unlink()
        np.save(output, np.ones((4, 4, 2)))
    elif change == 'map':
        with (watched / convert.MAP_FILENAME).open('a') as handle:
            handle.write('\n')
    elif change == 'recipe':
        settings['lower_percentile'] = 8
    elif change == 'mode':
        settings['watch_normalization_pool'] = 'per_field'
    else:
        ledger['normalization_cohorts'][ledger['normalization_pool']['id']].pop('collection_checkpoint')
        path.write_text(json.dumps(ledger))
    before = _bytes(watched)
    recorder = CohortRecorder()
    with pytest.raises(ValueError):
        core._watch_folder_and_analyse(settings, recorder)
    assert _bytes(watched) == before
    assert recorder.calls == []


@pytest.mark.parametrize('extra', (
    {'watch_normalization_pool': 'unknown'}, {'randomize': True},
    {'batch_size': 1}, {'batch_size': 2.5}, {'batch_size': True},
    {'pipeline_style': 'v2'}, {'timelapse': True},
    {'watch_pipeline': 'mask_measure_classify'},
))
def test_unsupported_cohort_recipes_refuse_before_workspace(tmp_path, extra):
    _, watched, _ = _acquisition(tmp_path)
    settings = _settings(watched)
    settings.update(extra)
    with pytest.raises(ValueError):
        core._watch_folder_and_analyse(settings, CohortRecorder())
    assert not (watched / 'spacr_watch').exists()


def test_open_ended_and_unexpected_input_pools_are_refused(tmp_path):
    source, watched, rows = _acquisition(tmp_path)
    mapping = watched / convert.MAP_FILENAME
    mapping.unlink()
    with pytest.raises(ValueError, match='finite.*conversion_map'):
        core._watch_folder_and_analyse(_settings(watched), CohortRecorder())
    shutil.copy2(source / convert.MAP_FILENAME, mapping)
    tifffile.imwrite(watched / 'unexpected.tif', np.ones((16, 16), dtype=np.uint16))
    with pytest.raises(ValueError, match='unexpected or duplicate'):
        core._watch_folder_and_analyse(_settings(watched), CohortRecorder())
    assert not (watched / 'spacr_watch').exists()


def test_cohort_checkpoint_completes_field_aliases_after_process_exit(tmp_path):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, rows)
    settings = _settings(watched)
    recorder = CohortRecorder()
    result = core._watch_folder_and_analyse(settings, recorder)
    path = Path(result['ledger'])
    ledger = json.loads(path.read_text())
    for entry in ledger['fields'].values():
        entry['status'] = 'waiting'
    path.write_text(json.dumps(ledger))
    recovered = core._watch_folder_and_analyse(settings, recorder)
    assert recovered['done'] == result['done']
    assert recovered['incomplete'] == []
    assert len(recorder.calls) == 1


@pytest.mark.parametrize('boundary', ('snapshot', 'analysis'))
def test_stop_before_cohort_publication_restarts_from_original_images(
        tmp_path, monkeypatch, boundary):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, rows)
    token, recorder = CancellationToken(), CohortRecorder()
    copying = core._watch_copy_snapshot
    copied = []

    def stop_after_last_copy(*args):
        digest = copying(*args)
        copied.append(digest)
        if len(copied) == len(rows):
            token.cancel()
        return digest

    def stop_after_analysis(directory, settings):
        recorder(directory, settings)
        token.cancel()

    if boundary == 'snapshot':
        monkeypatch.setattr(core, '_watch_copy_snapshot', stop_after_last_copy)
    with installed_token(token), pytest.raises(PipelineCancelled):
        core._watch_folder_and_analyse(
            _settings(watched), recorder if boundary == 'snapshot' else stop_after_analysis)
    assert len(recorder.calls) == (0 if boundary == 'snapshot' else 1)
    work = watched / 'spacr_watch'
    assert not (work / 'merged').exists()
    ledger = json.loads((work / 'watch_ledger.json').read_text())
    unit = ledger['normalization_cohorts'][ledger['normalization_pool']['id']]
    assert unit['status'] == 'interrupted' and 'collection_checkpoint' not in unit
    assert all(entry['status'] == 'interrupted' for entry in ledger['fields'].values())
    monkeypatch.setattr(core, '_watch_copy_snapshot', copying)
    restart = CohortRecorder()
    assert len(core._watch_folder_and_analyse(_settings(watched), restart)['done']) == 3
    assert len(restart.calls) == 1


def test_changed_snapshot_defers_the_whole_pool_then_retries(tmp_path, monkeypatch):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, rows)
    copying, recorder = core._watch_copy_snapshot, CohortRecorder()
    calls = []

    def changed_once(*args):
        calls.append(args)
        return None if len(calls) == 1 else copying(*args)

    monkeypatch.setattr(core, '_watch_copy_snapshot', changed_once)
    result = core._watch_folder_and_analyse(_settings(watched), recorder)
    assert len(result['done']) == 3 and len(recorder.calls) == 1
    assert len(calls) == len(rows) + 1


def test_partial_analysis_never_publishes_a_smaller_pool(tmp_path):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, rows)
    recorder = CohortRecorder()

    def missing_one(directory, settings):
        recorder(directory, settings)
        next((Path(directory) / 'merged').glob('*.npy')).unlink()

    result = core._watch_folder_and_analyse(_settings(watched), missing_one)
    assert not result['done'] and len(result['failed']) == 3
    assert len(recorder.calls) == 1
    assert not (watched / 'spacr_watch/merged').exists()
    ledger = json.loads(Path(result['ledger']).read_text())
    unit = ledger['normalization_cohorts'][ledger['normalization_pool']['id']]
    assert 'collection_checkpoint' not in unit
    assert 'partial outputs are not published' in unit['error']


def test_input_change_during_analysis_retries_without_publishing_old_cohort(tmp_path, capsys):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, rows)
    recorder = CohortRecorder()

    def changed_source(directory, settings):
        assert not (watched / 'spacr_watch/merged').exists()
        recorder(directory, settings)
        if len(recorder.calls) == 1:
            tifffile.imwrite(watched / rows[0]['target'], np.ones((64, 64), dtype=np.uint16))

    result = core._watch_folder_and_analyse(_settings(watched), changed_source)
    assert len(result['done']) == 3 and not result['failed']
    assert len(recorder.calls) == 2
    assert 'inputs changed during analysis; outputs are not published' in capsys.readouterr().out


@pytest.mark.parametrize('damage', ('hashes', 'staged', 'unknown_cohort',
                                  'input_digest', 'duplicate_hash', 'alias_checkpoint'))
def test_incomplete_cohort_provenance_preserves_results(tmp_path, damage):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, rows)
    result = core._watch_folder_and_analyse(_settings(watched), CohortRecorder())
    path = Path(result['ledger'])
    ledger = json.loads(path.read_text())
    key = ledger['normalization_pool']['id']
    unit = ledger['normalization_cohorts'][key]
    if damage == 'hashes':
        unit['snapshot_sha256'].pop(next(iter(unit['snapshot_sha256'])))
    elif damage == 'input_digest':
        unit['snapshot_sha256'][next(iter(unit['snapshot_sha256']))] = '0' * 64
    elif damage == 'duplicate_hash':
        name = next(iter(unit['snapshot_sha256']))
        unit['snapshot_sha256']['sub/' + name] = unit['snapshot_sha256'][name]
    elif damage == 'alias_checkpoint':
        unit.pop('collection_checkpoint')
        unit['status'] = 'waiting'
    elif damage == 'staged':
        next((watched / 'spacr_watch/fields' / key / 'merged').glob('*.npy')).unlink()
    else:
        ledger['normalization_cohorts']['unknown'] = {'status': 'waiting'}
    path.write_text(json.dumps(ledger))
    before = _bytes(watched)
    with pytest.raises(ValueError):
        core._watch_folder_and_analyse(_settings(watched), CohortRecorder())
    assert _bytes(watched) == before


@pytest.mark.parametrize('kind', ('no_plate', 'unparseable', 'multiple_stacks', 'collision'))
def test_ambiguous_legacy_stack_identity_is_refused(tmp_path, kind):
    settings = _settings(tmp_path)
    names = ['plate1_A01_T0001F001L01A01Z01C01.tif']
    manifest = {'first': names}
    if kind == 'no_plate':
        settings.update(metadata_type='custom', custom_regex=r'(?P<wellID>A01)')
    elif kind == 'unparseable':
        manifest = {'first': ['bad.tif']}
    elif kind == 'multiple_stacks':
        manifest['first'].append('plate1_A02_T0001F001L01A01Z01C01.tif')
    else:
        manifest = {'first': ['plate1_A01_T0001F001L01C01.tif'],
                    'second': ['plate1_A01_T1F1L1C1.tif']}
        settings.update(metadata_type='custom', custom_regex=(
            r'(?P<plateID>plate1)_(?P<wellID>A01)_T(?P<timeID>\d+)'
            r'F(?P<fieldID>\d+)L\d+C(?P<chanID>\d+)\.tif'))
    with pytest.raises(ValueError):
        core._watch_closed_pool_spec(settings, manifest, 'maphash', 'maskhash')


def test_duplicate_declared_filename_in_nested_folder_is_refused(tmp_path):
    source, watched, rows = _acquisition(tmp_path)
    _arrive(source, watched, rows)
    nested = watched / 'nested'
    nested.mkdir()
    shutil.copy2(source / rows[0]['target'], nested / rows[0]['target'])
    with pytest.raises(ValueError, match='unexpected or duplicate'):
        core._watch_folder_and_analyse(_settings(watched), CohortRecorder())
    assert not (watched / 'spacr_watch').exists()
