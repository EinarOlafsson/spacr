"""A fixed Convert map bounds one complete timelapse field series."""
import csv
import json
import shutil
import sqlite3

import numpy as np
import pytest
import tifffile

from spacr import convert, core
from tests.test_watch_folder_and_analyse import (
    MASK, MEASURE, Recorder, _channels, _rows, real_pipeline,
)
from tests.test_watch_nested_f548 import _fast


def _converted_series(tmp_path, wells=('A01',)):
    """Convert real TZYX acquisitions into exact T/Z/channel targets."""
    raw, output = tmp_path / 'raw', tmp_path / 'converted'
    for index, well in enumerate(wells):
        folder = raw / well
        folder.mkdir(parents=True)
        for channel, image in enumerate(_channels(index), start=1):
            later = np.where(image > 0, image + 10, 0).astype(image.dtype)
            frames = np.stack((np.stack((image // 2, image)),
                               np.stack((later // 2, later))))
            tifffile.imwrite(folder / f'field01_C{channel}.tif', frames,
                             metadata={'axes': 'TZYX'})
    result = convert.convert_folder({'src': str(raw), 'dst': str(output),
                                     'z_handling': 'keep', 'preview_rows': 0})
    assert result.is_complete
    rows = result.rows()
    assert len(rows) == 8 * len(wells)
    assert {int(row['t']) for row in rows} == {1, 2}
    assert {int(row['z']) for row in rows} == {1, 2}
    return output, rows


def test_mapped_series_waits_for_final_frame_then_matches_batch(tmp_path, real_pipeline):
    from spacr.measure import measure_crop

    output, rows = _converted_series(tmp_path)
    batch, watched = tmp_path / 'batch', tmp_path / 'watched'
    batch.mkdir()
    watched.mkdir()
    shutil.copy2(output / convert.MAP_FILENAME, watched / convert.MAP_FILENAME)
    for row in rows:
        shutil.copy2(output / row['target'], batch / row['target'])
        if int(row['t']) == 1:
            shutil.copy2(output / row['target'], watched / row['target'])
    mask = dict(MASK, timelapse=True)
    measure = dict(MEASURE, timelapse=True)
    core.preprocess_generate_masks(dict(mask, src=str(batch)))
    measure_crop(dict(measure, src=str(batch / 'merged')))
    recipe = tmp_path / 'measure.json'
    recipe.write_text(json.dumps(measure))
    watch_settings = dict(mask, **_fast(watched), watch_pipeline='mask_measure',
                          watch_measure_settings=str(recipe))
    first = core._watch_folder_and_analyse(watch_settings)
    assert not first['done'] and len(first['incomplete']) == 1
    for row in rows:
        if int(row['t']) == 2:
            shutil.copy2(output / row['target'], watched / row['target'])
    second = core._watch_folder_and_analyse(watch_settings)
    assert len(second['done']) == 1 and not second['failed'] and not second['incomplete']
    combined = watched / 'spacr_watch'
    batch_names = sorted(path.name for path in (batch / 'merged').glob('*.npy'))
    assert len(batch_names) == 2
    assert sorted(path.name for path in (combined / 'merged').glob('*.npy')) == batch_names
    for name in batch_names:
        np.testing.assert_array_equal(np.load(batch / 'merged' / name),
                                      np.load(combined / 'merged' / name))
    for table in ('cell', 'nucleus'):
        assert _rows(batch / 'measurements/measurements.db', table) == _rows(
            combined / 'measurements/measurements.db', table)
    with sqlite3.connect(combined / 'measurements/measurements.db') as connection:
        for table in ('cell', 'nucleus'):
            frames = connection.execute(
                f'SELECT DISTINCT prcf FROM "{table}" ORDER BY prcf').fetchall()
            assert frames == [('plate1_r1_c1_f1_t1',), ('plate1_r1_c1_f1_t2',)]
        assert connection.execute('SELECT COUNT(*) FROM spacr_watch_fields').fetchone() == (1,)
    assert core._watch_folder_and_analyse(watch_settings)['done'] == second['done']


def test_two_mapped_series_keep_their_field_members_separate(tmp_path):
    output, rows = _converted_series(tmp_path, wells=('A01', 'A02'))
    recorder = Recorder()
    result = core._watch_folder_and_analyse(dict(MASK, **_fast(output), timelapse=True),
                                            recorder)
    assert len(result['done']) == len(recorder.calls) == 2
    assert not result['failed'] and not result['incomplete']
    for key, files, _at in recorder.calls:
        well = 'A01' if '_A01_' in key else 'A02'
        assert len(files) == 8
        assert set(files) == {row['target'] for row in rows if row['well'] == well}


@pytest.mark.parametrize('kind', ['missing_frame_plane', 'sparse_time'])
def test_mapped_series_requires_a_dense_time_and_plane_grid(tmp_path, kind):
    output, rows = _converted_series(tmp_path)
    with (output / convert.MAP_FILENAME).open(newline='') as handle:
        mapped = list(csv.DictReader(handle))
    if kind == 'missing_frame_plane':
        mapped = [row for row in mapped if not (
            row['channel'] == '1' and row['z'] == '2' and row['t'] == '2')]
    else:
        row = next(row for row in mapped if row['t'] == '2')
        row['t'] = '3'
        row['target'] = convert.target_name('plate1', 'A01', 1,
                                            int(row['channel']), z=int(row['z']), t=3)
    with (output / convert.MAP_FILENAME).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=convert.MAP_COLUMNS)
        writer.writeheader()
        writer.writerows(mapped)
    with pytest.raises(ValueError, match='invalid conversion_map.csv'):
        core._watch_folder_and_analyse(dict(MASK, **_fast(output), timelapse=True),
                                       Recorder())
    assert not (output / 'spacr_watch').exists()


def test_unmapped_timelapse_and_plain_map_time_series_are_refused(tmp_path):
    with pytest.raises(ValueError, match='conversion_map.csv'):
        core._watch_folder_and_analyse(dict(MASK, **_fast(tmp_path), timelapse=True),
                                       Recorder())
    output, _rows_ = _converted_series(tmp_path)
    with pytest.raises(ValueError, match='invalid conversion_map.csv'):
        core._watch_folder_and_analyse(dict(MASK, **_fast(output)), Recorder())
    assert not (output / 'spacr_watch').exists()


@pytest.mark.parametrize('change', ['map', 'mask_recipe', 'measure_recipe'])
def test_series_resume_rejects_changed_provenance(tmp_path, change):
    output, rows = _converted_series(tmp_path)
    for row in rows:
        assert (output / row['target']).is_file()
    measure = tmp_path / 'measure.json'
    measure.write_text(json.dumps(dict(MEASURE, timelapse=True)))
    settings = dict(MASK, **_fast(output), timelapse=True,
                    watch_pipeline='mask_measure', watch_measure_settings=str(measure))
    recorder = Recorder()
    first = core._watch_folder_and_analyse(settings, recorder)
    assert len(first['done']) == len(recorder.calls) == 1
    ledger = (output / 'spacr_watch/watch_ledger.json').read_bytes()
    if change == 'map':
        with (output / convert.MAP_FILENAME).open('a') as handle:
            handle.write('\n')
    elif change == 'mask_recipe':
        settings['normalize'] = False
    else:
        measure.write_text(json.dumps(dict(MEASURE, timelapse=True,
                                           cell_min_size=1)))
    resumed = Recorder()
    with pytest.raises(ValueError, match='differ|changed'):
        core._watch_folder_and_analyse(settings, resumed)
    assert not resumed.calls
    assert (output / 'spacr_watch/watch_ledger.json').read_bytes() == ledger


def test_extra_or_changed_frame_cannot_mutate_a_committed_series(tmp_path, capsys):
    output, rows = _converted_series(tmp_path)
    settings = dict(MASK, **_fast(output), timelapse=True)
    extra = output / convert.target_name('plate1', 'A01', 1, 1, z=1, t=3)
    tifffile.imwrite(extra, np.ones((16, 16), np.uint16))
    waiting = Recorder()
    first = core._watch_folder_and_analyse(settings, waiting)
    assert first['incomplete'] and not waiting.calls
    extra.unlink()
    recorder = Recorder()
    done = core._watch_folder_and_analyse(settings, recorder)
    assert len(done['done']) == len(recorder.calls) == 1
    late = next(row for row in rows if int(row['t']) == 2)
    tifffile.imwrite(output / late['target'], np.zeros((16, 16), np.uint16))
    resumed = Recorder()
    assert core._watch_folder_and_analyse(settings, resumed)['done'] == done['done']
    assert not resumed.calls
    assert 'changed after it was analysed' in capsys.readouterr().out


def test_series_measure_recipe_must_keep_timepoint_identity(tmp_path):
    output, _rows_ = _converted_series(tmp_path)
    measure = tmp_path / 'measure.json'
    measure.write_text(json.dumps(dict(MEASURE, timelapse=False)))
    settings = dict(MASK, **_fast(output), timelapse=True,
                    watch_pipeline='mask_measure', watch_measure_settings=str(measure))
    with pytest.raises(ValueError, match='Measure settings must enable timelapse'):
        core._watch_folder_and_analyse(settings, Recorder())
    assert not (output / 'spacr_watch').exists()
