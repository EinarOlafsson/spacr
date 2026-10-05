"""A fixed Convert map makes projected Z fields complete before live analysis."""
import csv
import json
import shutil

import numpy as np
import pytest
import tifffile

from spacr import convert, core
from tests.test_watch_folder_and_analyse import (
    MASK, MEASURE, Recorder, _channels, _rows, real_pipeline,
)
from tests.test_watch_nested_f548 import _fast


def _converted_z_series(tmp_path):
    """Make real two-plane TIFF acquisitions and Convert's fixed target map."""
    raw, output = tmp_path / 'raw', tmp_path / 'converted'
    for index, well in enumerate(('A01', 'A02')):
        folder = raw / well
        folder.mkdir(parents=True)
        for channel, image in enumerate(_channels(index), start=1):
            planes = np.stack((image // 2, image))
            tifffile.imwrite(folder / f'field01_C{channel}.tif', planes,
                             metadata={'axes': 'ZYX'})
    result = convert.convert_folder({'src': str(raw), 'dst': str(output),
                                     'z_handling': 'keep', 'preview_rows': 0})
    assert result.is_complete
    rows = result.rows()
    assert len(rows) == 8
    assert {int(row['z']) for row in rows} == {1, 2}
    return output, rows


def test_mapped_z_field_waits_for_every_plane_and_resumes(tmp_path):
    output, rows = _converted_z_series(tmp_path)
    watched = tmp_path / 'watched'
    watched.mkdir()
    shutil.copy2(output / convert.MAP_FILENAME, watched / convert.MAP_FILENAME)
    recorder = Recorder()
    for row in rows:
        if int(row['z']) == 1:
            shutil.copy2(output / row['target'], watched / row['target'])
    first = core._watch_folder_and_analyse(_fast(watched), recorder)
    assert not first['done'] and len(first['incomplete']) == 2
    assert not recorder.calls
    for row in rows:
        if int(row['z']) == 2:
            shutil.copy2(output / row['target'], watched / row['target'])
    second = core._watch_folder_and_analyse(_fast(watched), recorder)
    assert len(second['done']) == len(recorder.calls) == 2
    assert not second['incomplete'] and not second['failed']
    assert all(len(files) == 4 for _key, files, _at in recorder.calls)
    assert core._watch_folder_and_analyse(_fast(watched), recorder)['done'] == second['done']
    assert len(recorder.calls) == 2


@pytest.mark.parametrize('kind', ['missing_plane', 'sparse_z', 'time_series'])
def test_incomplete_or_temporal_map_is_refused_before_output(tmp_path, kind):
    output, rows = _converted_z_series(tmp_path)
    with (output / convert.MAP_FILENAME).open(newline='') as handle:
        mapped = list(csv.DictReader(handle))
    if kind == 'missing_plane':
        mapped = [row for row in mapped if not (
            row['well'] == 'A01' and row['channel'] == '1' and row['z'] == '2')]
    elif kind == 'sparse_z':
        row = next(row for row in mapped if row['well'] == 'A01'
                   and row['channel'] == '1' and row['z'] == '2')
        row['z'] = '3'
        row['target'] = convert.target_name('plate1', 'A01', 1, 1, z=3)
    else:
        row = mapped[0]
        row['t'] = '2'
        row['target'] = convert.target_name('plate1', 'A01', 1, 1,
                                            z=int(row['z']), t=2)
    with (output / convert.MAP_FILENAME).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=convert.MAP_COLUMNS)
        writer.writeheader()
        writer.writerows(mapped)
    with pytest.raises(ValueError, match='invalid conversion_map.csv'):
        core._watch_folder_and_analyse(_fast(output), Recorder())
    assert not (output / 'spacr_watch').exists()


def test_projected_z_watch_matches_batch_mask_and_measure(tmp_path, real_pipeline):
    from spacr.measure import measure_crop

    output, rows = _converted_z_series(tmp_path)
    batch, watched = tmp_path / 'batch', tmp_path / 'watched'
    batch.mkdir()
    watched.mkdir()
    shutil.copy2(output / convert.MAP_FILENAME, watched / convert.MAP_FILENAME)
    source_bytes = {}
    for row in rows:
        source = output / row['target']
        source_bytes[row['target']] = source.read_bytes()
        shutil.copy2(source, batch / row['target'])
        shutil.copy2(source, watched / row['target'])
    core.preprocess_generate_masks(dict(MASK, src=str(batch)))
    measure_crop(dict(MEASURE, src=str(batch / 'merged')))
    recipe = tmp_path / 'measure.json'
    recipe.write_text(json.dumps(MEASURE))
    result = core.preprocess_generate_masks(dict(
        MASK, **_fast(watched), watch_pipeline='mask_measure',
        watch_measure_settings=str(recipe)))
    assert len(result['done']) == 2 and not result['failed'] and not result['incomplete']
    combined = watched / 'spacr_watch'
    batch_names = sorted(path.name for path in (batch / 'merged').glob('*.npy'))
    watch_names = sorted(path.name for path in (combined / 'merged').glob('*.npy'))
    assert watch_names == batch_names and len(batch_names) == 2
    for name in batch_names:
        np.testing.assert_array_equal(np.load(batch / 'merged' / name),
                                      np.load(combined / 'merged' / name))
    for table in ('cell', 'nucleus'):
        assert _rows(batch / 'measurements/measurements.db', table) == _rows(
            combined / 'measurements/measurements.db', table)
    assert all((watched / name).read_bytes() == content
               for name, content in source_bytes.items())
