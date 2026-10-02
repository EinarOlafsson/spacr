"""Exact Convert filenames retain their field identity in nested acquisitions."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import tifffile

from spacr import core, convert
from tests.test_watch_folder_and_analyse import (
    Recorder, _channels, _rows, MASK, MEASURE, real_pipeline,
)
from tests.test_watch_nested_f548 import _fast


@pytest.fixture
def converted(tmp_path):
    """Produce genuine two-field Convert TIFFs and its unmodified default map."""
    raw, output = tmp_path / 'raw', tmp_path / 'converted'
    for index, well in enumerate(('A01', 'A02')):
        folder = raw / well
        folder.mkdir(parents=True)
        for channel, image in enumerate(_channels(index), start=1):
            tifffile.imwrite(folder / f'field01_C{channel}.tif', image)
    result = convert.convert_folder({'src': str(raw), 'dst': str(output), 'preview_rows': 0})
    assert result.is_complete
    return output, result.rows()


def _nested_copy(converted, destination):
    """Relocate mapped basenames beneath arbitrary folders without changing metadata."""
    output, rows = converted
    destination.mkdir()
    shutil.copy2(output / convert.MAP_FILENAME, destination / convert.MAP_FILENAME)
    paths = []
    for row in rows:
        folder = destination / 'arbitrary-group' / ('run-' + row['well'])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / row['target']
        shutil.copy2(output / row['target'], path)
        paths.append(path)
    return paths


def test_real_convert_map_accepts_exact_nested_companions_and_resumes(converted, tmp_path):
    watched = tmp_path / 'watched'
    paths = _nested_copy(converted, watched)
    before = {path: path.read_bytes() for path in [watched / convert.MAP_FILENAME, *paths]}
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(watched), recorder)
    assert len(result['done']) == len(recorder.calls) == 2
    assert not result['incomplete'] and not result['failed']
    ledger = json.loads(Path(result['ledger']).read_text())
    assert ledger['conversion_map_sha256'] == hashlib.sha256(before[watched / convert.MAP_FILENAME]).hexdigest()
    recorded = {name for entry in ledger['fields'].values() for name in entry['files']}
    assert recorded == {str(path.relative_to(watched)) for path in paths}
    assert core._watch_folder_and_analyse(_fast(watched), recorder)['done'] == result['done']
    assert len(recorder.calls) == 2
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('invalid', ['duplicate_location', 'cross_folder', 'missing', 'unexpected'])
def test_nested_map_still_refuses_ambiguous_or_incomplete_field(converted, tmp_path, invalid):
    watched = tmp_path / 'watched'
    paths = _nested_copy(converted, watched)
    first = paths[0]
    if invalid == 'duplicate_location':
        shutil.copy2(first, watched / first.name)
    elif invalid == 'cross_folder':
        first.rename(watched / first.name)
    elif invalid == 'missing':
        first.unlink()
    else:
        shutil.copy2(first, first.with_name(first.name.replace('C01', 'C03')))
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(watched), recorder)
    assert len(result['done']) == len(recorder.calls) == 1
    assert result['incomplete'] == ['plate1_A01_0001_001']
    assert all(key != 'plate1_A01_0001_001' for key, _files, _at in recorder.calls)


def test_nested_map_real_preprocessing_and_measure_equal_flat_batch(converted, tmp_path, real_pipeline):
    from spacr.measure import measure_crop

    output, _mapping = converted
    watched, batch = tmp_path / 'watched', tmp_path / 'batch'
    paths = _nested_copy(converted, watched)
    before = {path: path.read_bytes() for path in [watched / convert.MAP_FILENAME, *paths]}
    batch.mkdir()
    for path in output.glob('*.tif'):
        shutil.copy2(path, batch / path.name)
    core.preprocess_generate_masks(dict(MASK, src=str(batch)))
    measure_crop(dict(MEASURE, src=str(batch / 'merged')))
    settings = tmp_path / 'measure.json'
    settings.write_text(json.dumps(MEASURE))
    result = core.preprocess_generate_masks(dict(
        MASK, **_fast(watched), watch_pipeline='mask_measure',
        watch_measure_settings=str(settings)))
    assert len(result['done']) == 2 and not result['failed'] and not result['incomplete']
    for path in (batch / 'merged').glob('*.npy'):
        np.testing.assert_array_equal(np.load(path), np.load(watched / 'spacr_watch/merged' / path.name))
    for table in ('cell', 'nucleus'):
        assert _rows(batch / 'measurements/measurements.db', table) == _rows(
            watched / 'spacr_watch/measurements/measurements.db', table)
    assert all(path.read_bytes() == content for path, content in before.items())
