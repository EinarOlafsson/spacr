"""Watch resume cannot silently mix mask and measurement pipeline outputs."""
import json
from pathlib import Path

import pytest

from spacr import core
from tests.test_watch_folder_and_analyse import Recorder, _name
from tests.test_watch_nested_f548 import _fast, _write


def _settings(folder, pipeline):
    """Use real TIFF readiness and short polling for the selected pipeline."""
    return {**_fast(folder), 'watch_pipeline': pipeline}


def _images(folder, well):
    """Create one complete two-channel acquisition field."""
    for channel in (1, 2):
        _write(folder, _name(well, channel), channel)


def _bytes(folder):
    """Snapshot acquired and output file contents for preservation assertions."""
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob('*') if path.is_file()}


@pytest.mark.parametrize('first,second', [
    ('mask', 'mask_measure'), ('mask_measure', 'mask'),
])
def test_changed_pipeline_refuses_completed_resume_before_writes(tmp_path, first, second):
    _images(tmp_path, 'A01')
    result = core._watch_folder_and_analyse(_settings(tmp_path, first), Recorder())
    assert result['done'] == ['plate1_A01_0001_001']
    _images(tmp_path, 'A02')
    before = _bytes(tmp_path)
    recorder = Recorder()
    with pytest.raises(ValueError, match='pipeline.*separate watch workspace'):
        core._watch_folder_and_analyse(_settings(tmp_path, second), recorder)
    assert not recorder.calls
    assert _bytes(tmp_path) == before


def test_populated_legacy_ledger_without_pipeline_preserves_unknown_provenance(tmp_path):
    _images(tmp_path, 'A01')
    result = core._watch_folder_and_analyse(_settings(tmp_path, 'mask'), Recorder())
    ledger_path = Path(result['ledger'])
    ledger = json.loads(ledger_path.read_text())
    ledger.pop('pipeline')
    ledger_path.write_text(json.dumps(ledger))
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match='pipeline.*separate watch workspace'):
        core._watch_folder_and_analyse(_settings(tmp_path, 'mask'), Recorder())
    assert _bytes(tmp_path) == before


def test_changed_pipeline_refuses_failed_field_retry(tmp_path):
    _images(tmp_path, 'A01')
    failed = Recorder(fail=['plate1_A01_0001_001'])
    result = core._watch_folder_and_analyse(_settings(tmp_path, 'mask'), failed)
    assert result['failed']
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match='pipeline.*separate watch workspace'):
        core._watch_folder_and_analyse(_settings(tmp_path, 'mask_measure'), Recorder())
    assert _bytes(tmp_path) == before


@pytest.mark.parametrize('pipeline', ['mask', 'mask_measure'])
def test_unchanged_pipeline_resumes_old_fields_and_processes_new_fields(tmp_path, pipeline):
    _images(tmp_path, 'A01')
    first = core._watch_folder_and_analyse(_settings(tmp_path, pipeline), Recorder())
    prior = json.loads(Path(first['ledger']).read_text())['fields']['plate1_A01_0001_001']
    _images(tmp_path, 'A02')
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_settings(tmp_path, pipeline), recorder)
    assert result['done'] == ['plate1_A01_0001_001', 'plate1_A02_0001_001']
    assert [call[0] for call in recorder.calls] == ['plate1_A02_0001_001']
    ledger = json.loads(Path(result['ledger']).read_text())
    assert ledger['pipeline'] == pipeline
    assert ledger['fields']['plate1_A01_0001_001'] == prior


@pytest.mark.parametrize('old_pipeline', [None, 'mask'])
def test_empty_ledger_can_select_pipeline_without_old_results(tmp_path, old_pipeline):
    work = tmp_path / 'spacr_watch'
    work.mkdir()
    ledger = {'src': str(tmp_path), 'fields': {}}
    if old_pipeline is not None:
        ledger['pipeline'] = old_pipeline
    (work / 'watch_ledger.json').write_text(json.dumps(ledger))
    _images(tmp_path, 'A01')
    result = core._watch_folder_and_analyse(_settings(tmp_path, 'mask_measure'), Recorder())
    assert result['done'] == ['plate1_A01_0001_001']
    assert json.loads(Path(result['ledger']).read_text())['pipeline'] == 'mask_measure'
