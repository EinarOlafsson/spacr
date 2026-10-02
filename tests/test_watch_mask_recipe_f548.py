"""A watched plate must not mix different Mask recipes across fields."""
import json
from pathlib import Path

import pytest

from spacr import core
from tests.test_watch_folder_and_analyse import Recorder
from tests.test_watch_pipeline_resume_f548 import _bytes, _images, _settings


def test_mask_recipe_is_captured_before_first_field(tmp_path):
    settings = {**_settings(tmp_path, 'mask'), 'diameter': 18}
    _images(tmp_path, 'A01')
    _images(tmp_path, 'A02')
    seen = []
    recorder = Recorder()

    def analyse(field_dir, snapshot):
        seen.append(snapshot['diameter'])
        recorder(field_dir, snapshot)
        settings['diameter'] = 38

    result = core._watch_folder_and_analyse(settings, analyse)
    assert result['done'] == ['plate1_A01_0001_001', 'plate1_A02_0001_001']
    assert seen == [18, 18]
    assert settings['diameter'] == 38
    assert 'mask_settings_sha256' in json.loads(Path(result['ledger']).read_text())


def test_changed_mask_recipe_refuses_resume_before_any_output_write(tmp_path):
    _images(tmp_path, 'A01')
    settings = {**_settings(tmp_path, 'mask'), 'diameter': 18}
    first = core._watch_folder_and_analyse(settings, Recorder())
    assert first['done']
    _images(tmp_path, 'A02')
    before = _bytes(tmp_path)
    recorder = Recorder()
    with pytest.raises(ValueError, match='Mask settings differ.*separate watch workspace'):
        core._watch_folder_and_analyse({**settings, 'diameter': 38}, recorder)
    assert recorder.calls == []
    assert _bytes(tmp_path) == before


def test_missing_legacy_mask_provenance_refuses_a_populated_resume(tmp_path):
    _images(tmp_path, 'A01')
    first = core._watch_folder_and_analyse(_settings(tmp_path, 'mask'), Recorder())
    path = Path(first['ledger'])
    ledger = json.loads(path.read_text())
    ledger.pop('mask_settings_sha256')
    path.write_text(json.dumps(ledger))
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match='Mask settings differ'):
        core._watch_folder_and_analyse(_settings(tmp_path, 'mask'), Recorder())
    assert _bytes(tmp_path) == before


def test_timing_changes_do_not_change_a_mask_recipe(tmp_path):
    _images(tmp_path, 'A01')
    settings = {**_settings(tmp_path, 'mask'), 'diameter': 18}
    core._watch_folder_and_analyse(settings, Recorder())
    _images(tmp_path, 'A02')
    later = {**settings, 'watch_poll_seconds': 0.02,
             'watch_idle_minutes': 0.02}
    result = core._watch_folder_and_analyse(later, Recorder())
    assert result['done'] == ['plate1_A01_0001_001', 'plate1_A02_0001_001']


def test_empty_watch_can_change_mask_recipe(tmp_path):
    settings = {**_settings(tmp_path, 'mask'), 'diameter': 18}
    assert not core._watch_folder_and_analyse(settings, Recorder())['done']
    _images(tmp_path, 'A01')
    result = core._watch_folder_and_analyse({**settings, 'diameter': 38}, Recorder())
    assert result['done'] == ['plate1_A01_0001_001']


def test_nonfinite_mask_recipe_refuses_before_creating_workspace(tmp_path):
    with pytest.raises(ValueError, match='finite JSON-compatible'):
        core._watch_folder_and_analyse(
            {**_settings(tmp_path, 'mask'), 'diameter': float('nan')}, Recorder())
    assert not (tmp_path / 'spacr_watch').exists()
