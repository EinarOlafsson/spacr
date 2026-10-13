"""Every field in a watched measurement workspace uses one captured recipe."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from spacr import core
from tests.test_watch_pipeline_resume_f548 import _bytes, _images, _settings


@pytest.fixture
def adapter(monkeypatch):
    """Exercise the real field adapter with bounded mask/measurement stand-ins."""
    from spacr import measure

    calls = []
    hooks = []

    def masks(settings):
        """Write the merged artifact that the real adapter requires."""
        merged = Path(settings['src']) / 'merged'
        merged.mkdir()
        np.save(merged / (merged.parent.name + '.npy'), np.zeros((2, 2, 2)))

    def measurements(settings):
        """Record applied settings and write a real per-field SQLite artifact."""
        calls.append(json.loads(json.dumps(settings)))
        output = Path(settings['src']).parent / 'measurements'
        output.mkdir()
        with sqlite3.connect(output / 'measurements.db') as connection:
            connection.execute('CREATE TABLE measurement (field TEXT, save_png INTEGER)')
            connection.execute('INSERT INTO measurement VALUES (?, ?)',
                               (output.parent.name, int(settings.get('save_png', False))))
        for hook in hooks:
            hook(settings)

    monkeypatch.setattr(core, 'preprocess_generate_masks', masks)
    monkeypatch.setattr(measure, 'measure_crop', measurements)
    return calls, hooks


def _recipe(folder, **extra):
    """Save supported JSON Measure settings and return watch settings."""
    path = folder / 'measure.json'
    path.write_text(json.dumps({'channels': [0, 1], 'save_png': False, **extra}))
    return {**_settings(folder, 'mask_measure'), 'watch_measure_settings': str(path)}


def test_measure_recipe_is_captured_before_first_field(tmp_path, adapter):
    calls, hooks = adapter
    settings = _recipe(tmp_path)
    _images(tmp_path, 'A01')
    _images(tmp_path, 'A02')

    def change_source(_settings):
        """Simulate editing the same settings file while a watch is active."""
        Path(settings['watch_measure_settings']).write_text(
            json.dumps({'channels': [0, 1], 'save_png': True}))

    hooks.append(change_source)
    result = core._watch_folder_and_analyse(settings)
    assert len(result['done']) == 2
    assert [call['save_png'] for call in calls] == [False, False]
    with sqlite3.connect(tmp_path / 'spacr_watch/measurements/measurements.db') as connection:
        assert connection.execute('SELECT save_png FROM measurement').fetchall() == [(0,), (0,)]


def test_changed_recipe_refuses_resume_before_any_output_write(tmp_path, adapter):
    calls, _hooks = adapter
    settings = _recipe(tmp_path)
    _images(tmp_path, 'A01')
    core._watch_folder_and_analyse(settings)
    _recipe(tmp_path, save_png=True)
    _images(tmp_path, 'A02')
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match='Measure settings.*separate watch workspace'):
        core._watch_folder_and_analyse(settings)
    assert len(calls) == 1
    assert _bytes(tmp_path) == before


def test_equivalent_recipe_ignores_file_format_src_and_adjustable_timing(tmp_path, adapter):
    calls, _hooks = adapter
    settings = _recipe(tmp_path, src='/irrelevant/old')
    _images(tmp_path, 'A01')
    first = core._watch_folder_and_analyse(settings)
    before = json.loads(Path(first['ledger']).read_text())['fields']
    replacement = tmp_path / 'moved-recipe.json'
    replacement.write_text('{\n "save_png": false, "src": "/different", "channels": [0, 1]\n}')
    settings.update(watch_measure_settings=str(replacement), watch_poll_seconds=0.02,
                    watch_settle_seconds=0.02, watch_idle_minutes=0.002)
    _images(tmp_path, 'A02')
    result = core._watch_folder_and_analyse(settings)
    assert len(calls) == 2 and len(result['done']) == 2
    assert json.loads(Path(result['ledger']).read_text())['fields']['plate1_A01_0001_001'] == before['plate1_A01_0001_001']


def test_measure_receives_independent_nested_copies(tmp_path, adapter):
    calls, hooks = adapter
    settings = _recipe(tmp_path)
    _images(tmp_path, 'A01')
    _images(tmp_path, 'A02')

    def mutate_applied(measure_settings):
        """Simulate downstream defaults modifying a nested settings value."""
        measure_settings['channels'].append(9)

    hooks.append(mutate_applied)
    core._watch_folder_and_analyse(settings)
    assert [call['channels'] for call in calls] == [[0, 1], [0, 1]]


def test_missing_legacy_recipe_is_not_assumed(tmp_path, adapter):
    settings = _recipe(tmp_path)
    _images(tmp_path, 'A01')
    result = core._watch_folder_and_analyse(settings)
    path = Path(result['ledger'])
    ledger = json.loads(path.read_text())
    ledger.pop('measure_settings_sha256', None)
    path.write_text(json.dumps(ledger))
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match='Measure settings.*separate watch workspace'):
        core._watch_folder_and_analyse(settings)
    assert _bytes(tmp_path) == before


def test_default_measure_channels_are_bound_on_resume(tmp_path, adapter):
    settings = _settings(tmp_path, 'mask_measure')
    _images(tmp_path, 'A01')
    core._watch_folder_and_analyse(settings)
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match='Measure settings.*separate watch workspace'):
        core._watch_folder_and_analyse({**settings, 'channels': [0]})
    assert _bytes(tmp_path) == before


def test_mask_only_does_not_read_or_bind_measure_recipe(tmp_path, adapter):
    settings = {**_settings(tmp_path, 'mask'), 'watch_measure_settings': '/missing/file.json'}
    _images(tmp_path, 'A01')
    result = core._watch_folder_and_analyse(settings)
    assert result['done'] and not adapter[0]
    assert 'measure_settings_sha256' not in json.loads(Path(result['ledger']).read_text())


def test_empty_watch_can_change_measure_recipe(tmp_path, adapter):
    settings = _recipe(tmp_path)
    assert not core._watch_folder_and_analyse(settings)['done']
    _recipe(tmp_path, save_png=True)
    _images(tmp_path, 'A01')
    assert core._watch_folder_and_analyse(settings)['done']
    assert adapter[0][0]['save_png'] is True
