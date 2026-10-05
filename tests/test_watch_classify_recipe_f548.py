"""A watched CV inference run uses one model and collects each field once."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import core
from tests.test_watch_folder_and_analyse import _name, real_pipeline
from tests.test_watch_pipeline_resume_f548 import _bytes, _settings


def _arrive(folder, well, value):
    for channel in (1, 2):
        tifffile.imwrite(folder / _name(well, channel),
                         np.full((8, 8), value + channel - 1, np.uint16))


def _recipe(folder, model, **extra):
    path = folder / 'classify.json'
    values = {'classifier_family': 'cv', 'model_path': str(model),
              'train': False, 'test': False,
              'generate_training_dataset': False,
              'apply_model_to_dataset': True, 'crop_source': 'merged',
              'extract_channels': [0], 'cell_mask_dim': 1, **extra}
    path.write_text(json.dumps(values))
    return {**_settings(folder, 'mask_measure_classify'),
            'watch_classify_settings': str(path)}


def _predictions(database):
    with sqlite3.connect(database) as connection:
        return sorted(connection.execute(
            'SELECT prcfo, value, pred, cv_predictions FROM png_list').fetchall())


@pytest.fixture
def adapter(monkeypatch):
    from spacr import deep_spacr, measure

    calls = []
    hooks = []

    def masks(settings):
        folder = Path(settings['src'])
        value = int(tifffile.imread(sorted(folder.glob('*.tif'))[0])[0, 0])
        merged = folder / 'merged'
        merged.mkdir()
        np.save(merged / (folder.name + '.npy'),
                np.full((8, 8, 2), value, np.float32))

    def measurements(settings):
        merged = Path(settings['src'])
        folder = merged.parent
        value = int(np.load(next(merged.glob('*.npy')))[0, 0, 0])
        output = folder / 'measurements'
        output.mkdir()
        with sqlite3.connect(output / 'measurements.db') as connection:
            connection.execute('CREATE TABLE png_list (prcfo TEXT PRIMARY KEY, value REAL)')
            connection.execute('INSERT INTO png_list VALUES (?, ?)',
                               (folder.name, value))

    def infer(settings):
        model = Path(settings['model_path'])
        calls.append((model, json.loads(json.dumps(settings))))
        threshold = float(model.read_text())
        database = Path(settings['src']) / 'measurements' / 'measurements.db'
        with sqlite3.connect(database) as connection:
            connection.execute('ALTER TABLE png_list ADD COLUMN pred REAL')
            connection.execute('ALTER TABLE png_list ADD COLUMN cv_predictions INTEGER')
            connection.execute('UPDATE png_list SET pred = value / 10, '
                               'cv_predictions = value >= ?', (threshold,))
        for hook in hooks:
            hook(settings)

    monkeypatch.setattr(core, 'preprocess_generate_masks', masks)
    monkeypatch.setattr(measure, 'measure_crop', measurements)
    monkeypatch.setattr(deep_spacr, 'deep_spacr', infer)
    return masks, measurements, calls, hooks


def test_watched_predictions_match_the_same_per_field_batch_pipeline(tmp_path, adapter):
    masks, measurements, calls, _hooks = adapter
    model = tmp_path / 'model.pt'
    model.write_text('2')
    watched = tmp_path / 'watched'
    watched.mkdir()
    _arrive(watched, 'A01', 1)
    _arrive(watched, 'A02', 3)
    settings = _recipe(watched, model)

    result = core._watch_folder_and_analyse(settings)
    assert len(result['done']) == 2
    combined = _predictions(watched / 'spacr_watch/measurements/measurements.db')

    from spacr.classify import classify

    batch = []
    saved = json.loads(Path(settings['watch_classify_settings']).read_text())
    for well in ('A01', 'A02'):
        field = tmp_path / ('batch_' + well)
        field.mkdir()
        for channel in (1, 2):
            shutil.copy2(watched / _name(well, channel), field / _name(well, channel))
        masks({'src': str(field)})
        measurements({'src': str(field / 'merged')})
        classify({**saved, 'src': str(field)})
        batch.extend(_predictions(field / 'measurements/measurements.db'))

    assert sorted((row[1:] for row in combined)) == sorted(row[1:] for row in batch)
    assert [row[3] for row in combined] == [0, 1]
    assert all(str(model_path).startswith(str(watched / 'spacr_watch'))
               for model_path, _settings_used in calls[:2])


def test_model_and_recipe_are_frozen_until_resume(tmp_path, adapter):
    _masks, _measurements, calls, hooks = adapter
    model = tmp_path / 'model.pt'
    model.write_text('2')
    _arrive(tmp_path, 'A01', 1)
    _arrive(tmp_path, 'A02', 3)
    settings = _recipe(tmp_path, model, n_top_examples=1)

    def change_sources(_applied):
        if len(calls) == 1:
            model.write_text('4')
            _recipe(tmp_path, model, n_top_examples=9)

    hooks.append(change_sources)
    result = core._watch_folder_and_analyse(settings)
    assert len(result['done']) == 2
    assert _predictions(tmp_path / 'spacr_watch/measurements/measurements.db')[-1][3] == 1
    assert calls[0][0] == calls[1][0]
    assert calls[0][1]['n_top_examples'] == calls[1][1]['n_top_examples'] == 1
    _arrive(tmp_path, 'A03', 5)
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match='Classify settings or model differ'):
        core._watch_folder_and_analyse(settings)
    assert _bytes(tmp_path) == before


@pytest.mark.parametrize('change', [
    {'classifier_family': 'ml'}, {'train': True}, {'test': True},
    {'generate_training_dataset': True}, {'apply_model_to_dataset': False},
    {'crop_source': 'png'},
])
def test_training_and_unstable_crop_recipes_are_refused_before_output(tmp_path,
                                                                    change):
    model = tmp_path / 'model.pt'
    model.write_text('2')
    _arrive(tmp_path, 'A01', 1)
    settings = _recipe(tmp_path, model, **change)
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match='live Classify requires'):
        core._watch_folder_and_analyse(settings)
    assert _bytes(tmp_path) == before


def test_no_predictions_means_the_field_is_not_collected(tmp_path, adapter,
                                                        monkeypatch):
    from spacr import deep_spacr

    monkeypatch.setattr(deep_spacr, 'deep_spacr', lambda _settings: None)
    model = tmp_path / 'model.pt'
    model.write_text('2')
    _arrive(tmp_path, 'A01', 1)
    settings = _recipe(tmp_path, model)
    result = core._watch_folder_and_analyse(settings)
    assert result['failed'] == ['plate1_A01_0001_001']
    assert not (tmp_path / 'spacr_watch/measurements/measurements.db').exists()


def test_real_mask_and_measure_predictions_match_batch_size_one(
        tmp_path, real_pipeline, monkeypatch):
    from spacr import deep_spacr
    from spacr.measure import measure_crop
    from tests.test_watch_folder_and_analyse import MASK, MEASURE, _channels, _rows

    def infer(settings):
        database = Path(settings['src']) / 'measurements' / 'measurements.db'
        with sqlite3.connect(database) as connection:
            connection.execute('ALTER TABLE png_list ADD COLUMN pred REAL')
            connection.execute('ALTER TABLE png_list ADD COLUMN cv_predictions INTEGER')
            keys = [row[0] for row in connection.execute('SELECT prcfo FROM png_list')]
            for key in keys:
                score = (sum(key.encode('utf-8')) % 10) / 10
                connection.execute('UPDATE png_list SET pred = ?, '
                                   'cv_predictions = ? WHERE prcfo = ?',
                                   (score, int(score >= 0.5), key))

    monkeypatch.setattr(deep_spacr, 'deep_spacr', infer)
    source = tmp_path / 'source'
    source.mkdir()
    for shift, well in enumerate(('A01', 'A02')):
        for channel, image in enumerate(_channels(shift), start=1):
            tifffile.imwrite(source / _name(well, channel), image)
    model = tmp_path / 'model.pt'
    model.write_text('2')
    watched = tmp_path / 'watched'
    batch = tmp_path / 'batch'
    shutil.copytree(source, watched)
    shutil.copytree(source, batch)
    measure = {**MEASURE, 'save_png': True}
    measure_file = tmp_path / 'measure.json'
    measure_file.write_text(json.dumps(measure))
    settings = _recipe(watched, model)
    settings.update(MASK, src=str(watched), watch_folder=True,
                    watch_pipeline='mask_measure_classify',
                    watch_measure_settings=str(measure_file))
    result = core._watch_folder_and_analyse(settings)
    assert len(result['done']) == 2

    core.preprocess_generate_masks({**MASK, 'src': str(batch)})
    measure_crop({**measure, 'src': str(batch / 'merged')})
    from spacr.classify import classify
    saved = json.loads(Path(settings['watch_classify_settings']).read_text())
    classify({**saved, 'src': str(batch)})

    watched_db = watched / 'spacr_watch/measurements/measurements.db'
    batch_db = batch / 'measurements/measurements.db'
    assert _rows(watched_db, 'png_list') == _rows(batch_db, 'png_list')
