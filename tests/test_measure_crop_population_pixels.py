"""Separate fresh populations keep archived crop paths and pixels immutable."""
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from spacr import measure
from spacr.resume import read_recorded_settings
from tests.test_measure_crop_core_synth import _build_merged_stack, _settings_for, _write_stack


KEY = '_measurement_crop_output_dir'


def _paths(database, table='png_list'):
    with sqlite3.connect(database) as db:
        return [Path(row[0]) for row in db.execute(f'SELECT png_path FROM "{table}"')]


def test_three_actual_runs_preserve_every_prior_png_and_array_and_resume_reuses_folder(
        tmp_path, synth_masks_multi, rng):
    merged, name = _write_stack(tmp_path, _build_merged_stack(synth_masks_multi, rng))
    settings = _settings_for(merged, resume=False, database_write_queue_gib=0, save_arrays=True)
    database = tmp_path / 'measurements' / 'measurements.db'
    preserved = {}
    populations = []
    roots = []
    for index in range(3):
        if index:
            np.save(merged / name, _build_merged_stack(synth_masks_multi, rng))
        measure.measure_crop(dict(settings))
        paths = _paths(database)
        assert len(paths) == 4 and all(path.is_file() for path in paths)
        assert not set(paths).intersection(preserved)
        populations.append(paths)
        recorded = read_recorded_settings(str(database))
        root = recorded.get(KEY)
        roots.append(root)
        if index == 0:
            assert root is None
        else:
            assert root and Path(root).parent == tmp_path / 'data'
            assert all(Path(root) in path.parents for path in paths)
            assert roots[index] != roots[index - 1]
        for old_path, old_bytes in preserved.items():
            assert old_path.read_bytes() == old_bytes
        preserved.update({path: path.read_bytes() for path in paths})
        arrays = list((Path(root) if root else tmp_path / 'data').rglob('*.png.npy'))
        assert len(arrays) == 4
        for path in arrays:
            preserved.setdefault(path, path.read_bytes())
    assert _paths(database, 'png_list_1') == populations[0]
    assert _paths(database, 'png_list_2') == populations[1]
    assert populations[0][0].read_bytes() != populations[1][0].read_bytes()
    with sqlite3.connect(database) as db:
        before_tables = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        for table in ('cell', 'cytoplasm', 'nucleus', 'pathogen', 'png_list', 'intensity_rescale'):
            try:
                db.execute(f'DELETE FROM "{table}"')
            except sqlite3.OperationalError:
                pass
    measure.measure_crop(dict(settings, resume=True))
    assert read_recorded_settings(str(database))[KEY] == roots[-1]
    assert _paths(database) == populations[-1]
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall() == before_tables
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_fresh_plan_ignores_imported_directory_and_creates_no_folders(tmp_path):
    merged = tmp_path / 'merged'
    merged.mkdir()
    database = tmp_path / 'measurements' / 'measurements.db'
    settings = {'src': str(merged), 'resume': False, KEY: '/some/other/project/data/old'}
    measure._prepare_measurement_crop_output(settings, [])
    assert KEY not in settings
    plan = [{'db_path': str(database)}]
    measure._prepare_measurement_crop_output(settings, plan)
    first = settings[KEY]
    measure._prepare_measurement_crop_output(settings, plan)
    assert settings[KEY] != first
    assert not (tmp_path / 'data').exists()


def test_resume_restores_recorded_current_directory_and_refuses_another_project(tmp_path):
    from spacr.errors import ConfigurationError
    from spacr.io import _save_settings_to_db
    merged = tmp_path / 'merged'
    merged.mkdir()
    saved = str(tmp_path / 'data' / 'measure_run_saved')
    _save_settings_to_db({'src': str(merged), KEY: saved})
    settings = {'src': str(merged), 'resume': True, KEY: 'ignored'}
    measure._prepare_measurement_crop_output(settings, [])
    assert settings[KEY] == saved
    _save_settings_to_db({'src': str(merged), KEY: str(tmp_path.parent / 'other' / 'data')})
    with pytest.raises(ConfigurationError, match='outside'):
        measure._prepare_measurement_crop_output(settings, [])
    assert not (tmp_path / 'data').exists()
