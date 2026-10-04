"""Measure exposes alpha lineage controls and calls the real measured-colour engine."""
import sqlite3

import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings

from spacr._lineage_measurements import _write_lineage_sources

_KEYS = ('timelapse_lineage', 'timelapse_lineage_color_by', 'timelapse_lineage_max_distance',
         'timelapse_lineage_min_division_h')


def test_measure_controls_keep_alpha_gate_and_saved_values(qtbot, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'alpha.ini'), QSettings.IniFormat))
    screen = AppScreen('measure')
    try:
        screen.set_dimension('t', True)
        screen._open_the_heading_of(_KEYS[0])
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key) for key in _KEYS)
        assert screen._settings_model.set_value_for_key(_KEYS[0], True)
        assert screen._settings_model.set_value_for_key(_KEYS[1], 'cell_area')
        preferences._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key) for key in _KEYS)
        preferences._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        values = screen._settings_model.collect()
        assert values[_KEYS[0]] is True and values[_KEYS[1]] == 'cell_area'
        assert not any(screen.setting_row_is_visible(key) for key in _KEYS)
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


def test_measure_pipeline_uses_saved_filename_mapping_for_actual_area(tmp_path):
    from spacr.measure import measure_crop

    merged = tmp_path / 'merged'
    merged.mkdir()
    tracks_dir = tmp_path / 'tracks'
    tracks_dir.mkdir()
    labels, names, rows = [], [], []
    for frame, (time, size) in enumerate([(100, 2), (103, 3), (109, 4)]):
        mask = np.zeros((32, 32), dtype=np.uint16)
        mask[10:10 + size, 10:10 + size] = 7
        image = np.full(mask.shape, 100, dtype=np.uint16)
        image[mask > 0] = 1000 + frame * 100
        name = f'plate1_A01_1_{time}.npy'
        np.save(merged / name, np.stack([image, mask], axis=-1))
        labels.append(mask)
        names.append(name)
        rows.append({'frame': frame, 'track_id': 7, 'original_label': 7,
                     'x': 11, 'y': 11, 'parent_track_id': 0})
    path = tracks_dir / 'trackpy_tracks_cell_plate1_A01_1.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    _write_lineage_sources(path, 'cell', names, labels)
    before = {file: file.read_bytes() for file in [path, *(merged / n for n in names)]}
    measure_crop({
        'src': str(merged), 'channels': [0], 'cell_mask_dim': 1,
        'nucleus_mask_dim': None, 'pathogen_mask_dim': None, 'organelle_mask_dim': None,
        'cell_chann_dim': 0, 'nucleus_chann_dim': None, 'pathogen_chann_dim': None,
        'cell_min_size': 0, 'nucleus_min_size': 0, 'pathogen_min_size': 0,
        'cytoplasm_min_size': 0, 'uninfected': True, 'cytoplasm': False,
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'representative_images': False, 'plot': False, 'save': False,
        'verbose': False, 'n_jobs': 1, 'timelapse': True, 'timelapse_objects': ['cell'],
        'timelapse_lineage': True, 'timelapse_lineage_color_by': 'cell_area',
        'test_mode': False, 'normalize': False, 'homogeneity': False,
        'radial_dist': False, 'calculate_correlation': False,
        'merge_edge_pathogen_cells': False, 'experiment': 'lineage_integration'})
    db = tmp_path / 'measurements' / 'measurements.db'
    with sqlite3.connect(db) as connection:
        measured = pd.read_sql_query('SELECT prcf, cell_area FROM cell ORDER BY prcf', connection)
    assert sorted(measured['cell_area']) == [4, 9, 16]
    saved = pd.read_csv(tracks_dir / 'lineage_measured' / (path.stem + '_segments.csv'))
    assert len(saved) == 1
    assert saved.iloc[0]['color_cell_area'] == np.mean([4, 9, 16])
    assert saved.iloc[0]['n_frames'] == 3
    for file, content in before.items():
        assert file.read_bytes() == content
