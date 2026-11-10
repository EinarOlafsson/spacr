"""HRMAn-style infection and replication readout (item 898, alpha).

Vacuoles are parasites grouped by gap, hosted by majority overlap; a
replication classifier is trained on an external table and applied after a
size calibration; vacuoles under a parasite-signal floor are dropped; wells
and conditions are summarised as the reference titration_summary.py does.
On the reanalysis data (plate103) the spaCR functions reproduce the
reference vacuole table on real masks, the classifier calls, the per-well
and per-condition tables and the cross-validation report.
"""
import os

import numpy as np
import pandas as pd
import pytest

from spacr import host_pathogen as hp

DATA = os.path.join(os.path.dirname(__file__), 'data', 'vacuole_titration_898')


def _field():
    """Two host cells; three parasites, two of them 3 px apart, one in a tiny cell."""
    cells = np.zeros((40, 60), np.int32)
    cells[0:30, 0:30] = 1
    cells[0:30, 30:60] = 2
    cells[32:40, 0:5] = 3
    parasites = np.zeros_like(cells)
    parasites[5:10, 5:10] = 1
    parasites[5:10, 13:18] = 2
    parasites[20:25, 40:45] = 3
    parasites[34:38, 1:5] = 4
    signal = np.where(parasites > 0, 2000.0, 100.0)
    signal[20:25, 40:45] = 500.0
    return cells, parasites, signal


def test_nearby_parasites_are_one_vacuole_and_hosts_are_by_majority():
    cells, parasites, signal = _field()
    vacuoles, links = hp.vacuoles_from_parasites(parasites, gap_px=4)
    assert vacuoles[7, 7] == vacuoles[7, 15] != vacuoles[22, 42]
    assert links.set_index('label').loc[[1, 2], 'pathogen_id'].nunique() == 1
    apart, _ = hp.vacuoles_from_parasites(parasites, gap_px=1)
    assert apart[7, 7] != apart[7, 15]
    empty, none = hp.vacuoles_from_parasites(np.zeros((5, 5), int))
    assert none is None and not empty.any()
    cell_table, vac = hp.vacuole_objects(cells, parasites, signal, {'host_min_size': 300},
                                         well='B03', field=2)
    assert list(vac.columns) == hp._VACUOLE_COLUMNS
    by_area = vac.sort_values('area_px')
    assert list(by_area['n_parasites_cellpose']) == [1, 1, 2]
    assert list(by_area['host_cell']) == [0, 2, 1]
    assert by_area['tox_mean'].iloc[1] == 500.0
    assert set(cell_table['cell']) == {1, 2} and cell_table['infected'].all()
    assert (cell_table['well'] == 'B03').all() and (vac['field'] == 2).all()
    _, small = hp.vacuole_objects(cells, parasites, signal, {'vacuole_min_size': 30})
    assert len(small) == 1
    with pytest.raises(ValueError):
        hp.vacuole_objects(cells, parasites, settings={'host_assignment': 'centroid'})


def test_modal_area_and_the_size_calibration():
    areas = np.r_[np.full(50, 100.0), np.full(10, 400.0), [3.0]]
    assert hp.modal_area(areas) == pytest.approx(100, rel=0.05)
    cal = hp.size_calibration(areas, {'size_calibration': 'modal_single',
                                      'size_calibration_reference_px': 57.0})
    assert cal['k'] == pytest.approx(57.0 / cal['modal_px']) and cal['reference_px'] == 57.0
    assert hp.size_calibration(areas)['k'] == 1.0
    with pytest.raises(ValueError):
        hp.size_calibration(areas, {'size_calibration': 'median'})


def _training(seed=0, n=60):
    """A replication training table: area grows with parasites per vacuole."""
    rng = np.random.default_rng(seed)
    classes = np.repeat([1, 2, 4, 8], n)
    return pd.DataFrame({'Num Pix': classes * 50 * rng.uniform(0.8, 1.2, len(classes)),
                         'Circularity': rng.uniform(0.6, 1, len(classes)),
                         'Convexity': rng.uniform(0.8, 1, len(classes)),
                         'Extend': rng.uniform(0.6, 0.9, len(classes)),
                         'Diameter': np.sqrt(classes * 50) * 1.2,
                         'Parasites/vacuole': classes})


def test_the_classifier_reads_the_training_table_and_scales_sizes_by_k():
    training = _training()
    settings = {'replication_classifier': 'hist_gbm'}
    vacuoles = pd.DataFrame({'area_px': [100.0, 800.0], 'circularity': [0.8, 0.8],
                             'solidity': [0.9, 0.9], 'extent': [0.7, np.nan],
                             'feret_max': [np.sqrt(100) * 1.2, np.sqrt(800) * 1.2]})
    calls = hp.classify_replication(vacuoles, training, settings, k=0.5)
    assert list(calls) == [1, 8]
    unscaled = hp.classify_replication(vacuoles, training, settings, k=1.0)
    assert unscaled[0] == 2
    with pytest.raises(ValueError):
        hp.classify_replication(vacuoles, training, {})
    cv = hp.replication_classifier_cv(training, {'replication_cv_folds': 3})
    assert cv['majority'] == 0.25 and list(cv['quantiles'].index) == [1, 2, 4, 8]
    assert cv['text'].startswith('majority 0.250\n')
    scores = cv['scores'].set_index('target')
    assert set(scores.index) == {'4class', 'binary>1'} and (cv['scores']['accuracy'] > 0.8).all()
    three = hp.replication_classifier_cv(training, {'replication_cv_folds': 3,
                                                   'replication_classes': '1,2,4'})
    assert three['majority'] == pytest.approx(1 / 3)
    as_json = {'replication_training_columns': '{"Num Pix": "area_px"}',
               'replication_classifier': 'hist_gbm'}
    assert len(hp.classify_replication(vacuoles, training, as_json)) == 2


def test_the_signal_floor_is_a_number_or_an_uninfected_percentile():
    vac = pd.DataFrame({'well': ['B02'] * 4 + ['B03'] * 2, 'status': ['uninfected'] * 4 + ['infected'] * 2,
                        'tox_mean': [100, 200, 300, 400, 900, 2000]})
    assert hp.infection_floor(vac) is None
    assert hp.infection_floor(vac, {'infection_signal_floor': 1000}) == 1000.0
    pct = {'infection_signal_floor': 'uninfected_percentile', 'infection_floor_percentile': 50}
    assert hp.infection_floor(vac, pct) == 250.0
    assert hp.infection_floor(vac, {**pct, 'infection_uninfected_wells': 'B03'}) == 1450.0
    with pytest.raises(ValueError):
        hp.infection_floor(vac[vac['status'] == 'infected'], pct)


def test_wells_and_conditions_match_the_reference_tables():
    """The reference per-well rows of two PRU 0 IU wells, rebuilt from their vacuoles."""
    plate = pd.read_csv(os.path.join(DATA, 'plate103_plate_map.csv'))
    assert len(plate) == 60 and plate.loc[plate['well'] == 'B02', 'status'].item() == 'uninfected'
    cells = pd.DataFrame({'well': ['B03'] * 4 + ['C03'] * 4 + ['D03'], 'cell': range(9),
                          'infected': [True, True, False, False, True, False, False, False, False],
                          'strain': 'PRU', 'status': 'infected', 'dose': 0.0})
    vac = pd.DataFrame({'well': ['B03', 'B03', 'C03'], 'vacuole': [1, 2, 1], 'host_cell': [1, 0, 3],
                        'area_px': [100, 300, 50], 'n_parasites_clf': [1, 4, 2],
                        'n_parasites_cellpose': [1, 2, 1]})
    vac['repl_clf'] = vac['n_parasites_clf'] > 1
    vac['repl_cellpose'] = vac['n_parasites_cellpose'] > 1
    wells = hp.titration_wells(cells, vac, {'titration_min_well_cells': 0.3}).set_index('well')
    assert wells.loc['B03', 'pct_infected'] == 50 and wells.loc['B03', 'vacuoles_unhosted'] == 1
    assert wells.loc['B03', 'pct_replicating_clf'] == 50 and wells.loc['B03', 'mean_vacuole_area_px'] == 200
    assert wells.loc['D03', 'vacuoles'] == 0 and not wells.loc['D03', 'qc_pass']
    reference = pd.read_csv(os.path.join(DATA, 'hrman_reference_values.csv'))
    cond = hp.titration_conditions(wells.reset_index(), reference)
    row = cond.iloc[0]
    assert row['wells'] == 2 and row['cells'] == 8 and row['vacuoles'] == 3
    assert row['pct_infected mean'] == 37.5 and row['HRMAn pct_replicating'] == 50.05
    assert row['HRMAn mean_size err(as stored)'] == 1.116
    assert 'HRMAn mean_size' not in hp.titration_conditions(wells.reset_index())


def _merged_plate(folder):
    """Two fields of a merged plate: channel 1 parasite signal, mask 3 cells, mask 5 parasites."""
    cells, parasites, signal = _field()
    os.makedirs(folder)
    for well in ('B03', 'B02'):
        stack = np.zeros(cells.shape + (6,), np.float32)
        stack[..., 1] = signal if well == 'B03' else 50.0
        stack[..., 3] = cells
        stack[..., 5] = parasites if well == 'B03' else 0
        np.save(os.path.join(folder, f'plate103_{well}_0.npy'), stack)


def test_a_titration_settings_run_from_merged_arrays(tmp_path):
    from spacr.tabular import write_table

    merged = tmp_path / 'merged'
    _merged_plate(str(merged))
    training = tmp_path / 'Training_dataset.csv'
    write_table(_training(), str(training), canonicalise=False)
    settings = {'src': str(tmp_path), 'vacuole_from_parasites': True,
                'titration_merged_folder': str(merged),
                'titration_plate_map': os.path.join(DATA, 'plate103_plate_map.csv'),
                'titration_reference_values': os.path.join(DATA, 'hrman_reference_values.csv'),
                'replication_classifier': 'hist_gbm', 'replication_training_table': str(training),
                'replication_cv_folds': 3, 'size_calibration': 'modal_single',
                'infection_signal_floor': 1000, 'host_min_size': 300}
    result = hp.analyze_host_pathogen(settings)
    assert result['floor'] == 1000.0 and len(result['vacuoles']) == 2
    wells = result['wells'].set_index('well')
    assert wells.loc['B03', 'pct_infected'] == 50.0 and wells.loc['B02', 'vacuoles'] == 0
    assert result['calibration']['k'] == pytest.approx(57.0 / result['calibration']['modal_px'])
    for name in ('titration_vacuoles.csv', 'titration_per_well.csv', 'titration_per_condition.csv',
                 'titration_calibration.txt', 'replication_classifier_cv.txt'):
        assert os.path.isfile(os.path.join(result['folder'], name))
    write_table(result['vacuoles'][hp._VACUOLE_COLUMNS], str(tmp_path / 'vacuoles.csv'),
                canonicalise=False)
    cells = result['wells'][['well']].assign(field=0, cell=1, area_px=400, infected=False)
    write_table(cells, str(tmp_path / 'cells.csv'), canonicalise=False)
    again = hp.analyze_titration({'src': str(tmp_path), 'save': False,
                                  'titration_object_tables': str(tmp_path),
                                  'titration_plate_map': settings['titration_plate_map']})
    assert again['cv'] is None and again['vacuoles']['n_parasites_clf'].isna().all()
    assert 'folder' not in again
    with pytest.raises(ValueError):
        hp.analyze_titration({'src': str(tmp_path), 'titration_plate_map': 'x.csv'})
    bad = tmp_path / 'map.csv'
    write_table(pd.DataFrame({'well': ['B03']}), str(bad), canonicalise=False)
    with pytest.raises(ValueError):
        hp._plate_map(str(bad))
    empty = tmp_path / 'empty'
    os.makedirs(empty)
    np.save(empty / 'plate103_B02_0.npy', np.zeros((40, 60, 6), np.float32))
    cells_only, none = hp._titration_objects({**hp.TITRATION_DEFAULTS, 'vacuole_from_parasites': True,
                                              'titration_merged_folder': str(empty)})
    assert none.empty and list(none.columns) == hp._VACUOLE_COLUMNS and cells_only.empty


def test_titration_settings_are_registered_alpha():
    from spacr.settings import ALPHA_FEATURES

    names = set(hp.TITRATION_DEFAULTS) & set(ALPHA_FEATURES[898]['settings'])
    defaults = hp.default_settings()
    assert set(names) == set(hp.TITRATION_DEFAULTS)
    assert defaults['vacuole_gap_px'] == 4 and defaults['replication_classifier'] == 'none'


def _measure_field(tmp_path, **over):
    """One 80 x 80 field: two cells, three parasites (two 2 px apart), signal 9 on parasites."""
    from spacr import measure
    from spacr.settings import get_measure_crop_settings

    merged = tmp_path / 'merged'
    merged.mkdir(parents=True)
    cell = np.zeros((80, 80), np.uint16)
    cell[5:40, 5:40] = 1
    cell[45:78, 45:78] = 2
    nucleus = np.zeros_like(cell)
    nucleus[15:25, 15:25] = 1
    nucleus[55:65, 55:65] = 2
    par = np.zeros_like(cell)
    par[10:16, 10:16] = 1
    par[10:16, 18:24] = 2
    par[50:56, 50:56] = 3
    signal = np.where(par > 0, 9, 1).astype(np.uint16)
    dna = np.where(nucleus > 0, 5, 1).astype(np.uint16)
    np.save(merged / 'plate1_B02_1.npy', np.stack([dna, signal, cell, nucleus, par], axis=-1))
    settings = get_measure_crop_settings({})
    settings.update({'src': str(merged), 'channels': [0, 1], 'cell_mask_dim': 2,
                     'nucleus_mask_dim': 3, 'pathogen_mask_dim': 4, 'cell_min_size': 0,
                     'nucleus_min_size': 0, 'pathogen_min_size': 0, 'cytoplasm_min_size': 0,
                     'save_png': False, 'save_arrays': False, 'plot': False, 'verbose': False,
                     'n_jobs': 1, **over})
    measure.measure_crop(settings)
    import sqlite3

    with sqlite3.connect(tmp_path / 'measurements' / 'measurements.db') as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        cells = pd.read_sql('SELECT * FROM cell', db)
        vacuoles = (pd.read_sql('SELECT * FROM vacuole', db) if 'vacuole' in tables else None)
        hosts = (pd.read_sql('SELECT * FROM vacuole_host', db) if 'vacuole_host' in tables else None)
    return tables, cells, vacuoles, hosts


def test_measure_groups_vacuoles_and_counts_them_per_cell(tmp_path):
    tables, cells, vacuoles, hosts = _measure_field(
        tmp_path, vacuole_from_parasites=True, vacuole_signal_channel=1, host_min_size=10,
        vacuole_min_size=5)
    assert {'vacuole', 'vacuole_host'} <= tables
    assert len(vacuoles) == 2 and set(vacuoles['n_parasites_cellpose']) == {1, 2}
    assert (vacuoles['tox_mean'] > 1).all() and set(vacuoles['file_name']) == {'plate1_B02_1'}
    assert set(vacuoles['well']) == {'B02'} and set(vacuoles['field']) == {1}
    assert sorted(hosts['n_vacuoles']) == [1, 1] and hosts['infected'].astype(bool).all()
    counts = cells.set_index('object_label')[['n_vacuoles', 'n_pathogens']]
    assert counts.loc[1].tolist() == [1, 2] and counts.loc[2].tolist() == [1, 1]
    read_cells, read_vacuoles = hp._titration_objects({**hp.TITRATION_DEFAULTS,
                                                       'src': str(tmp_path)})
    assert len(read_vacuoles) == 2 and 'file_name' not in read_vacuoles
    assert len(read_cells) == 2


def test_measure_without_vacuole_grouping_writes_no_vacuole_columns(tmp_path):
    tables, cells, vacuoles, _ = _measure_field(tmp_path)
    assert 'vacuole' not in tables and vacuoles is None
    assert 'n_vacuoles' not in cells and 'n_pathogens' not in cells


def test_measure_vacuole_helpers_skip_what_they_cannot_read():
    from spacr.measure import _vacuole_planes, _with_vacuole_counts

    data = np.zeros((10, 10, 5), np.uint16)
    on = {'vacuole_from_parasites': True, 'cell_mask_dim': 2, 'pathogen_mask_dim': 4}
    assert _vacuole_planes(data, {**on, 'vacuole_from_parasites': False}) is None
    assert _vacuole_planes(data, {**on, 'pathogen_mask_dim': None}) is None
    assert _vacuole_planes(np.zeros((2, 10, 10, 5)), on) is None
    planes = _vacuole_planes(data, on)
    assert planes[2] is None and planes[0].shape == (10, 10)
    assert _vacuole_planes(data, {**on, 'vacuole_signal_channel': 1})[2].shape == (10, 10)
    empty = pd.DataFrame({'area': []})
    assert _with_vacuole_counts(empty, data[..., 2], data[..., 4], {}) is empty


def test_vacuoles_per_cell_and_measure_vacuoles():
    cell = np.zeros((40, 40), int)
    cell[0:20, 0:20] = 1
    cell[20:40, 20:40] = 2
    par = np.zeros_like(cell)
    par[8:11, 8:11] = 1
    par[8:11, 12:15] = 2
    par[25:28, 25:28] = 3
    par[0:3, 30:33] = 4
    counts = hp.vacuoles_per_cell(cell, par, gap_px=4).set_index('object_label')
    assert counts.loc[1].tolist() == [1, 2] and counts.loc[2].tolist() == [1, 1]
    none = hp.vacuoles_per_cell(cell, np.zeros_like(par))
    assert none['n_vacuoles'].sum() == 0 and none['n_pathogens'].sum() == 0
    vacuoles, hosts = hp.measure_vacuoles(cell, par, None, {'host_min_size': 10,
                                                            'vacuole_min_size': 1},
                                          'plate1_C03_x')
    assert vacuoles['tox_mean'].isna().all() and set(vacuoles['field']) == {'x'}
    assert set(hosts['file_name']) == {'plate1_C03_x'} and set(hosts['well']) == {'C03'}


def test_vacuole_signal_channel_is_registered_alpha():
    from spacr.settings import ALPHA_FEATURES, expected_types, get_measure_crop_settings, tooltips

    defaults = get_measure_crop_settings({})
    assert defaults['vacuole_from_parasites'] is False and defaults['vacuole_signal_channel'] is None
    assert defaults['vacuole_gap_px'] == 4 and defaults['host_min_size'] == 300
    assert 'vacuole_signal_channel' in ALPHA_FEATURES[898]['settings']
    assert expected_types['vacuole_signal_channel'] == (int, type(None))
    assert tooltips['vacuole_signal_channel'].endswith('Default None.')


def test_vacuole_tables_skip_empty_fields_and_a_database_without_them(tmp_path):
    import sqlite3

    from spacr.measure import _save_vacuole_tables

    os.makedirs(tmp_path / 'measurements')
    db = tmp_path / 'measurements' / 'measurements.db'
    cell = np.zeros((20, 20), np.uint16)
    _save_vacuole_tables((cell, cell.copy(), None), 'plate1_B02_1', str(tmp_path), {})
    assert not db.exists()
    cell[2:18, 2:18] = 1
    _save_vacuole_tables((cell, np.zeros_like(cell), None), 'plate1_B02_1', str(tmp_path),
                         {'host_min_size': 10})
    with sqlite3.connect(db) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.execute('CREATE TABLE cell (object_label INTEGER)')
    assert 'vacuole_host' in tables and 'vacuole' not in tables
    with pytest.raises(ValueError):
        hp._titration_objects({**hp.TITRATION_DEFAULTS, 'src': str(tmp_path)})


def test_cell_frames_get_vacuole_counts_by_their_label_column():
    from spacr.measure import _with_vacuole_counts

    cell = np.zeros((40, 40), np.uint16)
    cell[0:20, 0:20] = 1
    cell[20:40, 20:40] = 2
    par = np.zeros_like(cell)
    par[8:11, 8:11] = 1
    par[8:11, 12:15] = 2
    frame = pd.DataFrame({'cell_label': [2, 1, 3], 'cell_area': [400, 400, 0],
                          'n_vacuoles': [9, 9, 9]})
    out = _with_vacuole_counts(frame, cell, par, {'vacuole_gap_px': 4})
    assert out['n_vacuoles'].tolist() == [0, 1, 0] and out['n_pathogens'].tolist() == [0, 2, 0]
    assert frame['n_vacuoles'].tolist() == [9, 9, 9]


def test_a_field_measured_again_on_resume_keeps_one_set_of_vacuole_rows(tmp_path):
    """A partial field is cleared and measured again; its vacuole rows must not double."""
    import sqlite3

    from spacr import measure
    from spacr.settings import get_measure_crop_settings

    over = dict(vacuole_from_parasites=True, vacuole_signal_channel=1, host_min_size=10,
                vacuole_min_size=5)
    _, _, first_vacuoles, first_hosts = _measure_field(tmp_path, **over)
    db = tmp_path / 'measurements' / 'measurements.db'
    with sqlite3.connect(db) as con:
        con.execute('DELETE FROM cytoplasm')
    settings = get_measure_crop_settings({})
    settings.update({'src': str(tmp_path / 'merged'), 'channels': [0, 1], 'cell_mask_dim': 2,
                     'nucleus_mask_dim': 3, 'pathogen_mask_dim': 4, 'cell_min_size': 0,
                     'nucleus_min_size': 0, 'pathogen_min_size': 0, 'cytoplasm_min_size': 0,
                     'save_png': False, 'save_arrays': False, 'plot': False, 'verbose': False,
                     'n_jobs': 1, 'resume': True, **over})
    measure.measure_crop(settings)
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT COUNT(*) FROM cytoplasm').fetchone()[0] > 0
        vacuoles = pd.read_sql('SELECT * FROM vacuole', con)
        hosts = pd.read_sql('SELECT * FROM vacuole_host', con)
    assert len(vacuoles) == len(first_vacuoles) == 2
    assert len(hosts) == len(first_hosts) == 2
    pd.testing.assert_frame_equal(vacuoles, first_vacuoles)


def test_releasing_a_field_leaves_the_other_fields_rows(tmp_path):
    import sqlite3

    from spacr.measure import _release_vacuole_rows

    db = tmp_path / 'measurements.db'
    assert _release_vacuole_rows(str(db), 'plate1_B02_1') == 0
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE vacuole (file_name TEXT, area_px INTEGER)')
        con.executemany('INSERT INTO vacuole VALUES (?, ?)',
                        [('plate1_B02_1', 5), ('plate1_B02_1', 6), ('plate1_B02_10', 7)])
    assert _release_vacuole_rows(str(db), 'plate1_B02_1') == 2
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT file_name FROM vacuole').fetchall() == [('plate1_B02_10',)]
