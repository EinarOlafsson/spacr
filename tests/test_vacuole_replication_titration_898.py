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
