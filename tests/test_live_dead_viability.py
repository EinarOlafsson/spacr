"""Live/dead viability and a cytotoxicity index per well, from Measure's tables.

Cells of known state are drawn into fields: every nucleus in a Hoechst
channel, dead cells with a propidium-iodide nucleus that is also pyknotic
(smaller and brighter), live cells filled with calcein, and some live cells
holding a parasite. A control plate of untreated and cytotoxic wells, and a
dose series of a selective and a cytotoxic compound, go through Measure
itself, so the calls are made from the columns Measure really writes and
scored against the drawn truth. The larger validation behind the numbers in
features/future/540_live_dead_viability_and_cytotoxicity.txt uses the same
generator.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr.measure import (
    _VIABILITY_QC_TABLE,
    _VIABILITY_TABLE,
    _VIABILITY_WELL_TABLE,
    _classify_viability,
    _condensation_score,
    _cytotoxicity_index,
    _read_plate_map,
    _stain_cut,
    _two_population_fit,
    _viability_by_well,
    _viability_manual,
    _viability_states,
    _well_roles,
)

HILL = 1.5


def _disk(yy, xx, cy, cx, radius):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


def viability_field(rng, n_cells, dead_fraction, infection=0.0, size=256,
                    pi_dim=False, gain=1.0, pi_sigma=0.4, calcein_sigma=0.35,
                    dim_dead=0.0, pyknotic=1.0):
    """One field: Hoechst, PI and calcein planes, nucleus/cell/parasite labels.

    ``gain`` scales every stain (a brighter or dimmer plate), ``dim_dead``
    is the share of dead cells whose PI is a tenth as bright and which keep
    a fifth of the calcein of a live cell (dying, not yet emptied), and
    ``pyknotic`` the share of dead cells whose nucleus is condensed.

    :returns: ``(stack, truth)``; ``truth`` maps nucleus label to
        ``(state, infected)``.
    """
    hoechst = np.zeros((size, size))
    pi = np.zeros((size, size))
    calcein = np.zeros((size, size))
    nuclei = np.zeros((size, size), np.int32)
    cells = np.zeros((size, size), np.int32)
    parasites = np.zeros((size, size), np.int32)
    yy, xx = np.mgrid[0:size, 0:size]
    truth = {}
    radius = 9
    for _ in range(n_cells):
        dead = rng.random() < dead_fraction
        for _attempt in range(60):
            cy, cx = rng.uniform(radius + 3, size - radius - 3, 2)
            cell = _disk(yy, xx, cy, cx, radius)
            if not cells[_disk(yy, xx, cy, cx, radius + 2)].any():
                break
        else:
            continue
        label = int(cells.max()) + 1
        condensed = dead and rng.random() < pyknotic
        dim = dead and rng.random() < dim_dead
        r_nuc = 3.8 if condensed else 5.5
        nucleus = _disk(yy, xx, cy, cx, r_nuc)
        cells[cell] = label
        nuclei[nucleus] = label
        dna = gain * rng.normal(1.0, 0.08) * 300.0 * np.pi * 5.5 ** 2
        hoechst[nucleus] += dna / nucleus.sum()
        infected = False
        if dead:
            level = gain * (400.0 if not pi_dim else 60.0) * (0.1 if dim else 1)
            pi[nucleus] += rng.lognormal(np.log(level), pi_sigma)
            if dim:
                calcein[cell] += gain * rng.lognormal(np.log(60.0),
                                                      calcein_sigma)
        else:
            calcein[cell] += gain * rng.lognormal(np.log(300.0),
                                                  calcein_sigma)
            if rng.random() < infection:
                angle = rng.uniform(0, 2 * np.pi)
                py, px = cy + 7 * np.sin(angle), cx + 7 * np.cos(angle)
                blob = _disk(yy, xx, py, px, 1.8) & cell & ~nucleus
                if blob.sum() >= 3:
                    parasites[blob] = int(parasites.max()) + 1
                    infected = True
        truth[label] = ('dead' if dead else 'live', infected)
    planes = []
    for plane, background in ((hoechst, 100.0), (pi, 80.0), (calcein, 90.0)):
        from scipy import ndimage as ndi

        blurred = ndi.gaussian_filter(plane, 0.8) + background
        planes.append(rng.poisson(blurred).astype(np.uint16))
    stack = np.stack(planes + [nuclei.astype(np.uint16),
                               cells.astype(np.uint16),
                               parasites.astype(np.uint16)], axis=-1)
    return stack, truth


def killed(dose, cc50):
    """Fraction of host cells a dose kills (Hill slope :data:`HILL`)."""
    if dose <= 0:
        return 0.0
    return dose ** HILL / (dose ** HILL + cc50 ** HILL)


def well_design(dose, cc50, ec50, *, base_cells, base_dead=0.04,
                base_infection=0.5, detach=0.5):
    """Cells, dead fraction and infection of one well at one dose.

    A killed cell detaches with probability ``detach`` and otherwise stays
    as a dead, stained cell.
    """
    k = killed(dose, cc50)
    live = (1 - base_dead) * (1 - k)
    dead = base_dead * (1 - k) + (1 - detach) * k
    n = max(int(round(base_cells * (live + dead))), 1)
    infection = base_infection * (1 - (killed(dose, ec50) if ec50 else 0.0))
    return n, dead / (live + dead), infection


def write_plate(root, wells, *, fields=1, size=256, seed=0, plate='plate1',
                base_cells=40, seed_cv=0.0, **look):
    """Write merged fields for ``wells``: ``{well: (dose, cc50, ec50)}``.

    ``seed_cv`` spreads the cells seeded per well (log-normal), as uneven
    seeding does on a real plate.

    :returns: ``(merged folder, truth by field name)``.
    """
    merged = Path(root) / 'merged'
    merged.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truths = {}
    for well, (dose, cc50, ec50) in wells.items():
        seeded = base_cells * (rng.lognormal(0.0, seed_cv) if seed_cv else 1)
        n, dead, infection = well_design(dose, cc50, ec50,
                                         base_cells=seeded)
        for number in range(1, fields + 1):
            stack, truth = viability_field(rng, n, dead, infection,
                                           size=size, **look)
            name = f'{plate}_{well}_{number}'
            np.save(merged / f'{name}.npy', stack)
            truths[name] = truth
    return merged, truths


def measure_settings(merged, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        'src': str(merged), 'channels': [0, 1, 2], 'nucleus_channel': 0,
        'cell_channel': 2, 'pathogen_channel': None,
        'nucleus_mask_dim': 3, 'cell_mask_dim': 4, 'pathogen_mask_dim': 5,
        'cell_min_size': 0, 'nucleus_min_size': 0, 'pathogen_min_size': 0,
        'cytoplasm_min_size': 0, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'n_jobs': 2,
        'radial_dist': False, 'spatial_measurements': False,
        'object_distances': False, 'object_distance_maxima': False,
        'object_distance_intensity': False, 'calculate_correlation': False,
        'homogeneity': False, 'illumination_qc': False,
        'viability': True, 'viability_dead_channel': 1,
        'viability_live_channel': 2,
    })
    settings.update(over)
    return settings


def truth_for(table, truths):
    return [truths[f].get(int(o), (None, None))[0] for f, o in
            zip(table['file_name'], table['object_label'])]


def _signal_table(n_live=380, n_dead=20, plates=('p1',), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for plate in plates:
        for state, count in (('live', n_live), ('dead', n_dead)):
            for _ in range(count):
                dead = state == 'dead'
                rows.append({
                    'plateID': plate, 'rowID': 'r1',
                    'columnID': 'c1' if rng.random() < 0.5 else 'c2',
                    'fieldID': 'f1', 'object_label': len(rows) + 1,
                    'state': state,
                    'nucleus_area': rng.normal(45 if dead else 95, 6),
                    'nucleus_channel_0_mean_intensity':
                        rng.normal(620 if dead else 300, 30) + 100,
                    'nucleus_channel_0_outside_percentile_50': 100.0,
                    'nucleus_channel_1_mean_intensity':
                        (rng.lognormal(np.log(400), 0.4) if dead
                         else rng.normal(0, 3)) + 80,
                    'nucleus_channel_1_outside_percentile_50': 80.0,
                    'nucleus_channel_2_mean_intensity':
                        (rng.normal(3, 3) if dead
                         else rng.lognormal(np.log(300), 0.35)) + 90,
                    'nucleus_channel_2_outside_percentile_50': 90.0,
                })
    frame = pd.DataFrame(rows)
    frame['prcf'] = (frame['plateID'] + '_' + frame['rowID'] + '_'
                     + frame['columnID'] + '_' + frame['fieldID'])
    return frame


def test_two_populations_are_cut_where_they_cross():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0, 1, 950), rng.normal(8, 1, 50)])
    fit = _two_population_fit(values)
    assert fit['bimodal']
    assert 2.5 < fit['cut'] < 5.5
    assert fit['weights'][1] == pytest.approx(0.05, abs=0.01)
    single = _two_population_fit(rng.normal(0, 1, 1000))
    assert not single['bimodal']
    skewed = _two_population_fit(rng.lognormal(0, 0.6, 1000))
    assert not skewed['bimodal']


def test_a_plate_with_no_dead_cells_is_cut_above_its_one_population():
    rng = np.random.default_rng(2)
    unstained = rng.normal(0, 3, 500)
    cut = _stain_cut(unstained, single_is_positive=False)
    assert cut.source == 'single'
    assert cut.threshold > np.percentile(unstained, 99.9)
    live = rng.lognormal(np.log(300), 0.3, 500)
    cut = _stain_cut(live, single_is_positive=True)
    assert cut.source == 'single' and cut.positive_fraction > 0.99
    mixed = np.concatenate([unstained, rng.lognormal(np.log(400), 0.4, 25)])
    cut = _stain_cut(mixed, single_is_positive=False)
    assert cut.source == 'mixture'
    assert cut.positive_fraction == pytest.approx(25 / 525, abs=0.01)
    manual = _stain_cut(mixed, single_is_positive=False, manual=1000.0)
    assert manual.source == 'manual' and manual.threshold == 1000.0


def test_manual_thresholds_parse_and_bad_ones_are_refused():
    assert _viability_manual(None) == (None, None)
    assert _viability_manual([150, None]) == (150.0, None)
    assert _viability_manual(['auto', '20']) == (None, 20.0)
    assert _viability_manual([3]) == (3.0, None)
    for bad in ([1, 2, 3], ['x'], 5):
        with pytest.raises(ValueError):
            _viability_manual(bad)


def test_states_from_one_stain_or_both():
    dead = np.array([True, False, False, True])
    live = np.array([False, True, False, True])
    assert list(_viability_states(dead, live, dead=True, live=True)) == [
        'dead', 'live', 'unstained', 'dead']
    assert list(_viability_states(dead, None, dead=True, live=False)) == [
        'dead', 'live', 'live', 'dead']
    assert list(_viability_states(None, live, dead=False, live=True)) == [
        'dead', 'live', 'dead', 'live']


def test_pyknotic_nuclei_have_a_high_condensation_score():
    table = _signal_table()
    score = _condensation_score(table, 0)
    by_state = score.groupby(table['state']).median()
    assert by_state['live'] == pytest.approx(1.0, abs=0.1)
    assert by_state['dead'] > 3.0


def test_roles_come_from_the_well_notation():
    frame = pd.DataFrame({'rowID': ['r1', 'r2', 'r1', 'r3'],
                          'columnID': ['c1', 'c1', 'c12', 'c5']})
    roles = _well_roles(frame, {'viability_negative_wells': ['c1'],
                                'viability_positive_wells': 'A12'})
    assert list(roles) == ['negative', 'negative', 'positive', 'sample']
    with pytest.raises(ValueError, match='both'):
        _well_roles(frame, {'viability_negative_wells': 'c1',
                            'viability_positive_wells': 'A01'})


def _well_frame():
    rows = []
    for column, role, n_live, n_objects in (
            ('c1', 'negative', 96, 100), ('c2', 'negative', 104, 108),
            ('c3', 'positive', 4, 50), ('c4', 'positive', 6, 52),
            ('c5', 'sample', 50, 70)):
        rows.append({'plateID': 'p', 'rowID': 'r1', 'columnID': column,
                     'role': role, 'n_live': n_live, 'n_objects': n_objects,
                     'viability': n_live / n_objects, 'plate_key': 'p'})
    return pd.DataFrame(rows)


def test_the_cytotoxicity_index_is_scaled_to_the_plates_controls():
    wells = _cytotoxicity_index(_well_frame())
    assert (wells['cytotoxicity_basis'] == 'controls').all()
    assert wells.loc[wells['role'] == 'negative',
                     'cytotoxicity_index'].mean() == pytest.approx(0, abs=1e-9)
    assert wells.loc[wells['role'] == 'positive',
                     'cytotoxicity_index'].mean() == pytest.approx(100)
    sample = wells[wells['role'] == 'sample'].iloc[0]
    assert sample['live_cell_index'] == pytest.approx(0.5)
    assert sample['cytotoxicity_index'] == pytest.approx(
        100 * (1 - 0.5) / (1 - 0.05))

    no_positive = _well_frame()
    no_positive['role'] = no_positive['role'].replace('positive', 'sample')
    wells = _cytotoxicity_index(no_positive)
    assert (wells['cytotoxicity_basis'] == 'negative control').all()
    assert wells.iloc[-1]['cytotoxicity_index'] == pytest.approx(50)

    none = _well_frame().assign(role='sample')
    wells = _cytotoxicity_index(none)
    assert (wells['cytotoxicity_basis'] == 'dead fraction').all()
    assert wells.iloc[-1]['cytotoxicity_index'] == pytest.approx(
        100 * (1 - 50 / 70))


def test_a_well_whose_cells_were_all_lost_is_reported_empty():
    table = pd.DataFrame({
        'plateID': ['p'] * 4, 'rowID': ['r1'] * 4,
        'columnID': ['c1', 'c1', 'c2', 'c2'],
        'viability_state': ['live', 'dead', 'live', 'live'],
        'infected': [1.0, np.nan, 0.0, 1.0],
    })
    fields = pd.DataFrame({'plateID': ['p', 'p', 'p'],
                           'rowID': ['r1'] * 3,
                           'columnID': ['c1', 'c2', 'c3'],
                           'timeID': [None] * 3})
    wells = _viability_by_well(table, {'viability_negative_wells': 'c2'},
                               fields)
    lost = wells[wells['columnID'] == 'c3'].iloc[0]
    assert lost['n_objects'] == 0 and lost['live_cell_index'] == 0
    assert lost['cytotoxicity_index'] == pytest.approx(100)
    first = wells[wells['columnID'] == 'c1'].iloc[0]
    assert first['viability'] == pytest.approx(0.5)
    assert first['infection_live'] == pytest.approx(1.0)
    assert wells[wells['columnID'] == 'c2'].iloc[0][
        'infection_live'] == pytest.approx(0.5)


def test_a_plate_map_names_compound_and_concentration(tmp_path):
    wells = _cytotoxicity_index(_well_frame())
    path = tmp_path / 'map.csv'
    pd.DataFrame({'well': ['A05', 'A01'], 'Compound': ['X', 'DMSO'],
                  'Concentration': [3.0, 0.0]}).to_csv(path, index=False)
    joined = _read_plate_map(str(path), wells)
    assert joined.loc[joined['columnID'] == 'c5', 'compound'].iloc[0] == 'X'
    assert joined.loc[joined['columnID'] == 'c5',
                      'concentration'].iloc[0] == 3.0
    assert joined['compound'].isna().sum() == 3
    bad = tmp_path / 'bad.csv'
    pd.DataFrame({'well': ['A01'], 'x': [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match='compound'):
        _read_plate_map(str(bad), wells)


def test_a_database_without_objects_is_refused(tmp_path):
    db = tmp_path / 'measurements.db'
    with sqlite3.connect(db) as conn:
        pd.DataFrame({'x': [1]}).to_sql('png_list', conn, index=False)
    with pytest.raises(ValueError, match='no nucleus or cell'):
        _classify_viability(str(db), {'viability': True})
    assert measure._run_viability_step(str(db), {'viability': True}) is None


def _control_plate(tmp_path, **kw):
    design = {}
    for row in 'AB':
        for col in (1, 2, 3):
            design[f'{row}{col:02d}'] = (0.0, 1.0, None)
        for col in (10, 11, 12):
            design[f'{row}{col:02d}'] = (100.0, 1.0, None)
    return write_plate(tmp_path, design, **kw)


@pytest.mark.integration
def test_measure_separates_a_control_plate_with_both_stains(tmp_path):
    merged, truths = _control_plate(tmp_path, seed=4)
    settings = measure_settings(
        merged, viability_negative_wells=['c1', 'c2', 'c3'],
        viability_positive_wells=['c10', 'c11', 'c12'])
    measure.measure_crop(settings)

    db = tmp_path / 'measurements' / 'measurements.db'
    _classify_viability(str(db), settings, plot=True)
    with sqlite3.connect(db) as conn:
        table = pd.read_sql_query(f'SELECT * FROM {_VIABILITY_TABLE}', conn)
        wells = pd.read_sql_query(f'SELECT * FROM {_VIABILITY_WELL_TABLE}',
                                  conn)
        qc = pd.read_sql_query(f'SELECT * FROM {_VIABILITY_QC_TABLE}', conn)
    truth = truth_for(table, truths)
    called = np.where(table['viability_state'] == 'live', 'live', 'dead')
    accuracy = float(np.mean(called == np.asarray(truth)))
    assert accuracy >= 0.97, accuracy
    assert table['prcfo'].str.match(r'^plate1_r\d+_c\d+_f\d+_o\d+$').all()
    assert len(wells) == 12
    assert set(wells['role']) == {'negative', 'positive'}
    assert (wells['cytotoxicity_basis'] == 'controls').all()
    row = qc.iloc[0]
    assert row['dead_threshold_source'] == 'mixture'
    assert row['live_threshold_source'] == 'mixture'
    assert row['zprime_viability'] > 0.5
    assert row['n_negative'] == 6 and row['n_positive'] == 6
    figures = {p.stem for p in (tmp_path / 'results' / 'viability').glob('*')}
    assert {'viability_thresholds_plate1', 'viability_controls',
            'viability_plate_viability'} <= figures


@pytest.mark.integration
def test_one_stain_or_none_and_a_manual_cut(tmp_path):
    merged, truths = _control_plate(tmp_path, seed=5)
    measure.measure_crop(measure_settings(merged, viability=False))
    db = str(tmp_path / 'measurements' / 'measurements.db')
    base = measure_settings(merged)
    for over in ({'viability_live_channel': None},
                 {'viability_dead_channel': None},
                 {'viability_dead_channel': None,
                  'viability_live_channel': None}):
        table, report = _classify_viability(db, {**base, **over})
        truth = np.asarray(truth_for(table, truths))
        called = np.where(table['viability_state'] == 'live', 'live', 'dead')
        assert np.mean(called == truth) >= 0.93, (over, report['method'])
    assert report['method'] == 'morphology'
    table, report = _classify_viability(
        db, {**base, 'viability_thresholds': [1e9, None]})
    assert set(table['viability_state']) <= {'live', 'unstained'}
    assert (report['qc']['dead_threshold_source'] == 'manual').all()


@pytest.mark.integration
def test_dose_response_separates_a_selective_compound_from_a_toxic_one(
        tmp_path):
    design, rows = {}, []
    doses = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    for row in 'AB':
        design[f'{row}01'] = (0.0, 1.0, None)
        design[f'{row}12'] = (300.0, 1.0, None)
    for row, (compound, cc50, ec50) in zip(
            'CDEF', (('selective', 60.0, 0.3), ('selective', 60.0, 0.3),
                     ('toxic', 1.0, 1.0), ('toxic', 1.0, 1.0))):
        for col, dose in enumerate(doses, start=2):
            well = f'{row}{col:02d}'
            design[well] = (dose, cc50, ec50)
            rows.append({'well': well, 'compound': compound,
                         'concentration': dose})
    plate_map = tmp_path / 'plate_map.csv'
    pd.DataFrame(rows).to_csv(plate_map, index=False)
    merged, _truths = write_plate(tmp_path, design, seed=7, base_cells=80,
                                  size=320)
    measure.measure_crop(measure_settings(merged, viability=False))
    db = str(tmp_path / 'measurements' / 'measurements.db')
    settings = measure_settings(
        merged, viability_negative_wells='c1', viability_positive_wells='c12',
        viability_plate_map=str(plate_map), plot=True)
    _table, report = _classify_viability(db, settings)

    curves = report['dose_response'].set_index(['readout', 'group'])
    assert {'viability', 'cytotoxicity_index', 'infection'} <= set(
        curves.index.get_level_values(0))
    toxic_cc50 = curves.loc[('cytotoxicity_index', 'toxic'), 'ec50']
    assert 0.3 < toxic_cc50 < 3.0
    selective_ec50 = curves.loc[('infection', 'selective'), 'ec50']
    assert 0.05 < selective_ec50 < 1.5
    assert curves.loc[('cytotoxicity_index', 'selective'),
                      'status'] == 'unbounded'
    selectivity = report['selectivity'].set_index('compound')
    toxic = selectivity.loc['toxic']
    assert toxic['status'] == 'fitted'
    assert toxic['selectivity_index'] < 5
    assert selectivity.loc['selective', 'status'] == 'unbounded'
    assert list((tmp_path / 'results' / 'viability').glob(
        'viability_dose_response.*'))


def test_the_settings_are_measure_defaults_and_alpha_registered():
    from spacr.settings import (ALPHA_FEATURES, expected_types,
                                get_measure_crop_settings, tooltips)

    defaults = get_measure_crop_settings({})
    keys = ALPHA_FEATURES[540]['settings']
    assert {k for k in defaults if k.startswith('viability')} == set(keys)
    assert defaults['viability'] is False
    assert defaults['viability_dead_channel'] is None
    for key in keys:
        assert key in expected_types and key in tooltips, key
        assert len(tooltips[key]) <= 600, key
        assert re.search(r'Default [^.]+(\.\d+)?\.$', tooltips[key]), key


def test_the_real_plate1_nuclei_are_mostly_called_live(tmp_path):
    """The example plate's Hoechst nuclei, called from morphology alone."""
    source = Path(os.environ.get('SPACR_PLATE1_DB') or Path.home()
                  / '.cache/spacr/example_data/plate1/measurements'
                  / 'measurements.db')
    if not source.is_file():
        pytest.skip('plate1 example data not downloaded')
    copy = tmp_path / 'measurements' / 'measurements.db'
    copy.parent.mkdir()
    with sqlite3.connect(f'file:{source}?mode=ro', uri=True) as conn:
        nuclei = pd.read_sql_query('SELECT * FROM nucleus', conn)
    with sqlite3.connect(copy) as conn:
        nuclei.to_sql('nucleus', conn, index=False)
    table, report = _classify_viability(
        str(copy), {'channels': [0, 1, 2, 3], 'nucleus_channel': 0})
    assert report['method'] == 'morphology'
    share = (table['viability_state'] == 'live').mean()
    assert 0.8 < share < 1.0
