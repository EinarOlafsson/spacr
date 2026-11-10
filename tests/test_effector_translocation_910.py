"""Host-nucleus effector translocation against parasite load (item 910, alpha).

Cells from nuclei (territories) when there is no cell stain, vacuoles per
cell (item 898's grouping), a raw and a corrected nuclear effector readout
and their controls. On the HRMAn Protein_translocation pilot (3 fields) the
territories, the per-cell table and every summary table equal the reference
scripts' output; the synthetic movie here pins the rules.
"""
import os

import numpy as np
import pandas as pd
import pytest

from spacr import host_pathogen as hp


def _field(k, planted=True, overlap=False, size=60):
    """One nucleus with ``k`` parasites around it; effector rises with k when planted."""
    nuc = np.zeros((size, size), np.int32)
    nuc[20:40, 20:40] = 1
    par = np.zeros_like(nuc)
    spots = [(2, 2), (2, 50), (50, 2), (50, 50), (2, 26), (50, 26), (26, 2), (26, 50)]
    for i in range(k):
        y, x = spots[i]
        par[y:y + 6, x:x + 6] = i + 1
    if overlap:
        par[28:34, 28:34] = k + 1
    ha = np.full(nuc.shape, 100.0)
    ha[nuc > 0] += (50.0 * k if planted else 0) + 500
    ha[par > 0] = 5000.0
    dna = np.where(nuc > 0, 2000.0, 50.0)
    return np.stack([dna, np.zeros_like(ha), ha, np.zeros_like(ha), nuc, par], axis=-1)


def test_territories_grow_without_overlap_and_drop_small_nuclei():
    nuc = np.zeros((30, 30), np.int32)
    nuc[5:10, 5:10] = 1
    nuc[5:10, 20:25] = 2
    nuc[25:27, 25:27] = 3
    territory, kept = hp.nucleus_territories(nuc, expand_px=4, min_nucleus_size=10)
    assert 3 not in kept and 3 not in territory
    assert (territory[kept > 0] == kept[kept > 0]).all()
    assert (territory == 1).sum() > 25 and territory[7, 12] == 1 and territory[7, 17] == 2
    assert territory[7, 15] == 0
    assert territory[29, 0] == 0


def test_raw_readout_is_inflated_by_an_overlapping_vacuole_and_the_corrected_one_is_not():
    clean = hp.translocation_cells(_territories(_field(2)), {}, 'B02', 1).iloc[0]
    over = hp.translocation_cells(_territories(_field(2, overlap=True)), {}, 'B02', 1).iloc[0]
    assert over['ha_total_nucleus'] > clean['ha_total_nucleus'] * 1.1
    assert over['ha_mean_nucleus_excl'] == pytest.approx(clean['ha_mean_nucleus_excl'])
    assert over['frac_nucleus_excluded'] > 0 and clean['frac_nucleus_excluded'] == 0
    assert clean['n_vacuoles'] == 2 and clean['n_parasites'] == 2 and clean['bg_ha'] == 100
    none = hp.translocation_cells(_territories(_field(0)), {'translocation_background': 'none'})
    assert none.iloc[0]['bg_ha'] == 0 and none.iloc[0]['n_vacuoles'] == 0
    few = hp.translocation_cells(_territories(_field(1)), {'translocation_background_min_px': 10 ** 6})
    assert few.iloc[0]['bg_ha'] == np.percentile(_field(1)[..., 2], 5)


def _territories(stack):
    """The six-plane layout with territory cells."""
    territory, nuclei = hp.nucleus_territories(stack[..., 4], 20, 50)
    stack = stack.copy()
    stack[..., 3], stack[..., 4] = territory, nuclei
    return stack


def _cells(planted, fields=4):
    """Cells of three wells per group with k = 0..3 vacuoles."""
    rows = []
    for row, group in (('B', 'A'), ('D', 'B')):
        for w in range(3):
            well = f'{row}0{w + 2}'
            for f in range(fields):
                for k in range(4):
                    for rep in range(2):
                        t = hp.translocation_cells(_territories(_field(k, planted)), {}, well, f)
                        rows.append(t.assign(cell=len(rows) + 1))
    return pd.concat(rows, ignore_index=True)


SETTINGS = {'translocation_groups': 'B:A, D:B', 'translocation_shuffle_n': 20,
            'translocation_min_nucleus_size': 100, 'translocation_kmax': 3}


def test_the_corrected_readout_recovers_the_planted_slope_and_the_controls_stay_flat():
    out = hp.translocation_summary(_cells(True), SETTINGS)
    curves = out['curves'].set_index(['readout', 'effector', 'k'])
    corrected = curves.loc[('corrected', 'A'), 'ratio']
    expected = (500 + 50 * np.arange(4)) / 500
    assert np.allclose(corrected.to_numpy(), expected)
    assert np.allclose(curves.loc[('dna_control', 'A'), 'ratio'], 1)
    null = out['shuffle_null'].set_index(['readout', 'effector', 'k'])
    assert null.loc[('corrected', 'A', 3), 'null_ratio_mean'] < corrected.loc[3]
    assert set(out['per_well']['readout']) == {'hrman_equiv', 'corrected', 'dna_control'}
    assert (out['per_well_summary']['n_wells'] == 3).all()
    assert out['counts'].loc['A', 'cells'] == 96 and out['agreement'].empty
    flat = hp.translocation_summary(_cells(False), SETTINGS)
    c = flat['curves'].set_index(['readout', 'effector', 'k'])
    assert np.allclose(c.loc[('corrected', 'A'), 'ratio'], 1)
    lo = flat['shuffle_null'].set_index(['readout', 'effector', 'k']).loc[('corrected', 'A', 2)]
    assert lo['null_ratio_p2_5'] <= 1 <= lo['null_ratio_p97_5']
    thresholds = flat['null_thresholds'].set_index('effector')
    assert len(thresholds) == 2 and (flat['pct_positive']['pct_positive'] <= 100).all()


def test_the_reference_curve_gives_an_agreement_table():
    cells = _cells(True)
    reference = pd.DataFrame([dict(effector=g, k=k, hrman_total=1 + 0.125 * k,
                                   hrman_ratio=1 + 0.125 * k) for g in 'AB' for k in range(4)])
    out = hp.translocation_summary(cells, SETTINGS, reference)
    agree = out['agreement'].set_index(['readout', 'effector'])
    assert agree.loc[('corrected', 'A'), 'pearson_r'] == pytest.approx(1.0)
    assert agree.loc[('corrected', 'A'), 'ratio_k10'] == pytest.approx(1.3)
    assert agree.loc[('corrected', 'A'), 'hrman_k10'] == pytest.approx(1.375)
    groups = hp._translocation_groups({'B02': 'A'})
    assert groups == {'B02': 'A'}


def test_a_translocation_settings_run_from_five_plane_arrays(tmp_path):
    from spacr.tabular import write_table

    merged = tmp_path / 'merged'
    os.makedirs(merged)
    for well, k in (('B02', 0), ('B03', 2), ('D02', 0), ('D03', 3)):
        stack = _field(k)
        np.save(merged / f'plate_{well}_1_None.npy', np.delete(stack, 3, axis=-1))
    plate = tmp_path / 'map.csv'
    write_table(pd.DataFrame({'well': ['B02', 'B03', 'D02', 'D03'], 'group': ['A', 'A', 'B', 'B']}),
                str(plate), canonicalise=False)
    result = hp.analyze_host_pathogen({'src': str(tmp_path), 'translocation': True,
                                       'cell_from_nucleus_expand_px': 20,
                                       'cell_from_nucleus_min_nucleus_size': 50,
                                       'translocation_groups': str(plate),
                                       'translocation_min_nucleus_size': 100,
                                       'translocation_shuffle_n': 5})
    cells = pd.read_csv(os.path.join(result['folder'], 'cells_translocation.csv'))
    assert (cells['cell_source'] == 'nucleus_territory').all() and len(cells) == 4
    assert sorted(cells['n_vacuoles']) == [0, 0, 2, 3]
    for name in hp._TRANSLOCATION_FILES.values():
        assert os.path.isfile(os.path.join(result['folder'], name))


def test_translocation_settings_are_registered_alpha():
    from spacr.settings import ALPHA_FEATURES

    names = set(hp.TRANSLOCATION_DEFAULTS) & set(ALPHA_FEATURES[772]['settings'])
    assert set(names) == set(hp.TRANSLOCATION_DEFAULTS)
    assert hp.default_settings()['translocation'] is False
