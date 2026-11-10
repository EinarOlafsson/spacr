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

    names = set(hp.TRANSLOCATION_DEFAULTS) & set(ALPHA_FEATURES[910]['settings'])
    assert set(names) == set(hp.TRANSLOCATION_DEFAULTS)
    assert hp.default_settings()['translocation'] is False


def _mask_run(tmp_path, wells=(('B02', 0), ('B03', 2), ('D02', 0), ('D03', 3))):
    """A Make Masks run folder without a cell stain: stack/ and nucleus and pathogen masks."""
    for sub in ('stack', 'masks/nucleus_mask_stack', 'masks/pathogen_mask_stack'):
        os.makedirs(tmp_path / sub)
    for well, k in wells:
        stack = _field(k)
        np.save(tmp_path / 'stack' / f'plate_{well}_1.npy', stack[..., :3].astype(np.uint16))
        np.save(tmp_path / 'masks' / 'nucleus_mask_stack' / f'plate_{well}_1.npy',
                stack[..., 4].astype(np.uint16))
        np.save(tmp_path / 'masks' / 'pathogen_mask_stack' / f'plate_{well}_1.npy',
                stack[..., 5].astype(np.uint16))
    return tmp_path


def test_make_masks_grows_cells_from_nuclei_and_the_merge_records_it(tmp_path, capsys):
    import json

    from spacr.core import _cells_from_nuclei
    from spacr.io import _load_and_concatenate_arrays

    src = _mask_run(tmp_path)
    settings = {'cell_channel': None, 'nucleus_channel': 0, 'cell_from_nucleus_expand_px': 20,
                'cell_from_nucleus_min_nucleus_size': 50, 'timelapse': False}
    (src / 'masks' / 'nucleus_mask_stack' / 'notes.txt').write_text('not a mask')
    folders = _cells_from_nuclei(str(src / 'masks'), settings)
    assert 'nucleus territories' in capsys.readouterr().out
    assert folders == {'nucleus': str(src / 'masks' / hp.TERRITORY_NUCLEUS_FOLDER)}
    _load_and_concatenate_arrays(str(src), None, None, 0, 1, None, mask_folders=folders,
                                 cell_source='nucleus_territory')
    layout = json.loads((src / 'merged' / '.spacr_plane_layout.json').read_text())
    assert layout['cell_source'] == 'nucleus_territory'
    assert layout['mask_dims'] == {'cell': 3, 'nucleus': 4, 'pathogen': 5}
    merged = np.load(src / 'merged' / 'plate_B03_1.npy')
    territory, nuclei = hp.nucleus_territories(_field(2)[..., 4], 20, 50)
    assert merged.shape[-1] == 6 and merged.dtype == np.uint16
    assert np.array_equal(merged[..., 3], territory) and np.array_equal(merged[..., 4], nuclei)
    result = hp.analyze_host_pathogen({'src': str(src), 'translocation': True,
                                       'cell_from_nucleus_expand_px': 20,
                                       'translocation_groups': 'B:A, D:B',
                                       'translocation_min_nucleus_size': 100,
                                       'translocation_shuffle_n': 5})
    cells = result['cells']
    assert (cells['cell_source'] == 'nucleus_territory').all()
    assert sorted(cells['n_vacuoles']) == [0, 0, 2, 3]
    assert os.path.isfile(result['figure'])


def test_cells_from_nuclei_only_runs_without_a_cell_stain(tmp_path, capsys):
    from spacr.core import _cells_from_nuclei

    on = {'cell_channel': None, 'nucleus_channel': 0, 'cell_from_nucleus_expand_px': 20,
          'timelapse': False}
    assert _cells_from_nuclei(str(tmp_path), {**on, 'cell_channel': 1}) == {}
    assert _cells_from_nuclei(str(tmp_path), {**on, 'nucleus_channel': None}) == {}
    assert _cells_from_nuclei(str(tmp_path), {**on, 'cell_from_nucleus_expand_px': 0}) == {}
    assert _cells_from_nuclei(str(tmp_path), {**on, 'timelapse': True}) == {}
    assert 'not applied to time-lapse' in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        _cells_from_nuclei(str(tmp_path), on)
    with pytest.raises(ValueError):
        hp.write_nucleus_territories(str(tmp_path), 0)


def test_the_merged_layout_is_unchanged_without_territories(tmp_path):
    import json

    from spacr.io import _load_and_concatenate_arrays

    src = _mask_run(tmp_path, wells=(('B02', 1),))
    _load_and_concatenate_arrays(str(src), None, None, 0, 1, None)
    layout = json.loads((src / 'merged' / '.spacr_plane_layout.json').read_text())
    assert 'cell_source' not in layout and layout['mask_plane_order'] == ['nucleus', 'pathogen']


def test_the_translocation_figure_has_raw_and_corrected_rows_per_group():
    cells = _cells(True)
    reference = pd.DataFrame([dict(effector=g, k=k, hrman_total=1 + 0.125 * k,
                                   hrman_ratio=1 + 0.125 * k) for g in 'AB' for k in range(4)])
    out = hp.translocation_summary(cells, SETTINGS, reference)
    figure = hp.translocation_figure(out, SETTINGS)
    assert len(figure.axes) == 4
    dashed = [line for ax in figure.axes for line in ax.get_lines() if line.get_linestyle() == '--']
    assert len(dashed) == 4
    assert figure.axes[0].get_title() == 'A' and 'raw' in figure.axes[0].get_ylabel()
    assert len(figure.axes[0].patches) == 4 and figure.axes[0].collections
    sparse = dict(out, per_well_summary=out['per_well_summary'].iloc[0:0],
                  per_well=out['per_well'].iloc[0:0], shuffle_null=out['shuffle_null'].iloc[0:0])
    bare = hp.translocation_figure(sparse, {**SETTINGS, 'translocation_kmax': 10})
    assert len(bare.axes[0].patches) == 4 and not bare.axes[0].collections


def test_an_empty_translocation_summary_draws_empty_panels():
    out = hp.translocation_summary(_cells(True), SETTINGS)
    empty = {key: frame.iloc[0:0] for key, frame in out.items() if hasattr(frame, 'iloc')}
    figure = hp.translocation_figure(empty, SETTINGS)
    assert len(figure.axes) == 2 and figure.axes[0].get_legend() is None


def test_six_plane_arrays_with_a_cell_stain_have_no_cell_source(tmp_path):
    merged = tmp_path / 'merged'
    os.makedirs(merged)
    for well, k in (('B02', 0), ('B03', 2)):
        np.save(merged / f'plate_{well}_1.npy', _territories(_field(k)))
    result = hp.nuclear_translocation({'src': str(tmp_path), 'translocation_groups': 'B:A',
                                       'translocation_min_nucleus_size': 100,
                                       'translocation_shuffle_n': 2})
    assert 'cell_source' not in result['cells'] and len(result['cells']) == 2
