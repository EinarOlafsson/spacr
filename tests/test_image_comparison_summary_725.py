"""Item 725 round 3: quantitative comparisons of chosen wells and conditions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.sp_stats import (
    _compare_image_groups,
    _image_comparison_table,
    _image_effect_figure,
    _image_well_groups,
    _rank_image_differences,
)


def _frame(seed=0, fields=6):
    rng = np.random.default_rng(seed)
    rows = []
    layout = {'A01': ('HeLa', 0.0), 'A02': ('HeLa', 0.0), 'B01': ('HFF', 8.0), 'B02': ('HFF', 8.0)}
    for well, (cell, shift) in layout.items():
        for field in range(fields):
            rows.append(dict(plate='plate1', well=well, field=str(field + 1), cell_type=cell, status='ok',
                             channel_0_intensity_mean=100 + shift + rng.normal(0, 1),
                             channel_0_focus_brenner=20 + rng.normal(0, 1),
                             channel_0_radial_bin_0=5.0))
    return pd.DataFrame(rows)


def test_well_groups_compare_chosen_wells_directly():
    frame = _frame()
    definitions = _image_well_groups([('plate1', 'A01'), ('plate1', 'B01')])
    assert definitions == [{'plate': 'plate1', 'well': 'A01'}, {'plate': 'plate1', 'well': 'B01'}]
    result = _compare_image_groups(frame, ['channel_0_intensity_mean'], [], groups=definitions)
    assert set(result['summary']['group']) == {'plate1 + A01', 'plate1 + B01'}
    assert result['summary']['n'].tolist() == [6, 6]
    assert _image_well_groups(['A01', 'A02']) == [{'well': 'A01'}, {'well': 'A02'}]
    with pytest.raises(ValueError):
        _image_well_groups(['A01'])
    with pytest.raises(ValueError):
        _image_well_groups(['A01', 'A01'])


def test_compact_table_has_means_effect_sizes_and_corrected_p_values():
    frame = _frame()
    result = _compare_image_groups(frame, ['channel_0_intensity_mean', 'channel_0_focus_brenner'],
                                   ['cell_type'], unit='well')
    table = _image_comparison_table(result)
    assert list(table['measurement']) == ['channel_0_intensity_mean', 'channel_0_focus_brenner']
    first = table.iloc[0]
    assert {first['group_a'], first['group_b']} == {'HeLa', 'HFF'}
    assert (first['n_a'], first['n_b']) == (2, 2)
    means = {first['group_a']: first['mean_a'], first['group_b']: first['mean_b']}
    assert means['HFF'] - means['HeLa'] == pytest.approx(8.0, abs=1.0)
    assert np.sign(first['effect_size']) == np.sign(first['mean_a'] - first['mean_b'])
    assert first['log2_ratio'] == pytest.approx(np.log2(first['mean_a'] / first['mean_b']))
    assert first['p_adjusted'] >= first['p_value']
    per_image = _image_comparison_table(_compare_image_groups(
        frame, ['channel_0_intensity_mean', 'channel_0_focus_brenner'], ['cell_type']))
    assert per_image.iloc[0]['p_adjusted'] < 1e-6 and per_image.iloc[0]['significance'].startswith('***')
    assert (per_image.iloc[0]['n_a'], per_image.iloc[0]['n_b']) == (12, 12)
    assert _image_comparison_table({}).empty


def test_ranking_puts_the_measurement_that_separates_the_groups_first():
    ranked = _rank_image_differences(_frame(), ['cell_type'])
    assert ranked.iloc[0]['measurement'] == 'channel_0_intensity_mean'
    assert 'channel_0_radial_bin_0' not in set(ranked['measurement'])
    assert len(_rank_image_differences(_frame(), ['cell_type'], top=1)) == 1
    wells = _rank_image_differences(_frame(), groups=_image_well_groups(['A01', 'A02', 'B01']))
    assert set(wells['group_a']) | set(wells['group_b']) == {'A01', 'A02', 'B01'}
    assert len(wells) == 3 * 2


def test_effect_figure_is_drawn_and_saved(tmp_path):
    ranked = _rank_image_differences(_frame(), ['cell_type'])
    figure, saved = _image_effect_figure(ranked, path=tmp_path / 'effects.png')
    from pathlib import Path
    assert saved is not None and Path(saved).exists() and Path(saved).parent == tmp_path
    assert len(figure.axes[0].get_yticklabels()) == len(ranked)
    with pytest.raises(ValueError):
        _image_effect_figure(ranked.iloc[0:0])


def test_omnibus_only_constant_and_unsaved_comparisons():
    frame = _frame()
    omnibus = _compare_image_groups(frame, ['channel_0_intensity_mean'], ['well'], pairwise=False)
    assert len(omnibus['tests']) == 1 and _image_comparison_table(omnibus).empty
    assert _rank_image_differences(frame, ['cell_type'], measurements=['channel_0_radial_bin_0']).empty
    figure, saved = _image_effect_figure(_rank_image_differences(frame, ['cell_type']))
    assert saved is None and figure.axes
