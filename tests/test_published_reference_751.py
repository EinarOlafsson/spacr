"""Item 751: compare a screen with a published per-guide or per-gene table."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr import published_reference as pr
from spacr.tabular import read_table, write_table


def _wide_reference():
    """A per-gene table laid out like Funk et al. 2022's: one column group per guide."""
    return pd.DataFrame({
        'gene_symbol': ['AAAS', 'AAGAB', 'U2AF1', 'U2AF1L5', 'KIF11'],
        'sgRNA_1': ['ATATGAGCACAATAAC', 'AGCAGACATCAATCTA', 'AAACAAACCTGGCTAA',
                    'AAACAAACCTGGCTAA', 'GGGTTTCCCAAAGGGT'],
        'sgRNA_1_interphase_cells': [600, 4000, 790, 790, 300],
        'sgRNA_1_mitotic_cells': [16, 134, 5, 5, 90],
        'sgRNA_2': ['CCCGGGAAATTTCCAA', '', 'TTTACCAGGTTTAACC', None, 'GGGTTTCCCAATTTTT'],
        'sgRNA_2_interphase_cells': [100, np.nan, 50, np.nan, 200],
        'sgRNA_2_mitotic_cells': [1, np.nan, 2, np.nan, 50],
        'mitotic_index': [0.02, 0.03, 0.01, 0.01, 0.25],
    })


def test_a_wide_table_becomes_one_row_per_guide(tmp_path):
    path = write_table(_wide_reference(), tmp_path / 'reference.xlsx', canonicalise=False)
    long = pr.read_reference(path, sheet='Sheet1')
    assert len(long) == 8
    assert {'sgRNA', 'guide_number', 'interphase_cells', 'mitotic_cells', 'cells',
            'mitotic_index', 'gene_symbol'} <= set(long.columns)
    first = long[(long['gene_symbol'] == 'AAAS') & (long['guide_number'] == 1)].iloc[0]
    assert first['cells'] == 616 and first['mitotic_index'] == 0.02
    assert pr.read_reference(path).shape == long.shape
    narrow = pd.DataFrame({'gene': ['A'], 'cells': [3]})
    assert pr.read_reference(narrow).equals(narrow)


def test_guides_match_on_the_prefix_and_a_shared_guide_counts_once():
    reference = pr.read_reference(_wide_reference())
    screen = pd.DataFrame({'guide': ['atatgagcaca', 'AGCAGACATCA', 'AAACAAACCTG',
                                     'CCCGGGAAAUU', 'TTTACCAGGTT', 'NOTINREFXXX',
                                     'GGGTTTCCCAA'],
                           'n_cells': [60, 400, 80, 10, 5, 7, 70]})
    joined = pr.match_reference(screen, reference, prefix_length=11)
    summary = joined.attrs['match']
    assert summary['level'] == 'guide'
    assert summary['reference_rows_repeating_a_guide'] == 2
    assert summary['screen_keys_not_in_reference'] == 1
    shared = joined[joined['key'] == 'AAACAAACCTG'].iloc[0]
    assert shared['gene'] == 'U2AF1;U2AF1L5'
    assert shared['reference_cells'] == 795 and shared['reference_rows'] == 1
    kif = joined[joined['key'] == 'GGGTTTCCCAA'].iloc[0]
    assert kif['reference_cells'] == 390 + 250 and kif['reference_rows'] == 2
    assert joined.loc[joined['key'] == 'ATATGAGCACA', 'screen_n_cells'].item() == 60
    assert joined.attrs['pairs'][0] == ('n_cells', 'cells', 'count')

    kept = pr.match_reference(screen, reference, duplicates='keep')
    assert (kept['key'] == 'AAACAAACCTG').sum() == 2
    dropped = pr.match_reference(screen, reference, duplicates='drop')
    assert not {'AAACAAACCTG', 'GGGTTTCCCAA'} & set(dropped['key'])
    assert dropped.attrs['match']['reference_keys_dropped_as_shared'] == 2


def test_unmatched_reference_guides_can_count_as_zero_cells():
    reference = pd.DataFrame({'gene': ['A', 'B', 'C'], 'sgRNA': ['AAAA', 'CCCC', 'GGGG'],
                              'cells': [10, 20, 30], 'score': [1.0, 2.0, 3.0]})
    screen = pd.DataFrame({'guide': ['AAAA', 'CCCC'], 'n_cells': [1, 2], 'score': [0.5, 0.7]})
    zero = pr.match_reference(screen, reference, prefix_length=0, unmatched='zero')
    assert list(zero['match']) == ['guide', 'guide', 'unmatched']
    assert list(zero['screen_n_cells']) == [1, 2, 0]
    assert np.isnan(zero['screen_score'].iloc[2])
    assert [p[2] for p in zero.attrs['pairs']] == ['count', 'value']
    assert len(pr.match_reference(screen, reference, prefix_length=0)) == 2


def test_genes_match_by_case_and_alias_and_pool_their_guides():
    reference = pd.DataFrame({
        'gene_symbol': ['TP53', 'TP53', 'KIF11', 'MKI67', 'ORC1'],
        'aliases': ['P53', 'P53', 'EG5; KNSL1', None, 'TP53'],
        'cells': [100, 50, 70, 20, 10],
        'mitotic_index': [0.02, 0.02, 0.30, 0.05, 0.01],
    })
    screen = pd.DataFrame({'gene': ['tp53', 'Eg5', 'Eg5', 'MKI67', 'ZZZ3'],
                           'guide_seq': ['A', 'C', 'G', 'T', 'A'],
                           'n_cells': [9, 4, 5, 2, 1],
                           'mitotic_index': [0.03, 0.2, 0.4, 0.06, 0.0]})
    joined = pr.match_reference(screen, reference, level='auto')
    assert joined.attrs['match']['level'] == 'gene'
    rows = joined.set_index('key')
    assert rows.loc['TP53', 'match'] == 'case' and rows.loc['TP53', 'reference_cells'] == 150
    assert rows.loc['TP53', 'reference_rows'] == 2
    assert rows.loc['KIF11', 'match'] == 'alias' and rows.loc['KIF11', 'screen_n_cells'] == 9
    assert rows.loc['KIF11', 'screen_mitotic_index'] == pytest.approx(0.3)
    assert rows.loc['MKI67', 'match'] == 'symbol'
    assert 'ZZZ3' not in rows.index
    assert joined.attrs['match']['screen_genes_matched_by_alias'] == 2
    assert pr._alias_map(reference, 'gene_symbol') == {'P53': 'TP53', 'EG5': 'KIF11',
                                                        'KNSL1': 'KIF11'}
    assert pr._alias_map(reference.drop(columns='aliases'), 'gene_symbol') == {}


def test_explicit_columns_and_bad_options():
    screen = pd.DataFrame({'g': ['AAAA', 'CCCC', 'GGGG'], 'n': [1, 2, 3]})
    reference = pd.DataFrame({'gene': ['A', 'B', 'C'], 'sgRNA': ['AAAA', 'CCCC', 'GGGG'],
                              'funk': [5, 6, 7]})
    joined = pr.match_reference(screen, reference, level='guide', screen_key='g',
                                prefix_length=0, counts='n=funk', values=[('n', 'funk')])
    assert joined.attrs['pairs'] == [('n', 'funk', 'count')]
    joined = pr.match_reference(screen, reference, level='guide', screen_key='G',
                                counts=['n=funk', ''])
    assert list(joined['screen_n']) == [1, 2, 3]
    with pytest.raises(ValueError, match='level'):
        pr.match_reference(screen, reference, level='plate')
    with pytest.raises(ValueError, match='duplicates'):
        pr.match_reference(screen, reference, duplicates='merge')
    with pytest.raises(ValueError, match='no guide column'):
        pr.match_reference(screen, reference, level='guide', counts='n=funk')
    with pytest.raises(ValueError, match='no column'):
        pr.match_reference(screen, reference, screen_key='g', level='guide', counts='x=funk')
    with pytest.raises(ValueError, match='no column to compare'):
        pr.match_reference(screen, reference, screen_key='g', level='guide')


def _correlated(n=300, seed=1):
    """A screen and reference whose guide counts agree with noise."""
    rng = np.random.default_rng(seed)
    truth = rng.lognormal(6, 0.5, n)
    guides = [''.join(rng.choice(list('ACGT'), 12)) for _ in range(n)]
    reference = pd.DataFrame({'gene': [f'G{i}' for i in range(n)], 'sgRNA': guides,
                              'cells': truth * 10})
    screen = pd.DataFrame({'guide': guides, 'n_cells': truth * rng.lognormal(0, 0.2, n)})
    return screen, reference


def test_agreement_reports_both_methods_with_intervals_and_a_null_control():
    screen, reference = _correlated()
    joined = pr.match_reference(screen, reference, prefix_length=12)
    table = pr.reference_agreement(joined, n_boot=200, shuffles=200, seed=3, chunk=64)
    assert list(table['method']) == ['spearman', 'spearman', 'pearson', 'pearson']
    observed = table[table['row'] == 'observed'].set_index('method')
    control = table[table['row'] != 'observed'].set_index('method')
    from scipy.stats import pearsonr, spearmanr
    x, y = joined['screen_n_cells'], joined['reference_cells']
    assert observed.loc['spearman', 'estimate'] == pytest.approx(spearmanr(x, y)[0])
    assert observed.loc['pearson', 'estimate'] == pytest.approx(pearsonr(x, y)[0])
    for method in ('spearman', 'pearson'):
        row = observed.loc[method]
        assert row['ci_low'] < row['estimate'] < row['ci_high'] and row['n'] == 300
        assert abs(control.loc[method, 'estimate']) < 0.05
        assert control.loc[method, 'shuffled_p95_abs'] < row['estimate']
        assert row['p_shuffled'] == pytest.approx(1 / 201)
    assert table.equals(pr.reference_agreement(joined, n_boot=200, shuffles=200, seed=3))
    assert set(pr.GLOSSARY) == set(table.columns)


def test_agreement_without_bootstrap_shuffles_or_enough_rows():
    screen, reference = _correlated(n=40)
    joined = pr.match_reference(screen, reference, prefix_length=12)
    bare = pr.reference_agreement(joined, n_boot=0, shuffles=0)
    assert list(bare['row']) == ['observed', 'observed']
    assert bare['ci_low'].isna().all() and bare['p_shuffled'].isna().all()
    tiny = pr.reference_agreement(joined.head(2), [('n_cells', 'cells')], n_boot=10, shuffles=10)
    assert tiny['estimate'].isna().all() and len(tiny) == 4
    flat = joined.assign(screen_n_cells=1.0)
    assert np.isnan(pr._rowwise_pearson(flat['screen_n_cells'].to_numpy()[None],
                                        flat['reference_cells'].to_numpy()[None])[0])


def test_the_scatter_registers_its_data_and_saves(tmp_path):
    screen, reference = _correlated(n=60)
    joined = pr.match_reference(screen, reference, prefix_length=12)
    table = pr.reference_agreement(joined, n_boot=20, shuffles=20)
    path = tmp_path / 'scatter.png'
    fig = pr.plot_reference_agreement(joined, 'n_cells', 'cells', agreement=table,
                                      path=str(path), log=True)
    assert fig._spacr_spec['kind'] == 'scatter'
    assert fig._spacr_spec['x'] == 'reference_cells'
    assert len(fig._spacr_data) == 60
    assert list(tmp_path.glob('scatter.*'))
    text = fig.axes[0].texts[0].get_text()
    assert 'Spearman' in text and 'shuffled' in text and 'n = 60' in text
    plain = pr.plot_reference_agreement(joined, 'n_cells', 'cells')
    assert plain.axes[0].get_xscale() == 'linear' and not plain.axes[0].texts
    wide = joined.assign(reference_cells=np.geomspace(1, 1e4, 60),
                         screen_n_cells=np.geomspace(1, 1e3, 60))
    assert pr.plot_reference_agreement(wide, 'n_cells', 'cells').axes[0].get_xscale() == 'log'
    bare = pr.reference_agreement(joined, n_boot=0, shuffles=0)
    lone = pr.plot_reference_agreement(joined, 'n_cells', 'cells', agreement=bare)
    assert '[' not in lone.axes[0].texts[0].get_text()


def test_compare_to_reference_writes_every_table(tmp_path, capsys):
    screen, reference = _correlated(n=80)
    screen_path = write_table(screen, tmp_path / 'guides.csv', canonicalise=False)
    reference_path = write_table(reference, tmp_path / 'published.csv', canonicalise=False)
    result = pr.compare_to_reference({'published_screen': screen_path,
                                      'published_reference': reference_path,
                                      'published_prefix_length': 12,
                                      'published_bootstrap': 50, 'published_shuffles': 50})
    out = tmp_path / 'published_reference'
    assert {p.split('/')[-1].split('.')[0] for p in result['paths']} == {
        'joined', 'agreement', 'glossary', 'agreement_n_cells', 'summary'}
    assert len(read_table(out / 'joined.csv', canonicalise=False, report=None)) == 80
    assert len(read_table(out / 'agreement.csv', canonicalise=False, report=None)) == 4
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['match']['matched'] == 80 and summary['settings']['published_bootstrap'] == 50
    assert 'Spearman' in capsys.readouterr().out
    framed = pr.compare_to_reference({'published_screen': screen, 'published_reference': reference,
                                      'published_prefix_length': 12, 'published_bootstrap': 0,
                                      'published_shuffles': 0,
                                      'published_dst': str(tmp_path / 'framed')})
    assert len(framed['agreement']) == 2
    assert json.loads((tmp_path / 'framed' / 'summary.json').read_text())[
        'settings']['published_screen'] == '<frame>'
    with pytest.raises(ValueError, match='published_screen'):
        pr.compare_to_reference({'published_reference': reference_path})
    with pytest.raises(ValueError, match='published_reference'):
        pr.compare_to_reference({'published_screen': screen_path})


def test_blank_destination_with_frames_writes_beside_the_working_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    screen, reference = _correlated(n=20)
    pr.compare_to_reference({'published_screen': screen, 'published_reference': reference,
                             'published_prefix_length': 12, 'published_bootstrap': 0,
                             'published_shuffles': 0})
    assert (tmp_path / 'published_reference' / 'joined.csv').exists()


def test_command_line(tmp_path):
    screen, reference = _correlated(n=30)
    screen_path = write_table(screen, tmp_path / 'guides.csv', canonicalise=False)
    reference_path = write_table(reference, tmp_path / 'published.csv', canonicalise=False)
    args = ['--screen', str(screen_path), '--reference', str(reference_path),
            '--prefix-length', '12', '--bootstrap', '10', '--shuffles', '10',
            '--dst', str(tmp_path / 'cli')]
    assert pr.main(args) == 0
    assert (tmp_path / 'cli' / 'agreement.csv').exists()


def test_settings_are_registered_and_alpha():
    from spacr import settings as S

    defaults = S.defaults_for(pr.APP_KEY, {})
    assert set(defaults) == set(pr._EXPECTED_TYPES) == set(pr._TOOLTIPS) == set(pr._DEFAULTS)
    for key, kind in pr._EXPECTED_TYPES.items():
        assert S.expected_types[key] is kind and isinstance(defaults[key], kind)
        assert S.tooltips[key].startswith(f'({kind.__name__}) - ') and len(S.tooltips[key]) <= 600
    categorised = {key for keys in pr._CATEGORIES.values() for key in keys}
    assert categorised == set(defaults)
    for name, keys in pr._CATEGORIES.items():
        assert set(keys) <= set(S.categories[name])
    assert set(S.ALPHA_FEATURES[751]['module_settings'][pr.APP_KEY]) == set(defaults)
    assert set(defaults) <= S._alpha_names('settings', app_key=pr.APP_KEY)
    assert pr.published_reference_defaults({'published_seed': 4})['published_seed'] == 4
    pr._register()
