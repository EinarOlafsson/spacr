"""Gene-level complex membership importance (spacr.ml.complex_membership_importance)."""
import numpy as np
import pandas as pd

from spacr.ml import complex_membership_importance


def _inputs(seed=0, n_genes=240, n_columns=40, n_complexes=5):
    """Synthetic profiles in which column c carries the membership of complex c (columns 0-4)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_genes, n_columns))
    rows = []
    for c in range(n_complexes):
        members = rng.choice(n_genes, 30, replace=False)
        X[members, c] += 3.0
        rows += [(f'cx{c}', f'g{m}') for m in members]
    genes = [f'g{i}' for i in range(n_genes)]
    columns = [f'f{j}' for j in range(n_columns)]
    profiles = pd.DataFrame(X, columns=columns)
    profiles.insert(0, 'gene', genes)
    split = pd.DataFrame({'gene': genes, 'is_train': np.arange(n_genes) % 2 == 0})
    groups = pd.DataFrame({'column': columns,
                           'family': ['signal'] * 5 + ['other'] * (n_columns - 5),
                           'compartment': ['nucleus'] * 20 + ['cell'] * (n_columns - 20)})
    return profiles, pd.DataFrame(rows, columns=['label_set', 'gene']), split, groups


SETTINGS = {'n_estimators': 30, 'n_shuffles': 12, 'bootstrap': 50, 'perm_repeats': 2,
            'null_gain_fits': 4}


def test_planted_signal_ranks_top_and_null_is_near_zero():
    """The planted columns lead by SHAP and permutation; the signal family beats its shuffled share."""
    out = complex_membership_importance(*_inputs(), SETTINGS)
    assert (out['label_sets'].verdict == 'above').all()
    columns = out['importance_columns']
    for measure in ('shap', 'permutation'):
        part = columns[columns.measure == measure].sort_values('share', ascending=False)
        assert set(part.column.head(5)) == {f'f{j}' for j in range(5)}
    families = out['importance_groups']
    family = families[(families.level == 'family') & (families.measure == 'shap')].set_index('group')
    assert family.loc['signal', 'ci_lo'] > family.loc['signal', 'shuffled_share']
    assert family.loc['signal', 'share'] > family.loc['other', 'share']
    assert family.loc['signal', 'enrichment'] > 1
    assert abs(out['pooled'].null_mean.iloc[0]) < 0.1


def test_same_seed_gives_identical_output():
    """A rerun with the same inputs and settings gives identical tables."""
    first = complex_membership_importance(*_inputs(), SETTINGS)
    second = complex_membership_importance(*_inputs(), SETTINGS)
    for name in ('label_sets', 'pooled', 'importance_columns', 'importance_groups'):
        pd.testing.assert_frame_equal(first[name], second[name])


def test_family_shares_sum_to_one_and_follow_the_columns():
    """Family and compartment shares sum to 1 per measure and equal the summed column shares."""
    out = complex_membership_importance(*_inputs(), SETTINGS)
    columns, groups = out['importance_columns'], out['importance_groups']
    for measure in ('gain', 'shap', 'permutation'):
        for level in ('family', 'compartment'):
            part = groups[(groups.measure == measure) & (groups.level == level)]
            assert abs(part.share.sum() - 1) < 1e-9
        by_family = columns[columns.measure == measure].groupby('family').share.sum()
        shown = groups[(groups.measure == measure) & (groups.level == 'family')].set_index('group').share
        assert np.allclose(by_family.sort_index(), shown.sort_index())


def test_heldout_halves_and_topk():
    """Half-1 sets are refitted on the top-k columns and on random sets of the same size."""
    profiles, labels, split, groups = _inputs()
    label_sets = pd.DataFrame({'label_set': ['cx0', 'cx1', 'cx2', 'cx3', 'cx4'], 'kind': 'CORUM',
                               'half': [0, 0, 0, 1, 1]})
    small = complex_membership_importance(profiles, labels, split, groups,
                                          {**SETTINGS, 'top_k': 5, 'random_subsets': 4}, label_sets)
    top = small['heldout_topk']['CORUM']
    assert top['k'] == 5 and top['n_sets'] == 2 and np.isfinite(top['difference_mean'])
    assert set(small['label_sets'][small['label_sets'].half == 1].label_set) == {'cx3', 'cx4'}
    everything = complex_membership_importance(profiles, labels, split, groups,
                                               {**SETTINGS, 'top_k': 40, 'random_subsets': 3},
                                               label_sets)
    assert abs(everything['heldout_topk']['CORUM']['difference_mean']) < 1e-9
