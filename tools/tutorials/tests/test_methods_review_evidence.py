"""The export check preserves errors as evidence, not as approved prose."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods_review_evidence import check_export, check_gene_family, review_warnings


def family():
    return [dict(gene=str(i), p_value=p, q_value=q, adjusted_p_value=q,
                 tested_genes_in_family=3, multiple_testing_method='fdr_bh',
                 significant=str(q < .05), alpha=.05)
            for i, (p, q) in enumerate([(.01, .03), (.03, .045), (.8, .8)])]


def test_complete_family_positive():
    checked = check_gene_family(family())
    assert checked['n_genes'] == 3 and checked['n_significant'] == 2
    assert checked['all_bh_values_checked'] is True


@pytest.mark.parametrize('key,value', [('q_value', .01), ('adjusted_p_value', .01),
    ('significant', 'False'), ('tested_genes_in_family', 4), ('p_value', float('nan'))])
def test_corrupt_family_rejected_after_positive(key, value):
    check_gene_family(family())
    rows = family(); rows[0][key] = value
    with pytest.raises(ValueError):
        check_gene_family(rows)


def test_duplicate_gene_is_not_an_independent_test():
    check_gene_family(family())
    rows = family(); rows[1]['gene'] = rows[0]['gene']
    with pytest.raises(ValueError):
        check_gene_family(rows)


def test_export_prose_and_digest_both_preserved():
    digest = {'run': {'seed_declared': {'guide_permutation_seed': 0}, 'seed': 42}}
    prefix = '## Methods\nUnreviewed seed 42.\n\n## Results\nUnreviewed wording.\n\n## Appendix: run digest\n\n```json\n'
    text = prefix + json.dumps(digest) + '\n```\n'
    check_export(text, '## Methods\nUnreviewed seed 42.', '## Results\nUnreviewed wording.', digest)
    with pytest.raises(ValueError):
        check_export(text.replace('seed 42.', 'seed 0.'), '## Methods\nUnreviewed seed 42.', '## Results\nUnreviewed wording.', digest)
    wrong = deepcopy(digest); wrong['run']['seed'] = 0
    with pytest.raises(ValueError):
        check_export(prefix + json.dumps(wrong) + '\n```\n', '## Methods\nUnreviewed seed 42.', '## Results\nUnreviewed wording.', digest)


def test_declared_global_seed_and_matching_version_have_no_inferred_warning():
    digest = {'hits': [], 'spacr_version': 'a'}
    manifest = {'seeds': {'declared': {'random_seed': 42}}, 'env': {'spacr': 'a'}}
    assert review_warnings(digest, manifest, {'n_significant': 0}, 'No candidates.') == []
    manifest['seeds']['declared'] = {'guide_permutation_seed': 0}
    digest['spacr_version'] = 'b'; digest['hits'] = [{'gene': 'example'}]
    issues = review_warnings(digest, manifest, {'n_significant': 0}, 'The strongest hits were:')
    assert len(issues) == 3
    assert any('global run seed' in item for item in issues)
    assert any('not significant' in item for item in issues)
    assert any('analysis version' in item for item in issues)
