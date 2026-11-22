"""Report an unjudged interphase call when a synthetic well fails the S6 R² gate."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import pytest

from spacr import ops_cell_cycle as cc


def _well_checks(poor_fit):
    """Judge an actual seeded DNA histogram against a known two-peak mixture."""
    from spacr.measure import _DnaFit

    rng = np.random.default_rng(20261122)
    count = 8000
    content = (rng.uniform(1, 5, count) if poor_fit else
               rng.permutation(np.concatenate((rng.normal(2, 0.1, 5600),
                                               rng.normal(4, 0.15, 2400)))))
    cells = pd.DataFrame({'well': ['W1'] * count, 'site': np.repeat(np.arange(8), 1000)})
    stages = pd.DataFrame({'model_stage': ['interphase'] * count})
    dna = pd.DataFrame({'dna_content_centred': content})
    fit = _DnaFit(g1=2, g2=4, sd1=0.1, sd2=0.15, weights=(0.7, 0, 0.3, 0),
                  gates=(2.5, 3.5), n=count, iterations=0)
    return cc.self_checks(cells, stages, dna, {'W1': {'fit': fit}}, card={},
                          arrest_genes=None, permutations=3)


@pytest.mark.parametrize('poor_fit', [True, False])
def test_s6_report_explains_unjudged_phases_and_preserves_other_verdicts(tmp_path, poor_fit):
    """Show the exact rounded reason four times only for a genuinely poor synthetic fit."""
    checks = _well_checks(poor_fit)
    original = checks.copy(deep=True)
    s6 = checks.loc[checks['check'] == 'S6'].iloc[0]
    assert cc.FIT_R2 == 0.90
    assert (float(s6['value']) < 0.8) if poor_fit else (float(s6['value']) >= 0.90)
    assert bool(s6['passed']) is (not poor_fit)
    counts = pd.Series({'interphase': 8000, 'metaphase': 12}, name='cells')
    failed = cc.failed_stages(checks)
    path = tmp_path / 'report.html'
    cc._report_html(str(path), 'P1', checks, counts, failed, [], {})
    result = path.read_text(encoding='utf-8')
    reason = f"not judged (S6 R^2 {s6['value']:.2f} < 0.90)"
    rows = re.findall(r'<tr>.*?</tr>', result, flags=re.S)
    s6_row = next(row for row in rows if '<td>S6</td>' in row)
    if poor_fit:
        assert result.count(html.escape(reason)) == 4
        assert html.escape(reason) in s6_row
        for stage in ('G1', 'S', 'G2'):
            assert f'<p>{stage}: {html.escape(reason)}</p>' in result
    else:
        assert 'not judged (S6 R^2' not in result
        assert '<td>True</td>' in s6_row
        assert failed == []
    for index in checks.index[checks['check'] != 'S6']:
        expected_rows = re.findall(r'<tr>.*?</tr>', cc._html_table(checks.loc[[index]]), flags=re.S)
        assert expected_rows[-1] in rows
    assert cc._html_table(counts.rename_axis('stage').reset_index(name='cells')) in result
    pd.testing.assert_frame_equal(checks, original)
    assert cc.failed_stages(checks) == failed


def test_ratio_only_s6_failure_keeps_the_existing_report_verdict(tmp_path):
    """Do not describe a passing R² as below threshold when only the peak ratio failed."""
    checks = pd.DataFrame([{'check': 'S6', 'stages': 'G1,S,G2', 'value': 0.96,
                            'threshold': 0.90, 'passed': False, 'note': 'ratio out of range'}])
    path = tmp_path / 'ratio-report.html'
    cc._report_html(str(path), 'P1', checks, pd.Series({'S/G2': 10}), ['S/G2'], [], {})
    result = path.read_text(encoding='utf-8')
    assert 'not judged (S6 R^2' not in result
    assert 'ratio out of range' in result
    assert cc.FIT_R2 == 0.90


@pytest.mark.parametrize('value,passed', [(0.8999, False), (0.90, True)])
def test_report_rounding_does_not_move_the_s6_acceptance_boundary(tmp_path, value, passed):
    """Format two decimals while keeping the exact 0.90 acceptance decision."""
    checks = pd.DataFrame([{'check': 'S6', 'stages': 'G1,S,G2', 'value': value,
                            'threshold': 0.90, 'passed': passed, 'note': 'fit quality'}])
    path = tmp_path / 'boundary-report.html'
    cc._report_html(str(path), 'P1', checks, pd.Series({'interphase': 10}),
                    cc.failed_stages(checks), [], {})
    result = path.read_text(encoding='utf-8')
    assert ('not judged (S6 R^2 0.90 &lt; 0.90)' in result) is (not passed)
    assert bool(checks.iloc[0]['passed']) is passed
    assert cc.FIT_R2 == 0.90
