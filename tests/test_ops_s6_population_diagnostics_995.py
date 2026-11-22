"""Retain fixed-fit S6 population diagnostics without changing production numbers."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr import ops_cell_cycle as cc

LEGACY = Path(__file__).parent / 'data/release_contracts/995_population_legacy_007ff7a707.py.txt'
LEGACY_SHA256 = '9e2da66dd849653ab089a10f0fe813feb1ae047f80862c9a08a4a5c9af8ec4fd'


def _legacy_pipeline():
    """Load the exact shared fit and field-centred pipeline from the task base."""
    source = LEGACY.read_bytes()
    assert hashlib.sha256(source).hexdigest() == LEGACY_SHA256
    namespace = dict(vars(measure))
    exec(compile(source, str(LEGACY), 'exec'), namespace)
    return namespace['_field_centred_dna_phases']


def _inputs(*, small_second=False):
    """Make two labelled wells with selected outliers, invalid DNA and excluded cells."""
    rng = np.random.default_rng(99520261122)
    n = 2400
    first = np.concatenate((rng.normal(2, 0.10, 1400), rng.uniform(2, 4, 600),
                            rng.normal(4, 0.20, 400)))
    second = first[:12].copy() if small_second else first.copy() * 1.1
    content = np.concatenate((first, second))
    content[25:35] = 80
    content[40] = np.nan
    excluded = np.zeros(len(content), dtype=bool)
    excluded[::7] = True
    content[excluded] *= 5
    groups = [('A1', np.arange(n)), ('A2', np.arange(n, len(content)))]
    fields = np.arange(len(content)) // 300
    cells = pd.DataFrame({'well': np.repeat(['A1', 'A2'], [n, len(second)]), 'site': fields})
    return content, groups, fields, excluded, cells


@pytest.mark.parametrize('centre', [False, True])
@pytest.mark.parametrize('gates', [None, [2.5, 3.5]])
def test_optional_capture_preserves_the_source_bound_legacy_pipeline_exactly(centre, gates):
    """Keep every fitted field and output byte identical with capture off and on."""
    content, groups, fields, excluded, _ = _inputs()
    options = dict(groups=groups, fields=fields, mitotic=excluded, centre=centre, gates=gates)
    expected_frame, expected_fits = _legacy_pipeline()(content, **options)
    for capture in (False, True):
        frame, fits = measure._field_centred_dna_phases(content, **options,
                                                       capture_diagnostics=capture)
        pd.testing.assert_frame_equal(frame, expected_frame, check_exact=True)
        assert frame.to_csv(index=False).encode() == expected_frame.to_csv(index=False).encode()
        for well, positions in groups:
            assert asdict(fits[well]['fit']) == asdict(expected_fits[well]['fit'])
            assert fits[well]['fields_shifted'] == expected_fits[well]['fields_shifted']
            assert fits[well]['pooled'] == expected_fits[well]['pooled']
            if not capture:
                assert fits[well].keys() == expected_fits[well].keys()
                continue
            record = fits[well]['diagnostics']
            selected = frame['dna_content_centred'].to_numpy()[positions][~excluded[positions]]
            np.testing.assert_array_equal(record['fit_mask_content'], selected)
            low, high = record['trim_bounds']
            kept = selected[np.isfinite(selected) & (selected > 0)]
            kept = kept[(kept >= low) & (kept <= high)]
            np.testing.assert_array_equal(record['trimmed_content'], kept)
            assert len(kept) == fits[well]['fit'].n
            if well == 'A1':
                assert len(kept) < np.isfinite(selected).sum()


def test_pooled_fallback_records_the_actual_pooled_fit_population():
    """Expose pooled training rows instead of falsely labelling the tiny well as the fit input."""
    content, groups, fields, excluded, cells = _inputs(small_second=True)
    frame, fits = measure._field_centred_dna_phases(content, groups=groups, fields=fields,
                                                   mitotic=excluded, capture_diagnostics=True)
    assert fits['A2']['pooled'] is True and fits['A1']['pooled'] is False
    selected = frame['dna_content_centred'].to_numpy()[~excluded]
    np.testing.assert_array_equal(fits['A2']['diagnostics']['fit_mask_content'], selected)
    rows = cc._s6_population_diagnostics(cells, frame, fits)
    pool = rows.loc[(rows['well'] == 'A2') & (rows['population'] == 'fit_mask')].iloc[0]
    scored = rows.loc[(rows['well'] == 'A2') & (rows['population'] == 'scored')].iloc[0]
    assert pool['scope'] == 'pooled' and pool['cells'] == len(selected)
    assert scored['cells'] == 12 and pool['cells'] > scored['cells']


def test_diagnostics_and_report_keep_production_s6_bytes_without_refitting(tmp_path, monkeypatch):
    """Retain all-finite S6 rows and verdicts while separately scoring the same fixed fit."""
    content, groups, fields, excluded, cells = _inputs()
    options = dict(groups=groups, fields=fields, mitotic=excluded)
    old_frame, old_fits = _legacy_pipeline()(content, **options)
    frame, fits = measure._field_centred_dna_phases(content, **options, capture_diagnostics=True)
    stages = pd.DataFrame({'model_stage': np.where(excluded, 'artefact', 'interphase')})
    old_checks = cc.self_checks(cells, stages, old_frame, old_fits, card={},
                                arrest_genes=None, permutations=3, seed=995)
    checks = cc.self_checks(cells, stages, frame, fits, card={},
                            arrest_genes=None, permutations=3, seed=995)
    expected = old_checks.to_csv(index=False).encode()
    assert checks.to_csv(index=False).encode() == expected
    models = {name: asdict(info['fit']) for name, info in fits.items()}

    def forbid_refit(*args, **kwargs):
        """Refuse any attempt to refit while reporting diagnostics."""
        raise AssertionError('Diagnostic reporting must use the saved fit.')

    monkeypatch.setattr(measure, '_fit_dna_content', forbid_refit)
    rows = cc._s6_population_diagnostics(cells, frame, fits)
    scored = rows.loc[rows['population'] == 'scored']
    s6 = checks.loc[checks['check'] == 'S6'].iloc[0]
    assert float(s6['value']) == scored['r2'].min()
    assert float(s6['control']) == np.median(scored['g2_g1'])
    assert scored['cells'].sum() == len(cells)
    assert rows.loc[rows['population'] == 'fit_mask', 'cells'].sum() == (~excluded).sum()
    checks.attrs['s6_population_diagnostics'] = rows
    path = tmp_path / 'diagnostic-report.html'
    cc._report_html(str(path), 'P1', checks, pd.Series({'interphase': len(cells)}),
                    cc.failed_stages(checks), [], {})
    text = path.read_text()
    assert 'diagnostic, not used for the verdict' in text
    for label in ('Fit mask', 'Trimmed fitting range', 'Production scored population'):
        assert label in text
    assert checks.to_csv(index=False).encode() == expected
    assert {name: asdict(info['fit']) for name, info in fits.items()} == models
    assert cc.FIT_R2 == 0.90


def test_legacy_fits_report_unavailable_populations_without_guessing(tmp_path):
    """Keep missing fit-mask and trim provenance explicit when only a saved fit is available."""
    content, groups, fields, excluded, cells = _inputs()
    frame, fits = _legacy_pipeline()(content, groups=groups, fields=fields, mitotic=excluded)
    rows = cc._s6_population_diagnostics(cells, frame, fits)
    missing = rows.loc[rows['population'] != 'scored']
    assert not missing['available'].any()
    assert missing[['cells', 'finite_cells', 'r2', 'dna_min', 'dna_max']].isna().all().all()
    assert rows.loc[rows['population'] == 'scored', 'available'].all()
    checks = pd.DataFrame([{'check': 'S6', 'passed': False, 'value': 0.8}])
    checks.attrs['s6_population_diagnostics'] = rows
    path = tmp_path / 'legacy-report.html'
    cc._report_html(str(path), 'P1', checks, pd.Series({'interphase': len(cells)}),
                    ['G1', 'S', 'G2'], [], {})
    assert 'not retained' in path.read_text()


def test_diagnostic_report_retains_exact_large_population_sizes(tmp_path):
    """Render the full integer population size without the four-digit float rounding."""
    count = 123456
    values = np.tile([2.0, 4.0], count // 2)
    cells = pd.DataFrame({'well': ['A1'] * count})
    dna = pd.DataFrame({'dna_content_centred': values})
    fit = measure._DnaFit(2, 4, 0.1, 0.2, (0.5, 0, 0.5, 0), (2.5, 3.5), count, 0)
    rows = cc._s6_population_diagnostics(cells, dna, {'A1': {'fit': fit, 'pooled': False}})
    checks = pd.DataFrame([{'check': 'S6', 'passed': False, 'value': 0.2}])
    checks.attrs['s6_population_diagnostics'] = rows
    path = tmp_path / 'large-report.html'
    cc._report_html(str(path), 'P1', checks, pd.Series({'interphase': count}),
                    ['G1', 'S', 'G2'], [], {})
    assert '<td>123456</td>' in path.read_text()
