"""Exact S2J never-set order and source-bound builder documentation (item 1014)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = 'spacr.ops_sbs.never_barcodes'
CHANGE = ROOT / 'tests/data/release_contracts/1014s2j_never_builder_api_change_2026-11-22.json'
SYNTHETIC_LIBRARY = ('GATGGCGTTAC', 'GGTGCAAAAGC', 'GGTTAGATCTC')
SYNTHETIC_EXPECTED = {1: ('GATGGGGTTAC', 'GATGGCGTGAC', 'AATGGCGTTAC', 'GATGGCGTTTC', 'GGAGCAAAAGC', 'GGTGCAAAACC', 'GGTTCAAAAGC', 'GGTGGAAAAGC', 'GGTTAGACCTC', 'AGTTAGATCTC', 'GGTTAGATCGC', 'GGTTAGATGTC'), 2: ('GATGTAGTTAC', 'GCTGGCGTTCC', 'GATGTCGTAAC', 'GATGACGATAC', 'GGTGCAAAGGT', 'GGTAAAAAAGC', 'GATGGAAAAGC', 'GGTGAAAAACC', 'CGTTAGATATC', 'GTTTTGATCTC', 'TGTTAGACCTC', 'GGGTAGAACTC')}


def _s2j_change():
    """Bind the single permitted builder doc revision to its exact current source syntax."""
    symbols = json.loads(CHANGE.read_text(encoding='utf-8'))['symbols']
    assert set(symbols) == {SYMBOL}
    change = symbols[SYMBOL]
    assert change['file'] == 'spacr/ops_sbs.py'
    source = (ROOT / change['file']).read_text(encoding='utf-8')
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == 'never_barcodes')
    assert ast.get_docstring(node) == change['source']
    assert hashlib.sha256(change['source'].encode()).hexdigest() == change['source_sha256']
    assert hashlib.sha256(change['before'].encode()).hexdigest() == change['before_sha256']
    assert ast.unparse(node.args) == change['signature']
    assert (ast.unparse(node.returns) if node.returns else None) == change['return_annotation']
    return change


def _current_s2j_records(records):
    """Advance only the current builder source fields while retaining historical signatures and bodies."""
    change = _s2j_change()
    previous = records[SYMBOL]
    assert previous['source'] == change['before']
    assert previous['source_sha256'] == change['before_sha256']
    assert previous['signature'] == change['signature']
    assert previous['return_annotation'] == change['return_annotation']
    current = dict(previous)
    for key in ('source', 'source_sha256', 'blocks', 'layout', 'line'):
        current[key] = change[key]
    return {**records, SYMBOL: current}


def _current_s2j_api(entries):
    """Compose the exact S2J builder doc revision after existing never-control API adapters."""
    result = dict(entries)
    if SYMBOL in result:
        change = _s2j_change()
        assert result[SYMBOL] in (change['before'], change['source'])
        result[SYMBOL] = change['source']
    return result


@pytest.mark.parametrize('distance', [1, 2])
def test_synthetic_acceptance_order_matches_the_paper_builder(distance):
    """Complete each sorted guide before the next, using the reference seeded draw order."""
    from spacr.ops_sbs import never_barcodes

    observed = never_barcodes(SYNTHETIC_LIBRARY, distance=distance, k=4, seed=1)['barcodes']
    assert tuple(observed) == SYNTHETIC_EXPECTED[distance]


@pytest.mark.parametrize('library', [(), ('A',)])
def test_distance_two_keeps_empty_results_for_guides_too_short_to_mutate(library):
    """Retain the previous empty N2 result when no guide has two positions."""
    from spacr.ops_sbs import never_barcodes

    result = never_barcodes(library, distance=2, k=4, seed=1)
    assert result['barcodes'] == ()
    assert result['m_d'].shape == (len(library), 3)
    assert result['library_m_d'].shape == (len(library), 3)


def test_s2j_doc_revision_composes_without_altering_unrelated_api_snapshots():
    """Preserve every historical contract and project only the revised builder body."""
    from tests.test_ops_fdr_never_1014 import _current_never_api, _never_api_contracts

    change = _s2j_change()
    original = json.loads((ROOT / 'tests/data/release_contracts/1014_never_api_sources_2026-11-22.json')
                          .read_text(encoding='utf-8'))['symbols']
    current = _never_api_contracts()
    assert original.keys() == current.keys()
    assert all(current[key] == original[key] for key in original if key != SYMBOL)
    assert original[SYMBOL]['source'] == change['before']
    assert current[SYMBOL]['source'] == change['source']
    assert current[SYMBOL]['before'] is original[SYMBOL]['before'] is None
    entries = {SYMBOL: change['before'], 'unrelated.symbol': 'unchanged source'}
    expected = {**entries, SYMBOL: change['source']}
    assert _current_s2j_api(entries) == expected
    assert _current_never_api(entries) == expected
    assert _current_s2j_api(expected) == expected
    assert entries[SYMBOL] == change['before']


@pytest.fixture(scope='module')
def s2j_data():
    """Use only the two explicitly supplied reference roots when both opt-in variables are set."""
    import os

    reference = os.environ.get('SPACR_S2J_REF')
    inputs = os.environ.get('SPACR_S2J_INPUTS')
    if not reference or not inputs:
        pytest.skip('Set SPACR_S2J_REF and SPACR_S2J_INPUTS for the full S2J reference gate.')
    roots = Path(reference).resolve(), Path(inputs).resolve()
    snapshots = {}
    for root in roots:
        assert root.is_dir(), root
        snapshots[root] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in root.rglob('*') if path.is_file()}
        for line in (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
            digest, relative = line.split(maxsplit=1)
            path = (root / relative).resolve()
            assert path.is_relative_to(root), relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
    yield roots
    for root in roots:
        after = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in root.rglob('*') if path.is_file()}
        assert after == snapshots[root], 'Reference inputs or file inventories changed during parity tests.'



def _barcode_bytes(barcodes):
    """Reproduce the ordered one-column CSV bytes used by the reference export."""
    return ('barcode\n' + ''.join(str(value) + '\n' for value in barcodes)).encode('utf-8')


def _set_parity(dataset, name, actual, expected, reference_file):
    """Report exact row, order and byte parity with only the first mismatch when one exists."""
    from itertools import zip_longest

    actual, expected = list(actual), list(expected)
    first = next(({'index': index, 'spaCR': left, 'reference': right}
                  for index, (left, right) in enumerate(zip_longest(actual, expected))
                  if left != right), None)
    actual_sha = hashlib.sha256(_barcode_bytes(actual)).hexdigest()
    reference_sha = hashlib.sha256(reference_file.read_bytes()).hexdigest()
    report = {'dataset': dataset, 'set': name, 'rows_spacr': len(actual),
              'rows_reference': len(expected), 'order_equal': first is None,
              'sha256_spacr': actual_sha, 'sha256_reference': reference_sha,
              'first_difference': first}
    print('S2J_SET_PARITY ' + json.dumps(report, sort_keys=True), flush=True)
    assert first is None, report
    assert actual_sha == reference_sha, report


@pytest.mark.slow
@pytest.mark.timeout(2400)
@pytest.mark.parametrize('dataset,size', [('funk_full', 20445), ('peri_full', 80862)])
def test_full_s2j_never_sets_and_production_decoys_are_exact(s2j_data, dataset, size):
    """Match all D, N1 and N2 rows, acceptance order and raw CSV hashes on the production libraries."""
    from spacr import ops_sbs
    from spacr.tabular import read_table

    reference, inputs = s2j_data
    library = read_table(inputs / ('LIB_' + dataset + '.csv'), canonicalise=False,
                         report=None)['barcode'].astype(str).tolist()
    assert len(library) == size and library == sorted(set(library))
    if dataset == 'peri_full':
        barcodes = inputs / 'periscope_Barcodes.csv'
        if barcodes.is_file():
            from spacr.ops_prefix import spacer_prefix_library

            rebuilt = spacer_prefix_library(str(barcodes), sequence_col='sgRNA', gene_col='gene_symbol',
                                            prefix_length=12, n_cycles=12, cycle_mask=[], collision='exclude')
            assert list(rebuilt.keys) == library
            print('S2J_PREFIX_LIBRARY raw table equals the supplied export', flush=True)
        else:
            print('S2J_PREFIX_LIBRARY raw table unavailable; using the approved production key export', flush=True)
    decoys = ops_sbs.decoy_library(library, k=4, seed=1)
    if dataset == 'peri_full':
        decoys = sorted(decoys)
    for name, values in [('D', decoys)]:
        path = reference / 'sets' / dataset / ('set_' + name + '.csv')
        expected = read_table(path, canonicalise=False, report=None)['barcode'].astype(str).tolist()
        _set_parity(dataset, name, values, expected, path)
    for distance in (1, 2):
        result = ops_sbs.never_barcodes(library, distance=distance, k=4, seed=1)
        path = reference / 'sets' / dataset / ('set_N' + str(distance) + '.csv')
        expected = read_table(path, canonicalise=False, report=None)['barcode'].astype(str).tolist()
        _set_parity(dataset, 'N' + str(distance), result['barcodes'], expected, path)


def _reference_estimator(path, dataset):
    """Extract only the exact reference estimator AST and its literal constants, executing no script setup."""
    import numpy as np

    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    assignments = {node.targets[0].id: node.value for node in tree.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)}
    assert ast.literal_eval(assignments['K']) == 4
    fixed = assignments['FIXED']
    assert isinstance(fixed, ast.Subscript) and isinstance(fixed.slice, ast.Name)
    assert fixed.slice.id == 'DATASET'
    constants = ast.literal_eval(fixed.value)
    node = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == 'est')
    isolated = ast.Module(body=[node], type_ignores=[])
    namespace = {'np': np, 'K': 4, 'FIXED': constants[dataset]}
    exec(compile(isolated, str(path), 'exec'), namespace)
    return namespace['est']


@pytest.mark.parametrize('dataset,fields,rounded', [('funk', 333, 1.35566), ('periscope', 85, 1.96469)])
def test_reference_estimator_and_spacr_reproduce_the_s2j_scale_points(s2j_data, dataset, fields, rounded):
    """Reproduce the N2-only S2J scale with the exact reference estimator and the existing spaCR owner."""
    import numpy as np
    from spacr.ops_decode import never_scale
    from spacr.tabular import read_table

    reference, inputs = s2j_data
    table = read_table(inputs / ('per_field_' + dataset + '.csv'), canonicalise=False, report=None)
    table = table.loc[np.isclose(table['b'].to_numpy(float), 1.0)].copy()
    assert len(table) == fields and table['site'].is_unique
    sums = {column: table[column].to_numpy(float).sum() for column in table.columns
            if column not in ('site', 'b')}
    estimate = _reference_estimator(reference / 'scripts/post.py', dataset)(sums)
    observed = never_scale(table['real_D'], table['calls_D'], table['calls_N2'],
                          table['opp2_N2'], table['opp2_L'], k=4, ci=False)
    summary = read_table(reference / 'tables' / ('calibration_summary_' + dataset + '.csv'),
                         canonicalise=False, report=None)
    selected = summary.loc[(summary['quantity'] == 's_cal') & np.isclose(summary['b'], 1.0)]
    assert len(selected) == 1
    expected = float(selected.iloc[0]['estimate'])
    assert float(estimate['s_cal']) == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert observed['scale_point'] == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert round(expected, 5) == rounded
    print('S2J_ESTIMATOR_PARITY ' + json.dumps({'dataset': dataset, 'b': 1.0, 'fields': fields,
          'reference': expected, 'post_est': float(estimate['s_cal']),
          'spacr': observed['scale_point'], 'rounded': rounded,
          'mode': 'N2-only, point estimate, no bootstrap'}), flush=True)
