"""Source-only exact adapters for the opt-in external max-projection API revisions."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'tests/data/release_contracts/1012_external_api_changes_2026-11-22.json'
SYMBOLS = frozenset({
    'spacr.ops_maxproj.calibrate_view', 'spacr.ops_maxproj.label_clusters_by_view',
    'spacr.ops_maxproj.max_null_pvalues', 'spacr.ops_maxproj.max_projection',
    'spacr.ops_maxproj.maxproj_scores', 'spacr.ops_maxproj.pair_bins',
    'spacr.ops_maxproj.projected_graph', 'spacr.ops_maxproj.view_similarity',
    'spacr.ops_reference.embed_gene_profiles'})


@lru_cache(maxsize=1)
def _external_api_sources():
    """Bind exactly nine revisions to current syntax, prose hashes and immutable before bodies."""
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    assert contract['before_api_count'] == contract['after_api_count'] == 14961
    records = contract['symbols']
    assert set(records) == SYMBOLS
    trees = {}
    for symbol, record in records.items():
        file = record['file']
        if file not in trees:
            trees[file] = ast.parse((ROOT / file).read_text(encoding='utf-8'))
        name = symbol.rsplit('.', 1)[1]
        node = next(node for node in trees[file].body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.get_docstring(node) == record['source'], symbol
        assert ast.unparse(node.args) == record['signature'], symbol
        assert (ast.unparse(node.returns) if node.returns else None) == record['return_annotation'], symbol
        for field in ('source', 'before'):
            assert hashlib.sha256(record[field].encode()).hexdigest() == record[field + '_sha256'], symbol
    return records


def _current_external_maxproj_api(entries):
    """Apply only the exact bound external-API revisions after existing arrival adapters."""
    result = dict(entries)
    for symbol, record in _external_api_sources().items():
        if symbol in result:
            assert result[symbol] in (record['before'], record['source']), symbol
            result[symbol] = record['source']
    return result


def _historical_external_arrivals():
    """Read both immutable historical arrivals containing the nine revised symbols."""
    entries = {}
    for name in ('871_911_bmc_paper_items_api_arrivals_2026-11-09.json',
                 '992_ops_maxproj_api_arrivals_2026-11-20.json'):
        entries.update(json.loads((ROOT / 'tests/data/release_contracts' / name).read_text()))
    return entries


def test_external_adapter_changes_only_its_exact_symbols_and_is_idempotent():
    """Retain every unrelated arrival, the input mapping and the historical bodies."""
    entries = _historical_external_arrivals()
    original = dict(entries)
    projected = _current_external_maxproj_api(entries)
    assert projected.keys() == entries.keys()
    assert entries == original
    assert {key for key in entries if entries[key] != projected[key]} == SYMBOLS
    for symbol in SYMBOLS:
        assert entries[symbol] == _external_api_sources()[symbol]['before']
        assert projected[symbol] == _external_api_sources()[symbol]['source']
    assert _current_external_maxproj_api(projected) == projected


def test_external_adapter_composes_with_existing_never_calibration_revisions():
    """Keep the prior 1014 adapter and the external revisions independent and composable."""
    from tests.test_ops_fdr_never_1014 import _current_never_api

    entries = _historical_external_arrivals()
    first = _current_external_maxproj_api(_current_never_api(entries))
    second = _current_never_api(_current_external_maxproj_api(entries))
    assert first == second
    assert first['spacr.ops_reference.embed_gene_profiles'] == _external_api_sources()[
        'spacr.ops_reference.embed_gene_profiles']['source']


def test_external_adapter_refuses_unbound_third_bodies():
    """Reject unknown prose instead of weakening an arrival-body assertion."""
    import pytest

    with pytest.raises(AssertionError):
        _current_external_maxproj_api({'spacr.ops_maxproj.maxproj_scores': 'unbound third body'})
