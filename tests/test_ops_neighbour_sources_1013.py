"""Exact source-only contracts for the 1013 shared bleed owner and API revisions."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'tests/data/release_contracts/1013_neighbour_api_changes_2026-11-22.json'
BIRTH = ROOT / 'tests/data/release_contracts/1013_neighbour_bleed_api_arrivals_2026-11-22.json'
ADDITION = 'spacr.ops_sbs.neighbour_bleed_flags'
REVISIONS = frozenset({
    'spacr.ops_decode.decode_rule', 'spacr.ops_decode.resolve_settings',
    'spacr.ops_prefix.prefix_cell_calls', 'spacr.ops_prefix.prefix_settings',
    'spacr.ops_sbs.vote_mapped_reads'})
FILES = frozenset({'spacr/ops_sbs.py', 'spacr/ops_decode.py', 'spacr/ops_prefix.py',
                   'spacr/ops_settings.py', 'spacr/settings.py'})


@lru_cache(maxsize=1)
def _neighbour_api_sources():
    """Bind six exact API bodies and signatures while retaining file provenance."""
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    assert contract['base'] == '299179d9530ac618d82f11cc823f3ad8abd1c001'
    assert contract['before_api_count'] == 14961 and contract['after_api_count'] == 14962
    records = contract['symbols']
    assert set(records) == REVISIONS | {ADDITION}
    assert set(contract['source_sha256']) == FILES
    trees = {}
    for symbol, record in records.items():
        module, name = symbol.rsplit('.', 1)
        file = module.replace('.', '/') + '.py'
        assert record['file'] == file and file in FILES
        if file not in trees:
            trees[file] = ast.parse((ROOT / file).read_text(encoding='utf-8'))
        node = next(node for node in trees[file].body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.get_docstring(node) == record['source'], symbol
        assert ast.unparse(node.args) == record['signature'], symbol
        assert (ast.unparse(node.returns) if node.returns else None) == record['return_annotation'], symbol
        assert hashlib.sha256(record['source'].encode()).hexdigest() == record['source_sha256'], symbol
        if symbol == ADDITION:
            assert record['change'] == 'addition' and record['before'] is None
            assert record['before_sha256'] is None and record['before_signature'] is None
        else:
            assert record['change'] == 'revision'
            assert hashlib.sha256(record['before'].encode()).hexdigest() == record['before_sha256'], symbol
    return records


def _neighbour_predecessor_source(symbol, record, source):
    """Validate the current revision before supplying its exact predecessor to an older gate."""
    assert symbol == 'spacr.ops_prefix.prefix_settings'
    current = _neighbour_api_sources()[symbol]
    assert record['source'] == current['before']
    node = next(item for item in ast.parse(source).body
                if isinstance(item, ast.FunctionDef) and item.name == 'prefix_settings')
    if ast.get_docstring(node) == record['source']:
        return source
    current_tree = ast.parse((ROOT / record['file']).read_text(encoding='utf-8'))
    current_node = next(item for item in current_tree.body
                        if isinstance(item, ast.FunctionDef) and item.name == 'prefix_settings')
    assert ast.dump(node, include_attributes=False) == ast.dump(current_node, include_attributes=False)
    assert ast.get_docstring(node) == current['source']
    assert ast.unparse(node.args) == current['signature'] == record['signature']
    expr = node.body[0]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:expr.lineno - 1])) + expr.col_offset
    end = sum(map(len, lines[:expr.end_lineno - 1])) + expr.end_col_offset
    return source[:start] + repr(record['source']) + source[end:]


def _current_neighbour_api(entries):
    """Apply only bound current bodies after the existing 1014 arrival adapter."""
    result = dict(entries)
    for symbol, record in _neighbour_api_sources().items():
        if symbol in result:
            allowed = (record['source'],) if record['before'] is None else (record['before'], record['source'])
            assert result[symbol] in allowed, symbol
            result[symbol] = record['source']
    return result


def _neighbour_birth_arrivals(existing=None):
    """Validate the single new public callable and refuse collisions with prior arrivals."""
    entries = json.loads(BIRTH.read_text(encoding='utf-8'))
    assert entries == {ADDITION: _neighbour_api_sources()[ADDITION]['source']}
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_neighbour_arrivals(callables):
    """Remove only the exact source-derived new callable from the frozen legacy boundary."""
    entries = _neighbour_birth_arrivals()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == {ADDITION}
    item = found[ADDITION]
    assert item.category == 'function' and item.exposure == 'autoapi'
    assert item.parameters == item.accepted_documented_parameters == {'cell', 'barcode', 'reads', 'neighbours'}
    assert item.variant_count == 1 and item.docless_variant_count == 0
    assert not item.accepts_arbitrary_keywords
    assert item.docstring == entries[ADDITION]
    return [item for item in callables if item.symbol not in entries]


def _historical_neighbour_arrivals():
    """Read the immutable BMC contracts containing the five revised public bodies."""
    result = {}
    for name in ('871_911_bmc_paper_items_api_arrivals_2026-11-09.json',
                 '918_923_bmc_paper_items_4_api_arrivals_2026-11-11.json'):
        result.update(json.loads((ROOT / 'tests/data/release_contracts' / name).read_text()))
    return result


def test_neighbour_adapter_changes_only_bound_revisions_after_existing_adapter():
    """Keep historical files and input maps unchanged while composing in the required order."""
    from tests.test_ops_fdr_never_1014 import _current_never_api

    original = _historical_neighbour_arrivals()
    entries = _current_never_api(original)
    before = dict(entries)
    result = _current_neighbour_api(entries)
    assert entries == before
    assert result.keys() == entries.keys()
    assert {key for key in entries if entries[key] != result[key]} == REVISIONS
    for symbol in REVISIONS:
        assert entries[symbol] == _neighbour_api_sources()[symbol]['before']
        assert result[symbol] == _neighbour_api_sources()[symbol]['source']
    assert _current_neighbour_api(result) == result
    assert original == _historical_neighbour_arrivals()


def test_neighbour_adapter_refuses_unbound_bodies_and_birth_collisions():
    """Reject arbitrary prose, absent birth bodies and a duplicate public symbol."""
    import pytest

    for symbol in REVISIONS | {ADDITION}:
        with pytest.raises(AssertionError):
            _current_neighbour_api({symbol: 'unbound third body'})
    with pytest.raises(AssertionError):
        _current_neighbour_api({ADDITION: None})
    with pytest.raises(AssertionError):
        _neighbour_birth_arrivals({ADDITION: 'already present'})


def test_new_callable_validator_retains_unrelated_entries_and_checks_parameters():
    """Accept the exact new declaration while retaining every unrelated inventory object."""
    from types import SimpleNamespace
    import pytest

    params = {'cell', 'barcode', 'reads', 'neighbours'}
    item = SimpleNamespace(symbol=ADDITION, category='function', exposure='autoapi',
        parameters=params, accepted_documented_parameters=params,
        variant_count=1, docless_variant_count=0, accepts_arbitrary_keywords=False,
        docstring=_neighbour_birth_arrivals()[ADDITION])
    other = SimpleNamespace(symbol='unrelated.source')
    assert _validated_neighbour_arrivals([other, item]) == [other]
    item.parameters = {'wrong'}
    with pytest.raises(AssertionError):
        _validated_neighbour_arrivals([item])


def test_predecessor_bridge_rejects_current_source_and_signature_drift():
    """An older contract cannot conceal drift in the admitted current prefix revision."""
    import pytest
    from tests.test_ops_fdr_never_1014 import _never_api_contracts, _validate_never_api_source

    symbol = 'spacr.ops_prefix.prefix_settings'
    record = _never_api_contracts()[symbol]
    source = (ROOT / record['file']).read_text()
    assert _validate_never_api_source(symbol, record, source) == record['source']
    unrelated = source + '\nUNRELATED_1013 = None\n'
    assert _validate_never_api_source(symbol, record, unrelated) == record['source']
    changed = source.replace('def prefix_settings(', 'def prefix_settings(unbound, ', 1)
    assert changed != source
    with pytest.raises(AssertionError):
        _validate_never_api_source(symbol, record, changed)
    changed = source.replace(_neighbour_api_sources()[symbol]['source'].splitlines()[0], 'Unbound third body.', 1)
    assert changed != source
    with pytest.raises(AssertionError):
        _validate_never_api_source(symbol, record, changed)
