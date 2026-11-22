"""Bind the chance-corrected recovery prose to its exact single-symbol API revision."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'tests/data/release_contracts/1012fix_complex_recovery_api_revision_2026-11-22.json'
SYMBOL='spacr.ops_clusters.complex_metrics'


@lru_cache(maxsize=1)
def _complex_recovery_api_source():
    """Validate source syntax and both immutable bodies without importing production code."""
    contract=json.loads(CONTRACT.read_text(encoding='utf-8'))
    assert set(contract['symbols'])=={SYMBOL}
    record=contract['symbols'][SYMBOL]
    tree=ast.parse((ROOT/record['file']).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='complex_metrics')
    assert ast.get_docstring(node)==record['source']
    assert ast.unparse(node.args)==record['signature']
    assert ast.unparse(node.returns)==record['return_annotation']
    historical=json.loads((ROOT/'tests/data/release_contracts'/contract['historical_arrival']).read_text())
    assert historical[SYMBOL]==record['before']
    for key in ('before','source'):
        assert hashlib.sha256(record[key].encode()).hexdigest()==record[key+'_sha256']
    return record


def _current_complex_recovery_api(entries):
    """Revise only the bound recovery body, accepting its original or already revised form."""
    result=dict(entries)
    if SYMBOL in result:
        record=_complex_recovery_api_source()
        assert result[SYMBOL] in (record['before'],record['source'])
        result[SYMBOL]=record['source']
    return result


def test_recovery_adapter_preserves_all_other_symbols_and_the_input():
    """Keep the historical arrival intact while revising exactly one current API body."""
    entries=json.loads((ROOT/'tests/data/release_contracts/912_917_bmc_paper_items_3_api_arrivals_2026-11-10.json').read_text())
    original=dict(entries)
    revised=_current_complex_recovery_api(entries)
    assert entries==original and revised.keys()==entries.keys()
    assert {key for key in entries if revised[key]!=entries[key]}=={SYMBOL}
    assert _current_complex_recovery_api(revised)==revised


def test_recovery_adapter_rejects_an_unbound_third_body():
    """Unknown documentation cannot silently bypass the exact revision contract."""
    import pytest

    with pytest.raises(AssertionError):
        _current_complex_recovery_api({SYMBOL:'unbound third body'})


def test_recovery_adapter_composes_with_the_external_projection_revision():
    """The independent recovery and projection owners compose in either order."""
    from tests.test_ops_maxproj_sources_1012 import _current_external_maxproj_api

    entries={SYMBOL:_complex_recovery_api_source()['before']}
    assert _current_complex_recovery_api(_current_external_maxproj_api(entries))==_current_external_maxproj_api(_current_complex_recovery_api(entries))
