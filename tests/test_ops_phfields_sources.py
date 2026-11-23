"""Bind the staged phenotype queue error to its exact public API revision."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SYMBOL = 'spacr.ops_run_infra.FieldStager.path'
CHANGE = ROOT / 'tests/data/release_contracts/phfields_deadlock_api_change_2026-11-22.json'
HISTORICAL = ROOT / 'tests/data/release_contracts/884_911_bmc_paper_items_2_api_arrivals_2026-11-10.json'


def _phfields_change(source=None):
    """Validate only the admitted method body, signature and exact before/current hashes."""
    records = json.loads(CHANGE.read_text(encoding='utf-8'))['symbols']
    assert records.keys() == {SYMBOL}
    record = records[SYMBOL]
    assert record['file'] == 'spacr/ops_run_infra.py'
    if source is None:
        source = (ROOT / record['file']).read_text(encoding='utf-8')
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == 'FieldStager')
    node = next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == 'path')
    assert ast.get_docstring(node) == record['source']
    assert ast.unparse(node.args) == record['signature']
    assert ast.unparse(node.returns) == record['return_annotation']
    for key in ('before', 'source'):
        assert hashlib.sha256(record[key].encode()).hexdigest() == record[key + '_sha256']
    return record


def _current_phfields_api(entries):
    """Advance only the exact staging-method revision while retaining every other API body."""
    result = dict(entries)
    if SYMBOL in result:
        change = _phfields_change()
        assert result[SYMBOL] in (change['before'], change['source'])
        result[SYMBOL] = change['source']
    return result


def test_phfields_api_revision_preserves_all_other_historical_bodies():
    """Keep the 376 original entries immutable and advance precisely one documented method."""
    change = _phfields_change()
    original = json.loads(HISTORICAL.read_text(encoding='utf-8'))
    assert len(original) == 376
    assert original[SYMBOL] == change['before']
    current = _current_phfields_api(original)
    assert original.keys() == current.keys()
    assert {key for key in original if original[key] != current[key]} == {SYMBOL}
    assert current[SYMBOL] == change['source']
    assert original[SYMBOL] == change['before']
    assert _current_phfields_api(current) == current


@pytest.mark.parametrize('old,new', [
    ('def path(self, site: int)', 'def path(self, field: int)'),
    ('def path(self, site: int)', 'def path(self, site: str)'),
    ('def path(self, site: int) -> Tuple[Optional[str], list]:',
     'def path(self, site: int) -> str:'),
    ('Wait for one field to be staged.', 'Changed unreviewed method documentation.'),
])
def test_phfields_api_contract_rejects_unreviewed_source_drift(old, new):
    """An unchanged receipt cannot hide a parameter, annotation or documentation change."""
    source = (ROOT / 'spacr/ops_run_infra.py').read_text(encoding='utf-8')
    assert old in source
    with pytest.raises(AssertionError):
        _phfields_change(source.replace(old, new, 1))


def test_phfields_api_adapter_rejects_an_unbound_historical_body():
    """Only the exact prior or admitted current method body can enter the projection."""
    with pytest.raises(AssertionError):
        _current_phfields_api({SYMBOL: 'unreviewed historical source'})
