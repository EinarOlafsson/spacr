"""The three item-1014 API paragraphs render without docutils warnings and retain exact source contracts."""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = frozenset('spacr.ops_decode.' + name for name in (
    'plate_fdr_summary', 'threshold_settings', 'write_well_reports'))
CHANGE = ROOT / 'tests/data/release_contracts/docfix_1014_api_changes_2026-11-22.json'


def _docfix_changes():
    """Bind the three permitted formatting revisions to their exact source bodies and signatures."""
    records = json.loads(CHANGE.read_text(encoding='utf-8'))['symbols']
    assert records.keys() == SYMBOLS
    tree = ast.parse((ROOT/'spacr/ops_decode.py').read_text(encoding='utf-8'))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for symbol, record in records.items():
        node = functions[symbol.rsplit('.', 1)[-1]]
        assert record['file'] == 'spacr/ops_decode.py'
        assert ast.get_docstring(node) == record['source']
        assert ast.unparse(node.args) == record['signature']
        assert ast.unparse(node.returns) == record['return_annotation']
        for key in ('source', 'before'):
            assert hashlib.sha256(record[key].encode()).hexdigest() == record[key+'_sha256']
    return records


def _current_docfix_records(records):
    """Compose only the three formatting revisions after the existing 1014 source adapters."""
    result = dict(records)
    for symbol, change in _docfix_changes().items():
        if symbol not in result:
            continue
        previous = result[symbol]
        assert previous['source'] == change['before']
        assert previous['source_sha256'] == change['before_sha256']
        assert previous['signature'] == change['signature']
        assert previous['return_annotation'] == change['return_annotation']
        current = dict(previous)
        for key in ('source', 'source_sha256', 'blocks', 'layout', 'line'):
            current[key] = change[key]
        result[symbol] = current
    return result


def _reference_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Treat a Sphinx function reference as a literal while docutils checks paragraph structure."""
    from docutils import nodes

    return [nodes.literal(rawtext, text, **(options or {}))], []


def _docutils_warnings(text, monkeypatch):
    """Parse exactly the Napoleon-processed text while isolating the known Sphinx function role."""
    from docutils.core import publish_doctree
    from docutils.parsers.rst import roles
    from tests.test_every_docstring_is_valid_rst import _as_sphinx_sees_it

    monkeypatch.setattr(roles, '_roles', dict(roles._roles))
    roles.register_local_role('func', _reference_role)
    stream = io.StringIO()
    document = publish_doctree(_as_sphinx_sees_it(text), settings_overrides={
        'report_level': 2, 'halt_level': 6, 'warning_stream': stream,
        'file_insertion_enabled': False, 'raw_enabled': False})
    return stream.getvalue(), document


@pytest.mark.parametrize('symbol', sorted(SYMBOLS))
def test_three_revised_api_docstrings_render_without_any_docutils_warning(symbol, monkeypatch):
    """The nightly fixtures each emit an unexpected-unindent warning; all three corrected docstrings must emit exactly zero warnings."""
    record = _docfix_changes()[symbol]
    warnings, document = _docutils_warnings(record['source'], monkeypatch)
    assert warnings == '', warnings
    from docutils import nodes

    assert list(document.findall(nodes.system_message)) == []
    assert 'Manual never-barcode calibration' in record['source'] or 'Per-well never reports' in record['source'] or 'Never calibration writes' in record['source']


@pytest.mark.parametrize('symbol', sorted(SYMBOLS))
def test_the_committed_baseline_docstrings_reproduce_the_reported_unindent(symbol, monkeypatch):
    """The detector must reproduce the actual nightly warning for each immutable before body, rather than merely accept corrected text."""
    warnings, _ = _docutils_warnings(_docfix_changes()[symbol]['before'], monkeypatch)
    assert 'Field list ends without a blank line; unexpected unindent.' in warnings


def test_docfix_source_adapter_preserves_every_other_historical_contract():
    """Retain all original before bodies and signatures while advancing only the three exact current source records."""
    from tests.test_ops_fdr_never_1014 import _never_api_contracts

    previous = json.loads((ROOT/'tests/data/release_contracts/1014_never_api_sources_2026-11-22.json').read_text())['symbols']
    advanced = _current_docfix_records(previous)
    current = _never_api_contracts()
    assert previous.keys() == advanced.keys() == current.keys()
    for symbol, record in previous.items():
        if symbol not in SYMBOLS:
            assert advanced[symbol] == record
        else:
            assert advanced[symbol]['before'] == record['before']
            assert advanced[symbol]['signature'] == record['signature']
            assert advanced[symbol]['source'] == current[symbol]['source']
