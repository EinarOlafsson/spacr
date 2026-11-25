"""Exact G1 API source, signature, birth and retirement bindings for historical consumers."""
import ast
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT/'tests/data/release_contracts/graphstyle_g1_api_sources_2026-11-23.json'
CONTRACT_SHA256 = 'fec22abc25c7a6801a6b37fdccf39ea3d63af059ac8f1d79a16f0e29f2e7c4ce'


def _graphstyle_api_contracts():
    """Read only the individually reviewed G1 prose and callable source contracts."""
    raw = CONTRACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CONTRACT_SHA256
    result = json.loads(raw)
    assert result['baseline'] == '4bb6cc7a3f80a77903d600a6380971468a83ade9'
    assert result['api_delta_counts'] == {'added':22,'removed':3,'changed':38}
    assert len(result['symbols']) == 63 and len(result['callables']) == 49
    return result


@lru_cache(maxsize=64)
def _graphstyle_lexical_nodes(source, qualified):
    """Locate every exact lexical definition variant without importing a producer."""
    found = []

    def visit(node, names):
        """Retain conditional definitions under their real function/class path."""
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            names = (*names,node.name)
        if names == qualified and isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            found.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child,names)

    visit(ast.parse(source),())
    return tuple(found)


def _graphstyle_source_signature(node):
    """Bind parameter kinds, defaults, annotations, class fields and constructor declarations."""
    from tools import build_documentation_i18n as documentation

    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
        return dict(kind=type(node).__name__,arguments=ast.unparse(node.args),
            returns=ast.unparse(node.returns) if node.returns is not None else None,
            decorators=[ast.unparse(item) for item in node.decorator_list])
    if isinstance(node,ast.ClassDef):
        children = tuple(documentation._module_scope_nodes(node.body))
        constructors = [item for item in children if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)) and item.name in ('__init__','__new__')]
        return dict(kind='ClassDef',bases=[ast.unparse(item) for item in node.bases],
            decorators=[ast.unparse(item) for item in node.decorator_list],
            fields=[ast.dump(item) for item in children if isinstance(item,(ast.Assign,ast.AnnAssign))],
            constructors=[dict(name=item.name,signature=_graphstyle_source_signature(item)) for item in constructors])
    return dict(kind='Module')


def _validate_graphstyle_api_source(symbol, record, source=None):
    """Validate exact canonical prose and complete declared signatures before any historical projection."""
    from tools import build_documentation_i18n as documentation

    source = (ROOT/record['file']).read_text() if source is None else source
    nodes = _graphstyle_lexical_nodes(source,tuple(record['qualified']))
    if record['change'] == 'removed':
        assert not nodes,symbol
        assert record['source'] is None
        assert hashlib.sha256(record['before'].encode()).hexdigest() == record['before_sha256']
        return None
    assert nodes,symbol
    assert [_graphstyle_source_signature(node) for node in nodes] == record['source_variants'],symbol
    documents = {}
    for node in nodes:
        text = documentation._autoapi_class_doc(node) if isinstance(node,ast.ClassDef) else documentation._clean_doc(node)
        documentation._merge_source_doc(documents,symbol,text)
    assert documents[symbol] == record['source'],symbol
    assert hashlib.sha256(documents[symbol].encode()).hexdigest() == record['source_sha256'],symbol
    return documents[symbol]


def _current_graphstyle_api(entries):
    """Compose only exact G1 revisions with immutable historical API arrival bodies."""
    result = dict(entries)
    for symbol,record in _graphstyle_api_contracts()['symbols'].items():
        if record['change'] != 'changed' or symbol not in result:
            continue
        assert result[symbol] in (record['before'],record['source']),symbol
        if result[symbol] == record['before']:
            assert hashlib.sha256(result[symbol].encode()).hexdigest() == record['before_sha256'],symbol
        result[symbol] = _validate_graphstyle_api_source(symbol,record)
    return result


def _retired_graphstyle_api(docs):
    """Restore only the three archived nested helpers for historical inventory arithmetic."""
    retired = {}
    for symbol,record in _graphstyle_api_contracts()['symbols'].items():
        if record['change'] != 'removed':
            continue
        assert symbol not in docs,symbol
        assert _validate_graphstyle_api_source(symbol,record) is None
        retired[symbol] = record['before']
    assert set(retired) == {'spacr.plot.spacrGraph.create_plot.'+name for name in ('_generate_tabels','_get_positions','_place_symbols')}
    return retired


def _graphstyle_callable_record(item):
    """Serialize every actual public-callable field without dropping signature or prose properties."""
    return {key:sorted(value) if isinstance(value,frozenset) else value for key,value in asdict(item).items()}


def _validated_graphstyle_callables(callables):
    """Check all 49 changed callable records before replaying the fixed pre-G1 signature boundary.

    Only this historical inventory projection removes the 13 births. Current
    source documentation and rendered API checks retain the actual new API.
    Every unlisted callable remains unchanged and accountable to the old ratchet.
    """
    records = _graphstyle_api_contracts()['callables']
    actual = {item.symbol:item for item in callables if item.symbol in records}
    assert actual.keys() == records.keys()
    result = []
    for item in callables:
        record = records.get(item.symbol)
        if record is None:
            result.append(item)
            continue
        assert _graphstyle_callable_record(item) == record['current'],item.symbol
        if record['before'] is not None:
            values = {key:frozenset(value) if isinstance(getattr(item,key),frozenset) else value for key,value in record['before'].items()}
            result.append(replace(item,**values))
    return result


def test_graphstyle_api_contract_binds_all_named_prose_signatures_and_retirements():
    """Require every declared G1 body, parameter default and retirement at its actual source, with no application imports."""
    records = _graphstyle_api_contracts()['symbols']
    for symbol,record in records.items():
        assert _validate_graphstyle_api_source(symbol,record) == record['source']
    before = {symbol:record['before'] for symbol,record in records.items() if record['change']=='changed'}
    current = _current_graphstyle_api(before)
    assert current == {symbol:records[symbol]['source'] for symbol in before}
    assert _current_graphstyle_api(current) == current
    assert len(_retired_graphstyle_api({})) == 3


def test_graphstyle_source_binding_rejects_parameter_default_type_value_and_prose_drift():
    """Changing a named parameter, an integer default to the same float, its value, or source prose must fail the exact named-source contract."""
    symbol = 'spacr.figures.style.condition_plot'
    record = _graphstyle_api_contracts()['symbols'][symbol]
    source = (ROOT/record['file']).read_text()
    for mutation in ('name','default_type','default_value','prose'):
        syntax = ast.parse(source)
        node = next(node for node in syntax.body if isinstance(node,ast.FunctionDef) and node.name=='condition_plot')
        if mutation == 'name':
            node.args.args[0].arg = 'unreviewed_parameter'
        elif mutation in ('default_type','default_value'):
            parameters = node.args.posonlyargs+node.args.args
            position = next(index for index,item in enumerate(parameters) if item.arg=='seed')
            index = position-len(parameters)+len(node.args.defaults)
            original = node.args.defaults[index].value
            assert isinstance(original,int)
            node.args.defaults[index] = ast.Constant(value=float(original) if mutation=='default_type' else original+1)
        else:
            node.body[0].value = ast.Constant(value='Changed unreviewed API prose.')
        import pytest

        with pytest.raises(AssertionError):
            _validate_graphstyle_api_source(symbol,record,ast.unparse(syntax))


def test_graphstyle_callable_projection_preserves_unlisted_records_and_rejects_changed_prose():
    """The historical projection can remove only its 13 declared births and restore its 36 exact revisions; every unlisted record remains intact and an unreviewed changed body is rejected."""
    import pytest
    from tests.test_docstring_correctness import _PublicCallable

    records = _graphstyle_api_contracts()['callables']
    current = []
    for record in records.values():
        values = {key:frozenset(value) if key in ('parameters','required_parameters','accepted_documented_parameters') else value
                  for key,value in record['current'].items()}
        current.append(_PublicCallable(**values))
    unknown = replace(current[0],symbol='spacr.unlisted.example')
    result = _validated_graphstyle_callables([*current,unknown])
    assert len(result) == 37 and result[-1] is unknown
    expected = {name:record['before'] for name,record in records.items() if record['before'] is not None}
    assert {item.symbol:_graphstyle_callable_record(item) for item in result[:-1]} == expected
    changed = [replace(current[0],docstring='Changed unreviewed source.'),*current[1:]]
    with pytest.raises(AssertionError):
        _validated_graphstyle_callables(changed)


def test_graphstyle_retirement_binding_rejects_a_reintroduced_helper():
    """The three retired nested helpers cannot silently return under an unchanged historical allowance."""
    import pytest

    symbol = 'spacr.plot.spacrGraph.create_plot._generate_tabels'
    record = _graphstyle_api_contracts()['symbols'][symbol]
    syntax = ast.parse((ROOT/record['file']).read_text())
    parent = next(node for node in ast.walk(syntax) if isinstance(node,ast.FunctionDef) and node.name=='create_plot')
    parent.body.append(ast.parse('def _generate_tabels():\n    """Reintroduced helper."""\n    return None\n').body[0])
    with pytest.raises(AssertionError):
        _validate_graphstyle_api_source(symbol,record,ast.unparse(syntax))
