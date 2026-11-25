"""Function routing guards pair source reachability with actual graph-style runtime checks."""
from tools import graphstyle_routes as routes


def test_style_routes_resolve_aliases_and_inherited_helpers_without_certifying_nested_or_rebound_calls(tmp_path):
    """Exact lexical and imported identities must reach the shared owner; merely mentioning it, putting it in a nested function, or replacing an imported alias must not count."""
    package = tmp_path / 'spacr'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'base.py').write_text('''from .figures.style import apply_axes_style as finish
class Base:
    def present(self, axes):
        """Apply the actual shared owner."""
        finish(axes, None)
''')
    (package / 'view.py').write_text('''import matplotlib.pyplot as painting
from .base import Base as Parent
from .figures import style as house
from .figures.style import apply_axes_style as finish
class View(Parent):
    def draw(self, axes):
        """Reach the inherited shared adapter."""
        axes.scatter([0], [1])
        self.present(axes)
def aliased(axes):
    """Reach a module imported under an alias."""
    painting.subplots()
    house.apply_axes_style(axes, None)
def uncalled_nested(axes):
    """A styled but uncalled nested body cannot certify this creator."""
    def local():
        """An independently owned nested body."""
        finish(axes, None)
    axes.scatter([0], [1])
def parameter_shadow(finish, axes):
    """An argument with the imported name is not the imported callable."""
    axes.plot([0], [1])
    finish(axes, None)
def assignment_shadow(axes):
    """A replaced import no longer names the shared callable."""
    finish = print
    axes.scatter([0], [1])
    finish('spacr.figures.style.apply_axes_style')
def local_shadow(axes):
    """A same-name nested callable must win over the module import."""
    def finish(*args):
        """Keep this deliberately unstyled."""
        return None
    axes.scatter([0], [1])
    finish(axes, None)
''')
    functions, classes = routes.scan_tree(tmp_path)
    reached = routes.routed_functions(routes.call_graph(functions, classes),
        {'spacr.figures.style.apply_axes_style'})
    assert {'spacr.view.View.draw', 'spacr.view.aliased', 'spacr.base.Base.present',
            'spacr.view.uncalled_nested.local'} <= reached
    assert not {'spacr.view.uncalled_nested', 'spacr.view.parameter_shadow',
                'spacr.view.assignment_shadow', 'spacr.view.local_shadow'} & reached
    assert [entry['raw'] for entry in functions['spacr.view.aliased'][0]['primitives']] == ['painting.subplots']
    assert functions['spacr.view.uncalled_nested'][0]['calls'] == [
        {'raw': 'axes.scatter', 'resolved': 'axes.scatter'}]


def test_every_conditional_function_variant_needs_a_route_and_cycles_do_not_hang():
    """A styled alternate definition cannot cover an unstyled one, and mutually recursive functions without an owner stay uncovered."""
    graph = {'all': [('owner',), ('wrapper',)], 'partial': [('owner',), ('print',)],
             'wrapper': [('owner',)], 'first': [('second',)], 'second': [('first',)]}
    reached = routes.routed_functions(graph, {'owner'})
    assert reached == {'owner', 'all', 'wrapper'}


def test_reviewed_support_cannot_gain_even_an_unrecognized_drawing_call(tmp_path):
    """A metadata getter is accepted only at its reviewed exact body; adding an unknown painter call must invalidate it even when the plotting-primitive scanner cannot name that operation."""
    import pytest

    package = tmp_path / 'spacr'
    package.mkdir()
    path = package / 'metadata.py'
    body = 'def page_size():\n    """Return only a page tuple."""\n    return (40., 40.)\n'
    path.write_text(body)
    functions, _classes = routes.scan_tree(tmp_path)
    identity = 'spacr.metadata.page_size'
    contracts = {identity: {'reason': 'Returns metadata only.', 'file': 'spacr/metadata.py',
        'syntax_sha256': [functions[identity][0]['syntax_sha256']],
        'calls': [functions[identity][0]['calls']]}}
    assert routes.reviewed_support_functions(functions, contracts) == {identity}
    path.write_text(body.replace('    return', '    painter.draw_custom_chart()\n    return'))
    changed, _classes = routes.scan_tree(tmp_path)
    assert not changed[identity][0]['primitives']
    with pytest.raises(ValueError, match='Review the changed support function'):
        routes.reviewed_support_functions(changed, contracts)


def test_reviewed_support_rejects_a_changed_import_target_even_with_the_same_function_body(tmp_path):
    """A module import can turn a harmless metadata factory into an arbitrary chart call without changing the getter AST or its primitive names; exact resolved-call bindings must reject that drift."""
    import pytest

    package = tmp_path/'spacr'
    package.mkdir()
    path = package/'metadata.py'
    body = 'from builtins import tuple as result\ndef page_size():\n    """Return page metadata."""\n    return result((40., 40.))\n'
    path.write_text(body)
    functions, _classes = routes.scan_tree(tmp_path)
    identity = 'spacr.metadata.page_size'
    original = functions[identity][0]
    contracts = {identity: {'reason': 'Returns page metadata.', 'file': 'spacr/metadata.py',
        'syntax_sha256': [original['syntax_sha256']], 'calls': [original['calls']]}}
    assert routes.reviewed_support_functions(functions, contracts) == {identity}
    path.write_text(body.replace('from builtins import tuple', 'from arbitrary_renderer import create_chart'))
    changed, _classes = routes.scan_tree(tmp_path)
    assert changed[identity][0]['syntax_sha256'] == original['syntax_sha256']
    assert not changed[identity][0]['primitives']
    with pytest.raises(ValueError, match='Review the changed support function'):
        routes.reviewed_support_functions(changed, contracts)


def test_caller_owned_adapters_do_not_certify_new_unstyled_callers_or_a_changed_host(tmp_path):
    """An exact primitive/host path may close an internal adapter, but a new call outside that host or removal of its finalizer must fail; the adapter itself never becomes a routed root."""
    import pytest

    package = tmp_path/'spacr'
    package.mkdir()
    path = package/'scene.py'
    body = 'from .figures.style import finalize_figure\ndef primitive(ax):\n    """Paint only in the caller-owned axes."""\n    ax.plot([0], [1])\ndef render(ax):\n    """Own and finalize the actual chart."""\n    primitive(ax)\n    finalize_figure(ax.figure)\n'
    path.write_text(body)
    functions, classes = routes.scan_tree(tmp_path)
    graph = routes.call_graph(functions, classes)
    helper, host = 'spacr.scene.primitive', 'spacr.scene.render'
    records = {name: [{key: record[key] for key in ('file','syntax_sha256','calls','primitives')}
                     for record in functions[name]] for name in (helper, host)}
    contracts = {helper: {'reason': 'Draws only in the finalized host.', 'host': host,
        'callers': [host], 'paths': [[host, helper]], 'records': records}}
    assert routes.reviewed_context_functions(functions, graph, contracts) == {helper}
    reached = routes.routed_functions(graph, {'spacr.figures.style.finalize_figure'})
    assert host in reached and helper not in reached
    path.write_text(body+'\ndef rogue(ax):\n    """An unstyled new caller."""\n    primitive(ax)\n')
    changed, classes = routes.scan_tree(tmp_path)
    changed_graph = routes.call_graph(changed, classes)
    assert 'spacr.scene.rogue' not in routes.routed_functions(changed_graph, {'spacr.figures.style.finalize_figure'})
    with pytest.raises(ValueError, match='Review the changed caller-owned graph adapter'):
        routes.reviewed_context_functions(changed, changed_graph, contracts)
    path.write_text(body.replace('    finalize_figure(ax.figure)', '    print("not finalized")'))
    changed, classes = routes.scan_tree(tmp_path)
    with pytest.raises(ValueError, match='Review the changed caller-owned graph adapter'):
        routes.reviewed_context_functions(changed, routes.call_graph(changed, classes), contracts)


def test_inventory_functions_and_new_creators_use_a_shared_route_or_an_explicit_shrinking_entry():
    """Retain all 1,607 reviewed identities, reject new bypasses and exception growth, and keep Home G2 removals in their own file; runtime style tests verify actual semantics and geometry separately."""
    import csv
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    folder = root / 'reports/graphstyle/routes'
    raw = (folder / 'initial_plan.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '58aba19d929e122ee86d9b4809fce86755a46ef97a39eff068c4fd4298ce6277'
    plan = json.loads(raw)
    assert plan['schema'] == 1 and plan['source_spec'] == '4f14c247cf'
    inventory = (folder / 'inventory.csv').read_bytes()
    assert hashlib.sha256(inventory).hexdigest() == plan['source_inventory_sha256']
    original = list(csv.DictReader(inventory.decode().splitlines()))
    identities = {record['function'] for record in original}
    assert len(original) == len(identities) == 1607
    deferred = {}
    for group in range(1, 6):
        current = json.loads((folder / f'deferred_g{group}.json').read_text())
        initial = plan['maximum_deferred'][str(group)]
        assert current.items() <= initial.items(), f'G{group} exceptions may only shrink, without replacement IDs or reasons'
        assert not deferred.keys() & current.keys()
        assert all(value['reason'].strip() and value['location'].startswith('spacr/') for value in current.values())
        deferred.update(current)
    functions, classes = routes.scan_tree(root)
    support_raw = (folder / 'support_contracts_g1.json').read_bytes()
    assert hashlib.sha256(support_raw).hexdigest() == 'a2df8faa3d8e17a765838b53297920e172c39f9a5e0b8a557c4a60d22ca332d5'
    contracts = json.loads(support_raw)
    assert contracts.keys() <= plan['maximum_deferred']['1'].keys()
    assert not contracts.keys() & deferred.keys()
    support = routes.reviewed_support_functions(functions, contracts)
    graph = routes.call_graph(functions, classes)
    context_raw = (folder / 'context_contracts_g1.json').read_bytes()
    assert hashlib.sha256(context_raw).hexdigest() == '5ebb1c188cd97b859321aaaf92de3d4111c2d70a845a8a24291446021aac58ef'
    context_contracts = json.loads(context_raw)
    assert context_contracts.keys() <= plan['maximum_deferred']['1'].keys()
    assert not context_contracts.keys() & (deferred.keys() | support)
    contexts = routes.reviewed_context_functions(functions, graph, context_contracts)
    assert all((root / path).is_file() for contract in context_contracts.values() for path in contract['validation_tests'])
    owners = set(plan['owners'])
    assert owners <= functions.keys(), 'Every explicit shared owner must still exist'
    reached = routes.routed_functions(graph, owners)
    for identity in plan['retired']:
        assert identity not in functions, f'Review the returned retired helper {identity}'
    uncovered = []
    for identity in identities | {name for name, records in functions.items()
            if any(record['primitives'] for record in records)}:
        canonical = identity.replace('.__init__.', '.')
        records = functions.get(canonical, [])
        if identity in plan['retired']:
            continue
        if canonical in support or canonical in contexts:
            continue
        assert records, f'Review a removed or renamed inventoried function: {identity}'
        excluded = plan['non_plot'].get(identity)
        if excluded is not None:
            actual = [call for record in records for call in record['primitives']]
            assert actual == excluded['primitives'], f'Review a new drawing operation inside an exclusion: {identity}'
        elif canonical not in reached and identity not in deferred:
            uncovered.append(identity)
    assert not uncovered, 'Review and route plotting functions individually:\n' + '\n'.join(sorted(uncovered))
