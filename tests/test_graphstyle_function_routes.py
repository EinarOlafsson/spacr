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


def test_inventory_functions_and_new_creators_use_a_shared_route_or_an_explicit_shrinking_entry():
    """Retain all 1,607 reviewed identities, reject new bypasses and exception growth, and keep Home G2 removals in their own file; runtime style tests verify actual semantics and geometry separately."""
    import csv
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    folder = root / 'reports/graphstyle/routes'
    raw = (folder / 'initial_plan.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'cc41aef6994ecf2d4968b194564631cfc1d70e021c56bfbbe56f86910f079e79'
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
    graph = routes.call_graph(functions, classes)
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
        assert records, f'Review a removed or renamed inventoried function: {identity}'
        excluded = plan['non_plot'].get(identity)
        if excluded is not None:
            actual = [call for record in records for call in record['primitives']]
            assert actual == excluded['primitives'], f'Review a new drawing operation inside an exclusion: {identity}'
        elif canonical not in reached and identity not in deferred:
            uncovered.append(identity)
    assert not uncovered, 'Review and route plotting functions individually:\n' + '\n'.join(sorted(uncovered))
