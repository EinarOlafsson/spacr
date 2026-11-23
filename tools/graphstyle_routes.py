"""Inspect function-level plotting routes without importing or executing the application.

The inventory ratchet resolves lexical imports and inherited self/cls calls.
Each nested body owns its own calls: a styled nested helper cannot silently
certify a separate drawing operation in its enclosing function.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path

PRIMITIVES = {'Figure', 'subplots', 'subplot', 'add_subplot', 'figure', 'plot', 'scatter', 'bar', 'barh',
              'boxplot', 'violinplot', 'stripplot', 'swarmplot', 'hist', 'histplot', 'hist2d', 'hexbin',
              'imshow', 'pcolormesh', 'matshow', 'fill_between', 'fill_betweenx', 'errorbar', 'stackplot',
              'pie', 'venn2', 'savefig', 'save_figure', 'publish', 'add_curve', 'add_scatter', 'add_group_mark',
              'add_bars', 'PlotWidget', 'PlotDataItem', 'ScatterPlotItem', 'ImageItem', 'BarGraphItem'}


def owned_nodes(node):
    """Walk lexical executable statements without attributing nested definitions to their parent."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        yield from owned_nodes(child)


def aliases_in(node, inherited, package):
    """Resolve import aliases in their lexical scope without importing spaCR or evaluating expressions."""
    aliases = dict(inherited)
    for child in owned_nodes(node):
        if isinstance(child, ast.Import):
            for alias in child.names:
                aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(child, ast.ImportFrom):
            module = importlib.util.resolve_name('.' * child.level + (child.module or ''), package) if child.level else child.module
            for alias in child.names:
                if alias.name != '*':
                    aliases[alias.asname or alias.name] = module + '.' + alias.name
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            aliases.pop(argument.arg, None)
        for argument in (node.args.vararg, node.args.kwarg):
            if argument is not None:
                aliases.pop(argument.arg, None)
        for child in owned_nodes(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                aliases.pop(child.id, None)
    return aliases


def parse_file(path, root):
    """Retain each function and module body with source hashes and independent import/call evidence."""
    relative = path.relative_to(root)
    module = '.'.join(relative.with_suffix('').parts)
    if path.name == '__init__.py':
        module = module.removesuffix('.__init__')
    package = module if path.name == '__init__.py' else module.rsplit('.', 1)[0]
    syntax = ast.parse(path.read_text())
    functions = {}
    classes = {}

    def visit(node, prefix, inherited, class_id=None):
        """Carry exact lexical identities through classes and nested functions."""
        aliases = aliases_in(node, inherited, package)
        if not isinstance(node, ast.ClassDef):
            identity = prefix + '.<module>' if isinstance(node, ast.Module) else prefix
            functions.setdefault(identity, []).append({'function': identity, 'file': str(relative),
                'line': getattr(node, 'lineno', 1), 'syntax_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
                'aliases': aliases, 'class': class_id,
                '_calls': [child.func for child in owned_nodes(node) if isinstance(child, ast.Call)],
                'scope': prefix, 'module': module})
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, prefix + '.' + child.name, aliases, class_id)
            elif isinstance(child, ast.ClassDef):
                name = prefix + '.' + child.name
                classes[name] = {'bases': [ast.unparse(base) for base in child.bases], 'aliases': aliases, 'module': module}
                visit(child, name, aliases, name)
            elif not isinstance(child, ast.Lambda):
                visit_definitions(child, prefix, aliases, class_id)

    def visit_definitions(node, prefix, aliases, class_id):
        """Find conditional definitions without creating artificial scopes for control flow."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raise AssertionError('Definition requires its own lexical identity')
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, prefix + '.' + child.name, aliases, class_id)
            elif isinstance(child, ast.ClassDef):
                name = prefix + '.' + child.name
                classes[name] = {'bases': [ast.unparse(base) for base in child.bases], 'aliases': aliases, 'module': module}
                visit(child, name, aliases, name)
            elif not isinstance(child, ast.Lambda):
                visit_definitions(child, prefix, aliases, class_id)

    visit(syntax, module, {})
    return functions, classes


def resolve(expression, record, known):
    """Resolve aliases and lexical/self targets; retain unresolved receivers as explicit evidence."""
    raw = ast.unparse(expression)
    parts = raw.split('.')
    aliases = record['aliases']
    if parts[0] in ('self', 'cls') and record['class']:
        return record['class'] + '.' + '.'.join(parts[1:])
    scope = record['scope']
    while scope.startswith(record['module']):
        candidate = scope + '.' + raw
        if candidate in known:
            return candidate
        if scope == record['module']:
            break
        scope = scope.rsplit('.', 1)[0]
    if parts[0] in aliases:
        return aliases[parts[0]] + ('.' + '.'.join(parts[1:]) if len(parts) > 1 else '')
    return raw



def scan_tree(root):
    """Return exact function bodies, class bindings and source-only resolved calls for a package tree."""
    root = Path(root)
    functions, classes = {}, {}
    for path in sorted((root / 'spacr').rglob('*.py')):
        found, defined = parse_file(path, root)
        for identity, records in found.items():
            functions.setdefault(identity, []).extend(records)
        classes.update(defined)
    known = set(functions) | set(classes)
    for records in functions.values():
        for record in records:
            calls = record.pop('_calls')
            record['calls'] = [{'raw': ast.unparse(call), 'resolved': resolve(call, record, known)} for call in calls]
            record['primitives'] = [call for call in record['calls'] if call['raw'].rsplit('.', 1)[-1] in PRIMITIVES]
    return functions, classes


def resolve_method(identity, functions, classes, seen=()):
    """Bind an inherited self/cls method to its actual source implementation without importing its class."""
    if identity in functions or identity in seen:
        return identity
    parent, separator, member = identity.rpartition('.')
    if not separator or parent not in classes:
        return identity
    record = classes[parent]
    for base in record['bases']:
        parts = base.split('.')
        head = record['aliases'].get(parts[0], record['module'] + '.' + parts[0])
        base_id = head + ('.' + '.'.join(parts[1:]) if len(parts) > 1 else '')
        candidate = resolve_method(base_id + '.' + member, functions, classes, (*seen, identity))
        if candidate in functions:
            return candidate
    return identity


def call_graph(functions, classes):
    """Retain all variants of each lexical function and bind only real, resolvable source callees."""
    return {identity: [tuple(dict.fromkeys(resolve_method(call['resolved'], functions, classes)
                            for call in record['calls'])) for record in records]
            for identity, records in functions.items()}


def routed_functions(graph, owners):
    """Find functions whose every source variant reaches an explicit shared owner.

    Reachability is a static routing guard, paired with runtime geometry,
    colour, opacity, data-neutrality and replay tests. It does not certify
    that every conditional execution path has produced a compliant figure.
    """
    reached = set(owners)
    while True:
        added = {identity for identity, variants in graph.items()
                 if identity not in reached and variants
                 and all(any(target in reached for target in calls) for calls in variants)}
        if not added:
            return reached
        reached.update(added)
