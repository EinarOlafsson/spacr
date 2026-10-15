import ast
import hashlib
import json
from pathlib import Path

root = Path('/mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015')
folder = Path(__file__).parent
packet = json.loads((folder / 'source-pins.json').read_text())
changed = [root / item['file'] for item in packet['files'] if item['file'].endswith('.py')]
base = root / 'tools/tutorials'
cache = {}
def parsed(path):
    if path not in cache:
        data = path.read_bytes()
        cache[path] = (ast.parse(data), hashlib.sha256(data).hexdigest())
    return cache[path][0]
def module_path(name, origin, level=0):
    if level:
        parent = origin.parent
        for _ in range(level - 1):
            parent = parent.parent
        options = [parent / (name.replace('.', '/') + '.py'), parent / name.replace('.', '/') / '__init__.py']
    else:
        options = [root / (name.replace('.', '/') + '.py'), base / (name.replace('.', '/') + '.py'), base / 'tests' / (name.replace('.', '/') + '.py')]
    return next((p for p in options if p.is_file()), None)
def bindings(path):
    names = set()
    def walk(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name): names.add(x.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split('.')[0] for a in n.names)
            elif isinstance(n, ast.If):
                walk(n.body); walk(n.orelse)
            elif isinstance(n, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
                walk(n.body); walk(n.orelse); walk(n.finalbody)
                for h in n.handlers: walk(h.body)
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                walk(n.body)
    walk(parsed(path).body)
    return names
rows = []
missing = []
for path in changed:
    tree = parsed(path)
    owners = {}
    def owner_walk(node, owner):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = owner + '.' + node.name if owner else node.name
        owners[id(node)] = owner or '<module>'
        for child in ast.iter_child_nodes(node): owner_walk(child, owner)
    owner_walk(tree, '')
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom) or not n.module or n.module == '__future__': continue
        target = module_path(n.module, path, n.level)
        if target is None: continue
        exported = bindings(target)
        for alias in n.names:
            if alias.name == '*': continue
            row = {'importer': str(path.relative_to(root)), 'line': n.lineno, 'owner': owners[id(n)], 'module': n.module, 'symbol': alias.name, 'target': str(target.relative_to(root)), 'exists_as_source_binding': alias.name in exported}
            if alias.name not in exported:
                local = alias.asname or alias.name
                row['use_lines'] = sorted({x.lineno for x in ast.walk(tree) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id == local})
                missing.append(row)
            rows.append(row)
# Direct existing callers that import any changed Python module are checked too.
changed_modules = {p.stem: p for p in changed}
caller_rows = []
for path in sorted(base.rglob('*.py')):
    if path in changed: continue
    tree = parsed(path)
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom) or not n.module: continue
        target = module_path(n.module, path, n.level)
        if target not in changed: continue
        names = bindings(target)
        for a in n.names:
            if a.name == '*': continue
            row = {'caller': str(path.relative_to(root)), 'line': n.lineno, 'symbol': a.name, 'target': str(target.relative_to(root)), 'exists_as_source_binding': a.name in names}
            caller_rows.append(row)
            if a.name not in names: missing.append(row)
result = {'mode': 'AST-only local literal from-import dependency census; no module imports/tests/browser/model/media/network', 'five_changed_python_files': [str(p.relative_to(root)) for p in changed], 'changed_file_local_imports': rows, 'direct_existing_callers': caller_rows, 'missing': missing, 'parsed_module_hashes': {str(p.relative_to(root)): h for p, (_, h) in cache.items()}, 'limits': 'Checks literal local from-import names and direct existing callers; dynamic imports/JS runtime dependencies are not recertified. No synthetic context or execution.'}
(folder / 'easy-local-dependency-census.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'local_import_bindings': len(rows), 'existing_caller_bindings': len(caller_rows), 'missing': missing}, indent=2))
