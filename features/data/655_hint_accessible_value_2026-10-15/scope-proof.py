import ast, hashlib, io, json, subprocess, sys, tokenize
from pathlib import Path
repo = Path(__file__).resolve().parents[0] / 'unused'
repo = Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015')
scratch = Path(__file__).resolve().parent
sys.path.insert(0, str(repo / 'tools'))
from api_visibility import public_symbol
from nested_helper_docs import inventory
path = 'spacr/qt/widgets/hint_bar.py'
module = 'spacr.qt.widgets.hint_bar'
base = 'a8545a4ab9'
before = subprocess.check_output(['git', 'show', base + ':' + path], cwd=repo)
after = (repo / path).read_bytes()
a, b = ast.parse(before), ast.parse(after)
comments = lambda raw: [t.string for t in tokenize.tokenize(io.BytesIO(raw).readline) if t.type == tokenize.COMMENT]
assert comments(before) == comments(after)
old_nodes = {n.name:n for n in a.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
new_nodes = {n.name:n for n in b.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
new_names = []
for name, old in old_nodes.items():
 new = new_nodes[name]
 assert ast.get_docstring(old) == ast.get_docstring(new), name
 if name != '_HintResizeHandle':
  assert ast.dump(old) == ast.dump(new), name
  continue
 old_methods = {n.name:n for n in old.body if isinstance(n, ast.FunctionDef)}
 new_methods = {n.name:n for n in new.body if isinstance(n, ast.FunctionDef)}
 for method, node in old_methods.items():
  assert ast.get_docstring(node) == ast.get_docstring(new_methods[method]), method
  assert ast.dump(node.args) == ast.dump(new_methods[method].args), method
  if method not in {'__init__', 'keyPressEvent'}:
   assert ast.dump(node) == ast.dump(new_methods[method]), method
 for method in sorted(new_methods.keys() - old_methods.keys()):
  new_names.append(module + '.' + name + '.' + method)
for name in sorted(new_nodes.keys() - old_nodes.keys()):
 node = new_nodes[name]
 new_names.append(module + '.' + name)
 if isinstance(node, ast.ClassDef):
  new_names.extend(module + '.' + name + '.' + n.name for n in node.body if isinstance(n, ast.FunctionDef))
assert ast.get_docstring(a) == ast.get_docstring(b)
policy = {name:public_symbol(name) for name in new_names}
assert not any(policy.values())
helper_results = {}
for label, raw in [('before', before), ('after', after)]:
 root = scratch / ('inventory-' + label)
 target = root / path
 target.parent.mkdir(parents=True, exist_ok=True)
 target.write_bytes(raw)
 helper_results[label] = inventory(root, ignore_patterns=())
assert helper_results['before'] == helper_results['after'] == ()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
result = {'baseline_commit':base, 'baseline_source_sha256':sha(before), 'source_sha256':sha(after),
 'module_and_all_existing_docstrings_signatures_comments_unchanged':True,
 'all_HintBar_methods_and_mouse_methods_unchanged':True,
 'changed_existing_private_methods':['_HintResizeHandle.__init__','_HintResizeHandle.keyPressEvent'],
 'actual_normal_public_symbol_calls':policy, 'actual_scoped_nested_inventory':[],
 'scope':'Actual normal visibility policy and before/after one-file helper inventory; no full repository collector, runtime module imports or generated artifacts.'}
(scratch / 'scope-proof.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
