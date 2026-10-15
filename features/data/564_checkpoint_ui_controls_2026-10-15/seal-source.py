from pathlib import Path
import ast
import copy
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tokenize

root = Path.cwd()
scratch = Path(__file__).parent
sys.path.insert(0, str(root / 'tools'))
import build_documentation_i18n as builder
import nested_helper_docs as nested

rel = 'spacr/qt/screens/settings_model.py'
base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
before = subprocess.check_output(['git', 'show', base + ':' + rel])
after = (root / rel).read_bytes()
old, new = ast.parse(before), ast.parse(after)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
comments = lambda raw: [t.string for t in tokenize.tokenize(io.BytesIO(raw).readline)
                        if t.type == tokenize.COMMENT]
assert comments(before) == comments(after)

def named_docs(tree):
    result = {}
    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                key = '.'.join([*prefix, child.name])
                result[key] = ast.get_docstring(child, clean=False)
                walk(child, [*prefix, child.name])
            else:
                walk(child, prefix)
    walk(tree, [])
    return result

old_docs, new_docs = named_docs(old), named_docs(new)
new_method = 'SettingsWidgets._choose_counterfactual_checkpoint'
assert set(new_docs) - set(old_docs) == {new_method}
assert all(new_docs[key] == value for key, value in old_docs.items())
assert not builder._is_visible_function_name(new_method.rsplit('.', 1)[-1], module_is_package=False)

old_by_name = {n.name: n for n in old.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
normalized = copy.deepcopy(new)
for index, node in enumerate(normalized.body):
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == '_APP_CATEGORY_SPECS':
        old_value = next(n for n in old.body if isinstance(n, ast.AnnAssign)
                         and isinstance(n.target, ast.Name) and n.target.id == '_APP_CATEGORY_SPECS')
        old_parts = {k.value: v for k, v in zip(old_value.value.keys, old_value.value.values)}
        new_parts = {k.value: v for k, v in zip(node.value.keys, node.value.values)}
        assert old_parts.keys() == new_parts.keys()
        assert all(ast.dump(old_parts[k], include_attributes=False) == ast.dump(new_parts[k], include_attributes=False)
                   for k in old_parts if k != 'activation')
        expected = tuple(
            (title, keys + ('counterfactual_checkpoint', 'counterfactual_checkpoint_sha256'))
            if title == 'Counterfactuals' else (title, keys)
            for title, keys in ast.literal_eval(old_parts['activation']))
        assert ast.literal_eval(new_parts['activation']) == expected
        normalized.body[index] = copy.deepcopy(old_value)
    elif isinstance(node, ast.FunctionDef) and node.name == 'attach_api_tooltip':
        normalized.body[index] = copy.deepcopy(old_by_name[node.name])
    elif isinstance(node, ast.ClassDef) and node.name == 'SettingsWidgets':
        original_methods = {n.name: n for n in old_by_name[node.name].body if isinstance(n, ast.FunctionDef)}
        node.body = [copy.deepcopy(original_methods[n.name])
                     if isinstance(n, ast.FunctionDef) and n.name in {'_add_source_actions', '_canonical'} else n
                     for n in node.body
                     if not (isinstance(n, ast.FunctionDef) and n.name == '_choose_counterfactual_checkpoint')]
assert ast.dump(normalized, include_attributes=False) == ast.dump(old, include_attributes=False)

helper_bodies = []
for name, raw in [('before', before), ('after', after)]:
    snapshot = scratch / 'scoped-source' / name
    path = snapshot / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    entries = nested.entries(nested.inventory(snapshot, ignore_patterns=builder.AUTOAPI_IGNORE),
                             modules={'spacr.qt.screens.settings_model'})
    helper_bodies.append({entry.qualified_key: entry.docstring for entry in entries})
assert helper_bodies[0] == helper_bodies[1]

def tr_literals(tree):
    return {node.args[0].value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'tr'
            and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)}
assert tr_literals(old) == tr_literals(new)
assert 'Choose file…' in tr_literals(old)
catalog = ast.parse((root / 'spacr/qt/i18n_catalogs/en.py').read_text())
ui = next(ast.literal_eval(n.value.args[0]) for n in catalog.body if isinstance(n, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == 'UI_SOURCES' for t in n.targets))
assert 'Choose file…' in ui
assert '5 failed, 1 passed in 4.64s' in (scratch / 'before.log').read_text()
assert '7 passed in 3.84s' in (scratch / 'after.log').read_text()

test_rel = 'tests/qt/test_generative_counterfactuals_alpha_gate.py'
old_test = subprocess.check_output(['git', 'show', base + ':' + test_rel])
new_test = (root / test_rel).read_bytes()
expected_test = old_test.replace(b'"counterfactual_structure_weight")',
    b'"counterfactual_structure_weight", "counterfactual_checkpoint",\n        "counterfactual_checkpoint_sha256")')
assert expected_test == new_test

out = root / 'features/data/564_checkpoint_ui_controls_2026-10-15'
out.mkdir(parents=True, exist_ok=True)
for name in ['before.log', 'before.time', 'after.log', 'after.time']:
    (out / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
receipt = {
    'status': 'Private GUI adapter; coherent backend/settings/runtime/derived integration still required',
    'base_commit': base,
    'original_settings_prerequisite': '1e1f2a137516f9d9014b29dd8ce2c38e775f5b3f',
    'source_path': rel, 'before_sha256': sha(before), 'after_sha256': sha(after),
    'source_scope': ['Counterfactuals two-key category addition', 'Two literal object names',
                     'One trailing local-file picker action/private callback',
                     'Two-key empty-to-blank canonicalization for built and unopened forms'],
    'other_executable_ast_unchanged': True, 'comments_unchanged': True,
    'all_previous_docstrings_unchanged': len(old_docs),
    'new_private_method': new_method, 'new_method_normal_public_visibility': False,
    'normal_enabled_nested_bodies_unchanged': len(helper_bodies[0]),
    'api_arrivals': [], 'runtime_caption_arrivals': [],
    'reused_runtime_source': 'Choose file…',
    'settings_translation_owner': 'Original Home: two labels and two tooltips, 36 targets; not authored here',
    'baseline': '5 failed, 1 passed in 4.64s', 'final': '7 passed in 3.84s',
    'new_test': {'path': 'tests/qt/test_f564_checkpoint_controls.py',
                 'sha256': sha((root / 'tests/qt/test_f564_checkpoint_controls.py').read_bytes())},
    'existing_test_change': {'path': test_rel, 'before_sha256': sha(old_test),
                             'after_sha256': sha(new_test), 'scope': 'Only two exact new KEYS entries; registry assertion untouched'},
    'acceptance': ['Actual AppScreen Counterfactuals placement and alpha hide/show with retained values',
                   'Eager/lazy blank defaults, unbuilt collect, JSON roundtrip and cleared values',
                   'Real QAction with isolated file dialog transport: choice commits once; cancellation is inert',
                   'Path selection never alters entered SHA or generator; invalid imported identity remains explicit'],
    'limits': ['No file content reads, hashes, model loads, downloads or training in GUI path selection',
               'SHA/generator compatibility is enforced by the separately owned strict backend',
               'No complete F564 scientific/checkpoint availability or whole-item completion claim'],
}
(out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
(out / 'seal-source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'source_sha256': sha(after), 'normal_nested_bodies_unchanged': len(helper_bodies[0]),
                  'api_arrivals': 0, 'runtime_arrivals': 0, 'tests': receipt['final']}))
