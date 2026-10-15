import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
root = Path.cwd()
scratch = Path(__file__).resolve().parent
sys.path[:0] = [str(root / 'tools'), str(root), str(scratch)]
import api_visibility
import build_documentation_i18n as docs
import build_i18n_catalogs as runtime
import nested_helper_docs
from targets import TARGETS
from spacr.qt.screens.settings_model import _humanize, _strip_type_prefix
sha = lambda raw: hashlib.sha256(raw).hexdigest()
backend = 'db58907754165054879fd083347d948c0a5380ae'
files = ['spacr/settings.py', 'spacr/attribution.py', 'spacr/deep_spacr.py']
def old(ref, path):
    return subprocess.check_output(['git','show',f'{ref}:{path}'])
source_pins = {}
for path in files:
    raw = (root / path).read_bytes()
    assert raw == old(backend, path), path
    source_pins[path] = sha(raw)
keys = ['counterfactual_strength', 'counterfactual_structure_weight']
tree = ast.parse((root / 'spacr/settings.py').read_text())
tips = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'tooltips' for t in n.targets))
raw_tips = {k.value: ast.literal_eval(v) for k,v in zip(tips.keys,tips.values) if isinstance(k,ast.Constant) and k.value in keys}
rows = [{'table':'setting_labels','key':key,'source':_humanize(key)} for key in keys]
rows += [{'table':'setting_tooltips','key':key,'source':' '.join(_strip_type_prefix(raw_tips[key]).split())} for key in keys]
source_tree = ast.parse((root / 'spacr/attribution.py').read_text())
owners = ['_cf_structure_weight','_cf_diffusion_strength','_cf_structure_loss']
for owner in owners:
    node = next(n for n in source_tree.body if isinstance(n,ast.FunctionDef) and n.name == owner)
    literals = [n.args[0].value for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == 'tr' and n.args and isinstance(n.args[0],ast.Constant)]
    assert len(literals) == 1
    rows.append({'table':'ui','key':literals[0],'source':literals[0],'owner':owner})
for row in rows:
    row['source_sha256'] = sha(row['source'].encode())
assert len(rows) == 7
sources = {'setting_labels':{},'setting_tooltips':{},'ui':[]}
for row in rows:
    if row['table'] == 'ui': sources['ui'].append(row['source'])
    else: sources[row['table']][row['key']] = row['source']
sources['ui'] = tuple(sources['ui'])
inputs = scratch / 'inputs'
for language, values in TARGETS.items():
    assert len(values) == 7
    records = [{k:v for k,v in row.items() if k != 'owner'} | {'translation':target} for row,target in zip(rows,values)]
    destination = inputs / language / '2026-10-15-f564-diffusion-controls-home-b.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({'schema':1,'language':language,'records':records},ensure_ascii=False,indent=2)+'\n')
runtime.REVIEWED_RUNTIME_DIR = inputs
runtime.canonical_sources = lambda: sources
runtime._translate_batches = lambda *a, **k: (_ for _ in ()).throw(AssertionError('No model decoding in review-input acceptance'))
checks = {}
for language in runtime.MODEL_SPECS:
    try:
        result = runtime.reviewed_runtime_translations(language)
        expected = {row['source']:target for row,target in zip(rows,TARGETS[language])}
        assert result == expected
        checks[language] = {'status':'PASS','records':len(result)}
    except Exception as error:
        checks[language] = {'status':'FAIL','error':str(error)}
(scratch / 'input-check.json').write_text(json.dumps(checks,indent=2)+'\n')
(scratch / 'sources.json').write_text(json.dumps({'backend':backend,'source_files_sha256':source_pins,'sources':rows},indent=2)+'\n')
print(json.dumps({'normal_scoped_review_loaders':checks}),flush=True)
assert all(value['status'] == 'PASS' for value in checks.values())

scope_roots = {}
all_changed = []
for label, revision in [('before',backend+'^'),('after',backend)]:
    scope = scratch / ('api-' + label)
    (scope / 'spacr').mkdir(parents=True,exist_ok=True)
    for path in files:
        (scope / path).write_bytes(old(revision,path))
    scope_roots[label] = scope
for path in files:
    module = path[:-3].replace('/','.')
    def bodies(raw):
        result = {}
        def visit(nodes, chain):
            for n in nodes:
                if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    key = '.'.join([module,*chain,n.name])
                    result[key] = (n, ast.get_docstring(n,clean=False))
                    visit(n.body, [*chain,n.name])
        visit(ast.parse(raw).body,[])
        return result
    previous = bodies(old(backend+'^',path)); current = bodies(old(backend,path))
    for key,(node,body) in current.items():
        if body == previous.get(key,(None,None))[1]: continue
        public = api_visibility.public_symbol(key)
        normal_name = docs._is_visible_function_name(node.name,module_is_package=False) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) else not node.name.startswith('_')
        all_changed.append({'symbol':key,'public_symbol':public,'leaf_name_visible':normal_name,'source_doc_sha256':sha((body or '').encode()),'reason':'Private top-level helper or member of private top-level class'})
        assert not public
nested = {}
for label, scope in scope_roots.items():
    definitions = nested_helper_docs.inventory(scope, ignore_patterns=docs.AUTOAPI_IGNORE)
    modules = {entry.module for entry in definitions} & nested_helper_docs.ENABLED_MODULES
    entries = nested_helper_docs.entries(definitions,modules=modules)
    nested[label] = {entry.qualified_key: {'docstring':entry.docstring,'signatures':entry.signatures} for entry in entries}
assert nested['before'] == nested['after']
assert len(all_changed) == 6
visibility = {'scope':'Actual normal visibility functions and inventory/entries over exactly three frozen source files; no whole collector or runtime backend imports.', 'changed_doc_bodies':all_changed,'eligible_changed_blocks':0,'nested_before':len(nested['before']),'nested_after':len(nested['after']),'nested_source_signatures_docs_unchanged':True,'source_sha256':source_pins}
(scratch / 'api-visibility.json').write_text(json.dumps(visibility,indent=2)+'\n')
print(json.dumps({'changed_private_docs':len(all_changed),'eligible_api_blocks':0,'unchanged_enabled_nested_entries':len(nested['after'])}),flush=True)
