import ast
import copy
import gzip
import hashlib
import json
import subprocess
import tokenize
from io import BytesIO
from pathlib import Path

root = Path.cwd()
scratch = Path('/mnt/wd4tb/scratch/f564-ui-controls-20261015')
out = root / 'features/data/564_diffusion_ui_controls_2026-10-15'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
name = 'spacr/qt/screens/settings_model.py'
old = subprocess.check_output(['git', 'show', f'HEAD:{name}'])
new = (root / name).read_bytes()
a, b = ast.parse(old), ast.parse(new)
new_keys = {'counterfactual_strength', 'counterfactual_structure_weight'}
layout = next(n.value for n in b.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == '_APP_CATEGORY_SPECS')
activation = next(value for key, value in zip(layout.keys, layout.values) if key.value == 'activation')
row = next(n for n in activation.elts if n.elts[0].value == 'Counterfactuals')
assert {n.value for n in row.elts[1].elts if n.value in new_keys} == new_keys
row.elts[1].elts = [n for n in row.elts[1].elts if n.value not in new_keys]
method = next(n for n in b.body if isinstance(n, ast.FunctionDef) and n.name == 'attach_api_tooltip')
blocks = [n for n in method.body if isinstance(n, ast.If) and ast.unparse(n.test) == "app_key == 'activation'"]
assert len(blocks) == 1
assert {n.args[0].value for n in ast.walk(blocks[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'setObjectName'} == {'ActivationCounterfactualStrength', 'ActivationCounterfactualStructureWeight'}
method.body.remove(blocks[0])
assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
comments = lambda raw: [t.string for t in tokenize.tokenize(BytesIO(raw).readline) if t.type == tokenize.COMMENT]
assert comments(old) == comments(new)
def signatures(tree):
    return sorted((n.name, ast.dump(n.args, include_attributes=False), ast.get_docstring(n, clean=False)) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
assert signatures(a) == signatures(ast.parse(new))
backend_source = (root / 'spacr/settings.py').read_bytes()
tree = ast.parse(backend_source)
tips = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'tooltips' for t in n.targets))
raw_tips = {k.value: ast.literal_eval(v) for k, v in zip(tips.keys, tips.values) if isinstance(k, ast.Constant) and k.value in new_keys}
labels = {'counterfactual_strength': 'Counterfactual strength', 'counterfactual_structure_weight': 'Counterfactual structure weight'}
packet = {
 'status': 'PRIVATE UI dependency ready; backend/source localization must be published coherently',
 'incoming_public': subprocess.check_output(['git', 'rev-parse', '663c1b4f0'], text=True).strip(),
 'backend_upstream': 'db58907754165054879fd083347d948c0a5380ae',
 'private_backend_adoption': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
 'ui_source_sha256': sha(new),
 'backend_file_sha256': {p: sha((root / p).read_bytes()) for p in ['spacr/settings.py', 'spacr/attribution.py', 'spacr/deep_spacr.py']},
 'exact_ui_changes': ['Add two existing backend setting keys to Activation Maps Counterfactuals heading', 'Literal object names in existing attach_api_tooltip metadata seam, shared by eager and lazy controls'],
 'controls': [
  {'key': 'counterfactual_strength', 'object_name': 'ActivationCounterfactualStrength', 'type': 'QDoubleSpinBox/float', 'default': 0.95},
  {'key': 'counterfactual_structure_weight', 'object_name': 'ActivationCounterfactualStructureWeight', 'type': 'QDoubleSpinBox/float', 'default': 0.0}
 ],
 'alpha': 'Existing backend ALPHA_FEATURES[564] setting identities gate the rows; the literal Qt object names identify their controls. No widget registry or preference gate change.',
 'api_eligibility': 'The complete original module AST is recovered by removing exactly the two category-key literals and new existing-function conditional. All function signatures/docstrings and comments are identical; no callable/property/public/helper arrival or API prose change is introduced by this UI patch.',
 'runtime_dependency': {
  'ui_patch_new_tr_literals': 0,
  'backend_setting_labels': labels,
  'backend_tooltips_raw': raw_tips,
  'canonical_tooltip_candidates': {k: ' '.join(v.removeprefix('(float) - ').split()) for k, v in raw_tips.items()},
  'qualification': 'Source-derived candidate labels/tooltips only; no full normal collector run here. Normal source-bound review/collection/generation belongs to coordinator/backend owner. No translation is invented or installed.'
 },
 'generated_dependency': ['Normal settings-flow outputs must include both new settings and their existing activation runner consumers', 'Backend API/source prose changes remain separately owned; this UI introduces no API prose target'],
 'limits': ['No model download, network, training, GPU or biological/phenotype acceptance', 'No publication, catalogs, API manifests, Help or settings-flow outputs changed', 'No standalone source publication while backend localization remains pending']
}
(out / 'dependency-packet.json').write_text(json.dumps(packet, indent=2) + '\n')
artifacts = {}
for stem in ['baseline', 'baseline-corrected', 'final']:
    for suffix in ['sh', 'log', 'xml', 'time', 'exit']:
        file = scratch / (stem + '.' + suffix)
        raw = file.read_bytes()
        zipped = suffix in ['log', 'xml']
        stored = gzip.compress(raw, mtime=0) if zipped else raw
        dest = file.name + ('.gz' if zipped else '')
        (out / dest).write_bytes(stored)
        artifacts[dest] = {'sha256': sha(stored), 'raw_sha256': sha(raw)}
raw = (scratch / 'initial-fixture.py').read_bytes()
(out / 'initial-fixture.py.gz').write_bytes(gzip.compress(raw, mtime=0))
receipt = {
 'status': 'PASS: real Qt controls, alpha visibility and settings roundtrip; private',
 'source_sha256': {name: sha(new), 'tests/qt/test_f564_diffusion_controls.py': sha((root / 'tests/qt/test_f564_diffusion_controls.py').read_bytes()), 'tests/qt/test_generative_counterfactuals_alpha_gate.py': sha((root / 'tests/qt/test_generative_counterfactuals_alpha_gate.py').read_bytes())},
 'initial_baseline': {'result': '3 failed in4.68s', 'meaning': 'One actual eager-name defect plus two fixture defects: lazy callback took one argument instead of two; visual title is uppercased. Raw failure and original fixture retained.'},
 'corrected_baseline': {'result': '2 failed in4.12s', 'meaning': 'Only the two corrected cases rerun: lazy editor name empty; actual new controls absent from Counterfactuals form. Eager original failure unchanged, not repeated.'},
 'final': {'result': '7 passed in5.07s', 'new_cases': 3, 'existing_alpha_cases': 4, 'exit': 0, 'wall_seconds': 8.07, 'peak_rss_kib': 1335576, 'cap': '4G', 'threads': 1, 'session': 41231, 'terminal_polled': True},
 'actual_acceptance': ['Normal Activation AppScreen has both controls in Counterfactuals and discovers their literal names', 'Existing alpha preference hides/shows/hides both rows; zero strength and nonzero weight remain in collected settings', 'Eager and lazy SettingsWidgets controls are floats, defaults0.95/0.0, and JSON save/restore retains explicit0.0/0.125', 'Four existing counterfactual registry/gate/viewer cases pass unchanged except the upstream backend key expansion'],
 'artifacts': artifacts,
 'publication': 'Held for owner source review and normal backend/runtime/generated closure; no whole-feature completion claim'
}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(out / 'seal.py').write_text(Path(__file__).read_text())
print(json.dumps({'ui_sha256': sha(new), 'new_api_prose': 0, 'new_ui_tr_literals': 0, 'tests': receipt['final']['result']}))
