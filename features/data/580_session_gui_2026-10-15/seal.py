import ast
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tokenize
import io

ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015')
OUT = Path('/mnt/wd4tb/scratch/f580-session-ui-20261015')
DEST = ROOT / 'features/data/580_session_gui_2026-10-15'
DEST.mkdir(parents=True, exist_ok=False)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
BASE = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert BASE.startswith('2bb17bb917')
paths = ['spacr/qt/screens/settings_model.py', 'spacr/qt/widgets/calibration_preview.py',
         'tests/qt/test_f580_session_preview_controls.py', 'tests/qt/test_intensity_calibration_alpha_gate.py']
packet = json.loads((OUT / 'dependency-packet.json').read_text())
initial_hashes = packet['source_sha256'].copy()

def docs_and_signatures(raw):
    result = {}
    def visit(nodes, scope):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                key = '.'.join([*scope, node.name])
                result[key] = {'doc': ast.get_docstring(node, clean=False),
                               'args': ast.dump(node.args) if hasattr(node, 'args') else None}
                visit(node.body, [*scope, node.name])
    tree = ast.parse(raw)
    result['module'] = ast.get_docstring(tree, clean=False)
    visit(tree.body, [])
    return result

for path in paths[:2]:
    pre_geometry = (OUT / 'api-after' / path).read_bytes()
    current = (ROOT / path).read_bytes()
    assert docs_and_signatures(pre_geometry) == docs_and_signatures(current)
    packet['source_sha256'][path] = sha(current)
    base = subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT)
    old_comments = [t.string for t in tokenize.tokenize(io.BytesIO(base).readline) if t.type == tokenize.COMMENT]
    new_comments = [t.string for t in tokenize.tokenize(io.BytesIO(current).readline) if t.type == tokenize.COMMENT]
    assert old_comments == new_comments
for path in ('spacr/settings.py', 'spacr/measure.py', 'spacr/intensity_rescale.py'):
    assert (ROOT / path).read_bytes() == subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT)
packet['api']['post_census_geometry_followup'] = {
    'all_owner_docstrings_and_definition_signatures_unchanged': True,
    'initial_scope_source_sha256': initial_hashes,
    'meaning': 'Normal scoped inventory ran before the small form-wrapping followup; final AST comparison proves no document/signature/definition arrival or source-message change.'}
packet['pending'] = 'Independent source peer; eighteen-message runtime and one-block API authoring/normal generation, exact old-review retirements, derived artifacts and coherent backend-owner publication. The reproduced viewport defect is repaired and both original theme assertions now pass.'
(DEST / 'dependency-packet.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2) + '\n')
for name in ('baseline', 'final', 'themes', 'geometry-final'):
    assert (OUT / f'{name}.exit').exists()
    for suffix in ('log', 'xml'):
        (DEST / f'{name}.{suffix}.gz').write_bytes(gzip.compress((OUT / f'{name}.{suffix}').read_bytes(), mtime=0))
    for suffix in ('sh', 'time', 'exit'):
        shutil.copy2(OUT / f'{name}.{suffix}', DEST / f'{name}.{suffix}')
for name in ('dependencies.py', 'dependencies.log', 'dependencies-initial.py', 'dependencies-initial.log'):
    shutil.copy2(OUT / name, DEST / name)
assert (OUT / 'geometry-final.exit').read_text().strip() == '0'
assert '14 passed in 20.22s' in (OUT / 'geometry-final.log').read_text()
assert '2 failed in 7.55s' in (OUT / 'themes.log').read_text()
assert '7 failed, 1 passed in 5.04s' in (OUT / 'baseline.log').read_text()
theme_path = 'tests/qt/test_calibration_gain_preview_f580.py'
assert (ROOT / theme_path).read_bytes() == subprocess.check_output(['git', 'show', f'{BASE}:{theme_path}'], cwd=ROOT)
receipt = {
    'status': 'Private GUI implementation and bounded owning checks PASS; localization/API publication still pending',
    'base_commit': BASE, 'original_backend_commit': '0d81431062978ca6db5fb731d95bb97ed3b819e6',
    'source_test_sha256': {path: sha((ROOT / path).read_bytes()) for path in [*paths, theme_path]},
    'backend_owner_files_unmodified_after_private_adoption': True,
    'new_python_modules': 0, 'existing_production_comments_unchanged': True,
    'baseline': {'session': 40029, 'exit': 1, 'result': '7 failed, 1 passed in 5.04 s',
        'failures': 'Missing path actions/literal names, omitted snapshot inputs, missing invalidation, and version-2 plan indexing. Legacy plate rows pass.'},
    'initial_owning': {'session': 42864, 'exit': 0, 'result': '11 passed in 6.01 s'},
    'requested_theme_reproduction': {'session': 87489, 'exit': 1, 'result': '2 failed in 7.55 s',
        'exact_example': 'viewport QRect(0,0,342,706); Preview QRect(276,616,74,40), 8 px right overflow',
        'meaning': 'Unchanged actual dark/light containment assertions reproduced the CI family; no assertion relaxation.'},
    'final': {'session': 83886, 'exit': 0, 'result': '14 passed in 20.22 s',
        'cases': '10 focused session GUI/preview cases, 2 existing alpha cases, 2 exact dark/light preview cases',
        'resources': '4G cap, one thread, hidden CUDA, offscreen, private QSettings/config, shared heavy.lock; exact wall/RSS in geometry-final.time'},
    'geometry_repair': 'Only the calibration Preview editor makes its own QFormLayout wrap long rows when mounted. Field and button geometry then fit the original viewport; minimum width remains zero. No shared Section/AppScreen changes.',
    'behavior': [
        'Four exact backend setting keys are exposed under the existing alpha category; eager/lazy controls get the registered literal names.',
        'CSV file and reference-folder choosers preserve typed paths and cancellation; column/reference session remain editable exact text, with no mapping or input reads.',
        'All four values reach detached preview snapshots; each input edit clears gains, and an old worker cannot publish after a session input edit.',
        'Version-2 plans use sessions/reference_session; legacy version-1 plate/reference-plate rows and gain formatting remain intact.',
        'Headers switch back to plates for a subsequent legacy plan; mixed report identities are labelled explicitly without guessing.'
    ],
    'dependency_counts': {'runtime_messages': 18, 'runtime_targets': 162, 'old_tooltip_review_rows': 36,
                          'changed_api_module_blocks': 1, 'excluded_private_bodies': 3,
                          'unchanged_enabled_nested_entries': 14},
    'dependency_harness_correction': 'Initial metadata census mistakenly looked for UI instead of canonical UI_SOURCES in en.py; retained failure corrected only that reader. The normal helper visibility and source logic are unchanged.',
    'limits': ['No backend 23-case or F564 cohort replay.', 'Inert planner tests cover UI handoff/lifetime; the two explicitly requested legacy theme cases use their existing small synthetic real-planner fixture.',
               'No known-exposure bead acquisition, new scientific accuracy claim, GPU, model download/inference, native platform, full CI or release acceptance.']
}
receipt['evidence_sha256'] = {path.name: sha(path.read_bytes()) for path in sorted(DEST.iterdir())}
(DEST / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
shutil.copy2(__file__, DEST / 'seal.py')
print(json.dumps({'final_cases': 14, 'runtime_messages': 18, 'api_blocks': 1, 'unchanged_backend_files': 3}))
