from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess

import build_documentation_i18n as api
import build_i18n_catalogs as runtime

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
baseline = scratch / 'foundation-api-refresh-baseline-r1'
baseline.mkdir(exist_ok=False)
shutil.copytree('docs/source/_static/i18n/api', baseline / 'api')
shutil.copytree('spacr/qt/i18n_catalogs', baseline / 'runtime')
shutil.copytree('docs/i18n/reviewed/api', baseline / 'reviewed-api')
shutil.copytree('docs/i18n/reviewed/runtime', baseline / 'reviewed-runtime')
old_source = subprocess.check_output(['git', 'show', 'HEAD:spacr/embeddings.py'])
(baseline / 'embeddings.py').write_bytes(old_source)
old = json.loads((baseline / 'api/en.json').read_text())['symbols']
current = api.public_docstrings()
changed = [key for key, source in current.items() if old[key]['text'] != source]
assert changed == ['spacr.embeddings.encoder_entry'], changed
blocks = api.translatable_blocks(current[changed[0]])
print(json.dumps(dict(enumerate(blocks)), ensure_ascii=False, indent=2), flush=True)
old_tree, tree = ast.parse(old_source), ast.parse(Path('spacr/embeddings.py').read_text())
old_nodes = {node.name: ast.dump(node) for node in old_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
new_nodes = {node.name: ast.dump(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
assert set(old_nodes) == set(new_nodes)
assert {key for key in old_nodes if old_nodes[key] != new_nodes[key]} == {
    '_retrieval_scorecard', '_weights_on_disk', 'encoder_entry'}
old_runtime = __import__('runpy').run_path(str(baseline / 'runtime/en.py'))['UI_SOURCES']
sources = runtime.canonical_sources()
added = sorted(set(sources['ui']) - set(old_runtime))
retired = sorted(set(old_runtime) - set(sources['ui']))
assert len(added) == 3 and not retired, (added, retired)
print('NEW RUNTIME:', added, flush=True)
report = {'api_changed': changed, 'api_blocks': dict(enumerate(blocks)),
          'runtime_added': added, 'runtime_retired': retired,
          'only_three_existing_functions_changed': True,
          'all_other_inference_functions_AST_exact': True,
          'source_sha256': hashlib.sha256(Path('spacr/embeddings.py').read_bytes()).hexdigest()}
(scratch / 'foundation-api-source-delta-r1.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('PASS source delta and all inference function preservation', flush=True)
