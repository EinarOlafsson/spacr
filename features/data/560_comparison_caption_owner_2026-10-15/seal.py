import ast
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

root = Path.cwd()
scratch = Path('/mnt/wd4tb/scratch/f560-caption-owner-20261015')
evidence = root / 'features/data/560_comparison_caption_owner_2026-10-15'
evidence.mkdir(parents=True, exist_ok=True)
sha = lambda data: hashlib.sha256(data).hexdigest()
def old(commit, name):
    return subprocess.check_output(['git', 'show', f'{commit}:{name}'])

tool_name = 'tools/build_i18n_catalogs.py'
tool = (root / tool_name).read_bytes()
base_tool = old('b75d47950', tool_name)
before_ast, after_ast = ast.parse(base_tool), ast.parse(tool)
found = []
for node in ast.walk(after_ast):
    if isinstance(node, ast.Set) and any(isinstance(value, ast.Constant) and value.value == '_backbone_scorecards' for value in node.elts):
        names = [value.value for value in node.elts]
        assert set(names) == {'encoder_entry', '_mil_frame_from_db', '_mil_bags', '_backbone_scorecards'}
        node.elts = [value for value in node.elts if value.value != '_backbone_scorecards']
        found.append(names)
assert len(found) == 1
assert ast.dump(before_ast, include_attributes=False) == ast.dump(after_ast, include_attributes=False)
request = json.loads((root / 'features/data/560_comparison_input_boundary_2026-10-15/extractor-owner-request.json').read_text())
backend = (root / 'spacr/embeddings.py').read_bytes()
assert sha(backend) == request['source_sha256']
owner = next(n for n in ast.parse(backend).body if isinstance(n, ast.FunctionDef) and n.name == '_backbone_scorecards')
literals = {n.args[0].value for n in ast.walk(owner) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'tr' and n.args and isinstance(n.args[0], ast.Constant)}
assert literals == set(request['literal_tr_sources'])
test_name = 'tests/test_runtime_backend_caption_owners.py'
old_tests = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(old('b75d47950', test_name)).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
new_tests = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse((root / test_name).read_bytes()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
assert all(new_tests[name] == value for name, value in old_tests.items())
artifacts = {}
for name in ('baseline.sh', 'baseline.log', 'baseline.xml', 'baseline.time', 'baseline.exit', 'final.sh', 'final.log', 'final.xml', 'final.time', 'final.exit'):
    raw = (scratch / name).read_bytes()
    stored = gzip.compress(raw, mtime=0) if name.endswith(('.log', '.xml')) else raw
    dest = name + '.gz' if name.endswith(('.log', '.xml')) else name
    (evidence / dest).write_bytes(stored)
    artifacts[dest] = {'sha256': sha(stored), 'raw_sha256': sha(raw)}
receipt = {
 'status': 'PASS: private exact owner addition, no standalone publication',
 'base_integration': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
 'f560_source': '241723e559d90bad5b292c07744e13be1b2b2d53',
 'source_peer': subprocess.check_output(['git', 'rev-parse', 'aa2968ffb'], text=True).strip(),
 'source_sha256': {tool_name: sha(tool), test_name: sha((root / test_name).read_bytes()), 'spacr/embeddings.py': sha(backend)},
 'literal': sorted(literals),
 'selector': found[0],
 'proof': 'Restoring only the new owner constant recovers the complete b75 builder AST; every pre-existing owning test function AST is unchanged.',
 'baseline': {'result': '1 failed, 1 passed, 37 deselected', 'seconds': 1.34, 'exit': 1, 'failure': 'The production extractor omitted the exact new literal; unrelated-scope exclusion already passed.'},
 'final': {'result': '39 passed', 'seconds': 1.05, 'exit': 0, 'wall_seconds': 2.49, 'peak_rss_kib': 460508, 'cap': '2G', 'threads': 1, 'session': 49789, 'terminal_polled': True},
 'boundary': 'Production AST extractor over existing tiny-source fixture; indirect UI registry sources isolated by existing fixture. No canonical full collector, inference, model or installer execution; no API or runtime artifact generation.',
 'artifacts': artifacts,
}
(evidence / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(evidence / 'seal.py').write_text(Path(__file__).read_text())

refs = {
 'spacr/model_zoo.py': '983947450',
 'tests/test_f370_curated_builtin_scorecards.py': '983947450',
 'tools/build_score_cards.py': '7209aca03',
 'packaging/generate_readme_visuals.py': '7209aca03',
 'tools/build_model_cards.py': 'a3d3cc025',
 'tests/test_f370_scorecard_generator_bindings.py': 'a3d3cc025',
 'spacr/embeddings.py': 'aa2968ffb',
 'tests/test_f560_comparison_input_contract.py': 'aa2968ffb',
 'tests/test_f560_comparison_input_locales.py': 'aa2968ffb',
}
outputs = json.loads((root / 'features/data/370_generator_scorecard_artifacts_2026-10-15/receipt.json').read_text())['outputs']
for name, binding in outputs.items():
    assert sha(old('f8260ead', name)) == binding['before_sha256'], name
    assert sha((root / name).read_bytes()) == binding['after_sha256'], name
    refs[name] = '8d2c27eae'
pins = {}
for name, ref in refs.items():
    data = (root / name).read_bytes()
    assert data == old(ref, name), name
    pins[name] = {'upstream': subprocess.check_output(['git', 'rev-parse', ref], text=True).strip(), 'sha256': sha(data)}
chain = ['b75d47950', 'c30f09ae5', '983947450', '7209aca03', 'a3d3cc025', '8d2c27eae', '241723e55', 'aa2968ffb']
expected = set()
for ref in chain:
    expected.update(subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', ref], text=True).splitlines())
changed = set(subprocess.check_output(['git', 'diff', '--name-only', 'f8260ead', 'HEAD'], text=True).splitlines())
assert changed == expected, (changed - expected, expected - changed)
proof = {
 'status': 'PASS: private source and output integration; localization generation and publication remain separate',
 'base': subprocess.check_output(['git', 'rev-parse', 'f8260ead'], text=True).strip(),
 'branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
 'committed_chain_head_before_owner_followup': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
 'adopted_upstream_commits': [subprocess.check_output(['git', 'rev-parse', r], text=True).strip() for r in chain],
 'source_output_bindings': pins,
 'generated_scope': 'All 11 normal generated output files retain their exact 8d2c27eae bytes. Their recorded before hashes match f8260ead. No generated merge, rewrite or regeneration occurred.',
 'source_review': [
  'Core requires the actual builtin declaration object plus exact remote URI, checkpoint digest and packaged card digest before exposing curated values; generic/custom records retain original values.',
  'Six bound metrics retain matched-stock values and two-pixel boundary F1 explicitly; prior metrics remain historical and raw BUNDLED declarations are unchanged.',
  'Generator binds before copying, reads the same bound metrics, and uses the recorded three-page image paths. Current glossary omits historical containers.',
  'F560 guard refuses non-four-dimensional stacks or unequal label counts before encoder/scorer work; no new API prose or scientific acceptance is asserted.',
  'The original source/tool/artifact tests were not repeated. Exact final source and generated bytes match their recorded upstream heads; only the changed extractor cohort was executed here.'
 ],
 'product_and_test_paths': sorted(p for p in expected if not p.startswith('features/data/')),
 'unrelated_paths': [],
 'pending': ['Source-bound runtime/API catalogs owned by other lanes', 'Coherent parent review/publication', 'Whole F370/F560 scientific/native/external acceptance beyond this bounded repair'],
}
(root / 'features/data/560_comparison_caption_owner_2026-10-15/integration.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps({'source_bindings': len(pins), 'generated_outputs': len(outputs), 'adopted_changed_paths': len(changed), 'owner_tests': receipt['final']['result']}))
