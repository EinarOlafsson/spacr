from pathlib import Path
import ast
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parents[3]
baseline = 'b460359010b4614f3a8e5261c3d0a804bc4e1f1f'
tested = '7b37fc21cbe91432e7c1e2e96662b05b87a9bb02'
incoming = '23f694a67b8230134a67945dbb8991ee6d5a2e64'

def git(*args):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', *args], cwd=root)

def tree(pin):
    return {entry.split(b'\t', 1)[1].decode():
            entry.split(b'\t', 1)[0].split()[-1].decode()
            for entry in git('ls-tree', '-rz', pin).split(b'\0') if entry}

def tr_calls(source):
    return sorted(ast.dump(node, include_attributes=False)
                  for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.Call)
                  and ((isinstance(node.func, ast.Name) and node.func.id == 'tr')
                       or (isinstance(node.func, ast.Attribute)
                           and node.func.attr == 'tr')))

own = set(git('diff', '--name-only', baseline, tested).decode().splitlines())
public, actual = tree(incoming), tree('HEAD')
preserved = [path for path in public if path not in own]
assert all(public[path] == actual[path] for path in preserved)
for path in ['spacr/qt/i18n_catalogs',
             'tests/data/release_contracts/runtime_caption_arrivals_2026_10_08.json.gz']:
    assert git('rev-parse', 'HEAD:' + path) == git('rev-parse', tested + ':' + path)
normal = json.loads((Path(__file__).parent / 'runtime-receipt.json').read_text())
for path, expected in normal['source_file_sha256_unchanged'].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
incoming_sources = []
for path in git('diff', '--name-only', baseline, incoming, '--', 'spacr', 'tools').decode().splitlines():
    if path.endswith('.py'):
        assert tr_calls(git('show', baseline + ':' + path).decode()) == tr_calls(git('show', incoming + ':' + path).decode()), path
        incoming_sources.append(path)
for path in ['spacr/qt/i18n_catalogs', 'tools/build_i18n_catalogs.py',
             'tools/build_documentation_i18n.py', 'spacr/qt/i18n.py']:
    assert git('rev-parse', baseline + ':' + path) == git('rev-parse', incoming + ':' + path)
receipt = {
    'tested_coherent_closure': tested,
    'immutable_incoming_public': incoming,
    'publication_retry': 'First fail-fast helper fresh fetch found external evidence-only 23f694 advance after pinned9e78; stopped before staging/commit/push. Reconciled exact16 evidence/feature/handoff paths, no spacr/tools/docs/tests delta, no test or generator repeat.',
    'public_driver_base': baseline,
    'unrelated_incoming_public_files_exact': len(preserved),
    'incoming_changed_python_sources_tr_calls_exact': incoming_sources,
    'incoming_runtime_catalogs_and_owning_generators_unchanged': True,
    'generated_runtime_catalogs_and_arrivals_exact_tested_closure': True,
    'actual_source_file_hashes_exact': normal['source_file_sha256_unchanged'],
    'checks': '47 owning checks and actual normal API visibility inventory; full nine-locale runtime audit and ordinary arrivals; 18 actual classifier locale callbacks, zero skips. Unchanged checks not repeated after unrelated publication/API/tutorial merge.',
    'limits': 'Shared physical-channel source component only. No whole F535/F540, biological calibration, native model training, global CI or release acceptance. Incoming API prose and publication-verifier changes remain their owners separate reviewed artifacts; owning runtime captions and generator inputs did not change.',
    'validated_tree_ids': {path: git('rev-parse', 'HEAD:' + path).decode().strip()
                           for path in ['spacr', 'tools', 'docs', 'tests']},
    'artifact_sha256': normal['source_file_sha256_unchanged'],
}
(Path(__file__).parent / 'final-publication-proof.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'incoming': incoming, 'preserved': len(preserved), 'sources': incoming_sources}))
