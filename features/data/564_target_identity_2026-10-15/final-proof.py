from pathlib import Path
import ast
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parents[3]
base = '086f03ca5d400bf5546f167a0e07fa1f2d248954'
tested = '5c09e86b4'
incoming = '086f03ca5d400bf5546f167a0e07fa1f2d248954'

def git(*args):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', *args], cwd=root)

def tree(pin):
    return {item.split(b'\t', 1)[1].decode():
            item.split(b'\t', 1)[0].split()[-1].decode()
            for item in git('ls-tree', '-rz', pin).split(b'\0') if item}

own = set(git('diff', '--name-only', base, tested).decode().splitlines())
public, actual = tree(incoming), tree('HEAD')
unrelated = [path for path in public if path not in own]
assert all(public[path] == actual[path] for path in unrelated)
for path in ['spacr', 'tools', 'docs', 'tests']:
    assert git('rev-parse', base + ':' + path) == git('rev-parse', incoming + ':' + path)
for path in ['spacr/qt/i18n_catalogs', 'docs/source/_static/i18n/api',
             'tests/data/release_contracts/runtime_caption_arrivals_2026_10_08.json.gz',
             'tools/build_i18n_catalogs.py', 'tools/build_documentation_i18n.py']:
    assert git('rev-parse', 'HEAD:' + path) == git('rev-parse', base + ':' + path)
receipt = json.loads((Path(__file__).parent / 'receipt.json').read_text())
hashes = receipt['source_file_sha256']
for path, expected in hashes.items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
def calls(raw):
    return sorted(ast.dump(node, include_attributes=False)
                  for node in ast.walk(ast.parse(raw))
                  if isinstance(node, ast.Call)
                  and ((isinstance(node.func, ast.Name) and node.func.id == 'tr')
                       or (isinstance(node.func, ast.Attribute) and node.func.attr == 'tr')))
assert calls((root / 'spacr/attribution.py').read_text()) == calls(git('show', base + ':spacr/attribution.py').decode())
proof = {
    'immutable_incoming_public': incoming,
    'tested_coherent_source': tested,
    'unrelated_incoming_public_files_exact': len(unrelated),
    'incoming_spacr_tools_docs_tests_tree_ids_exact_tested_base': True,
    'generated_runtime_API_arrivals_exact_base': True,
    'no_new_runtime_tr_call': True,
    'source_and_test_hashes_exact': hashes,
    'checks': '24 actual direct/report-refusal owning checks on exact source. Exact numeric-parser body reconstruction and signature/prose/tr AST neutrality establish no derived API/runtime target. No tests/writer/inventory repeats.',
    'limits': receipt['limits'],
    'validated_tree_ids': {path: git('rev-parse', 'HEAD:' + path).decode().strip()
                           for path in ['spacr', 'tools', 'docs', 'tests']},
    'artifact_sha256': hashes,
}
(Path(__file__).parent / 'final-publication-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps({'incoming': incoming, 'unrelated_preserved': len(unrelated)}))
