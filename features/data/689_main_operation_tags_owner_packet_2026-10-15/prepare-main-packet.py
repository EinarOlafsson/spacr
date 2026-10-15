"""Package only existing immutable source blobs; never mutate a main checkout."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import subprocess

root = Path.cwd()
scratch = Path('/mnt/wd4tb/scratch/home-b-n689-main-operation-tags-20261015')
out = root / 'features/data/689_main_operation_tags_owner_packet_2026-10-15'
scratch.mkdir(parents=True, exist_ok=False)
out.mkdir(parents=True, exist_ok=False)
main = subprocess.check_output(['git', 'rev-parse', 'origin/main'], text=True).strip()
nightly = subprocess.check_output(['git', 'rev-parse', 'origin/nightly'], text=True).strip()
source = subprocess.check_output(['git', 'rev-parse', '5a03a24ae7'], text=True).strip()
sha = lambda value: hashlib.sha256(value).hexdigest()

def blob(revision, path, optional=False):
    result = subprocess.run(['git', 'show', revision + ':' + path], capture_output=True)
    if result.returncode:
        assert optional, result.stderr.decode()
        return None
    return result.stdout

old_seven = Path('/mnt/wd4tb/scratch/home-b-n689-main-cutover-20261015')
old_eight = Path('/mnt/wd4tb/scratch/home-b-n689-main-cutover-access-20261015')
old_receipt = json.loads((old_eight / 'receipt.json').read_text())
paths = list(old_receipt['files'])
assert len(paths) == 8
assert main == old_receipt['main_base']
assert blob(source, '.github/workflows/docs.yml') == blob(nightly, '.github/workflows/docs.yml')
updated = {'tools/publish_docs_channels.py', 'tests/test_docs_space_access.py',
           'docs/notes/publishing_documentation.md'}
assert all(blob(source, p) == blob(nightly, p) for p in paths if p not in updated)
patch = subprocess.check_output(['git', 'diff', '--binary', '--full-index', main, source, '--', *paths])
assert patch
(out / 'proposal.patch').write_bytes(patch)
snapshot = scratch / 'apply-check'
snapshot.mkdir()
files = {}
for name in paths:
    before = blob(main, name, optional=True)
    target = blob(source, name)
    if before is not None:
        file = snapshot / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(before)
    files[name] = {'preimage_sha256': sha(before) if before is not None else None,
                   'source_sha256': sha(target), 'new_file': before is None,
                   'previous_packet_source_sha256': old_receipt['files'][name]['source_sha256'],
                   'changed_since_previous_packet': sha(target) != old_receipt['files'][name]['source_sha256']}
    if name.endswith('.py'):
        ast.parse(target, filename=name)
commands = []
for check in (True, False):
    command = ['git', 'apply'] + (['--check'] if check else []) + [str(out / 'proposal.patch')]
    result = subprocess.run(command, cwd=snapshot, capture_output=True, text=True)
    commands.append({'command': command, 'directory': str(snapshot),
                     'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    assert result.returncode == 0, result.stderr
assert {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()} == set(paths)
for name in paths:
    applied = (snapshot / name).read_bytes()
    assert applied == blob(source, name)
    files[name]['applied_sha256'] = sha(applied)
    files[name]['applied_exact'] = True

dependencies = {}
for name in ('tools/docs_version.py', 'tools/docs_publication_compat.py',
             'tools/pytest_translation_compatibility.py', 'tools/run_capped.sh'):
    before, after = blob(main, name), blob(source, name)
    assert before == after, name
    dependencies[name] = {'main_sha256': sha(before), 'source_sha256': sha(after), 'identical': True}
name = 'tools/verify_tutorial_live.py'
before, after = blob(main, name), blob(source, name)
def selected_ast(raw, symbol):
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == symbol)
    return ast.dump(node, include_attributes=False)
assert selected_ast(before, 'source_equivalent_assets') == selected_ast(after, 'source_equivalent_assets')
dependencies[name] = {'main_sha256': sha(before), 'source_sha256': sha(after),
                       'whole_file_identical': before == after,
                       'required_source_equivalent_assets_ast_identical': True}

history = out / 'historical'
history.mkdir()
historical = {}
for label, directory in [('seven-file', old_seven), ('eight-file', old_eight)]:
    for filename in ('receipt.json', 'proposal.patch'):
        raw = (directory / filename).read_bytes()
        (history / (label + '-' + filename)).write_bytes(raw)
        historical[label + '/' + filename] = {'original_path': str(directory / filename),
                                               'preserved_sha256': sha(raw)}
for filename in ('independent-peer.json', 'independent-byte-check.json'):
    raw = (old_seven / filename).read_bytes()
    (history / ('seven-file-' + filename)).write_bytes(raw)
    historical['seven-file/' + filename] = {'original_path': str(old_seven / filename),
                                            'preserved_sha256': sha(raw)}
owner_receipt = blob(source, 'features/data/689_access_operation_tags_2026-10-15/receipt.json')
(out / 'operation-tags-owner-receipt.json').write_bytes(owner_receipt)
record = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'Portable unapplied eight-file main proposal; disposable apply/source-byte proof only',
    'main_base': main, 'observed_nightly': nightly, 'source_commit': source,
    'main_readback': 'git fetch origin main nightly immediately before packaging; no main mutation',
    'files': files, 'patch_sha256': sha(patch),
    'historical_packets_preserved': historical, 'disposable_commands': commands,
    'disposable_exact_paths': paths, 'python_syntax': 'All six Python files parsed; no imports or tests executed',
    'dependencies': dependencies,
    'owner_operation_tagging_receipt_sha256': sha(owner_receipt),
    'scope': ['Exact current workflow retained, including separate main Pages/nightly static Space targets and paired host-variable resolver.',
              'Immutable full-tree/public verification and bounded access-only mode preserved.',
              'Root operation-tagging publisher/test/notes from5a03a24ae7: metadata/tree/readme/commit/readback refusal labels, original exception status, no credential output.',
              'No generated API/runtime/README/tutorial files, settings, token/variable changes or broad cherry-pick.'],
    'remaining_owner_actions': ['Original Home reviews/reconciles current main drift before private end-stage promotion.',
                               'Normal main Pages/legacy nightly redirect deployment remains separate.',
                               'Existing HTTP403 remains a real HF_DOCS_TOKEN permission gate; code does not resolve it.',
                               'Any bounded published permission probe, paired repository variables and Work URL-owner application require their coordinated actions.'],
    'not_performed': ['main checkout edits or push', 'workflow dispatch', 'Sphinx/site build', 'test replay',
                      'Space upload/create/history change', 'token retry/read/output', 'tutorial/media changes'],
}
(out / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
(out / 'prepare-main-packet.py').write_bytes(Path(__file__).read_bytes())
(out / 'OWNER_INSTRUCTIONS.txt').write_text(
    'UNAPPLIED N689 eight-file main owner packet\n\n'
    f'Exact main preimage: {main}\nExact source: {source}\n'
    f'Observed nightly workflow: {nightly}\n\n'
    'Original Home owns end-stage main promotion. Use an isolated owner checkout; compare every\n'
    'preimage with receipt.json before applying. Do not overwrite newer main changes. If main has\n'
    'moved, reconcile those changes and renew the source-byte proof. The prior seven/eight-file\n'
    'packets and their receipts remain byte-identical under historical/.\n\n'
    'Review: git apply --stat /ABS/PACKET/proposal.patch\n'
    'Preflight: git apply --check /ABS/PACKET/proposal.patch\n'
    'Only after owner review: git apply /ABS/PACKET/proposal.patch\n'
    'Compare all eight resulting SHA256 values with receipt.json before any integration.\n\n'
    'This packet was checked/applied only in an eight-file disposable scratch snapshot. No main\n'
    'checkout, service, variable, secret or deployment was changed. The actual403 permission gate\n'
    'remains unresolved. Operation tags help distinguish a future metadata/tree/readme/commit/\n'
    'readback refusal; they do not grant access. Old static Space/history remain untouched.\n')
print(json.dumps({'packet': str(out), 'main': main, 'source': source, 'paths': len(paths),
                  'patch_sha256': sha(patch), 'disposable_apply': 'PASS exact8',
                  'changed_since_previous': [p for p in paths if files[p]['changed_since_previous_packet']]}))
