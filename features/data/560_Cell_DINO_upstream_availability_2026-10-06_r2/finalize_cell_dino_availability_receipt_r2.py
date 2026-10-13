from pathlib import Path
import gzip
import hashlib
import json

out = Path('features/data/560_Cell_DINO_upstream_availability_2026-10-06_r2')
path = out.with_suffix('.json')
receipt = json.loads(path.read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for artifact, record in receipt['artifacts'].items():
    assert digest(artifact) == record['sha256'] and Path(artifact).stat().st_size == record['bytes']
source = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/cell-dino-upstream-availability-r2.log')
assert 'PASS pinned official Cell-DINO availability/factory evidence preserved' in source.read_text()
log = out / (source.name + '.gz')
log.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
assert gzip.decompress(log.read_bytes()) == source.read_bytes()
helper = out / Path(__file__).name
helper.write_bytes(Path(__file__).read_bytes())
for artifact in (log, helper):
    receipt['artifacts'][str(artifact)] = {'sha256': digest(artifact), 'bytes': artifact.stat().st_size}
receipt['corrected_private_audit_terminal_rc0_completed_log_retained'] = True
path.write_text(json.dumps(receipt, indent=2) + '\n')
note = '''
2026-10-06 workstation Cell-DINO availability receipt filename correction: the accepted pinned primary-source audit is features/data/560_Cell_DINO_upstream_availability_2026-10-06_r2.json. The original private r1 failed on the upstream README's stale LICENSE_CELL_DINO_CODE_WEIGHTS link (HTTP404); that original script/log and partial raw source downloads are retained. The corrected r2 verifies the actual LICENSE_CELL_DINO_MODELS file from the exact pinned upstream tree and is terminal rc=0. All original receipt artifact hashes are independently read back, with the completed r2 log also archived. The current foundation-picker tooltip and refusal test likewise still assert unpublished weights; Home should correct those with the loader/source change, followed by workstation normal API/runtime translation refresh. This does not claim Cell-DINO checkpoint acquisition, support or GPU acceptance.
'''
for source in (Path('features/future/560_foundation_model_embeddings.txt'), Path('features/325_two_sessions_one_repo_working_protocol.temp'), Path('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt')):
    with source.open('a') as stream:
        stream.write(note)
print('PASS complete accepted Cell-DINO upstream evidence artifact readback and corrected receipt filename/terminal log archived.', flush=True)
