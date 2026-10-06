from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
assert '5 failed, 6 passed' in (scratch / 'yolo-preserved-publication-tests-r1.log').read_text()
assert '13 passed' in (scratch / 'yolo-preserved-publication-tests-r3.log').read_text()
dest = Path('features/data/615_publication_guard_refresh_2026-10-06')
dest.mkdir(exist_ok=True)
for name in ('yolo-preserved-publication-tests-r1.log', 'yolo-preserved-publication-tests-r2.log', 'yolo-preserved-publication-tests-r3.log'):
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
shutil.copyfile(__file__, dest / Path(__file__).name)
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {'historical_global_pin_failures': 5, 'scoped_tests_passed': 13,
           'conflicting_current_revision_mutation_rejected': True,
           'conflicting_hosted_narration_hash_mutation_rejected': True,
           'accepted_baseline_voice_set_preserved': True,
           'current_normal_checkpoint_receipt_readback_and_hosted_checks_bound': True,
           'guard_ceiling_or_exclusion_changed': False,
           'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/615_publication_guard_refresh_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation current-publication guard refresh: the preserved Map Barcodes, OPS, Model Zoo, Model Compare and Embeddings checks failed only because their helper pinned every future publication to the historical wave-4 global commit/tag. The helper now proves the current normal checkpoint/receipt/readback revision matches, the immutable host root contains that revision, and exact hosted browser manifest/index/narration hashes agree. The accepted unchanged 27-voice baseline, current source narration, local candidate hash and every sentence-cue check remain. Both conflicting-current-revision and conflicting-hosted-narration mutations are rejected. Thirteen scoped checks pass; the original five failures remain archived in 615_publication_guard_refresh_2026-10-06.json. No old evidence was rewritten, no voice was dropped and no exclusion or ratchet was changed.\n'
for path in ('features/325_two_sessions_one_repo_working_protocol.temp', 'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: current-publication binding and two independent conflict mutations archived.', flush=True)
