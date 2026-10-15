import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
DEST = ROOT / 'features/data/580_session_review_inputs_2026-10-15'
DEST.mkdir(exist_ok=False)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
packet = json.loads((ROOT / 'features/data/580_session_gui_2026-10-15/dependency-packet.json').read_text())
api = json.loads((OUT / 'api-source.json').read_text())
checks = json.loads((OUT / 'input-check.json').read_text())
assert all(row['status'] == 'PASS' for category in checks.values() for row in category.values())
assert (OUT / 'initial-inputs.exit').read_text().strip() == '1'
assert (OUT / 'followup.exit').read_text().strip() == '0'

new_files = {}
for category in ('runtime', 'api'):
    for source in (OUT / (category + '_inputs')).glob('*/*.json'):
        relative = Path('docs/i18n/reviewed') / category / source.parent.name / source.name
        path = ROOT / relative
        assert not path.exists()
        shutil.copy2(source, path)
        new_files[str(relative)] = sha(path.read_bytes())
assert len(new_files) == 18

retirements = packet['four_prior_tooltip_review_retirements'] + api['retirements']
assert len(retirements) == 45
grouped = {}
for row in retirements:
    grouped.setdefault(row['path'], []).append(row)
retirement_receipt = []
for relative, rows in grouped.items():
    path = ROOT / relative
    before = path.read_bytes()
    assert all(sha(before) == row['file_sha256'] for row in rows)
    payload = json.loads(before)
    original = payload['records']
    indexes = {row['record_index'] for row in rows}
    for row in rows:
        assert original[row['record_index']] == row['old_record']
        api_row = relative.startswith('docs/i18n/reviewed/api/')
        replacement = ('2026-10-15-f580-session-preview-home-b.json' if api_row
                       else '2026-10-15-f580-session-controls-home-b.json')
        payload.setdefault('retired_records', []).append(row['old_record'] | {
            'retired': '2026-10-15 F580 source 6a1b51bfd7 superseded this exact source; retained with its original target. Replacement: ' + replacement,
            'replacement_source_sha256': row['replacement_source_sha256']})
    payload['records'] = [row for index, row in enumerate(original) if index not in indexes]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    assert payload['records'] == [row for index, row in enumerate(original) if index not in indexes]
    retirement_receipt.append({'path': relative, 'before_sha256': sha(before), 'after_sha256': sha(path.read_bytes()),
        'retired': rows, 'unrelated_active_records_preserved': len(payload['records'])})
(DEST / 'retirements.json').write_text(json.dumps(retirement_receipt, ensure_ascii=False, indent=2) + '\n')

for name in ('targets.py', 'prepare_inputs.py', 'initial-prepare_inputs.py', 'finish_inputs.py',
             'api-source.json', 'input-check.json', 'initial-input-check.json',
             'portuguese-context.json', 'portuguese-followup.json',
             'independent-first-six-peer.json', 'independent-all-nine-peer.json',
             'initial-inputs.sh', 'initial-inputs.time', 'initial-inputs.exit',
             'followup.sh', 'followup.time', 'followup.exit'):
    shutil.copy2(OUT / name, DEST / name)
for name in ('initial-inputs.log', 'followup.log'):
    (DEST / (name + '.gz')).write_bytes(gzip.compress((OUT / name).read_bytes(), mtime=0))
shutil.copy2(OUT / 'seal.py', DEST / 'seal.py')

backend = Path('/mnt/wd4tb/scratch/home-b-f580-backend-census-20261015')
backend_receipt = json.loads((backend / 'receipt.json').read_text())
assert all(packet['source_sha256'][path] == digest for path, digest in backend_receipt['source_sha256'].items())
peer = {
    'status': 'Independent technical source-census review PASS; no execution of the census by this reviewer',
    'source_owner': backend_receipt['source_owner'], 'source_sha256': backend_receipt['source_sha256'],
    'reviewer_boundary': 'Reviewer authored the two GUI adapters, not the backend implementation or root census.',
    'reviewed_census_script_sha256': sha((backend / 'census.py').read_bytes()),
    'reviewed_census_receipt_sha256': sha((backend / 'receipt.json').read_bytes()),
    'reviewed_log_sha256': sha((backend / 'run.log').read_bytes()),
    'findings': [
        'The normal visibility and enabled nested policies are invoked over exactly the three claimed backend files, with no application imports or model decoding.',
        'The reported 90 visible bodies, 45 enabled nested entries and literal tr sets are compared before/after, not inferred from names alone.',
        'A separate AST doc-body comparison confirms the only seven changes are the enumerated private definitions; module/class/public doc bodies are unchanged.',
        'Source hashes exactly match the GUI packet integrated backend; no additional backend API or literal-tr arrival was found within this bounded change.',
        'The four new labels/eight setting tooltip bodies are handled by the separate 18-source runtime packet. This is not a global catalog audit or backend behavior review.'
    ]}
(DEST / 'backend-census-peer.json').write_text(json.dumps(peer, indent=2) + '\n')
receipt = {
    'status': '171 source-bound review inputs ready; normal catalog production and canonical retirement census pending',
    'source_commit': '6a1b51bfd7', 'packaging_commit': '341e529f07',
    'source_sha256': packet['source_sha256'],
    'runtime_targets': 162, 'api_targets': 9, 'runtime_sources': 18, 'api_symbols': [api['changed_block']['label']],
    'new_review_files_sha256': new_files,
    'retired_runtime_records': 36, 'retired_api_records': 9,
    'retirement_policy': 'Only exact old-source active rows moved to retired_records; original fields/targets and unrelated active rows preserved.',
    'remaining_removed_gui_captions': packet['removed_gui_literals_pending_normal_canonical_retirement_check'],
    'normal_validation': {
        'initial_session': 91499, 'initial_exit': 1,
        'initial_result': 'Eight runtime loaders PASS; Portuguese identifier rewritten by generic prose rule; nine API source/context gates passed before harness wrong-arity inline call.',
        'followup_session': 60581, 'followup_exit': 0,
        'followup_result': 'Only Portuguese loader rerun with quoted filename plus nine actual inline parsers; all PASS. No source/gate changes.',
        'scope': 'Normal reviewed-runtime loader against exactly the 18 source-bound packet entries and isolated new review directory; no full canonical collector or catalog writer.'},
    'independent_review': 'Root accepted all 171 draft semantics, then accepted the single Portuguese quoted-identifier delta; other 170 target bytes unchanged. AI technical review, no native-speaker signoff.',
    'no_model_decode': True, 'no_generated_catalog_changes': True,
    'evidence_sha256': {p.name: sha(p.read_bytes()) for p in DEST.iterdir() if p.is_file()}
}
(DEST / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'runtime': 162, 'api': 9, 'new_files': len(new_files), 'retirements': len(retirements)}, indent=2))
