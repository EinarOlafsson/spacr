from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
report = json.loads((scratch / 'data-art-api-browser-r1/acceptance.json').read_text())
assert report['passed'] and len(report['cases']) == 108
assert len({row['symbol'] for row in report['cases']}) == 12
assert len({row['language'] for row in report['cases']}) == 9
assert all(row['passed'] for row in report['cases'])
dest = Path('features/data/615_data_art_api_browser_2026-10-06')
dest.mkdir(exist_ok=True)
for source in (scratch / 'data-art-api-browser-r1/acceptance.json', scratch / 'verify_data_art_api_browser.py', Path(__file__)):
    shutil.copyfile(source, dest / source.name)
(dest / 'data-art-api-browser-r1.log.gz').write_bytes(gzip.compress((scratch / 'data-art-api-browser-r1.log').read_bytes(), mtime=0))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {'current_strict_build_API_browser_cases_passed': 108,
           'changed_symbols': 12, 'languages': 9,
           'actual_translated_panels_and_protected_literals_verified': True,
           'all_built_catalog_hashes_match_accepted_preservation_proof': True,
           'nightly_deployment_verified': False,
           'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/615_data_art_api_browser_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation replacement API rendered-browser readback: all twelve changed night-theme/preferences/theme API symbols render in each of nine languages, 108 actual browser panel cases. Every built English/localized catalog hash matches the separately accepted full-record preservation proof. Current source hashes, non-English targets, panel language and every protected literal pass in the actual strict Sphinx output; no browser script error occurs. Receipt 615_data_art_api_browser_2026-10-06.json retains all case hashes, source-bound helper and completed browser log. This verifies the locally built artifact; actual nightly deployment remains separate and the normal hosted workflow is still in progress.\n'
for path in ('features/325_two_sessions_one_repo_working_protocol.temp', 'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: 108 current API browser cases archived.', flush=True)
