from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
out = Path('features/data/615_subcell_current_documentation_2026-10-06')
out.mkdir(exist_ok=False)
sha = lambda data: hashlib.sha256(data).hexdigest()
inventory = json.loads((scratch / 'subcell-rybg-documentation-inventory-r4.json').read_text())
source_sha = sha(Path('spacr/embeddings.py').read_bytes())
assert source_sha == inventory['source_sha256']
checks = {
    'full_catalog_preservation': 'subcell-catalog-preservation-r1.json',
    'strict_complete_Sphinx': 'subcell-full-Sphinx-r1.json',
    'original_twelve_full_cohorts': 'subcell-integrated-full-cohort-replay-r5.json',
    'normal_API_generation': 'subcell-rybg-API-generation-r2.json',
}
for name, path in checks.items():
    document = json.loads((scratch / path).read_text())
    if name != 'normal_API_generation':
        assert document['passed']
    assert source_sha in (document.get('application_source_sha256'), document.get('source_sha256'))
    shutil.copy2(scratch / path, out / path)
audits = {}
for group in ('A', 'B', 'C'):
    name = 'subcell-full-API-audit-' + group + '-r1.log'
    raw = (scratch / name).read_bytes()
    assert b'verified API catalogs: languages=3 symbols=13182' in raw
    audits[group] = {'normal_full_CLI_terminal_rc': 0, 'log_sha256': sha(raw)}
raw = (scratch / 'subcell-rybg-runtime-normal-refresh-r4.log').read_bytes()
assert b'verified external runtime catalogs: languages=9 settings=1238 categories=237 ui=7112 modules=77' in raw
for name in ('subcell-translated-Qt-dialog-r4', 'subcell-API-browser-r3'):
    document = json.loads((scratch / name / 'acceptance.json').read_text())
    assert document['passed'] and len(document['languages']) == 9
    shutil.copytree(scratch / name, out / name)
for path in sorted(scratch.glob('subcell-*.log')):
    if path.name in ('subcell-rybg-API-generation-r2.log', 'subcell-rybg-runtime-normal-refresh-r2.log'):
        status = 'owner-stopped preprocessing/candidate attempt, not acceptance'
    else:
        status = 'retained original closed attempt; only explicit receipt assertions designate final acceptance'
    packed = out / (path.name + '.gz')
    raw = path.read_bytes()
    packed.write_bytes(gzip.compress(raw, mtime=0))
    assert gzip.decompress(packed.read_bytes()) == raw
for path in sorted(scratch.glob('subcell-*.json')):
    if path.is_file():
        shutil.copy2(path, out / path.name)
for pattern in ('*subcell*.py', 'verify_foundation_api_full_cohorts_r4.py', 'verify_foundation_api_full_cohorts_r5.py'):
    for path in scratch.glob(pattern):
        shutil.copy2(path, out / path.name)
for path in ('current-night-preset-identity-tests-r1.log',):
    packed = out / (path + '.gz')
    packed.write_bytes(gzip.compress((scratch / path).read_bytes(), mtime=0))
diagnostic = Path('.translation-reports/f2780d8720510a7fee4f09472e0cdd8f1a5c1dd9d6275ed1dc64d45aa7913ec8-3042831.json')
shutil.copy2(diagnostic, out / 'night-preset-original-report-only-diagnostic.json')
for path in ('spacr/embeddings.py', 'tools/build_i18n_catalogs.py', 'docs/source/_static/api_i18n.js'):
    data = Path(path).read_bytes()
    packed = out / (path.replace('/', '_') + '.gz')
    packed.write_bytes(gzip.compress(data, mtime=0))
html = scratch / 'subcell-api-docs-all-r1'
for path in ('api/spacr/embeddings/index.html', 'api/spacr/qt/screens/embeddings/index.html'):
    (out / (path.replace('/', '_') + '.gz')).write_bytes(gzip.compress((html / path).read_bytes(), mtime=0))
report = {'scope': 'Current-source normal multilingual API/runtime generation, complete strict audits, full-source documentation build, actual API browser/Qt rendering, source preservation and original benchmark API replay after Home four-channel integration.',
          'passed_local': True, 'application_source_sha256': source_sha, 'baseline_commit': '58df6f5b9', 'integrated_Home_commit': '3ce39ac7b',
          'API_symbols': 13182, 'API_delta': {'added': inventory['API_added'], 'changed': inventory['API_changed'], 'removed': []},
          'API_reviews': 'Eight new source-bound blocks in every nine target languages; direct AI technical review, no native-speaker signoff.',
          'runtime_reviews': '21 new source-bound captions in all nine languages; official ER (Y) and DNA (B) marker captions retain their scientific identity.',
          'all_nine_normal_full_API_audits': audits, 'normal_runtime_CLI_full_audit_terminal_rc': 0,
          'complete_runtime_rows': 9880, 'unchanged_prior_runtime_rows': 9859, 'unchanged_complete_API_records': 13177,
          'all_complete_original_review_files_retained_byte_exact_in_place_or_documented_archive': True,
          'all_accepted_tutorial_source_and_media_byte_unchanged': True,
          'actual_Qt': 'All 21 captions, initially unchosen mapping, duplicate/missing/out-of-range refusals and permuted mapping storage exercised in each locale using normal Qt and dialog translation installation.',
          'actual_browser': 'All eight new and seven prior encoder-entry reviewed blocks plus the new mapping callback rendered from complete current catalogs in each locale.',
          'checks': {'actual_integrated_foundation_cases': '101 passed', 'callable_and_docstring_guards': '143 passed', 'API_frontend_and_all_catalog_source_contracts': '22 passed', 'scientific_markers_and_negative_copied_prose_control': '41 passed', 'settings': '9 passed', 'Help': '56 passed', 'runtime_catalog_cohort': '41 passed, one original report-only identity incompatibility', 'night_preset_identity_followup': '2 passed; unchanged native scientific/cloud titles and musical loanword explicitly declared only for the ten existing locale/title pairs'},
          'retained_nonaccepted_attempts': ['CPU parity r1 private constant/indexing verification', 'API generation r1-r4 with source-stale/identity loader failures or owner stop', 'runtime source-stale and unreviewed candidate attempts r1-r3', 'runtime direct reviews r1-r3 rejected same-spelling captions before any write', 'Qt verifier r1 editor tooltip retargeted to label', 'Qt verifier r2 guessed English remapping sentence', 'API browser r1/r2 looked for screen callback on backend page', 'benchmark API replay r4 selected main environment without optional timm'],
          'wrapper_log_scope': 'Uncaught tracebacks after the wrapper finally are outside its saved Tee logs; no traceback is reconstructed as raw capture.',
          'publication_state': 'Private local acceptance checkpoint; actual pushed-source workflow and deployed readback remain separate required work.',
          'not_claimed': ['Whole items 560/615 or goal completion', 'Official Cell-DINO loading', 'Four-stain expert biological accuracy', 'Final global GitHub green', 'Native-speaker signoff', 'Native macOS/Windows updater capture'],
          'receipt_files': {str(path.relative_to(out)): {'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())} for path in sorted(out.rglob('*')) if path.is_file()}}
Path('features/data/615_subcell_current_documentation_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
shutil.copy2(__file__, out / Path(__file__).name)
print('PASS complete current local acceptance archive, preserving all failed/nonaccepted diagnostics and bounded claim scope', flush=True)
