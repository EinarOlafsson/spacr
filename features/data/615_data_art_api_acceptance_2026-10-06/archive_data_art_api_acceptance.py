from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
proof = json.loads((scratch / 'data-art-api-preservation-proof.json').read_text())
review = json.loads((scratch / 'data-art-api-review-input-proof.json').read_text())
assert review['accepted'] and review['records_each_language'] == 16 and not review['failures']
assert len(proof['languages']) == 10
assert all(row['current_symbols'] == 13181 and row['complete_other_records_preserved'] == 13169 and row['new_source_and_context_hashes_current'] for row in proof['languages'].values())
assert 'verified API catalogs: languages=9 symbols=13181' in (scratch / 'data-art-api-strict-audit-r1.log').read_text()
assert 'build succeeded.' in (scratch / 'data-art-sphinx-all-r1.log').read_text()
assert '12 passed' in (scratch / 'data-art-api-catalog-tests-r1.log').read_text()
assert '6 passed' in (scratch / 'data-art-docstring-tests-r1.log').read_text()
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for lang, row in proof['languages'].items():
    assert digest(Path('docs/source/_static/i18n/api') / (lang + '.json')) == row['catalog_sha256']
dest = Path('features/data/615_data_art_api_acceptance_2026-10-06')
dest.mkdir(exist_ok=True)
for name in ('data-art-api-preservation-proof.json', 'check_data_art_api_preservation.py',
             'data-art-api-delta-r1.json', 'data-art-api-new-blocks-r1.json',
             'admit_data_art_api_reviews.py', Path(__file__).name):
    shutil.copyfile(scratch / name, dest / name)
logs = ('data-art-api-refresh-r1.log', 'data-art-api-refresh-r2.log',
        'data-art-api-reviewed-cpu-repair-r1.log', 'data-art-api-reviewed-cpu-repair-r2.log',
        'data-art-api-review-admission-r1.log', 'data-art-api-current-english-r1.log',
        'data-art-api-strict-audit-r1.log', 'data-art-api-preservation-r1.log',
        'data-art-sphinx-all-r1.log', 'data-art-api-catalog-tests-r1.log',
        'data-art-docstring-tests-r1.log')
for name in logs:
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
receipt = {'source_application_commit': proof['source_commit'],
           'normal_nine_language_reviewed_repair_and_strict_audit_passed': True,
           'strict_audit_terminal_exit_code': 0,
           'reviewed_unique_new_prose_blocks_each_language': 16,
           'total_source_context_bound_reviewed_inputs': 144,
           'current_symbols_each_catalog': 13181, 'catalog_languages': 10,
           'changed_symbols_each_catalog': proof['languages']['en']['changed_symbols'],
           'all_other_complete_records_preserved_each_catalog': 13169,
           'strict_all_language_full_Sphinx_build_passed': True,
           'API_catalog_tests_passed': 12, 'docstring_tests_passed': 6,
           'review_method': 'Direct Codex AI technical review; no native-speaker signoff',
           'failed_gpu_attempts_accepted': False, 'interrupted_cpu_attempt_accepted_as_full_run': False,
           'interrupted_process_143_cause_known': False,
           'runtime_receipt': '615_data_art_runtime_2026-10-05.json',
           'guides_receipt': '615_data_art_guides_2026-10-05.json',
           'new_catalog_nightly_deployment_verified': False,
           'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}}
Path('features/data/615_data_art_api_acceptance_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation replacement data-art API acceptance: sixteen new unique prose blocks in all nine languages are admitted through the normal reviewed-input helper, preserving exact source/context, technical syntax, semantic review and prior-record gates. Normal CPU repair and a terminal rc=0 strict all-nine audit pass at 13,181 API symbols. Exactly twelve existing records change in each of ten catalogs; all other 13,169 complete records remain identical to source checkpoint 3053b0815. Twelve API catalog tests, six current docstring tests and the full strict all-language Sphinx build pass. Receipt 615_data_art_api_acceptance_2026-10-06.json archives completed generation, strict audit, full-record preservation, scoped tests, source deltas and the failed GPU/interrupted CPU attempts; exit 143 cause is unresolved and no failed attempt is called accepted. Runtime and guide source-current work is already separately accepted. Direct AI technical review is not native-speaker signoff. New catalogs still need actual nightly deployed readback after push; remaining installation/scientific tutorial work stays open. Mask 07 and current Measure/Home/Conda corrections remain intact. Home retains CPU CI/coverage/Qt/source ownership; all GPU processes remain workstation-owned.\n'
for path in ('features/325_two_sessions_one_repo_working_protocol.temp', 'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: complete source-current replacement API acceptance archived.', flush=True)
