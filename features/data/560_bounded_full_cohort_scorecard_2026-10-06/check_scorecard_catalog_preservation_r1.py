from pathlib import Path
import hashlib
import json
import runpy

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
baseline = scratch / 'scorecard-api-repair-baseline-r1'
delta = json.loads((scratch / 'scorecard-api-runtime-source-delta-r1.json').read_text())
assert not delta['api_changed']
old_source, source = delta['runtime_retired'][0], delta['runtime_added'][0]
languages = ('en', 'sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr')
report = {'API': {}, 'runtime': {}}
for language in languages:
    api_path = Path('docs/source/_static/i18n/api') / (language + '.json')
    old_api = (baseline / 'api' / (language + '.json')).read_bytes()
    assert api_path.read_bytes() == old_api
    assert len(json.loads(old_api)['symbols']) == 13181
    report['API'][language] = {'all_complete_records_preserved': 13181,
                               'catalog_sha256': hashlib.sha256(old_api).hexdigest()}
    path = Path('spacr/qt/i18n_catalogs') / (language + '.py')
    before = runpy.run_path(str(baseline / 'runtime' / (language + '.py')))
    after = runpy.run_path(str(path))
    ui_table = 'UI_SOURCES' if language == 'en' else 'UI'
    assert set(before[ui_table]) - set(after[ui_table]) == {old_source}
    assert set(after[ui_table]) - set(before[ui_table]) == {source}
    for key in before:
        if not key.isupper() or key in (ui_table, 'SOURCE_HASHES'):
            continue
        assert before[key] == after[key], (language, key)
    if language != 'en':
        for key, value in before[ui_table].items():
            if key != old_source:
                assert after[ui_table][key] == value, (language, key)
        reviewed = json.loads((Path('docs/i18n/reviewed/runtime') / language /
                               '2026-09-27-runtime-codex-delta.json').read_text())
        entry = next(r for r in reviewed['records'] if r['source'] == source)
        assert after[ui_table][source] == entry['translation']
    expected_hashes = dict(before['SOURCE_HASHES'])
    del expected_hashes['UI', old_source]
    assert set(after['SOURCE_HASHES']) - set(expected_hashes) == {('UI', source)}
    assert after['SOURCE_HASHES']['UI', source] == hashlib.sha256(source.encode()).hexdigest()
    for key, sha in expected_hashes.items():
        assert after['SOURCE_HASHES'][key] == sha
    report['runtime'][language] = {'all_other_complete_rows_and_hashes_preserved': len(expected_hashes),
                                   'current_rows': len(after['SOURCE_HASHES']),
                                   'exact_source_replacements': 1,
                                   'catalog_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    print(language, 'all 13181 complete API records identical; exactly one runtime caption replaced', flush=True)
(scratch / 'scorecard-catalog-preservation-proof-r1.json').write_text(json.dumps(report, indent=2) + '\n')
