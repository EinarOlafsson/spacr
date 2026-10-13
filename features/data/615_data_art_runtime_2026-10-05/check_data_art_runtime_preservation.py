from pathlib import Path
import hashlib
import json
import runpy
import subprocess

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
commit = '3053b081512d6f598a1aba76dac2393de31ba48a'
new = set(json.loads((scratch / 'data-art-runtime-sources-r1.json').read_text()))
reports = {}
for language in ('en', 'sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
    path = Path(f'spacr/qt/i18n_catalogs/{language}.py')
    before = {}
    exec(compile(subprocess.check_output(['git', 'show', f'{commit}:{path}'], text=True), str(path), 'exec'), before)
    after = runpy.run_path(str(path))
    before_ui = set(before['UI_SOURCES']) if language == 'en' else set(before['UI'])
    after_ui = set(after['UI_SOURCES']) if language == 'en' else set(after['UI'])
    assert after_ui - before_ui == new and before_ui <= after_ui
    preserved = 0
    for table in ('SETTING_LABELS', 'SETTING_TOOLTIPS', 'CATEGORY_HELP', 'UI', 'MODULE_SUMMARIES'):
        if table not in before:
            continue
        for key, value in before[table].items():
            assert after[table][key] == value, (language, table, key)
            if 'SOURCE_HASHES' in before:
                assert after['SOURCE_HASHES'][table, key] == before['SOURCE_HASHES'][table, key]
            preserved += 1
    if language != 'en':
        records = json.loads(Path(f'docs/i18n/reviewed/runtime/{language}/2026-10-05-data-art-replacement.json').read_text())['records']
        assert len(records) == 24 and {r['source'] for r in records} == new
        for record in records:
            assert after['UI'][record['source']] == record['translation']
            assert after['SOURCE_HASHES']['UI', record['source']] == record['source_sha256']
        assert preserved == 9833
    reports[language] = {'new_reviewed_data_art_strings': 24, 'complete_previous_rows_preserved': preserved,
                         'all_previous_source_hashes_preserved': True,
                         'catalog_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    print(language, '24 new current strings; every prior complete row preserved', flush=True)
(scratch / 'data-art-runtime-preservation-proof.json').write_text(json.dumps({'source_commit': commit, 'languages': reports}, indent=2) + '\n')
