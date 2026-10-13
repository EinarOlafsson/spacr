from pathlib import Path
import hashlib
import json
import runpy
import subprocess

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
current = 'Interactive plots need pyqtgraph.\n\nInstall it with python -m pip install spacr, then reopen this module.\n\nEverything else works without it: the run still produces every figure, and they appear on the grid above the console.'
reports = {}
for language in ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
    path = Path(f'spacr/qt/i18n_catalogs/{language}.py')
    before = {}
    exec(compile(subprocess.check_output(['git', 'show', f'{commit}:{path}'], text=True), str(path), 'exec'), before)
    after = runpy.run_path(str(path))
    retired = next(key for key in before['UI'] if key.startswith('Interactive plots need pyqtgraph.'))
    assert "Install it with  pip install 'spacr[qt]'  and reopen this module." in retired
    assert set(after['UI']) - set(before['UI']) == {current}
    assert set(before['UI']) - set(after['UI']) == {retired}
    preserved = 0
    for table in ('SETTING_LABELS', 'SETTING_TOOLTIPS', 'CATEGORY_HELP', 'UI', 'MODULE_SUMMARIES'):
        for key, value in before[table].items():
            if table == 'UI' and key == retired:
                continue
            assert after[table][key] == value, (language, table, key)
            assert after['SOURCE_HASHES'][table, key] == before['SOURCE_HASHES'][table, key]
            preserved += 1
    review = json.loads(Path(f'docs/i18n/reviewed/runtime/{language}/2026-10-05-core-gui-install.json').read_text())
    assert after['UI'][current] == review['records'][0]['translation']
    assert 'python -m pip install spacr' in after['UI'][current]
    reports[language] = {'all_other_complete_rows_and_hashes_preserved': preserved,
                         'retired_source': retired, 'current_source': current,
                         'current_reviewed_target': after['UI'][current],
                         'catalog_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    print(language, 'preserved', preserved, 'complete rows and hashes; exactly one required source replacement', flush=True)
(scratch / 'core-gui-runtime-preservation-proof.json').write_text(json.dumps(
    {'source_commit': commit, 'languages': reports}, ensure_ascii=False, indent=2) + '\n')
