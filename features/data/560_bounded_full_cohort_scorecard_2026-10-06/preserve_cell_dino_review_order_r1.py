from pathlib import Path
import json
import subprocess

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
delta = json.loads((scratch / 'scorecard-api-runtime-source-delta-r1.json').read_text())
old_source, source = delta['runtime_retired'][0], delta['runtime_added'][0]
for language in ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
    path = Path('docs/i18n/reviewed/runtime') / language / '2026-09-27-runtime-codex-delta.json'
    original = json.loads(subprocess.check_output(['git', 'show', f'ad7ffb18c:{path}']))
    current = json.loads(path.read_text())
    entries = [r for r in current['records'] if r['source'] == source]
    assert len(entries) == 1
    expected = [entries[0] if r['source'] == old_source else r for r in original['records']]
    assert sorted(current['records'], key=lambda r: (r['table'], r['key'])) == sorted(expected, key=lambda r: (r['table'], r['key']))
    current = dict(original)
    current['records'] = [{key: entries[0][key] for key in r}
                          if r['source'] == old_source else r
                          for r in original['records']]
    current['note'] = original['note'] + ' 2026-10-06: the revised Cell-DINO caption received direct Codex AI technical review only; earlier independent-review statements apply to its previous wording. All other records retain their existing review history.'
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n')
    print(language, 'reviewed author-input order preserved; exactly one record replaced')
