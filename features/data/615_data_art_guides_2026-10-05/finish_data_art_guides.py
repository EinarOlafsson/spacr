from pathlib import Path
import hashlib
import json
import subprocess
import sys

sys.meta_path = [finder for finder in sys.meta_path
                 if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path('tools').resolve()))
import build_guide_i18n as builder
from spacr.qt.night_themes import DATA_ART_THEMES
import spacr.qt.night_themes as registry

assert Path(registry.__file__).resolve().is_relative_to(Path.cwd().resolve())
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
pot = scratch / 'data-art-guide-gettext'
prose = json.loads((scratch / 'data-art-guide-prose-reviewed.json').read_text())
themes = list(DATA_ART_THEMES.values())
prepared = {}
for language in builder.LANGUAGES:
    glossary = builder.build_glossary(language, pot)
    worklist = scratch / f'data-art-guide-worklist-{language}.json'
    count = builder.export_worklist(language, worklist)
    assert count == 14, (language, count)
    rows = json.loads(worklist.read_text())
    assert {row['domain'] for row in rows} == {'features'}
    runtime = json.loads(Path(f'docs/i18n/reviewed/runtime/{language}/2026-10-05-data-art-replacement.json').read_text())
    translations = {record['source']: record['translation'] for record in runtime['records']}
    targets = {'Data-art themes': prose[language]['title']}
    intro = prose[language]['intro']
    for label in ['Preferences', 'Appearance', 'Theme', 'Animation', 'None']:
        displayed = builder.runtime_ui_name(label, language)
        assert displayed and glossary.get(label, displayed) == displayed
        intro = intro.replace('@' + label + '@', displayed)
    assert '@' not in intro
    for row in rows:
        if row['msgid'].startswith('Under **Preferences**'):
            targets[row['msgid']] = intro
    for theme in themes:
        source = f'**{theme.label}** — {theme.description}'
        assert glossary[theme.label] == translations[theme.label]
        targets[source] = f'**{translations[theme.label]}** — {translations[theme.description]}'
    assert set(targets) == {row['msgid'] for row in rows}, (language, set(targets) ^ {row['msgid'] for row in rows})
    filled = [targets[row['msgid']] for row in rows]
    for row, target in zip(rows, filled):
        problems = builder.message_problems(row['msgid'], target, glossary)
        assert not problems, (language, row['msgid'], problems)
    previous = {str(path): {message.id: message.string for message in builder.read_catalog(path)
                            if message.id and message.string and not message.fuzzy}
                for path in (builder.LOCALE_DIR / language / 'LC_MESSAGES').glob('*.po')}
    prepared[language] = rows, filled, previous

reports = {}
for language, (rows, filled, previous) in prepared.items():
    strings = scratch / f'data-art-guide-filled-{language}.json'
    strings.write_text(json.dumps(filled, ensure_ascii=False, indent=2) + '\n')
    count, errors = builder.import_worklist(language, scratch / f'data-art-guide-worklist-{language}.json',
                                           strings=strings, reviewer='codex')
    assert count == 14 and not errors, (language, count, errors)
    preserved = 0
    for path, old in previous.items():
        after = {message.id: message.string for message in builder.read_catalog(Path(path)) if message.id}
        for source, target in old.items():
            assert after[source] == target, (language, source)
            preserved += 1
    assert preserved == 2965, (language, preserved)
    review = {'schema': 1, 'language': language,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_file_sha256': hashlib.sha256(Path('docs/source/features.rst').read_bytes()).hexdigest(),
              'review_method': 'Direct Codex AI technical review; no native-speaker signoff. Selection guidance translated directly, twelve labels/descriptions reused exactly from current source-bound runtime reviews and displayed UI glossary.',
              'records': [{'domain': row['domain'], 'source': row['msgid'],
                           'source_sha256': hashlib.sha256(row['msgid'].encode()).hexdigest(),
                           'target': target, 'ui': row['ui']} for row, target in zip(rows, filled)],
              'previous_complete_translations_preserved': preserved}
    path = Path(f'docs/i18n/reviewed/guides/{language}/2026-10-05-data-art-replacement.json')
    assert not path.exists()
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    reports[language] = {'new_messages': count, 'previous_complete_translations_preserved': preserved}
    print(language, '14 current reviewed messages; all 2,965 prior translations preserved', flush=True)
audit = builder.audit(pot, builder.LANGUAGES)
for language, result in audit['languages'].items():
    assert result['total'] == result['translated'] == 2979, (language, result)
    assert not result['stale'] and not result['invalid'] and not result['label_missing'], (language, result)
(scratch / 'data-art-guide-preservation-proof.json').write_text(json.dumps({'languages': reports, 'audit': audit}, ensure_ascii=False, indent=2) + '\n')
print('PASS: complete current fourteen-message data-art guide in all nine languages', flush=True)
