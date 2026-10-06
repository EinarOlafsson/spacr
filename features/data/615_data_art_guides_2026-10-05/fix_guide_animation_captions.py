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

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
pot = scratch / 'data-art-guide-gettext'
before = builder.audit(pot, builder.LANGUAGES)
reports = {}
for language, report in before['languages'].items():
    assert report['total'] == report['translated'] == 2979 and not report['stale'] and not report['label_missing']
    if not report['invalid']:
        reports[language] = {'changed_prior_messages': 0, 'preserved_prior_complete_translations': 2965}
        continue
    displayed = builder.runtime_ui_name('Animation', language)
    assert displayed and displayed != 'Animation'
    glossary = builder.load_glossary(language)
    rows = []
    changes = []
    for problem in report['invalid']:
        assert problem['page'] in {'localization', 'setting_animations'}
        assert all(detail == f"UI name 'Animation' must read **{displayed}**" for detail in problem['problems'])
        catalog = builder.read_catalog(builder.LOCALE_DIR / language / 'LC_MESSAGES' / (problem['page'] + '.po'))
        matches = [message for message in catalog if message.id and message.id.startswith(problem['msgid'])]
        assert len(matches) == 1
        message = matches[0]
        old = message.string
        new = old.replace('**Animation**', '**' + displayed + '**')
        assert new != old and not builder.message_problems(message.id, new, glossary)
        rows.append({'domain': problem['page'], 'msgid': message.id, 'hint': old,
                     'ui': {label: glossary[label] for label in builder.ui_names(message.id) if label in glossary},
                     'msgstr': new})
        changes.append({'domain': problem['page'], 'source': message.id,
                        'source_sha256': hashlib.sha256(message.id.encode()).hexdigest(),
                        'previous_translation': old, 'target': new,
                        'displayed_animation_caption': displayed})
    assert len(rows) == 2
    worklist = scratch / f'guide-animation-caption-fix-{language}.json'
    worklist.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    count, errors = builder.import_worklist(language, worklist, reviewer='codex')
    assert count == 2 and not errors
    review = {'schema': 1, 'language': language,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'review_method': 'Direct Codex AI technical review; no native-speaker signoff. Only the displayed Animation caption is changed to the actual first-run-owned runtime row; all surrounding reviewed wording is retained. Previous target text is retained as historical evidence.',
              'records': changes}
    path = Path(f'docs/i18n/reviewed/guides/{language}/2026-10-05-animation-caption-registration.json')
    assert not path.exists()
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    reports[language] = {'changed_prior_messages': count, 'preserved_prior_complete_translations': 2963}
    print(language, 'two exact displayed-caption repairs', flush=True)
after = builder.audit(pot, builder.LANGUAGES)
for language, report in after['languages'].items():
    assert report['total'] == report['translated'] == 2979
    assert not report['stale'] and not report['invalid'] and not report['label_missing'], (language, report)
(scratch / 'data-art-guide-animation-caption-proof.json').write_text(json.dumps({'before_invalid': {lang: row['invalid'] for lang, row in before['languages'].items()}, 'changes': reports, 'audit': after}, ensure_ascii=False, indent=2) + '\n')
print('PASS: all nine current complete guide audits including actual Animation captions', flush=True)
