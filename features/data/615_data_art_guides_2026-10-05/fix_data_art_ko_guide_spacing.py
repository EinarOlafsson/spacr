from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path('tools').resolve()))
import build_guide_i18n as builder

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
path = Path('docs/i18n/reviewed/guides/ko/2026-10-05-data-art-replacement.json')
review = json.loads(path.read_text())
record = next(record for record in review['records'] if record['source'].startswith('Under **Preferences**'))
old = record['target']
new = old.replace('**에서', '** 에서').replace('**으로', '** 으로').replace('**을', '** 을')
assert new != old
glossary = builder.load_glossary('ko')
assert not builder.message_problems(record['source'], new, glossary)
worklist = scratch / 'data-art-guide-ko-spacing-worklist.json'
worklist.write_text(json.dumps([{'domain': 'features', 'msgid': record['source'], 'hint': old,
                               'ui': record['ui'], 'msgstr': new}], ensure_ascii=False, indent=2) + '\n')
count, errors = builder.import_worklist('ko', worklist, reviewer='codex')
assert count == 1 and not errors
record['target'] = new
record['previous_target_before_strict_RST_spacing_fix'] = old
review['strict_RST_spacing_note'] = 'Korean particles are separated from closing bold UI markers to meet docutils boundaries; displayed labels and meaning are retained.'
path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
print('Corrected one reviewed Korean paragraph via normal guide importer', flush=True)
