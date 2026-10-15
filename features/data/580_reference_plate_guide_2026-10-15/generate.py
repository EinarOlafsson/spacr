"""Generate one guide paragraph through normal gettext update/import functions."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
SCRATCH = Path('/mnt/wd4tb/scratch/home-b-f580-ui-20261015/guide-generation')
sys.path.insert(0, str(ROOT / 'tools'))
import build_guide_i18n as builder


def identity(catalog):
    return {message.id: [message.string, sorted(message.flags), message.user_comments]
            for message in catalog if message.id}


SCRATCH.mkdir(parents=True, exist_ok=True)
templates = SCRATCH / 'templates'
templates.mkdir(exist_ok=True)
source = SCRATCH.parent / 'gettext' / 'measure_live.pot'
shutil.copyfile(source, templates / 'measure_live.pot')
locales = SCRATCH / 'locales'
receipt = {'normal_functions': ['extract', 'update_language', 'import_worklist', 'audit'],
           'scope': 'One new paragraph in measure_live; all existing messages retained exactly.',
           'review': 'Independent coordinator read all nine targets, including corrected Korean markup; AI technical review, not native-speaker signoff.',
           'template_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'locales': {}}
pending = []
for language in builder.LANGUAGES:
    relative = Path(language) / 'LC_MESSAGES' / 'measure_live.po'
    target = ROOT / 'docs' / 'i18n' / 'guides' / relative
    generated = locales / relative
    generated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target, generated)
    old = identity(builder.read_catalog(target))
    worklist = EVIDENCE / 'inputs' / f'{language}.json'
    rows = json.loads(worklist.read_text())
    assert len(rows) == 1 and rows[0]['domain'] == 'measure_live'
    row = rows[0]
    assert row['msgid'] not in old
    problems = builder.message_problems(row['msgid'], row['msgstr'], builder.load_glossary(language))
    assert not problems, (language, problems)
    builder.update_language(language, templates, locale_dir=locales)
    applied, rejected = builder.import_worklist(language, worklist, locale_dir=locales, reviewer='codex')
    assert applied == 1 and not rejected, (language, applied, rejected)
    new = identity(builder.read_catalog(generated))
    assert set(new) == set(old) | {row['msgid']}, language
    assert all(new[key] == value for key, value in old.items()), language
    assert new[row['msgid']][0] == row['msgstr'] and 'fuzzy' not in new[row['msgid']][1]
    receipt['locales'][language] = {
        'existing_messages_preserved': len(old), 'added_reviewed_messages': 1,
        'source_sha256': hashlib.sha256(row['msgid'].encode()).hexdigest(),
        'target_sha256': hashlib.sha256(row['msgstr'].encode()).hexdigest(),
        'generated_po_sha256': hashlib.sha256(generated.read_bytes()).hexdigest()}
    pending.append((generated, target))
report = builder.audit(templates, builder.LANGUAGES, locale_dir=locales)
(EVIDENCE / 'normal-page-audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
for generated, target in pending:
    shutil.copyfile(generated, target)
receipt['passed'] = True
(EVIDENCE / 'generation-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, sort_keys=True), flush=True)
