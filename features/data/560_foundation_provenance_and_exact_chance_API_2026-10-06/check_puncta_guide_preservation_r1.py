from pathlib import Path
import hashlib
import json

import build_guide_i18n as guides

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
baseline = scratch / 'foundation-guide-baseline-r1'
report = {}
def rows(path):
    return {message.id: {'string': message.string, 'flags': sorted(message.flags),
                         'user_comments': message.user_comments, 'auto_comments': message.auto_comments,
                         'locations': message.locations, 'context': message.context}
            for message in guides.read_catalog(path) if message.id}
for language in guides.LANGUAGES:
    paths = sorted((baseline / language / 'LC_MESSAGES').rglob('*.po'))
    preserved = added = 0
    for path in paths:
        relative = path.relative_to(baseline)
        current = guides.LOCALE_DIR / relative
        old, new = rows(path), rows(current)
        assert old.keys() <= new.keys(), (language, relative)
        for key, record in old.items():
            assert new[key] == record, (language, relative, key)
        additions = new.keys() - old.keys()
        if additions:
            assert relative.name == 'make_masks.po' and len(additions) == 12
            reviewed = json.loads((scratch / 'foundation-puncta-guide-worklists-r1' / (language + '-reviewed.json')).read_text())
            assert additions == {row['msgid'] for row in reviewed}
            assert all(new[row['msgid']]['string'] == row['msgstr'] and not new[row['msgid']]['flags'] for row in reviewed)
        preserved += len(old)
        added += len(additions)
    assert preserved == 2979 and added == 12
    report[language] = {'prior_complete_messages_exact': preserved, 'new_puncta_messages': added,
                        'current_total': preserved + added, 'catalog_files': len(paths),
                        'all_prior_user_and_auto_comments_flags_context_and_locations_exact': True,
                        'make_masks_sha256': hashlib.sha256((guides.LOCALE_DIR / language / 'LC_MESSAGES/make_masks.po').read_bytes()).hexdigest()}
    print('PASS guide preservation', language, preserved, 'complete prior messages and twelve additions', flush=True)
(scratch / 'foundation-puncta-guide-preservation-r1.json').write_text(json.dumps(report, indent=2) + '\n')
