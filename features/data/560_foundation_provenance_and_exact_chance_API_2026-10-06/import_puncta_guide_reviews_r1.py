from pathlib import Path
import hashlib
import json
import re

import build_guide_i18n as guides

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
folder = scratch / 'foundation-puncta-guide-worklists-r1'
prepared = {}
for language in guides.LANGUAGES:
    rows = json.loads((folder / (language + '.json')).read_text())
    reviewed_path = Path('docs/i18n/reviewed/api') / language / '2026-10-05-center-pixel-puncta.json'
    reviewed = json.loads(reviewed_path.read_text())
    targets = {}
    for record in reviewed['records']:
        assert hashlib.sha256(record['source'].encode()).hexdigest() == record['source_sha256']
        targets[record['source']] = record['translation']
    catalog = guides.read_catalog(guides.LOCALE_DIR / language / 'LC_MESSAGES/make_masks.po')
    source = "``secondary_growth`` — default ``'intensity'``"
    match = re.fullmatch(r"``secondary_growth`` — (.+) ``'intensity'``", catalog.get(source).string)
    assert match
    default_term = match[1]
    headers = descriptions = 0
    for row in rows:
        assert row['domain'] == 'make_masks'
        if row['msgid'] in targets:
            row['msgstr'] = targets[row['msgid']]
            descriptions += 1
        else:
            assert re.fullmatch(r'``puncta_\w+`` — default ``.+``', row['msgid'])
            row['msgstr'] = row['msgid'].replace(' — default ', ' — ' + default_term + ' ')
            headers += 1
        assert not guides.message_problems(row['msgid'], row['msgstr'], guides.load_glossary(language)), (language, row)
    assert headers == descriptions == 6
    path = folder / (language + '-reviewed.json')
    assert not path.exists()
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    prepared[language] = {'path': str(path), 'API_review_source_path': str(reviewed_path),
                          'API_review_source_sha256': hashlib.sha256(reviewed_path.read_bytes()).hexdigest(),
                          'six_descriptions_exact_to_prior_source_bound_Codex_API_reviews': True,
                          'six_literal_headers_use_existing_reviewed_guide_default_term': default_term}
for language, record in prepared.items():
    applied, rejected = guides.import_worklist(language, Path(record['path']), reviewer='codex')
    assert applied == 12 and not rejected, (language, applied, rejected)
    print('PASS normal guide import', language, '12 current puncta reference messages', flush=True)
(scratch / 'foundation-puncta-guide-import-r1.json').write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n')
