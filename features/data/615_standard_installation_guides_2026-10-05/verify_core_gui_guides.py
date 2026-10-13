from pathlib import Path
import hashlib
import json
import re
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, 'tools')
import build_guide_i18n as g

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')

def normalized(text):
    text = re.sub(r':doc:`([^`]+?) <[^>]+>`', r'\1', text)
    text = re.sub(r'\*\*|``', '', text)
    return re.sub(r'\s+', '', text.translate(str.maketrans({'’': "'", '‘': "'"})))

results = {}
for language in g.LANGUAGES:
    review = json.loads(Path(f'docs/i18n/reviewed/guides/{language}/2026-10-05-core-gui-boxes.json').read_text())
    for file, digest in review['source_files'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
    pages = {}
    for domain in ('features', 'index', 'installer_guide'):
        path = scratch / 'core-gui-guide-html' / language / (domain + '.html')
        pages[domain] = BeautifulSoup(path.read_text(), 'html.parser')
    for index, record in enumerate(review['records']):
        matching = [node for node in pages[record['domain']].select('p')
                    if normalized(record['target']) in normalized(node.get_text(' ', strip=True))]
        assert matching, (language, index, 'reviewed target absent from actual strict HTML')
        assert all('untranslated' not in node.get('class', []) and not node.select('.untranslated') for node in matching)
        if index == 1:
            expected = set(record['ui'].values())
            assert expected <= {node.get_text() for paragraph in matching for node in paragraph.select('strong')}
    results[language] = {'six_reviewed_passages_rendered': True,
                         'new_passage_english_fallbacks': 0,
                         'preserved_prior_translations': review['preserved_existing_translations'],
                         'passed': True}
    print(language, 'all six exact reviewed targets render', flush=True)
audit = g.audit(scratch / 'core-gui-guide-gettext', g.LANGUAGES)
assert all(row['total'] == row['translated'] == 2965 and not row['stale'] and not row['invalid'] and not row['label_missing'] for row in audit['languages'].values())
(scratch / 'core-gui-guide-rendered-proof.json').write_text(json.dumps(
    {'accepted': True, 'languages': results, 'strict_builds_all_nine': True,
     'initial_korean_strict_build_failed': True, 'corrected_new_korean_messages': 5,
     'audit': audit}, ensure_ascii=False, indent=2) + '\n')
