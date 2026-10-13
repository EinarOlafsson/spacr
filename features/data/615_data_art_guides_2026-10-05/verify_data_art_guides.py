from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

from bs4 import BeautifulSoup

sys.meta_path = [finder for finder in sys.meta_path
                 if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path('tools').resolve()))
import build_guide_i18n as builder

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
baseline = 'ccd1c7f4e'
caption = json.loads((scratch / 'data-art-guide-animation-caption-proof.json').read_text())

def normalized(value):
    value = re.sub(r':doc:`([^`]+?) <[^>]+>`', r'\1', value)
    return re.sub(r'\s+', '', re.sub(r'\*|``', '', value).translate(str.maketrans({'’': "'", '‘': "'"})))

results = {}
for language in builder.LANGUAGES:
    review = json.loads(Path(f'docs/i18n/reviewed/guides/{language}/2026-10-05-data-art-replacement.json').read_text())
    assert review['source_file_sha256'] == hashlib.sha256(Path('docs/source/features.rst').read_bytes()).hexdigest()
    changes = {}
    fix_path = Path(f'docs/i18n/reviewed/guides/{language}/2026-10-05-animation-caption-registration.json')
    if fix_path.exists():
        fixes = json.loads(fix_path.read_text())['records']
        changes = {(record['domain'], record['source']): record for record in fixes}
    preserved = repaired = 0
    for path in sorted((builder.LOCALE_DIR / language / 'LC_MESSAGES').glob('*.po')):
        previous_path = scratch / f'data-art-old-{language}-{path.name}'
        relative_path = path.relative_to(Path.cwd())
        previous_path.write_bytes(subprocess.check_output(['git', 'show', f'{baseline}:{relative_path}']))
        before = {message.id: message.string for message in builder.read_catalog(previous_path)
                  if message.id and message.string and not message.fuzzy}
        after = {message.id: message.string for message in builder.read_catalog(path) if message.id}
        for source, target in before.items():
            repair = changes.get((path.stem, source))
            if repair:
                assert target == repair['previous_translation'] and after[source] == repair['target']
                assert target.replace('**Animation**', '**' + repair['displayed_animation_caption'] + '**') == after[source]
                repaired += 1
            else:
                assert after[source] == target, (language, path.name, source)
                preserved += 1
    assert preserved + repaired == 2965 and repaired == caption['changes'][language]['changed_prior_messages']
    domains = {'features'} | {key[0] for key in changes}
    pages = {domain: BeautifulSoup((scratch / 'data-art-guide-html' / language / (domain + '.html')).read_text(), 'html.parser')
             for domain in domains}
    for record in review['records'] + list(changes.values()):
        nodes = [node for node in pages[record['domain']].select('p,h1,h2,h3,h4,h5')
                 if normalized(record['target']) in normalized(node.get_text(' ', strip=True))]
        assert nodes, (language, record['source'], 'exact reviewed target absent from actual strict HTML')
        assert all('untranslated' not in node.get('class', []) and not node.select('.untranslated') for node in nodes)
    results[language] = {'fourteen_current_data_art_messages_rendered': True,
                         'previous_complete_translations_preserved': preserved,
                         'actual_animation_caption_repairs_rendered': repaired,
                         'new_source_English_fallbacks': 0,
                         'feature_HTML_sha256': hashlib.sha256((scratch / 'data-art-guide-html' / language / 'features.html').read_bytes()).hexdigest()}
    print(language, 'all new and corrected reviewed text renders; other prior translations exact', flush=True)
audit = builder.audit(scratch / 'data-art-guide-gettext', builder.LANGUAGES)
assert all(row['total'] == row['translated'] == 2979 and not row['stale'] and not row['invalid'] and not row['label_missing'] for row in audit['languages'].values())
(scratch / 'data-art-guide-rendered-proof.json').write_text(json.dumps({'accepted': True, 'languages': results, 'audit': audit}, ensure_ascii=False, indent=2) + '\n')
print('PASS: actual strict all-nine guide HTML and complete prior-record preservation', flush=True)
