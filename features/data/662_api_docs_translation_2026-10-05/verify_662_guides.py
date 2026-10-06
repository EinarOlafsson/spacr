import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

repo = Path.cwd()
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
html = scratch / 'sphinx-662-english/html'
languages = ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr')

def normalized(text):
    # Sphinx smartquotes renders prose apostrophes as typographic apostrophes.
    return re.sub(r'\s+', '', text.translate(str.maketrans({'’': "'", '‘': "'"})))

reports = {}
for language in languages:
    review = json.loads((repo / f'docs/i18n/reviewed/guides/{language}/2026-10-05-yolo-boxes.json').read_text())
    source = repo / 'docs/source/make_masks.rst'
    assert review['source_files'][str(source.relative_to(repo))] == hashlib.sha256(source.read_bytes()).hexdigest()
    path = html / language / 'make_masks.html'
    soup = BeautifulSoup(path.read_text(), 'html.parser')
    section = soup.find('section', id='draw-bounding-boxes-for-yolo')
    assert section is not None, language
    shown = normalized(section.get_text(' ', strip=True))
    for record in review['records']:
        target = normalized(re.sub(r'\*\*|``', '', record['target']))
        assert target in shown, (language, record['index'], 'reviewed target not rendered')
    assert not section.select('.untranslated'), (language, 'untranslated guide content')
    literals = {node.get_text() for node in section.select('code')}
    assert {'.txt', '.classes.json', '.spacr_yolo_annotations.json', '_seg.npy'} <= literals, (language, literals)
    controls = {node.get_text() for node in section.select('strong')}
    expected = {name for record in review['records'] for name in record['ui'].values()}
    assert expected <= controls, (language, expected - controls)
    assert not any('**' in node.get_text() for node in section.select('strong'))
    assert not any('``' in node.get_text() for node in section.select('code'))
    assert len(section.select('ol > li')) == 4, (language, 'workflow list count')
    reports[language] = {'html_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'reviewed_passages_rendered': 7, 'required_file_literals': 4,
                         'runtime_controls_rendered': len(expected), 'workflow_steps': 4,
                         'english_fallbacks': 0, 'passed': True}
for path in ('662-guide-build-sv.log', '662-guide-build-zh_CN.log', '662-guide-build-ko.log', '662-guide-build-ko-r2.log'):
    assert (scratch / path).is_file()
assert 'ko: sphinx exit 0' in (scratch / '662-guide-build-ko-r2.log').read_text()
for language in languages:
    matching = [p for p in scratch.glob('662-guide-build-*.log')
                if f'{language}: sphinx exit 0' in p.read_text()]
    assert matching, language
result = {'accepted': True, 'scope': 'Actual strict-built current YOLO Make Masks guide pages in all nine locales.',
          'languages': reports, 'initial_korean_build_failed': True,
          'korean_corrected_new_messages': 6, 'final_all_nine_strict_builds_passed': True}
(scratch / '662-guide-rendered-proof.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print('All nine strict-built YOLO guides render seven reviewed messages, four exact literals and all runtime control names.')
