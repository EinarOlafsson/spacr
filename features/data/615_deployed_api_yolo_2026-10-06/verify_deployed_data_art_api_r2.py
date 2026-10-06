import hashlib
import json
from pathlib import Path
import re
import urllib.request

from playwright.sync_api import sync_playwright

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
assembled = scratch / 'docs-37419061863/assembled'
html = assembled / 'nightly'
out = scratch / 'deployed-data-art-api-r2'
out.mkdir(exist_ok=False)
read = lambda p: json.loads(p.read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
proof = read(scratch / 'data-art-api-preservation-proof.json')
url = 'https://einarolafsson.github.io/spacr/nightly/'
channels = read(assembled / 'channels.json')
assert channels['channels']['nightly']['commit'] == '7d24dcf7760e63f38a85c311da3bd477d0be9479'
languages = tuple(language for language in proof['languages'] if language != 'en')
catalogs = {}
catalog_hashes = {}
for language, record in proof['languages'].items():
    relative = '_static/i18n/api/' + language + '.json'
    payload = urllib.request.urlopen(url + relative, timeout=90).read()
    assert payload == (html / relative).read_bytes()
    assert payload == (Path('docs/source') / relative).read_bytes()
    assert sha(payload) == record['catalog_sha256']
    catalogs[language] = json.loads(payload)
    catalog_hashes[language] = sha(payload)
    print('PASS deployed API catalog:', language, flush=True)
modules = ('spacr.qt.night_themes', 'spacr.qt.preferences', 'spacr.qt.theme')
symbols = proof['languages']['en']['changed_symbols']
rows = []
with sync_playwright() as engine:
    browser = engine.chromium.launch(headless=True, executable_path='/usr/bin/google-chrome', args=['--disable-gpu'])
    context = browser.new_context(viewport={'width': 1440, 'height': 1000}, locale='en-US')
    for module in modules:
        relative = 'api/' + module.replace('.', '/') + '/index.html'
        payload = urllib.request.urlopen(url + relative, timeout=90).read()
        assert payload == (html / relative).read_bytes()
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url + relative, wait_until='networkidle', timeout=120000)
        select = page.locator('.spacr-api-language select')
        assert select.count() == 1
        assert page.locator('script[src*="api_i18n.js"]').get_attribute('data-api-language') == 'all'
        for language in languages:
            select.select_option(language)
            page.wait_for_function('''lang => {
                const panels = [...document.querySelectorAll('.spacr-api-translation')];
                return panels.length && panels.every(panel => panel.lang === lang);
            }''', arg=language.replace('_', '-'))
            for symbol in [s for s in symbols if s == module or s.startswith(module + '.')]:
                visible = page.evaluate('''({symbol, module}) => {
                    const signature = document.getElementById(symbol);
                    const panel = symbol === module
                        ? document.querySelector('article[role="main"] > section > .spacr-api-translation')
                        : signature?.parentElement?.querySelector(':scope > dd > .spacr-api-translation');
                    return panel && {text: panel.innerText, lang: panel.lang};
                }''', {'symbol': symbol, 'module': module})
                assert visible and visible['lang'] == language.replace('_', '-')
                english = catalogs['en']['symbols'][symbol]
                localized = catalogs[language]['symbols'][symbol]
                assert localized['source_sha256'] == english['source_sha256']
                assert localized['text'] != english['text']
                literals = re.findall(r'``([^`]+)``', english['text'])
                normalized = re.sub(r'\s+', ' ', visible['text'])
                assert all(re.sub(r'\s+', ' ', literal) in normalized for literal in literals), (symbol, language)
                assert len(visible['text']) > 20
                rows.append({'symbol': symbol, 'language': language, 'page_url': url + relative,
                             'page_sha256': sha(payload), 'catalog_source_sha256': localized['source_sha256'],
                             'protected_literals': len(literals), 'visible_translation_sha256': sha(visible['text'].encode()),
                             'passed': True})
            if language == 'ko':
                page.screenshot(path=str(out / (module + '-ko.png')), full_page=True)
        assert not errors, errors
        page.close()
    context.close()
    browser.close()
assert len(rows) == 108
(out / 'acceptance.json').write_text(json.dumps({'passed': True, 'completed_documentation_workflow': 37419061863,
    'source_commit': channels['channels']['nightly']['commit'], 'normal_assembled_reference': str(assembled),
    'actual_deployed_API_catalogs_exact': catalog_hashes, 'cases': rows}, indent=2) + '\n')
print('PASS: all 108 deployed API browser cases; complete current catalogs and generated HTML exact.', flush=True)
