import functools
import hashlib
import http.server
import json
import re
import sys
import threading
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

root = Path('docs/source')
url = 'https://einarolafsson.github.io/spacr/nightly/'
out = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/live-current-api-browser')
out.mkdir(exist_ok=False)
languages = ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr')
cases = (
    ('qt/cpu_modes', 'spacr.qt.cpu_modes.puncta'),
    ('plot', 'spacr.plot.figure_output_preferences'),
    ('qt/shortcuts', 'spacr.qt.shortcuts'),
    ('core', 'spacr.core.preprocess_generate_masks'),
)
catalogs = {language: json.loads((root / '_static/i18n/api' / (language + '.json')).read_text())
            for language in ('en', *languages)}
catalog_readback = {}
for language in ('en', *languages):
    relative = '_static/i18n/api/' + language + '.json'
    payload = urllib.request.urlopen(url + relative, timeout=90).read()
    expected = (root / relative).read_bytes()
    assert payload == expected, language
    catalog_readback[language] = hashlib.sha256(payload).hexdigest()
    print('Live API catalog hash PASS:', language, flush=True)
reports = []
try:
    with sync_playwright() as engine:
        browser = engine.chromium.launch(headless=True, executable_path='/usr/bin/google-chrome', args=['--disable-gpu'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, locale='en-US')
        for module, symbol in cases:
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page_url = url + 'api/spacr/' + module + '/index.html'
            page_payload = urllib.request.urlopen(page_url, timeout=90).read()
            page.goto(page_url, wait_until='networkidle', timeout=120000)
            select = page.locator('.spacr-api-language select')
            assert select.count() == 1
            assert page.locator('script[src*="api_i18n.js"]').get_attribute('data-api-language') == 'all'
            if symbol != 'spacr.qt.shortcuts':
                assert page.locator('[id]').evaluate_all('(nodes, symbol) => nodes.some(node => node.id === symbol)', symbol)
            else:
                assert page.locator('section[id="module-spacr.qt.shortcuts"] > h1').count() == 1
            for language in languages:
                select.select_option(language)
                page.wait_for_function('''({symbol, language}) => {
                    const signature = document.getElementById(symbol);
                    const panel = signature?.parentElement?.querySelector(':scope > dd > .spacr-api-translation')
                        || document.querySelector(':scope article .spacr-api-translation');
                    return panel?.lang === language.replace('_', '-');
                }''', arg={'symbol': symbol, 'language': language}, timeout=60000)
                visible = page.evaluate('''(symbol) => {
                    const signature = document.getElementById(symbol);
                    const panel = signature?.parentElement?.querySelector(':scope > dd > .spacr-api-translation')
                        || document.querySelector('article .spacr-api-translation');
                    return {text: panel.innerText, lang: panel.lang};
                }''', symbol)
                record = catalogs[language]['symbols'][symbol]
                english = catalogs['en']['symbols'][symbol]
                assert record['source_sha256'] == english['source_sha256']
                assert record['text'] != english['text']
                literals = re.findall(r'``([^`]+)``', english['text'])
                normalized = re.sub(r'\s+', ' ', visible['text'])
                assert all(re.sub(r'\s+', ' ', literal) in normalized for literal in literals), (symbol, language, literals, visible)
                assert len(visible['text']) > 80 and visible['lang'] == language.replace('_', '-')
                page.screenshot(path=str(out / f'{module.replace("/", "_")}-{language}.png'), full_page=True)
                reports.append({'symbol': symbol, 'language': language, 'page_url': page_url, 'page_sha256': hashlib.sha256(page_payload).hexdigest(),
                                'catalog_source_sha256': record['source_sha256'], 'literal_count': len(literals),
                                'translated_panel_text_sha256': hashlib.sha256(visible['text'].encode()).hexdigest(), 'passed': True})
            assert not errors, errors
            page.close()
        browser.close()
    assert len(reports) == 36
    (out / 'acceptance.json').write_text(json.dumps({'passed': True, 'scope': 'Actual deployed nightly Sphinx API pages in all nine locales, including puncta, figure preference literals, shortcut prose and current preprocessing.', 'catalog_readback_sha256': catalog_readback, 'cases': reports}, ensure_ascii=False, indent=2) + '\n')
    print('36 LIVE Sphinx-page locale cases passed with literal preservation.', flush=True)
finally:
    pass
