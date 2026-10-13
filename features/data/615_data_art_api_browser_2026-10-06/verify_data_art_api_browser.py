import functools
import hashlib
import http.server
import json
from pathlib import Path
import re
import threading

from playwright.sync_api import sync_playwright

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
html = scratch / 'data-art-docs-html'
out = scratch / 'data-art-api-browser-r1'
out.mkdir(exist_ok=False)
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
proof = read(scratch / 'data-art-api-preservation-proof.json')
languages = tuple(language for language in proof['languages'] if language != 'en')
catalogs = {lang: read(html / '_static/i18n/api' / (lang + '.json')) for lang in proof['languages']}
for language, record in proof['languages'].items():
    assert digest(html / '_static/i18n/api' / (language + '.json')) == record['catalog_sha256']
modules = ('spacr.qt.night_themes', 'spacr.qt.preferences', 'spacr.qt.theme')
symbols = proof['languages']['en']['changed_symbols']
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(html)))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
rows = []
try:
    with sync_playwright() as engine:
        browser = engine.chromium.launch(headless=True, executable_path='/usr/bin/google-chrome', args=['--disable-gpu'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, locale='en-US')
        for module in modules:
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            relative = 'api/' + module.replace('.', '/') + '/index.html'
            page.goto(f'http://127.0.0.1:{server.server_port}/' + relative, wait_until='networkidle')
            select = page.locator('.spacr-api-language select')
            assert select.count() == 1
            assert page.locator('script[src*="api_i18n.js"]').get_attribute('data-api-language') == 'all'
            panel_js = '''({symbol, module}) => {
                if (symbol === module) {
                    return document.querySelector('article[role="main"] > section > .spacr-api-translation');
                }
                const signature = document.getElementById(symbol);
                return signature?.parentElement?.querySelector(':scope > dd > .spacr-api-translation');
            }'''
            for language in languages:
                select.select_option(language)
                expected_lang = language.replace('_', '-')
                page.wait_for_function('''lang => {
                    const panels = [...document.querySelectorAll('.spacr-api-translation')];
                    return panels.length && panels.every(panel => panel.lang === lang);
                }''', arg=expected_lang)
                for symbol in [s for s in symbols if s == module or s.startswith(module + '.')]:
                    visible = page.evaluate('''({symbol, module}) => {
                        const signature = document.getElementById(symbol);
                        const panel = symbol === module
                            ? document.querySelector('article[role="main"] > section > .spacr-api-translation')
                            : signature?.parentElement?.querySelector(':scope > dd > .spacr-api-translation');
                        return panel && {text: panel.innerText, lang: panel.lang};
                    }''', {'symbol': symbol, 'module': module})
                    assert visible and visible['lang'] == expected_lang, (symbol, language)
                    english = catalogs['en']['symbols'][symbol]
                    localized = catalogs[language]['symbols'][symbol]
                    assert localized['source_sha256'] == english['source_sha256']
                    assert localized['text'] != english['text']
                    normalized = re.sub(r'\s+', ' ', visible['text'])
                    literals = re.findall(r'``([^`]+)``', english['text'])
                    assert all(re.sub(r'\s+', ' ', literal) in normalized for literal in literals), (symbol, language)
                    assert len(visible['text']) > 20
                    rows.append({'symbol': symbol, 'language': language, 'page': relative,
                                 'html_sha256': digest(html / relative), 'catalog_source_sha256': localized['source_sha256'],
                                 'protected_literals': len(literals), 'visible_translation_sha256': hashlib.sha256(visible['text'].encode()).hexdigest(),
                                 'passed': True})
                if language == 'ko':
                    page.screenshot(path=str(out / (module + '-ko.png')), full_page=True)
            assert not errors, errors
            page.close()
        browser.close()
    assert len(rows) == 108
    (out / 'acceptance.json').write_text(json.dumps({'passed': True, 'strict_build_output': str(html),
                                                   'scope': 'All twelve changed API symbols rendered in each of nine languages; actual browser panels and protected literals.',
                                                   'nightly_deployment_verified': False, 'cases': rows}, indent=2) + '\n')
    print('PASS: 108 actual strict-build API browser cases.', flush=True)
finally:
    server.shutdown()
    server.server_close()
    thread.join()
