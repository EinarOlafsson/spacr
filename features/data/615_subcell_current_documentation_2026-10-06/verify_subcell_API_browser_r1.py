from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import hashlib
import json
import re
import threading
from playwright.sync_api import sync_playwright

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
html = scratch / 'subcell-api-docs-all-r1'
out = scratch / 'subcell-API-browser-r1'
out.mkdir(exist_ok=False)
assert (html / 'objects.inv').is_file()
source_sha = hashlib.sha256(Path('spacr/embeddings.py').read_bytes()).hexdigest()
assert source_sha == json.loads((scratch / 'subcell-rybg-documentation-inventory-r4.json').read_text())['source_sha256']
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(html)))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = 'http://127.0.0.1:' + str(server.server_address[1]) + '/'
result = []
def plain(value):
    return re.sub(r'\s+', ' ', re.sub(r':[\w:]+:`([^`]+)`', r'\1', value).replace('``', '').replace('**', ''))
try:
    with sync_playwright() as engine:
        browser = engine.chromium.launch(headless=True, executable_path='/usr/bin/google-chrome', args=['--disable-gpu'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        prior = json.loads((scratch / 'foundation-api-reviewed-inputs-r3.json').read_text())['languages']
        for language in ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
            reviewed = json.loads((Path('docs/i18n/reviewed/api') / language / '2026-10-06-subcell-rybg.json').read_text())['records']
            assert len(reviewed) == 8
            grouped = {}
            for row in reviewed + prior[language]['API']:
                symbol = row['label'].rpartition('#')[0]
                grouped.setdefault(symbol, []).append(row['translation'])
            panels = []
            page.goto(base + 'api/spacr/embeddings/index.html', wait_until='networkidle', timeout=120000)
            select = page.locator('.spacr-api-language select')
            assert select.count() == 1
            select.select_option(language)
            for symbol, translations in grouped.items():
                page.wait_for_function('''args => {
                    const signature = document.getElementById(args.symbol);
                    const panel = signature?.parentElement?.querySelector(':scope > dd > .spacr-api-translation');
                    return panel && panel.lang === args.lang;
                }''', arg={'symbol': symbol, 'lang': language.replace('_', '-')})
                visible = page.evaluate('''symbol => document.getElementById(symbol).parentElement.querySelector(':scope > dd > .spacr-api-translation').innerText''', symbol)
                for translation in translations:
                    assert plain(translation) in plain(visible), (language, symbol, translation, visible)
                panels.append({'symbol': symbol, 'exact_reviewed_blocks_visible': len(translations), 'visible_sha256': hashlib.sha256(visible.encode()).hexdigest()})
            if language in ('sv', 'zh_CN', 'ko'):
                page.locator('[id="spacr.embeddings.EmbeddingSpec"]').scroll_into_view_if_needed()
                page.screenshot(path=str(out / ('spec-' + language + '.png')))
            symbol = 'spacr.qt.screens.embeddings.EmbeddingsScreen._subcell_channels_dialog.accept_mapping'
            page.goto(base + 'api/spacr/qt/screens/embeddings/index.html#' + symbol, wait_until='networkidle', timeout=120000)
            page.locator('.spacr-api-language select').select_option(language)
            page.wait_for_function('''args => {
                const signature = document.getElementById(args.symbol);
                const panel = signature?.parentElement?.querySelector(':scope > dd > .spacr-api-translation');
                return panel && panel.lang === args.lang;
            }''', arg={'symbol': symbol, 'lang': language.replace('_', '-')})
            visible = page.evaluate('''symbol => document.getElementById(symbol).parentElement.querySelector(':scope > dd > .spacr-api-translation').innerText''', symbol)
            catalog_path = html / '_static/i18n/api' / (language + '.json')
            assert catalog_path.read_bytes() == (Path('docs/source/_static/i18n/api') / catalog_path.name).read_bytes()
            catalog = json.loads(catalog_path.read_text())
            assert len(catalog['symbols']) == 13182
            assert plain(catalog['symbols'][symbol]['text']) in plain(visible)
            result.append({'language': language, 'reviewed_panels': panels, 'new_callback_visible': True, 'catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest()})
            print('PASS actual SubCell browser panels and new mapping callback', language, flush=True)
        assert not errors, errors
        context.close()
        browser.close()
finally:
    server.shutdown()
    server.server_close()
    thread.join()
(out / 'acceptance.json').write_text(json.dumps({'passed': True, 'application_source_sha256': source_sha, 'languages': result, 'eight_new_and_seven_prior_reviewed_blocks_rendered': True, 'no_deployment_claim': True, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
