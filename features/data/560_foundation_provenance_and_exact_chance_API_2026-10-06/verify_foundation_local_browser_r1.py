from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import hashlib
import json
import re
import threading

from playwright.sync_api import sync_playwright

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
html = scratch / 'foundation-api-docs-all-r1'
out = scratch / 'foundation-local-browser-r1'
out.mkdir(exist_ok=False)
assert (html / 'objects.inv').is_file()
reviewed = json.loads((scratch / 'foundation-api-reviewed-inputs-r2.json').read_text())
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(html)))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = 'http://127.0.0.1:' + str(server.server_address[1]) + '/'
rows = []
try:
    with sync_playwright() as engine:
        browser = engine.chromium.launch(headless=True, executable_path='/usr/bin/google-chrome', args=['--disable-gpu'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, locale='en-US')
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(base + 'api/spacr/embeddings/index.html#spacr.embeddings.encoder_entry', wait_until='networkidle', timeout=120000)
        select = page.locator('.spacr-api-language select')
        assert select.count() == 1
        assert page.locator('script[src*="api_i18n.js"]').get_attribute('data-api-language') == 'all'
        for language, targets in reviewed['languages'].items():
            select.select_option(language)
            page.wait_for_function('''lang => {
                const signature = document.getElementById('spacr.embeddings.encoder_entry');
                const panel = signature?.parentElement?.querySelector(':scope > dd > .spacr-api-translation');
                return panel && panel.lang === lang;
            }''', arg=language.replace('_', '-'))
            visible = page.evaluate('''() => {
                const signature = document.getElementById('spacr.embeddings.encoder_entry');
                const panel = signature.parentElement.querySelector(':scope > dd > .spacr-api-translation');
                return {text: panel.innerText, lang: panel.lang};
            }''')
            normalized = re.sub(r'\s+', ' ', visible['text'])
            for target in targets['API']:
                plain = target['translation'].replace('``', '').replace('**', '')
                assert re.sub(r'\s+', ' ', plain) in normalized, (language, target['label'], visible)
            assert 'UNKNOWN' in normalized and visible['lang'] == language.replace('_', '-')
            catalog = html / '_static/i18n/api' / (language + '.json')
            assert catalog.read_bytes() == (Path('docs/source/_static/i18n/api') / catalog.name).read_bytes()
            rows.append({'language': language, 'visible_translation_sha256': hashlib.sha256(visible['text'].encode()).hexdigest(),
                         'catalog_sha256': hashlib.sha256(catalog.read_bytes()).hexdigest(),
                         'both_reviewed_provider_and_checkpoint_blocks_visible_exact': True})
            if language in ('sv', 'zh_CN', 'ko'):
                page.locator('[id="spacr.embeddings.encoder_entry"]').scroll_into_view_if_needed()
                page.screenshot(path=str(out / ('encoder-entry-' + language + '.png')))
            print('PASS actual current API browser panel', language, flush=True)
        assert not errors, errors
        context.close()
        browser.close()
finally:
    server.shutdown()
    server.server_close()
    thread.join()
assert len(rows) == 9
(out / 'acceptance.json').write_text(json.dumps({'passed': True, 'local_complete_Sphinx_output': str(html),
    'source_current_API_panels': rows, 'no_actual_deployment_claim': True,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
