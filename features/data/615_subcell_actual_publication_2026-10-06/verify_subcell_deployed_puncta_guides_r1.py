from pathlib import Path
import hashlib
import json
import re
import urllib.request
from playwright.sync_api import sync_playwright

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
assembled = scratch / 'subcell-docs-37478293839/assembled'
out = scratch / 'subcell-deployed-puncta-guides-r1'
out.mkdir(exist_ok=False)
base = 'https://einarolafsson.github.io/spacr/'
channels_bytes = urllib.request.urlopen(base + 'channels.json', timeout=90).read()
assert channels_bytes == (assembled / 'channels.json').read_bytes()
channels = json.loads(channels_bytes)
assert channels['channels']['nightly']['commit'] == 'e259c2d4ecfa34b8228a8e8eab4520ff06edcf04'
records = []
def plain(value):
    return re.sub(r'\s+', ' ', value.replace('``', '').replace('**', ''))
with sync_playwright() as engine:
    browser = engine.chromium.launch(headless=True, executable_path='/usr/bin/google-chrome', args=['--disable-gpu'])
    context = browser.new_context(viewport={'width': 1440, 'height': 1000})
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    for language in ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
        relative = 'nightly/' + language + '/make_masks.html'
        payload = urllib.request.urlopen(base + relative, timeout=90).read()
        assert payload == (assembled / relative).read_bytes(), relative
        (out / (language + '.html')).write_bytes(payload)
        reviewed_path = scratch / 'foundation-puncta-guide-worklists-r1' / (language + '-reviewed.json')
        reviewed = json.loads(reviewed_path.read_text())
        assert len(reviewed) == 12
        page.goto(base + relative, wait_until='networkidle', timeout=120000)
        visible = page.locator('article[role="main"]').inner_text()
        for row in reviewed:
            assert plain(row['msgstr']) in plain(visible), (language, row, visible)
        for marker in ('puncta_sigmas', 'puncta_k', 'puncta_center_pixels', 'puncta_min_corrected', 'puncta_min_distance', 'puncta_edge_margin'):
            assert marker in visible, (language, marker)
        if language in ('sv', 'zh_CN', 'ko'):
            page.get_by_text('puncta_sigmas', exact=False).first.scroll_into_view_if_needed()
            page.screenshot(path=str(out / (language + '.png')))
        records.append({'language': language, 'actual_url': base + relative,
                        'actual_HTML_exact_to_normal_artifact': True,
                        'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                        'all_twelve_exact_reviewed_translations_visible': True,
                        'all_six_puncta_parameters_visible': True,
                        'reviewed_messages_sha256': hashlib.sha256(reviewed_path.read_bytes()).hexdigest(),
                        'rendered_text_sha256': hashlib.sha256(visible.encode()).hexdigest()})
        print('PASS actual deployed puncta reference guide', language, flush=True)
    assert not errors, errors
    context.close()
    browser.close()
(out / 'acceptance.json').write_text(json.dumps({'passed': True, 'completed_documentation_workflow': 37478293839,
    'actual_resolved_nightly_source': channels['channels']['nightly']['commit'],
    'actual_channels_sha256': hashlib.sha256(channels_bytes).hexdigest(),
    'all_108_reviewed_message_readbacks': records,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
