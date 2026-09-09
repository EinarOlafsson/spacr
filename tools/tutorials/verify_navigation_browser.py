#!/usr/bin/env python3
"""Exercise the tutorial hierarchy in desktop and mobile Chromium.

This checks navigation, preserved lesson URLs, host breadcrumbs, language
switching, and layout. Remote audio is blocked deliberately: media validation
is a separate gate, and this check must not download the voice library.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WORKSPACE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials')


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=WORKSPACE)
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'refresh_2026-09-09/browser')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((args.workspace / 'catalog/lessons_en.json').read_text())
    expected = [l['id'] for l in catalog['lessons']]
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                functools.partial(QuietHandler, directory=str(args.workspace)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = {'checks': [], 'remote_media_tested': False}
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome',
                                               headless=True)
            context = browser.new_context(viewport={'width': 1440, 'height': 1000})
            context.route('https://huggingface.co/**', lambda route: route.abort())
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/web/#lesson=05_home',
                      wait_until='domcontentloaded')
            page.wait_for_function("document.querySelectorAll('.lesson-link').length === 73")
            ids = page.locator('.lesson-link').evaluate_all('(nodes) => nodes.map(n => n.dataset.lesson)')
            assert sorted(ids) == sorted(expected) and len(ids) == len(set(ids))
            headings = page.locator('.series-toggle strong').all_text_contents()
            assert headings == ['Main modules', 'Submodules'], headings
            assert page.locator('[data-section="main"] .lesson-link').count() == 21
            assert page.locator('[data-section="submodules"] [data-host="classify_merged"] .lesson-link').count() == 7
            assert page.locator('[data-section="main"] [data-lesson="41_classify"]').count() == 1
            assert page.locator('[data-section="submodules"] [data-lesson="28_training_runs"]').count() == 1
            evidence['checks'].append('All existing lessons appear exactly once in the two sections')
            page.locator('[data-lesson="28_training_runs"]').click()
            page.wait_for_function("location.hash === '#lesson=28_training_runs'")
            assert 'Classify' in page.locator('#lesson-route').inner_text()
            evidence['checks'].append('Moved lesson retains URL and shows current parent breadcrumb')
            page.locator('#lesson-search').fill('Classify')
            assert page.locator('[data-lesson="28_training_runs"]').count() == 1
            page.locator('#lesson-search').fill('')
            evidence['checks'].append('Searching a parent finds its submodules')
            page.screenshot(path=str(args.output / 'desktop.png'))
            language = page.locator('#language-select')
            if not language.count():
                language = page.locator('select').filter(has=page.locator('option[value="es"]')).first
            language.select_option('es')
            page.wait_for_function("document.querySelector('.series-toggle strong').textContent === 'Módulos principales'")
            assert page.locator('.series-toggle strong').all_text_contents() == ['Módulos principales', 'Submódulos']
            evidence['checks'].append('Spanish navigation switches with the narration language')
            language.select_option('en')
            page.wait_for_function("document.querySelector('.series-toggle strong').textContent === 'Main modules'")
            page.set_viewport_size({'width': 390, 'height': 844})
            page.locator('#menu-button').click()
            page.screenshot(path=str(args.output / 'mobile.png'))
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            assert page.locator('[data-section="main"] .series-toggle').get_attribute('aria-expanded') == 'true'
            page.locator('[data-section="main"] .series-toggle').click()
            assert page.locator('[data-section="main"] .series-toggle').get_attribute('aria-expanded') == 'false'
            page.locator('[data-section="submodules"] .series-toggle').click()
            assert page.locator('[data-section="submodules"] .series-toggle').get_attribute('aria-expanded') == 'false'
            evidence['checks'].append('390-pixel mobile layout fits and both sections collapse accessibly')
            assert not errors, errors
            evidence['checks'].append('No JavaScript runtime errors')
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    evidence['passed'] = True
    (args.output / 'navigation-checks.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
