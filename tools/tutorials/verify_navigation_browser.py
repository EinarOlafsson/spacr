#!/usr/bin/env python3
"""Exercise the tutorial hierarchy in desktop and mobile Chromium.

This checks navigation, preserved lesson URLs, host breadcrumbs, language
switching, and layout. Existing local media avoids a download of the voice
library; this is not a complete media-validation or live-deployment check.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WORKSPACE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials')


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def copyfile(self, source, output):
        try:
            super().copyfile(source, output)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Navigation can cancel an in-flight poster request.


def catalog_handler(data):
    # A second positional parameter would receive Playwright's Request,
    # overriding a default argument intended to bind this catalog.
    return lambda route: route.fulfill(json=data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=WORKSPACE)
    parser.add_argument('--stage', type=Path, help='Inspect staged catalogs without replacing the live player or media')
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'refresh_2026-09-09/browser')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog_root = (args.stage or args.workspace) / 'catalog'
    catalog = json.loads((catalog_root / 'lessons_en.json').read_text())
    expected = [l['id'] for l in catalog['lessons']]
    from build_navigation import build
    navigation = build(catalog)
    main_count = sum(len(group['lessons']) for group in navigation['sections'][0]['groups'])
    classify_count = len(next(group for group in navigation['sections'][1]['groups']
                              if group['id'] == 'classify_merged')['lessons'])
    for lesson in catalog['lessons']:
        lesson['poster'] = f"{lesson['id']}/poster.jpg"
        lesson['silent'] = f"{lesson['id']}/video/{lesson['id']}_silent.mp4"
    media = (args.stage or args.workspace) / 'production'
    source = (args.workspace / 'web/index.html').read_text()
    media_url = '/' + str(media.relative_to(args.workspace))
    for attribute in ('production-root', 'audio-root', 'video4k-root'):
        source = re.sub(rf'data-{attribute}="[^"]*"', f'data-{attribute}="{media_url}"', source)

    class SessionHandler(QuietHandler):
        def translate_path(self, path):
            candidate = Path(super().translate_path(path))
            if args.stage and candidate.is_relative_to(media) and not candidate.is_file():
                return str(args.workspace / 'production' / candidate.relative_to(media))
            return str(candidate)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                functools.partial(SessionHandler, directory=str(args.workspace)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = {'checks': [], 'remote_media_tested': False,
                'catalog_lesson_count': len(expected),
                'runtime_source_commit': navigation['source_commit'],
                'missing_tutorials': navigation['missing_tutorials']}
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome',
                                               headless=True)
            context = browser.new_context(viewport={'width': 1440, 'height': 1000})
            context.route('https://huggingface.co/**', lambda route: route.abort())
            context.route(f'http://127.0.0.1:{server.server_port}/web/',
                          lambda route: route.fulfill(body=source, content_type='text/html'))
            context.route('**/web/lesson_catalog.js*', lambda route: route.fulfill(
                body='window.SPACR_LESSON_CATALOG = ' + json.dumps(catalog) + ';',
                content_type='application/javascript'))
            context.route('**/web/module_navigation.js*', lambda route: route.fulfill(
                body='window.SPACR_TUTORIAL_NAVIGATION = ' + json.dumps(navigation) + ';',
                content_type='application/javascript'))
            for path in catalog_root.glob('*.json'):
                localized = json.loads(path.read_text())
                context.route('**/web/catalog/' + path.name + '*',
                              catalog_handler(localized))
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/web/#lesson=05_home',
                      wait_until='domcontentloaded')
            page.wait_for_function("document.querySelectorAll('.lesson-link').length === " + str(len(expected)))
            ids = page.locator('.lesson-link').evaluate_all('(nodes) => nodes.map(n => n.dataset.lesson)')
            assert sorted(ids) == sorted(expected) and len(ids) == len(set(ids))
            headings = page.locator('.series-toggle strong').all_text_contents()
            assert headings == ['Main modules', 'Submodules'], headings
            assert page.locator('[data-section="main"] .lesson-link').count() == main_count
            assert page.locator('[data-section="submodules"] [data-host="classify_merged"] .lesson-link').count() == classify_count
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
