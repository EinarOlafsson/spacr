#!/usr/bin/env python3
"""Observe the real screen/completion checks fail against in-memory mutants.

No source or candidate file is modified. Only two deliberate browser-served
code variants change; positive baseline checks run before and after them.
"""
import argparse
import functools
import http.server
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

from check_completed_matrix import digest
from append_staged_lessons import JS_SUFFIX, parse_javascript
from audit_staged_catalogs import CATALOGS
from coming_soon import COPY, placeholder_probe
from stage_lesson import read, write
from verify_staged_lesson import Handler


def verify(root):
    source = (root / 'web/app_v2.js').read_text()
    catalog = read(root / 'web/catalog/lessons_en.json')['lessons']
    held = [lesson['id'] for lesson in catalog if lesson.get('status') == 'coming_soon']
    # With no real unavailable lesson, serve an in-memory probe (never written).
    probe = None if held else next(lesson['id'] for lesson in reversed(catalog)
                                   if lesson.get('status') != 'coming_soon')
    unavailable = held[0] if held else probe
    ready_count = sum(lesson.get('status') != 'coming_soon' for lesson in catalog) - (probe is not None)
    served = {}
    if probe:
        js_text = (root / 'web/lesson_catalog.js').read_text()
        js = parse_javascript(js_text)
        js['lessons'] = placeholder_probe(js['lessons'], probe)
        served['lesson_catalog.js'] = (js_text[:js_text.index('Object.freeze(') + len('Object.freeze(')]
                                       + json.dumps(js, ensure_ascii=False) + JS_SUFFIX)
        for filename in CATALOGS:
            language = filename.split('_', 1)[1].removesuffix('.json')
            if language not in COPY:
                continue
            localized = read(root / 'web/catalog' / filename)
            localized['lessons'] = placeholder_probe(localized['lessons'], probe, language)
            served['catalog/' + filename] = json.dumps(localized, ensure_ascii=False)
    variants = [
        ('availability guard', 'lesson.status !== "coming_soon"', 'true'),
        ('completion guard', 'function toggleComplete() {\n  if (!isPlayable(activeLesson)) return;',
         'function toggleComplete() {'),
    ]
    def serve(body, kind):
        # One-argument handler: Playwright passes the request too when it can.
        def handler(route):
            route.fulfill(body=body, content_type=kind)
        return handler

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
             functools.partial(Handler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    observed = []
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True)
            def check(code):
                context = browser.new_context()
                context.route('**/app_v2.js*', lambda route: route.fulfill(body=code, content_type='application/javascript'))
                for name, body in served.items():
                    context.route(f'**/web/{name}*', serve(body, 'application/javascript'
                                                         if name.endswith('.js') else 'application/json'))
                page = context.new_page()
                try:
                    page.goto(origin + '/web/#lesson=' + unavailable, wait_until='networkidle')
                    page.wait_for_function('(identity) => activeLesson?.id === identity', arg=unavailable)
                    assert page.locator('#planned-card').is_visible()
                    assert page.locator('#available-count').inner_text() == str(ready_count)
                    page.evaluate('toggleComplete()')
                    assert page.evaluate('(identity) => !completed.has(identity)', unavailable)
                finally:
                    context.close()
            check(source)
            for name, old, new in variants:
                assert source.count(old) == 1
                try:
                    check(source.replace(old, new))
                except AssertionError:
                    observed.append({'guard': name, 'observed_red': True})
                    print(name, 'OBSERVED RED', flush=True)
                else:
                    raise AssertionError(f'Mutant survived: {name}')
            check(source)
            browser.close()
        write(root / 'checks/placeholder-mutation-checks.json', {
              'player_sha256': digest(root / 'web/app_v2.js'), 'mutations': observed,
              'baseline_before_and_after_passed': True, 'source_files_modified': False,
              'tested_placeholder': unavailable,
              'placeholder_source': ('in-memory probe; the candidate has no unavailable lesson'
                                     if probe else 'actual unavailable lesson'),
              'passed': True, 'published': False})
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    verify(parser.parse_args().candidate.resolve())
