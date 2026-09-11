#!/usr/bin/env python3
"""Browser-check website-source placeholders; this does not deploy Pages."""
import argparse
import functools
import http.server
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright
from coming_soon import COPY, PLACEHOLDERS
from stage_lesson import REPO, write
from verify_staged_lesson import Handler


def verify(web, output, *, mutate=False):
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
        functools.partial(Handler, directory=str(web)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    screens, errors = [], []
    output.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True)
            for language in COPY:
                ctx = browser.new_context(viewport={'width': 1440, 'height': 1100})
                storage = {'spacr-tutorial-language-v2': language if language not in ('da','de','is','ko','nb','sv') else 'en',
                           'spacr-tutorial-voice-v2': 'af_heart',
                           'spacr-tutorial-captions-v1': json.dumps({'language': language}),
                           'spacr-tutorial-progress-v2': json.dumps(list(PLACEHOLDERS))}
                ctx.add_init_script('Object.entries(' + json.dumps(storage) +
                                    ').forEach(([k,v]) => localStorage.setItem(k,v));')
                if mutate:
                    # Mutate the actual served source, not a test-only model.
                    source = (web / 'app_v2.js').read_text()
                    broken = source.replace('lesson.status !== "coming_soon"', 'true', 1)
                    assert broken != source
                    ctx.route('**/app_v2.js*', lambda route: route.fulfill(body=broken, content_type='text/javascript'))
                page = ctx.new_page()
                page.on('pageerror', lambda error: errors.append(str(error)))
                media = []
                page.on('request', lambda req: media.append(req.url)
                        if any(ext in req.url for ext in ('.mp4', '.m4a')) else None)
                page.goto(origin + '/#lesson=76_ops', wait_until='networkidle')
                for identity in PLACEHOLDERS:
                    page.evaluate('(id) => selectLesson(id)', identity)
                    page.wait_for_function('(s) => document.querySelector("#planned-title").textContent === s', arg=COPY[language][0])
                    assert page.locator('#planned-card').is_visible()
                    assert not page.locator('#ready-player').is_visible()
                    assert not page.locator('#complete-button').is_visible()
                    assert page.locator('#planned-copy').inner_text() == COPY[language][1]
                    assert page.locator('#available-count').inner_text() == '71'
                    assert page.locator('#total-count').inner_text() == '77'
                    assert page.locator('#progress-label').inner_text() == '0 of 71 complete'
                    assert page.evaluate('''() => {
                        const before = [...completed].sort().join();
                        toggleComplete(); markCompleteAtEnd();
                        return before === [...completed].sort().join()
                            && videoSource() === '' && audioSource() === ''
                            && !elements.video.getAttribute('src') && !elements.audio.getAttribute('src');
                    }''')
                    assert not media, media
                    screens.append({'lesson': identity, 'language': language})
                for width in (320, 390, 768, 1440):
                    page.set_viewport_size({'width': width, 'height': 1100})
                    page.evaluate('closeSidebar()')
                    page.wait_for_timeout(400)
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    assert page.locator('#planned-copy').evaluate('''el => {
                        const r = el.getBoundingClientRect();
                        return r.width > 0 && r.x >= 0 && r.right <= innerWidth
                            && el.contains(document.elementFromPoint(r.x+r.width/2, r.y+r.height/2));
                    }''')
                    if language == 'en' and width in (390, 1440):
                        page.screenshot(path=str(output / f'public-coming-soon-{width}.png'), full_page=True)
                if language == 'en':
                    # Positive counterpart: real existing public video/audio,
                    # then return to a placeholder and verify cancellation.
                    page.evaluate("selectLesson('01_pypi_github')")
                    page.wait_for_function('narrationAudioAvailable && elements.audio.readyState >= 1', timeout=60000)
                    assert page.locator('#ready-player').is_visible()
                    assert page.locator('#complete-button').is_visible()
                    assert not page.locator('#planned-card').is_visible()
                    page.evaluate('seekTo(1)')
                    page.wait_for_timeout(2000)
                    clocks = page.evaluate('''() => ({audio: elements.audio.currentTime,
                        video: elements.video.currentTime, expected: videoTimeFromAudio(elements.audio.currentTime),
                        error: elements.video.error?.message || elements.audio.error?.message || null})''')
                    assert clocks['audio'] > 1 and clocks['video'] > 0 and clocks['error'] is None, clocks
                    assert abs(clocks['video'] - clocks['expected']) < .5, clocks
                    page.evaluate("selectLesson('76_ops')")
                    assert page.evaluate('elements.audio.paused && !elements.audio.getAttribute("src")')
                assert not errors, errors
                ctx.close()
                print(language, 'six website placeholders PASS', flush=True)
            browser.close()
        report = {'scope': 'Local website source, not deployed Pages', 'published': False,
                  'placeholder_cases': screens, 'ready_playback': clocks, 'passed': True}
        write(output / 'public-browser-checks.json', report)
        return report
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web', type=Path, default=REPO / 'docs/source/_extra/tutorials')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mutate-playability', action='store_true')
    args = parser.parse_args()
    verify(args.web, args.output, mutate=args.mutate_playability)
