#!/usr/bin/env python3
"""Exercise the actual private player: unavailable screens and ready playback.

No remote network, UI mocks, synthesis, or deployment. The server exposes only
the candidate directory and supports the real player's media byte ranges.

``--published <pages tree>`` runs the same checks on a Pages tree built from
the candidate, whose index pins narration and 4K to one immutable hosted
commit. The player's own files are still served locally (as Pages would), the
media are fetched from the host, and only that commit's URLs (plus the host's
download redirects) are allowed. Loading anything else, including
``resolve/main``, fails the run.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

from check_completed_matrix import digest
from coming_soon import COPY, first_placeholder
from stage_lesson import read, write
from validate_candidate import validate
from verify_staged_lesson import Handler

PINNED_ROOT = r'https://huggingface\.co/datasets/[^/]+/[^/]+/resolve/[0-9a-f]{40}'
REDIRECT_HOSTS = ('.hf.co', '.huggingface.co')


def check_sentence_cues(page):
    """Observe native captions at each actual narrated sentence's midpoint."""
    sentences = page.evaluate('audioTimings.scenes.flatMap(scene => scene.sentences)')
    assert sentences
    cases = []
    for sentence in sentences:
        requested = (sentence['speech_start'] + sentence['speech_end']) / 2
        page.wait_for_function('!videoClockCorrectionPending && !elements.video.seeking && !elements.audio.seeking')
        page.evaluate('(seconds) => seekTo(seconds)', requested)
        page.wait_for_timeout(300)
        page.evaluate('elements.video.pause()')
        page.wait_for_timeout(150)
        actual = page.evaluate('''() => ({audio: elements.audio.currentTime,
            video: elements.video.currentTime,
            cues: [...(elements.captionTrack.track.activeCues || [])].map(c => c.text)})''')
        assert abs(actual['audio'] - requested) < 1, (requested, actual)
        assert sentence['text'] in actual['cues'], (sentence, actual)
        cases.append({'requested_audio_time': requested, 'text': sentence['text'], **actual})
    return cases


def published_tree(root, pages):
    """Return the pinned media root after proving the tree is the candidate's web bytes."""
    index = (pages / 'index.html').read_text(encoding='utf-8')
    roots = set(re.findall(r'data-(?:audio|video4k)-root="([^"]+)"', index))
    if len(roots) != 1 or not re.fullmatch(PINNED_ROOT, next(iter(roots))):
        raise ValueError(f'The published index must pin one immutable media root, found {roots}')
    for record in read(root / 'release-manifest.json')['files']:
        if record['path'].startswith('web/') and record['path'] != 'web/index.html':
            if digest(pages / record['path'][len('web/'):]) != record['sha256']:
                raise ValueError(f"Pages tree differs from the candidate: {record['path']}")
    return next(iter(roots))


def verify(root, *, placeholders_only=False, published=None):
    root = Path(root).resolve()
    validate(root, include_hosted_media=published is None)
    records = {r['path']: r for r in read(root / 'release-manifest.json')['files']}
    if published is None:
        served, entry, media_root = root, '/web/', None
    else:
        published = Path(published).resolve()
        media_root = published_tree(root, published)
        served, entry = published.parent, '/' + published.name + '/'
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
            functools.partial(Handler, directory=str(served)))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    output = root / 'checks'
    output.mkdir(exist_ok=True)
    errors, foreign, screens, playback, hosted = [], [], [], [], []
    english = read(root / 'web/catalog/lessons_en.json')
    ready = [l for l in english['lessons'] if l.get('status') != 'coming_soon']
    placeholders = [l['id'] for l in english['lessons'] if l.get('status') == 'coming_soon']
    unavailable = first_placeholder(english['lessons'])

    def allowed_remote(url):
        if not media_root:
            return False
        host = urlparse(url).hostname or ''
        return url.startswith(media_root + '/') or (host != 'huggingface.co' and host.endswith(REDIRECT_HOSTS))

    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True)
            def context(language='en', width=1440):
                ctx = browser.new_context(viewport={'width': width, 'height': 1100})
                def no_remote(route):
                    url = route.request.url
                    if url.startswith(origin + '/'):
                        route.continue_()
                    elif allowed_remote(url):
                        hosted.append(url)
                        route.continue_()
                    else:
                        foreign.append(url)
                        route.abort()
                ctx.route('**/*', no_remote)
                caption_only = language in ('da', 'de', 'is', 'ko', 'nb', 'sv')
                storage = {'spacr-tutorial-language-v2': 'en' if caption_only else language,
                           'spacr-tutorial-voice-v2': 'af_heart',
                           'spacr-tutorial-captions-v1': json.dumps({'language': language, 'enabled': True}),
                           # Old completion records must not count held entries.
                           'spacr-tutorial-progress-v2': json.dumps(placeholders)}
                ctx.add_init_script('Object.entries(' + json.dumps(storage) +
                                    ').forEach(([k,v]) => localStorage.setItem(k,v));')
                return ctx

            for language in COPY:
                ctx = context(language)
                page = ctx.new_page()
                page.on('pageerror', lambda error: errors.append(str(error)))
                media_requests = []
                page.on('request', lambda req: media_requests.append(req.url)
                        if any(ext in req.url for ext in ('.mp4', '.m4a')) else None)
                page.goto(origin + entry + '#lesson=' + unavailable, wait_until='networkidle')
                for identity in placeholders:
                    page.evaluate('(id) => selectLesson(id)', identity)
                    page.wait_for_function('(text) => document.querySelector("#planned-title").textContent === text', arg=COPY[language][0])
                    assert page.locator('#planned-card').is_visible()
                    assert not page.locator('#ready-player').is_visible()
                    assert not page.locator('#complete-button').is_visible()
                    assert page.locator('#planned-copy').inner_text() == COPY[language][1]
                    assert page.locator('#available-count').inner_text() == str(len(ready))
                    assert page.locator('#total-count').inner_text() == str(len(english['lessons']))
                    assert page.locator('#progress-label').inner_text() == f'0 of {len(ready)} complete'
                    assert page.evaluate('''() => {
                        const before = [...completed].sort().join();
                        toggleComplete(); markCompleteAtEnd();
                        return before === [...completed].sort().join()
                            && videoSource() === '' && audioSource() === ''
                            && !elements.video.getAttribute('src') && !elements.audio.getAttribute('src');
                    }''')
                    assert media_requests == [], media_requests
                    screens.append({'lesson': identity, 'language': language, 'passed': True})
                for width in (320, 390, 768, 1440):
                    page.set_viewport_size({'width': width, 'height': 1100})
                    page.evaluate('closeSidebar()')
                    page.wait_for_timeout(400)  # Let the actual drawer transition finish.
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (language, width)
                    box = page.locator('#planned-copy').bounding_box()
                    assert box and box['width'] > 0 and box['height'] > 0
                    assert box['x'] >= 0 and box['x'] + box['width'] <= width, (language, width, box)
                    assert page.locator('#planned-copy').evaluate('''el => {
                        const r = el.getBoundingClientRect();
                        return el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2));
                    }'''), (language, width, 'placeholder text obscured')
                    if language in ('en', 'de', 'ja') and width in (390, 1440):
                        prefix = 'published-' if published else ''
                        page.screenshot(path=str(output / f'{prefix}coming-soon-{language}-{width}.png'), full_page=True)
                assert not errors, errors
                print(language, len(placeholders), 'Coming soon screens PASS', flush=True)
                ctx.close()

            if not placeholders_only:
                ctx = context()
                page = ctx.new_page()
                page.on('pageerror', lambda error: errors.append(str(error)))
                requested_urls = []
                page.on('request', lambda req: requested_urls.append(req.url))
                page.goto(origin + entry + '#lesson=' + unavailable, wait_until='networkidle')
                for lesson in ready:
                    identity = lesson['id']
                    page.evaluate('(id) => selectLesson(id)', identity)
                    page.wait_for_function('narrationAudioAvailable && elements.audio.readyState >= 1 && chapterData.length > 0', timeout=60000)
                    assert page.locator('#ready-player').is_visible()
                    assert not page.locator('#planned-card').is_visible()
                    assert page.locator('#complete-button').is_visible()
                    assert page.locator('#voice-select').is_enabled()
                    assert page.locator('.chapter-button').count() == len(lesson['scenes'])
                    loaded = page.evaluate('''async () => {
                        const response = await fetch(elements.audio.src);
                        const bytes = await response.arrayBuffer();
                        const hash = await crypto.subtle.digest('SHA-256', bytes);
                        return Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2,'0')).join('');
                    }''')
                    relative = f'{identity}/audio/en/af_heart.m4a'
                    if media_root:
                        expected = records['media_host/' + relative]['sha256']
                        assert f'{media_root}/{relative}' in requested_urls, (identity, 'not loaded from the pinned host')
                    else:
                        expected = digest(root / 'media_host' / relative)
                    assert loaded == expected, identity
                    paired = page.evaluate('''() => ({voice: audioTimings.voice,
                        hash: audioTimings.media_sha256,
                        chapters: chapterData.map(c => c.text),
                        spoken: audioTimings.scenes.map(s => s.text)})''')
                    assert paired['voice'] == 'af_heart' and paired['hash'] == loaded, identity
                    assert paired['chapters'] == paired['spoken'], identity
                    page.wait_for_function('!videoClockCorrectionPending && !elements.video.seeking && !elements.audio.seeking')
                    requested = page.evaluate('chapterData[Math.min(2, chapterData.length - 1)].start')
                    seek_started = time.monotonic()
                    page.evaluate('(seconds) => seekTo(seconds)', requested)
                    page.wait_for_timeout(1500)
                    page.wait_for_function('elements.captionTrack.readyState === 2 && !captionTrackLoading')
                    # A cold seek may still be loading media or decoding a frame,
                    # including local files while other media jobs are active.
                    # Keep the clock tolerance unchanged while waiting for it.
                    page.wait_for_function('''!videoClockCorrectionPending &&
                        !elements.video.seeking && !elements.audio.seeking &&
                        Math.abs(elements.video.currentTime -
                            videoTimeFromAudio(elements.audio.currentTime)) < .5''',
                        timeout=15000, polling=50)
                    clocks = page.evaluate('''() => ({audio: elements.audio.currentTime,
                        video: elements.video.currentTime, expected: videoTimeFromAudio(elements.audio.currentTime),
                        width: elements.video.videoWidth, height: elements.video.videoHeight,
                        error: elements.video.error?.message || elements.audio.error?.message || null})''')
                    assert clocks['error'] is None and clocks['audio'] > 0, (identity, clocks)
                    clocks['requested_audio_time'] = requested
                    clocks['seek_elapsed_seconds'] = time.monotonic() - seek_started
                    assert requested - .25 <= clocks['audio'] < requested + clocks['seek_elapsed_seconds'] + .5, (identity, clocks)
                    assert abs(clocks['video'] - clocks['expected']) < .5, (identity, clocks)
                    assert (clocks['width'], clocks['height']) == (2560, 1440), (identity, clocks)
                    for _ in range(2):
                        page.evaluate('renderCaptions()')
                        page.wait_for_function('elements.captionTrack.readyState === 2 && !captionTrackLoading')
                        assert page.locator('#caption-track').count() == 1
                    sentence_cues = check_sentence_cues(page) if identity in {
                        '04_platform_installers', '12_map_barcodes', '21_model_compare', '22_model_zoo', '76_ops', '77_embeddings'} else []
                    # The positive playable counterpart is followed by a real
                    # transition back to unavailable, cancelling active audio.
                    page.evaluate('(identity) => selectLesson(identity)', unavailable)
                    assert page.evaluate('elements.audio.paused && !elements.audio.getAttribute("src")')
                    playback.append({'lesson': identity, 'audio_sha256': loaded, 'clocks': clocks,
                                     'chapter_text_matches_audio': True, 'native_caption_reloads': 2,
                                     'sentence_cue_checks': sentence_cues, 'passed': True})
                    print(identity, 'candidate playback PASS', flush=True)
                    assert not errors, errors
                ctx.close()
            browser.close()
        if foreign or errors:
            raise ValueError(f'Unexpected external requests/errors: {foreign}, {errors}')
        result = {'scope': 'Candidate browser checks, not native-language listening or deployment',
                  'manifest_sha256': digest(root / 'release-manifest.json'),
                  'placeholder_cases': screens, 'ready_playback_cases': playback,
                  'passed': True, 'published': False}
        if published:
            result.update(scope='Pages tree playback against the pinned hosted media, before the Pages deploy',
                          pages_tree=str(published), media_root=media_root,
                          hosted_requests=len(hosted),
                          hosted_hosts=sorted({urlparse(url).hostname for url in hosted}),
                          index_sha256=digest(published / 'index.html'))
            name = 'published-media-browser-checks.json'
        else:
            name = 'placeholder-browser-checks.json' if placeholders_only else 'candidate-browser-checks.json'
        write(output / name, result)
        return result
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--placeholders-only', action='store_true')
    parser.add_argument('--published', type=Path,
                        help='Pages tree pinned to hosted media; plays it against the host')
    parser.add_argument('--serve', action='store_true', help='Preview locally until Ctrl-C; never uploads')
    args = parser.parse_args()
    if args.serve:
        root = args.candidate.resolve()
        manifest = read(root / 'release-manifest.json')
        for record in manifest['files']:
            if digest(root / record['path']) != record['sha256']:
                raise ValueError(f"Candidate changed: {record['path']}")
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                 functools.partial(Handler, directory=str(root)))
        print(f'Local preview: http://127.0.0.1:{server.server_port}/web/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    else:
        verify(args.candidate, placeholders_only=args.placeholders_only, published=args.published)
