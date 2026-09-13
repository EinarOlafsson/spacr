#!/usr/bin/env python3
"""Check a real staged English lesson in Chromium without uploading its media.

The live player code is served unchanged. Only catalog/media locations are
redirected to the staging directory; untouched lessons use their existing files.
This verifies playback/synchronization, not human pronunciation or translation.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import re
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright
from stage_lesson import DEFAULT_STAGE, REPO, read, write

WORKSPACE = DEFAULT_STAGE.parent


def check_related_links(actual, expected):
    """Require exactly the authored links, including a genuinely linkless lesson."""
    if sorted(set(actual)) != sorted(set(expected)):
        raise ValueError(f'Related lesson links differ: {actual!r} != {expected!r}')


class Handler(http.server.SimpleHTTPRequestHandler):
    web_lesson = None

    def log_message(self, *args):
        pass

    def translate_path(self, path):
        candidate = Path(super().translate_path(path))
        player = WORKSPACE / 'web'
        if candidate.is_relative_to(player):
            return str(REPO / 'docs/source/_extra/tutorials' / candidate.relative_to(player))
        staged_media = DEFAULT_STAGE / 'production'
        if self.web_lesson:
            relative = Path(self.web_lesson) / 'video' / f'{self.web_lesson}_silent.mp4'
            if candidate == staged_media / relative:
                return str(DEFAULT_STAGE / 'web-renditions' / relative)
        if candidate.is_relative_to(staged_media) and not candidate.is_file():
            return str(WORKSPACE / 'production' / candidate.relative_to(staged_media))
        return str(candidate)

    def send_head(self):
        # Match the production host's byte-range contract. Without it Chrome
        # can clamp a chapter seek back to zero even with metadata loaded.
        self._range_remaining = None
        requested = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not requested or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
        if not match or not any(match.groups()):
            self.send_error(416)
            return None
        left, right = match.groups()
        start = int(left) if left else max(0, size - int(right))
        end = min(size - 1, int(right)) if left and right else size - 1
        if start > end or start >= size:
            self.send_error(416)
            return None
        stream = path.open('rb')
        stream.seek(start)
        self._range_remaining = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(self._range_remaining))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self._range_remaining is None:
                return super().copyfile(source, outputfile)
            while self._range_remaining:
                chunk = source.read(min(65536, self._range_remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                self._range_remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Native video seeking cancels its preceding range request.
            pass


def main():
    global DEFAULT_STAGE, WORKSPACE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE,
                        help='Isolated tutorial stage; does not modify the shared stage')
    parser.add_argument('--lesson', default='05_home')
    parser.add_argument('--language', default='en')
    parser.add_argument('--voice', default='af_heart')
    parser.add_argument('--caption-language', help='Independently test a staged caption language with this voice')
    parser.add_argument('--retained-media', action='store_true',
                        help='Require unchanged original catalogs and media, with no staged override')
    parser.add_argument('--web-rendition', action='store_true',
                        help='Check the verified private 1440p copy, preserving original browser reports')
    args = parser.parse_args()
    DEFAULT_STAGE = args.stage.resolve()
    WORKSPACE = DEFAULT_STAGE.parent
    retained = None
    if args.retained_media:
        from verify_retained_media import retained_media_sources
        from check_completed_matrix import voice_matrix
        retained = retained_media_sources(
            DEFAULT_STAGE, WORKSPACE, REPO / 'docs/source/_extra/tutorials/catalog',
            args.lesson, voice_matrix(WORKSPACE / 'tools/render_all_voices.py'))
    english = read(DEFAULT_STAGE / 'catalog/lessons_en.json')
    lesson = next(item for item in english['lessons'] if item['id'] == args.lesson)
    if args.caption_language:
        from catalog_preflight import validate_caption_structure
        filename = f'captions_{args.caption_language}.json'
        caption_path = DEFAULT_STAGE / 'catalog' / filename
        if not caption_path.exists():
            caption_path = DEFAULT_STAGE / 'catalog' / f'lessons_{args.caption_language}.json'
        validate_caption_structure(english, read(caption_path), args.caption_language)
    from build_navigation import build
    navigation = build(english)
    for item in english['lessons']:
        item['poster'] = f"{item['id']}/poster.jpg"
        item['silent'] = f"{item['id']}/video/{item['id']}_silent.mp4"
    source = (REPO / 'docs/source/_extra/tutorials/index.html').read_text(encoding='utf-8')
    production = '/' + str(DEFAULT_STAGE.relative_to(WORKSPACE)) + '/production'
    for attribute in ('production-root', 'audio-root', 'video4k-root'):
        source = re.sub(rf'data-{attribute}="[^"]*"', f'data-{attribute}="{production}"', source)
    rendition = None
    if args.web_rendition:
        from check_completed_matrix import digest
        folder = DEFAULT_STAGE / 'web-renditions' / args.lesson
        rendition = read(folder / 'rendition-checks.json')
        if (rendition.get('accepted') is not True or rendition.get('lesson') != args.lesson
                or digest(folder / 'video' / f'{args.lesson}_silent.mp4') != rendition.get('rendition_sha256')):
            raise ValueError('Web rendition is missing or changed since verification')
        source = re.sub(r'data-video4k-root="[^"]*"', 'data-video4k-root=""', source)
    tag = f'{args.language}-{args.voice}'
    if args.caption_language:
        tag += f'-captions-{args.caption_language}'
    output = DEFAULT_STAGE / ('browser-web' if args.web_rendition else 'browser') / args.lesson / tag
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    handler = type('WebRenditionHandler', (Handler,), {'web_lesson': args.lesson}) if args.web_rendition else Handler
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                functools.partial(handler, directory=str(WORKSPACE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    evidence = {'lesson': args.lesson, 'scope': f'{args.language}/{args.voice} playback and scene links only',
                'uploaded': False, 'translation_or_listening_review': False}
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True)
            context = browser.new_context(viewport={'width': 1440, 'height': 1100})
            base = f'http://127.0.0.1:{server.server_port}'
            context.route(base + '/web/', lambda route: route.fulfill(body=source, content_type='text/html'))
            catalog_js = 'window.SPACR_LESSON_CATALOG = ' + json.dumps(english, ensure_ascii=False) + ';'
            context.route('**/web/lesson_catalog.js*', lambda route: route.fulfill(
                body=catalog_js, content_type='application/javascript'))
            navigation_js = 'window.SPACR_TUTORIAL_NAVIGATION = ' + json.dumps(navigation, ensure_ascii=False) + ';'
            context.route('**/web/module_navigation.js*', lambda route: route.fulfill(
                body=navigation_js, content_type='application/javascript'))
            context.route('**/web/catalog/lessons_en.json*', lambda route: route.fulfill(
                json=english))
            if args.language != 'en':
                localized = read(DEFAULT_STAGE / 'catalog' / f'lessons_{args.language}.json')
                context.route(f'**/web/catalog/lessons_{args.language}.json*',
                              lambda route: route.fulfill(json=localized))
            caption_lesson = None
            if args.caption_language:
                prefix = 'captions' if args.caption_language in {'da', 'de', 'is', 'ko', 'nb', 'sv'} else 'lessons'
                filename = f'{prefix}_{args.caption_language}.json'
                captions = read(DEFAULT_STAGE / 'catalog' / filename)
                caption_lesson = next(item for item in captions['lessons'] if item['id'] == args.lesson)
                context.route(f'**/web/catalog/{filename}*', lambda route: route.fulfill(json=captions))
            page = context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(base + '/web/#lesson=' + args.lesson, wait_until='domcontentloaded')
            player_response = context.request.get(base + '/web/app_v2.js')
            assert player_response.ok
            player_bytes = (REPO / 'docs/source/_extra/tutorials/app_v2.js').read_bytes()
            assert player_response.body() == player_bytes
            evidence['repository_player_sha256'] = hashlib.sha256(player_bytes).hexdigest()
            page.wait_for_function('document.querySelectorAll(".chapter-button").length === ' + str(len(lesson['scenes'])), timeout=60000)
            evidence['navigation_contains_staged_lesson'] = args.lesson in navigation['preserved_lesson_ids']
            assert evidence['navigation_contains_staged_lesson']
            assert page.locator(f'#curriculum [data-lesson="{args.lesson}"]').count() == 1
            host_key = navigation['routes'].get(args.lesson, {}).get('host_app_key') or lesson.get('host_app_key')
            if host_key:
                assert page.locator('#lesson-content').get_attribute('data-app-key') == host_key
                assert page.locator(f'#curriculum [data-host="{host_key}"] [data-lesson="{args.lesson}"]').count() == 1
                host = next(item for item in english['lessons'] if item.get('app_key') == host_key)
                assert host['title'] in page.locator('#lesson-route').inner_text()
                evidence['host_app_key'] = host_key
            page.wait_for_function('elements.video.readyState >= 2 && elements.audio.readyState >= 2', timeout=60000)
            page.select_option('#language-select', args.language)
            page.wait_for_function('(voice) => [...elements.voice.options].some(o => o.value === voice)',
                                   arg=args.voice, timeout=30000)
            page.select_option('#voice-select', args.voice)
            example_files = list(dict.fromkeys(lesson.get('example_files', [])))
            example_links = page.locator('#prerequisite-copy a[download]')
            assert example_links.all_text_contents() == example_files
            evidence['example_files'] = []
            for name in example_files:
                assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:csv|zip)', name) and len(name) <= 128
                link = page.get_by_role('link', name=name, exact=True)
                assert link.get_attribute('download') == name
                response = context.request.get(base + '/web/' + link.get_attribute('href'))
                assert response.ok
                digest = hashlib.sha256(response.body()).hexdigest()
                expected = REPO / 'docs/source/_extra/tutorials/examples' / name
                assert digest == hashlib.sha256(expected.read_bytes()).hexdigest()
                evidence['example_files'].append({'name': name, 'sha256': digest})
            try:
                page.wait_for_function('(voice) => audioTimings?.voice === voice && narrationAudioAvailable && elements.audio.readyState >= 2',
                                       arg=args.voice, timeout=30000)
            except Exception:
                diagnostic = page.evaluate('({audioSrc: elements.audio.currentSrc, audioReady: elements.audio.readyState, error: elements.audio.error?.message, language: elements.language.value, voice: elements.voice.value, lesson: activeLesson?.id, status: elements.status.textContent, toast: elements.toast.textContent})')
                diagnostic['javascript_errors'] = errors
                write(output / 'language-load-failure.json', diagnostic)
                print(json.dumps(diagnostic, indent=2), flush=True)
                raise
            # The player intentionally prefetches narration into a blob URL.
            # Match its actual bytes, not an assumed URL shape or just the
            # selected label, which could still show a previous voice.
            audio_hash = page.evaluate('''async () => {
                const bytes = await (await fetch(elements.audio.currentSrc)).arrayBuffer();
                const digest = await crypto.subtle.digest('SHA-256', bytes);
                return [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, '0')).join('');
            }''')
            expected_hash = hashlib.sha256(((WORKSPACE if retained else DEFAULT_STAGE) / 'production' / args.lesson /
                            'audio' / args.language / f'{args.voice}.m4a').read_bytes()).hexdigest()
            assert audio_hash == expected_hash, (audio_hash, expected_hash)
            evidence['loaded_audio_sha256'] = audio_hash
            if caption_lesson is not None:
                page.locator('#caption-settings-button').click()
                page.select_option('#caption-language-select', args.caption_language)
                page.wait_for_function('(language) => effectiveCaptionLanguage() === language && elements.chapters.lang === language',
                                       arg=args.caption_language, timeout=30000)
                texts = page.evaluate('chapterData.map(chapter => chapter.text)')
                assert texts == [scene['narration'] for scene in caption_lesson['scenes']], texts
                assert page.evaluate('elements.language.value') == args.language
                assert page.evaluate('audioTimings.voice') == args.voice
                evidence['caption_language'] = args.caption_language
                evidence['caption_scenes_match_staging'] = True
                # Check the actual generated WebVTT, not just the selector.
                vtt = page.evaluate('async () => await (await fetch(elements.captionTrack.src)).text()')
                assert vtt.startswith('WEBVTT') and '-->' in vtt, vtt[:100]
                evidence['caption_webvtt_sha256'] = hashlib.sha256(vtt.encode()).hexdigest()
                page.locator('#caption-settings-close').click()
            evidence['scene_count'] = page.locator('.chapter-button').count()
            expected = sorted({identity for scene in lesson['scenes'] for identity in scene.get('related_lessons', [])})
            actual = page.locator('#chapter-list [data-related-lesson]').evaluate_all(
                '(nodes) => [...new Set(nodes.map(n => n.dataset.relatedLesson))].sort()')
            check_related_links(actual, expected)
            page.evaluate('elements.audio.muted = true; elements.video.muted = true')
            page.locator('.chapter-button').nth(5).click()
            try:
                page.wait_for_function('!elements.audio.paused && elements.audio.currentTime > chapterData[5].start + 1', timeout=15000)
            except Exception:
                diagnostic = page.evaluate('({audio: {time: elements.audio.currentTime, paused: elements.audio.paused, ready: elements.audio.readyState, src: elements.audio.currentSrc, error: elements.audio.error?.message}, video: {time: elements.video.currentTime, paused: elements.video.paused, ready: elements.video.readyState, src: elements.video.currentSrc, error: elements.video.error?.message}, voice: elements.voice.value, available: narrationAudioAvailable, status: elements.status.textContent, chapter: chapterData[5]})')
                write(output / 'playback-failure.json', diagnostic)
                print(json.dumps(diagnostic, indent=2), flush=True)
                page.screenshot(path=str(output / 'playback-failure.png'))
                raise
            clock = page.evaluate('({audio: elements.audio.currentTime, video: elements.video.currentTime, expectedVideo: videoTimeFromAudio(elements.audio.currentTime), audioDuration: elements.audio.duration, videoDuration: elements.video.duration, mediaError: elements.video.error?.message || elements.audio.error?.message || null})')
            assert not clock['mediaError'], clock
            assert abs(clock['video'] - clock['expectedVideo']) < 0.5, clock
            evidence['seek_playback_clocks'] = clock
            if rendition:
                video = page.evaluate('''async () => {
                    const bytes = await (await fetch(elements.video.currentSrc)).arrayBuffer();
                    const hash = await crypto.subtle.digest('SHA-256', bytes);
                    return {sha256: [...new Uint8Array(hash)].map(x => x.toString(16).padStart(2, '0')).join(''),
                            width: elements.video.videoWidth, height: elements.video.videoHeight};
                }''')
                assert video['sha256'] == rendition['rendition_sha256'], video
                assert [video['width'], video['height']] == [2560, 1440], video
                evidence['checked_web_rendition'] = video
            page.evaluate('elements.audio.pause(); elements.video.pause()')
            page.screenshot(path=str(output / 'desktop.png'), full_page=True)
            page.locator('#transcript-tab').click()
            transcript_links = page.locator('#transcript-list [data-related-lesson]').evaluate_all(
                '(nodes) => nodes.map(n => n.dataset.relatedLesson)')
            check_related_links(transcript_links, expected)
            evidence['chapter_and_transcript_links'] = expected
            page.set_viewport_size({'width': 390, 'height': 844})
            mobile_geometry = page.evaluate('''() => ({
                viewport: window.innerWidth, width: document.documentElement.scrollWidth,
                overflowing: [...document.querySelectorAll('body *')].filter(node => {
                    const r = node.getBoundingClientRect();
                    return r.width > 0 && (r.right > window.innerWidth + 1 || r.left < -1);
                }).slice(0, 30).map(node => ({tag: node.tagName, id: node.id,
                    className: String(node.className), text: node.textContent.slice(0, 160),
                    width: node.getBoundingClientRect().width}))
            })''')
            if mobile_geometry['width'] > mobile_geometry['viewport']:
                write(output / 'mobile-overflow.json', mobile_geometry)
                page.screenshot(path=str(output / 'mobile-overflow.png'), full_page=True)
                raise AssertionError(f'Mobile overflow: {mobile_geometry}')
            page.screenshot(path=str(output / 'mobile.png'), full_page=True)
            assert not errors, errors
            evidence['passed'] = True
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    if retained:
        after = retained_media_sources(
            DEFAULT_STAGE, WORKSPACE, REPO / 'docs/source/_extra/tutorials/catalog',
            args.lesson, voice_matrix(WORKSPACE / 'tools/render_all_voices.py'))
        if after != retained:
            raise ValueError('Retained source media changed during playback verification')
        evidence['whole_media_retained'] = True
        evidence['original_sources_unchanged'] = True
    write(output / 'playback-checks.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
