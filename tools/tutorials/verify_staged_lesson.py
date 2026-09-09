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
from stage_lesson import DEFAULT_STAGE, read, write

WORKSPACE = DEFAULT_STAGE.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def translate_path(self, path):
        candidate = Path(super().translate_path(path))
        staged_media = DEFAULT_STAGE / 'production'
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', default='05_home')
    parser.add_argument('--language', default='en')
    parser.add_argument('--voice', default='af_heart')
    parser.add_argument('--caption-language', help='Independently test a staged caption language with this voice')
    args = parser.parse_args()
    english = read(DEFAULT_STAGE / 'catalog/lessons_en.json')
    lesson = next(item for item in english['lessons'] if item['id'] == args.lesson)
    from build_navigation import build
    navigation = build(english)
    for item in english['lessons']:
        item['poster'] = f"{item['id']}/poster.jpg"
        item['silent'] = f"{item['id']}/video/{item['id']}_silent.mp4"
    source = (WORKSPACE / 'web/index.html').read_text(encoding='utf-8')
    production = '/' + str(DEFAULT_STAGE.relative_to(WORKSPACE)) + '/production'
    for attribute in ('production-root', 'audio-root', 'video4k-root'):
        source = re.sub(rf'data-{attribute}="[^"]*"', f'data-{attribute}="{production}"', source)
    tag = f'{args.language}-{args.voice}'
    if args.caption_language:
        tag += f'-captions-{args.caption_language}'
    output = DEFAULT_STAGE / 'browser' / args.lesson / tag
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                functools.partial(Handler, directory=str(WORKSPACE)))
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
            page.wait_for_function('document.querySelectorAll(".chapter-button").length === ' + str(len(lesson['scenes'])), timeout=60000)
            evidence['navigation_contains_staged_lesson'] = args.lesson in navigation['preserved_lesson_ids']
            assert evidence['navigation_contains_staged_lesson']
            assert page.locator(f'#curriculum [data-lesson="{args.lesson}"]').count() == 1
            if lesson.get('host_app_key'):
                assert page.locator('#lesson-content').get_attribute('data-app-key') == lesson['host_app_key']
                assert page.locator(f'#curriculum [data-host="{lesson["host_app_key"]}"] [data-lesson="{args.lesson}"]').count() == 1
                host = next(item for item in english['lessons'] if item.get('app_key') == lesson['host_app_key'])
                assert host['title'] in page.locator('#lesson-route').inner_text()
            page.wait_for_function('elements.video.readyState >= 2 && elements.audio.readyState >= 2', timeout=60000)
            page.select_option('#language-select', args.language)
            page.wait_for_function('(voice) => [...elements.voice.options].some(o => o.value === voice)',
                                   arg=args.voice, timeout=30000)
            page.select_option('#voice-select', args.voice)
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
            expected_hash = hashlib.sha256((DEFAULT_STAGE / 'production' / args.lesson /
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
            assert actual == expected, (actual, expected)
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
            page.evaluate('elements.audio.pause(); elements.video.pause()')
            page.screenshot(path=str(output / 'desktop.png'), full_page=True)
            page.locator('#transcript-tab').click()
            assert page.locator('#transcript-list [data-related-lesson]').count() > 0
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            page.screenshot(path=str(output / 'mobile.png'), full_page=True)
            assert not errors, errors
            evidence['passed'] = True
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    write(output / 'playback-checks.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
