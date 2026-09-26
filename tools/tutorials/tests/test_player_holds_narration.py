"""A slow or stalled picture holds the narration instead of racing ahead of it.

Real browser, real player source (tools/tutorials/authoring/web/app_v2.js),
the committed release checkpoint's page and catalogs, and synthetic media
served by a range-capable local server that answers video requests slowly,
like a congested media host. Narration must not advance while the picture is
still seeking or buffering, the clock correction must not chase a moving
target, and the viewer's play state must be kept.
"""
from __future__ import annotations

import functools
import http.server
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import pytest

TUTORIALS = Path(__file__).resolve().parents[1]
PAGE = TUTORIALS / 'release_candidate/web'
PLAYER = TUTORIALS / 'authoring/web/app_v2.js'
CHROME = Path('/opt/google/chrome/chrome')
DURATION = 60.0
VIDEO_DELAY_SECONDS = 2.5


def make_media(root, lesson):
    """Synthetic 60 s silent video and narration with one sentence per scene."""
    scenes = lesson['scenes']
    video = root / 'video.mp4'
    audio = root / 'audio.m4a'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'testsrc2=size=320x180:rate=30:duration={DURATION}', '-threads', '2',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-g', '60', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(video)], check=True, timeout=120)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'sine=frequency=330:duration={DURATION}', '-threads', '2',
                    '-c:a', 'aac', '-b:a', '64k', str(audio)], check=True, timeout=120)
    step = DURATION / len(scenes)
    timing_scenes = []
    for index, scene in enumerate(scenes):
        start, end = index * step, index * step + step - 0.5
        timing_scenes.append({
            'scene': index + 1, 'speech_start': start, 'speech_end': end,
            'scene_end': (index + 1) * step, 'duration': end - start, 'text': scene['narration'],
            'sentences': [{'sentence': 1, 'speech_start': start, 'speech_end': end,
                           'audible_start': start, 'audible_end': end, 'duration': end - start,
                           'text': scene['narration']}]})
    timings = {'schema': 1, 'language': 'en', 'voice': 'af_heart', 'total_duration': DURATION,
               'scenes': timing_scenes}
    (root / 'timings.json').write_text(json.dumps(timings))
    return video, audio, root / 'timings.json'


class SlowMediaHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the checkpoint page, the player source and range-capable media."""
    routes = {}
    video = None
    video_requests = []

    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path.endswith('_silent.mp4'):
            self.video_requests.append((time.monotonic(), self.headers.get('Range')))
            time.sleep(VIDEO_DELAY_SECONDS)
            return self.send_range(self.video, 'video/mp4')
        if path in self.routes:
            return self.send_range(*self.routes[path])
        return super().do_GET()

    def send_range(self, source, kind):
        data = source.read_bytes()
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range') or '')
        start, end = 0, len(data) - 1
        if match:
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            else:
                start = len(data) - int(match.group(2))
            end = min(end, len(data) - 1)
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
        else:
            self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        try:
            self.wfile.write(data[start:end + 1])
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def served(tmp_path):
    if not (shutil.which('ffmpeg') and CHROME.exists() and (PAGE / 'index.html').exists()):
        pytest.skip('Requires ffmpeg, Chrome and the committed release checkpoint page')
    pytest.importorskip('playwright')
    js = (PAGE / 'lesson_catalog.js').read_text()
    catalog = json.loads(js[js.index('Object.freeze(') + len('Object.freeze('):js.rindex(');')])
    lesson = next(item for item in catalog['lessons']
                  if item.get('status') != 'coming_soon' and len(item['scenes']) >= 4)
    video, audio, timings = make_media(tmp_path, lesson)
    identity = lesson['id']
    handler = type('Handler', (SlowMediaHandler,), {
        'video': video, 'video_requests': [],
        'routes': {'/web/app_v2.js': (PLAYER, 'application/javascript'),
                   f'/media_host/{identity}/audio/en/af_heart.m4a': (audio, 'audio/mp4'),
                   f'/media_host/{identity}/audio/en/af_heart.json': (timings, 'application/json')}})
    root = tmp_path / 'site'
    root.mkdir()
    (root / 'web').symlink_to(PAGE, target_is_directory=True)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(handler, directory=str(root)))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/web/#lesson={identity}', handler
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('start', ['playing', 'paused'])
def test_a_slow_seek_holds_narration_and_keeps_playing(served, start):
    """seekTo() also starts a paused lesson; that play must not release the hold."""
    from playwright.sync_api import sync_playwright
    url, handler = served
    with sync_playwright() as engine:
        browser = engine.chromium.launch(executable_path=str(CHROME), headless=True,
                                         args=['--autoplay-policy=no-user-gesture-required'])
        page = browser.new_page()
        page.add_init_script("localStorage.setItem('spacr-tutorial-voice-v2', 'af_heart');"
                             "localStorage.setItem('spacr-tutorial-quality-v1', '1440p');")
        page.goto(url, wait_until='domcontentloaded')
        page.wait_for_function('narrationAudioAvailable && elements.audio.readyState >= 1 && chapterData.length > 0',
                               timeout=90000)
        page.evaluate('''() => {
            window.__seeks = 0;
            window.__trace = [];
            elements.video.addEventListener('seeking', () => window.__seeks++);
            setInterval(() => window.__trace.push({t: performance.now(),
                audio: elements.audio.currentTime, audioPaused: elements.audio.paused,
                video: elements.video.currentTime, loading: elements.video.seeking ||
                    elements.video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA,
                videoPaused: elements.video.paused}), 50);
        }''')
        page.evaluate('elements.video.muted = true; elements.video.play()')
        page.wait_for_function('!elements.video.paused && !elements.video.seeking && '
                               'elements.video.readyState >= 3 && !elements.audio.paused', timeout=60000)
        if start == 'paused':
            page.evaluate('elements.video.pause()')
            page.wait_for_function('elements.video.paused && elements.audio.paused')
        target = page.evaluate('chapterData[2].start')
        page.evaluate('() => { window.__seeks = 0; window.__trace = []; }')
        page.evaluate('(seconds) => seekTo(seconds)', target)
        # Let the slow seek run its course, then require a settled, synchronized clock.
        page.wait_for_function('''!videoClockCorrectionPending && !elements.video.seeking &&
            elements.video.readyState >= 3 && !elements.audio.paused &&
            Math.abs(elements.video.currentTime - videoTimeFromAudio(elements.audio.currentTime)) < .5''',
                               timeout=60000, polling=50)
        page.wait_for_timeout(1000)
        trace = page.evaluate('window.__trace')
        seeks = page.evaluate('window.__seeks')
        final = page.evaluate('''() => ({audio: elements.audio.currentTime, video: elements.video.currentTime,
            expected: videoTimeFromAudio(elements.audio.currentTime), paused: elements.video.paused,
            audioPaused: elements.audio.paused})''')
        browser.close()
    loading = [row for row in trace if row['loading']]
    assert loading, 'The throttled media host must actually produce a loading picture'
    # While the picture was loading, narration stayed at the chapter start.
    assert max(row['audio'] for row in loading) - target < 0.3, (target, loading[-1])
    # The viewer's play state never changed after the seek started it; only
    # the narration was held.
    assert not any(row['videoPaused'] for row in trace[3:]), 'the video play state was changed'
    # Correct once, then wait: no chase of a moving narration clock.
    assert seeks <= 2, seeks
    assert not final['paused'] and not final['audioPaused'], final
    assert abs(final['video'] - final['expected']) < .5 and final['audio'] > target, final
