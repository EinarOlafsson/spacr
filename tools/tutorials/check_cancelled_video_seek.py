"""Exercise actual media cancellation; never manually set synchronization flags.

Serves the repository player with an existing private candidate's real media.
An interrupted seek must not prevent synchronization in the next lesson.
"""
import argparse
import functools
import hashlib
import http.server
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

from coming_soon import first_placeholder
from stage_lesson import REPO, read, write
from verify_staged_lesson import Handler


def check(candidate, output):
    source = (REPO / 'docs/source/_extra/tutorials/app_v2.js').read_text()
    catalog = read(candidate / 'web/catalog/lessons_en.json')['lessons']
    hold = first_placeholder(catalog)
    ready = [item['id'] for item in catalog if item.get('status') != 'coming_soon']
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
        functools.partial(Handler, directory=str(candidate)))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    result = {'repository_player_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'scope': 'Actual source replacement during a real video seek',
              'flags_injected': False, 'published': False, 'cases': [], 'passed': False}
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True)
            context = browser.new_context()
            context.route('**/app_v2.js*', lambda route: route.fulfill(
                body=source, content_type='application/javascript'))
            page = context.new_page()
            page.goto(origin + '/web/#lesson=' + hold, wait_until='networkidle')
            for destination in (hold, ready[1]):
                page.evaluate('(id) => selectLesson(id)', ready[0])
                page.wait_for_function('narrationAudioAvailable && !elements.video.seeking && !videoClockCorrectionPending')
                before = page.evaluate('''(destination) => {
                    // A restored watch position may equal a fixed target.
                    // Always request a genuinely different position.
                    setVideoTimeWithoutNarrationSeek(elements.video.duration *
                        (elements.video.currentTime < elements.video.duration / 2 ? .75 : .25));
                    const before = {pending: videoClockCorrectionPending,
                                    seeking: elements.video.seeking};
                    window.transitionDone = selectLesson(destination);
                    return before;
                }''', destination)
                assert before == {'pending': True, 'seeking': True}, before
                page.evaluate('transitionDone')
                page.wait_for_timeout(200)
                after = page.evaluate('''() => ({pending: videoClockCorrectionPending,
                    seeking: elements.video.seeking, lesson: activeLesson.id,
                    ready: narrationAudioAvailable})''')
                result['cases'].append({'destination': destination, 'before': before, 'after': after})
                assert after['lesson'] == destination
                assert after['pending'] is False and after['seeking'] is False, after
                if destination != hold:
                    assert after['ready'] is True
                    page.evaluate('seekTo(8)')
                    page.wait_for_timeout(1500)
                    page.wait_for_function('''elements.audio.currentTime >= 7.75 &&
                        Math.abs(elements.video.currentTime -
                            videoTimeFromAudio(elements.audio.currentTime)) < .5''', timeout=10000)
                    clocks = page.evaluate('''() => ({audio: elements.audio.currentTime,
                        video: elements.video.currentTime,
                        expected: videoTimeFromAudio(elements.audio.currentTime)})''')
                    result['cases'][-1]['resumed_clocks'] = clocks
                    assert clocks['audio'] >= 7.75, clocks
                    assert abs(clocks['video'] - clocks['expected']) < .5, clocks
            browser.close()
        result['passed'] = True
    finally:
        write(output, result)
        server.shutdown()
        server.server_close()
    print('Real interrupted seek: placeholder and ready transitions passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.candidate.resolve(), args.output)
