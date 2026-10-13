#!/usr/bin/env python3
"""Verify the actual English audio/metadata pair and generated native cues."""
import argparse
import functools
import hashlib
import http.server
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright
from stage_lesson import REPO, write
from verify_staged_lesson import Handler


def verify(output, media_override=None):
    web = REPO / 'docs/source/_extra/tutorials'
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
        functools.partial(Handler, directory=str(web)))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True)
            context = browser.new_context(viewport={'width': 1440, 'height': 1000})
            if media_override:
                # Serve real staged bytes through the unchanged production
                # URLs; do not mock audio clocks, transcripts or native cues.
                for suffix, content_type in (('.m4a', 'audio/mp4'), ('.json', 'application/json')):
                    path = media_override / ('af_heart' + suffix)
                    context.route('https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/main/'
                        + '04_platform_installers/audio/en/af_heart' + suffix,
                        lambda route, request, p=path, kind=content_type: route.fulfill(
                            path=str(p), content_type=kind, headers={'Access-Control-Allow-Origin':'*'}))
            context.add_init_script('''
                localStorage.setItem('spacr-tutorial-language-v2', 'en');
                localStorage.setItem('spacr-tutorial-voice-v2', 'af_heart');
                localStorage.setItem('spacr-tutorial-captions-v1', JSON.stringify({language:'en'}));
            ''')
            page = context.new_page()
            page.on('pageerror', lambda error: print('Player error:', error, flush=True))
            page.on('console', lambda message: print('Browser:', message.text, flush=True)
                    if message.type == 'error' else None)
            page.goto(origin + '/#lesson=04_platform_installers')
            try:
                page.wait_for_function('narrationAudioAvailable && audioTimings', timeout=30000)
                # Exercise the same visible caption toggle as the listener.
                page.locator('#caption-settings-button').click()
                page.locator('#caption-enabled').check()
                page.locator('#caption-settings-close').click()
                page.evaluate('seekTo(1)')
                page.wait_for_function('elements.captionTrack.track.cues?.length > 0', timeout=10000)
            except Exception:
                print(page.evaluate('''async () => ({available:narrationAudioAvailable,
                    timingVoice:audioTimings?.voice, captionLoading:captionTrackLoading,
                    trackMode:elements.captionTrack.track.mode, trackState:elements.captionTrack.readyState,
                    settings:captionSettings, chapters:chapterData.length,
                    vtt:captionUrl ? await (await fetch(captionUrl)).text() : null,
                    audioError:elements.audio.error?.message, videoError:elements.video.error?.message,
                    toast:document.querySelector('#toast').textContent})'''), flush=True)
                raise
            identity = page.evaluate('''async () => {
                const bytes = await (await fetch(elements.audio.src)).arrayBuffer();
                const hash = await crypto.subtle.digest('SHA-256', bytes);
                return {voice: audioTimings.voice, expected: audioTimings.media_sha256,
                    actual: Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2,'0')).join(''),
                    chapterTexts: chapterData.map(c => c.text),
                    spokenTexts: audioTimings.scenes.map(s => s.text)};
            }''')
            assert identity['voice'] == 'af_heart'
            assert identity['expected'] == identity['actual']
            assert identity['chapterTexts'] == identity['spokenTexts']
            for _ in range(3):
                page.evaluate('renderCaptions()')
                page.wait_for_function('elements.captionTrack.readyState === 2 && !captionTrackLoading', timeout=10000)
                assert page.locator('#caption-track').count() == 1
            cases = []
            for seconds in (27, 37, 55, 76):
                page.wait_for_function('!videoClockCorrectionPending && !elements.video.seeking && !elements.audio.seeking', timeout=10000)
                page.evaluate('(s) => seekTo(s)', seconds)
                page.wait_for_timeout(300)
                page.evaluate('elements.video.pause()')
                page.wait_for_timeout(150)
                sample = page.evaluate('''() => {
                    const at = elements.audio.currentTime;
                    const spoken = audioTimings.scenes.flatMap(s => s.sentences || [])
                        .find(s => at >= s.speech_start && at < s.speech_end);
                    const cues = [...(elements.captionTrack.track.activeCues || [])].map(c => c.text);
                    return {audioTime:at, videoTime:elements.video.currentTime,
                        expectedText:spoken?.text, activeCues:cues};
                }''')
                sample['requestedAudioTime'] = seconds
                assert abs(sample['audioTime'] - seconds) < 1, sample
                assert sample['expectedText'] and sample['expectedText'] in sample['activeCues'], sample
                cases.append(sample)
            result = {'scope':'Local fixed player with ' + ('staged' if media_override else 'current hosted')
                          + ' Heart audio; not deployment or human pronunciation review',
                      'player_sha256':hashlib.sha256((web/'app_v2.js').read_bytes()).hexdigest(),
                      'audio_identity':identity, 'native_track_reload_checks':3,
                      'reported_timestamp_checks':cases, 'passed':True}
            write(output, result)
            print(result)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media-override', type=Path)
    args = parser.parse_args()
    verify(args.output, args.media_override)
