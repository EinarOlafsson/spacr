"""Complete-clip opt-in keeps synthetic actions visible at their recorded speed."""
import json
import subprocess

from test_player_holds_narration import served, CHROME, PAGE


def complete_fixture(served, tmp_path):
    url, handler = served
    script = (PAGE / 'lesson_catalog.js').read_text()
    catalog = json.loads(script[script.index('Object.freeze(') + len('Object.freeze('):script.rindex(');')])
    identity = url.split('lesson=')[1]
    lesson = next(row for row in catalog['lessons'] if row['id'] == identity)
    lesson['scenes'] = lesson['scenes'][:3]
    video = tmp_path / 'complete-native-test.mp4'
    audio = tmp_path / 'complete-native-audio.m4a'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'testsrc2=size=320x180:rate=30:duration=12', '-threads', '2', '-c:v', 'libx264',
        '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'sine=frequency=330:duration=3', '-threads', '2', '-c:a', 'aac', str(audio)], check=True)
    rows = [dict(scene=i+1, speech_start=i, speech_end=i+.65, scene_end=i+1,
                 duration=1, text=row['narration'], sentences=[dict(sentence=1,
                 speech_start=i, speech_end=i+.65, text=row['narration'])])
            for i, row in enumerate(lesson['scenes'])]
    timing = tmp_path / 'complete-native-audio.json'
    timing.write_text(json.dumps(dict(schema=1, language='en', voice='af_heart', total_duration=3, scenes=rows)))
    visual = tmp_path / 'complete-native-visual.json'
    visual.write_text(json.dumps(dict(kind='native_visual_timing', native_live_video=True,
        complete_native_clips=True, coverage=dict(accepted=True), total_duration=12,
        scenes=[dict(row, speech_start=i*4, speech_end=(i+1)*4, scene_end=(i+1)*4,
                     duration=4, native_live_video=True) for i,row in enumerate(rows)])))
    lesson['visual_timings'] = f'{identity}/video/native-live-timings.json'
    catalog_path = tmp_path / 'complete-native-catalog.js'
    catalog_path.write_text('window.SPACR_LESSON_CATALOG = Object.freeze('+json.dumps(catalog)+');')
    handler.video = video
    spanish = json.loads(json.dumps(catalog))
    for row in next(row for row in spanish['lessons'] if row['id'] == identity)['scenes']:
        row['narration'] = 'Texto de prueba para el intervalo hablado.'
    spanish_path = tmp_path / 'complete-native-es.json'
    spanish_path.write_text(json.dumps(spanish))
    spanish_timing = tmp_path / 'complete-native-es-timings.json'
    spanish_rows = [dict(row, text='Texto de prueba para el intervalo hablado.') for row in rows]
    spanish_timing.write_text(json.dumps(dict(schema=1, language='es', voice='ef_dora',
        total_duration=3, scenes=spanish_rows)))
    handler.routes.update({'/web/catalog/lessons_es.json': (spanish_path, 'application/json'),
        f'/media_host/{identity}/audio/es/ef_dora.m4a': (audio, 'audio/mp4'),
        f'/media_host/{identity}/audio/es/ef_dora.json': (spanish_timing, 'application/json'),'/web/lesson_catalog.js': (catalog_path, 'application/javascript'),
        f'/media_host/{identity}/audio/en/af_heart.m4a': (audio, 'audio/mp4'),
        f'/media_host/{identity}/audio/en/af_heart.json': (timing, 'application/json'),
        f'/media_host/{identity}/video/native-live-timings.json': (visual, 'application/json')})
    return url


def open_player(engine, url):
    browser = engine.chromium.launch(executable_path=str(CHROME), headless=True,
        args=['--disable-gpu', '--disable-accelerated-video-decode', '--autoplay-policy=no-user-gesture-required'])
    page = browser.new_page()
    page.add_init_script("localStorage.setItem('spacr-tutorial-voice-v2','af_heart');"
                         "localStorage.setItem('spacr-tutorial-quality-v1','1440p');")
    page.goto(url, wait_until='domcontentloaded')
    page.wait_for_function('completeNativePlayback() && narrationAudioAvailable && chapterData.length===3', timeout=90000)
    return browser, page


def test_complete_native_tails_run_without_automatic_seeks_and_final_audio_cannot_skip(served, tmp_path):
    from playwright.sync_api import sync_playwright
    url = complete_fixture(served, tmp_path)
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        page.evaluate('''() => {
          window.__trace=[]; window.__automaticSeeks=0;
          elements.video.addEventListener('seeking',()=>window.__automaticSeeks++);
          window.__sample=setInterval(()=>window.__trace.push({video:elements.video.currentTime,
            audio:elements.audio.currentTime, paused:elements.audio.paused,
            rate:elements.video.playbackRate, audioRate:elements.audio.playbackRate}),40);
          elements.video.muted=true; elements.video.play();
        }''')
        page.wait_for_function('elements.video.currentTime>9.3 && elements.audio.paused', timeout=60000)
        page.evaluate('elements.audio.dispatchEvent(new Event("ended"))')
        page.wait_for_function('elements.video.ended', timeout=30000)
        trace = page.evaluate('window.__trace')
        seeks = page.evaluate('window.__automaticSeeks')
        assert page.evaluate('elements.video.currentTime') > 11.9
        assert page.evaluate('document.querySelector("#watch-time")?.textContent || elements.watchTime.textContent').endswith('/ 0:12')
        browser.close()
    assert seeks == 0
    assert all(row['rate'] == row['audioRate'] == 1 for row in trace)
    # Each scene's long tail, including the final action, remains on screen.
    for index in range(3):
        tail = [row for row in trace if index*4+1.3 < row['video'] < (index+1)*4-.2]
        assert len(tail) > 20, (index, tail)
        assert all(row['paused'] and index+.60 <= row['audio'] < index+1 for row in tail)
    assert any(4.1 < row['video'] < 4.6 and not row['paused'] for row in trace)
    assert any(8.1 < row['video'] < 8.6 and not row['paused'] for row in trace)


def test_complete_native_seek_pause_resume_chapter_and_track_reload(served, tmp_path):
    from playwright.sync_api import sync_playwright
    url = complete_fixture(served, tmp_path)
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        page.evaluate('elements.video.muted=true;elements.video.play()')
        page.wait_for_function('elements.video.currentTime>1.2 && elements.audio.paused', timeout=60000)
        # Clicking the current chapter while waiting in its tail must replay
        # its speech, rather than carrying the old waiting state backwards.
        page.evaluate('seekTo(chapterData[0].start)')
        page.wait_for_function('!elements.video.seeking && elements.video.currentTime<.6 && !elements.audio.paused', timeout=60000)
        page.wait_for_function('elements.video.currentTime>1.2 && elements.audio.paused', timeout=60000)
        page.evaluate('elements.video.pause()')
        paused = page.evaluate('elements.video.currentTime')
        page.wait_for_timeout(300)
        assert abs(page.evaluate('elements.video.currentTime')-paused) < .05
        assert page.evaluate('elements.audio.paused')
        page.evaluate('elements.video.play()')
        page.wait_for_function('elements.video.currentTime>2 && elements.audio.paused')
        # A user seek back into speech restarts only the requested narration.
        page.evaluate('elements.video.currentTime=.1')
        page.wait_for_function('!elements.video.seeking && !elements.audio.paused && elements.audio.currentTime<.6', timeout=60000)
        page.evaluate('seekTo(chapterData[1].start)')
        page.wait_for_function('!elements.video.seeking && elements.video.currentTime>=4 && elements.video.currentTime<5 && !elements.audio.paused', timeout=60000)
        # Track/voice reload keeps the current native visual position and does
        # not restart speech while the selected scene is already in its tail.
        page.evaluate('elements.video.currentTime=6')
        page.wait_for_function('!elements.video.seeking && elements.audio.paused', timeout=60000)
        page.evaluate('loadNarration(true)')
        page.wait_for_function('completeNativePlayback() && narrationAudioAvailable && elements.video.currentTime>=6 && !elements.video.paused && elements.audio.paused', timeout=60000)
        assert page.evaluate('chapterData.length') == 3
        assert page.evaluate('chapterDisplayTime(chapterData[1])') == 4
        assert page.locator('.chapter-time').nth(1).text_content() == '0:04'
        assert page.locator('.transcript-entry button').nth(1).text_content() == '0:04'
        # Captions map the spoken interval into the corresponding full scene.
        assert page.evaluate('videoTimeFromAudio(1.5)') == 4.5
        # Silent mode and switching back discard complete-mode wait state.
        page.evaluate('elements.voice.value="silent";loadNarration(true)')
        page.wait_for_function('completeNativePlayback() && elements.voice.value==="silent" && !narrationAudioAvailable && !elements.video.paused', timeout=60000)
        page.evaluate('elements.voice.value="af_heart";loadNarration(true)')
        page.wait_for_function('completeNativePlayback() && narrationAudioAvailable && !elements.video.paused', timeout=60000)
        assert page.evaluate('elements.video.playbackRate') == 1
        page.evaluate('elements.language.value="es";switchLanguage()')
        page.wait_for_function('completeNativePlayback() && audioTimings?.language==="es" && narrationAudioAvailable && !elements.video.paused', timeout=60000)
        assert page.evaluate('elements.voice.value') == 'ef_dora'
        assert page.evaluate('chapterData[0].text') == 'Texto de prueba para el intervalo hablado.'
        assert page.evaluate('elements.video.playbackRate') == 1
        browser.close()
