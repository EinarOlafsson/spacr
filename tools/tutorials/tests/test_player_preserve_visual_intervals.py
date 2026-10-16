"""Retained visual intervals stay intact without asserting full raw-clip coverage."""
import json
import pytest

from test_player_complete_native_clips import complete_fixture, open_player
from test_player_holds_narration import served
from verify_staged_lesson import check_native_boundary_playback


def test_preserved_intervals_hold_audio_and_finish_selected_video_at_one_speed(served, tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright
    import test_player_holds_narration
    monkeypatch.setattr(test_player_holds_narration, 'VIDEO_DELAY_SECONDS', 0)
    url = complete_fixture(served, tmp_path)
    visual_path = tmp_path / 'complete-native-visual.json'
    visual = json.loads(visual_path.read_text())
    del visual['complete_native_clips']
    visual['preserve_visual_intervals'] = True
    visual['scenes'][0]['native_live_video'] = False
    visual['scenes'][-1]['native_live_video'] = False
    visual_path.write_text(json.dumps(visual))
    spoken = json.loads((tmp_path / 'complete-native-audio.json').read_text())
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        try:
            page.wait_for_function('elements.video.buffered.length && elements.video.buffered.end(elements.video.buffered.length-1)>=11.99', timeout=30000)
            assert page.evaluate('visualTimings.complete_native_clips === undefined')
            result = check_native_boundary_playback(page, visual, spoken)
            assert result['preserve_visual_intervals'] is True
            assert result['complete_native_clips'] is False
            assert result['actual_video_speed'] == 1
            assert result['boundary']['recorded_tail_observed']
            assert result['boundary']['next_narration_resumed']
            assert result['boundary']['checked_boundary_after_scene'] == 2
            assert result['final_tail']['final_visual_eof_observed']
            assert result['final_tail']['final_native_eof_observed'] is False
            assert page.evaluate('elements.video.currentTime') > 11.9
            assert page.evaluate('elements.video.playbackRate === 1 && elements.audio.playbackRate === 1')
            page.evaluate('visualTimings.native_live_video=false')
            assert page.evaluate('completeNativePlayback()') is False
        finally:
            browser.close()


def test_absent_preservation_flag_keeps_legacy_narration_owned_boundary_skip(served, tmp_path):
    from test_player_native_live_video import test_native_video_runs_at_one_speed_and_seeks_only_scene_boundaries
    test_native_video_runs_at_one_speed_and_seeks_only_scene_boundaries(served, tmp_path)


def test_preserved_static_chapter_seek_and_caption_cues_use_elapsed_speech(served, tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright
    import test_player_holds_narration
    monkeypatch.setattr(test_player_holds_narration, 'VIDEO_DELAY_SECONDS', 0)
    url = complete_fixture(served, tmp_path)
    visual_path = tmp_path / 'complete-native-visual.json'
    visual = json.loads(visual_path.read_text())
    del visual['complete_native_clips']
    visual['preserve_visual_intervals'] = True
    visual['scenes'][0]['native_live_video'] = False
    visual['scenes'][-1]['native_live_video'] = False
    visual_path.write_text(json.dumps(visual))
    catalog_path = tmp_path / 'complete-native-catalog.js'
    script = catalog_path.read_text()
    catalog = json.loads(script[script.index('Object.freeze(')+len('Object.freeze('):script.rindex(');')])
    lesson = next(row for row in catalog['lessons'] if row['id'] == url.split('lesson=')[1])
    timing_path = tmp_path / 'complete-native-audio.json'
    timing = json.loads(timing_path.read_text())
    for index, text in enumerate(('Static intro.', 'Native interval.', 'Static conclusion.')):
        lesson['scenes'][index]['narration'] = text
        timing['scenes'][index]['text'] = text
        timing['scenes'][index]['sentences'][0]['text'] = text
    catalog_path.write_text('window.SPACR_LESSON_CATALOG = Object.freeze('+json.dumps(catalog)+');')
    timing_path.write_text(json.dumps(timing))
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        try:
            page.wait_for_function('elements.video.buffered.length && elements.video.buffered.end(elements.video.buffered.length-1)>=11.99', timeout=30000)
            assert page.evaluate('videoTimeFromAudio(.25)') == .25
            assert page.evaluate('audioTimeFromVideo(.25)') == .25
            assert page.evaluate('videoTimeFromAudio(2.25)') == 8.25
            assert page.evaluate('audioTimeFromVideo(8.25)') == 2.25
            assert abs(page.evaluate('audioTimeFromVideo(10)') - 2.999) < .00001
            page.wait_for_function('elements.captionTrack.readyState===2 && !captionTrackLoading', timeout=30000)
            vtt = page.evaluate('fetch(captionUrl).then(response=>response.text())')
            assert '00:00:08.000 --> 00:00:08.650\nStatic conclusion.' in vtt
            page.evaluate('elements.video.muted=true')
            page.locator('.chapter-button').nth(2).click()
            page.wait_for_function('!elements.video.seeking && !elements.audio.seeking && !elements.audio.paused && elements.video.currentTime>=8', timeout=15000)
            page.evaluate('elements.video.pause()')
            chapter = page.evaluate('({video:elements.video.currentTime,audio:elements.audio.currentTime})')
            assert abs(chapter['video'] - (8 + chapter['audio'] - 2)) < .15, chapter
            page.evaluate('elements.video.muted=true;seekTo(2.25);elements.video.pause()')
            page.wait_for_function('!elements.video.seeking && !elements.audio.seeking && elements.video.readyState>=2', timeout=15000)
            page.wait_for_timeout(150)
            state = page.evaluate('({video:elements.video.currentTime,audio:elements.audio.currentTime,cues:[...(elements.captionTrack.track.activeCues||[])].map(c=>c.text)})')
            assert abs(state['video'] - (8 + state['audio'] - 2)) < .15, state
            assert 'Static conclusion.' in state['cues']
            page.evaluate('delete visualTimings.preserve_visual_intervals')
            assert page.evaluate('completeNativePlayback()') is False
            assert page.evaluate('videoTimeFromAudio(2.25)') == 9
            assert page.evaluate('audioTimeFromVideo(9)') == 2.25
        finally:
            browser.close()


@pytest.mark.parametrize('visual_mode', ['preserved', 'complete_native'])
def test_cold_paused_sentence_seeks_never_rewind_the_requested_audio_offset(served, tmp_path, visual_mode):
    from playwright.sync_api import sync_playwright
    url = complete_fixture(served, tmp_path)
    visual_path = tmp_path / 'complete-native-visual.json'
    visual = json.loads(visual_path.read_text())
    if visual_mode == 'preserved':
        del visual['complete_native_clips']
        visual['preserve_visual_intervals'] = True
        visual['scenes'][0]['native_live_video'] = False
        visual['scenes'][-1]['native_live_video'] = False
    visual_path.write_text(json.dumps(visual))
    catalog_path = tmp_path / 'complete-native-catalog.js'
    script = catalog_path.read_text()
    catalog = json.loads(script[script.index('Object.freeze(')+len('Object.freeze('):script.rindex(');')])
    lesson = next(row for row in catalog['lessons'] if row['id'] == url.split('lesson=')[1])
    timing_path = tmp_path / 'complete-native-audio.json'
    timing = json.loads(timing_path.read_text())
    for index, text in enumerate(('Opening guidance.', 'Recorded controls.', 'Closing guidance.')):
        lesson['scenes'][index]['narration'] = text
        timing['scenes'][index]['text'] = text
        timing['scenes'][index]['sentences'][0]['text'] = text
    catalog_path.write_text('window.SPACR_LESSON_CATALOG = Object.freeze('+json.dumps(catalog)+');')
    timing_path.write_text(json.dumps(timing))
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        try:
            page.wait_for_function('!videoClockCorrectionPending && !elements.video.seeking && !elements.audio.seeking && elements.video.readyState>=2', timeout=30000)
            page.evaluate('''() => {
                const descriptor = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "currentTime");
                window.__seekWrites=[];window.__pendingSeekEvents=[];
                Object.defineProperty(elements.audio,"currentTime",{
                    configurable:true,
                    get(){return Reflect.apply(descriptor.get,this,[]);},
                    set(value){window.__seekWrites.push({value,requested:window.__requestedAudio,
                        pending:videoClockCorrectionPending,videoSeeking:elements.video.seeking,
                        stack:new Error().stack});return Reflect.apply(descriptor.set,this,[value]);}
                });
                for(const media of [elements.audio,elements.video])
                    media.addEventListener("seeking",()=>window.__pendingSeekEvents.push({
                        pending:videoClockCorrectionPending,videoSeeking:elements.video.seeking,
                        audioPaused:elements.audio.paused}));
            }''')
            for index in (0, 2, 0, 1, 2):
                sentence = timing['scenes'][index]['sentences'][0]
                seconds = (sentence['speech_start'] + sentence['speech_end']) / 2
                page.evaluate('''seconds => {window.__seekWrites=[];window.__requestedAudio=seconds;
                    seekTo(seconds);elements.video.pause();}''', seconds)
                page.wait_for_function('!videoClockCorrectionPending && !elements.video.seeking && !elements.audio.seeking && elements.video.readyState>=2', timeout=15000)
                page.wait_for_timeout(150)
                state = page.evaluate('''() => ({writes:window.__seekWrites,
                    audio:elements.audio.currentTime,video:elements.video.currentTime,
                    expected:videoTimeFromAudio(window.__requestedAudio),
                    cues:[...(elements.captionTrack.track.activeCues||[])].map(c=>c.text),
                    text:audioTimings.scenes[Math.floor(window.__requestedAudio)].text})''')
                assert state['writes'], state
                assert all(abs(row['value']-seconds)<.015 for row in state['writes']), state
                assert abs(state['audio']-seconds)<.15, state
                assert abs(state['video']-state['expected'])<.15, state
                assert sentence['text'] in state['cues'], state
            events = page.evaluate('window.__pendingSeekEvents')
            pending = [row for row in events if row['pending'] or row['videoSeeking']]
            assert pending
            assert all(row['audioPaused'] for row in pending), pending
        finally:
            page.evaluate('delete elements.audio.currentTime')
            browser.close()
