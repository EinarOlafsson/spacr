"""A pending narration must not spend the picture clock or skip its onset.

Real software Chrome and player code, with explicit synthetic media fixtures.
HTTP delays exercise normal first-play/chapter/voice control lifecycles.
"""
import json
import time

import pytest

from test_player_complete_native_clips import complete_fixture, open_player
from test_player_holds_narration import served, CHROME


def delay_resource(handler, suffix, seconds=1.2):
    original = handler.do_GET

    def delayed(self):
        if self.path.split('?', 1)[0].endswith(suffix):
            time.sleep(seconds)
        return original(self)

    handler.do_GET = delayed


def trace_audio(page):
    page.evaluate("""() => {
      window.__starts=[];window.__readinessTrace=[];
      elements.audio.addEventListener('playing',()=>window.__starts.push({
        audio:elements.audio.currentTime,video:elements.video.currentTime,
        voice:audioTimings?.voice,pending:narrationLoadPending}));
      window.__readinessTimer=setInterval(()=>window.__readinessTrace.push({
        audio:elements.audio.currentTime,video:elements.video.currentTime,
        pending:narrationLoadPending,held:videoHeldForAudio}),25);
    }""")


def test_normal_first_play_waits_for_audio_body(served, tmp_path):
    from playwright.sync_api import sync_playwright

    url = complete_fixture(served, tmp_path)
    delay_resource(served[1], '/audio/en/af_heart.m4a', seconds=8)
    with sync_playwright() as engine:
        browser = engine.chromium.launch(executable_path=str(CHROME), headless=True,
            args=['--disable-gpu', '--disable-accelerated-video-decode',
                  '--autoplay-policy=no-user-gesture-required'])
        page = browser.new_page()
        page.add_init_script("localStorage.setItem('spacr-tutorial-voice-v2','af_heart');")
        page.goto(url, wait_until='domcontentloaded')
        page.wait_for_function('elements.video.readyState>=3&&narrationLoadPending')
        assert page.evaluate('!elements.loading.classList.contains("hidden")')
        trace_audio(page)
        # The loading layer prevents a normal pointer play gesture from
        # starting the already-ready picture before its narration is fetched.
        page.locator('#video-loading').click()
        # Separately exercise the native media play-event guard while pending;
        # this controlled API request is not evidence of a physical UI click.
        page.evaluate('void elements.video.play().catch(()=>{})')
        page.wait_for_function('videoHeldForAudio&&elements.video.paused')
        position = page.evaluate('elements.video.currentTime')
        page.wait_for_timeout(300)
        assert abs(page.evaluate('elements.video.currentTime') - position) < .06
        page.wait_for_function('!narrationLoadPending', timeout=15000)
        assert page.evaluate('elements.video.currentTime') < .06
        page.locator('.chapter-button').nth(0).click()
        page.wait_for_function('!elements.audio.paused&&!elements.video.paused')
        page.wait_for_function('__starts.length>0')
        first = page.evaluate('__starts[0]')
        assert first['voice'] == 'af_heart' and not first['pending']
        assert first['audio'] < .1 and first['video'] < .1
        browser.close()


def test_voice_change_holds_position_until_its_timing_arrives(served, tmp_path):
    from playwright.sync_api import sync_playwright

    url = complete_fixture(served, tmp_path)
    delay_resource(served[1], '/audio/es/ef_dora.json', seconds=3)
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        page.locator('.chapter-button').nth(1).click()
        page.wait_for_function('elements.video.currentTime>4.1&&!elements.audio.paused')
        start = page.evaluate('elements.video.currentTime')
        trace_audio(page)
        page.evaluate('elements.language.value="es";void switchLanguage()')
        page.wait_for_function('narrationLoadPending&&videoHeldForAudio')
        page.wait_for_timeout(500)
        assert abs(page.evaluate('elements.video.currentTime') - start) < .08
        assert page.evaluate('elements.audio.paused')
        page.wait_for_function('audioTimings?.voice==="ef_dora"&&!narrationLoadPending&&!elements.video.paused', timeout=15000)
        page.wait_for_function('__starts.length>0')
        first = page.evaluate('__starts[0]')
        assert first['voice'] == 'ef_dora' and not first['pending']
        assert abs(first['audio'] - (1 + start - 4)) < .1
        assert abs(first['video'] - start) < .1
        assert not page.evaluate('__readinessTrace.some(x=>x.pending&&!x.held)')
        browser.close()


def test_a_video_seek_during_voice_load_wins(served, tmp_path):
    from playwright.sync_api import sync_playwright

    url = complete_fixture(served, tmp_path)
    delay_resource(served[1], '/audio/es/ef_dora.json', seconds=3)
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        page.locator('.chapter-button').nth(1).click()
        page.wait_for_function('elements.video.currentTime>4.1&&!elements.audio.paused')
        page.evaluate('elements.language.value="es";void switchLanguage()')
        page.wait_for_function('narrationLoadPending')
        # The chapter list is replaced by its loading state during this fetch.
        # Exercise the media seek event produced by native scrubbing instead.
        page.evaluate('elements.video.currentTime=8')
        assert page.evaluate('narrationLoadPending')
        page.wait_for_function('elements.video.currentTime>=8&&elements.video.paused')
        try:
            page.wait_for_function('audioTimings?.voice==="ef_dora"&&!narrationLoadPending&&!elements.video.paused', timeout=15000)
        except Exception:
            print(page.evaluate('({pending:narrationLoadPending,held:videoHeldForAudio,resume:resumeAfterAudioHold,pauseEvents:audioHoldPauseEventsPending,video:elements.video.currentTime,audio:elements.audio.currentTime,videoPaused:elements.video.paused,audioPaused:elements.audio.paused,voice:audioTimings?.voice,language:elements.language.value,ready:elements.audio.readyState,seeking:elements.audio.seeking,waiting:completeNativeNarrationWaiting})'))
            raise
        assert 8 <= page.evaluate('elements.video.currentTime') < 8.5
        assert 2 <= page.evaluate('elements.audio.currentTime') < 2.5
        browser.close()


@pytest.mark.parametrize("already_playing", [False, True])
@pytest.mark.parametrize("complete_native", [False, True])
def test_audio_decoder_wait_holds_both_clocks_without_a_catchup_seek(served, tmp_path, already_playing, complete_native):
    from playwright.sync_api import sync_playwright

    url = complete_fixture(served, tmp_path)
    if not complete_native:
        path = next(path for name, (path, _) in served[1].routes.items()
                    if name.endswith('/video/native-live-timings.json'))
        legacy = json.loads(path.read_text(encoding="utf-8"))
        legacy.pop('complete_native_clips')
        path.write_text(json.dumps(legacy), encoding="utf-8")
    with sync_playwright() as engine:
        browser = engine.chromium.launch(executable_path=str(CHROME), headless=True,
            args=['--disable-gpu', '--disable-accelerated-video-decode',
                  '--autoplay-policy=no-user-gesture-required'])
        page = browser.new_page()
        page.add_init_script("localStorage.setItem('spacr-tutorial-voice-v2','af_heart');"
                             "localStorage.setItem('spacr-tutorial-quality-v1','1440p');")
        page.goto(url, wait_until='domcontentloaded')
        page.wait_for_function('narrationAudioAvailable&&audioTimings&&!narrationLoadPending')
        assert page.evaluate('completeNativePlayback()') == complete_native
        if already_playing:
            page.locator('.chapter-button').nth(0).click()
            page.wait_for_function('elements.audio.currentTime>.15&&!elements.audio.paused')
        # Controlled decoder starvation: the Blob transport is already fully
        # downloaded, so inject the actual readiness/events rather than claim
        # that another HTTP request can starve this existing Blob resource.
        page.evaluate("""() => {
          window.__readyGetter=Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype,'readyState').get;
          Object.defineProperty(elements.audio,'readyState',{configurable:true,get:()=>2});
          elements.audio.pause();
          elements.audio.dispatchEvent(new Event('waiting'));
          window.__audioSeeks=0;
          elements.audio.addEventListener('seeking',()=>window.__audioSeeks++);
        }""")
        if not already_playing:
            page.locator('.chapter-button').nth(0).click()
        page.wait_for_function('videoHeldForAudio&&elements.video.paused&&!elements.audio.seeking')
        # The explicit chapter click may itself seek to zero. Count only
        # unwanted catch-up seeks after that requested target has settled.
        page.evaluate('__audioSeeks=0')
        before = page.evaluate('[elements.video.currentTime,elements.audio.currentTime]')
        page.wait_for_timeout(250)
        after = page.evaluate('[elements.video.currentTime,elements.audio.currentTime]')
        assert all(abs(a-b) < .06 for a, b in zip(after, before))
        page.evaluate("delete elements.audio.readyState;elements.audio.dispatchEvent(new Event('canplay'))")
        page.wait_for_function('!elements.video.paused&&!elements.audio.paused&&!videoHeldForAudio')
        assert page.evaluate('__audioSeeks') == 0
        browser.close()


def test_superseded_deferred_metadata_cannot_reenable_discarded_audio(served, tmp_path):
    from playwright.sync_api import sync_playwright

    url = complete_fixture(served, tmp_path)
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        # Control only asynchronous completion of the real metadata-ready
        # helper, representing an older decoder callback delivered late.
        # The actual player, audio source and selection handlers remain real.
        page.evaluate("""() => {
          const actual=mediaReadyOrDeferred;
          mediaReadyOrDeferred=async (...args)=>{
            const result=await actual(...args);
            if(elements.voice.value==='ef_dora') {
              window.__oldMetadataPending=true;
              await new Promise(resolve=>window.__releaseOldMetadata=resolve);
            }
            return result;
          };
          elements.language.value='es';void switchLanguage();
        }""")
        page.wait_for_function('window.__oldMetadataPending===true')
        page.evaluate("elements.voice.value='silent';void switchVoice()")
        page.wait_for_function("elements.voice.value==='silent'&&!narrationLoadPending&&!narrationAudioAvailable")
        page.evaluate('__releaseOldMetadata()')
        page.wait_for_timeout(250)
        assert page.evaluate("elements.voice.value==='silent'&&!narrationAudioAvailable&&!narrationLoadPending")
        assert page.evaluate('elements.audio.paused&&!elements.audio.getAttribute("src")')
        browser.close()


def test_metadata_only_midscene_voice_load_keeps_the_requested_offset(served, tmp_path):
    from playwright.sync_api import sync_playwright

    url = complete_fixture(served, tmp_path)
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        page.locator('.chapter-button').nth(1).click()
        page.wait_for_function('elements.video.currentTime>4.25&&!elements.audio.paused')
        # Control the decoder's metadata-known/data-not-ready state. This tests
        # target assignment, not a claim of genuine HTTP decoder starvation.
        page.evaluate("""() => {
          const actual=mediaReadyOrDeferred;
          mediaReadyOrDeferred=async (...args)=>{
            const result=await actual(...args);
            if(elements.voice.value==='ef_dora') {
              Object.defineProperty(elements.audio,'readyState',{
                configurable:true,get:()=>HTMLMediaElement.HAVE_METADATA});
            }
            return result;
          };
          elements.language.value='es';void switchLanguage();
        }""")
        page.wait_for_function('audioTimings?.voice==="ef_dora"&&!narrationLoadPending')
        position = page.evaluate('elements.video.currentTime')
        assert abs(page.evaluate('elements.audio.currentTime') - (1 + position - 4)) < .08
        page.evaluate("delete elements.audio.readyState;elements.audio.dispatchEvent(new Event('canplay'))")
        page.wait_for_function('!elements.audio.paused&&!elements.video.paused')
        browser.close()
