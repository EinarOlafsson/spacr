"""Real browser playback keeps recorded elapsed time across shorter narration.

The media fixture is explicitly synthetic test material. It exercises the
actual player and media clocks; it is never emitted as tutorial footage.
"""
import json
from pathlib import Path
import subprocess

import pytest
from test_player_holds_narration import served, CHROME, PAGE


def test_native_video_runs_at_one_speed_and_seeks_only_scene_boundaries(served,tmp_path):
    from playwright.sync_api import sync_playwright
    url,handler=served
    script=(PAGE/'lesson_catalog.js').read_text()
    catalog=json.loads(script[script.index('Object.freeze(')+len('Object.freeze('):script.rindex(');')])
    identity=url.split('lesson=')[1];lesson=next(row for row in catalog['lessons'] if row['id']==identity)
    count=len(lesson['scenes']);native_step=60/count;audio_step=native_step*.6
    audio=tmp_path/'short-native-test-audio.m4a'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i',
        f'sine=frequency=330:duration={audio_step*count}','-threads','2','-c:a','aac',str(audio)],check=True)
    rows=[{'scene':i+1,'speech_start':i*audio_step,'speech_end':(i+1)*audio_step-.1,
           'scene_end':(i+1)*audio_step,'duration':audio_step,'text':scene['narration'],
           'sentences':[{'sentence':1,'speech_start':i*audio_step,'speech_end':(i+1)*audio_step-.1,
                         'audible_start':i*audio_step,'audible_end':(i+1)*audio_step-.1,
                         'duration':audio_step-.1,'text':scene['narration']}]}
          for i,scene in enumerate(lesson['scenes'])]
    timing=tmp_path/'short-native-test-timings.json';timing.write_text(json.dumps({'schema':1,
        'language':'en','voice':'af_heart','total_duration':audio_step*count,'scenes':rows}))
    native=tmp_path/'native-test-visual-timings.json';native.write_text(json.dumps({
        'kind':'native_visual_timing','native_live_video':True,'coverage':{'accepted':True},
        'total_duration':60,'scenes':[dict(row,speech_start=i*native_step,
           speech_end=(i+1)*native_step,scene_end=(i+1)*native_step,
           duration=native_step,native_live_video=True) for i,row in enumerate(rows)]}))
    lesson['visual_timings']=f'{identity}/video/native-live-timings.json'
    catalog_path=tmp_path/'native-test-lesson-catalog.js';catalog_path.write_text(
        'window.SPACR_LESSON_CATALOG = Object.freeze('+json.dumps(catalog)+');')
    handler.routes.update({'/web/lesson_catalog.js':(catalog_path,'application/javascript'),
        f'/media_host/{identity}/audio/en/af_heart.m4a':(audio,'audio/mp4'),
        f'/media_host/{identity}/audio/en/af_heart.json':(timing,'application/json'),
        f'/media_host/{identity}/video/native-live-timings.json':(native,'application/json')})
    with sync_playwright() as engine:
        browser=engine.chromium.launch(executable_path=str(CHROME),headless=True,
            args=['--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=browser.new_page();page.add_init_script(
            "localStorage.setItem('spacr-tutorial-voice-v2','af_heart');"
            "localStorage.setItem('spacr-tutorial-quality-v1','1440p');")
        page.goto(url,wait_until='domcontentloaded')
        page.wait_for_function('narrationAudioAvailable && visualTimings?.native_live_video && chapterData.length>0',timeout=90000)
        page.evaluate('elements.video.muted=true;elements.video.play()')
        page.wait_for_function('!elements.video.paused && !elements.audio.paused && !elements.video.seeking',timeout=60000)
        page.evaluate('(target)=>seekTo(target)',audio_step-.8)
        page.wait_for_function('!videoClockCorrectionPending && !elements.video.seeking && !elements.audio.paused',timeout=60000)
        # Observe after the player's frame callback, rather than between a
        # clock crossing its scene boundary and that callback beginning a seek.
        page.evaluate('''() => {
          window.__nativeTrace=[];
          let previous=-Infinity;
          const sample=(now)=>{
            if(now-previous>=50){
              previous=now;
              window.__nativeTrace.push({
                audio:elements.audio.currentTime,video:elements.video.currentTime,
                rate:elements.video.playbackRate,audioRate:elements.audio.playbackRate,
                scene:timingScene(audioTimings,elements.audio.currentTime).index,
                seeking:elements.video.seeking,correcting:videoClockCorrectionPending});
            }
            window.__nativeTimer=requestAnimationFrame(sample);
          };
          window.__nativeTimer=requestAnimationFrame(sample);
        }''')
        page.wait_for_timeout(3500)
        trace=page.evaluate('window.__nativeTrace');page.evaluate('cancelAnimationFrame(window.__nativeTimer)')
        browser.close()
    settled=[row for row in trace if not row['seeking'] and not row['correcting']]
    assert len(settled)>20
    assert len({row['scene'] for row in settled})>=2
    assert all(row['rate']==1 and row['audioRate']==1 for row in settled)
    for row in settled:
        expected=row['audio']+row['scene']*(native_step-audio_step)
        assert abs(row['video']-expected)<.5,row
    # Genuine elapsed motion remains1:1 within a scene, rather than being
    # stretched to the selected voice's duration.
    pairs=[(left,right) for left,right in zip(settled,settled[1:])
           if left['scene']==right['scene'] and right['audio']-left['audio']>.02]
    assert pairs
    assert all(abs((right['video']-left['video'])-(right['audio']-left['audio']))<.2
               for left,right in pairs)


def test_native_video_guard_rejects_disabled_frame_sync(served,tmp_path):
    """Frame-ordered observations still reject a missing boundary correction."""
    from test_player_holds_narration import PLAYER
    url,handler=served
    source=PLAYER.read_text()
    needle='function startNativeFrameSync() {'
    assert source.count(needle)==1
    disabled=tmp_path/'disabled-native-frame-sync.js'
    disabled.write_text(source.replace(needle,needle+'\n  return;'))
    handler.routes['/web/app_v2.js']=(disabled,'application/javascript')
    with pytest.raises(AssertionError,match="'audio':"):
        test_native_video_runs_at_one_speed_and_seeks_only_scene_boundaries((url,handler),tmp_path)
