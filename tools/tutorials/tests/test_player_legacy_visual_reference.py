"""Preserved legacy video clocks remain independent of corrected narration."""
import json

import pytest

from test_player_holds_narration import served,CHROME,PAGE


def reference_fixture(served,tmp_path,mutation=None):
    url,handler=served
    script=(PAGE/'lesson_catalog.js').read_text()
    catalog=json.loads(script[script.index('Object.freeze(')+len('Object.freeze('):script.rindex(');')])
    identity=url.split('lesson=')[1]
    lesson=next(row for row in catalog['lessons'] if row['id']==identity)
    count=len(lesson['scenes']);elapsed=0;rows=[]
    for index in range(count):
        duration=60*(index+1)/sum(range(1,count+1))
        rows.append(dict(scene=index+1,speech_start=elapsed,scene_end=elapsed+duration))
        elapsed+=duration
    reference=dict(schema=1,total_duration=60,scenes=rows)
    if mutation=='schema':reference['schema']=2
    elif mutation=='scene_count':reference['scenes']=rows[:-1]
    elif mutation=='native_unaccepted':
        reference.update(kind='native_visual_timing',native_live_video=True,coverage=dict(accepted=False))
    elif mutation=='unknown_kind':reference['kind']='unverified_visual_timing'
    path=tmp_path/'original-visual-timings.json';path.write_text(json.dumps(reference))
    lesson['visual_timings']=f'{identity}/video/original-visual-timings.json'
    data=tmp_path/'legacy-catalog.js'
    data.write_text('window.SPACR_LESSON_CATALOG = Object.freeze('+json.dumps(catalog)+');')
    handler.routes.update({'/web/lesson_catalog.js':(data,'application/javascript'),
                           '/media_host/'+lesson['visual_timings']:(path,'application/json')})
    return url,reference,count


def test_explicit_legacy_reference_keeps_original_clock(served,tmp_path):
    from playwright.sync_api import sync_playwright
    url,reference,count=reference_fixture(served,tmp_path)
    with sync_playwright() as engine:
        browser=engine.chromium.launch(executable_path=str(CHROME),headless=True,
            args=['--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=browser.new_page()
        page.add_init_script("localStorage.setItem('spacr-tutorial-voice-v2','af_heart');")
        page.goto(url,wait_until='domcontentloaded')
        page.wait_for_function('audioTimings&&visualTimings&&chapterData.length>0&&elements.audio.readyState>=2&&elements.video.readyState>=2',timeout=90000)
        assert page.evaluate('visualTimings')==reference
        requested=60/count/2
        page.evaluate('(target)=>seekTo(target)',requested)
        page.wait_for_function('!videoClockCorrectionPending&&!elements.video.seeking&&!elements.audio.seeking',timeout=60000)
        page.wait_for_timeout(100)
        actual=page.evaluate('()=>({audio:elements.audio.currentTime,video:elements.video.currentTime,rate:elements.audio.playbackRate,captions:chapterData.length})')
        assert abs(actual['audio']-requested)<1 and actual['captions']==count
        assert actual['rate']==1
        expected=actual['audio']/(60/count)*reference['scenes'][0]['scene_end']
        assert abs(actual['video']-expected)<.5
        assert abs(actual['video']-actual['audio'])>1
        browser.close()


@pytest.mark.parametrize('mutation',['schema','scene_count','native_unaccepted','unknown_kind'])
def test_explicit_reference_rejects_unverified_metadata(served,tmp_path,mutation):
    from playwright.sync_api import sync_playwright
    url,reference,count=reference_fixture(served,tmp_path,mutation)
    with sync_playwright() as engine:
        browser=engine.chromium.launch(executable_path=str(CHROME),headless=True,args=['--disable-gpu'])
        page=browser.new_page()
        page.goto(url,wait_until='domcontentloaded')
        page.wait_for_function('elements.chapters.textContent.includes("Chapter metadata is unavailable")',timeout=90000)
        assert page.evaluate('audioTimings') is None
        browser.close()
