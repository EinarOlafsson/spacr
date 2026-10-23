"""Pure selected-voice timing contract; no Qt/browser/media decoding."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest
P=Path(__file__).resolve().parent
PLAYER=P.parent/"authoring/web/app_v2.js"
NODE=os.environ.get("SPACR_TUTORIAL_NODE") or shutil.which("node")
assert NODE and Path(NODE).is_file(), "Set SPACR_TUTORIAL_NODE to an existing Node runtime"

def call(visual,audio,index=0):
    node=NODE
    assert Path(node).is_file(), 'Existing bundledNode runtime is required'
    script=PLAYER.read_text()
    helper=script.split('function nativeIdleTailCut(',1)[1].split('function syncCompleteNativeNarration(',1)[0]
    js='function nativeIdleTailCut('+helper+'\nconsole.log(JSON.stringify(nativeIdleTailCut('+json.dumps(visual)+','+json.dumps(audio)+','+str(index)+')));'
    return json.loads(subprocess.check_output([node,'-e',js],text=True))

def fixture(last=26.95,idle=27,gap=1.3):
    v={'scenes':[{'speech_start':120.83333333333334,'scene_end':163.13333333333333,'idle_tail_start':idle},{'speech_start':163.13333333333333,'scene_end':180}]}
    a={'scenes':[{'speech_start':70.895,'speech_end':70.895+last+.18,'sentences':[{'audible_end':70.895+last}]},{'speech_start':70.895+last+gap-.1,'sentences':[{'audible_start':70.895+last+gap}]}]}
    return v,a

def test_long_source_gap_is_capped_at_exactly_one_second():
    v,a=fixture();r=call(v,a)
    last_video=v['scenes'][0]['speech_start']+26.95
    next_lead=.1
    assert r['at']-last_video+next_lead==pytest.approx(1)
    assert r['next']==pytest.approx(163.13333333333333)

def test_selected_voice_changes_cut_without_changing_audio():
    for last in [18.0,26.95,39.0]:
        v,a=fixture(last,idle=10);before=json.dumps(a)
        r=call(v,a)
        assert r['at']-(120.83333333333334+last)+.1==pytest.approx(1)
        assert json.dumps(a)==before

def test_essential_action_after_speech_blocks_cut():
    v,a=fixture(idle=30)
    assert call(v,a) is None

def test_missing_explicit_idle_fence_preserves_all_native_footage():
    v,a=fixture();del v['scenes'][0]['idle_tail_start']
    assert call(v,a) is None

def test_no_speech_metadata_or_final_scene_has_no_synthetic_boundary():
    v,a=fixture();a['scenes'][0]['sentences']=[]
    assert call(v,a) is None
    assert call(v,a,1) is None

def test_full_speech_clip_must_fit_before_cut():
    v,a=fixture();a['scenes'][0]['speech_end']=110
    assert call(v,a) is None

def test_malformed_explicit_action_boundary_is_rejected():
    v,a=fixture();v['scenes'][0]['idle_tail_start']=None
    assert call(v,a) is None


def ready(**overrides):
    state=dict(sameTrack=True,sameScene=True,loadPending=False,audioSeeking=False,
      videoSeeking=False,videoPaused=False,pictureLoading=False,explicitSeek=False,
      readyState=4,bufferedSpeechEnd=True,audioTime=98.03,speechEnd=98.025)
    state.update(overrides)
    node=NODE
    script=PLAYER.read_text().split('function nativeIdleTailReady(',1)[1].split('function currentNativeIdleTailState(',1)[0]
    js='function nativeIdleTailReady('+script+'\nconsole.log(JSON.stringify(nativeIdleTailReady('+json.dumps(state)+')));'
    return json.loads(subprocess.check_output([node,'-e',js],text=True))

def test_finished_actual_speech_allows_idle_cut():
    assert ready() is True

@pytest.mark.parametrize('state',[
 {'audioTime':97.9}, {'readyState':2}, {'bufferedSpeechEnd':False},
 {'audioSeeking':True}, {'videoSeeking':True}, {'explicitSeek':True},
 {'loadPending':True}, {'sameTrack':False}, {'sameScene':False},
 {'pictureLoading':True}, {'videoPaused':True}])
def test_lag_buffering_reload_voice_switch_or_seek_never_advances_speech(state):
    assert ready(**state) is False


def wait_clock(previous=None,now=1000,**overrides):
    state=dict(key='07/en/heart/5',sameTrack=True,sameScene=True,loadPending=False,
      audioSeeking=False,videoSeeking=False,videoPaused=False,pictureLoading=False,
      explicitSeek=False,readyState=4,bufferedSpeechEnd=True,audioTime=98.03,
      speechEnd=98.025,lastSound=97.845,nextLead=.1,nextSound=99.145,final=False)
    state.update(overrides)
    node=NODE
    script=PLAYER.read_text().split('function nativeIdleTailReady(',1)[1].split('function currentNativeIdleTailState(',1)[0]
    js='function nativeIdleTailReady('+script+'\nconsole.log(JSON.stringify(nativeIdleTailWait('+json.dumps(previous)+','+json.dumps(state)+','+str(now)+')));'
    return json.loads(subprocess.check_output([node,'-e',js],text=True))

def test_lagged_speech_completion_still_gets_its_actual_remaining_pause():
    pending=wait_clock(now=12000,audioTime=97.9)
    assert pending=={'clock':None,'ready':False}
    # Video has already passed its nominal cut; actual speech completes now.
    first=wait_clock(now=15000)
    assert first['ready'] is False
    assert wait_clock(first['clock'],now=15719)['ready'] is False
    assert wait_clock(first['clock'],now=15721)['ready'] is True

@pytest.mark.parametrize('reset',[
 {'videoPaused':True},{'loadPending':True},{'audioSeeking':True},
 {'videoSeeking':True},{'sameTrack':False},{'readyState':2}])
def test_pause_loading_seek_or_switch_resets_actual_pause_deadline(reset):
    first=wait_clock(now=1000)
    assert wait_clock(first['clock'],now=1500,**reset)['clock'] is None
    resumed=wait_clock(now=3000)
    assert resumed['ready'] is False
    assert wait_clock(resumed['clock'],now=3721)['ready'] is True

def test_new_voice_scene_or_reload_cannot_reuse_old_deadline():
    first=wait_clock(now=1000)
    switched=wait_clock(first['clock'],now=10000,key='07/en/emma/5')
    assert switched['ready'] is False
    assert switched['clock']['completedAt']==10000


def test_final_padding_can_reach_real_video_eof_after_all_speech():
    v={'scenes':[{'speech_start':0,'scene_end':30,'idle_tail_start':4.9}]}
    a={'scenes':[{'speech_start':0,'speech_end':5,'sentences':[{'audible_end':4.82}]}]}
    cut=call(v,a)
    assert cut['final'] is True and cut['next']==30
    assert cut['at']==pytest.approx(5.82)
    assert cut['at']>=a['scenes'][0]['speech_end']

def test_actual_player_blob_identity_accepts_current_fetch_and_rejects_stale_blob():
    script=PLAYER.read_text().split('function currentNativeIdleTailState(',1)[1].split('function syncCompleteNativeNarration(',1)[0]
    prelude="""
      let narrationObjectUrl='blob:http://test/current';
      const narrationRequest=7,activeLesson={id:'07_mask'},completeNativeSceneIndex=0;
      const narrationLoadPending=false,videoClockCorrectionPending=false;
      const audioTimings={voice:'af_heart',language:'en',scenes:[{speech_start:0,speech_end:5,sentences:[{audible_end:4.82}]},{speech_start:5.52,sentences:[{audible_start:5.62}]}]};
      const elements={voice:{value:'af_heart'},language:{value:'en'},
        audio:{currentSrc:'blob:http://test/current',currentTime:5.01,readyState:4,seeking:false,buffered:{length:1,start:()=>0,end:()=>20}},
        video:{seeking:false,paused:false}};
      function videoIsLoading(){return false;}
    """
    js=prelude+'function currentNativeIdleTailState('+script+"\nconsole.log(JSON.stringify([currentNativeIdleTailState(audioTimings.scenes[0],0,false).sameTrack,(narrationObjectUrl='blob:http://test/next',currentNativeIdleTailState(audioTimings.scenes[0],0,false).sameTrack)]));"
    result=json.loads(subprocess.check_output([NODE,'-e',js],text=True))
    assert result==[True,False]


def test_decoded_fractional_boundary_stays_in_next_scene_first_frame():
    source=PLAYER.read_text()
    budget='function nativeIdleGapBudget('+source.split('function nativeIdleGapBudget(',1)[1].split('let nativeIdlePauseClock',1)[0]
    begin='function beginNativeIdleTransition('+source.split('function beginNativeIdleTransition(',1)[1].split('function syncCompleteNativeNarration(',1)[0]
    scene='function timingDuration('+source.split('function timingDuration(',1)[1].split('function mapTiming(',1)[0]
    js="""
      let nativeIdleTransition=null,completeNativeNarrationWaiting=false;
      const narrationRequest=1,activeLesson={id:'07_mask'};
      const visualTimings={fps:30,total_duration:180,scenes:[{scene_end:163.13333333333333},{scene_end:180}]};
      const audioTimings={scenes:[{speech_start:70.895},{speech_start:98.545}]};
      const elements={voice:{value:'af_heart'},language:{value:'en'},audio:{pause(){},currentTime:98.025},video:{duration:180}};
      let destination;
      function nativeIdleTailReady(){return true;}
      function holdVideoForAudio(){}
      function updateCaptionTrackMode(){}
      function scheduleNativeIdleTransition(){}
      function clamp(x,a,b){return Math.min(b,Math.max(a,x));}
      let performance={now:()=>1000};
      function nativeIdleQuietRate(audioTime, quietEnd, now, endAt) {
  // Only post-clip silence is elastic. Return to ordinary speed before the
  // next complete speech clip, preserving its entire conservative lead.
  const reserve = 0.1;
  const source = quietEnd - audioTime - reserve;
  const wall = (endAt - now) / 1000 - reserve;
  return source <= 0 || wall <= 0 ? 1 : clamp(source / wall, 0.0625, 1);
}


      function setVideoTimeWithoutNarrationSeek(value){destination=Number(value.toFixed(6));}
    """+scene+budget+begin+"""
      beginNativeIdleTransition({final:false,next:163.13333333333333},{speechEnd:98.025,lastSound:97.845,nextLead:.1,nextSound:99.145,final:false},{completedAt:1000},0);
      console.log(JSON.stringify({index:timingScene(visualTimings,destination).index,offset:destination-163.13333333333333,audio:elements.audio.currentTime}));
    """
    result=json.loads(subprocess.check_output([NODE,'-e',js],text=True))
    assert result['index']==1
    assert 0<=result['offset']<1/30
    assert result['audio']==98.025


def test_only_verified_quiet_is_stretched_and_rate_returns_before_next_clip():
    source=PLAYER.read_text()
    helper='function nativeIdleQuietRate('+source.split('function nativeIdleQuietRate(',1)[1].split('function serviceNativeIdleTransition(',1)[0]
    js='function clamp(x,a,b){return Math.min(b,Math.max(a,x));}'+helper+"console.log(JSON.stringify([nativeIdleQuietRate(1,1.52,0,720),nativeIdleQuietRate(1.43,1.52,600,720),nativeIdleQuietRate(1.52,1.52,720,720)]));"
    result=json.loads(subprocess.check_output([NODE,'-e',js],text=True))
    assert .0625<=result[0]<1
    assert result[1:]==[1,1]


def test_cancelling_transition_immediately_restores_natural_audio_rate():
    source=PLAYER.read_text()
    helper='function cancelNativeIdleTransition('+source.split('function cancelNativeIdleTransition(',1)[1].split('function scheduleNativeIdleTransition(',1)[0]
    js="let nativeIdleTransitionTimer=null,nativeIdleTransition={},nativeIdlePauseClock={};const elements={audio:{playbackRate:.7}};function updateCaptionTrackMode(){};"+helper+"cancelNativeIdleTransition();console.log(JSON.stringify([elements.audio.playbackRate,nativeIdleTransition,nativeIdlePauseClock]));"
    assert json.loads(subprocess.check_output([NODE,'-e',js],text=True))==[1,None,None]


@pytest.mark.parametrize('video_time,expected_audio',[(6,3),(25,15)])
def test_hybrid_card_and_native_user_seeks_invert_picture_mapping(video_time,expected_audio):
    source=PLAYER.read_text()
    mapping='function timingDuration('+source.split('function timingDuration(',1)[1].split('function updateSceneSyncRate(',1)[0]
    late='function nativeLateIdleTailCut('+source.split('function nativeLateIdleTailCut(',1)[1].split('function beginNativeLateIdleTransition(',1)[0]
    sync='function syncCompleteNativeNarration('+source.split('function syncCompleteNativeNarration(',1)[1].split('function holdVideoForAudio(',1)[0]
    js="""
      let nativeIdleTransition=null,narrationLoadPending=false,narrationAudioAvailable=true,videoClockCorrectionPending=false;
      let completeNativeSceneIndex=null,completeNativeNarrationWaiting=false,nativeIdlePauseClock=null,videoHeldForAudio=false;
      const visualTimings={native_live_video:true,complete_native_clips:true,total_duration:40,scenes:[{speech_start:0,scene_end:20,native_live_video:false},{speech_start:20,scene_end:40,native_live_video:true}]};
      const audioTimings={total_duration:30,scenes:[{speech_start:0,speech_end:9.5,scene_end:10},{speech_start:10,speech_end:29.5,scene_end:30}]};
      const elements={voice:{value:'af_heart'},video:{duration:40,currentTime:VIDEO_TIME,paused:true,seeking:false},audio:{duration:30,currentTime:0,paused:true,pause(){}}};
      function clamp(x,a,b){return Math.min(b,Math.max(a,x));}
      function completeNativePlayback(){return true;}
      function nativeIdleTailCut(){return null;}
      function currentNativeIdleTailState(){return {};}
      function nativeIdleTailWait(){return {clock:null};}
      function updateSceneSyncRate(){}
      function audioNeedsData(){return false;}
      function releaseVideoForAudio(){}
    """.replace('VIDEO_TIME',str(video_time))+mapping+late+sync+"syncCompleteNativeNarration(true);console.log(elements.audio.currentTime);"
    actual=float(subprocess.check_output([NODE,'-e',js],text=True))
    assert actual==pytest.approx(expected_audio)


def test_runtime_fences_require_matching_lesson_timing_and_scene_clock():
    source=PLAYER.read_text()
    helper='function boundNativeIdleTimings('+source.split('function boundNativeIdleTimings(',1)[1].split('async function loadLessonDetail(',1)[0]
    reference={'scenes':[{'visual':'table','speech_start':120,'scene_end':163}]}
    policy={'schema':1,'lesson':'07_mask','master_sha256':'a'*64,'visual_timing_sha256':'b'*64,'scenes':[{'visual':'table','source_start':120,'source_end':163,'idle_tail_start':26.2}]}
    outcomes=[]
    for mutation in ['valid','missing','timing','lesson','clock']:
        p=json.loads(json.dumps(policy))
        if mutation=='timing':p['visual_timing_sha256']='c'*64
        elif mutation=='lesson':p['lesson']='08_measure'
        elif mutation=='clock':p['scenes'][0]['source_end']=164
        lesson={'id':'07_mask'}
        if mutation!='missing':lesson['native_idle_tails']=p
        js=helper+'console.log(JSON.stringify(boundNativeIdleTimings('+json.dumps(reference)+','+json.dumps(lesson)+','+json.dumps('b'*64)+').scenes[0].idle_tail_start ?? null));'
        outcomes.append(json.loads(subprocess.check_output([NODE,'-e',js],text=True)))
    assert outcomes==[26.2,None,None,None,None]


def test_late_fence_inside_gap_but_after_full_clip_preserves_footage():
    v,a=fixture(idle=27.6)
    assert a['scenes'][0]['speech_end']-a['scenes'][0]['speech_start']<27.6<27.85
    assert call(v,a) is None


def test_paused_narration_cannot_enter_continuous_idle_transition():
    source=PLAYER.read_text()
    helper='function beginNativeIdleTransition('+source.split('function beginNativeIdleTransition(',1)[1].split('function syncCompleteNativeNarration(',1)[0]
    js="let nativeIdleTransition=null;const elements={audio:{paused:true,playbackRate:1}};function nativeIdleTailReady(){return true;}"+helper+"beginNativeIdleTransition({final:false},{},{},0);console.log(JSON.stringify([nativeIdleTransition,elements.audio.playbackRate]));"
    assert json.loads(subprocess.check_output([NODE,'-e',js],text=True))==[None,1]


def test_short_source_gap_is_not_padded_by_picture_cut_or_observed_wait():
    v,a=fixture(gap=.8);r=call(v,a)
    assert r['at']-(v['scenes'][0]['speech_start']+26.95)+.1==pytest.approx(.8)
    first=wait_clock(now=1000,nextSound=98.645)
    assert wait_clock(first['clock'],now=1519,nextSound=98.645)['ready'] is False
    assert wait_clock(first['clock'],now=1521,nextSound=98.645)['ready'] is True
