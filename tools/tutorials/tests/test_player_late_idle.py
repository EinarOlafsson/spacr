"""Behavior tests for completed native narration and reviewed late actions."""
from pathlib import Path
import json,os,shutil,subprocess
import pytest
P=Path(__file__).resolve().parent
PLAYER=Path(os.environ.get('SPACR_TEST_PLAYER',str(P.parent/'authoring/web/app_v2.js')))
NODE=os.environ.get('SPACR_TUTORIAL_NODE') or shutil.which('node')

def part(source,start,end):return start+source.split(start,1)[1].split(end,1)[0]

def run(body):
    assert NODE and Path(NODE).is_file(),'Use the existing bundled Node, never install or skip'
    source=PLAYER.read_text()
    functions='\n'.join([
        part(source,'function nativeIdleTailCut(','function nativeIdleTailReady('),
        part(source,'function nativeIdleTailReady(','let nativeIdlePauseClock'),
        part(source,'function nativeIdleTailWait(','function currentNativeIdleTailState('),
        part(source,'function currentNativeIdleTailState(','function cancelNativeIdleTransition('),
        part(source,'function nativeIdleQuietRate(','function serviceNativeIdleTransition('),
        part(source,'function beginNativeIdleTransition(','function nativeLateIdleTailCut('),
        part(source,'function syncCompleteNativeNarration(','function holdVideoForAudio('),
        part(source,'function timingScene(','function mapTiming('),
        part(source,'function cancelNativeIdleTransition(','function scheduleNativeIdleTransition('),
        part(source,'function scheduleNativeIdleTransition(','function nativeIdleQuietRate('),
        part(source,'function nativeLateIdleTailCut(','function syncCompleteNativeNarration('),
        part(source,'function serviceNativeIdleTransition(','function beginNativeIdleTransition('),
        part(source,'function releaseVideoForAudio(','function startNarrationAudio('),
        part(source,'function startNarrationAudio(','function updateNarrationHold('),
        part(source,'function playVideoWithoutNarrationSync(','function parkVideoForNarration('),
        part(source,'function seekTo(audioSeconds)','function chapterLabel('),
    ])
    prelude='''
      let now=1000,actualAudio=10,nativeIdleTransition=null,nativeIdleTransitionTimer=null,nativeIdlePauseClock=null;
      let narrationRequest=7,narrationLoadPending=false,narrationAudioAvailable=true,videoCatchingUpToNarration=false,narrationObjectUrl='blob:current';
      let completeNativeSceneIndex=0,completeNativeNarrationWaiting=true,videoClockCorrectionPending=false;
      let videoHeldForAudio=false,resumeAfterAudioHold=false,programmaticVideoPlayPending=false;
      let resolveStart=null,deferStart=false,rejectStart=false;
      const performance={now:()=>now},HTMLMediaElement={HAVE_FUTURE_DATA:3};
      const stats={audioSeeks:[],audioStarts:0,videoStarts:0,targets:[],toasts:[]};
      const activeLesson={id:'07_mask'};
      const visualTimings={fps:30,total_duration:70,scenes:[{speech_start:0,scene_end:50,idle_tail_start:42.6},{speech_start:50,scene_end:70,idle_tail_start:18}]};
      const audioTimings={total_duration:30,language:'en',voice:'af_heart',scenes:[{speech_start:0,speech_end:10,scene_end:10.52,sentences:[{audible_end:9.82}]},{speech_start:10.52,speech_end:29,scene_end:30,sentences:[{audible_start:10.62,audible_end:28.82}]}]};
      const elements={voice:{value:'af_heart'},language:{value:'en'},loading:{classList:{add(){}}},
        audio:{buffered:{length:1,start(){return 0},end(){return 30}},duration:30,currentSrc:'blob:current',paused:true,seeking:false,readyState:4,playbackRate:1,
          get currentTime(){return actualAudio},set currentTime(x){stats.audioSeeks.push(x);actualAudio=x},
          pause(){this.paused=true},play(){stats.audioStarts++;this.paused=false;
            if(rejectStart)return Promise.reject(new Error('actual start failed'));
            return deferStart?new Promise(resolve=>{resolveStart=resolve}):Promise.resolve()}},
        video:{duration:70,currentTime:42.6,paused:false,seeking:false,readyState:4,ended:false,
          play(){this.paused=false;stats.videoStarts++;return Promise.resolve()},pause(){this.paused=true}}};
      function setTimeout(){return 1} function clearTimeout(){}
      function updateCaptionTrackMode(){} function showToast(text){stats.toasts.push(text)}
      function videoIsLoading(){return videoClockCorrectionPending||elements.video.readyState<3}
      function audioNeedsData(){return false}
      function holdVideoForAudio(){videoHeldForAudio=true;resumeAfterAudioHold=true;elements.video.pause()}
      function setVideoTimeWithoutNarrationSeek(x){stats.targets.push(x);elements.video.currentTime=x;videoClockCorrectionPending=true}
      function clamp(x,a,b){return Math.min(b,Math.max(a,x))}
      function timingDuration(t,fallback=0){return t.total_duration||fallback}
      function completeNativePlayback(){return true} function updateSceneSyncRate(){}
      function videoTimeFromAudio(x){return x} function setVideoTimeFromNarration(x){elements.video.currentTime=x}
      function state(){return {sameTrack:true,sameScene:true,loadPending:false,audioSeeking:false,videoSeeking:false,
        videoPaused:elements.video.paused,pictureLoading:false,explicitSeek:false,readyState:4,bufferedSpeechEnd:true,
        audioTime:actualAudio,speechEnd:10,lastSound:9.82,nextLead:.1,nextSound:10.62,final:false}}
      function cut(){return nativeLateIdleTailCut(visualTimings,audioTimings,0)}
      function decoded(){videoClockCorrectionPending=false}
      async function tick(){await Promise.resolve();await Promise.resolve();await Promise.resolve()}
    '''
    js=prelude+functions+'\n(async()=>{'+body+'})().catch(error=>{console.error(error);process.exit(1)});'
    return json.loads(subprocess.check_output([NODE,'-e',js],text=True))


def test_late_fence_requires_full_speech_actual_running_picture_and_fence_reached():
    r=run('''const c=cut();
      const before=beginNativeLateIdleTransition(c,state(),0,42.59);
      actualAudio=9.99;const unfinished=beginNativeLateIdleTransition(c,state(),0,42.6);
      actualAudio=10;elements.video.paused=true;const paused=beginNativeLateIdleTransition(c,state(),0,42.6);
      elements.video.paused=false;const begun=beginNativeLateIdleTransition(c,state(),0,42.6);
      console.log(JSON.stringify({before,unfinished,paused,begun,starts:stats.audioStarts,seeks:stats.audioSeeks}));''')
    assert r=={'before':False,'unfinished':False,'paused':False,'begun':True,'starts':0,'seeks':[]}


def test_decoded_late_transition_keeps_all_pcm_and_holds_picture_until_next_clip():
    r=run('''beginNativeLateIdleTransition(cut(),state(),0,42.6);
      serviceNativeIdleTransition();const startsWhileLoading=stats.audioStarts;
      decoded();serviceNativeIdleTransition();await tick();
      const resumed={time:actualAudio,starts:stats.audioStarts,pictureStarts:stats.videoStarts,phase:nativeIdleTransition.phase};
      actualAudio=10.51;serviceNativeIdleTransition();const beforeNext=stats.videoStarts;
      actualAudio=10.52;serviceNativeIdleTransition();
      console.log(JSON.stringify({startsWhileLoading,resumed,beforeNext,after:stats.videoStarts,seeks:stats.audioSeeks,
        rate:elements.audio.playbackRate,scene:completeNativeSceneIndex,state:nativeIdleTransition}));''')
    assert r['startsWhileLoading']==0
    assert r['resumed']=={'time':10,'starts':1,'pictureStarts':0,'phase':'quiet'}
    assert r['beforeNext']==0 and r['after']==1
    assert r['seeks']==[] and r['rate']==1 and r['scene']==1 and r['state'] is None


def test_explicit_user_seek_cancels_late_state_without_restoring_old_audio_position():
    r=run('''beginNativeLateIdleTransition(cut(),state(),0,42.6);
      seekTo(20);decoded();serviceNativeIdleTransition();
      console.log(JSON.stringify({audio:actualAudio,seeks:stats.audioSeeks,starts:stats.audioStarts,state:nativeIdleTransition}));''')
    assert r=={'audio':20,'seeks':[20],'starts':0,'state':None}


def test_cancelled_user_pause_cannot_resume_when_pending_play_promise_later_resolves():
    r=run('''beginNativeLateIdleTransition(cut(),state(),0,42.6);decoded();deferStart=true;
      serviceNativeIdleTransition();
      // The actual player's user-pause path cancels, clears held resume intent,
      // and pauses narration. Model those effects before the old promise returns.
      cancelNativeIdleTransition();videoHeldForAudio=false;resumeAfterAudioHold=false;elements.audio.pause();
      resolveStart();await tick();serviceNativeIdleTransition();
      console.log(JSON.stringify({paused:elements.audio.paused,rate:elements.audio.playbackRate,seeks:stats.audioSeeks,
        state:nativeIdleTransition,pictureStarts:stats.videoStarts}));''')
    assert r=={'paused':True,'rate':1,'seeks':[],'state':None,'pictureStarts':0}


@pytest.mark.parametrize('change',["narrationRequest++","narrationObjectUrl='blob:new'","elements.voice.value='bf_emma'","narrationLoadPending=true"])
def test_changed_request_voice_or_blob_aborts_before_any_audio_resume(change):
    r=run('beginNativeLateIdleTransition(cut(),state(),0,42.6);'+change+''';decoded();serviceNativeIdleTransition();
      console.log(JSON.stringify({state:nativeIdleTransition,starts:stats.audioStarts,seeks:stats.audioSeeks,rate:elements.audio.playbackRate}));''')
    assert r=={'state':None,'starts':0,'seeks':[],'rate':1}


def test_failed_audio_start_preserves_position_and_requires_viewer_action():
    r=run('''beginNativeLateIdleTransition(cut(),state(),0,42.6);decoded();rejectStart=true;
      serviceNativeIdleTransition();await tick();
      console.log(JSON.stringify({state:nativeIdleTransition,audio:actualAudio,seeks:stats.audioSeeks,
        pictureStarts:stats.videoStarts,toasts:stats.toasts.length}));''')
    assert r=={'state':None,'audio':10,'seeks':[],'pictureStarts':0,'toasts':1}


def test_final_late_transition_plays_actual_last_frame_without_manufacturing_eof():
    r=run('''visualTimings.scenes.length=1;visualTimings.total_duration=50;
      audioTimings.scenes.length=1;elements.video.duration=50;
      const c=cut();beginNativeLateIdleTransition(c,state(),0,42.6);decoded();serviceNativeIdleTransition();
      console.log(JSON.stringify({final:c.final,target:stats.targets[0],pictureStarts:stats.videoStarts,
        ended:elements.video.ended,audioStarts:stats.audioStarts,seeks:stats.audioSeeks,state:nativeIdleTransition}));''')
    assert r['final'] and r['target']==pytest.approx(50-1/30)
    assert r['pictureStarts']==1 and r['ended'] is False and r['audioStarts']==0 and r['seeks']==[] and r['state'] is None


def test_missing_fence_or_next_clip_already_entered_cannot_skip_picture_tail():
    r=run('''delete visualTimings.scenes[0].idle_tail_start;const missing=cut();
      visualTimings.scenes[0].idle_tail_start=42.6;actualAudio=10.53;
      const begun=beginNativeLateIdleTransition(cut(),state(),0,42.6);
      console.log(JSON.stringify({missing,begun,seeks:stats.audioSeeks,targets:stats.targets}));''')
    assert r=={'missing':None,'begun':False,'seeks':[],'targets':[]}


@pytest.mark.parametrize('picture_time,expect_transition',[(9.8,False),(9.9,True)])
def test_actual_early_sync_waits_for_picture_to_reach_bound_fence_even_when_audio_finished(picture_time,expect_transition):
    r=run('visualTimings.scenes[0].idle_tail_start=9.9;elements.audio.paused=false;elements.video.currentTime='+str(picture_time)+';syncCompleteNativeNarration(false);console.log(JSON.stringify({transition:!!nativeIdleTransition,seeks:stats.audioSeeks,targets:stats.targets.length,audio:actualAudio}));')
    assert r['transition'] is expect_transition
    assert r['targets']==int(expect_transition)
    assert r['seeks']==[] and r['audio']==10


@pytest.mark.parametrize('original_gap,remaining',[(.8,.52),(1.3,.72)])
def test_regular_transition_uses_same_validated_budget_without_speech_seek(original_gap,remaining):
    body="""visualTimings.scenes[0].idle_tail_start=1;
      audioTimings.scenes[1].speech_start=9.82+GAP-.1;
      audioTimings.scenes[1].sentences[0].audible_start=9.82+GAP;
      completeNativeNarrationWaiting=false; // This case starts in active unfinished speech, unlike the late-action fixture.
      actualAudio=9.99;elements.audio.paused=false;elements.video.currentTime=10;
      syncCompleteNativeNarration();const unfinished=nativeIdleTransition;
      actualAudio=10;syncCompleteNativeNarration();
      const begun={deadline:nativeIdleTransition.resumeAt,rate:elements.audio.playbackRate,
        seeks:stats.audioSeeks,starts:stats.audioStarts};
      cancelNativeIdleTransition();
      console.log(JSON.stringify({unfinished,begun,cancelRate:elements.audio.playbackRate}));"""
    r=run(body.replace('GAP',str(original_gap)))
    assert r['unfinished'] is None
    assert r['begun']['deadline']==pytest.approx(1000+remaining*1000)
    assert r['begun']['seeks']==[] and r['begun']['starts']==0 and r['cancelRate']==1
    if original_gap<1:assert r['begun']['rate']==pytest.approx(1)


def test_final_gap_budget_stays_one_second_and_malformed_gap_fails_closed():
    r=run("""console.log(JSON.stringify([
      nativeIdleGapBudget(9.82,10,true),nativeIdleGapBudget(9.82,10.62,false),
      nativeIdleGapBudget(9.82,9,false),nativeIdleGapBudget(9.82,NaN,false)]));""")
    assert r[0]==1 and r[1]==pytest.approx(.8) and r[2:]==[None,None]
