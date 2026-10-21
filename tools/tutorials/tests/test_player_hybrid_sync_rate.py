from pathlib import Path
import json,os,shutil,subprocess
PLAYER=Path(os.environ.get('SPACR_TEST_PLAYER',str(Path(__file__).resolve().parent.parent/'authoring/web/app_v2.js')))
NODE=os.environ.get('SPACR_TUTORIAL_NODE') or shutil.which('node')
def test_programmed_rate_matches_forward_mapping_for_cards_and_native_intervals():
 assert NODE and Path(NODE).is_file(),'Use the existing bundled Node, never install or skip'
 s=PLAYER.read_text();a=s.index('function updateSceneSyncRate(');b=s.index('// This is the only routine',a)
 function=s[a:b]
 js='''let syncRate=1,programmedVideoRate=1,narrationAudioAvailable=true;
 const visualTimings={complete_native_clips:true,native_live_video:true,total_duration:100,scenes:[{speech_start:0,scene_end:40,native_live_video:false},{speech_start:40,scene_end:100,native_live_video:true}]};
 const audioTimings={total_duration:50,scenes:[{speech_start:0,scene_end:20},{speech_start:20,scene_end:50}]};
 const elements={video:{duration:100,playbackRate:1},audio:{playbackRate:1}};
 function timingScene(t,x){const i=x<20?0:1;return {index:i,scene:t.scenes[i]}}
 function timingDuration(t){return t.total_duration}
 function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
 function completeNativePlayback(){return true}
 '''+function+'''
 updateSceneSyncRate(5);const card=elements.video.playbackRate;
 updateSceneSyncRate(25);const native=elements.video.playbackRate;
 visualTimings.preserve_visual_intervals=true;updateSceneSyncRate(5);const preserved=elements.video.playbackRate;
 console.log(JSON.stringify({card,native,preserved,speech:elements.audio.playbackRate}));'''
 assert json.loads(subprocess.check_output([NODE,'-e',js],text=True))=={'card':2,'native':1,'preserved':1,'speech':1}
