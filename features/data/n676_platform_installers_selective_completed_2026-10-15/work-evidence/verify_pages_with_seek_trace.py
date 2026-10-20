from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'tools/tutorials')]
import json,time
from urllib.parse import urlsplit,urlunsplit
import verify_release_candidate as owning
W=Path(__file__).resolve().parent
class FocusedDiagnosticComplete(BaseException): pass
original=owning.check_sentence_cues

def diagnostic_cues(page):
    began=time.monotonic();events=[]
    page.evaluate("""() => {
      window.__audioSeekTrace={writes:[],events:[]};
      const audio=elements.audio;
      const descriptor=Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype,'currentTime');
      const snapshot=()=>({at:performance.now(),audio:descriptor.get.call(audio),audioSeeking:audio.seeking,audioReady:audio.readyState,video:elements.video.currentTime,videoSeeking:elements.video.seeking,videoReady:elements.video.readyState,correction:videoClockCorrectionPending,scene:completeNativeSceneIndex,waiting:completeNativeNarrationWaiting});
      Object.defineProperty(audio,'currentTime',{...descriptor,get(){return Reflect.apply(descriptor.get,this,[])},set(value){
        const before=snapshot();
        const result=Reflect.apply(descriptor.set,this,[value]);
        window.__audioSeekTrace.writes.push({value,before,after:snapshot(),stack:new Error().stack});
        if(window.__audioSeekTrace.writes.length>2000)window.__audioSeekTrace.writes.shift();
        return result;
      }});
      for(const media of [audio,elements.video])for(const name of ['seeking','seeked','loadeddata','canplay','waiting','stalled','pause','error','emptied','abort']){
        media.addEventListener(name,()=>{window.__audioSeekTrace.events.push({media:media===audio?'audio':'video',name,...snapshot()});if(window.__audioSeekTrace.events.length>3000)window.__audioSeekTrace.events.shift()});
      }
    }""")
    def safe(url):
        q=urlsplit(url);return urlunsplit((q.scheme,q.netloc,q.path,'',''))
    def response(r):
        if 'huggingface.co' in r.url or '.hf.co' in r.url:
            events.append(dict(seconds=round(time.monotonic()-began,3),url=safe(r.url),status=r.status,range=r.headers.get('content-range'),length=r.headers.get('content-length')))
    def failed(r):
        events.append(dict(seconds=round(time.monotonic()-began,3),url=safe(r.url),failure=r.failure))
    page.on('response',response);page.on('requestfailed',failed)
    try:
        result=original(page)
        if page.evaluate("activeLesson.id")=="04_platform_installers":
            out=W/"pages-r3-focused-pass-trace.json"
            with out.open("x") as f:json.dump(dict(scope="04 focused diagnostics only; not84 acceptance",cases=result,trace=page.evaluate("window.__audioSeekTrace")),f,indent=2)
            raise FocusedDiagnosticComplete()
        return result
    except Exception as error:
        state=page.evaluate("""() => ({lesson:activeLesson?.id, correctionPending:videoClockCorrectionPending,
          sceneIndex:completeNativeSceneIndex, narrationWaiting:completeNativeNarrationWaiting,
          videoHeldForAudio, resumeAfterAudioHold, narrationHeldForVideo,
          audio:{time:elements.audio.currentTime,paused:elements.audio.paused,seeking:elements.audio.seeking,ready:elements.audio.readyState,network:elements.audio.networkState,error:elements.audio.error?.message,buffered:[...Array(elements.audio.buffered.length)].map((_,i)=>[elements.audio.buffered.start(i),elements.audio.buffered.end(i)])},
          video:{time:elements.video.currentTime,paused:elements.video.paused,seeking:elements.video.seeking,ready:elements.video.readyState,network:elements.video.networkState,error:elements.video.error?.message,buffered:[...Array(elements.video.buffered.length)].map((_,i)=>[elements.video.buffered.start(i),elements.video.buffered.end(i)])},
          expectedVideo:videoTimeFromAudio(elements.audio.currentTime)})""")
        state.update(scope='Unmodified owning sentence-cue failure diagnostics; no threshold or playback changes',elapsed_seconds=time.monotonic()-began,error=type(error).__name__,network_events=events[-150:],setter_trace=page.evaluate("window.__audioSeekTrace"))
        out=W/'pages-r3-seek-failure-trace.json'
        with out.open('x') as f:json.dump(state,f,indent=2);f.write('\n')
        print(json.dumps({k:v for k,v in state.items() if k not in ['network_events','setter_trace']}),flush=True)
        raise
    finally:
        page.remove_listener('response',response);page.remove_listener('requestfailed',failed)

owning.check_sentence_cues=diagnostic_cues
try:
    owning.verify(Path(sys.argv[1]),published=Path(sys.argv[2]))
except FocusedDiagnosticComplete:
    print("04 focused diagnostics passed original cue checks; NOT84 publication acceptance",flush=True)
