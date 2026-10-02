from pathlib import Path
import datetime,hashlib,json,re,time
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
out=Path(__file__).parent;readback=json.loads((out/'readback.json').read_text());site=readback['site'];media=readback['media_root'];identity='14_make_masks';records=[];errors=[];foreign=[];hosted=[]
expected_audio=next(x['sha256'] for x in readback['fresh_artifacts'] if x['path'].endswith('af_heart.m4a'))
def route(r):
 u=r.request.url;host=urlparse(u).hostname or ''
 if host==urlparse(site).hostname or u.startswith(('data:','blob:')):r.continue_()
 elif u.startswith(media+'/') or (host!='huggingface.co' and host.endswith(('.hf.co','.huggingface.co'))):hosted.append(u);r.continue_()
 else:foreign.append(u);r.abort()
with sync_playwright() as pw:
 browser=pw.chromium.launch(executable_path='/opt/google/chrome/chrome',headless=True,args=['--disable-gpu','--disable-accelerated-video-decode'])
 for label,width,height,mobile in [('desktop',1440,1000,False),('mobile',390,844,True)]:
  ctx=browser.new_context(viewport={'width':width,'height':height},is_mobile=mobile,has_touch=mobile,device_scale_factor=3 if mobile else 1);ctx.route('**/*',route)
  ctx.add_init_script("localStorage.setItem('spacr-tutorial-language-v2','en');localStorage.setItem('spacr-tutorial-voice-v2','af_heart');")
  page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.goto(site+'#lesson='+identity,wait_until='networkidle',timeout=120000)
  page.wait_for_function("activeLesson?.id === '14_make_masks' && narrationAudioAvailable && elements.audio.readyState >= 1 && chapterData.length===32",timeout=180000)
  page.evaluate('closeSidebar()');page.wait_for_timeout(400)
  actual=page.evaluate("async()=>{const b=await(await fetch(elements.audio.src)).arrayBuffer();return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b)),x=>x.toString(16).padStart(2,'0')).join('');}")
  assert actual==expected_audio
  assert page.evaluate('audioTimings.media_sha256')==actual
  layout=page.evaluate("()=>{const r=elements.video.getBoundingClientRect();return {scroll:document.documentElement.scrollWidth,viewport:innerWidth,video:[r.x,r.y,r.width,r.height]};}")
  assert layout['scroll']<=width;assert layout['video'][0]>=0 and layout['video'][0]+layout['video'][2]<=width+.5
  page.locator('#tutorial-video').scroll_into_view_if_needed()
  if mobile:page.tap('#tutorial-video')
  else:page.click('#tutorial-video')
  started='native synthetic tap' if mobile else 'native synthetic click'
  try:page.wait_for_function('!elements.audio.paused && elements.audio.currentTime>1',timeout=3000)
  except Exception:
   started+='; explicit production video.play fallback';page.evaluate('elements.video.play()');page.wait_for_function('!elements.audio.paused && elements.audio.currentTime>1',timeout=90000)
  cases=[]
  for i in readback['restoration_indices']:
   text=page.evaluate('(i)=>chapterData[i].text',i);expected=readback['restoration_scenes'][i-21]['catalog']['narration'];assert text==expected
   requested=page.evaluate('(i)=>chapterData[i].start',i);page.evaluate('(t)=>{seekTo(t);elements.video.play();}',requested)
   page.wait_for_function("(t)=>!videoClockCorrectionPending&&!elements.video.seeking&&!elements.audio.seeking&&elements.audio.currentTime>t+.3&&Math.abs(elements.video.currentTime-videoTimeFromAudio(elements.audio.currentTime))<.5",arg=requested,timeout=90000,polling=50)
   before=page.evaluate('elements.audio.currentTime');page.wait_for_timeout(1000)
   playing=page.evaluate("()=>({audio:elements.audio.currentTime,video:elements.video.currentTime,expected:videoTimeFromAudio(elements.audio.currentTime),paused:elements.video.paused||elements.audio.paused,error:elements.video.error?.message||elements.audio.error?.message||null,video_src:elements.video.currentSrc})")
   assert playing['audio']>before+.25 and not playing['paused'] and not playing['error'];assert abs(playing['video']-playing['expected'])<.5;assert playing['video_src'].startswith(media+'/')
   sentence=page.evaluate('(i)=>audioTimings.scenes[i].sentences[0]',i);mid=(sentence['speech_start']+sentence['speech_end'])/2
   for attempt in range(4):
    page.wait_for_function('!videoClockCorrectionPending&&!elements.video.seeking&&!elements.audio.seeking',timeout=30000)
    page.evaluate('(t)=>{seekTo(t);elements.video.pause();}',mid);page.wait_for_function('!elements.video.seeking&&!elements.audio.seeking&&elements.video.readyState>=2',timeout=90000);page.wait_for_timeout(200)
    cue=page.evaluate("()=>({audio:elements.audio.currentTime,video:elements.video.currentTime,cues:[...(elements.captionTrack.track.activeCues||[])].map(c=>c.text)})")
    if abs(cue['audio']-mid)<1:break
   assert abs(cue['audio']-mid)<1 and sentence['text'] in cue['cues'],(label,i,cue,sentence)
   filename=f'{label}-restoration-{i-20:02d}.png';page.locator('#tutorial-video').screenshot(path=str(out/filename));cases.append({'scene_index':i,'scene_text':text,'requested_audio_start':requested,'playing':playing,'caption_midpoint':mid,'native_caption':cue,'screenshot':filename,'screenshot_sha256':hashlib.sha256((out/filename).read_bytes()).hexdigest()});print('PASS',label,'scene',i,flush=True)
  records.append({'viewport':label,'size':[width,height],'device_scale_factor':3 if mobile else 1,'layout':layout,'start_method':started,'audio_sha256':actual,'cases':cases});ctx.close()
 browser.close()
assert not errors and not foreign,(errors,foreign)
result={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'site':site,'media_root':media,'scope':'Actual live player, five restoration scenes each at desktop and touch/mobile geometry; narration/video advancement, native caption cue, exact audio hash and <0.5 second synchronization. Scene seeks use production chapter seekTo handler.','browser':'installed Google Chrome, headless CPU rendering','human_or_native_listening_review':False,'passed_scenes':10,'viewports':records,'unexpected_requests':foreign,'javascript_errors':errors,'hosted_request_count':len(hosted)}
(out/'playback.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('PASS all10 live restoration playback cases')
