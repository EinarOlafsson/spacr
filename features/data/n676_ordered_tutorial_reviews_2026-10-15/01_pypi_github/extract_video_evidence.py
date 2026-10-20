import sys
from pathlib import Path
ROOT = Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path.insert(0, str(ROOT))
import contextlib, datetime, fcntl, hashlib, json, os, subprocess, urllib.request

W = Path(__file__).resolve().parent
B = Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-23-r60-20261015/release-candidate-append-7ew0avek')
LEASE = Path('/media/carruthers/mnt3/claude/ram_leases.md')
LABEL = f'codex-lesson01-video-assessment | bounded software extraction | 2 GB | owner_pid={os.getpid()}\n'
def lease(add):
    with LEASE.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        text = f.read().replace(LABEL, '')
        f.seek(0); f.write(text + (LABEL if add else '')); f.truncate()
def binding(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}
def run(args):
    r = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode: raise RuntimeError(f'{args[0]} failed {r.returncode}: {r.stderr[-2000:]}')
    return r
lease(True)
try:
    video = B/'media_host/01_pypi_github/video/01_pypi_github_silent.mp4'
    webvideo = B/'media_host/01_pypi_github/web/01_pypi_github_silent.mp4'
    sidecar = B/'media_host/01_pypi_github/audio/en/af_heart.json'
    canonical = ROOT/'tools/tutorials/lessons/01_pypi_github.json'
    audio = json.loads(sidecar.read_text())
    lesson = json.loads(canonical.read_text())
    probe = json.loads(run(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration,size','-of','json',str(video)]).stdout)
    (W/'video-probe.json').write_text(json.dumps(probe,indent=2)+'\n')
    samples = []
    for i,s in enumerate(audio['scenes']):
        lo,hi = s['speech_start'],s['scene_end']
        for position,t in [('start',lo+.35),('middle',(lo+hi)/2),('end',hi-.35)]:
            frame = round(t*30)
            samples.append({'scene':i+1,'visual':lesson['scenes'][i]['visual'],'position':position,'requested_seconds':t,'frame':frame,'frame_seconds':frame/30})
    frames=W/'video-samples';frames.mkdir(exist_ok=False)
    expr='+'.join(f'eq(n\\,{x["frame"]})' for x in samples)
    args=['/usr/bin/ffmpeg','-hide_banner','-nostdin','-v','warning','-hwaccel','none','-threads','2','-i',str(video),'-an','-vf',f'select={expr}','-vsync','0','-frames:v',str(len(samples)),str(frames/'%03d.png')]
    result=run(args)
    (W/'frame-extraction.log').write_text(result.stderr+'\nExit status: 0\n')
    actual=sorted(frames.glob('*.png'))
    assert len(actual)==len(samples),(len(actual),len(samples))
    for x,p in zip(samples,actual): x['file']=binding(p)
    from PIL import Image, ImageDraw
    for group,subset in [('all-scenes-middle',[x for x in samples if x['position']=='middle']),('all-scenes-start',[x for x in samples if x['position']=='start']),('all-scenes-end',[x for x in samples if x['position']=='end'])]:
        sheet=Image.new('RGB',(1920,4*572),'#202020');d=ImageDraw.Draw(sheet)
        for j,x in enumerate(subset):
            with Image.open(x['file']['path']) as im:
                im.thumbnail((960,540)); sheet.paste(im,((j%2)*960,(j//2)*572+30))
            d.text(((j%2)*960+8,(j//2)*572+8),f'Scene {x["scene"]}: {x["visual"]} at {x["frame_seconds"]:.3f}s',fill='white')
        sheet.save(W/f'{group}.jpg',quality=92)
    api_results=[]
    for name,url in [('pypi','https://pypi.org/pypi/spacr/json'),('github-latest','https://api.github.com/repos/EinarOlafsson/spacr/releases/latest'),('conda-forge','https://api.anaconda.org/package/conda-forge/spacr')]:
        checked=datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'spaCR-tutorial-source-review/1.0','Cache-Control':'no-cache','Accept':'application/json'})
            with urllib.request.urlopen(request,timeout=25) as response:
                raw=response.read(16*1024*1024+1); assert len(raw)<=16*1024*1024
                final=response.url; status=response.status
            payload=json.loads(raw); out=W/f'primary-{name}.json';out.write_bytes(raw)
            if name=='pypi': facts={'version':payload['info']['version'],'project_urls':payload['info']['project_urls'],'requires_python':payload['info']['requires_python']}
            elif name=='github-latest': facts={k:payload.get(k) for k in ['tag_name','html_url','published_at','prerelease','draft']};facts['assets']=[{'name':x['name'],'bytes':x['size'],'url':x['browser_download_url']} for x in payload['assets']]
            else: facts={k:payload.get(k) for k in ['name','full_name','latest_version','public','home','dev_url','doc_url','versions','modified_at']}
            api_results.append({'url':url,'final_url':final,'checked_utc':checked,'status':status,'response':binding(out),'facts':facts})
        except Exception as e: api_results.append({'url':url,'checked_utc':checked,'error':str(e)})
    manifest={'schema':1,'root_head':run(['git','-C',str(ROOT),'rev-parse','HEAD']).stdout.strip(),'baseline_manifest':binding(B/'release-manifest.json'),'video':binding(video),'web_video':binding(webvideo),'canonical':binding(canonical),'reference_audio_sidecar_timing_only':binding(sidecar),'catalog':binding(B/'web/catalog/lessons_en.json'),'probe':probe,'samples':samples,'extraction_argv':args,'software_only':True,'primary_api_checks':api_results,'scope':'24 sampled frames, three per canonical narration scene; no audio audition or full-motion viewing claim'}
    (W/'video-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'samples':len(samples),'manifest':binding(W/'video-evidence-manifest.json'),'primary_api_facts':[{'url':x['url'],'facts':x.get('facts'),'error':x.get('error')} for x in api_results]},indent=2))
finally:
    lease(False)
