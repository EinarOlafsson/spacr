import sys
from pathlib import Path
ROOT = Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path.insert(0, str(ROOT))
import datetime, fcntl, hashlib, json, os, subprocess
W = Path(__file__).resolve().parent
B = Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-23-r60-20261015/release-candidate-append-7ew0avek')
LEASE = Path('/media/carruthers/mnt3/claude/ram_leases.md')
ROW = f'codex-lesson02-video-assessment | bounded software samples | 2 GB | owner_pid={os.getpid()}\n'
def lease(add):
    with LEASE.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        t=f.read().replace(ROW,'');f.seek(0);f.write(t+(ROW if add else ''));f.truncate()
def binding(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def run(args):
    r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if r.returncode: raise RuntimeError(r.stderr[-2000:])
    return r
lease(True)
try:
    video=B/'media_host/02_install_spacr/video/02_install_spacr_silent.mp4'
    web=B/'media_host/02_install_spacr/web/02_install_spacr_silent.mp4'
    timing=B/'media_host/02_install_spacr/video/native-live-timings.json'
    canon=ROOT/'tools/tutorials/lessons/02_install_spacr.json'
    t=json.loads(timing.read_text());c=json.loads(canon.read_text())
    assert len(t['scenes'])==len(c['scenes'])==17
    assert [x['visual'] for x in t['scenes']]==[x['visual'] for x in c['scenes']]
    probe={}
    for name,p in [('master',video),('web',web)]:
        probe[name]=json.loads(run(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration,size','-of','json',str(p)]).stdout)
    (W/'video-probe.json').write_text(json.dumps(probe,indent=2)+'\n')
    samples=[]
    for i,s in enumerate(t['scenes'],1):
        lo,hi=s['speech_start'],s['scene_end']
        for pos,sec in [('start',lo+.4),('middle',(lo+hi)/2),('end',hi-.4)]:
            n=round(sec*30);samples.append({'scene':i,'visual':s['visual'],'position':pos,'frame':n,'seconds':n/30})
    frames=W/'video-samples';frames.mkdir(exist_ok=False)
    expr='+'.join(f'eq(n\\,{x["frame"]})' for x in samples)
    args=['/usr/bin/ffmpeg','-hide_banner','-nostdin','-v','warning','-hwaccel','none','-threads','2','-i',str(video),'-an','-vf',f'select={expr}','-vsync','0','-frames:v',str(len(samples)),str(frames/'%03d.png')]
    r=run(args);(W/'frame-extraction.log').write_text(r.stderr+'\nExit status: 0\n')
    files=sorted(frames.glob('*.png'));assert len(files)==len(samples)
    for x,p in zip(samples,files): x['file']=binding(p)
    from PIL import Image,ImageDraw
    for pos in ('start','middle','end'):
        sub=[x for x in samples if x['position']==pos]
        for part in range(3):
            subset=sub[part*6:(part+1)*6]
            sheet=Image.new('RGB',(1920,3*572),'#202020');d=ImageDraw.Draw(sheet)
            for j,x in enumerate(subset):
                with Image.open(x['file']['path']) as im:
                    im.thumbnail((960,540));sheet.paste(im,((j%2)*960,(j//2)*572+30))
                d.text(((j%2)*960+8,(j//2)*572+8),f'Scene{x["scene"]} {x["visual"]} {x["seconds"]:.3f}s',fill='white')
            sheet.save(W/f'{pos}-{part+1}.jpg',quality=92)
    manifest={'schema':1,'root_head':run(['git','-C',str(ROOT),'rev-parse','HEAD']).stdout.strip(),'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline_manifest':binding(B/'release-manifest.json'),'master':binding(video),'web':binding(web),'visual_timing':binding(timing),'canonical':binding(canon),'catalog':binding(B/'web/catalog/lessons_en.json'),'probe':probe,'samples':samples,'ffmpeg_argv':args,'extraction_log':binding(W/'frame-extraction.log'),'scope':'51frames, three per17canonical sections, one software sampling pass; no audio audition or full-motion watching claim'}
    (W/'video-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(binding(W/'video-evidence-manifest.json')))
finally: lease(False)
