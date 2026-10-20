from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import datetime,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent;S=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-04-selective-20261015');B=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-02-selective-r2-20261015/release-candidate-append-lxmwr53e');L='04_platform_installers';P=S/'production'/L;OUT=W/'retained-components-independent-readback.json';ORIG=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-04-r23/production')/L
book=Path('/media/carruthers/mnt3/claude/ram_leases.md');label='codex-ws-04-retained-components-'+str(os.getpid());token='owner_pid='+str(os.getpid())
def lease(add):
 with book.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
  if add:f.seek(0,2);f.write('\n'+label+' | actual04 sealed retained-component readback only | 1 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on exit\n')
  else:f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
  f.flush();os.fsync(f.fileno())
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(p.read_text())
lease(True)
try:
 assert not OUT.exists();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert source=='b929b7dc53465132322c3ca0b866f567de773b9f'
 manifest=bind(B/'release-manifest.json');assert manifest['sha256']=='98d88a909b3eeaf14f829297d965461e62a07dff7dfd25ce50d2c2bc15a25439'
 records={r['path']:r for r in read(B/'release-manifest.json')['files']};audio=[]
 for f in sorted((P/'audio').rglob('*')):
  if f.is_file():
   rel=f.relative_to(P);assert f.suffix in ['.json','.m4a'];r=records['media_host/'+L+'/'+rel.as_posix()];b=bind(f);assert (b['bytes'],b['sha256'])==(r['bytes'],r['sha256']);audio.append({'staged':b,'baseline_manifest_path':r['path']})
 assert len(audio)==100 and sum(r['staged']['path'].endswith('.m4a') for r in audio)==50
 catalogs=[]
 for f in sorted((S/'catalog').glob('*.json')):
  old=B/'web/catalog'/f.name;assert f.read_bytes()==old.read_bytes();catalogs.append({'staged':bind(f),'baseline':bind(old),'entire_catalog_bytes_equal':True})
 assert len(catalogs)==14
 english=P/'lesson.en.json';canon=ROOT/'tools/tutorials/lessons'/f'{L}.json';assert english.read_bytes()==canon.read_bytes()==(ORIG/'lesson.en.json').read_bytes()
 reviews=[]
 for f in sorted(P.glob('review.*.json')):
  locale=f.name[len('review.'):-5];r=ROOT/'tools/tutorials/lessons/reviews'/f'{L}.{locale}.json';assert f.read_bytes()==r.read_bytes()==(ORIG/f.name).read_bytes();reviews.append({'staged':bind(f),'canonical_review':bind(r)})
 assert len(reviews)==13
 spec=read(P/'scenes.json');old=read(ORIG/'scenes.json');changed=[i+1 for i,(a,b) in enumerate(zip(old['scenes'],spec['scenes'])) if a!=b];assert len(spec['scenes'])==14 and changed==[9,10]
 prep=read(P/'selective-stage-preparation.json');assert prep['source_commit']==source and prep['native_frames']==851
 assert spec['scenes'][8]['clip']['duration']==851/30 and spec['scenes'][8]['clip']['sha256']==prep['replacements'][0]['new_video_sha256'];assert [r['scene'] for r in prep['replacements']]==[9,10]
 assert prep['replacements'][1]['kind']=='generated_command_reference_not_terminal_output'
 assert spec['capture_source']['completed_capture'] is False and spec['capture_source']['frame_pixels_modified'] is True
 result={'status':'pass','source_commit':source,'stage':str(S),'baseline_manifest':manifest,'audio_and_sidecars':audio,'counts':{'audio_tracks':50,'audio_and_sidecar_files':100,'catalogs':14,'reviews':13,'scenes':14},'catalogs':catalogs,'english':{'staged':bind(english),'canonical':bind(canon)},'reviews':reviews,'specification':bind(P/'scenes.json'),'stage_preparation':bind(P/'selective-stage-preparation.json'),'changed_visual_scenes':changed,'native_home_frames':851,'native_home_seconds':851/30,'doctor_reference_scene':10,'limitations':['Master, render log, final timing, poster and web rendition excluded because render is active.','No decode, audio listening, browser, synthesis, source/stage mutation or publication acceptance.','Entire14 catalog bytes retained, stronger than lesson-object-only equality; review prose/metadata exact original and ROOT.','Static readback does not replace protected final mixed-visual validator or actual final-video review.']}
 OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(bind(OUT)))
finally:lease(False)
