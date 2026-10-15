from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import datetime,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent;O=W/'final-boundary-samples';P=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-04-selective-20261015/production/04_platform_installers');M=P/'video/04_platform_installers_silent.mp4';FRAMES=[5110,5111,5961,5962,6400,6840,6841,10017]
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
book=Path('/media/carruthers/mnt3/claude/ram_leases.md');label='codex-ws-04-final-boundaries-'+str(os.getpid());token='owner_pid='+str(os.getpid())
def lease(add):
 with book.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
  if add:f.seek(0,2);f.write('\n'+label+' | actual04 composite eight boundary samples | 4 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on exit\n')
  else:f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
  f.flush();os.fsync(f.fileno())
lease(True)
try:
 assert not O.exists();O.mkdir();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert source=='b929b7dc53465132322c3ca0b866f567de773b9f'
 receipt=json.loads((P/'retained-visual-refresh.json').read_text());master=bind(M);assert receipt['new_master']==master
 assert receipt['replacements'][0]['frame_count']==851
 expr='+'.join('eq(n\\,'+str(n)+')' for n in FRAMES)
 cmd=['ffmpeg','-nostdin','-n','-v','error','-threads','2','-i',str(M),'-vf','select='+expr,'-vsync','0','-frames:v','8','-an','-threads','2','-filter_threads','1',str(O/'sample-%02d.png')]
 with (O/'ffmpeg.log').open('xb') as f:r=subprocess.run(cmd,stdout=f,stderr=f);f.write(('\nTerminal exit status: '+str(r.returncode)+'\n').encode())
 assert r.returncode==0
 assert len(list(O.glob('sample-*.png')))==8
 samples=[]
 for i,n in enumerate(FRAMES,1):
  old=O/f'sample-{i:02d}.png';out=O/f'frame-{n:05d}.png';old.rename(out);samples.append({'frame_index':n,'seconds':n/30,'image':bind(out)})
 assert bind(M)==master
 out={'status':'actual_boundary_samples_completed_not_visual_acceptance','source_commit':source,'master':master,'retained_visual_refresh':bind(P/'retained-visual-refresh.json'),'expected_master_frames':10018,'sampling_command':cmd,'returncode':0,'output_log':bind(O/'ffmpeg.log'),'samples':samples,'scope':'Eight actual decoded master frames: before/start/end native Home, start/mid/end doctor card, first preserved tail and final EOF frame. Not whole-watch/continuous smoothness/full-decode acceptance.'}
 q=O/'sampling-receipt.json';q.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(bind(q)),flush=True)
finally:lease(False)
