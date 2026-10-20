from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import datetime,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent
B=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-02-selective-r2-20261015/release-candidate-append-lxmwr53e')
master=B/'media_host/04_platform_installers/video/04_platform_installers_silent.mp4';timing=B/'media_host/04_platform_installers/video/native-live-timings.json'
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert bind(B/'release-manifest.json')['sha256']=='98d88a909b3eeaf14f829297d965461e62a07dff7dfd25ce50d2c2bc15a25439'
assert bind(master)['sha256']=='71fc4905d51039d5ccef673c0c7c6030f47c4b377ba670a899766d8b62a28e41'
out=W/'video-samples';assert not out.exists();out.mkdir()
book=Path('/media/carruthers/mnt3/claude/ram_leases.md');label='codex-ws-04-video-assessment-'+str(os.getpid());token='owner_pid='+str(os.getpid())
def lease(add):
 with book.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
  if add:f.seek(0,2);f.write('\n'+label+' | readonly04video42samples ffmpeg2threads | 4 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on ownerexit\n')
  else:f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
  f.flush();os.fsync(f.fileno())
lease(True)
try:
 scenes=json.loads(timing.read_text())['scenes'];samples=[]
 for i,row in enumerate(scenes,1):
  first=round(row['speech_start']*30);last=round(row['scene_end']*30)-1
  for role,n in [('start',first+min(3,last-first)),('middle',(first+last)//2),('end',last-min(3,last-first))]:
   p=out/f'{i:02}-{role}.png';cmd=['ffmpeg','-nostdin','-n','-v','error','-threads','2','-ss',f'{n/30:.9f}','-i',str(master),'-frames:v','1','-an','-c:v','png','-threads','2',str(p)]
   subprocess.run(cmd,check=True)
   samples.append({'scene':i,'visual':row['visual'],'position':role,'requested_master_frame':n,'requested_seconds':n/30,'image':bind(p),'command':cmd})
  print('sampled',i,row['visual'],flush=True)
 report={'schema':1,'lesson':'04_platform_installers','master':bind(master),'timing':bind(timing),'samples':samples,'scope':'42 requested near-start/middle/near-end frame samples,3per14sections; no full-watch/decode/audio/nativeacceptance claim.'}
 (W/'video-samples.json').write_text(json.dumps(report,indent=2)+'\n')
finally:lease(False)
