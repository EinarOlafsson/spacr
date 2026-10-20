#!/usr/bin/env python3
"""Root-run four exact final-master frame samples; not a full decode acceptance."""
from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import datetime,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent
STAGE=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-02-selective-r2-20261015')
FOLDER=STAGE/'production/02_install_spacr'
OUTPUT=W/'final-boundary-samples-r2'
FRAMES=[4419,4420,5111,5112]
def binding(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(p.read_text())
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==os.environ['SOURCE_COMMIT']
 assert (W/'scene7-overlay-r2.log').read_text().rstrip().endswith('Terminal exit status: 0'), 'Actual owning overlay terminal0 required'
 receipt=read(FOLDER/'retained-visual-refresh.json');master=FOLDER/'video/02_install_spacr_silent.mp4';timing=FOLDER/'video/native-live-timings.json'
 assert receipt['new_master']==binding(master)
 times=read(timing);assert round(times['total_duration']*30)==12901 and times.get('complete_native_clips') is not True
 assert not OUTPUT.exists(), 'Preserve any previous samples'
 label='codex-ws-02-selective-boundary-'+str(os.getpid());token='owner_pid='+str(os.getpid())
 book=Path('/media/carruthers/mnt3/claude/ram_leases.md')
 def lease(add):
  with book.open('r+') as f:
   fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
   if add:
    assert not any(r.startswith(label+' | ') for r in rows)
    f.seek(0,2);f.write('\n'+label+' | four exact final-master boundary PNG samples; ffmpeg2threads | 4 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on exact owner exit\n')
   else:
    f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
   f.flush();os.fsync(f.fileno())
 lease(True)
 try:
  OUTPUT.mkdir()
  command=['ffmpeg','-nostdin','-n','-v','error','-threads','2','-i',str(master),'-an','-vf','select='+ '+'.join('eq(n\\,'+str(n)+')' for n in FRAMES),'-vsync','0','-frames:v','4','-c:v','png','-threads','2','-filter_threads','2',str(OUTPUT/'sample-%02d.png')]
  log=OUTPUT/'ffmpeg.log'
  with log.open('xb') as f:rc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
  (OUTPUT/'command.json').write_text(json.dumps({'command':command,'returncode':rc,'scope':'Exact ordinal frame extraction only; no full decode or image-content acceptance'},indent=2)+'\n')
  assert rc==0
  assert sorted(p.name for p in OUTPUT.glob('sample-*.png'))==['sample-01.png','sample-02.png','sample-03.png','sample-04.png']
  samples=[]
  for i,n in enumerate(FRAMES,1):
   p=OUTPUT/f'frame-{n:05d}.png';(OUTPUT/f'sample-{i:02d}.png').rename(p)
   samples.append({'master_frame_zero_based':n,'time_seconds':n/30,'role':{4419:'last original frame before scene7',4420:'first replacement frame',5111:'last replacement frame',5112:'first original frame after scene7'}[n],'image':binding(p)})
  report={'schema':1,'lesson':'02_install_spacr','source_commit':os.environ['SOURCE_COMMIT'],'master':binding(master),'timing':binding(timing),'overlay_receipt':binding(FOLDER/'retained-visual-refresh.json'),'samples':samples,'command_log':binding(log),'scope':'Four actual ordinal frames only; root visual review pending. Lossy inherited frames are not claimed pixel-identical; no human/audio/whole-video acceptance.'}
  (OUTPUT/'samples.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report,indent=2))
 finally:lease(False)
if __name__=='__main__':main()
