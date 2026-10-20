from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import datetime,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent;O=W/'native-home-visual-samples';V=Path('/media/carruthers/mnt3/claude/n676_stages/host/i04-home-r1/captures/home/13_current_nightly_home.mp4');P=V.with_suffix('.pointer.json');C=V.with_suffix('.capture.json')
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
book=Path('/media/carruthers/mnt3/claude/ram_leases.md');label='codex-ws-04-native-samples-'+str(os.getpid());token='owner_pid='+str(os.getpid())
def lease(add):
 with book.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
  if add:f.seek(0,2);f.write('\n'+label+' | actual04Home bounded native visual sampling | 4 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on exit\n')
  else:f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
  f.flush();os.fsync(f.fileno())
lease(True)
try:
 assert not O.exists();O.mkdir();c=json.loads(C.read_text());assert len(c['frames'])==851 and c['full_decode_passed'] and bind(V)['sha256']==c['sha256']
 commands=[]
 def run(cmd):
  with (O/'ffmpeg.log').open('ab') as f:r=subprocess.run(cmd,stdout=f,stderr=f)
  commands.append({'command':cmd,'returncode':r.returncode});assert r.returncode==0
 base=['ffmpeg','-nostdin','-n','-v','error','-threads','2','-i',str(V)]
 run(base+['-vf',"select='not(mod(n,30))',scale=960:540",'-vsync','0','-an','-threads','2',str(O/'second-%02d.png')])
 run(['ffmpeg','-nostdin','-n','-v','error','-threads','2','-framerate','1','-i',str(O/'second-%02d.png'),'-vf','scale=480:270,tile=5x6','-frames:v','1','-threads','2',str(O/'contact-sheet.png')])
 events=json.loads(P.read_text());clicks=[r for r in events if r['event']=='release-left'];states=[('first',0),('last',850)]+[(f'after-tab-click-{i+1}',min(850,round((r['t']+.6)*30))) for i,r in enumerate(clicks)]
 samples=[]
 for name,n in states:
  out=O/(name+'.png');run(base+['-vf',f'select=eq(n\\,{n})','-vsync','0','-frames:v','1','-an','-threads','2',str(out)]);samples.append({'name':name,'frame_index':n,'seconds':n/30,'image':bind(out)})
 receipt={'status':'actual_sampling_completed_not_full_watch','video':bind(V),'capture_receipt':bind(C),'pointer':bind(P),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'frame_count':851,'sampled_seconds':list(range(29)),'contact_sheet':bind(O/'contact-sheet.png'),'second_samples':[bind(p) for p in sorted(O.glob('second-*.png'))],'click_release_events':clicks,'samples':samples,'commands':commands,'output_log':bind(O/'ffmpeg.log'),'limitations':['Bounded visual samples only, not full-watch/smoothness/semantic acceptance.','Click times are pointer-relative receipt times; tab names require actual image/source readback.','Existing capture full-decode receipt reused; sampling is not a repeated full-decode acceptance.']}
 (O/'sampling-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'output':str(O),'receipt':bind(O/'sampling-receipt.json')}),flush=True)
finally:lease(False)
