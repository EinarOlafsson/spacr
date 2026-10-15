from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import collections,datetime,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent;B=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-02-selective-r2-20261015/release-candidate-append-lxmwr53e');M=B/'media_host/04_platform_installers/video/04_platform_installers_silent.mp4'
raw=Path('/media/carruthers/mnt3/claude/n676_stages/host/r23-05-preferences/captures/home');events=json.loads((raw/'09_performance.pointer.json').read_text());assert not [x for x in events if 'scroll' in x['event'] or 'wheel' in x['event']]
source=subprocess.check_output(['git','show','0cde90ced82641b8caf831cdac81dc0c3911900d:tools/tutorials/capture_home_live.py'],cwd=ROOT,text=True);part=source[source.index('    def steps(dialog):',source.index('def record_preferences_live')):];part=part[:part.index('        capture.end()')+len('        capture.end()')];assert 'Spaceout' not in part and 'spaceout' not in part and 'scroll' not in part
book=Path('/media/carruthers/mnt3/claude/ram_leases.md');label='codex-ws-04-performance-states-'+str(os.getpid());token='owner_pid='+str(os.getpid())
def lease(add):
 with book.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
  if add:f.seek(0,2);f.write('\n'+label+' | bounded04Performance actual5UIstates | 4 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on ownerexit\n')
  else:f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
  f.flush();os.fsync(f.fileno())
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
lease(True)
try:
 out=W/'performance-distinct-states';assert not out.exists();out.mkdir();rows=[]
 for name,offset in [('menu',3.9),('dialog',5.5),('performance',7.0),('profile_list',9.9),('profile_kept',11.2)]:
  q=out/(name+'.png');cmd=['ffmpeg','-nostdin','-n','-v','error','-threads','2','-ss',str(143.1+offset),'-i',str(M),'-frames:v','1','-an','-c:v','png','-threads','2',str(q)];subprocess.run(cmd,check=True);rows.append({'name':name,'requested_master_seconds':143.1+offset,'image':bind(q)})
 result={'schema':1,'scope':'All5pointer-click-driven UIstate transitions plus original3samples; no full818frame content/wholewatch guarantee','pointer':bind(raw/'09_performance.pointer.json'),'host':bind(raw/'09_performance.host.json'),'pointer_event_counts':dict(collections.Counter(x['event'] for x in events)),'scroll_or_wheel_events':0,'source_commit':'0cde90ced82641b8caf831cdac81dc0c3911900d','source_function_excerpt':part,'source_function_excerpt_sha256':hashlib.sha256(part.encode()).hexdigest(),'samples':rows,'spaceout_visual_verdict':'pending actual image inspection'}
 (W/'performance-state-source-scan.json').write_text(json.dumps(result,indent=2)+'\n')
finally:lease(False)
