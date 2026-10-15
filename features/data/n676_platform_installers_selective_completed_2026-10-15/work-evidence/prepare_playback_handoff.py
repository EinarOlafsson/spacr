from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import datetime,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent;O=W/'source-handoff-playback-followup';K4='b929b7dc53465132322c3ca0b866f567de773b9f';K5='253f5814bc2add67d01032948271b8aa1b06aa00';paths=['tools/tutorials/authoring/web/app_v2.js','tools/tutorials/tests/test_player_preserve_visual_intervals.py']
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT)
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(p):b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':sha(b)}
book=Path('/media/carruthers/mnt3/claude/ram_leases.md');label='codex-ws-04-playback-handoff-'+str(os.getpid());token='owner_pid='+str(os.getpid())
def lease(add):
 with book.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
  if add:f.seek(0,2);f.write('\n'+label+' | exact04 K4-to-K5 source evidence handoff | 1 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on exit\n')
  else:f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
  f.flush();os.fsync(f.fileno())
lease(True)
try:
 assert not O.exists();assert git('rev-parse','HEAD').decode().strip()==K5;assert git('rev-parse',K5+'^').decode().strip()==K4;assert set(git('diff','--name-only',K4,K5).decode().splitlines())==set(paths)
 app=git('rev-parse',K4+':spacr').decode().strip();assert git('rev-parse',K5+':spacr').decode().strip()==app
 old={str(p.relative_to(W/'source-handoff')):bind(p) for p in (W/'source-handoff').rglob('*') if p.is_file()};assert old
 O.mkdir();e=json.loads((W/'retained-static-clock-owning-tests.json').read_text());peer=json.loads((W/'player-static-interval-independent-source-peer.json').read_text());assert e['tests']['passed']==225 and e['tests']['terminal_exit_status']==0 and peer['status']=='pass'
 for name,data in [('0001-selective04-static-playback-clock-followup.patch',git('format-patch','--stdout',K4+'..'+K5)),('commit-log.txt',git('show','--no-patch','--format=fuller',K5))]:(O/name).write_bytes(data)
 for rel in paths:
  b=git('show',K5+':'+rel);assert b==(ROOT/rel).read_bytes();matches=[r for r in e['files'] if r['path']==str(ROOT/rel)];assert len(matches)==1 and matches[0]['sha256']==sha(b) and matches[0]['bytes']==len(b)
  target=O/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b);assert target.read_bytes()==git('show',K5+':'+rel)
 log=Path(e['tests']['log']['path']);assert bind(log)==e['tests']['log'];assert peer['owning_evidence']==bind(W/'retained-static-clock-owning-tests.json')
 for src,name in [(W/'retained-static-clock-owning-tests.json','owning-tests.json'),(log,'owning-tests.log'),(W/'player-static-interval-independent-source-peer.json','independent-source-peer.json')]:
  dst=O/'evidence'/name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(src.read_bytes());assert dst.read_bytes()==src.read_bytes()
 (O/'README.txt').write_text('Exact tutorial04 playback follow-up only. Native capture remains K4 b929b7dc; final player is K5 253f5814, its direct child. Only player JS and focused test differ; app tree, capture tools, gates, English, reviews, audio and encoded video are unchanged by this commit. Existing 225-test owning receipt and independent source peer are copied unchanged. Actual production matrix rerun is pending, not accepted here. Original closed K4 source-handoff is untouched. No execution of tests/browser/media/provider operations in this packaging step.\n')
 assert old=={str(p.relative_to(W/'source-handoff')):bind(p) for p in (W/'source-handoff').rglob('*') if p.is_file()}
 result={'schema':1,'status':'source_handoff_prepared_actual_matrix_acceptance_pending','capture_source_commit':K4,'playback_source_commit':K5,'direct_parent_verified':True,'exact_changed_paths':paths,'app_tree_unchanged':app,'source_tests':{'passed':225,'terminal_exit_status':0},'original_closed_K4_handoff_preserved':old,'files':[dict(relative_path=str(p.relative_to(O)),**bind(p)) for p in sorted(O.rglob('*')) if p.is_file()],'limits':['Existing source-test/peer evidence only; production matrix r2 is active and no acceptance inferred.','No capture, reencode, audio changes, source mutation, commit/push or test/browser execution.']}
 m=O/'source-manifest.json';m.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(bind(m)),flush=True)
finally:lease(False)
