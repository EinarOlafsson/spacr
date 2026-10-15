from pathlib import Path
import sys
R=Path('/media/carruthers/mnt3/claude/spacr-lesson20-report-20261013');sys.path.insert(0,'/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
import datetime,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent;P=R/'features/data/n676_current_publication_selector_2026-10-15/receipt.json';O=W/'current-publication-selector-independent-readback.json'
def bind(p):b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def git(*a):return subprocess.check_output(['git',*a],cwd=R)
book=Path('/media/carruthers/mnt3/claude/ram_leases.md');label='codex-ws-selector-readback-'+str(os.getpid());token='owner_pid='+str(os.getpid())
def lease(add):
 with book.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
  if add:f.seek(0,2);f.write('\n'+label+' | completed02 selector bindings only | 1 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on exit\n')
  else:f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
  f.flush();os.fsync(f.fileno())
lease(True)
try:
 assert not O.exists();d=json.loads(P.read_text());assert bind(P)['sha256'].startswith('800acc64');assert git('show','HEAD:'+str(P.relative_to(R)))==P.read_bytes();bindings=[]
 for key in ['manifest','publication_receipt','checkpoint','hosted_browser_checks','current_index']:
  b=d[key];rel=Path(b['path']);assert not rel.is_absolute() and '..' not in rel.parts;p=R/rel;actual=bind(p);assert (actual['bytes'],actual['sha256'])==(b['bytes'],b['sha256']);assert git('show','HEAD:'+str(rel))==p.read_bytes();bindings.append({'role':key,**actual,'matches_current_committed_HEAD':True})
 manifest=json.loads((R/d['manifest']['path']).read_text());receipt=json.loads((R/d['publication_receipt']['path']).read_text());checkpoint=json.loads((R/d['checkpoint']['path']).read_text());browser=json.loads((R/d['hosted_browser_checks']['path']).read_text());index=(R/d['current_index']['path']).read_text()
 assert d['manifest']['sha256']=='98d88a909b3eeaf14f829297d965461e62a07dff7dfd25ce50d2c2bc15a25439';assert receipt['manifest_sha256']==checkpoint['manifest_sha256']==browser['manifest_sha256']==d['manifest']['sha256'];assert receipt['commit']==checkpoint['media_revision']['commit']==d['media_revision']=='b62bcc2d85b3c54162485f7e6116cf04c0bf0dc3';assert receipt['media_root']==browser['media_root']==d['media_root'] and d['media_root'] in index;assert receipt['branch']==d['branch'] and receipt['tag']==d['tag'];assert browser['index_sha256']==d['current_index']['sha256'] and browser['passed'] and len(browser['ready_playback_cases'])==84
 assert receipt['media_files']==5369 and receipt['media_bytes']==8679855371 and len(receipt['readback']['files_on_revision_not_in_candidate'])==2964;assert checkpoint['published'] is False and checkpoint['pages_tree_ready'] is True
 subprocess.run(['git','merge-base','--is-ancestor',d['publication_nightly_commit'],'HEAD'],cwd=R,check=True)
 historical=R/'tools/tutorials/release_candidate/publication-receipt.json';h=json.loads(historical.read_text());assert h['commit'].startswith('c34');test=R/'tests/test_tutorial_release_audit.py';txt=test.read_text();assert 'tools/tutorials/release_candidate' in txt
 result={'status':'pass','selector':bind(P),'report_HEAD':git('rev-parse','HEAD').decode().strip(),'publication_nightly_commit':d['publication_nightly_commit'],'bindings':bindings,'selected_completed_lesson':'02_install_spacr','manifest_sha256':d['manifest']['sha256'],'media_revision':d['media_revision'],'historical_coverage_read_only':d['verified_historical_coverage'],'checkpoint_published_false_preserved':True,'unpublished04_not_selected':True,'historical_candidate_receipt':bind(historical),'historical_candidate_media_revision':h['commit'],'test_source':bind(test),'test_selector_limitation':'Current test source still explicitly selects tools/tutorials/release_candidate in candidate/media-record expressions; historicalc34 is not current committed docs indexb62. This receipt does not alter or run tests.','limits':['Local source/committed-blob and metadata readback only.','Historical completed02 remote/browser results were read, not rerun; no current remote/live deployment or CI success asserted.','04 ongoing repair/matrix is not publication and is not selected.']}
 O.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(bind(O)))
finally:lease(False)
