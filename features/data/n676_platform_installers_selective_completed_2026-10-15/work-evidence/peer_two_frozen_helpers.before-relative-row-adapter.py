from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import ast,datetime,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent
book=Path('/media/carruthers/mnt3/claude/ram_leases.md');label='codex-ws-04-frozen-helper-peer-'+str(os.getpid());token='owner_pid='+str(os.getpid())
def lease(add):
 with book.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
  if add:f.seek(0,2);f.write('\n'+label+' | two bounded04 source/delivery peers | 1 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on exit\n')
  else:f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
  f.flush();os.fsync(f.fileno())
def bind(p):b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def checked(b):p=Path(b['path']);a=bind(p);assert all(a[k]==b[k] for k in ['path','bytes','sha256']);return p
def read(p):return json.loads(p.read_text())
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT)
def save(name,d):p=W/name;assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(bind(p)))
lease(True)
try:
 helper=W/'prepare_verified_media_reuse.py';assert bind(helper)['sha256']=='64fbd1930dda301f43c2d748636715ad9ca36aa346417b13bc053b4120c11546';new=helper.read_text();ast.parse(new);oldpaths=list((W/'helper-review-originals').glob('prepare_verified_media_reuse.py*'));old=next(p for p in oldpaths if bind(p)['sha256']=='319354c6e5343d5b5c5b262ecac4fefc45a5086b94749001d41c65d519edc58c');before=old.read_text()
 insertion1="    browser = read(candidate / 'checks/candidate-browser-checks.json')\n    cases = browser['ready_playback_cases']\n    require(manifest['ready_lessons'] == manifest['routes'] == 84\n            and len(cases) == len({row['lesson'] for row in cases}) == 84\n            and all(row['passed'] is True for row in cases), 'Exact current84 unique passed routes required')\n"
 insertion2="    candidate_records = {row['path']: row for row in manifest['files']}\n"
 insertion3="        if row['path'].startswith('web/'):\n            current = candidate_records.get(row['path'])\n            require(current is not None and current['sha256'] == row['sha256']\n                    and current['bytes'] == row['bytes'], 'Checkpoint web file is not this actual candidate')\n"
 reduced=new
 for s in [insertion1,insertion2,insertion3]:assert reduced.count(s)==1;reduced=reduced.replace(s,'',1)
 assert reduced==before
 save('verified-media-reuse-helper-independent-source-peer.json',{'status':'pass','source_peer_passed':True,'blocking_findings':[],'helper':bind(helper),'preserved_original':bind(old),'prior_findings':bind(W/'verified-media-reuse-helper-independent-source-findings.json'),'checks':{'only_two_required_guard_areas_changed':True,'exact84ready_routes_unique_passed_browser_IDs':True,'every_checkpoint_web_row_matches_newcandidate_manifest_SHA_bytes':True,'owning_validator_hashes_actual_candidate_before_new_checks':True,'unchanged_original_readback_passthrough_exact_maps_pins_fresh_output':True},'limits':['Source/AST-only delta peer, helper not imported/executed.','No test/browser/media/network/publication operation or future acceptance.']})
 # Third closed handoff and one-list assembler addition.
 base=W/'source-handoff-pending-seek-followup';mp=base/'source-manifest.json';m=read(mp);assert bind(mp)['sha256']=='e92b35938a62224a732b93c025c9e7dc0ef673b8375ce727d03a2c83632af689';k6=m['source_commit'];k5=m['source_parent'];assert k6=='2427d57f17ad78d97cf81f18720c1cd9e0c7cc02' and git('rev-parse',k6+'^').decode().strip()==k5
 assert set(git('diff','--name-only',k5,k6).decode().splitlines())==set(m['exact_changed_paths']);assert git('rev-parse',k5+':spacr')==git('rev-parse',k6+':spacr')
 files=[p for p in base.rglob('*') if p.is_file()];assert not [p for p in base.rglob('*') if p.is_symlink()];assert len(files)==9 and len(m['files'])==m['listed_file_count']==8
 assert {str(p.relative_to(base)) for p in files}=={r['relative_path'] for r in m['files']}|{'source-manifest.json'}
 for r in m['files']:assert checked(r)==base/r['relative_path']
 for row in m['raw_source_files']:
  p=checked(row['artifact']);assert p.read_bytes()==git('show',k6+':'+row['relative_path']);assert git('rev-parse',k6+':'+row['relative_path']).decode().strip()==row['git_blob']
 assert checked(m['patch']).read_bytes()==git('format-patch','--full-index','--no-signature','-1','--stdout',k6)
 owner=read(checked(m['owning_tests']));peer=read(checked(m['independent_source_peer']));assert owner['tests']['passed']==226 and owner['tests']['terminal_exit_status']==0 and peer['status']=='pass';assert (base/'evidence/owning-tests.json').read_bytes()==checked(m['owning_tests']).read_bytes();assert (base/'evidence/independent-source-peer.json').read_bytes()==checked(m['independent_source_peer']).read_bytes();assert (base/'evidence/owning-tests.log').read_bytes()==checked(m['source_tests']['log']).read_bytes()
 previous=[]
 for b in m['previous_closed_source_manifests']:
  p=checked(b);prior=read(p)
  for row in prior['files']:checked(row)
  previous.append(bind(p))
 prep=read(W/'completed04-assembler-source-preparation-v4.json');a=checked(prep['assembler']);original=checked(prep['preserved_assembler']);assert bind(a)['sha256']=='682227451cb1f0ae77d0c383f2e7d295303f74535f81c875e002fd87b39f4dc8';ast.parse(a.read_text());assert a.read_text()==original.read_text().replace("'source-handoff-playback-followup','scene10-doctor-command-reference'","'source-handoff-playback-followup','source-handoff-pending-seek-followup','scene10-doctor-command-reference'",1)
 checked(prep['usage']);checked(prep['exact_source_delta'])
 save('pending-seek-handoff-assembler-independent-source-peer.json',{'status':'pass','source_peer_passed':True,'blocking_findings':[],'third_handoff_manifest':bind(mp),'assembler':bind(a),'preserved_assembler':bind(original),'preparation':bind(W/'completed04-assembler-source-preparation-v4.json'),'checked_files':8,'actual_handoff_files_with_manifest':9,'capture_source_commit':m['capture_source_commit'],'playback_source_parent':k5,'playback_source_commit':k6,'previous_closed_manifests_verified_unchanged':previous,'checks':{'exact_two_source_Git_blobs_and_direct_parent':True,'exact_format_patch_reconstruction':True,'all_manifest_sizes_SHA_and_closed_nonsymlink_file_set':True,'actual226_owning_log_and_independent_peer_copied_exact':True,'only_third_folder_literal_added_to_assembler':True},'limits':['Source/delivery readback only; no tests/browser/media/helper execution or source mutation.','Source handoff correctly keeps production matrix/publication acceptance pending.','No publication, remote readback, human listening or whole-fresh-capture claim.']})
finally:lease(False)
