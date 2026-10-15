from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import ast,datetime,difflib,fcntl,hashlib,json,os,subprocess
W=Path(__file__).resolve().parent;D=W/'docs-media-selector-proposal';R=Path('/media/carruthers/mnt3/claude/spacr-lesson20-report-20261013');O=W/'docs-media-selector-proposal-independent-source-peer.json'
def bind(p):b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
book=Path('/media/carruthers/mnt3/claude/ram_leases.md');label='codex-ws-docs-selector-peer-'+str(os.getpid());token='owner_pid='+str(os.getpid())
def lease(add):
 with book.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
  if add:f.seek(0,2);f.write('\n'+label+' | exacttwo-expression docs selector sourcepeer | 1 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on exit\n')
  else:f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
  f.flush();os.fsync(f.fileno())
lease(True)
try:
 assert not O.exists();p=json.loads((D/'proposal.json').read_text());assert bind(D/'proposal.json')['sha256']=='e58b48a8b474f82f230b55d0ef1bfba06dd8513963e2e67daa3682cac08de674'
 for key in ['preimage','proposed','patch','selector_receipt','owner_relay']:
  b=p[key];assert bind(Path(b['path']))==b
 old=(D/'test_docs_media_budget.py.preimage').read_text();new=Path(p['proposed']['path']).read_text();target=R/p['target'];assert target.read_text()==old;assert subprocess.check_output(['git','show','HEAD:'+p['target']],cwd=R)==target.read_bytes()
 replacements=[('checkpoint = REPO_ROOT / "tools" / "tutorials" / "release_candidate"','checkpoint = REPO_ROOT / "features/data/n676_install_selective_completed_2026-10-15/export/checkpoint"'),('REPO_ROOT / "tools/tutorials/release_candidate",','REPO_ROOT / "features/data/n676_install_selective_completed_2026-10-15/export/checkpoint",')]
 expected=old
 for a,b in replacements:assert expected.count(a)==1;expected=expected.replace(a,b,1)
 assert new==expected
 a,b=ast.parse(old),ast.parse(new);assert [ast.dump(n,include_attributes=False) for n in ast.walk(a) if isinstance(n,ast.Assert)]==[ast.dump(n,include_attributes=False) for n in ast.walk(b) if isinstance(n,ast.Assert)]
 assert ''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+p['target'],tofile='b/'+p['target']))==Path(p['patch']['path']).read_text()
 for node in p['selected_nodes']:assert node.split('::')[-1] in {n.name for n in b.body if isinstance(n,ast.FunctionDef)}
 selector=json.loads(Path(p['selector_receipt']['path']).read_text());assert p['current_manifest_sha256']==selector['manifest']['sha256'] and p['current_media_revision']==selector['media_revision'];assert selector['checkpoint_root']==p['current_checkpoint_root']
 for entry in p['current_asset_bindings'].values():assert bind(Path(entry['path']))==entry
 result={'status':'pass','source_peer_passed':True,'blocking_findings':[],'proposal':bind(D/'proposal.json'),'preimage':bind(target),'proposed':bind(Path(p['proposed']['path'])),'patch':bind(Path(p['patch']['path'])),'report_current_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'checks':{'current_committed_test_preimage_exact':True,'only_two_explicit_checkpoint_selector_expressions_changed':True,'all_assert_AST_nodes_identical':True,'exact_patch_reconstruction':True,'both_selected_test_nodes_exist':True,'current02_selector_b62_manifest98d88_matches':True,'actual_test_source_unchanged':target.read_text()==old},'selected_nodes':p['selected_nodes'],'limits':['Source-only portable proposal, not applied; actual tests/CI/browser/remote operations not executed.','Applies to exact preimage only; selects completed02 currently published data, not unpublished04.','Historical tools/tutorials/release_candidate files remain untouched; no assertions weakened.']}
 O.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(bind(O)))
finally:lease(False)
