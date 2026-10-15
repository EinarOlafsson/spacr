import ast,gzip,hashlib,json,pathlib,subprocess
root=pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015')
base='288f17d7a2^';final='bf9b5dd0ba';inputs='c9e3dfadf3';native='57b4e08773'
def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root)
def sha(data):return hashlib.sha256(data).hexdigest()
def defs(data):
 out={}
 def visit(body,prefix=''):
  for n in body:
   if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
    key=prefix+n.name;out[key]=n
    if isinstance(n,ast.ClassDef):visit(n.body,key+'.')
 visit(ast.parse(data).body);return out
files=['spacr/qt/widgets/figure_queue.py','spacr/qt/screens/app_screen.py']
proof={'source_commit':subprocess.check_output(['git','rev-parse',final],cwd=root,text=True).strip(),'source_files':{},'existing_definition_changes':{},'artifacts':{}}
for path in files:
 a,b=defs(blob(base,path)),defs(blob(final,path))
 changed=[k for k in a if k in b and ast.dump(a[k],include_attributes=False)!=ast.dump(b[k],include_attributes=False)]
 expected=['FigureQueue'] if path.endswith('figure_queue.py') else ['AppScreen','AppScreen._build_runtime_panel']
 assert changed==expected,(path,changed)
 proof['existing_definition_changes'][path]=changed
 proof['source_files'][path]=sha(blob(final,path))
queue=files[0];current=blob(final,queue).decode();old=blob(native,queue).decode()
for name in ['_ClearFiguresLabel','_AccessibleClearFigures','_clear_figures_accessible_factory']:
 a=ast.get_source_segment(old,defs(old)[name]);b=ast.get_source_segment(current,defs(current)[name]);assert a==b,name
proof['clear_adapter_source_segments_equal_native_57b']=True
grid='spacr/qt/widgets/figure_grid_view.py'
assert blob(base,grid)==blob(final,grid)
proof['grid_untouched_by_change']=True
proof['grid_branch_hash']=sha(blob(final,grid));proof['preserve_coordinator_grid_native_57b_hash']=sha(blob(native,grid))
receiptpath='features/data/180_session_figure_carry_2026-10-15/receipt.json'
receipt=json.loads(blob('288f17d7a2',receiptpath))
for name,h in receipt['artifact_sha256'].items():
 path=receiptpath.rsplit('/',1)[0]+'/'+name;data=blob('288f17d7a2',path);assert sha(data)==h,path;proof['artifacts'][path]=h
for name,needle in [('final.log.gz','18 passed in 15.82s'),('pdf-after.log.gz','3 passed, 13 deselected in 3.15s')]:
 path=receiptpath.rsplit('/',1)[0]+'/'+name;data=blob(final,path);log=gzip.decompress(data).decode();assert needle in log and 'Exit status: 0' in log;proof['artifacts'][path]=sha(data)
for name in ['before.log.gz','after.log.gz','pdf-before.log.gz','pdf-followup.json']:
 path=receiptpath.rsplit('/',1)[0]+'/'+name;proof['artifacts'][path]=sha(blob(final,path))
test='tests/qt/test_figure_queue_workspace_180.py';follow=json.loads(blob(final,receiptpath.rsplit('/',1)[0]+'/pdf-followup.json'))
assert sha(blob(final,test))==follow['tests_sha256'];proof['test_sha256']=sha(blob(final,test))
package=json.loads(blob(inputs,'features/data/180_session_figure_inputs_2026-10-15/receipt.json'))
for path,h in package['source_files'].items():assert proof['source_files'][path]==h
for path,h in package['input_files'].items():assert sha(blob(inputs,path))==h
paths=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',inputs],cwd=root,text=True).splitlines();assert not any(p.startswith('spacr/') for p in paths)
proof['input_only_commit_verified']=inputs;proof['input_files_verified']=len(package['input_files'])
proof['limits']='Read-only source/AST/evidence binding; no runtime imports, tests, native probe, collector, generation or translation semantic re-review.'
path=pathlib.Path('/mnt/wd4tb/scratch/home-b-n180-figure-carry-peer-20261015/proof.json');path.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
