import ast
import difflib
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

repo = Path('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006')
scratch = Path('/mnt/wd4tb/scratch/ci-final-repairs-20261006')
out = repo / 'features/data/43_six_module_coverage_cpu_2026-10-06'
out.mkdir(parents=True, exist_ok=True)
old_sha = '392ca4d6a2319fd88c2d352b1a0b021682123d6b'
qt_sha = '7a6c2c6f33'
paths = ['spacr/convert.py','spacr/io.py','spacr/qt/recipes.py','spacr/qt/screens/make_masks.py','spacr/qt/timing.py','spacr/qt/widgets/ambient.py']
old = json.loads(Path('/mnt/wd4tb/scratch/ci-392-ratchet-20261007/coverage.json').read_text())['files']
scn = json.loads((scratch/'scn-portable-final-coverage.json').read_text())['files']
qt = json.loads((scratch/'qt-ratchet-final-coverage.json').read_text())['files']
dense = json.loads(subprocess.check_output(['git','show','HEAD:features/data/663_fungal_dense_cache_cpu_2026-10-07/current_coverage.json'],cwd=repo))
report = json.loads(Path('/mnt/wd4tb/scratch/ci-392-ratchet-20261007/module-coverage-ratchet.json').read_text())
baselines = {m['path']:m['baseline'] for m in report['modules']}
inputs = {'hosted392':old,'focused42':scn,'focused222':qt,'dense51':{'spacr/qt/widgets/ambient.py':dense}}
result = {}
for path in paths:
 current = (repo/path).read_bytes()
 records = [(old_sha,old[path])]
 if path in scn: records.append((None,scn[path]))
 if path in qt: records.append((qt_sha,qt[path]))
 if path.endswith('/ambient.py'): records.append((None,dense))
 universe = dense if path.endswith('/ambient.py') else records[-1][1]
 statements = set(universe['executed_lines']+universe['missing_lines'])
 branches = set(map(tuple,universe['executed_branches']+universe['missing_branches']))
 executed,arcs=set(),set()
 for sha,record in records:
  before=current if sha is None else subprocess.check_output(['git','show',sha+':'+path],cwd=repo)
  if before==current:
   executed.update(record['executed_lines']);arcs.update(map(tuple,record['executed_branches']));continue
  left,right=before.decode().splitlines(),current.decode().splitlines()
  mapping={}
  for a,b,n in difflib.SequenceMatcher(None,left,right,autojunk=False).get_matching_blocks():
   mapping.update({a+i+1:b+i+1 for i in range(n)})
  banned=None
  if path.endswith('/ambient.py'):
   old_class=next(n for n in ast.parse(before).body if isinstance(n,ast.ClassDef) and n.name=='_FungalGrowthEngine')
   new_class=next(n for n in ast.parse(current).body if isinstance(n,ast.ClassDef) and n.name=='_FungalGrowthEngine')
   assert left[:old_class.lineno-1]+left[old_class.end_lineno:]==right[:new_class.lineno-1]+right[new_class.end_lineno:]
   banned=(new_class.lineno,new_class.end_lineno)
  else:
   assert path.endswith('/make_masks.py')
   assert all(line in mapping for line in range(1,len(left)+1)), 'Only inserted guard lines may differ'
  def mapped(line):
   n=mapping.get(abs(line))
   if n is None or banned is not None and banned[0]<=n<=banned[1]: return None
   return n if line>0 else -n
  executed.update(n for line in record['executed_lines'] if (n:=mapped(line)) is not None)
  for arc in record['executed_branches']:
   new_arc=tuple(mapped(line) for line in arc)
   if all(line is not None for line in new_arc): arcs.add(new_arc)
 missing_lines=sorted(statements-executed);missing_arcs=sorted(branches-arcs)
 base=baselines[path]
 good=len(missing_lines)<=base['uncovered_statements'] and len(missing_arcs)<=base['uncovered_branches']
 result[path]={'source_sha256':hashlib.sha256(current).hexdigest(),'statements':len(statements),'branches':len(branches),'missing_lines':missing_lines,'missing_branches':missing_arcs,'original_statement_allowance':base['uncovered_statements'],'original_branch_allowance':base['uncovered_branches'],'within_original_allowances':good}
 assert good,(path,missing_lines,missing_arcs)
for name,data in inputs.items():
 selected={p:v for p,v in data.items() if p in paths}
 (out/(name+'.json.gz')).write_bytes(gzip.compress((json.dumps({'files':selected},sort_keys=True)+'\n').encode(),mtime=0))
for name in ('scn-portable-final.log','qt-ratchet-final.log'):
 (out/(name+'.gz')).write_bytes(gzip.compress((scratch/name).read_bytes(),mtime=0))
receipt={'scope':'Conservative source-bound union of an old hosted aggregate and bounded current behavior tests; not a fresh complete-suite, serial Qt, or hosted-success verdict. Old ambient coverage is inherited ONLY outside the byte-identical fungal-class exterior. Old Make Masks coverage is mapped across five inserted guard lines.','hosted_source':old_sha,'qt222_source':subprocess.check_output(['git','rev-parse',qt_sha],cwd=repo,text=True).strip(),'scn_passed':42,'qt_passed':222,'dense_passed':51,'CUDA_VISIBLE_DEVICES':'','QT_QPA_PLATFORM':'offscreen','hard_memory_cap':'4G','modules':result}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'README.md').write_text('# Six numerical coverage repairs\n\nAll six regressions in the complete hosted392 aggregate fit their unchanged original allowances in this conservative source-bound union. The archive retains the input module records and the full 42-case CPU and 222-case Qt integration logs. The current dense51 input and its original raw run/proofs are also archived in `../663_fungal_dense_cache_cpu_2026-10-07/`. Source, mapping restrictions, counts and unresolved scope are explicit in receipt.json. No ceiling, exclusion or failed-shard prerequisite was changed. This does not establish fresh hosted or full serial acceptance.\n')
(out/'archive_six_modules.py').write_bytes(Path(__file__).read_bytes())
manifest={'files':{}}
for p in sorted(out.iterdir()):
 if p.name!='manifest.json':manifest['files'][p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for p,m in result.items():print(p,len(m['missing_lines']),len(m['missing_branches']),'PASS')
