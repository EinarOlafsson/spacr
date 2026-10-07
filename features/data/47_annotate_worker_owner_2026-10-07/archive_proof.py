import ast
import difflib
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

repo=Path('/mnt/wd4tb/spacr-worktrees/codex-mask-load-drain-20261007')
scratch=Path('/mnt/wd4tb/scratch/annotate-worker-drain-20261007')
out=repo/'features/data/47_annotate_worker_owner_2026-10-07'
out.mkdir(parents=True,exist_ok=True)
path='spacr/qt/screens/annotate.py'
old=(scratch/'before_annotate.py').read_bytes()
current=(repo/path).read_bytes()
expected=old
for kind in ('similar','retrain','suggest'):
 needle=('            if stopped:\n                _retire('+kind+')\n').encode()
 assert expected.count(needle)==1
 expected=expected.replace(needle,needle+('            else:\n                '+kind+'.setParent(None)\n').encode())
assert current==expected
hosted_sha='e1f54c80650f60942ba6c98d4e999b36a3fa2846'
assert subprocess.check_output(['git','show',hosted_sha+':'+path],cwd=repo)==old
hosted_path=Path('/mnt/wd4tb/scratch/ci-e1f-ratchet-20261007/coverage.json')
hosted=json.loads(hosted_path.read_text())['files'][path]
focused=json.loads((scratch/'coverage.json').read_text())['files'][path]
left,right=old.decode().splitlines(),current.decode().splitlines()
mapping={a+i+1:b+i+1 for a,b,size in difflib.SequenceMatcher(None,left,right,autojunk=False).get_matching_blocks() for i in range(size)}
assert len(mapping)==len(left)
def mapped(n):return mapping[abs(n)]*(1 if n>0 else -1)
lines=set(focused['executed_lines']+focused['missing_lines'])
arcs=set(map(tuple,focused['executed_branches']+focused['missing_branches']))
old_lines={mapped(n) for n in hosted['executed_lines']+hosted['missing_lines']}
old_arcs={tuple(mapped(n) for n in edge) for edge in hosted['executed_branches']+hosted['missing_branches']}
assert lines==old_lines|{7666,7681,7699}
discarded=sorted(old_arcs-arcs)
assert discarded==[(7663,7667),(7678,7682),(7696,7700)]
new_arcs={(7663,7666),(7678,7681),(7696,7699)}
assert arcs==(old_arcs-set(discarded))|new_arcs
assert {7666,7681,7699}<=set(focused['executed_lines'])
assert new_arcs|{(7663,7664),(7678,7679),(7696,7697)}<=set(map(tuple,focused['executed_branches']))
executed={mapped(n) for n in hosted['executed_lines']}|set(focused['executed_lines'])
taken=({tuple(mapped(n) for n in edge) for edge in hosted['executed_branches']} & arcs)|set(map(tuple,focused['executed_branches']))
missing_lines,missing_arcs=sorted(lines-executed),sorted(arcs-taken)
baseline_path=repo/'tools/coverage_baseline.json'
allowance=json.loads(baseline_path.read_text())['modules'][path]
assert len(missing_lines)<=allowance['uncovered_statements'] and len(missing_arcs)<=allowance['uncovered_branches']
def docs(source):
 return [(type(n).__name__,n.name,ast.get_docstring(n)) for n in ast.walk(ast.parse(source)) if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
assert docs(old)==docs(current)
def ruff(source):
 result=subprocess.run(['/home/olafsson/.local/bin/ruff','check','--output-format','json','--stdin-filename',path,'-'],cwd=repo,input=source,capture_output=True)
 assert result.returncode in (0,1)
 return json.loads(result.stdout)
a,b=ruff(old),ruff(current)
assert len(a)==len(b)
for previous,now in zip(a,b):
 assert (previous['code'],previous['message'])==(now['code'],now['message'])
 for field in ('location','end_location'):
  assert previous[field]['column']==now[field]['column'] and mapping[previous[field]['row']]==now[field]['row']
for label,record in [('focused',focused),('hosted_e1',hosted)]:
 (out/(label+'.json.gz')).write_bytes(gzip.compress(json.dumps({'files':{path:record}},sort_keys=True).encode(),mtime=0))
for name in ('native_probe.py','run_probe.py','before.json','before.log','after.json','after.log','focused-tests.log'):
 shutil.copy2(scratch/name,out/name)
(out/'before_annotate.py.gz').write_bytes(gzip.compress(old,mtime=0))
(out/'after_annotate.py.gz').write_bytes(gzip.compress(current,mtime=0))
shutil.copy2(repo/'tests/qt/test_annotate_worker_owner_shutdown.py',out/'test_annotate_worker_owner_shutdown.py')
shutil.copy2(__file__,out/'archive_proof.py')
receipt={'source_parent':'0324166b59 (Annotate byte-identical through mask-loader branch)',
 'source_commit':'f2248bfe06','before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(current).hexdigest(),
 'test_sha256':hashlib.sha256((repo/'tests/qt/test_annotate_worker_owner_shutdown.py').read_bytes()).hexdigest(),
 'hosted_source':hosted_sha,'original_hosted_report_sha256':hashlib.sha256(hosted_path.read_bytes()).hexdigest(),
 'baseline_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),'original_allowance':allowance,
 'runtime_sources':{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in ('spacr/qt/bridge.py','tests/conftest.py','tests/qt/conftest.py')},
 'insertion_only_all_old_lines_identical':True,'statements':len(lines),'branches':len(arcs),
 'discarded_old_arcs':discarded,'new_target_arcs_directly_covered':sorted(new_arcs),
 'new_lines_directly_covered':[7666,7681,7699],'missing_lines':missing_lines,'missing_branches':missing_arcs,
 'callable_docstrings_unchanged':True,'unchanged_existing_ruff_findings':len(a),
 'focused_result':'12 passed,25 recorded unwired-signal disconnect warnings in5.67s; no warning filters changed',
 'native_before_exit':-6,'native_after_exit':0,'existing_drain_ms':15000,
 'limits':['Controlled CPU Event replaces model work; no GPU or model-science claim.',
 'Native15s owner deletion proved for real retrain QThread; all three worker classes checked for timeout, interruption and disconnected late signals.',
 'Page worker and central bridge unchanged. Normal completed and original native-deleted paths retained.',
 'Separate small-image Shiboken SIGSEGV, installed Save and full Qt serial acceptance remain OPEN.']}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
