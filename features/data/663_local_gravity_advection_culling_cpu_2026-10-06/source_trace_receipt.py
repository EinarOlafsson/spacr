"""Relate an actual branch trace to precisely added application source lines."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
label=sys.argv[1];root=Path(__file__).parent;p=Path('spacr/qt/widgets/ambient.py')
diff=subprocess.check_output(['git','diff','--unified=0','HEAD','--',str(p)],text=True)
added=set();line=0
for row in diff.splitlines():
 if row.startswith('@@'):line=int(re.search(r'\+(\d+)',row).group(1))
 elif row.startswith('+') and not row.startswith('+++'):added.add(line);line+=1
 elif row.startswith(' ') and not row.startswith('---'):line+=1
report=json.loads((root/f'coverage-{label}.json').read_text())['files'][str(p)]
executed=set(report['executed_lines']);missing=set(report['missing_lines'])
arcs=report['executed_branches']+report['missing_branches'];touch=[a for a in arcs if any(v in added for v in a)]
receipt={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'added_executable_statements':len((executed|missing)&added),'missing_added_statements':sorted(missing&added),'touching_branch_arcs':len(touch),'missing_touching_arcs':[a for a in report['missing_branches'] if any(v in added for v in a)]}
(root/f'{label}-new-source-coverage.json').write_text(json.dumps(receipt,indent=2)+'\n')
(root/f'ambient-{label}.py').write_bytes(p.read_bytes());(root/f'{label}-source.diff').write_text(diff);print(receipt)
