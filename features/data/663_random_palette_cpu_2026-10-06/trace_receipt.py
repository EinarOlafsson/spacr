"""Bind recorded branch coverage to exactly added palette application lines."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = Path('spacr/qt/widgets/ambient.py')
diff = subprocess.check_output(['git', 'diff', '--unified=0', '9ee363637f^',
                                '9ee363637f', '--', str(path)], text=True)
added = set()
line = 0
for row in diff.splitlines():
    if row.startswith('@@'):
        line = int(re.search(r'\+(\d+)', row).group(1))
    elif row.startswith('+') and not row.startswith('+++'):
        added.add(line)
        line += 1
    elif row.startswith(' ') and not row.startswith('---'):
        line += 1
report = json.loads((ROOT / 'coverage.json').read_text())['files'][str(path)]
executed = set(report['executed_lines'])
missing = set(report['missing_lines'])
arcs = report['executed_branches'] + report['missing_branches']
source = subprocess.check_output(['git', 'show', '9ee363637f:' + str(path)])
receipt = {'commit': '9ee363637f', 'source_sha256': hashlib.sha256(source).hexdigest(),
           'added_executable_statements': len((executed | missing) & added),
           'missing_added_statements': sorted(missing & added),
           'touching_branch_arcs': len([arc for arc in arcs if any(v in added for v in arc)]),
           'missing_touching_arcs': [arc for arc in report['missing_branches']
                                     if any(v in added for v in arc)]}
(ROOT / 'source.diff').write_text(diff)
(ROOT / 'source_coverage.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
