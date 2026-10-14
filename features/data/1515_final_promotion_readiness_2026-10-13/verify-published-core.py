"""Check the tagged Python source against the exact full-CI candidate."""
import argparse
import ast
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--candidate', default='eb72a1a85f5c3a3324f295ab18ad730aea7d5570')
parser.add_argument('--release', default='v1.5.1.5')
parser.add_argument('--version', default='1.5.1.5')
args = parser.parse_args()

def git(*parts):
    return subprocess.check_output(['git', '-C', str(args.repo), *parts])

def content(ref, path):
    return git('show', f'{ref}:{path}')

def source_files(ref):
    paths = git('ls-tree', '-r', '--name-only', '-z', ref, '--', 'spacr').decode().split('\0')
    return {path for path in paths if path.endswith('.py')}

def version_value(raw, name):
    tree = ast.parse(raw)
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(matches) != 1 or not isinstance(matches[0].value, ast.Constant):
        raise ValueError(f'Expected one literal assignment for {name}')
    return matches[0].value.value

problems = []
candidate_files = source_files(args.candidate)
release_files = source_files(args.release)
if candidate_files != release_files:
    problems.append({'python_source_inventory_changed': sorted(candidate_files ^ release_files)})
changed = []
for path in sorted(candidate_files & release_files):
    before = content(args.candidate, path)
    after = content(args.release, path)
    if before != after:
        changed.append(path)
        if path != 'spacr/_version.py':
            problems.append({'untested_python_source_change': path})
for path, name in [('setup.py', 'VERSION'), ('spacr/_version.py', '__version__')]:
    before = content(args.candidate, path)
    after = content(args.release, path)
    old = version_value(before, name)
    new = version_value(after, name)
    if new != args.version:
        problems.append({'incorrect_release_version': path, 'actual': new})
    before_text = before.decode('utf-8')
    old_literal = f'"{old}"'
    if before_text.count(old_literal) != 1:
        problems.append({'ambiguous_expected_literal': path})
    elif before_text.replace(old_literal, f'"{args.version}"', 1).encode('utf-8') != after:
        problems.append({'changes_beyond_version_literal': path})
print(json.dumps({
    'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'candidate': git('rev-parse', args.candidate).decode().strip(),
    'release_commit': git('rev-list', '-n', '1', args.release).decode().strip(),
    'release_ref': args.release,
    'python_source_files': len(release_files),
    'changed_python_source_files': changed,
    'core_file_sha256': {path: hashlib.sha256(content(args.release, path)).hexdigest()
                         for path in ['setup.py', 'spacr/_version.py', 'CITATION.cff']},
    'verified_source_equivalence_except_version': not problems,
    'problems': problems,
    'scope': 'Tagged Python source and exact core version literals; not artifact, installer, citation or runtime acceptance.'
}, indent=2))
raise SystemExit(0 if not problems else 2)
