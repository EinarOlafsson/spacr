"""Compare exact-source archived ambient coverage without modifying a baseline."""
import argparse
import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
args = parser.parse_args()
folder = Path(__file__).resolve().parent
path = "spacr/qt/widgets/ambient.py"
old = subprocess.check_output(["git", "show", "392ca4d6a2319fd88c2d352b1a0b021682123d6b:" + path], cwd=args.repo)
current = (args.repo / path).read_bytes()
assert hashlib.sha256(old).hexdigest() == "08815be8f5e01de4ae87517290197d3c1747eff65b4e39afbd5f2f0e1d3b8a28"
assert hashlib.sha256(current).hexdigest() == "b2a191470947a2f832e7fe135835f05527e0a9e6fa1e7f41408108a4c412d855"
old_lines, current_lines = old.decode().splitlines(), current.decode().splitlines()
mapping = {}
for left, right, length in difflib.SequenceMatcher(None, old_lines, current_lines, autojunk=False).get_matching_blocks():
    mapping.update({left + offset + 1: right + offset + 1 for offset in range(length)})
changed_class = next(node for node in ast.parse(current).body if isinstance(node, ast.ClassDef) and node.name == "_FungalGrowthEngine")
old_class = next(node for node in ast.parse(old).body if isinstance(node, ast.ClassDef) and node.name == "_FungalGrowthEngine")
assert old_lines[:old_class.lineno - 1] + old_lines[old_class.end_lineno:] == current_lines[:changed_class.lineno - 1] + current_lines[changed_class.end_lineno:]
def inherited(line):
    mapped = mapping.get(abs(line))
    if mapped is None or changed_class.lineno <= mapped <= changed_class.end_lineno:
        return None
    return mapped if line > 0 else -mapped
hosted, cache, focused = [json.loads((folder / name).read_text()) for name in ("hosted392.json", "cache41.json", "refusals8.json")]
statements = set(focused["executed_lines"]) | set(focused["missing_lines"])
branches = set(map(tuple, focused["executed_branches"] + focused["missing_branches"]))
executed = set(cache["executed_lines"]) | set(focused["executed_lines"])
executed.update(mapped for line in hosted["executed_lines"] if (mapped := inherited(line)) is not None)
arcs = set(map(tuple, cache["executed_branches"] + focused["executed_branches"]))
for arc in hosted["executed_branches"]:
    mapped = tuple(inherited(line) for line in arc)
    if all(line is not None for line in mapped):
        arcs.add(mapped)
missing_lines = sorted(statements - executed)
missing_branches = sorted(branches - arcs)
result = {"source_sha256": hashlib.sha256(current).hexdigest(), "statements": len(statements), "branches": len(branches), "missing_lines": missing_lines, "missing_branches": missing_branches, "baseline_statement_allowance": 1, "baseline_branch_allowance": 0, "within_existing_baseline": len(missing_lines) <= 1 and not missing_branches, "scope": "Mapped hosted392 unchanged source outside the changed fungal class plus exact-b2a targeted runs; not a fresh complete-suite or hosted verdict"}
print(json.dumps(result, indent=2))
assert result["within_existing_baseline"]
