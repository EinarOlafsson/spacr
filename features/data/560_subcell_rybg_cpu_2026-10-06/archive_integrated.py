"""Archive source-matched CPU evidence without duplicating model weights."""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path.cwd()
base = "23750631790924590490e356144c8da72ac805af"
target = root / "features/data/560_subcell_rybg_cpu_2026-10-06"
parity = json.loads((target / "native_parity.json").read_text())
assert parity["spacr_source_sha256"] == digest(root / "spacr/embeddings.py")
sources = {
    "spacr/embeddings.py": Path("/mnt/wd4tb/scratch/subcell-rybg-20261006/preflight_coverage.json"),
    "spacr/qt/screens/embeddings.py": Path("/mnt/wd4tb/scratch/f560-subcell-gui-coverage-20261006/native-size-fixture.json"),
}
coverage = {}
for name, receipt in sources.items():
    diff = subprocess.check_output(
        ["git", "diff", "--no-ext-diff", "--unified=0", base, "--", name],
        text=True,
    )
    added = set()
    line_number = None
    for line in diff.splitlines():
        if line.startswith("@@"):
            line_number = int(re.search(r"\+(\d+)", line).group(1))
        elif line_number is not None and line.startswith("+"):
            added.add(line_number)
            line_number += 1
        elif line_number is not None and not line.startswith("-"):
            line_number += 1
    record = json.loads(receipt.read_text())["files"][name]
    executed = set(record["executed_lines"])
    missing = set(record["missing_lines"])
    branch_hits = {tuple(arc) for arc in record["executed_branches"]}
    branch_misses = {tuple(arc) for arc in record["missing_branches"]}
    executable = added & (executed | missing)
    origin_arcs = {arc for arc in branch_hits | branch_misses if arc[0] in added}
    touching_arcs = {arc for arc in branch_hits | branch_misses if set(arc) & added}
    assert not executable & missing
    assert not origin_arcs & branch_misses
    assert not touching_arcs & branch_misses
    coverage[name] = {
        "source_sha256": digest(root / name),
        "source_git_blob": subprocess.check_output(["git", "hash-object", name], text=True).strip(),
        "raw_receipt_path": str(receipt),
        "raw_receipt_sha256": digest(receipt),
        "added_executable_lines": sorted(executable),
        "covered_added_lines": len(executable),
        "covered_added_origin_arcs": sorted(origin_arcs),
        "covered_arcs_touching_added_lines": sorted(touching_arcs),
        "missing_added_lines": [],
        "missing_added_arcs": [],
    }
assert len(coverage["spacr/embeddings.py"]["added_executable_lines"]) == 26
assert len(coverage["spacr/qt/screens/embeddings.py"]["added_executable_lines"]) == 117

old_source = subprocess.check_output(["git", "show", f"{base}:spacr/embeddings.py"], text=True)
new_source = (root / "spacr/embeddings.py").read_text()
owned = {}
for name in ("_retrieval_scorecard", "_weights_on_disk", "encoder_entry"):
    texts = []
    for source in (old_source, new_source):
        node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
        texts.append(ast.get_source_segment(source, node))
    assert texts[0] == texts[1]
    owned[name] = hashlib.sha256(texts[1].encode()).hexdigest()

result = {
    "source_checkpoint": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "comparison_base": base,
    "scope": "Focused CPU implementation/checkpoint parity; no GPU, human-label four-stain benchmark, whole-suite or final GitHub verdict",
    "root_validation": {"passed": 75, "skipped": 1, "skip_reason": "main environment has no timm; genuine strict timm cases passed separately", "memory_cap": "4G", "cuda_visible_devices": "", "qt_qpa_platform": "offscreen"},
    "added_source_coverage": coverage,
    "workstation_owned_functions_unchanged_sha256": owned,
    "native_parity": parity,
    "artifact_sha256": {path.name: digest(path) for path in sorted(target.iterdir()) if path.is_file()},
    "official_source_commit": "c4e0a9106ecffdd2ad91fbbd60bda6dc9fc74009",
    "raw_upstream_and_weights_directory": "/mnt/wd4tb/scratch/subcell-rybg-20261006",
}
(root / "features/data/560_subcell_rybg_cpu_2026-10-06.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({name: {"added_lines": len(row["added_executable_lines"]), "added_origin_arcs": len(row["covered_added_origin_arcs"])} for name, row in coverage.items()}, sort_keys=True))
