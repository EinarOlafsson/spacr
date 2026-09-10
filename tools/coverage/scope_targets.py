"""Turn a coverage JSON into a work list of exact arcs.

Hand out arcs, not modules. Measured on the same twenty modules with the same
agents: "cover this module" closed 0 of 83 targeted arcs and "reach line 818
and arc 816->822" closed 79. The scope was the only variable, so this exists
to make the narrow scope the easy one to produce.

Two filters keep the list honest:

* a module whose source differs between the measured commit and the working
  tree is DROPPED, because its line numbers now name different code -- the
  trap instruction 310 records three separate ways of falling into;
* a module importing a package this interpreter does not have is dropped,
  because no test written here can reach it.

    python tools/coverage/scope_targets.py <coverage.json> [--min-gap 4]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
STDLIB = set(sys.stdlib_module_names)


def _third_party_imports(path: pathlib.Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (OSError, SyntaxError):
        return {"<unreadable>"}
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names if n not in STDLIB and n not in ("spacr", "__future__")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("coverage_json")
    ap.add_argument("--min-gap", type=int, default=4)
    ap.add_argument("--measured-at", default=None,
                    help="commit the JSON was measured on; defaults to HEAD")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.coverage_json).read_text())["files"]
    at = args.measured_at or "HEAD"

    def _git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                              text=True).stdout

    drifted = set(_git("diff", "--name-only", at, "HEAD").split())
    dirty = {ln[3:].strip() for ln in _git("status", "--porcelain").splitlines()
             if ln[:2].strip()}

    absent: dict[str, bool] = {}
    rows, skipped = [], {"drifted": 0, "blocked": 0, "small": 0}
    for path, entry in data.items():
        if not path.startswith("spacr/"):
            continue
        summary = entry["summary"]
        gap = summary["missing_lines"] + summary.get("missing_branches", 0)
        if gap < args.min_gap:
            skipped["small"] += 1
            continue
        if path in drifted or path in dirty:
            skipped["drifted"] += 1
            continue
        blocked = False
        for name in _third_party_imports(ROOT / path):
            if name not in absent:
                try:
                    absent[name] = importlib.util.find_spec(name) is None
                except Exception:                                # noqa: BLE001
                    absent[name] = True
            if absent[name]:
                blocked = True
                break
        if blocked:
            skipped["blocked"] += 1
            continue
        rows.append({
            "module": path,
            "gap": gap,
            "percent": round(summary["percent_covered"], 1),
            "lines": entry["missing_lines"],
            "branches": [list(a) for a in entry.get("missing_branches", [])],
        })

    rows.sort(key=lambda r: -r["gap"])
    print(json.dumps(rows, indent=1))
    total = sum(r["gap"] for r in rows)
    print(f"# {len(rows)} modules, {total} arcs actionable here", file=sys.stderr)
    print(f"# skipped: {skipped['drifted']} drifted, {skipped['blocked']} "
          f"dependency-blocked, {skipped['small']} under --min-gap",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
