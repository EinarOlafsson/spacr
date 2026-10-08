"""Read API/runtime English source identities without writing catalogs."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
sys.path.insert(0, str(root / "tools"))
sys.path.insert(0, str(root))

import build_documentation_i18n as api
import build_i18n_catalogs as runtime

assert api.ROOT.resolve() == root
assert runtime.ROOT.resolve() == root

docs = api.public_docstrings()
sources = runtime.canonical_sources()
foreign = {
    name: str(path)
    for name, module in sys.modules.items()
    if (name == "spacr" or name.startswith("spacr."))
    if (path := getattr(module, "__file__", None)) is not None
    if not Path(path).resolve().is_relative_to(root)
}
if foreign:
    raise RuntimeError(f"foreign spaCR modules in inventory process: {foreign}")


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


runtime_entries = {}
runtime_samples = {}
for name, table in sources.items():
    if isinstance(table, dict):
        runtime_entries[name] = {str(key): digest(value)
                                 for key, value in sorted(table.items())}
        runtime_samples[name] = {
            str(key): value for key, value in table.items()
            if "Save on navigation" in str(value)
            or "Save an edited mask before Keep" in str(value)
        }
    elif isinstance(table, (tuple, list, set, frozenset)):
        runtime_entries[name] = {str(value): digest(value) for value in table}
        runtime_samples[name] = {
            str(value): value for value in table
            if "Save on navigation" in str(value)
            or "Save an edited mask before Keep" in str(value)
        }
    else:
        runtime_entries[name] = {"<value>": digest(table)}

payload = {
    "tree": str(root),
    "head": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "source_blob": subprocess.check_output(
        ["git", "-C", str(root), "hash-object",
         "spacr/qt/screens/make_masks.py"], text=True).strip(),
    "api_count": len(docs),
    "api_hashes": {key: digest(value) for key, value in sorted(docs.items())},
    "api_make_masks": {key: value for key, value in sorted(docs.items())
                       if key.startswith("spacr.qt.screens.make_masks.")},
    "runtime_counts": {key: len(value) for key, value in runtime_entries.items()},
    "runtime_hashes": runtime_entries,
    "runtime_samples": runtime_samples,
}
output.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                             sort_keys=True, default=str) + "\n")
print(json.dumps({"head": payload["head"], "api_count": len(docs),
                  "runtime_counts": payload["runtime_counts"],
                  "source_blob": payload["source_blob"]}, sort_keys=True))
