"""Check the recorded source bindings without importing Qt or running tests."""

import hashlib
import json
from pathlib import Path
import subprocess


root = Path(__file__).resolve().parents[3]
receipt = json.loads(Path(__file__).with_name("receipt.json").read_text())
for binding in receipt["source_bindings"]:
    content = subprocess.check_output(
        ["git", "show", f"{binding['commit']}:{binding['path']}"], cwd=root)
    assert hashlib.sha256(content).hexdigest() == binding["sha256"], binding
print(json.dumps({"source_bindings_verified": len(receipt["source_bindings"]),
                  "requirements": len(receipt["requirements"]),
                  "runtime_tests_run": False,
                  "overall_status": receipt["overall_status"]}))
