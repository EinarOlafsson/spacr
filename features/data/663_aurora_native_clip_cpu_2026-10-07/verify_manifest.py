"""Verify the N663 Aurora proof payload and its historical source identity."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(data):
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def git_source(commit):
    """Read the exact ambient module from a committed Git object."""
    return subprocess.check_output(
        ["git", "show", f"{commit}:spacr/qt/widgets/ambient.py"], cwd=ROOT)


def aurora_class(source):
    """Find Aurora's class node in source text."""
    module = ast.parse(source.decode())
    return next(node for node in module.body
                if isinstance(node, ast.ClassDef) and node.name == "AuroraEngine")


def main():
    """Verify payload bytes, source commits, AST and measured parity claims."""
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, expected in manifest["payload_sha256"].items():
        assert digest((HERE / name).read_bytes()) == expected, name
    receipt = json.loads((HERE / "receipt.json").read_text())
    for label, commit in receipt["source_commits"].items():
        source = git_source(commit)
        assert digest(source) == receipt["module_sha256"][label], label
        node = aurora_class(source)
        actual = digest(ast.dump(node, include_attributes=False).encode())
        assert actual == receipt["aurora_class_ast_sha256"][label], label
    old = json.loads((HERE / "source-abba.json").read_text())
    final = json.loads((HERE / "source-abba-final-079.json").read_text())
    parity = json.loads((HERE / "parity-final-079.json").read_text())
    assert old["current_source_sha256"] == receipt["module_sha256"]["optimized"]
    assert final["current_source_sha256"] == parity["source_sha256"]
    assert final["current_source_sha256"] == receipt["integrated_module_sha256"]
    assert len(parity["rows"]) == 36
    native = [row for row in parity["rows"] if row["size"] == [3840, 2160]]
    small = [row for row in parity["rows"] if row["size"] == [1280, 720]]
    assert len(native) == 24 and all(row["changed_pixels"] == 0 for row in native)
    assert len(small) == 12 and max(row["changed_pixels"] for row in small) <= 47
    assert all(row["max_byte_diff"] <= 1 for row in small)
    print(f"verified {len(manifest['payload_sha256'])} payloads and 36 source-bound frames")


if __name__ == "__main__":
    main()
