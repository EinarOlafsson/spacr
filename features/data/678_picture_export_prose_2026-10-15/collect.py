"""Record the bounded picture-export prose review and unchanged executable AST."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess


BASE = "0118cabd401"
PATH = "spacr/qt/widgets/picture_export.py"
MODULE = "spacr.qt.widgets.picture_export"
before = subprocess.check_output(["git", "show", f"{BASE}:{PATH}"])
after = Path(PATH).read_bytes()
before_tree, after_tree = ast.parse(before), ast.parse(after)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def docs(node, name):
    result = {name: ast.get_docstring(node)}
    for child in node.body:
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result.update(docs(child, name + "." + child.name))
    return result


def without_docs(tree):
    tree = copy.deepcopy(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


old_docs, new_docs = docs(before_tree, MODULE), docs(after_tree, MODULE)
assert old_docs.keys() == new_docs.keys()
assert without_docs(before_tree) == without_docs(after_tree)
compile(after, PATH, "exec")
reasons = {
    MODULE: "_save_pdf calls QPainter.drawImage with the supplied raster; physical page dimensions use pixel count/export DPI, not microscopy calibration. _save_png copies pixels and sets dots-per-metre metadata. Menu choice ultimately controls suffix in save_as.",
    MODULE + "._save_png": "Implementation stamps integer dots-per-metre on a copy and still attempts save after a metadata error; it establishes no universal default DPI of external layout programs.",
    MODULE + "._save_pdf": "Width and height each use max(1.0, pixels / dpi * 25.4), then drawImage fills the page with zero margins. The old unconditional no-stretch claim omitted this independent clamp.",
    MODULE + ".ask_where_to_save": "QFileDialog cancellation returns empty; no-suffix choice follows selected filter; existing suffix is returned as supplied, with final menu enforcement performed by save_as. PDF also contains raster pixels, so 'PDF rather than raster' was misleading.",
    MODULE + ".install_picture_save": "Nested _menu calls picture() before choose_format, calls a callable stem after format selection, then calls save_as. The picture callback is not repeated after format/path dialogs.",
}
changed = {key for key in old_docs if old_docs[key] != new_docs[key]}
assert changed == reasons.keys()
ledger = {
    "schema": 1, "date": "2026-10-15", "item": "N678",
    "base_commit": subprocess.check_output(["git", "rev-parse", BASE]).decode().strip(),
    "source_path": PATH, "before_sha256": sha(before), "after_sha256": sha(after),
    "scope": "One picture-export module; five proven factual docstring corrections, nine docstrings retained. No runtime caption or behavior changes.",
    "changed": [{"symbol": key, "before": old_docs[key], "after": new_docs[key], "implementation_evidence": reasons[key]} for key in old_docs if key in changed],
    "retained": [{"symbol": key, "docstring_sha256": sha(old_docs[key]), "decision": "Retained: inspected description matches the corresponding implementation within this factual scope."} for key in old_docs if key not in changed],
    "rendered_api_dependencies": [MODULE, MODULE + ".ask_where_to_save", MODULE + ".install_picture_save"],
    "private_documentation_only": [MODULE + "._save_png", MODULE + "._save_pdf"],
    "validation": {
        "syntax_compile": "PASS, compile() without importing Qt or producing bytecode files",
        "complete_executable_ast_after_removing_docstrings": "IDENTICAL",
        "documented_symbol_keys": "UNCHANGED",
        "pytest": "Not run: prose-only correction, exact executable AST unchanged; no duplicate export tests required.",
    },
    "limitations": [
        "API extraction, nine-locale review and normal catalog generation belong to the API lane and remain pending for the three rendered symbols.",
        "No claim of rendered/hosted documentation or native Qt acceptance from this source-only correction.",
        "This bounded ledger does not complete N678 or review all repository prose.",
    ],
}
assert len(ledger["changed"]) == 5 and len(ledger["retained"]) == 9
Path(__file__).with_name("ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
print("5 corrected / 9 retained docstrings; compile PASS; complete executable AST identical.")
