"""Reproduce the bounded web-source census without normal catalog generation."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import build_i18n_catalogs as builder


SOURCE = "6691ddc61b3fc74a704c72e496b7832268c0a4ea"
FILES = (
    "app.py", "detector.py", "examples.py", "inputs.py", "masks.py",
    "plaque.py", "registry.py", "results.py", "run.py", "service.py",
)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def assigned(tree, name):
    return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name
                        for target in node.targets))


expected = set()
raw_expected = set()
file_hashes = {}
normalizations = []
calls = 0
for filename in FILES:
    path = "tools/web_mask_generator/" + filename
    raw = (ROOT / path).read_bytes()
    assert raw == subprocess.check_output(["git", "show", SOURCE + ":" + path], cwd=ROOT)
    file_hashes[path] = sha(raw)
    for node in ast.walk(ast.parse(raw)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "tr" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            continue
        value = node.args[0].value
        raw_expected.add(value)
        expected.add(value.strip())
        calls += 1
        if value != value.strip():
            normalizations.append({
                "path": path, "line": node.lineno, "raw_source": value,
                "raw_sha256": sha(value), "canonical_source": value.strip(),
                "canonical_sha256": sha(value.strip()),
            })

family = ast.literal_eval(assigned(ast.parse(
    (ROOT / "tools/web_mask_generator/app.py").read_bytes()), "FAMILY_LABELS"))
expected.update(family.values())
actual = builder._web_mask_generator_ui_sources()
assert actual == expected
assert len(actual) == 183 and calls == 185 and len(raw_expected) == 175
assert len(normalizations) == 2
assert "# spaCR mask generator" in actual and not (set(family) & actual)
en_raw = (ROOT / "spacr/qt/i18n_catalogs/en.py").read_bytes()
english_sources = set(ast.literal_eval(assigned(ast.parse(en_raw), "UI_SOURCES").args[0]))
new = actual - english_sources
assert len(new) == 178
assert not any(name == "gradio" or name.startswith("tools.web_mask_generator")
               for name in sys.modules)
result = {
    "schema": 1, "web_source_commit": SOURCE, "web_source_sha256": file_hashes,
    "literal_tr_calls": calls, "literal_unique": len(raw_expected),
    "family_label_values": family, "total_owned_unique": len(actual),
    "matches_independent_literal_call_and_family_value_census": True,
    "already_generated_sources": sorted(actual & english_sources),
    "new_against_existing_english_UI": len(new),
    "new_family_captions": sorted(set(family.values()) - english_sources),
    "normalizations": normalizations,
    "normalization_contract": "Matches existing extractor source.strip(); spacr.qt.i18n._exact_translation resolves the stripped key and restores leading/trailing whitespace. Review inputs must bind to the canonical_source/hash shown here.",
    "markdown_heading_preserved": True, "family_identifiers_excluded": True,
    "web_and_gradio_modules_not_imported": True,
    "english_catalog_sha256_at_census": sha(en_raw),
    "qualification": "Measured against the pinned English catalog before new-source generation; the count is evidence, not a source-arrival ceiling or a translation review claim.",
}
Path(__file__).with_name("source-census.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n")
Path(__file__).with_name("new-sources.json").write_text(json.dumps([
    {"source": text, "source_sha256": sha(text)} for text in sorted(new)
], ensure_ascii=False, indent=2) + "\n")
print("183 owned sources / 178 new; two canonical whitespace normalizations; no web imports.")
