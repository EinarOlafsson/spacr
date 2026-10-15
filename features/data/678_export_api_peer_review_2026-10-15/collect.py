"""Bind the independent 54-target export API semantic review to source bytes."""

import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


COMMIT = subprocess.check_output(["git", "rev-parse", "fec368766e6"]).decode().strip()
SOURCE = "c1be6383a7c601f473a27419bd8692dfbf49edc7"
LANGUAGES = ["de", "es", "fr", "hi", "is", "ko", "pt", "sv", "zh_CN"]
EVIDENCE = "features/data/678_export_api_reviews_2026-10-15/"


def blob(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def normalized(text):
    return " ".join(text.split())


snapshot_raw = blob(COMMIT, EVIDENCE + "sources.json.gz")
snapshot = json.loads(gzip.decompress(snapshot_raw))
author = json.loads(blob(COMMIT, EVIDENCE + "receipt.json"))
assert author["source_commit"] == SOURCE
for symbol, text in snapshot["source_docs"].items():
    assert sha(text) == author["source_docs_sha256"][symbol]
assert len(snapshot["source_docs"]) == len(snapshot["changed_blocks"]) == 6

records = {}
file_hashes = {}
for language in LANGUAGES:
    path = f"docs/i18n/reviewed/api/{language}/2026-10-15-figure-export-prose.json"
    raw = blob(COMMIT, path)
    data = json.loads(raw)
    assert data["language"] == language and len(data["records"]) == 6
    records[language] = {row["label"]: row for row in data["records"]}
    file_hashes[path] = sha(raw)

decisions = [
    "Propagation receives Image UMAP values only; other figure-style edits excluded, missing Image UMAP controls yield an empty dictionary, and None disables Propagate with an explanation.",
    "Result is dict style data with exact spacr_style_kind/general/per_graph keys; per-graph values that are not dictionaries are omitted, not generalized to arbitrary mappings.",
    "General overrides and graph-type-keyed per_graph overrides remain separate; non-dictionary per-graph values omitted.",
    "File settings distinguished from plot context-menu settings; Matplotlib also has live figure settings, while resized detached export copy receives additional export-only text scaling.",
    "Colour targets retain axes titles/labels/ticks, legend entries, axes text, spines and tick marks; figure-level text, legend titles and plotted data lines excluded. text_colour/line_colour override ink only in their own scopes; empty background makes both backgrounds transparent. Grid True/False/None remains draw/disable/preserve. Dimensions in inches change only when BOTH positive; DPI zero and font scale <=0 preserve existing values.",
    "Fast-plot file controls and format/resolution/page size retained; live Matplotlib figure menu exists and export dialog adds text scale for its copy, without claiming live edits are absent.",
]
source_files = {}
coverage = []
for block, decision in zip(snapshot["changed_blocks"], decisions):
    symbol = block["label"].split("#")[0]
    module = next(name for name in ("spacr.qt.widgets.figure_settings", "spacr.qt.widgets.save_figure_dialog") if symbol == name or symbol.startswith(name + "."))
    path = module.replace(".", "/") + ".py"
    raw = blob(SOURCE, path)
    node = ast.parse(raw)
    for component in symbol[len(module):].strip(".").split(".") if symbol != module else []:
        node = next(child for child in node.body if getattr(child, "name", None) == component)
    docs = [ast.get_docstring(node) or ""]
    if isinstance(node, ast.ClassDef):
        docs += [ast.get_docstring(child) or "" for child in node.body if getattr(child, "name", None) == "__init__"]
    assert any(normalized(block["source"]) in normalized(doc) for doc in docs), block["label"]
    assert sha(block["source"]) == block["source_sha256"]
    source_files[path] = sha(raw)
    targets = {}
    for language in LANGUAGES:
        row = records[language][block["label"]]
        assert all(row[key] == block[key] for key in ("source", "context", "source_sha256"))
        assert re.findall(r":(?:class|data|meth|func):`[^`]+`|``[^`]+``", row["source"]) == re.findall(r":(?:class|data|meth|func):`[^`]+`|``[^`]+``", row["translation"])
        targets[language] = {"decision": "PASS", "translation_sha256": sha(row["translation"])}
    coverage.append({"label": block["label"], "source_sha256": block["source_sha256"], "context_sha256": sha(block["context"]), "semantic_review": decision, "targets": targets})

receipt = {
    "schema": 1, "date": "2026-10-15", "reviewed_commit": COMMIT,
    "source_commit": SOURCE, "source_files_sha256": source_files,
    "source_snapshot_sha256": sha(snapshot_raw), "review_files_sha256": file_hashes,
    "reviewer": "Home-B website_686_687; authored the factual English correction, did not author these API translations",
    "review_kind": "Independent AI technical semantic review; no human or native-speaker signoff",
    "scope": "54 target records, six changed rendered blocks/symbols, nine locales",
    "decision": "PASS all 54; no target corrections requested",
    "coverage": coverage,
    "validation": {
        "semantic": "All 54 source/target pairs read and compared directly; source-specific preserved distinctions listed per block.",
        "binding": "Six blocks match actual pinned AST docstrings; six snapshot document hashes match author receipt; all 54 source/context/hash identities and protected RST names match.",
        "normal_gates": "Author reports normal source/context gates already passed; not rerun or replaced by this review receipt.",
        "generation": "API owner handles normal generation separately; no writer/model/global collector or duplicate tests run here.",
    },
    "limitations": ["Technical semantic approval only; no native-speaker fluency claim.", "No hosted/rendered documentation, runtime, scientific or hardware acceptance performed.", "The six changed blocks are the complete scope of this receipt, not the full N678 item or all prose in these modules."],
}
Path(__file__).with_name("receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print("54 source-bound targets PASS; six changed blocks, nine locales; no generation or duplicate tests.")
