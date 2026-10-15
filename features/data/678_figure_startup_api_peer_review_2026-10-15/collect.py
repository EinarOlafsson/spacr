"""Bind the independent semantic review to immutable inputs, without generation."""

import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


REVIEW_COMMIT = "a1028d8caee6d4a65de0267e9babf6dc07443b79"
FIGURE_SOURCE = "6ac69c5b82343b329c546f45fafa1a13567995e6"
STARTUP_SOURCE = "c3b78f277dc7380a8718ef962d78637cfa7b264c"
LANGUAGES = ["de", "es", "fr", "hi", "is", "ko", "pt", "sv", "zh_CN"]
EVIDENCE = "features/data/678_figure_startup_api_reviews_2026-10-15/"
OUTPUT = Path(__file__).with_name("receipt.json")


def blob(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def normalized(text):
    return " ".join(text.split())


snapshot_bytes = blob(REVIEW_COMMIT, EVIDENCE + "figure-startup-sources.json.gz")
snapshot = json.loads(gzip.decompress(snapshot_bytes))
author_receipt = json.loads(blob(REVIEW_COMMIT, EVIDENCE + "receipt.json"))
blocks = snapshot["changed_blocks"]
assert len(blocks) == 13 and len(snapshot["source_docs"]) == 10
for symbol, text in snapshot["source_docs"].items():
    assert sha(text) == author_receipt["source_docs_sha256"][symbol]

checks = [
    "LRU eviction; ram_cap counts full-resolution pixmaps, default 100; retained PNGs reload on demand; thumbnails and temporary-directory lifecycle remain distinct.",
    "Traversal order and missing-entry omission preserved; repeated references are retained, not deduplicated.",
    "Default 100 is an entry count, not bytes or a whole-process memory ceiling; LRU eviction leaves PNGs reloadable.",
    "Constructor cache parameter counts full-resolution pixmaps, default 100, with least-recently-used eviction.",
    "Default montage cap is 2,000 selected objects, not pixels; widget and QPixmap updates belong to GUI thread; image reads/crop selection belong to worker.",
    "Loading cover fills the window; photomosaic animation is associated with startup.",
    "Eager imports use worker thread with GUI-timer progress; default on-demand policy ends loading without those imports; host follows startup preferences.",
    "Three progress phases and thin line, logo-to-mosaic transition, animated removal and replay over existing window retain their roles and RST references.",
    "Full-window loading cover has optional photomosaic animation; no claim that imports must always finish first.",
    "Environment/offscreen/reduced-motion/startup-switch conditions disable startup animation; unreadable preference disables it; full Appearance paths retained.",
    "Explicit replay still works with startup switch off; two-second hold, reduced-motion one-second fade, click/key skip, unchanged window size/state/focus retained.",
    "Appearance > Animation path retained; reduced motion overrides startup preference.",
    "Accessibility path and listed-widget scope retained; replacement border width uses FOCUS_RING_PX and colour uses focus_ring_colour.",
]

source_files = {}
for block in blocks:
    symbol = block["label"].split("#")[0]
    module = next(name for name in (
        "spacr.qt.widgets.figure_queue", "spacr.qt.widgets.cell_montage_view",
        "spacr.qt.widgets.loading_screen", "spacr.qt.preferences", "spacr.qt.theme"
    ) if symbol == name or symbol.startswith(name + "."))
    path = module.replace(".", "/") + ".py"
    commit = FIGURE_SOURCE if module.endswith(("figure_queue", "cell_montage_view")) else STARTUP_SOURCE
    raw = blob(commit, path)
    tree = ast.parse(raw)
    node = tree
    for component in symbol[len(module):].strip(".").split(".") if symbol != module else []:
        node = next(child for child in node.body if getattr(child, "name", None) == component)
    docs = [ast.get_docstring(node) or ""]
    if isinstance(node, ast.ClassDef):
        docs += [ast.get_docstring(child) or "" for child in node.body if getattr(child, "name", None) == "__init__"]
    assert any(normalized(block["source"]) in normalized(doc) for doc in docs), block["label"]
    assert sha(block["source"]) == block["source_sha256"]
    source_files[path] = {"commit": commit, "sha256": sha(raw)}

worker_path = "spacr/qt/job_runner.py"
worker_raw = blob(FIGURE_SOURCE, worker_path)
worker_ast = ast.parse(worker_raw)
runner = next(node for node in worker_ast.body if getattr(node, "name", None) == "JobRunner")
submit = next(node for node in runner.body if getattr(node, "name", None) == "submit")
assert "It runs on a worker thread" in ast.get_docstring(submit)
source_files[worker_path] = {"commit": FIGURE_SOURCE, "sha256": sha(worker_raw)}

records = {}
review_files = {}
for language in LANGUAGES:
    path = f"docs/i18n/reviewed/api/{language}/2026-10-15-figure-startup-prose.json"
    raw = blob(REVIEW_COMMIT, path)
    data = json.loads(raw)
    assert data["language"] == language and len(data["records"]) == 13
    records[language] = {row["label"]: row for row in data["records"]}
    assert set(records[language]) == {row["label"] for row in blocks}
    review_files[path] = sha(raw)

findings = {
    "es": ("en el proceso de trabajo", "en el hilo de trabajo"),
    "pt": ("no processo de trabalho", "na thread de trabalho"),
}
coverage = []
for block, semantic_check in zip(blocks, checks):
    targets = {}
    for language in LANGUAGES:
        row = records[language][block["label"]]
        for field in ("source", "context", "source_sha256"):
            assert row[field] == block[field]
        assert re.findall(r":(?:class|data|meth|func):`[^`]+`|``[^`]+``", row["source"]) == re.findall(r":(?:class|data|meth|func):`[^`]+`|``[^`]+``", row["translation"])
        for literal in ("100", "2,000"):
            if literal in row["source"]:
                assert literal in row["translation"]
        result = {"translation_sha256": sha(row["translation"]), "decision": "PASS"}
        if language in findings and block["label"] == "spacr.qt.widgets.cell_montage_view#2":
            old, new = findings[language]
            assert row["translation"].count(old) == 1
            result.update({
                "decision": "CORRECTION_REQUESTED",
                "finding": "Worker wording can suggest an OS process; pinned JobRunner.submit executes a worker thread.",
                "replace": old,
                "with": new,
                "proposed_translation_sha256": sha(row["translation"].replace(old, new)),
            })
        targets[language] = result
    coverage.append({
        "label": block["label"], "source_sha256": block["source_sha256"],
        "context_sha256": sha(block["context"]), "semantic_review": semantic_check,
        "targets": targets,
    })

receipt = {
    "schema": 1, "date": "2026-10-15", "reviewed_commit": REVIEW_COMMIT,
    "reviewer": "Home-B website_686_687, independent of the author of these 117 targets",
    "review_kind": "Independent AI technical semantic review; no human or native-speaker signoff",
    "scope": "117 records, 13 blocks, 10 rendered API symbols, nine locales",
    "decision": "115 PASS; two exact ES/PT worker-thread wording corrections requested",
    "source_snapshot_sha256": sha(snapshot_bytes), "source_files": source_files,
    "review_files_sha256": review_files, "coverage": coverage,
    "validation": {
        "source_binding": "All 13 source blocks match docstrings in the two pinned source commits; ten snapshot document hashes match author receipt.",
        "record_binding": "All 117 source/context/hash identities match the immutable snapshot; protected RST names and numerical cache/montage limits retained.",
        "semantics": "All 117 translated targets read and independently compared with the exact English blocks; decisions recorded per target.",
        "normal_gates": "Previously passed normal collector/context gates supplied by author; not rerun or substituted by these receipt checks.",
    },
    "limitations": [
        "No claim of native-speaker fluency review or human linguistic signoff.",
        "No runtime, GUI, scientific, hardware or hosted-page acceptance was performed in this review.",
        "No generated catalogs or source behavior changed; source-bound semantic review does not close the whole N678 item.",
        "This receipt does not approve the two requested corrections until their actual committed targets are checked.",
    ],
}
OUTPUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print("117 immutable records checked: 115 semantic PASS; 2 exact worker-thread corrections requested.")
