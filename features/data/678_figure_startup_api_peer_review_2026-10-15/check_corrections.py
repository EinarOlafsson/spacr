"""Check only the two requested target corrections against the peer receipt."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parent
ORIGINAL = json.loads((ROOT / "receipt.json").read_text())
CORRECTION = subprocess.check_output(["git", "rev-parse", "457f338e814"]).decode().strip()
INTEGRATED = subprocess.check_output(["git", "rev-parse", "a65c5e630fe"]).decode().strip()
LABEL = "spacr.qt.widgets.cell_montage_view#2"
BLOCK = next(row for row in ORIGINAL["coverage"] if row["label"] == LABEL)


def blob(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


rows = []
for language in ("es", "pt"):
    path = f"docs/i18n/reviewed/api/{language}/2026-10-15-figure-startup-prose.json"
    raw = blob(CORRECTION, path)
    assert blob(INTEGRATED, path) == raw
    data = json.loads(raw)
    row = next(item for item in data["records"] if item["label"] == LABEL)
    finding = BLOCK["targets"][language]
    assert row["source_sha256"] == BLOCK["source_sha256"] == sha(row["source"])
    assert sha(row["context"]) == BLOCK["context_sha256"]
    assert sha(row["translation"]) == finding["proposed_translation_sha256"]
    assert finding["with"] in row["translation"] and finding["replace"] not in row["translation"]
    catalog_path = f"docs/source/_static/i18n/api/{language}.json"
    catalog_raw = blob(CORRECTION, catalog_path)
    assert blob(INTEGRATED, catalog_path) == catalog_raw
    catalog = json.loads(catalog_raw)["symbols"]["spacr.qt.widgets.cell_montage_view"]
    assert catalog["text"].split("\n\n")[2] == row["translation"]
    assert catalog["source_blocks_sha256"][2] == row["source_sha256"]
    rows.append({
        "language": language, "label": LABEL, "decision": "PASS",
        "review_file": path, "review_file_sha256": sha(raw),
        "catalog_file": catalog_path, "catalog_file_sha256": sha(catalog_raw),
        "source_sha256": row["source_sha256"], "context_sha256": sha(row["context"]),
        "translation_sha256": sha(row["translation"]), "translation": row["translation"],
        "semantic_review": "Actual corrected target now explicitly describes a worker thread. GUI-thread responsibilities, 2,000-object default and RST references remain unchanged.",
    })

result = {
    "schema": 1, "date": "2026-10-15", "original_peer_receipt_sha256": sha((ROOT / "receipt.json").read_bytes()),
    "original_reviewed_commit": ORIGINAL["reviewed_commit"], "correction_commit": CORRECTION,
    "integrated_commit": INTEGRATED,
    "review_kind": "Independent AI technical semantic review; no human or native-speaker signoff",
    "scope": "Only two requested ES/PT cell_montage_view#2 corrections",
    "records": rows,
    "decision": "Both corrections PASS; together with the original 115 accepted targets, all 117 scoped targets are approved for technical source semantics.",
    "validation": "Two corrected review targets match the exact requested replacement hashes, preserve source/context, and appear verbatim in their normally generated catalog block. Integrated review/catalog file bytes match the correction commit.",
    "normal_generation_evidence": "457f338e814:features/data/678_montage_thread_review_2026-10-15/receipt.json",
    "limitations": ORIGINAL["limitations"][:3],
    "no_duplicate_validation": "No full 117-record rerun, normal collector, generation, model, GUI or targeted tests were run for this addendum.",
}
(ROOT / "correction-approval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("2 corrected targets PASS; original 115-target approval retained; scoped total 117 approved.")
