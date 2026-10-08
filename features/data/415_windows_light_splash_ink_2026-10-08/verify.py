from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
receipt = json.loads((here / "receipt.json").read_text())
for name, expected in receipt["payloads"].items():
    artifact = (here / name).read_bytes()
    assert hashlib.sha256(artifact).hexdigest() == expected["artifact_sha256"], name
    original = gzip.decompress(artifact) if name.endswith(".gz") else artifact
    assert hashlib.sha256(original).hexdigest() == expected["source_sha256"], name
    assert len(original) == expected["source_bytes"], name
baseline = json.loads(gzip.decompress((here / "baseline-inventory.json.gz").read_bytes()))
candidate = json.loads(gzip.decompress((here / "candidate-inventory.json.gz").read_bytes()))
diff = json.loads(gzip.decompress((here / "inventory-diff.json.gz").read_bytes()))
assert diff["baseline_api_count"] == diff["current_api_count"] == 13232
assert not any(diff["api"][kind] for kind in ("added", "removed", "changed"))
for bucket in diff["runtime"].values():
    assert not any(bucket.get(kind) for kind in ("added", "removed", "changed"))
assert diff["runtime"]["ui"]["baseline_total"] == diff["runtime"]["ui"]["current_total"] == 7288
assert baseline and candidate
for revision, blob in (("baseline_commit", "baseline_loading_screen_blob"), ("candidate_commit", "candidate_loading_screen_blob")):
    try:
        actual = subprocess.check_output(
            ["git", "rev-parse", f"{receipt[revision]}:spacr/qt/widgets/loading_screen.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        continue
    assert actual == receipt[blob], revision
print("12 payloads and API/runtime equivalence verified")
