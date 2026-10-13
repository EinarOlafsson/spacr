"""Regenerate the deterministic receipt and payload-hash manifest."""

import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    before = gzip.decompress((HERE / "test-before.py.gz").read_bytes())
    after = gzip.decompress((HERE / "test-after.py.gz").read_bytes())
    payloads = sorted(path.name for path in HERE.iterdir()
                      if path.is_file() and path.name not in
                      {"MANIFEST.json", "receipt.json"})
    payloads.append("receipt.json")
    receipt = {
        "hosted_source": "f0b81dacefe089c54b1d22a4aebd363a29d53cfb",
        "hosted_job_id": 113582650451,
        "hosted_result": "worker gw1 died on the two-well resume test; 800 passed, 1 skipped",
        "local_original_file": "10 passed in 849.99s; separate original-source run",
        "targeted_result": "1 passed in 275.32s with original 300s limit and 4 GiB cap",
        "test_before_sha256": _sha(before),
        "test_after_sha256": _sha(after),
        "payloads": sorted(payloads),
        "limits": "No worker exit signal or native stack in hosted job log; timeout is inferred from 300-second gap and measured pacing.",
    }
    (HERE / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for name in sorted(payloads):
        data = (HERE / name).read_bytes()
        row = {"path": name, "bytes": len(data), "sha256": _sha(data)}
        if name.endswith(".gz"):
            raw = gzip.decompress(data)
            row.update(raw_bytes=len(raw), raw_sha256=_sha(raw))
        rows.append(row)
    manifest = {"hosted_source": receipt["hosted_source"], "payloads": rows}
    (HERE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
