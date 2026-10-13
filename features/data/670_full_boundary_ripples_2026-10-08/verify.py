"""Verify the frozen full-boundary field ripple evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import subprocess


HERE = pathlib.Path(__file__).resolve().parent


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="store_true")
    options = parser.parse_args()
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for name, expected in manifest.items():
        stored = (HERE / name).read_bytes()
        raw = gzip.decompress(stored) if name.endswith(".gz") else stored
        assert len(stored) == expected["stored_bytes"], name
        assert _sha(stored) == expected["stored_sha256"], name
        assert _sha(raw) == expected["raw_sha256"], name
    receipt = json.loads((HERE / "receipt.json").read_text())
    delta = json.loads(gzip.decompress(
        (HERE / "normal-inventory-delta.json.gz").read_bytes()))
    assert delta["api_before_count"] == delta["api_after_count"] == 13232
    assert len(delta["api_changed"]) == 1
    assert len(delta["ui_added"]) == len(delta["ui_removed"]) == 1
    assert delta["other_runtime_buckets_equal"]
    for identity in receipt["source_paths"].values():
        assert not identity.get("focused_changed_missing_lines")
        assert not identity.get("focused_changed_missing_arcs")
    if options.source:
        root = HERE.parents[2]
        for name, identity in receipt["source_paths"].items():
            raw = subprocess.check_output(
                ["git", "-C", str(root), "show", f"HEAD:{name}"])
            assert _sha(raw) == identity["after_sha256"], name
    print(f"Verified {len(manifest)} payloads and {len(receipt['source_paths'])} source paths")


if __name__ == "__main__":
    main()
