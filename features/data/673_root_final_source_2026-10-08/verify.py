"""Verify the frozen N673 transfer payload and optional source tree."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    receipt = json.loads((folder / "receipt.json").read_text())
    for name, expected in receipt["payloads"].items():
        raw = (folder / name).read_bytes()
        assert len(raw) == expected["bytes"], name
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"], name
        if name.endswith(".gz"):
            gzip.decompress(raw)
    for option, key in ((args.source_root, "source_paths"),
                        (args.baseline_root, "baseline_source_paths")):
        if option is None:
            continue
        for name, expected in receipt[key].items():
            path = option / name
            if expected is None:
                assert not path.exists(), name
            else:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
    print("N673 payload and supplied source hashes pass")


if __name__ == "__main__":
    main()
