"""Verify quantity-control payloads and frozen renderer hashes without Qt."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    """Check filesystem or Git payload identity and both frozen sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git", metavar="REF")
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["payloads"].items():
        if args.git:
            if args.repo is None:
                parser.error("--git requires --repo")
            relative = root.relative_to(args.repo.resolve()) / name
            data = subprocess.check_output(
                ["git", "show", f"{args.git}:{relative.as_posix()}"], cwd=args.repo)
        else:
            data = (root / name).read_bytes()
        assert len(data) == expected["bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
    receipt = json.loads((root / "receipt.json").read_text())
    for name, key in (("ambient.before.py.gz", "before_sha256"),
                      ("ambient.after.py.gz", "source_sha256")):
        source = gzip.decompress((root / name).read_bytes())
        assert hashlib.sha256(source).hexdigest() == receipt[key]
    print(f"Verified {len(manifest['payloads'])} payloads and both frozen source hashes.")


if __name__ == "__main__":
    main()
