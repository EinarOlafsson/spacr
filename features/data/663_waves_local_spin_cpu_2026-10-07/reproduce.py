"""Verify the bound archive or replay its bounded native CPU render probes."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    """Check payload/source identity before optional bounded scratch rendering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git", metavar="REF")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["payloads"].items():
        if args.git:
            if args.repo is None:
                parser.error("--git requires --repo")
            relative = root.relative_to(args.repo.resolve()) / name
            value = subprocess.check_output(
                ["git", "show", f"{args.git}:{relative.as_posix()}"], cwd=args.repo)
        else:
            value = (root / name).read_bytes()
        assert len(value) == expected["bytes"], name
        assert hashlib.sha256(value).hexdigest() == expected["sha256"], name
    receipt = json.loads((root / "receipt.json").read_text())
    for name, key in (("ambient.before.py.gz", "before_sha256"),
                      ("ambient.after.py.gz", "source_sha256")):
        assert hashlib.sha256(gzip.decompress((root / name).read_bytes())).hexdigest() == receipt[key]
    print(f"Verified {len(manifest['payloads'])} payloads and both frozen source hashes.")
    if not args.render:
        return
    if args.repo is None:
        parser.error("--render requires --repo")
    repo = args.repo.resolve()
    source = repo / "spacr/qt/widgets/ambient.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt["source_sha256"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    target = Path(tempfile.mkdtemp(prefix="waves-spin-replay-", dir="/mnt/wd4tb/scratch"))
    (target / "before.py").write_bytes(gzip.decompress((root / "ambient.before.py.gz").read_bytes()))
    for name in ("native_review.py", "facet_cost.py"):
        shutil.copyfile(root / name, target / name)
        with (target / (name + ".log")).open("w") as stream:
            subprocess.run([sys.executable, str(target / name), str(repo)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True,
                           timeout=180, env=os.environ.copy())
    print(f"Fresh replay retained at {target}; original receipts unchanged.")


if __name__ == "__main__":
    main()
