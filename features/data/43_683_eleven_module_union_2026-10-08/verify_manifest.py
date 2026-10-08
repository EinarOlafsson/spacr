"""Verify the archived coverage evidence file hashes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    """Compare payloads on disk or in Git with the frozen manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", action="store_true")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    root = folder.parents[2]
    prefix = folder.relative_to(root).as_posix()

    def read(name):
        """Load one payload without changing its bytes."""
        if args.git:
            return subprocess.check_output(
                ["git", "-C", str(root), "show", f"HEAD:{prefix}/{name}"]
            )
        return (folder / name).read_bytes()

    manifest = json.loads(read("MANIFEST.json"))
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256(read(name)).hexdigest()
        assert actual == expected, name
    print(f"Verified {len(manifest['sha256'])} coverage evidence payloads")


if __name__ == "__main__":
    main()
