"""Verify portable archive payload hashes in a filesystem or Git revision."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--git")
args = parser.parse_args()
directory = Path(__file__).resolve().parent
repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=directory, text=True).strip())
relative = directory.relative_to(repo).as_posix()

def read(name):
    if args.git:
        return subprocess.check_output(["git", "show", f"{args.git}:{relative}/{name}"], cwd=repo)
    return (directory / name).read_bytes()

manifest = json.loads(read("manifest.json"))
for name, record in manifest["payloads"].items():
    payload = read(name)
    assert len(payload) == record["bytes"], name
    assert hashlib.sha256(payload).hexdigest() == record["sha256"], name
print(json.dumps({"verified_payloads": len(manifest["payloads"]), "bytes": sum(row["bytes"] for row in manifest["payloads"].values()), "mode": args.git or "filesystem"}))
