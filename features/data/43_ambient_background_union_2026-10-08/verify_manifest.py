"""Verify every frozen source-bound coverage payload from files or Git."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = HERE.relative_to(ROOT).as_posix()


def _read(name, git):
    if git:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"HEAD:{PREFIX}/{name}"]
        )
    return (HERE / name).read_bytes()


def verify(git=False):
    hashes = json.loads(_read("MANIFEST.json", git))
    assert set(hashes) == {
        "README.md",
        "receipt.json",
        "verify_union.py",
        "verify_manifest.py",
        "hosted-three.json.gz",
        "root-focused-two.json.gz",
        "background-86.json.gz",
        "integrated-7.json.gz",
        "integrated-7.log.gz",
        "background-measured-ambient.py.gz",
    }
    for name, expected in hashes.items():
        assert hashlib.sha256(_read(name, git)).hexdigest() == expected, name
    print(len(hashes), "source-bound background coverage payloads verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", action="store_true")
    verify(parser.parse_args().git)
