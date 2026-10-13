"""Item 434: the object a damaged checkout is missing, kept as text.

A checkout lost one object and could no longer fetch, because every pack
the server sends has deltas based on it. The object is a commit, and git
names an object by the SHA-1 of "<type> <length>\\0<content>" -- so the
commit written back byte for byte gets the name it had.

This recomputes that name from the bytes in the repository. A file that
rots here fails here, rather than on the machine that has no other way to
get it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "tools" / "repair"
MISSING = "984b725b5a11346ebf011808812d562a43869fc7"


def _git_name(kind: str, payload: bytes) -> str:
    """The name git would give this content."""
    header = f"{kind} {len(payload)}".encode() + b"\0"
    return hashlib.sha1(header + payload).hexdigest()


def test_the_kept_object_still_names_itself():
    """The whole point: these bytes ARE that object, not a copy of it."""
    kept = REPAIR / f"{MISSING}.commit"
    assert kept.is_file(), f"{kept} is gone"
    assert _git_name("commit", kept.read_bytes()) == MISSING, (
        "the file no longer hashes to the object it is named after, so "
        "writing it into a damaged checkout would add a DIFFERENT object "
        "and leave the original still missing")


def test_it_is_a_commit_and_says_what_it_is():
    text = (REPAIR / f"{MISSING}.commit").read_text(encoding="utf-8",
                                                    errors="replace")
    assert text.startswith("tree "), "a commit object begins with its tree"
    assert "author " in text and "committer " in text


def test_the_instructions_are_beside_it():
    readme = REPAIR / "README.md"
    assert readme.is_file()
    written = readme.read_text(encoding="utf-8")
    assert "git hash-object -t commit -w --stdin" in written
    assert "not** with `git fetch`" in written, (
        "the instructions have to say that fetch is the broken thing, or "
        "the first thing the reader tries is the thing that cannot work")
