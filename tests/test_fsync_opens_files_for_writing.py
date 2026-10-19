"""Every file spaCR flushes to disk with os.fsync is opened writable.

Found natively on Windows 11 2026-10-19: os.fsync is FlushFileBuffers there, which needs write access, so a
file opened "rb" fails with [Errno 9] Bad file descriptor. Saving a mask in Make Masks, writing a filtered
mask, the run settings snapshot, CLI settings and AnnData exports all did this and failed on every Windows
computer; POSIX accepts a read-only descriptor, so Linux and macOS never showed it.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "spacr"
READ_ONLY_THEN_FSYNC = re.compile(
    r"""open\([^)\n]*["']rb["']\)\s+as\s+(\w+):\s*\n\s*os\.fsync\(\1\.fileno\(\)\)""")


def test_no_read_only_handle_is_fsynced():
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in READ_ONLY_THEN_FSYNC.finditer(text):
            offenders.append(f"{path.relative_to(ROOT.parent)}:{text.count(chr(10), 0, match.start()) + 1}")
    assert offenders == []


def test_a_writable_handle_can_be_fsynced(tmp_path):
    staged = tmp_path / "staged.bin"
    staged.write_bytes(b"spacr")
    with open(staged, "r+b") as handle:
        os.fsync(handle.fileno())
    assert staged.read_bytes() == b"spacr"
