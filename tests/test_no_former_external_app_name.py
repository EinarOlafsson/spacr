"""The external Toxoplasma application is ParaScape in every tracked text file.

The name it replaced is held only as hex bytes, so this file does not spell it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMER_BYTES = bytes.fromhex("73746172706c617374")
FORMER = re.compile(re.escape(FORMER_BYTES), re.IGNORECASE)
TEXT_SUFFIXES = {
    ".py", ".rst", ".md", ".txt", ".json", ".html", ".js", ".css", ".qss", ".svg",
    ".po", ".pot", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".tsv", ".temp",
}


def _tracked():
    names = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                           capture_output=True).stdout.split(b"\0")
    return [name.decode() for name in names if name]


def test_no_tracked_path_names_the_former_application():
    assert not [name for name in _tracked() if FORMER.search(name.encode())]


def test_no_tracked_text_names_the_former_application():
    offenders = []
    for name in _tracked():
        path = ROOT / name
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        if FORMER.search(path.read_bytes()):
            offenders.append(name)
    assert not offenders


def test_a_former_name_is_found_in_any_letter_case():
    assert FORMER.search(b"Open " + FORMER_BYTES.upper())
    assert FORMER.search(b"Open " + FORMER_BYTES.capitalize())
    assert not FORMER.search(b"Open ParaScape")
