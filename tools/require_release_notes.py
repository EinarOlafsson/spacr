#!/usr/bin/env python
"""Refuse a release that has no written summary of its changes.

Every GitHub release carries the hand-written notes in
``docs/notes/release_notes_<version>.md``: a plain-language account of what
changed since the previous version, in the sections the earlier releases use.
1.5.1.7 was published with only GitHub's generated "Full Changelog" link
because the workflow fell back to ``--generate-notes``; the maintainer then
made written notes a rule for every release (2026-11-05). ``release.yml``
runs this before anything is built and publishes the file as the release body.

    python tools/require_release_notes.py 1.5.1.8
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "docs" / "notes"
MIN_CHARS = 1500
CHANGE_SECTIONS = ("### New", "### Improved", "### Fixed")


def notes_path(version: str, notes_dir: Path = NOTES_DIR) -> Path:
    """Return where the notes for ``version`` live."""
    return notes_dir / f"release_notes_{version}.md"


def problems(version: str, notes_dir: Path = NOTES_DIR) -> list:
    """List what keeps the notes for ``version`` from being publishable."""
    path = notes_path(version, notes_dir)
    if not path.is_file():
        return [f"{path.relative_to(notes_dir.parent.parent)} does not exist"]
    text = path.read_text(encoding="utf-8")
    found = []
    if not re.search(rf"^## spaCR {re.escape(version)}\s*$", text, re.M):
        found.append(f"the notes do not start a '## spaCR {version}' section")
    if not any(section in text for section in CHANGE_SECTIONS):
        found.append("the notes have none of: " + ", ".join(CHANGE_SECTIONS))
    if len(text.strip()) < MIN_CHARS:
        found.append(f"the notes are {len(text.strip())} characters; "
                     f"a summary of the changes needs at least {MIN_CHARS}")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version")
    arguments = parser.parse_args(argv)
    found = problems(arguments.version)
    for problem in found:
        print(f"::error::Release notes for {arguments.version}: {problem}",
              file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
