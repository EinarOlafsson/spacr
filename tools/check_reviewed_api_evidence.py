#!/usr/bin/env python3
"""Report reviewed API records that no longer match their docstring.

WHY THIS IS A SCRIPT AND NOT A TEST. It is both, in effect -- the reviewed
path already raises on a stale record -- but it raises on the FIRST one, as a
``ValueError`` from inside `reviewed_api_block_translations`, and that takes
the entire reviewed-API path down for the locale it happened to reach. Someone
editing docstrings wants the whole list before they push, and wants it without
running the localization suite.

WHAT KNOCKS A RECORD LOOSE. Editing the prose of a symbol that carries reviewed
translations. The record is bound to a block's exact text, its SHA-256, and its
index within the docstring, so rewording a sentence, adding one above it, or
removing one all break the binding.

WHAT THE TWO OUTCOMES MEAN, and the distinction is the point of the script:

  MOVED    the recorded sentence is still in the docstring, at a different
           index. The translation is still good and the record should be
           RE-BOUND by editing its label, not retired. Retiring it throws away
           reviewed work for a sentence nobody changed.
  STALE    the recorded sentence is gone or rewritten. The translation stands
           for wording the program no longer uses, so the record is retired in
           place with ``"retired": "<date>"`` and a ``retired_reason``.

Exit status is 1 when anything is reported, so it can gate a push.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def main() -> int:
    """Print every loose reviewed API record; return 1 if there are any."""
    import build_documentation_i18n as api

    docs = api.public_docstrings()
    moved: list[str] = []
    stale: list[str] = []
    for path in sorted((ROOT / "docs" / "i18n" / "reviewed" / "api").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            if not isinstance(record, dict) or record.get("retired"):
                continue
            label = str(record.get("label", ""))
            symbol, separator, raw_index = label.rpartition("#")
            if not separator or symbol not in docs:
                continue
            try:
                index = int(raw_index)
            except ValueError:
                stale.append(f"  {path.parent.name}/{path.name}  {label}  (bad index)")
                continue
            blocks, _layout = api.translatable_blocks(docs[symbol])
            source = str(record.get("source", ""))
            if 0 <= index < len(blocks) and blocks[index] == source:
                if record.get("source_sha256") == api._source_hash(source):
                    continue
                stale.append(f"  {path.parent.name}/{path.name}  {label}  (hash)")
                continue
            where = [n for n, block in enumerate(blocks) if block == source]
            if where:
                moved.append(
                    f"  {path.parent.name}/{path.name}  {label}"
                    f"  -> re-bind to #{where[0]}"
                )
            else:
                stale.append(
                    f"  {path.parent.name}/{path.name}  {label}"
                    f"  (sentence gone; {len(blocks)} blocks now)"
                )
    if moved:
        print(f"MOVED -- re-bind, do not retire ({len(moved)}):")
        print("\n".join(moved))
    if stale:
        print(f"STALE -- retire in place with a reason ({len(stale)}):")
        print("\n".join(stale))
    if not moved and not stale:
        print("every reviewed API record still matches its docstring")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
