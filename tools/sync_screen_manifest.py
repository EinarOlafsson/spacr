#!/usr/bin/env python
"""Set `spacr.screen_data.SCREEN_ASSETS` sizes from what is published.

The picker states a size before a download is paid for, so the number has to
be the ARCHIVE's real one -- a figure taken from the folder it was made from
is out by however much tar added, and a hand-typed one is out by whatever was
mistyped. Run this after publishing or replacing any piece.

    python tools/sync_screen_manifest.py [--check]

``--check`` reports drift and exits non-zero instead of writing, which is what
CI would run.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MANIFEST = REPO_ROOT / "spacr" / "screen_data.py"


def published_sizes(repo: str, names) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    found = api.get_paths_info(repo, list(names), repo_type="dataset")
    return {item.path: int(item.size) for item in found
            if getattr(item, "size", None)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report drift without writing")
    args = parser.parse_args()

    from spacr.screen_data import SCREEN_ASSETS, SCREEN_REPO, human_size

    sizes = published_sizes(SCREEN_REPO, [a.archive for a in SCREEN_ASSETS])
    text = MANIFEST.read_text(encoding="utf-8")
    drifted, absent = [], []
    for asset in SCREEN_ASSETS:
        real = sizes.get(asset.archive)
        if real is None:
            absent.append(asset.archive)
            continue
        if real == asset.bytes:
            continue
        drifted.append((asset.archive, asset.bytes, real))
        # Only the number on this asset's own line, found through its archive
        # name, so two assets of the same size cannot be confused.
        #
        # ANY digit run, not the value spelled out: Python literals here carry
        # underscore separators (493_000_000), so matching str(asset.bytes)
        # found nothing and left one entry stale while reporting it changed.
        pattern = re.compile(
            r'(ScreenAsset\("' + re.escape(asset.archive) +
            r'",\s*\d+,\s*"[a-z]+",\s*)[\d_]+', re.S)
        text, count = pattern.subn(rf"\g<1>{real}", text, count=1)
        if not count:
            print(f"  could not rewrite {asset.archive}", file=sys.stderr)

    for name in absent:
        print(f"  NOT PUBLISHED YET: {name}")
    for name, was, now in drifted:
        print(f"  {name}: {human_size(was)} -> {human_size(now)}")
    if not drifted:
        print("every published size already matches the manifest.")
        return 0
    if args.check:
        print(f"{len(drifted)} size(s) drifted; run without --check to fix.")
        return 1
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"updated {len(drifted)} size(s) in {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
