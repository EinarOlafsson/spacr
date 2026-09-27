#!/usr/bin/env python3
"""Run spaCR's Mask module on an offline bundle's test data.

Shipped inside every offline bundle and run by the installer's
``--check-mask`` option, or by hand after an install::

    <install root>/venv/bin/python offline_mask_check.py --data <install root>/test_data/plate1

The test images are copied to a scratch folder first, because Mask
reorganises the folder it is pointed at, so the installed copy stays
reusable. A model name in the settings that points at a model this bundle
does not carry (``<models>/...``) is replaced by ``--fallback-model``.

Exit 0 when Mask ran and wrote masks, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def model_overrides(settings: Path, fallback: str) -> list[str]:
    """``--set`` overrides for model names that cannot resolve offline."""
    overrides = []
    with open(settings, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0].endswith("_model_name") and "<" in row[1]:
                overrides.append(f"{row[0]}={fallback}")
    return overrides


def main(argv=None) -> int:
    """Copy the test plate, run ``spacr-run mask`` on it, check its masks."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--fallback-model", default="cpsam")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the scratch copy and its masks.")
    args = parser.parse_args(argv)

    settings = args.data / "settings" / "gen_mask_settings.csv"
    if not settings.is_file():
        print(f"no Mask settings at {settings}", file=sys.stderr)
        return 1
    work = Path(tempfile.mkdtemp(prefix="spacr-offline-mask-", dir=args.work))
    try:
        src = work / "plate1"
        shutil.copytree(args.data, src)
        command = [sys.executable, "-m", "spacr.cli", "mask",
                   "--settings", str(src / "settings" / "gen_mask_settings.csv"),
                   "--set", f"src={src}", "--no-hash-inputs",
                   *[part for item in model_overrides(settings, args.fallback_model)
                     for part in ("--set", item)]]
        print("+", " ".join(command), flush=True)
        code = subprocess.run(command).returncode
        masks = [p for p in src.rglob("*") if p.is_file() and "mask" in p.as_posix().lower()
                 and p.suffix == ".npy"]
        merged = list(src.rglob("merged/*.npy"))
        print(f"Mask exit code {code}; {len(masks)} mask arrays, "
              f"{len(merged)} merged stacks under {src}")
        return 0 if code == 0 and masks and merged else 1
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
