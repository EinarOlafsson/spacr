"""Inspect the workspace saved with a run without launching the GUI.

A run can contain ``workspace.json`` beside its manifest. The document records
the databases and panels that were open, their view state, and the files they
reference. ``spacr-workspace`` prints that information from a terminal.

Usage::

    spacr-workspace ~/.spacr/runs/2026-08-19_143507_ab12cd34__regression
    spacr-workspace <run-folder> --files          # only the file inventory
    spacr-workspace <run-folder> --json           # the document itself

This is useful for checking which database a shared run used and whether the
referenced files still match, including on machines without a display.

Exit codes:
  0  — a workspace was found and printed
  2  — the folder does not exist, or carries no workspace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .workspace import DOC_NAME, check_files, inventory_text, load


def main(argv=None) -> int:
    """Print a saved workspace and return a process exit code.

    :param argv: command-line arguments without the program name. ``None``
        reads :data:`sys.argv`.
    :returns: ``0`` when a workspace was printed, or ``2`` when the requested
        path or workspace document does not exist.
    """
    parser = argparse.ArgumentParser(
        prog="spacr-workspace",
        description="Print what a saved run had open.")
    parser.add_argument("run_dir", help="the run folder, or its workspace.json")
    parser.add_argument("--files", action="store_true",
                        help="print only the file inventory and its state")
    parser.add_argument("--json", action="store_true",
                        help="print the document itself, unchanged")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.exists():
        print(f"no such run folder: {run_dir}", file=sys.stderr)
        return 2

    doc = load(run_dir)
    if doc is None:
        # NAMED, not "not found". A run saved with the feature off is a
        # different thing from a run whose bundle failed to write, and the
        # user's next step differs.
        print(f"{run_dir} carries no {DOC_NAME} — it was saved with "
              f"save_workspace='off', or predates workspace bundles.",
              file=sys.stderr)
        return 2

    root = run_dir if run_dir.is_dir() else run_dir.parent
    if args.json:
        print(json.dumps(doc, indent=2))
        return 0
    if args.files:
        for entry in check_files(doc, run_dir=root):
            note = entry.get("skipped")
            print(f"{entry['state']:<8} {entry.get('role', ''):<28} "
                  f"{entry['path']}" + (f"  [{note}]" if note else ""))
        return 0
    print(inventory_text(doc, run_dir=root))
    return 0


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
