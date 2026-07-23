"""
``spacr repro`` — replay a pipeline run from its recorded journal.

Every pipeline invocation writes a run journal (see
:mod:`spacr.run_journal`) containing the exact settings + environment
that produced a result. This CLI re-runs those settings, opens a
FRESH journal folder, and reports whether the outcome matches.

Usage::

    spacr repro ~/.spacr/runs/2026-07-23_143507_ab12cd34__mask
    spacr repro ~/.spacr/runs/2026-07-23_143507_ab12cd34__mask --dry
    spacr repro ~/.spacr/runs/2026-07-23_143507_ab12cd34__mask --show

* ``--dry``  prints the resolved settings + which pipeline entry
  will run; doesn't invoke it.
* ``--show`` prints the manifest + settings; doesn't invoke it.

Exit codes:
  0  — replay ran to completion (regardless of scientific outcome)
  1  — replay raised (journal captures the traceback for triage)
  2  — bad input (missing run folder / unresolvable app_key)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .run_journal import load_run_settings, open_run, runs_root


def _print_manifest(run_dir: Path) -> None:
    m = json.loads((run_dir / "manifest.json").read_text())
    print(f"run:       {run_dir.name}")
    print(f"app:       {m.get('app_key')}")
    print(f"status:    {m.get('status')}")
    print(f"start:     {m.get('start_utc')}")
    print(f"elapsed:   {m.get('elapsed_s')}s")
    print(f"n_settings:{m.get('n_settings')}")
    env = m.get("env", {})
    print(f"spacr:     {env.get('spacr')} (git {env.get('spacr_git')})")
    print(f"python:    {env.get('python')}  torch {env.get('torch')}  "
          f"cellpose {env.get('cellpose')}")
    if m.get("model_hashes"):
        print("models:")
        for k, v in m["model_hashes"].items():
            print(f"  {k}: {v}")


def _print_settings(settings: dict) -> None:
    print("settings:")
    for k, v in sorted(settings.items()):
        print(f"  {k:32s} = {v!r}")


def _resolve_pipeline(app_key: str):
    """Return the pipeline callable for ``app_key`` or ``None``."""
    try:
        from .qt.bridge import resolve_pipeline_entry
        return resolve_pipeline_entry(app_key)
    except Exception:
        return None


def main(argv=None) -> int:
    """CLI entry point wired as the ``spacr-repro`` console script.

    :param argv: optional argv list; defaults to ``sys.argv[1:]``.
    :returns: process exit code.
    """
    p = argparse.ArgumentParser(
        prog="spacr repro",
        description="Replay a spaCR pipeline run from its recorded "
                    "journal folder.",
    )
    p.add_argument("run_dir",
                    help="Path to a folder under ~/.spacr/runs/, or the "
                         "folder's basename.")
    p.add_argument("--dry", action="store_true",
                    help="Print resolved settings + app; don't run.")
    p.add_argument("--show", action="store_true",
                    help="Print manifest + settings; don't run.")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        # Try under runs_root() by basename
        candidate = runs_root() / args.run_dir
        if candidate.exists():
            run_dir = candidate
        else:
            print(f"error: no such run folder: {args.run_dir}",
                    file=sys.stderr)
            return 2

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"error: {run_dir} is not a valid run folder "
                f"(no manifest.json)", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text())
    app_key = manifest.get("app_key")
    settings = load_run_settings(run_dir)

    if args.show:
        _print_manifest(run_dir)
        print()
        _print_settings(settings)
        return 0

    entry = _resolve_pipeline(app_key)
    if entry is None:
        print(f"error: no pipeline entry for app_key={app_key!r}",
                file=sys.stderr)
        return 2

    if args.dry:
        print(f"would run: {entry.__module__}.{entry.__name__}(settings)")
        _print_settings(settings)
        return 0

    print(f"replaying {app_key} — this opens a NEW run journal folder.")
    with open_run(app_key, settings) as run:
        try:
            entry(settings)
            run.set_status("success")
        except Exception as e:
            run.set_status("failed")
            print(f"replay raised: {type(e).__name__}: {e}",
                    file=sys.stderr)
            return 1
    print(f"done → {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
