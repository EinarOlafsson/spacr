#!/usr/bin/env python3
"""Run one small spaCR pipeline inside a freshly built container image.

WHY THIS FILE EXISTS. ``spacr --version`` proves that the console script is on
``PATH`` and that the package metadata resolved; it proves nothing about the
scientific stack the image was built for. An image whose ``torch`` wheel does
not match its ``numpy``, or whose ``tifffile`` cannot read a plain TIFF, still
answers ``--version`` correctly and fails an hour into the user's first real
run. So the release workflow runs this too, before anything is pushed.

WHAT IT RUNS, and why that module. ``external_masks`` is the only pipeline
stage that is a genuine end-to-end run with **no model download, no GPU and no
network**: it takes images and label masks that already exist, builds the
merged stacks, and hands them to Measure, which writes
``measurements/measurements.db``. So a green run here has exercised
``tifffile`` reading, ``numpy`` stacking, ``skimage`` region measurement,
``pandas`` assembly and the SQLite writer -- the whole headless spine -- on a
32x32 field that finishes in seconds.

The synthetic field is two intensity channels and two label masks, and the
size filters are switched off for it. spaCR's Measure defaults are the
maintainer's own 40x screening defaults (cell 8000 px2, nucleus 2000), and a
few hundred synthetic pixels would be correctly erased by them. That is a
property of this fixture, not of the pipeline.

Usage::

    python3 /opt/spacr/smoke_pipeline.py                 # temporary workspace
    python3 /opt/spacr/smoke_pipeline.py --workspace DIR  # keep the output

Exit status is 0 only when every check passed. Every check prints one line.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

#: Tables Measure must have written for the run to count as finished. A
#: database that exists but holds no object table is the failure mode this
#: check is aimed at: the run "completed" and measured nothing.
REQUIRED_TABLES = ("cell", "nucleus")

#: Seconds the pipeline is allowed before it is treated as hung. The real run
#: takes a couple of seconds on one 32x32 field; the ceiling is generous
#: because a cold container imports torch, cellpose and skimage first.
RUN_TIMEOUT_SECONDS = 900


def _say(ok: bool, message: str) -> bool:
    """Print one check line and return the result unchanged."""
    print(f"{'PASS' if ok else 'FAIL'}  {message}", flush=True)
    return ok


def write_field(root: Path) -> tuple[Path, Path, Path]:
    """Write one synthetic two-channel field and its two label masks.

    :param root: directory to write ``images/``, ``cell_masks/`` and
        ``nucleus_masks/`` into.
    :returns: the three folders, in that order.
    """
    import numpy as np
    import tifffile

    shape = (32, 32)
    yy, xx = np.indices(shape)
    images = root / "images"
    cells = root / "cell_masks"
    nuclei = root / "nucleus_masks"
    for folder in (images, cells, nuclei):
        folder.mkdir(parents=True, exist_ok=True)

    def emit(path: Path, array) -> None:
        tifffile.imwrite(str(path), np.asarray(array), photometric="minisblack")

    emit(images / "fov001_C1.tif", (yy * 32 + xx).astype(np.uint16))
    emit(images / "fov001_C2.tif", ((xx * 17 + yy * 3) % 4096).astype(np.uint16))

    cell_mask = np.zeros(shape, dtype=np.uint16)
    cell_mask[3:29, 3:29] = 1
    emit(cells / "fov001_cell_mask.tif", cell_mask)

    nucleus_mask = np.zeros(shape, dtype=np.uint16)
    nucleus_mask[10:20, 11:21] = 1
    emit(nuclei / "fov001_nucleus_mask.tif", nucleus_mask)
    return images, cells, nuclei


def write_settings(path: Path, folders: tuple[Path, Path, Path],
                   destination: Path) -> None:
    """Write the settings file ``spacr-run`` will be given.

    JSON rather than CSV on purpose: ``inputs`` is a list of paths, and a
    settings CSV round trip is exactly where a list becomes a string.
    """
    settings = {
        "inputs": [str(folder) for folder in folders],
        "dst": str(destination),
        "layout": "flat",
        "crop_mode": ["cell"],
        "cell_min_size": 0,
        "nucleus_min_size": 0,
        "pathogen_min_size": 0,
        "n_jobs": 1,
        "plot": False,
        "verbose": True,
    }
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def run_pipeline(settings: Path, workspace: Path) -> tuple[bool, str]:
    """Invoke ``spacr-run external_masks`` and return success plus its output.

    The console script is looked up on ``PATH`` rather than called as
    ``python -m``: an image that installed the package but not its entry
    points is a real and silent packaging failure.
    """
    executable = shutil.which("spacr-run")
    if executable is None:
        return False, "spacr-run is not on PATH"
    environment = dict(os.environ)
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", str(workspace / "mpl"))
    environment.pop("DISPLAY", None)
    try:
        completed = subprocess.run(
            [executable, "external_masks", "--settings", str(settings)],
            cwd=str(workspace), env=environment, text=True,
            capture_output=True, timeout=RUN_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        return False, f"spacr-run did not finish in {RUN_TIMEOUT_SECONDS}s"
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        return False, f"spacr-run exited {completed.returncode}\n{output}"
    return True, output


def tables_in(database: Path) -> list[str]:
    """Return the table names of a SQLite database, or an empty list."""
    if not database.is_file():
        return []
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return sorted(str(row[0]) for row in rows)


def rows_in(database: Path, table: str) -> int:
    """Return how many rows ``table`` holds."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        return int(connection.execute(
            f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def main(argv: list[str] | None = None) -> int:
    """Run every check and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workspace", default=None,
        help="directory to work in; a temporary one is used and removed "
             "when this is not given.")
    arguments = parser.parse_args(argv)

    temporary = arguments.workspace is None
    workspace = Path(
        tempfile.mkdtemp(prefix="spacr-smoke-") if temporary
        else arguments.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"workspace: {workspace}", flush=True)

    checks: list[bool] = []
    try:
        from spacr.version import get_version

        version = get_version()
        checks.append(_say(
            version not in ("", "unknown"),
            f"package metadata resolves the version: {version}"))

        folders = write_field(workspace / "input")
        settings = workspace / "settings.json"
        destination = workspace / "project"
        write_settings(settings, folders, destination)
        checks.append(_say(True, "synthetic field and label masks written"))

        ran, output = run_pipeline(settings, workspace)
        checks.append(_say(ran, "spacr-run external_masks finished"))
        if not ran:
            print(output, file=sys.stderr, flush=True)
            return 1

        merged = sorted((destination / "merged").glob("*.npy"))
        checks.append(_say(
            bool(merged), f"merged stacks written: {len(merged)}"))

        database = destination / "measurements" / "measurements.db"
        present = tables_in(database)
        checks.append(_say(
            bool(present), f"measurements.db written with tables: {present}"))

        for table in REQUIRED_TABLES:
            if table not in present:
                checks.append(_say(False, f"table {table!r} is missing"))
                continue
            count = rows_in(database, table)
            checks.append(_say(
                count > 0, f"table {table!r} holds {count} measured object(s)"))
    finally:
        if temporary:
            shutil.rmtree(workspace, ignore_errors=True)

    failed = checks.count(False)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
