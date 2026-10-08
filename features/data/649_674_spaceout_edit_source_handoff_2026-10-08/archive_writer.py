"""Freeze the tested private UI source for the workstation artifact refresh."""

import gzip
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
BASE = "e31e19543a"
CHECKPOINT = "0527ae6fa0"


def git(*args):
    """Read exact immutable objects from the shared repository."""
    return subprocess.check_output(["git", *args], cwd=ROOT)


def digest(data):
    """Hash source and evidence bytes before writing their compressed payload."""
    return hashlib.sha256(data).hexdigest()


def payload(name, data):
    """Write reproducible gzip bytes and return both integrity checks."""
    packed = gzip.compress(data, mtime=0)
    (OUT / name).write_bytes(packed)
    return {"payload": name, "sha256": digest(packed), "raw_sha256": digest(data)}


def main():
    """Archive the exact source patch, changed files and scoped test logs."""
    baseline = git("rev-parse", BASE).decode().strip()
    checkpoint = git("rev-parse", CHECKPOINT).decode().strip()
    names = git("diff", "--name-only", baseline, checkpoint, "--", "spacr", "tests").decode().splitlines()
    sources = {}
    for index, name in enumerate(names):
        row = payload(f"source-{index:02}.gz", git("show", f"{checkpoint}:{name}"))
        sources[name] = row
    patch = payload("source-and-tests.patch.gz", git("diff", "--binary", baseline, checkpoint, "--", "spacr", "tests"))
    phases = []
    for filename, count, tested, scope in (
        ("home-ui-spaceout-undo-hue-final-20261008.log", 212, checkpoint, "Fourteen selected CPU/Qt files: final Spaceout phenomena, actual mouse/Preferences wiring, Edit Undo/Redo and transferred UI ripple/Gate/window/help/setup regressions."),
    ):
        data = (pathlib.Path("/mnt/wd4tb/scratch") / filename).read_bytes()
        assert f"{count} passed".encode() in data
        phases.append({"passed": count, "tested_checkpoint": git("rev-parse", tested).decode().strip(), "scope": scope, **payload(filename + ".gz", data)})
    inventory_data = pathlib.Path("/mnt/wd4tb/scratch/ui-integration-20261008/final-0527-diff.json").read_bytes()
    inventory_row = payload("normal-inventory-diff.json.gz", inventory_data)
    inventory_row.update({"api_baseline": 13215, "api_current": 13221,
                          "ui_baseline": 7245, "ui_current": 7261,
                          "combined_workstation_source_must_be_remeasured": True})
    receipt = {
        "schema": 1,
        "baseline": baseline,
        "checkpoint": checkpoint,
        "items": ["N649", "N674"],
        "scope": "Incremental private source handoff after the accepted e31 N666-671 transfer. Includes N649 Edit menu and N674 Spaceout field default/effects/controls, with a real float32 hue-index boundary repair. App publication and final normal catalogs remain pending workstation refresh.",
        "environment": {"python": "/home/olafsson/anaconda3/envs/spacr/bin/python", "PySide6": "6.11.2", "pytest": "8.4.2 overlay", "cap": "4G", "CUDA_VISIBLE_DEVICES": "", "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg", "xdist": False},
        "sources": sources,
        "patch": patch,
        "inventory_diff": inventory_row,
        "phases": phases,
        "limits": ["Phase totals overlap and must not be summed as unique tests.", "No native compositor, GPU, full serial or human appearance acceptance.", "Existing required CI green belongs to exact 07809d9264; protected fresh53ab run has a guide-browser timeout and does not test this source.", "N672/N673 are registered open work and are not implemented in this patch.", "No test ceiling, correctness assertion, workflow timeout or memory cap has been raised."],
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Archived {len(sources)} source bindings and {len(phases)} scoped logs at {checkpoint}")


if __name__ == "__main__":
    main()
