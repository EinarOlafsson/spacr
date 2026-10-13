"""Freeze the tested private UI source for the workstation artifact refresh."""

import gzip
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
BASE = "23d939816a"
CHECKPOINT = "e31e19543a"


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
        ("n666-671-final-cohort-20261008.log", 313, checkpoint, "Final combined 12-file CPU/Qt interaction cohort; includes setup, ripples, window states, footer, settings and Gate rendering."),
        ("gate-rounded-interaction-final-20261008.log", 162, "b5eb568b65", "Gate drawing, 3D rotation/dragging, mouse and menus; same Gate source as final checkpoint."),
        ("ripples-gravity-switch-final-20261008.log", 119, "b5eb568b65", "Gravity, grab and switch behavior before the discrete-popup follow-up; overlaps final cohort."),
        ("ui-final-docguards-20261008.log", 3, "4f7ba6db89", "Callable and nested-helper documentation guards before final popup prose and Gate painter follow-up; historical scope only."),
    ):
        data = (pathlib.Path("/mnt/wd4tb/scratch") / filename).read_bytes()
        assert f"{count} passed".encode() in data
        phases.append({"passed": count, "tested_checkpoint": git("rev-parse", tested).decode().strip(), "scope": scope, **payload(filename + ".gz", data)})
    receipt = {
        "schema": 1,
        "baseline": baseline,
        "checkpoint": checkpoint,
        "items": ["N666", "N667", "N668", "N670", "N671"],
        "scope": "Private source handoff; app source has not been published. Final normal catalogs and exact inventory guards are pending workstation refresh. N674 is a subsequent in-progress change, not included here.",
        "environment": {"python": "/home/olafsson/anaconda3/envs/spacr/bin/python", "PySide6": "6.11.2", "pytest": "8.4.2 overlay", "cap": "4G", "CUDA_VISIBLE_DEVICES": "", "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg", "xdist": False},
        "sources": sources,
        "patch": patch,
        "phases": phases,
        "limits": ["Phase totals overlap and must not be summed as unique tests.", "No native compositor, GPU, full serial or human appearance acceptance.", "Existing required CI green belongs to exact 07809d9264; fresh protected 53ab run does not test this source.", "N672/N673 are registered open work and are not implemented in this patch.", "No test ceiling, correctness assertion, workflow timeout or memory cap has been raised."],
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Archived {len(sources)} source bindings and {len(phases)} scoped logs at {checkpoint}")


if __name__ == "__main__":
    main()
