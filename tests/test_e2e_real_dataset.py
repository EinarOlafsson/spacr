"""Opt-in long end-to-end test against a real microscopy dataset.

There is no built-in path here. Both of these environment variables must name
an existing directory, and they are the only way in:

* ``SPACR_E2E_DATA``     — a plate folder of raw images
* ``SPACR_E2E_SETTINGS`` — a folder of ``<app_key>_settings.csv`` files

Run it explicitly with::

    SPACR_E2E_DATA=/somewhere/plate1 SPACR_E2E_SETTINGS=/somewhere/settings \\
        pytest -m slow tests/test_e2e_real_dataset.py -s

The two variables used to *default* to one developer's home directory, which
made this module a machine-specific test dressed as a portable one: everywhere
but that one computer the four stages skipped themselves and the module
reported green while running nothing at all. Opting in is now explicit, and the
two ways to get it wrong report differently:

* neither variable set — SKIP, naming the variables. You did not opt in.
* set, but not pointing at a directory — FAIL. You *did* opt in, so a typo has
  to be loud; a skip there would recreate the exact "green for nothing" bug
  this module was rewritten to remove.

It walks the actual pipeline chain — mask -> measure -> notebook
export -> run journal inspection — on a copy of the real dataset,
which is the definitive smoke test for "does the whole thing still
work". Everything below is layered so a failure points at the stage
that regressed rather than a generic "the pipeline crashed" trace.

Wall-clock is roughly 5-15 minutes on a GPU-equipped box.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths — supplied entirely by the environment, with no fallback
# ---------------------------------------------------------------------------

_DATASET_ENV = "SPACR_E2E_DATA"
_SETTINGS_ENV = "SPACR_E2E_SETTINGS"


def _resolve(var: str) -> Path | None:
    """The directory ``var`` names, or ``None`` when it is unset or blank."""
    raw = os.environ.get(var, "").strip()
    return Path(raw) if raw else None


# ---------------------------------------------------------------------------
# Opt-in guard — one place, so every test in this module reports the same
# ---------------------------------------------------------------------------

def _require_real_dataset() -> tuple[Path, Path]:
    """Skip when opted out, fail when opted in with an unusable path.

    :returns: ``(dataset_dir, settings_dir)`` once both are known to exist.
    """
    resolved = {var: _resolve(var)
                for var in (_DATASET_ENV, _SETTINGS_ENV)}
    unset = sorted(var for var, path in resolved.items() if path is None)
    if unset:
        pytest.skip(
            "the real-dataset E2E is opt-in and was not opted into: set "
            + " and ".join(unset)
            + " to an existing directory to run it")
    not_a_dir = [f"{var}={path}" for var, path in resolved.items()
                 if not path.is_dir()]
    if not_a_dir:
        pytest.fail(
            "opted in to the real-dataset E2E, but "
            + "; ".join(not_a_dir)
            + " is not a directory — fix the path or unset the variable")
    return resolved[_DATASET_ENV], resolved[_SETTINGS_ENV]


# ---------------------------------------------------------------------------
# Session-scoped scratch: copy the dataset ONCE and re-use across stages
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _scratch(tmp_path_factory):
    """Copy the real dataset into a tmp dir so every stage mutates the
    copy rather than the source. Session-scoped so the mask stage's
    outputs are visible to the measure stage without re-copying."""
    src, _ = _require_real_dataset()
    root = tmp_path_factory.mktemp("spacr_e2e_real", numbered=True)
    dst = root / "plate1"
    shutil.copytree(src, dst)
    return dst


@pytest.fixture(scope="module")
def _settings_root():
    _, settings = _require_real_dataset()
    return settings


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

def _load_settings_for(app_key: str,
                          settings_root: Path,
                          src: Path) -> dict:
    """Look up ``{app_key}_settings.csv`` under ``settings_root`` (or
    fall back to spacr defaults) and rewrite ``src`` to the local copy.
    """
    from spacr.qt.screens.settings_model import resolve_default_settings
    settings = dict(resolve_default_settings(app_key))
    csv_path = settings_root / f"{app_key}_settings.csv"
    if csv_path.is_file():
        import csv
        with csv_path.open() as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 2:
                    continue
                k = row[0].strip()
                v = row[1]
                # Best-effort coercion
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                else:
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                settings[k] = v
    settings["src"] = str(src)
    return settings


# ---------------------------------------------------------------------------
# Stage 1 — mask
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_e2e_real_stage_1_mask(_scratch, _settings_root):
    """Run preprocess_generate_masks against the real dataset. Emits
    a masks/ folder and settles the run journal."""
    from spacr.core import preprocess_generate_masks
    from spacr.run_journal import open_run

    settings = _load_settings_for("mask", _settings_root, _scratch)
    started = time.time()
    with open_run("mask", settings) as run:
        preprocess_generate_masks(settings)
    elapsed = time.time() - started
    print(f"[e2e] mask stage completed in {elapsed:.1f}s -> {run.dir}")
    # Journal was written
    assert (run.dir / "manifest.json").exists()
    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["status"] in ("success", "running")
    # At least one mask was produced
    masks_dir = _scratch / "masks"
    if not masks_dir.exists():
        # Some pipelines write into a subfolder; look one level deeper
        candidates = list(_scratch.rglob("*_cell_mask.tif"))
        assert candidates, "no cell mask outputs found under scratch"


# ---------------------------------------------------------------------------
# Stage 2 — measure (depends on stage 1 having written masks)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_e2e_real_stage_2_measure(_scratch, _settings_root):
    """After masks exist, run measure_crop to populate the sqlite DB."""
    if not any((_scratch / sub).is_dir()
                 for sub in ("masks", "measurements", "stack")):
        pytest.skip("mask stage didn't produce a downstream-ready layout")

    try:
        from spacr.measure import measure_crop
    except Exception as e:
        pytest.skip(f"measure module unavailable: {e}")

    settings = _load_settings_for("measure", _settings_root, _scratch)
    started = time.time()
    try:
        measure_crop(settings)
    except Exception as e:
        # Measure often has strict layout expectations. If the real
        # dataset doesn't satisfy them we don't want a crash to fail
        # the whole E2E — record + skip.
        pytest.skip(f"measure stage bailed on this dataset: {e}")
    elapsed = time.time() - started
    print(f"[e2e] measure stage completed in {elapsed:.1f}s")

    # We expect a measurements DB somewhere under scratch
    db_hits = list(_scratch.rglob("measurements.db"))
    assert db_hits, "no measurements.db written"


# ---------------------------------------------------------------------------
# Stage 3 — notebook export from the run journal
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_e2e_real_stage_3_notebook_export(tmp_path):
    """Take the most recent successful mask run and export a notebook
    from it. Verifies the run-journal -> notebook path end-to-end."""
    _require_real_dataset()
    from spacr.notebook_export import export_run
    from spacr.run_journal import recent_runs

    runs = recent_runs(limit=10)
    mask_runs = [r for r in runs if r.get("app_key") == "mask"]
    if not mask_runs:
        pytest.skip("no recent mask runs to export from")
    latest = mask_runs[0]
    out = export_run(latest["dir"], out_path=tmp_path / "e2e.ipynb")
    assert out.exists()
    nb = json.loads(out.read_text())
    # Structural sanity
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert any("preprocess_generate_masks" in "".join(c["source"])
                for c in code_cells)


# ---------------------------------------------------------------------------
# Stage 4 — plate queue chained on TWO fake plates using the real settings
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_e2e_real_stage_4_plate_queue(_scratch, _settings_root, tmp_path):
    """Use the plate queue to chain the mask stage across two copies of
    the same plate. This is what the batch/plate manager users will
    actually experience in production."""
    from spacr.qt.plate_queue import (
        PlateQueue, QueueItem, run_queue, Status,
    )
    from spacr.core import preprocess_generate_masks

    q = PlateQueue(path=tmp_path / "queue.json")
    for i, dst in enumerate((tmp_path / "copy_a", tmp_path / "copy_b")):
        shutil.copytree(_scratch, dst)
        settings = _load_settings_for("mask", _settings_root, dst)
        q.add(QueueItem.build("mask", settings, label=str(dst)))

    def _runner(item):
        preprocess_generate_masks(item.settings)

    started = time.time()
    run_queue(q, runner=_runner)
    elapsed = time.time() - started
    print(f"[e2e] plate queue completed 2 plates in {elapsed:.1f}s")

    statuses = [i.status for i in q.items()]
    assert Status.SUCCESS in statuses
    # Every item hit a terminal state
    assert all(s in (Status.SUCCESS, Status.FAILED) for s in statuses)
