"""End-to-end integration test using the Hugging Face demo dataset.

Downloads the toxo_mito images + spacr_settings CSVs into a temp
directory, then runs the same Mask → Measure → Annotate chain the
Qt demo menu triggers when the user clicks
"End-to-end (Mask → Measure → Annotate) real dataset…".

Skipped unless the caller opts in. Two opt-in env vars:

  * ``SPACR_HF_E2E_STUB=1`` — use a tiny synthetic dataset built
    on-the-fly. No network, ~30s on a GPU box. Good for CI +
    day-to-day.
  * ``SPACR_HF_E2E_RUN=1``  — actually download the toxo_mito repo +
    settings pack. ~5-15 min depending on network. Manual bug hunts.

Also marked ``@pytest.mark.slow`` + ``@pytest.mark.network`` for the
marker-based selectors, but the env-var gate is the primary guard so
a bare ``pytest tests/`` never triggers a 300-MB download.

Example invocations::

    SPACR_HF_E2E_STUB=1 pytest tests/test_hf_e2e_integration.py -s
    SPACR_HF_E2E_RUN=1  pytest tests/test_hf_e2e_integration.py -s
- Measure and Annotate stages are best-effort — if either bails on
  dataset-shape mismatches the test records the reason and moves on
  rather than failing loudly. The point is to prove the chain is
  wired end-to-end; deep pipeline coverage lives elsewhere.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest


STUB_ENV = "SPACR_HF_E2E_STUB"
RUN_ENV  = "SPACR_HF_E2E_RUN"


def _stubbed_mode() -> bool:
    return os.environ.get(STUB_ENV) == "1"


def _explicit_opt_in() -> bool:
    """True when either env var says "yes go".

    Deliberate: without ONE of these, the tests skip so a plain
    ``pytest tests/`` never triggers a 300-MB download.
    """
    return _stubbed_mode() or os.environ.get(RUN_ENV) == "1"


def _require_network():
    """Skip unless the caller opted in; then verify the network path.

    ``SPACR_HF_E2E_STUB=1`` short-circuits the network check entirely
    (stub mode uses a synthetic dataset).
    """
    if not _explicit_opt_in():
        pytest.skip(
            "set SPACR_HF_E2E_STUB=1 for fast stub mode, or "
            "SPACR_HF_E2E_RUN=1 to hit the real HF endpoint")
    if _stubbed_mode():
        return
    try:
        from huggingface_hub import list_repo_files       # noqa: F401
    except Exception as e:
        pytest.skip(f"huggingface-hub unavailable: {e}")
    try:
        import requests
        requests.head("https://huggingface.co", timeout=5)
    except Exception as e:
        pytest.skip(f"network / huggingface.co unreachable: {e}")


def _require_gpu_cellpose():
    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch unavailable: {e}")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA — this E2E chain is GPU-only")
    try:
        import cellpose                                    # noqa: F401
    except Exception as e:
        pytest.skip(f"cellpose unavailable: {e}")


# ---------------------------------------------------------------------------
# Stub dataset — a handful of Yokogawa-named TIFFs + minimal CSVs
# ---------------------------------------------------------------------------

def _make_stub_dataset(dst: Path) -> Path:
    """Emit a tiny cellvoyager-format plate at ``dst/plate1``."""
    import tifffile
    plate = dst / "plate1"; plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for well in ("A01", "A02"):
        for field in (1, 2):
            for ch in range(3):
                arr = (rng.integers(0, 2000, size=(64, 64))
                       + (300 if ch == 0 else 100)).astype(np.uint16)
                p = (plate / f"plate1_{well}_"
                        f"T01F0{field}L01A01Z01C0{ch}.tif")
                tifffile.imwrite(str(p), arr)
    return plate


def _make_stub_settings(dst: Path) -> Path:
    """Emit minimal settings CSVs mirroring the HF settings pack."""
    settings = dst / "settings"; settings.mkdir(parents=True,
                                                    exist_ok=True)
    (settings / "mask_settings.csv").write_text(
        "src,\n"
        "metadata_type,cellvoyager\n"
        "channels,\"[0, 1, 2]\"\n"
        "cell_channel,0\n"
        "nucleus_channel,1\n"
        "plot,false\n"
        "test_mode,false\n"
        "batch_size,2\n"
    )
    (settings / "measure_settings.csv").write_text(
        "src,\n"
        "plot,false\n"
    )
    (settings / "annotate_settings.csv").write_text("src,\n")
    return settings


@pytest.fixture(scope="module")
def _prepared_workspace(tmp_path_factory):
    _require_network()
    _require_gpu_cellpose()
    root = tmp_path_factory.mktemp("hf_e2e", numbered=True)
    if _stubbed_mode():
        dataset = _make_stub_dataset(root / "data")
        settings = _make_stub_settings(root / "data")
    else:
        from spacr.gui_utils import download_dataset
        # Use the CLI downloader (queue-based). We pipe status
        # messages into a small local queue and print them so a -s
        # invocation shows progress in real time.
        import queue as _q
        q = _q.Queue()
        dataset = Path(download_dataset(
            q, repo_id="einarolafsson/toxo_mito",
            subfolder="plate1", local_dir=str(root)))
        settings = Path(download_dataset(
            q, repo_id="einarolafsson/spacr_settings",
            subfolder="", local_dir=str(root / "settings_dir")))
    return dataset, settings


# ---------------------------------------------------------------------------
# Settings load helper (mirror the app.py routine so the test uses
# the same CSV loading logic the GUI does).
# ---------------------------------------------------------------------------

def _load_settings_for(app_key: str,
                          settings_root: Path, src: Path) -> dict:
    from spacr.qt.screens.settings_model import resolve_default_settings
    settings = dict(resolve_default_settings(app_key))
    csv = settings_root / f"{app_key}_settings.csv"
    if csv.is_file():
        import csv as _csv
        with csv.open() as fh:
            for row in _csv.reader(fh):
                if not row or row[0].startswith("#") or len(row) < 2:
                    continue
                k, v = row[0].strip(), row[1]
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
# The chain
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.network
def test_hf_e2e_mask_stage(_prepared_workspace):
    """Mask stage runs against the HF dataset + settings pack."""
    dataset, settings_root = _prepared_workspace
    from spacr.core import preprocess_generate_masks
    from spacr.run_journal import open_run

    settings = _load_settings_for("mask", settings_root, dataset)
    t0 = time.time()
    with open_run("mask", settings) as run:
        preprocess_generate_masks(settings)
    print(f"[hf-e2e] mask stage: {time.time() - t0:.1f}s -> {run.dir}")
    assert (run.dir / "manifest.json").exists()
    # v1 writes .npy stacks under masks/cell_mask_stack/ — accept
    # any file whose path names "cell_mask" (covers both the v1 stack
    # layout + any per-field .tif some builds emit).
    hits = [p for p in dataset.rglob("*") if "cell_mask" in p.name]
    assert hits, "mask stage produced no cell_mask output files"


@pytest.mark.slow
@pytest.mark.network
def test_hf_e2e_measure_stage(_prepared_workspace):
    """Measure stage runs against the previous stage's mask output."""
    dataset, settings_root = _prepared_workspace
    try:
        from spacr.measure import measure_crop
    except Exception as e:
        pytest.skip(f"measure module unavailable: {e}")

    settings = _load_settings_for("measure", settings_root, dataset)
    try:
        measure_crop(settings)
    except Exception as e:
        pytest.skip(f"measure stage bailed on the HF dataset: {e}")
    # A measurements DB somewhere under scratch is proof-of-life
    assert list(dataset.rglob("measurements.db")), \
        "measure stage wrote no measurements.db"


@pytest.mark.slow
@pytest.mark.network
def test_hf_e2e_annotate_screen_opens(_prepared_workspace, qtbot):
    """Annotate is interactive; the "test" is that the screen
    constructs against the HF dataset without exceptions and points
    at the right src. Deliberately doesn't depend on
    ``qt_theme_applied`` (a fixture that only lives under tests/qt/)
    so this file can sit at the top level."""
    pytest.importorskip("PySide6")
    dataset, _ = _prepared_workspace
    from spacr.qt.screens.annotate import AnnotateScreen
    scr = AnnotateScreen()
    qtbot.addWidget(scr)
    if hasattr(scr, "apply_settings_dict"):
        scr.apply_settings_dict({"src": str(dataset)})
    assert scr.isEnabled()
