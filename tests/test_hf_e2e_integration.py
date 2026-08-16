"""End-to-end integration test using the Hugging Face demo dataset.

Downloads the toxo_mito images + spacr_settings CSVs into a temp
directory, then runs the same Mask → Measure → Annotate chain the
Qt demo menu triggers when the user clicks
"End-to-end (Mask → Measure → Annotate) real dataset…".

The suite probes CUDA, Cellpose and Hugging Face automatically. It runs when
all three are available and skips cleanly otherwise.

``SPACR_HF_E2E_STUB=1`` remains as a mode selector: it substitutes a tiny
synthetic dataset and therefore does not require network access. Without it,
the real toxo_mito dataset and settings pack are downloaded.

The tests remain marked ``@pytest.mark.slow`` + ``@pytest.mark.network`` for
reporting and targeted selection.

Example invocations::

    SPACR_HF_E2E_STUB=1 pytest tests/test_hf_e2e_integration.py -s
    pytest tests/test_hf_e2e_integration.py -s
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


def _stubbed_mode() -> bool:
    return os.environ.get(STUB_ENV) == "1"


def _require_network():
    """Skip when the real dataset endpoint cannot be reached.

    ``SPACR_HF_E2E_STUB=1`` short-circuits the network check entirely
    (stub mode uses a synthetic dataset).
    """
    if _stubbed_mode():
        return
    from tests.resource_capabilities import (
        endpoint_available,
        package_available,
    )
    if not package_available("huggingface_hub"):
        pytest.skip("huggingface-hub unavailable")
    if not endpoint_available():
        pytest.skip("network / huggingface.co unreachable")


def _require_gpu_cellpose():
    from tests.resource_capabilities import (
        cuda_available,
        package_available,
    )
    # The real microscopy dataset remains GPU-only. The four-field stub is a
    # release-gate contract and is deliberately small enough for Cellpose CPU,
    # so a hosted runner can execute it rather than skip the only assertion
    # that proves masks survive the stage.
    if not _stubbed_mode() and not cuda_available():
        pytest.skip("no CUDA — this E2E chain is GPU-only")
    if not package_available("cellpose"):
        pytest.skip("cellpose unavailable")


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


#: What each stage's settings file is actually CALLED in the
#: ``einarolafsson/spacr_settings`` pack. This used to be derived as
#: ``f"{app_key}_settings.csv"``, and the pack has never shipped a file by
#: either of those names — so ``_load_settings_for`` found nothing, silently
#: fell back to ``resolve_default_settings``, and the mask stage refused with
#: "at least one of cell_channel / nucleus_channel / pathogen_channel /
#: organelle_channel must be set" because the defaults name no channel. A
#: whole E2E chain was running on defaults and nobody could see it, because
#: the lookup was a miss rather than an error. The names are asserted below
#: instead of guessed, so a rename in the pack says so.
_PACK_CSV = {
    "mask": "gen_masks_settings.csv",
    "measure": "crop_measure_settings.csv",
}


def _make_stub_settings(dst: Path) -> Path:
    """Emit minimal settings CSVs mirroring the HF settings pack.

    Same file names as the pack, including its ``Key,Value`` header row, so
    stub mode exercises the same lookup the real mode does.
    """
    settings = dst / "settings"; settings.mkdir(parents=True,
                                                    exist_ok=True)
    (settings / _PACK_CSV["mask"]).write_text(
        "Key,Value\n"
        "src,\n"
        "metadata_type,cellvoyager\n"
        "channels,\"[0, 1, 2]\"\n"
        "cell_channel,0\n"
        "nucleus_channel,1\n"
        "plot,false\n"
        "test_mode,false\n"
        "batch_size,2\n"
        # The mask stage normally deletes its intermediate stacks after they
        # have been merged. This test explicitly asserts the stack contract,
        # so keep the artefact it is looking for instead of calling a planned
        # cleanup "no masks were produced".
        "keep_intermediate,true\n"
    )
    (settings / _PACK_CSV["measure"]).write_text(
        "Key,Value\n"
        "src,\n"
        # Three image channels are merged first, then cell and nucleus masks.
        # The generic Measure defaults assume four image channels; spelling
        # the fixture's actual layout exercises the same provenance check a
        # real settings pack must satisfy.
        "cell_mask_dim,3\n"
        "nucleus_mask_dim,4\n"
        "pathogen_mask_dim,None\n"
        "organelle_mask_dim,None\n"
        "plot,false\n"
    )
    (settings / "annotate_settings.csv").write_text("Key,Value\nsrc,\n")
    return settings


@pytest.fixture(scope="module")
def _prepared_workspace(tmp_path_factory):
    _require_network()
    _require_gpu_cellpose()
    # WHY THIS SEED IS HERE (instruction 104). This test failed about one run
    # in eight -- same commit, same stub, same machine. It was FLAKY, not
    # broken. The stub images are already seeded, so the variance was
    # downstream: cellpose has no seed of its own and draws from the NumPy and
    # Torch global streams (spacr.runctx.SEED_CAVEATS["cellpose"]), and nothing
    # in the mask pipeline seeds them. Segmentation therefore varied run to
    # run, occasionally finding no cells -- and the assertion below, "some
    # cell_mask output exists", is exactly weak enough to pass most of the time
    # and fail sometimes.
    #
    # Per that caveat, seeding makes cellpose reproducible on CPU, which is the
    # path the stub uses.
    #
    # This makes the TEST deterministic. The PIPELINE is still unseeded for
    # real runs, which is the larger finding and is recorded in instruction
    # 104 rather than fixed here.
    from spacr.runctx import seed_everything
    seed_everything(0)
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
    import ast

    from spacr.qt.screens.settings_model import resolve_default_settings
    settings = dict(resolve_default_settings(app_key))
    csv = settings_root / _PACK_CSV[app_key]
    assert csv.is_file(), (
        f"the settings pack has no {csv.name}; the chain below would run on "
        f"resolve_default_settings({app_key!r}) alone and prove nothing. "
        f"What it does ship: "
        f"{sorted(p.name for p in settings_root.glob('*.csv'))}")
    import csv as _csv
    with csv.open() as fh:
        for row in _csv.reader(fh):
            if not row or row[0].startswith("#") or len(row) < 2:
                continue
            k, v = row[0].strip(), row[1]
            if k == "Key":                      # the pack's header row
                continue
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        # `channels,"[0, 1, 2, 3]"` and `png_dims,"[0, 2, 3]"`
                        # are lists on the page and were being handed to the
                        # pipeline as the string "[0, 1, 2, 3]".
                        try:
                            v = ast.literal_eval(v)
                        except (ValueError, SyntaxError):
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
    # The downloaded pack may choose production cleanup. An E2E output test
    # must retain the output it asserts in both stub and real-data modes.
    settings["keep_intermediate"] = True
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
    # No guard: spacr.measure is spaCR's own code, not an optional dependency.
    # An ImportError here is the bug, not a reason to stand the stage down.
    from spacr.measure import measure_crop

    settings = _load_settings_for("measure", settings_root, dataset)
    # Unguarded: _prepared_workspace has already run the mask stage over this
    # dataset, so measure_crop is being handed spaCR's own output. "Bailed on
    # the HF dataset" is the result this stage exists to report, not a reason
    # to withhold it.
    measure_crop(settings)
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
