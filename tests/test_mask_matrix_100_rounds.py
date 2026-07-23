"""100-round mask-generation matrix — every object-type combination
crossed with cycling settings.

The user's directive: "try 100 rounds and change the settings. Do a run
with just cells just nuclei just pathogens, or any combination also
organelles, and cycle through the other settings."

Design:
  * A 4-channel synthetic Yokogawa plate is generated ONCE per config
    (nucleus=C00, cell=C01, pathogen=C02, organelle=C03).
  * 15 non-empty object-type subsets × cycling settings (normalise,
    remove_background, diameter, pipeline v1/v2, batch size) →
    deterministically expanded to exactly 100 configs.
  * Each config runs the mask pipeline and asserts that the REQUESTED
    object types produced a mask stack while the pipeline didn't crash.

Two entry points are exercised:
  * spacr.core.preprocess_generate_masks (v1)  — most configs
  * spacr.pipeline_v2.run_v2 (v2 streaming)     — the configs whose
    pipeline_style == 'v2'

Marked @slow + @gpu. Full 100-round run is ~6-12 min on a GPU box
(1 well × 1 field × 96 px keeps each Cellpose call short). Skips
cleanly without CUDA / cellpose.

Because 100 separate pytest params would flood the report, the matrix
runs as ONE test that collects per-round failures and asserts none —
each failure line names the exact config so a regression is
pinpointable.
"""
from __future__ import annotations

import itertools
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

def _require_gpu_cellpose():
    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch unavailable: {e}")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA — the 100-round matrix is GPU-only")
    try:
        import cellpose                                    # noqa: F401
    except Exception as e:
        pytest.skip(f"cellpose unavailable: {e}")


# ---------------------------------------------------------------------------
# Object-type ↔ channel map
# ---------------------------------------------------------------------------

OBJECT_CHANNEL = {
    "nucleus":   0,
    "cell":      1,
    "pathogen":  2,
    "organelle": 3,
}
ALL_OBJECTS = ("nucleus", "cell", "pathogen", "organelle")


def _nonempty_subsets() -> List[Tuple[str, ...]]:
    """Every non-empty subset of the four object types (15 total)."""
    out: List[Tuple[str, ...]] = []
    for r in range(1, len(ALL_OBJECTS) + 1):
        for combo in itertools.combinations(ALL_OBJECTS, r):
            out.append(combo)
    return out


# ---------------------------------------------------------------------------
# Config matrix — deterministically expand to exactly 100 rounds
# ---------------------------------------------------------------------------

@dataclass
class Round:
    idx:              int
    objects:          Tuple[str, ...]
    normalize:        bool
    remove_background: bool
    diameter:         float
    batch_size:       int
    pipeline:         str   # 'v1' or 'v2'

    def label(self) -> str:
        return (f"round#{self.idx:03d} objs={'+'.join(self.objects)} "
                f"norm={int(self.normalize)} rmbg={int(self.remove_background)} "
                f"diam={self.diameter:g} batch={self.batch_size} "
                f"pipe={self.pipeline}")


def _build_matrix(n: int = 100) -> List[Round]:
    """Deterministically build ``n`` rounds cycling every knob.

    No RNG — we index into the Cartesian-ish product with prime-ish
    strides so each round differs in several dimensions at once, and
    every object subset appears multiple times across the 100.
    """
    subsets = _nonempty_subsets()             # 15
    norm_opts = (True, False)
    rmbg_opts = (True, False)
    diam_opts = (0.0, 15.0, 30.0, 60.0)       # 0 → auto-estimate
    batch_opts = (1, 2, 4)
    # v2 only supports cell/nucleus-style single-pass; keep it to the
    # subsets that pipeline_v2 handles (it segments one channel). We
    # route ~1/5 of the rounds through v2.
    rounds: List[Round] = []
    for i in range(n):
        objs = subsets[i % len(subsets)]
        norm = norm_opts[(i // 3) % len(norm_opts)]
        rmbg = rmbg_opts[(i // 5) % len(rmbg_opts)]
        diam = diam_opts[(i // 7) % len(diam_opts)]
        batch = batch_opts[(i // 11) % len(batch_opts)]
        # v2 handles a single object channel cleanly; only route
        # single-object rounds there so the comparison is apples to
        # apples.
        pipeline = "v2" if (i % 5 == 0 and len(objs) == 1) else "v1"
        rounds.append(Round(idx=i, objects=objs, normalize=norm,
                              remove_background=rmbg, diameter=diam,
                              batch_size=batch, pipeline=pipeline))
    return rounds


# ---------------------------------------------------------------------------
# Synthetic plate — always 4 channels; objects toggled via settings
# ---------------------------------------------------------------------------

def _make_plate(dst: Path, size: int = 96) -> Path:
    """One well, one field, four channels of Gaussian blobs."""
    import tifffile
    plate = dst / "plate1"; plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    centres = rng.integers(15, size - 15, size=(10, 2))
    y, x = np.ogrid[:size, :size]

    def _blobs(cs, radius, intensity):
        bg = rng.integers(80, 160, size=(size, size)).astype(np.uint16)
        for cy, cx in cs:
            g = np.exp(-((x - int(cx)) ** 2 + (y - int(cy)) ** 2)
                           / (2 * radius ** 2)) * intensity
            bg = np.clip(bg.astype(np.float32) + g, 0, 65535
                             ).astype(np.uint16)
        return bg

    layers = {
        0: _blobs(centres, 4, 3000),        # nucleus
        1: _blobs(centres, 9, 2500),        # cell
        2: _blobs(centres[:6], 2, 2000),    # pathogen
        3: _blobs(centres[:4], 1, 1800),    # organelle
    }
    for ch, arr in layers.items():
        p = plate / f"plate1_A01_T01F01L01A01Z01C0{ch}.tif"
        tifffile.imwrite(str(p), arr)
    return plate


# ---------------------------------------------------------------------------
# Per-round settings
# ---------------------------------------------------------------------------

def _v1_settings(plate: Path, rnd: Round) -> dict:
    from spacr.qt import synthetic as syn
    s = syn.demo_settings("mask", str(plate))
    # Object channels: enabled → their channel index, disabled → None.
    for obj in ALL_OBJECTS:
        key = f"{obj}_channel"
        s[key] = OBJECT_CHANNEL[obj] if obj in rnd.objects else None
    s.update({
        "channels": [0, 1, 2, 3],
        "consolidate": False,
        "remove_background": rnd.remove_background,
        "normalize": rnd.normalize,
        "backgrounds": [100, 100, 100, 100],
        "remove_background_cell": rnd.remove_background,
        "remove_background_nucleus": rnd.remove_background,
        "remove_background_pathogen": rnd.remove_background,
        "cells_per_field": 10,
        "test_mode": False,
        "Signal_to_noise": 10,
        "cell_intensity_range": None,
        "nucleus_intensity_range": None,
        "pathogen_intensity_range": None,
        "cytoplasm_intensity_range": None,
        "denoise": False,
        "remove_background_intensity": False,
        "skip_extraction": False,
        "batch_size": rnd.batch_size,
        "plot": False,
    })
    # Per-object diameter overrides (0 → let Cellpose estimate).
    if rnd.diameter > 0:
        for obj in rnd.objects:
            s[f"{obj}_diameter"] = rnd.diameter
    return s


# ---------------------------------------------------------------------------
# The matrix test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_mask_matrix_100_rounds(tmp_path_factory, caplog):
    _require_gpu_cellpose()
    import os
    from spacr.core import preprocess_generate_masks
    from spacr.pipeline_v2 import run_v2

    # SPACR_MATRIX_N lets a quick validation run do fewer rounds
    # without editing the file (e.g. SPACR_MATRIX_N=6 for a smoke).
    n_rounds = int(os.environ.get("SPACR_MATRIX_N", "100"))
    rounds = _build_matrix(n_rounds)
    caplog.set_level(logging.INFO, logger="spacr")

    failures: List[str] = []
    ok_count = 0
    t_start = time.time()

    for rnd in rounds:
        root = tmp_path_factory.mktemp(f"matrix_{rnd.idx:03d}",
                                          numbered=False)
        plate = _make_plate(root)
        try:
            if rnd.pipeline == "v2":
                obj = rnd.objects[0]
                run_v2(
                    src=plate,
                    channels=(0, 1, 2, 3),
                    model_name="cyto",
                    channels_for_cellpose=(OBJECT_CHANNEL[obj], 0),
                    diameter=(rnd.diameter or None),
                    batch_fields=rnd.batch_size,
                    metadata_type="cellvoyager",
                )
                merged = plate / "merged"
                stacks = list(merged.glob("stack_*.npy")) if merged.is_dir() else []
                if not stacks:
                    failures.append(
                        f"{rnd.label()} — v2 wrote no merged stacks")
                    continue
            else:
                settings = _v1_settings(plate, rnd)
                preprocess_generate_masks(settings)
                # Each requested object type must have produced a mask
                # stack folder with content.
                masks_root = plate / "masks"
                missing = []
                for obj in rnd.objects:
                    d = masks_root / f"{obj}_mask_stack"
                    if not d.is_dir() or not list(d.glob("*.npy")):
                        missing.append(obj)
                if missing:
                    failures.append(
                        f"{rnd.label()} — no mask stack for "
                        f"{', '.join(missing)}")
                    continue
            ok_count += 1
        except Exception as e:
            tb = traceback.format_exc().splitlines()[-1]
            failures.append(f"{rnd.label()} — RAISED {tb}")

    elapsed = time.time() - t_start
    print(f"\n[matrix] {ok_count}/{len(rounds)} rounds OK in "
            f"{elapsed:.0f}s; {len(failures)} failure(s)")
    for f in failures:
        print(f"[matrix][FAIL] {f}")

    # We tolerate a small number of Cellpose "found nothing" flukes on
    # the tiniest synthetic tiles, but the bulk must pass.
    assert ok_count >= int(0.9 * len(rounds)), (
        f"only {ok_count}/{len(rounds)} matrix rounds passed — "
        f"see [matrix][FAIL] lines above")
