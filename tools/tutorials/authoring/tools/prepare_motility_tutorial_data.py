#!/usr/bin/env python3
"""Write Motility Assay settings for the tracked tutorial time series."""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path("/mnt/firecuda2/codex/repo/spacr")
ROOT = Path("/mnt/firecuda2/Claude/toxoplasma_projects/tutorials")
SOURCE = ROOT / "synthetic" / "motility" / "source"


def main() -> int:
    sys.path.insert(0, str(REPO))
    from spacr.settings import get_automated_motility_assay_default_settings
    from spacr.qt.synthetic import demo_settings, generate_timelapse_demo

    merged = SOURCE / "merged"
    if not any(merged.glob("*.npy")):
        generate_timelapse_demo(
            SOURCE,
            plate="motility_plate",
            wells=("A01",),
            fields=1,
            times=8,
            channels=(0, 1, 2),
        )
        from spacr.core import preprocess_generate_masks_timelapse
        timelapse_settings = demo_settings(
            "timelapse", str(SOURCE), channels=(0, 1, 2)
        )
        timelapse_settings.update({
            "timelapse_frame_limits": [0, 8],
            "timelapse_objects": ["cell"],
            "timelapse_mode": "iou",
            "fps": 1,
            "plot": False,
            "test_mode": False,
            "n_jobs": 1,
            "batch_size": 8,
        })
        preprocess_generate_masks_timelapse(timelapse_settings)

    settings = get_automated_motility_assay_default_settings({})
    settings.update({
        "src": str(SOURCE),
        "channels": [0, 1, 2],
        "nucleus_channel": 0,
        "cell_channel": 1,
        "pathogen_channel": 2,
        "tracked_object": "cell",
        "seconds_per_frame": 1,
        "pixels_per_um": 1.0,
        "max_displacement": 100.0,
        "straightness_filter": False,
        "zscore_thresh": 3.0,
        "infection_intensity_qc": False,
        "infection_intensity_strategy": "histogram",
        "infection_intensity_qc_scope": "none",
        "reuse_existing_measurements": False,
        "infection_intensity_qc_graphs": True,
        "motility_xlim": [-100, 100],
        "motility_ylim": [-100, 100],
        "n_jobs": 1,
    })
    path = ROOT / "synthetic" / "motility" / "tutorial_settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
