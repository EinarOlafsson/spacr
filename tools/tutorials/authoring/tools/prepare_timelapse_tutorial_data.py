#!/usr/bin/env python3
"""Create the deterministic synthetic time series used by tutorial 17."""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path("/mnt/firecuda2/codex/repo/spacr")
ROOT = Path("/mnt/firecuda2/Claude/toxoplasma_projects/tutorials")
DESTINATION = ROOT / "synthetic" / "timelapse"


def main() -> int:
    sys.path.insert(0, str(REPO))
    from spacr.qt.synthetic import demo_settings, generate_timelapse_demo

    layout = generate_timelapse_demo(
        DESTINATION,
        plate="tutorial_plate",
        wells=("A01",),
        fields=1,
        times=8,
        channels=(0, 1),
    )
    settings = demo_settings("timelapse", str(DESTINATION), channels=(0, 1))
    settings.update({
        "src": str(DESTINATION),
        "timelapse": True,
        "timelapse_frame_limits": [0, 8],
        "timelapse_objects": ["cell"],
        "timelapse_mode": "iou",
        "fps": 1,
        "normalize": True,
        "lower_percentile": 2,
        "test_mode": False,
        "plot": False,
        "save": True,
        "n_jobs": 1,
        "batch_size": 8,
    })
    settings_path = DESTINATION / "tutorial_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    manifest = {
        "schema": 1,
        "kind": "deterministic synthetic timelapse",
        "frames": 8,
        "channels": {"0": "nuclei", "1": "cell/ER"},
        "purpose": "interface, segmentation, and tracking demonstration",
        "settings": str(settings_path),
        "generated_files": [str(path) for path in layout.image_files],
    }
    (DESTINATION / "tutorial_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(settings_path)
    print(f"images={len(layout.image_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
