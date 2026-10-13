#!/usr/bin/env python3
"""Build a small, explicitly non-biological Activation Maps demo artifact."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials/test"
)
DEFAULT_OUTPUT = ROOT / "synthetic" / "activation"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images", type=int, default=8)
    args = parser.parse_args()

    import torch

    from spacr.torch_artifacts import save_model_artifact
    from spacr.utils import choose_model

    output = args.output.resolve()
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    crops = sorted((args.dataset / "data").rglob("cell_png/*.png"))[: args.images]
    if len(crops) < args.images:
        raise FileNotFoundError("not enough real cell crops for activation demo")

    archive = data_dir / "real_cell_crops.tar"
    with tarfile.open(archive, "w") as handle:
        for crop in crops:
            handle.add(crop, arcname=crop.name)

    torch.manual_seed(42)
    model = choose_model(
        "resnet18", torch.device("cpu"), init_weights=False,
        channels=3, height=224, width=224, num_classes=2,
    )
    if model is None:
        raise RuntimeError("could not construct deterministic ResNet18 demo")
    model_path = output / "demo_untrained_resnet18.pth"
    save_model_artifact(
        model,
        str(model_path),
        epoch=0,
        classes=["demo_0", "demo_1"],
        channels=["red", "green", "blue"],
        preprocessing={"image_size": 224, "normalize_input": True},
        include_rng=False,
        artifact_role="tutorial_untrained_demo",
    )
    manifest = {
        "schema": 1,
        "purpose": "Activation Maps interface demonstration only",
        "biological_interpretation_allowed": False,
        "warning": (
            "The model is deterministically initialized but untrained. "
            "Its maps demonstrate software behavior, not learned biology."
        ),
        "source_dataset": str(args.dataset.resolve()),
        "dataset_tar": str(archive),
        "model_path": str(model_path),
        "images": [str(path) for path in crops],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
