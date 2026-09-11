#!/usr/bin/env python3
"""Build a small, reproducible Train Cellpose tutorial set from real fields.

The recorded settings identify image channel 1 and label plane 4 as cell.
Plane 6 is pathogen, NOT cell: the historical generator paired the wrong
compartment. Verify those recorded settings before extracting new pairs.
Existing labels and edge-cut objects are not independently reviewed ground
truth. These pairs demonstrate file preparation and training mechanics only.
Never overwrite the retained historical examples or an existing destination.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


SOURCE = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials/merged"
)
DESTINATION = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/'
                   'refresh_2026-09-09/derived/train_cellpose_corrected')
IMAGE_CHANNEL = 1
MASK_PLANE = 4

# Fixed fields and crop origins keep every regeneration visually identical.
PAIRS = (
    ("plate1_B01_1_1.npy", 120, 180),
    ("plate1_B01_2_1.npy", 520, 240),
    ("plate1_B01_3_1.npy", 940, 520),
    ("plate1_B02_1_1.npy", 180, 980),
    ("plate1_B02_2_1.npy", 700, 1080),
    ("plate1_B02_3_1.npy", 1120, 900),
)
CROP_SIZE = 512


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def recorded_planes(source):
    """Pin the target compartment to original settings, not image appearance."""
    settings = Path(source).parent / 'settings'
    def read(name):
        with (settings / name).open(newline='') as stream:
            return {row['Key']: row['Value'] for row in csv.DictReader(stream)}
    measured = read('measure_crop_settings.csv')
    segmented = read('gen_mask_settings.csv')
    actual = (int(segmented['cell_channel']), int(measured['cell_mask_dim']),
              int(measured['nucleus_mask_dim']), int(measured['pathogen_mask_dim']))
    if actual != (IMAGE_CHANNEL, MASK_PLANE, 5, 6):
        raise ValueError('Recorded compartment mapping changed; review before preparing data')
    return {str(settings / name): digest(settings / name) for name in
            ('measure_crop_settings.csv', 'gen_mask_settings.csv')}


def prepare(source, destination, *, pairs=PAIRS, crop_size=CROP_SIZE):
    """Create new, exact image/label crops with explicit unreviewed-label scope."""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    recorded = recorded_planes(source)
    # mkdir without exist_ok refuses existing files, folders and symlinks.
    destination.mkdir(parents=True)
    image_dir = destination / 'train/images'
    mask_dir = destination / 'train/masks'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    records = []
    for index, (name, row, column) in enumerate(pairs, start=1):
        path = source / name
        original_sha = digest(path)
        merged = np.load(path, mmap_mode="r", allow_pickle=False)
        image = np.asarray(
            merged[row : row + crop_size, column : column + crop_size, IMAGE_CHANNEL]
        )
        mask = np.asarray(
            merged[row : row + crop_size, column : column + crop_size, MASK_PLANE]
        )
        if image.shape != (crop_size, crop_size) or mask.shape != image.shape:
            raise ValueError(f'incomplete crop from {path}: {image.shape}, {mask.shape}')
        if image.dtype != np.uint16 or mask.dtype != np.uint16:
            raise ValueError('Expected original uint16 image and instance-label planes')
        filename = f"cell_pair_{index:02d}.tif"
        tifffile.imwrite(image_dir / filename, image)
        tifffile.imwrite(mask_dir / filename, mask)
        if digest(path) != original_sha:
            raise ValueError('Original source changed during preparation')
        labels = np.unique(mask)
        records.append(
            {
                "file": filename,
                "source": name,
                "source_sha256": original_sha,
                "image_sha256": digest(image_dir / filename),
                "mask_sha256": digest(mask_dir / filename),
                "origin_yx": [row, column],
                "objects": int(np.count_nonzero(labels)),
                "image_range": [int(image.min()), int(image.max())],
            }
        )

    settings = {
        "src": str(destination),
        "model_name": "spacr_tutorial_cells",
        "model_type": "cpsam",
        "Signal_to_noise": 10,
        "background": 200,
        "remove_background": False,
        "learning_rate": 0.01,
        "weight_decay": 0.00001,
        "batch_size": 2,
        "n_epochs": 1,
        "from_scratch": False,
        "diameter": 30,
        "resize": True,
        "width_height": [512, 512],
        "target_size": 256,
        "augment": False,
        "verbose": True,
    }
    (destination / "tutorial_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
    manifest = {
        'source_root': str(source), 'image_channel': IMAGE_CHANNEL,
        'mask_plane': MASK_PLANE, 'recorded_settings_sha256': recorded,
        'crop_size': crop_size, 'pairs': records,
        'target_compartment': 'cell, according to original recorded settings',
        'independent_annotation_review': False, 'held_out_accuracy_validated': False,
        'edge_cut_instances_present': 'possible; source labels are preserved exactly',
        'purpose': 'workflow demonstration; independently review labels before production training',
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--destination', type=Path, default=DESTINATION,
                        help='New directory only; an existing destination is never overwritten')
    args = parser.parse_args()
    manifest = prepare(args.source, args.destination)
    print(f"Prepared {len(manifest['pairs'])} paired fields in {args.destination}")
    print(f"Settings: {args.destination / 'tutorial_settings.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
