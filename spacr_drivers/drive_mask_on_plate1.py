"""Drive the mask module on real plate1 fields, end to end.

Two fields of raw Yokogawa TIFs go through the whole pipeline on the settings
the recorded run used: orig -> stack -> merged -> masks, segmentation QC, the
cell adjustment against nuclei and pathogens, and the merge. The masks kept in
the dataset are the reference, and the driver counts objects in both.

THE OBJECT COUNTS ARE NOT EXPECTED TO MATCH, and the reason is the model. The
reference was segmented with Cellpose 3's ``cyto`` and ``nuclei``; Cellpose 4
ships one network, ``cpsam``, and estimates diameter differently. A different
network gives a different segmentation. The counts are printed side by side so
the size of that difference is visible rather than assumed.

``src`` MUST HOLD THE IMAGES DIRECTLY. Pointing it at a plate folder whose
images are already in ``orig/`` finds zero fields; spaCR says so and names the
sub-folders it found. That is why this driver stages the TIFs flat.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import (cap_gpu, dataset_root, preflight, read_settings, require,
                      run, scratch, settings_file, stage, undeclared)

DEFAULT_ROOT = "/home/olafsson/datasets/plate1"

REQUIRED = ("orig/*.tif", "masks")

SETTINGS_CANDIDATES = ("../settings/gen_masks_settings.csv",
                       "settings/gen_mask_settings.csv")

#: Two fields: enough for the QC and the cell adjustment to have something to
#: do, few enough to segment in a few minutes on a card. Override with a
#: comma-separated third argument.
FIELDS = ("F001", "F009")


def count_objects(path, planes=(("cell", 4), ("nucleus", 5), ("pathogen", 6))):
    """Objects per mask plane in one merged ``.npy``."""
    import numpy as np

    array = np.load(path)
    counts = {}
    for name, index in planes:
        if index < array.shape[-1]:
            labels = np.unique(array[..., index])
            counts[name] = int((labels != 0).sum())
    return counts


def main(argv):
    """Stage two fields of raw TIFs and segment them on the recorded settings."""
    root = require(dataset_root(argv, DEFAULT_ROOT), REQUIRED,
                   what="the plate1 pipeline dataset")
    print(f"dataset root: {root}")
    recorded = (Path(argv[2]).expanduser() if len(argv) > 2 and argv[2]
                else settings_file(root, SETTINGS_CANDIDATES,
                                   what="the mask run"))
    print(f"settings:     {recorded}")

    fields = tuple(argv[3].split(",")) if len(argv) > 3 else FIELDS
    work = scratch("mask_on_plate1")
    stage(root, [f"orig/*{field}*.tif" for field in fields], work, flatten=True)
    staged = sorted(work.glob("*.tif"))
    print(f"staged {len(staged)} images for fields {', '.join(fields)}")
    if not staged:
        raise SystemExit(
            f"no images under {root}/orig match fields {', '.join(fields)}; "
            f"name the fields as the third argument, e.g. F001,F009.")

    settings = read_settings(recorded)
    settings["src"] = str(work)
    settings["plot"] = False
    settings["batch_size"] = len(fields)
    stale = undeclared(settings, "mask")
    if stale:
        print(f"settings this spaCR no longer declares, so nothing reads them: "
              f"{stale}")
    preflight(settings, "mask")

    on_gpu = cap_gpu()
    if on_gpu:
        print("segmenting on the GPU, capped at 80% of the card")
    else:
        print("segmenting on the CPU. Cellpose 4's single network is a SAM "
              "backbone: a full 1994x1994 field takes hours there, against "
              "minutes on a card. Name one small field as the third argument "
              "if that is not what you want.")

    from spacr.core import preprocess_generate_masks

    preprocess_generate_masks(settings)

    print("\nobjects per field, this run against the reference:")
    for produced in sorted((work / "merged").glob("*.npy")):
        reference = root / "merged" / produced.name
        line = f"  {produced.stem}: {count_objects(produced)}"
        if reference.is_file():
            line += f" vs reference {count_objects(reference)}"
        print(line)
    print("\nA difference here is the segmentation model, not the pipeline: "
          "the reference was made with Cellpose 3 and this run uses whatever "
          "Cellpose is installed.")


if __name__ == "__main__":
    run(main)
