"""Item 288: a foreign mask whose well disagrees still pairs by a unique field.

Collaborators' masks are often written by a different tool than their
images, and the well in the two names does not always agree. The importer
tries, in order: the exact plate/well/field, the same well and field, and
finally the field alone -- but only when exactly ONE image field carries it,
so the loose match can never pick between two candidates.

Here the image is from well B02 and the mask names well A01, both field 1.
The mask is paired with the only field 1 there is, and the pairing is
recorded, not guessed silently.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import tifffile

from spacr import foreign as fg


def _write(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, array)


def test_a_mask_for_another_well_pairs_with_the_only_matching_field(
        tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "cell_masks"
    image_name = "plate1_B02_T0001F001L01A01Z01C01.tif"
    _write(str(images / image_name), np.full((24, 24), 5, np.uint16))
    label = np.zeros((24, 24), np.uint16)
    label[2:8, 2:8] = 1
    _write(str(masks / "plate1_A01_T0001F001L01A01Z01C01.tif"), label)
    table = pd.DataFrame({"ImageNumber": [image_name], "ObjectNumber": [1],
                          "AreaShape_Area": [36.0]})

    plan = fg.plan_import(str(images), {"cell": str(masks)}, table,
                          metadata_type="cellvoyager", um_per_px=0.5)

    assert len(plan.masks.fields) == 1
    (stem, present), = plan.masks.fields.items()
    mapping = present["cell"]
    assert mapping.well == "B02", "paired to the image's well, not its own"
    assert mapping.field == 1
    assert mapping.labels == (1,)
    assert plan.masks.masks_without_images == []
