"""Item 919 frame fix: painting-frame labels onto the sequencing frame.

PERISCOPE publishes its painting crops in their own frame; its 04Align moves
them onto Cycle01_DAPI at analysis time and logs the shift in each site's
Image.csv. A prefix decode that reads labels segmented on the painting crops
must move them by that shift first. Pinned here: the shift helpers, the sign
(labels moved by minus the logged shift land on the sequencing frame), the
refusal of a site without a record, the 'register' source, the decode giving
the same calls from painting-frame labels as from sequencing-frame ones, the
per-site record, the guard against spaCR painting labels read as 'sbs', and
the unchanged 'sbs' default.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import cpg_cropped as C
from spacr import ops_prefix as P
from spacr.tabular import read_table
from tests.test_ops_spacer_prefix_decode_920 import (SPACERS, library_table,
                                                     synthetic_site)

BASES = ("G", "T", "A", "C")
LOGGED = {1: (6, -5), 2: (-5, 6)}   # PERISCOPE (Align_Yshift, Align_Xshift)_DAPI_Painting
BATCH = "20200101_TEST"


def write_image_csv(top, site, logged=None, empty=False):
    """A PERISCOPE-like Image.csv with the 04Align columns."""
    folder = top / "workspace" / "analysis" / BATCH / f"CP9-Well1-{site}"
    folder.mkdir(parents=True, exist_ok=True)
    y, x = ("", "") if empty else logged
    (folder / "Image.csv").write_text(
        "ImageNumber,Align_Xshift_DAPI_Painting,Align_Yshift_DAPI_Painting,Count_Cells\n"
        f"1,{x},{y},4\n")


def write_tree(top, labels_by_site, stack):
    """Sequencing stack, CorrDNA and Cycle01_DAPI per site in the cpg0021 layout."""
    well = top / "images" / BATCH / "images_corrected_cropped" / "CP9_Well1"
    for site, labels in labels_by_site.items():
        for c in range(stack.shape[0]):
            for k, base in enumerate(BASES):
                name = f"Cycle{c + 1:02d}_{base}"
                (well / name).mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(well / name / f"{name}_Site_{site}.tiff", stack[c, ..., k])
        for name in ("CorrDNA", "Cycle01_DAPI"):
            (well / name).mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(well / name / f"{name}_Site_{site}.tiff",
                             np.full(labels.shape, 100, np.uint16))
    return well.parent


def write_masks(folder, labels_by_site):
    folder.mkdir(parents=True, exist_ok=True)
    for site, labels in labels_by_site.items():
        tifffile.imwrite(folder / f"CP9_Well1_{site}_cells.tiff", labels.astype(np.uint16))
    return str(folder / "{plate}_{well}_{site}_cells.tiff")


def test_shift_image_moves_and_zero_fills():
    image = np.arange(1, 13, dtype=np.uint16).reshape(3, 4)
    moved = C.shift_image(image, 1, -1)
    assert moved.dtype == image.dtype and moved.shape == image.shape
    np.testing.assert_array_equal(moved[1:, :-1], image[:-1, 1:])
    assert not moved[0].any() and not moved[:, -1].any()
    np.testing.assert_array_equal(C.shift_image(image, 0, 0), image)
    assert not C.shift_image(image, 3, 0).any()
    back = C.shift_image(C.shift_image(image, 1, 2), -1, -2)
    np.testing.assert_array_equal(back[:-1, :-2], image[:-1, :-2])


def test_image_csv_path_from_root_or_template(tmp_path):
    root = tmp_path / "images" / BATCH / "images_corrected_cropped"
    want = os.path.join(str(tmp_path), "workspace", "analysis", BATCH, "CP9-Well1-7", "Image.csv")
    assert C.periscope_image_csv(str(root), "CP9", "W1", 7) == want
    assert C.periscope_image_csv(str(root / "CP9_Well1"), "CP9", "Well1", 7) == want
    assert C.periscope_image_csv("/x", "CP9", "W1", 7,
                                 "/t/{plate}/{well}/{well_number}/{site}.csv") == "/t/CP9/Well1/1/7.csv"
    with pytest.raises(ValueError, match="cpg_align_table"):
        C.periscope_image_csv(str(tmp_path / "elsewhere"), "CP9", "W1", 7)


def test_logged_shift_sign_and_refusals(tmp_path):
    root = str(tmp_path / "images" / BATCH / "images_corrected_cropped")
    write_image_csv(tmp_path, 1, LOGGED[1])
    found = C.painting_to_sbs_shift(pd.DataFrame(), "CP9", "W1", 1, cpg_root=root)
    assert (found["dy"], found["dx"]) == (-6, 5)
    assert (found["logged_dy"], found["logged_dx"]) == LOGGED[1]
    assert found["source"] == "periscope_logged" and found["record"].endswith("Image.csv")
    with pytest.raises(ValueError, match="no 04Align record"):
        C.painting_to_sbs_shift(pd.DataFrame(), "CP9", "W1", 2, cpg_root=root)
    write_image_csv(tmp_path, 3, empty=True)
    with pytest.raises(ValueError, match="refusing the site"):
        C.painting_to_sbs_shift(pd.DataFrame(), "CP9", "W1", 3, cpg_root=root)
    with pytest.raises(ValueError, match="cpg_align_source"):
        C.painting_to_sbs_shift(pd.DataFrame(), "CP9", "W1", 1, cpg_root=root, source="guess")


def test_register_source_finds_the_shift(tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128]
    sbs = np.zeros((128, 128))
    for y, x in rng.integers(10, 118, (40, 2)):
        sbs += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 8.0)
    sbs = (sbs * 3000).astype(np.uint16)
    painting = C.shift_image(sbs, 3, -2)       # painting frame = sbs moved by the logged shift
    well = tmp_path / "CP9_Well1"
    for name, plane in (("CorrDNA", painting), ("Cycle01_DAPI", sbs)):
        (well / name).mkdir(parents=True)
        tifffile.imwrite(well / name / f"{name}_Site_1.tiff", plane)
    index = C.cpg_cropped_index(str(well))
    found = C.painting_to_sbs_shift(index, "CP9", "W1", 1, cpg_root=str(well), source="register")
    assert (found["dy"], found["dx"]) == (-3, 2) and found["source"] == "register"


def _run(tmp_path, name, masks, frame=None, **extra):
    library = tmp_path / "Barcodes.csv"
    library_table(SPACERS).to_csv(library, index=False)
    settings = {"ops_layout": "cpg_cropped", "ops_library_mode": "spacer_prefix",
                "cpg_root": str(tmp_path / "images" / BATCH / "images_corrected_cropped"),
                "cpg_phenotype_channels": ["CorrDNA"], "ops_library": str(library),
                "ops_prefix_masks": masks, "ops_read_threshold": 10.0, "n_workers": 1,
                "dst_root": str(tmp_path / name), **extra}
    if frame:
        settings["ops_prefix_masks_frame"] = frame
    return P.run_prefix_decode(settings)


def _table(out, name):
    return read_table(os.path.join(out["folder"], name), canonicalise=False, report=None)


def test_painting_labels_give_the_sequencing_frame_calls(tmp_path):
    stack, sbs_labels = synthetic_site()
    sites = (1, 2)
    write_tree(tmp_path, {s: sbs_labels for s in sites}, stack)
    for site in sites:
        write_image_csv(tmp_path, site, LOGGED[site])
    painting = {s: C.shift_image(sbs_labels, *LOGGED[s]) for s in sites}
    sbs_masks = write_masks(tmp_path / "masks_sbs", {s: sbs_labels for s in sites})
    painting_masks = write_masks(tmp_path / "masks_painting", painting)

    reference = _run(tmp_path, "reference", sbs_masks)
    fixed = _run(tmp_path, "fixed", painting_masks, frame="painting")
    for name in ("cells_CP9_W1.csv", "site_summary.csv", "cells_per_guide.csv"):
        pd.testing.assert_frame_equal(_table(reference, name), _table(fixed, name))
    record = _table(fixed, "label_frame_CP9_W1.csv")
    assert list(record["site"]) == [1, 2]
    assert list(zip(record["dy"], record["dx"])) == [(-6, 5), (5, -6)]
    assert set(record["source"]) == {"periscope_logged"}
    classes = json.loads(open(os.path.join(fixed["folder"], "quality_classes.json")).read())
    assert classes["settings"]["ops_prefix_masks_frame"] == "painting"

    # the bug being fixed: painting labels read as if in the sequencing frame
    wrong = _run(tmp_path, "wrong", painting_masks)
    assert not _table(wrong, "cells_CP9_W1.csv").equals(_table(reference, "cells_CP9_W1.csv"))


def test_sbs_default_writes_what_it_wrote_before(tmp_path):
    stack, labels = synthetic_site()
    write_tree(tmp_path, {1: labels}, stack)
    masks = write_masks(tmp_path / "masks", {1: labels})
    out = _run(tmp_path, "run", masks)
    classes = json.loads(open(os.path.join(out["folder"], "quality_classes.json")).read())
    assert not set(P._ITEM_919_FRAME_KEYS) & set(classes["settings"])
    assert not os.path.exists(os.path.join(out["folder"], "label_frame_CP9_W1.csv"))


def test_site_without_a_record_is_refused(tmp_path):
    stack, labels = synthetic_site()
    write_tree(tmp_path, {1: labels, 2: labels}, stack)
    write_image_csv(tmp_path, 1, LOGGED[1])
    masks = write_masks(tmp_path / "masks", {1: labels, 2: labels})
    with pytest.raises(ValueError, match="no 04Align record"):
        _run(tmp_path, "run", masks, frame="painting")


def test_spacr_painting_labels_read_as_sbs_are_refused(tmp_path):
    stack, labels = synthetic_site()
    write_tree(tmp_path, {1: labels}, stack)
    write_image_csv(tmp_path, 1, LOGGED[1])
    seg = tmp_path / "seg"
    masks = write_masks(seg / "labels", {1: C.shift_image(labels, *LOGGED[1])})
    (seg / "settings.json").write_text(json.dumps({"seg_sites_src": "/crops/CP9_Well1"}))
    with pytest.raises(ValueError, match="ops_prefix_masks_frame to 'painting'"):
        _run(tmp_path, "sbs", masks)
    out = _run(tmp_path, "painting", masks, frame="painting")
    assert out["sites"] == 1


def test_frame_settings_are_checked():
    with pytest.raises(ValueError, match="ops_prefix_masks_frame"):
        P.prefix_settings({"ops_library": "x.csv", "ops_prefix_masks_frame": "both"})
    with pytest.raises(ValueError, match="cpg_align_source"):
        P.prefix_settings({"ops_library": "x.csv", "cpg_align_source": "guess"})
    chosen = P.prefix_settings({"ops_library": "x.csv"})
    assert chosen["ops_prefix_masks_frame"] == "sbs"
    assert chosen["cpg_align_source"] == "periscope_logged" and chosen["cpg_align_table"] is None
