"""External masks: every organelle slot is proposed and paired, not four.

Item 76's last open note: ``spacr/external_masks.py`` recognised Organelle 2-4
by four hand-written filename patterns. Measured 2026-09-19, the cost was not
"a fifth slot is not recognised" but worse: ``organelle_5_masks`` matched the
PRIMARY slot's pattern (``organelle`` followed by a separator), so a folder of
Organelle 5 masks was proposed as Organelle 1, and the plan then refused the
run because neither numbered file paired with its intensity field.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from spacr import external_masks as em


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")


@pytest.mark.parametrize("name, expected", [
    ("organelle_masks/fov001.tif", "organelle"),
    ("organelle_1_masks.tif", "organelle"),
    ("mitochondria_masks.tif", "organelle"),
    ("organelle_2_masks.tif", "organelleb"),
    ("organelle2_masks.tif", "organelleb"),
    ("Organelle 3 labels.tif", "organellec"),
    ("organelled_masks.tif", "organelled"),
    ("organelle_5_masks.tif", "organellee"),
    ("organelle5_masks.tif", "organellee"),
    ("organellee_mask_stack/fov001.npy", "organellee"),
    ("organelle_12_masks.tif", "organellel"),
    ("organelle_27_masks.tif", "organelleaa"),
    ("organelleaa_mask_stack/fov001.npy", "organelleaa"),
    ("organelle_702_masks.tif", "organellezz"),
    ("nucleus_organelle_5.tif", "nucleus"),
    ("cell_masks/fov001.tif", "cell"),
])
def test_a_filename_names_the_slot_it_spells(name, expected):
    assert em._suggest_object(name) == expected


@pytest.mark.parametrize("name", [
    "organelle_0_masks.tif",
    "organelle_703_masks.tif",
    "organelles_masks.tif",
    "organellemarker_masks.tif",
])
def test_a_token_that_names_no_slot_proposes_none(name):
    """Before this change the first two were proposed as Organelle 1."""
    assert em._suggest_object(name) is None


def test_organelle_1_5_and_12_import_as_three_slots_and_all_pair(tmp_path):
    yy, xx = np.indices((32, 32))
    images = tmp_path / "images"
    _write(images / "fov001_C1.tif", yy * 32 + xx)
    _write(images / "fov001_C2.tif", (xx * 17 + yy * 3) % 4096)
    cell = np.zeros((32, 32), dtype=np.uint16)
    cell[3:29, 3:29] = 1
    _write(tmp_path / "cell_masks" / "fov001_cell_mask.tif", cell)
    folders = [images, tmp_path / "cell_masks"]
    for number, (top, left) in {1: (5, 5), 5: (20, 20), 12: (5, 20)}.items():
        mask = np.zeros((32, 32), dtype=np.uint16)
        mask[top:top + 3, left:left + 3] = 1
        folder = tmp_path / f"organelle_{number}_masks"
        _write(folder / f"fov001_organelle_{number}_mask.tif", mask)
        folders.append(folder)

    groups = em.detect_inputs(folders)
    proposed = sorted(group.object_type for group in groups
                      if group.role == "mask")
    assert proposed == ["cell", "organelle", "organellee", "organellel"]

    plan = em.plan_external_masks({
        "inputs": [group.to_dict() for group in groups],
        "dst": str(tmp_path / "project"),
        "layout": "flat",
    })
    assert plan.ok, plan.summary()
    assert plan.object_types == ["cell", "organelle", "organellee",
                                 "organellel"]
    assert len(plan.stems) == 1
    stem = plan.stems[0]
    assert plan.masks["organellee"][stem].path.endswith(
        "fov001_organelle_5_mask.tif")
    assert plan.masks["organellel"][stem].match == "normalised"


def test_an_unassigned_group_is_asked_about_in_one_line(tmp_path):
    """The refusal named all 705 object types, 9,122 characters of them.

    ``organelle_0`` proposes nothing now, so this message is what a user with
    zero-based folder names reads first.
    """
    yy, xx = np.indices((32, 32))
    _write(tmp_path / "images" / "fov001_C1.tif", yy * 32 + xx)
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[3:9, 3:9] = 1
    folder = tmp_path / "organelle_0_masks"
    _write(folder / "fov001_organelle_0_mask.tif", mask)

    groups = em.detect_inputs([tmp_path / "images", folder])
    assert [group.object_type for group in groups
            if group.role == "mask"] == [None]
    plan = em.plan_external_masks({
        "inputs": [group.to_dict() for group in groups],
        "dst": str(tmp_path / "project"),
        "layout": "flat",
    })
    refusal = next(message for message in plan.errors
                   if "choose whether" in message)
    assert "cell, nucleus, pathogen or an organelle slot" in refusal
    assert "organellezz" not in refusal
    assert len(refusal) < len(str(folder)) + 200
