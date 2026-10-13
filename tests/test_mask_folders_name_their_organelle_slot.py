"""External masks: a folder's own name types its masks; laser lines are not slots.

Item 76, open points recorded 2026-09-19 and fixed 2026-09-30:

* folder-only naming -- ``organelle/fov001.tif``, ``organelle_2/fov001.tif``
  -- proposed nothing for the bare folder and slot 1 for ``organelle_2/``,
  because ``/`` was not a token boundary;
* ``organelle_488_masks`` was proposed as Organelle 488.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from spacr import external_masks as em


def _labels(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((32, 32), dtype=np.uint16)
    array[4:10, 4:10] = 1
    tifffile.imwrite(path, array, photometric="minisblack")


@pytest.mark.parametrize("name, expected", [
    ("organelle/fov001.tif", "organelle"),
    ("organelle_1/fov001.tif", "organelle"),
    ("organelle_2/fov001.tif", "organelleb"),
    ("proj/organelle_5_masks/fov001.tif", "organellee"),
    ("proj/cell_masks/fov001.tif", "cell"),
    ("proj/nuclei/fov001.tif", "nucleus"),
    ("organelle_488_masks.tif", "organelle"),
    ("organelle_561/fov001.tif", "organelle"),
    ("organelle_masks/fov001_organelle_640.tif", "organelle"),
    ("organelle_488/fov001_organelle_3_mask.tif", "organellec"),
    ("organelle_12_masks.tif", "organellel"),
])
def test_a_folder_or_laser_line_token_proposes_the_right_type(name, expected):
    assert em._suggest_object(name) == expected


def test_a_folder_path_is_a_mask_word_boundary():
    assert em._MASK_WORDS.search("proj/cell_masks/fov001")
    assert em._MASK_WORDS.search("labels/fov001")


def test_three_folder_only_groups_are_typed_by_their_folders(tmp_path):
    for folder in ("organelle", "organelle_1", "organelle_2"):
        _labels(tmp_path / folder / "fov001.tif")
    groups = em.detect_inputs(
        [tmp_path / f for f in ("organelle", "organelle_1", "organelle_2")])
    typed = sorted((g.root.rsplit("/", 1)[-1], g.object_type)
                   for g in groups if g.role == "mask")
    assert typed == [("organelle", "organelle"),
                     ("organelle_1", "organelle"),
                     ("organelle_2", "organelleb")]


def test_project_subfolders_are_typed_by_their_names(tmp_path):
    _labels(tmp_path / "proj" / "cell_masks" / "fov001.tif")
    _labels(tmp_path / "proj" / "organelle_2_masks" / "fov001.tif")
    groups = em.detect_inputs([tmp_path / "proj"])
    assert sorted(g.object_type for g in groups if g.role == "mask") == [
        "cell", "organelleb"]
    assert {g.confidence for g in groups} == {0.99}
