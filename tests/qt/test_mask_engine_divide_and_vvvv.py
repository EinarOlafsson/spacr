"""The divide stroke, the magic wand over owned ground, and the vvvv export.

``divide_object`` cuts only the objects a stroke really separates: the
larger piece keeps its ID, the smaller gets the first free one, an object
the stroke only nicks is left whole, and a stroke off the field or over
background gives back an unchanged copy.  ``magic_wand`` counts its budget
only against pixels it changes, so a click on ground the mask already owns
rewrites the whole object.  ``_export_vvvv`` refuses labels it cannot store
as a 16-bit PNG before writing anything, and records an empty version when
spaCR carries none.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import mask_engine as engine

pytestmark = pytest.mark.qt


def _two_blobs():
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:8, 2:8] = 4
    mask[12:18, 12:18] = 9
    return mask


def _crossed_and_nicked():
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:8, 2:16] = 4      # crossed by the cut at x = 9
    mask[12:16, 9:14] = 6    # only its left column lies under the cut
    return mask


def test_divide_object_splits_only_what_the_stroke_separates(subtests):
    """Each stroke gives the exact mask and (split, new id) pairs, never in place."""
    divided = _crossed_and_nicked()
    divided[2:8, 9] = 0
    divided[2:8, 10:16] = 7  # the smaller piece takes the first free id
    # case, mask, p0 (x, y), p1, expected mask, expected splits
    cases = [
        ("off the field", _two_blobs(), (-40, -40), (-30, -35), _two_blobs(),
         []),
        ("through background", _two_blobs(), (10, 0), (10, 9), _two_blobs(),
         []),
        ("a nick keeps the object whole", _two_blobs(), (2, 2), (3, 2),
         _two_blobs(), []),
        ("splits the crossed object, skips the nicked one",
         _crossed_and_nicked(), (9, 0), (9, 19), divided, [(4, 7)]),
    ]
    for case, mask, p0, p1, expected, splits in cases:
        with subtests.test(msg=case):
            before = mask.copy()
            out, found = engine.divide_object(mask, p0, p1)
            assert found == splits
            assert out.dtype == np.uint16
            np.testing.assert_array_equal(out, expected)
            assert out is not mask
            np.testing.assert_array_equal(mask, before)


def test_magic_wand_charges_its_budget_only_for_pixels_it_changes(subtests):
    """Owned ground is rewritten for free; changed pixels stop at the budget."""
    image = np.full((40, 40), 100, dtype=np.uint16)
    owned = np.full((40, 40), 3, dtype=np.uint8)
    blank = np.zeros((40, 40), dtype=np.uint8)
    # case, mask, action, expected {value: count}
    cases = [
        ("add over an owned object rewrites all of it", owned, "add",
         {255: 1600}),
        ("erase over empty ground stays empty", blank, "erase", {0: 1600}),
        ("add into blank ground stops at the budget", blank, "add",
         {0: 1596, 255: 4}),
        ("erase over an owned object stops at the budget", owned, "erase",
         {0: 4, 3: 1596}),
    ]
    for case, mask, action, counts in cases:
        with subtests.test(msg=case):
            before = mask.copy()
            out = engine.magic_wand(image, mask, 20, 20, tolerance=5,
                                    max_pixels=4, action=action)
            values, found = np.unique(out, return_counts=True)
            assert dict(zip(values.tolist(), found.tolist())) == counts
            np.testing.assert_array_equal(mask, before)
            assert out is not mask

    with subtests.test(msg="a seed off the image returns the mask itself"):
        assert engine.magic_wand(image, owned, 40, 0, tolerance=5) is owned


def test_export_vvvv_refuses_unstorable_labels_and_records_its_manifest(
        tmp_path, monkeypatch, subtests):
    """Bad labels write nothing; a good field writes its files and exact manifest."""
    for case, labels, fragment in [
            ("a stack of labels", np.zeros((2, 4, 4), dtype=np.uint16),
             "vvvv export needs a 2-D label image"),
            ("negative ids", np.full((3, 3), -1, dtype=np.int32),
             "ids must lie between 0 and 65535"),
            ("ids past 16 bits", np.full((3, 3), 70000, dtype=np.int32),
             "vvvv export stores labels as 16-bit PNG")]:
        with subtests.test(msg=f"refused: {case}"):
            root = tmp_path / case
            root.mkdir()
            with pytest.raises(ValueError) as raised:
                engine._export_vvvv(root, "field.tif", labels)
            assert fragment in str(raised.value)
            assert list(root.iterdir()) == []

    with subtests.test(msg="no spaCR version is an empty version"):
        import spacr

        monkeypatch.delattr(spacr, "__version__", raising=False)
        labels = np.zeros((6, 6), dtype=np.uint16)
        labels[1:3, 1:3] = 2
        folder = Path(engine._export_vvvv(tmp_path / "out", "field.png",
                                          labels, np.ones((6, 6)),
                                          pixel_size=(1, 2), unit="nm"))
        assert folder == tmp_path / "out" / "field"
        assert sorted(p.name for p in folder.iterdir()) == [
            "README.txt", "field.vl", "field_image.png", "field_labels.png",
            "field_objects.csv", "field_outlines.png", "manifest.json"]
        manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
        stable = {key: manifest[key] for key in manifest
                  if key not in ("exported_at", "export_id", "coordinates")}
        assert stable == {
            "format": "spacr-vvvv-export", "version": 1, "image": "field.png",
            "source_path": "", "spacr_version": "", "width": 6, "height": 6,
            "pixel_size": {"x": 1.0, "y": 2.0, "unit": "nm"},
            "channel_names": ["channel_1"], "objects": 1, "vvvv_gamma": "7.4",
            "files": {"labels": "field_labels.png",
                      "outlines": "field_outlines.png",
                      "objects": "field_objects.csv",
                      "image": "field_image.png", "project": "field.vl",
                      "readme": "README.txt"}}
        np.testing.assert_array_equal(
            imageio.imread(folder / "field_labels.png"), labels)
