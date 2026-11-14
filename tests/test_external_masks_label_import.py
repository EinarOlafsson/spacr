"""Table-driven tests of the label-image import in ``spacr.external_masks``.

The import pairs raw channel images with another tool's label images by
filename, plans the pairing without reading pixels, and then writes spaCR's
own ``stack/``, ``masks/`` and ``merged/`` arrays for Measure.  These tests
walk case tables with pytest's ``subtests`` fixture: every planning error and
warning is compared as the exact message, every refusal of the write step as
its exact text, and every successful import by the files it leaves and the
settings it hands Measure.  The saved-group normaliser
``InputGroup.from_value`` is covered here as well.  Files are tiny and written
per case.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from spacr import external_masks as em  # noqa: E402
from spacr.errors import ConfigurationError  # noqa: E402

IMAGE_REGEX = r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_w(?P<chanID>\d)"
TIMED_REGEX = (r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_t(?P<timeID>\d+)"
               r"_w(?P<chanID>\d)")
CELL_REGEX = r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_Cell\.tiff$"


def _touch(folder, names):
    """Create empty files ``names`` in ``folder``; return the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return folder


# ---------------------------------------------------------------------------
# _plan_label_import
# ---------------------------------------------------------------------------

#: case id, image files, label files (None = no label folder), extra settings,
#: expected errors, warnings, skipped, channels, objects and ready stems.
#: ``{img}``/``{lab}`` stand for the case's image and label folders.
PLAN_CASES = [
    ("one complete field", ["B02_s1_w1.tif"], ["B02_s1_Cell.tiff"], {},
     [], [], [], ["1"], ["cell"], ["img_B02_1"]),
    ("a second file for one channel, and an unrelated file",
     ["B02_s1_w1.tif", "B02_s1_w1_copy.tif", "unrelated.tif"], ["B02_s1_Cell.tiff"], {},
     ["{img}/B02_s1_w1.tif and {img}/B02_s1_w1_copy.tif are both channel 1 of img_B02_1."],
     ["1 file(s) in {img} do not match the image pattern and are not imported."],
     [], ["1"], ["cell"], ["img_B02_1"]),
    ("no image matches the pattern", ["unrelated.tif"], ["B02_s1_Cell.tiff"], {},
     [f"No file in {{img}} matches label_import_image_regex {IMAGE_REGEX!r}.",
      f"No file in {{lab}} matches cell_label_regex {CELL_REGEX!r} and an image field."],
     ["1 file(s) in {img} do not match the image pattern and are not imported.",
      "1 cell label file(s) match the pattern but no image field; they are not imported."],
     [], [], ["cell"], []),
    ("an unknown channel and no label folder", ["B02_s1_w1.tif"], None,
     {"label_import_channels": [1, 9]},
     ["label_import_channels names channel(s) 9; the images carry 1.",
      "label_import is on but no label folder is set: fill cell_label_folder, "
      "nucleus_label_folder or pathogen_label_folder."],
     ["1 field(s) lack a channel or a label and are not imported, e.g. img_B02_1: no 9."],
     ["img_B02_1: no 9"], ["1", "9"], [], []),
    ("label files fitting several, no, or a repeated field",
     ["B02_s1_t1_w1.tif", "B02_s1_t2_w1.tif", "C03_s1_t1_w1.tif"],
     ["B02_s1_Cell.tiff", "P99_s1_Cell.tiff", "C03_s1_Cell.tiff",
      "other_C03_s1_Cell.tiff", "junk.tiff"],
     {"label_import_image_regex": TIMED_REGEX},
     ["{lab}/B02_s1_Cell.tiff fits 2 image fields; add plateID or timeID to cell_label_regex.",
      "{lab}/C03_s1_Cell.tiff and {lab}/other_C03_s1_Cell.tiff are both cell labels "
      "of one field; make cell_label_regex match only one of them."],
     ["1 cell label file(s) match the pattern but no image field; they are not imported.",
      "2 field(s) lack a channel or a label and are not imported, e.g. img_B02_1_1: no cell."],
     ["img_B02_1_1: no cell", "img_B02_1_2: no cell"], ["1"], ["cell"], ["img_C03_1_1"]),
    ("a label folder with no matching file", ["B02_s1_w1.tif"], ["nothing.tiff"], {},
     [f"No file in {{lab}} matches cell_label_regex {CELL_REGEX!r} and an image field."],
     ["1 field(s) lack a channel or a label and are not imported, e.g. img_B02_1: no cell."],
     ["img_B02_1: no cell"], ["1"], ["cell"], []),
    ("every field misses something", ["B02_s1_w1.tif", "C03_s1_w2.tif"],
     ["B02_s1_Cell.tiff", "C03_s1_Cell.tiff"], {},
     ["No field has every channel and every label."],
     ["2 field(s) lack a channel or a label and are not imported, e.g. img_B02_1: no 2."],
     ["img_B02_1: no 2", "img_C03_1: no 1"], ["1", "2"], ["cell"], []),
    ("channels sort numerically and padded ids lose their zeros",
     ["B02_s01_w2.tif", "B02_s01_w1.tif"], ["b02_s1_Cell.tiff"],
     {"cell_label_regex": r"(?P<wellID>[a-pA-P]\d{2})_s(?P<fieldID>\d+)_Cell\.tiff$"},
     [], [], [], ["1", "2"], ["cell"], ["img_B02_1"]),
]


def test_plan_label_import_names_every_problem_exactly(tmp_path, subtests):
    """Each folder layout gives exactly these errors, warnings, skipped and ready fields."""
    for index, (case_id, images, labels, extra, errors, warnings, skipped,
                channels, objects, ready) in enumerate(PLAN_CASES):
        with subtests.test(msg=case_id):
            root = tmp_path / f"case{index}"
            img = _touch(root / "img", images)
            src = root / "work" / "plate"
            src.mkdir(parents=True)
            settings = {"src": str(src), "label_import_images": str(img),
                        "label_import_image_regex": IMAGE_REGEX}
            lab = root / "lab"
            if labels is not None:
                _touch(lab, labels)
                settings.update(cell_label_folder=str(lab), cell_label_regex=CELL_REGEX)
            settings.update(extra)

            plan = em._plan_label_import(settings)

            def fill(texts):
                return [text.replace("{img}", str(img)).replace("{lab}", str(lab))
                        for text in texts]
            assert plan["errors"] == fill(errors)
            assert plan["warnings"] == fill(warnings)
            assert plan["skipped"] == skipped
            assert plan["channels"] == channels
            assert plan["objects"] == objects
            assert [stem for stem, _paths, _labels in plan["fields"]] == ready
            assert plan["src"] == str(src)
            assert plan["image_folder"] == str(img)
            for _stem, paths, found in plan["fields"]:
                assert [os.path.dirname(path) for path in paths] == [str(img)] * len(channels)
                assert list(found) == objects


def test_plan_label_import_refuses_an_unusable_src_or_pattern(tmp_path, subtests):
    """Bad src values and unusable patterns raise with their exact message; good src forms resolve."""
    images = _touch(tmp_path / "img", ["B02_s1_w1.tif"])
    named = _touch(tmp_path / "named", ["x.tif", "y.tif"])
    empty = _touch(tmp_path / "empty", [])
    plate = tmp_path / "work" / "plate"
    (plate / "merged").mkdir(parents=True)
    try:
        re.compile("(unclosed")
    except re.error as exc:
        unclosed = str(exc)

    def settings(folder, regex=IMAGE_REGEX, src=str(plate)):
        return {"src": src, "label_import_images": str(folder),
                "label_import_image_regex": regex}

    inferred_tail = (" Type a regular expression with those named groups, for example "
                     r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+).")
    refusals = [
        ("two plate folders", {"src": ["a", "b"], "label_import_images": str(images)},
         "label_import builds one plate folder at a time; src names 2. "
         "Measure each plate in its own run."),
        ("a blank src", {"src": "  "},
         "label_import needs src: the plate folder the merged arrays and "
         "measurements are written to."),
        ("no src at all", {},
         "label_import needs src: the plate folder the merged arrays and "
         "measurements are written to."),
        ("a pattern that does not compile", settings(images, "(unclosed"),
         f"label_import_image_regex: '(unclosed' is not a regular expression ({unclosed})."),
        ("a pattern missing a group", settings(images, r"(?P<wellID>[A-P]\d{2})"),
         "label_import_image_regex: the pattern has no (?P<fieldID>...), "
         "(?P<chanID>...) group; it needs wellID, fieldID, chanID."),
        ("no pattern can be inferred from the names", settings(named, ""),
         "label_import_image_regex: no filename pattern with wellID, fieldID, chanID "
         "could be inferred from 2 file(s) such as x.tif." + inferred_tail),
        ("no pattern can be inferred from no files", settings(empty, ""),
         "label_import_image_regex: no filename pattern with wellID, fieldID, chanID "
         "could be inferred from 0 file(s)." + inferred_tail),
        ("an image folder that is not a folder", settings(tmp_path / "absent"),
         f"{tmp_path / 'absent'} is not a folder."),
    ]
    for case_id, values, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ConfigurationError) as caught:
                em._plan_label_import(values)
            assert str(caught.value) == message

    accepted = [
        ("a one-item list", [str(plate)]),
        ("a merged folder is read as its plate", str(plate / "merged")),
        ("a trailing separator", str(plate) + os.sep),
    ]
    for case_id, src in accepted:
        with subtests.test(msg=case_id):
            assert em._plan_label_import(settings(images, src=src))["src"] == str(plate)


def test_label_import_files_lists_importable_files_only(tmp_path):
    """Dot-files and unsupported extensions are skipped; spaCR output folders are not searched."""
    folder = _touch(tmp_path / "f", ["b.tif", "a.tif", ".hidden.tif", "notes.txt", "c.npy"])
    _touch(folder / "sub", ["d.tif"])
    _touch(folder / "merged", ["e.npy"])
    _touch(folder / ".cache", ["f.tif"])
    flat = em._label_import_files(str(folder), recursive=False)
    deep = em._label_import_files(str(folder), recursive=True)
    assert [os.path.relpath(p, folder) for p in flat] == ["a.tif", "b.tif", "c.npy"]
    assert [os.path.relpath(p, folder) for p in deep] == [
        "a.tif", "b.tif", "c.npy", os.path.join("sub", "d.tif")]


# ---------------------------------------------------------------------------
# _import_label_images_for_measure
# ---------------------------------------------------------------------------

def _drop(root, shapes, label_shape=(8, 8)):
    """Real one-field images (one per shape) and one cell label; return the settings."""
    images = root / "img"
    labels = root / "lab"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for channel, shape in enumerate(shapes, start=1):
        tifffile.imwrite(images / f"B02_s1_w{channel}.tif",
                         np.full(shape, channel, np.uint16))
    label = np.zeros(label_shape, np.uint16)
    label[..., 2:6, 2:6] = 1
    tifffile.imwrite(labels / "B02_s1_Cell.tiff", label)
    src = root / "work" / "plate"
    src.mkdir(parents=True)
    return {"src": str(src), "label_import_images": str(images),
            "label_import_image_regex": IMAGE_REGEX,
            "cell_label_folder": str(labels), "cell_label_regex": CELL_REGEX}


def test_import_label_images_refuses_before_or_while_writing(tmp_path, monkeypatch, subtests):
    """Plan errors, multi-plane files, mismatched channels and an empty merge are refused exactly."""
    from spacr import io

    cases = [
        ("a plan with errors writes nothing", [(8, 8)], (8, 8),
         {"label_import_channels": [5]}, None,
         "Label-image import refused; nothing was written:\n"
         "  label_import_channels names channel(s) 5; the images carry 1."),
        ("a multi-plane label file", [(8, 8)], (2, 8, 8), {}, None,
         "{lab}/B02_s1_Cell.tiff: a cell label file must hold one 2-D plane, got "
         "shape (1, 1, 2, 8, 8). Export one plane per file."),
        ("channels of different sizes", [(8, 8), (9, 8)], (8, 8), {}, None,
         "img_B02_1: intensity channels have inconsistent shapes ((8, 8), (9, 8))."),
        ("a merge that writes nothing", [(8, 8)], (8, 8), {},
         lambda *args, **kwargs: None,
         "The merge wrote no merged array for 1 field(s), starting with img_B02_1."),
    ]
    for index, (case_id, shapes, label_shape, extra, merge, message) in enumerate(cases):
        with subtests.test(msg=case_id):
            root = tmp_path / f"case{index}"
            settings = dict(_drop(root, shapes, label_shape), **extra)
            with monkeypatch.context() as patch:
                if merge is not None:
                    patch.setattr(io, "_load_and_concatenate_arrays", merge)
                with pytest.raises(ConfigurationError) as caught:
                    em._import_label_images_for_measure(settings)
            assert str(caught.value) == message.replace("{lab}", str(root / "lab"))
            assert not os.path.exists(os.path.join(settings["src"], "label_import.json"))
            if index == 0:
                assert os.listdir(settings["src"]) == []

    with subtests.test(msg="arrays the import did not write are refused"):
        settings = _drop(tmp_path / "foreign", [(8, 8)])
        foreign = os.path.join(settings["src"], "stack", "other.npy")
        os.makedirs(os.path.dirname(foreign))
        np.save(foreign, np.zeros((2, 2)))
        with pytest.raises(ConfigurationError) as caught:
            em._import_label_images_for_measure(settings)
        assert str(caught.value).startswith(
            f"{settings['src']} already holds 1 array(s) that this import did not "
            f"write, starting with {foreign}.")


def test_import_label_images_writes_reuses_and_rebuilds_a_plate(tmp_path, subtests):
    """A good import writes stack/masks/merged, is reused unchanged, rebuilds after a change."""
    settings = _drop(tmp_path, [(8, 8), (8, 8)])
    src = settings["src"]

    with subtests.test(msg="first import"):
        resolved, manifest = em._import_label_images_for_measure(settings)
        assert manifest["written"] == [
            os.path.join("stack", "img_B02_1.npy"),
            os.path.join("masks", "cell_mask_stack", "img_B02_1.npy"),
            os.path.join("merged", "img_B02_1.npy")]
        assert manifest["channels"] == ["1", "2"]
        assert manifest["skipped"] == []
        assert resolved["src"] == src
        assert (resolved["cell_mask_dim"], resolved["nucleus_mask_dim"],
                resolved["pathogen_mask_dim"]) == (2, None, None)
        stack = np.load(os.path.join(src, "stack", "img_B02_1.npy"))
        np.testing.assert_array_equal(stack[0, 0], [1, 2])
        mask = np.load(os.path.join(src, "masks", "cell_mask_stack", "img_B02_1.npy"))
        assert int(mask.sum()) == 16
        assert os.path.isfile(os.path.join(src, "label_import.json"))

    with subtests.test(msg="the same files are reused, not rewritten"):
        merged = os.path.join(src, "merged", "img_B02_1.npy")
        before = os.stat(merged).st_mtime_ns
        _again, reused = em._import_label_images_for_measure(settings)
        assert reused == manifest
        assert os.stat(merged).st_mtime_ns == before

    with subtests.test(msg="a rebuild tolerates its own files being gone"):
        os.remove(os.path.join(src, manifest["written"][0]))
        rebuilt, again = em._import_label_images_for_measure(
            dict(settings, label_import_channels=[1]))
        assert again["channels"] == ["1"]
        assert rebuilt["cell_mask_dim"] == 1
        assert os.path.isfile(os.path.join(src, "merged", "img_B02_1.npy"))
        assert np.load(os.path.join(src, "stack", "img_B02_1.npy")).shape == (8, 8, 1)

    requested = [
        ("channels within the import are left alone", [0, 1], [0, 1]),
        ("channels past the import are dropped", [0, 5], [0]),
        ("only channels past the import measure them all", [5, 6], [0, 1]),
        ("non-integer entries are kept", ["a", 7], ["a"]),
        ("no channels setting stays absent", None, None),
    ]
    for case_id, channels, expected in requested:
        with subtests.test(msg=case_id):
            values = dict(settings)
            if channels is not None:
                values["channels"] = channels
            resolved, _manifest = em._import_label_images_for_measure(values)
            assert resolved.get("channels") == expected


# ---------------------------------------------------------------------------
# InputGroup.from_value
# ---------------------------------------------------------------------------

def test_input_group_from_value_normalises_a_saved_group(tmp_path, subtests):
    """Mappings become groups with absolute paths, safe defaults and a known object type only."""
    existing = em.InputGroup(key="k", root="/r", paths=["/r/a.tif"], role="mask",
                             object_type="cell")
    with subtests.test(msg="a group is returned as it is"):
        assert em.InputGroup.from_value(existing) is existing

    cases = [
        ("an unknown object type is dropped",
         {"paths": ["/tmp/a_mask.tif"], "role": "mask", "object_type": "mitochondrion"},
         {"object_type": None, "role": "mask", "paths": ["/tmp/a_mask.tif"]}),
        ("a recognised object type survives",
         {"paths": ["/tmp/a_mask.tif"], "role": "mask", "object_type": em.OBJECT_TYPES[0]},
         {"object_type": em.OBJECT_TYPES[0]}),
        ("an empty object type is none", {"object_type": ""}, {"object_type": None}),
        ("defaults fill every missing field", {},
         {"key": "", "root": os.path.abspath("."), "paths": [], "role": "ignore",
          "object_type": None, "confidence": 0.0, "reason": ""}),
        ("values are coerced", {"key": 3, "root": "rel", "paths": ["x.tif"],
                                "confidence": "0.5", "reason": None},
         {"key": "3", "root": os.path.abspath("rel"),
          "paths": [os.path.abspath("x.tif")], "confidence": 0.5, "reason": ""}),
    ]
    for case_id, value, expected in cases:
        with subtests.test(msg=case_id):
            group = em.InputGroup.from_value(value)
            for field, wanted in expected.items():
                assert getattr(group, field) == wanted, field
            assert em.InputGroup.from_value(group.to_dict()) == group

    for case_id, value in [("a string", "folder"), ("a list", ["a"]), ("none", None)]:
        with subtests.test(msg=f"{case_id} is refused"):
            with pytest.raises(ConfigurationError) as caught:
                em.InputGroup.from_value(value)
            assert str(caught.value) == (
                "Each external-mask input must be an InputGroup or mapping.")
    assert "mitochondrion" not in em.OBJECT_TYPES
