"""Item 600: what a drop on Make Masks is, decided without a widget.

Every case of the item's list A is built on disk with small TIFFs and fed to
:func:`spacr.drop_classification.classify_drop`; the channel-sorting changes
that let images from other folders join a sort (absolute names, masks in
their own folder's ``masks/``) are tested here too.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile

from spacr import channel_sorting as cs
from spacr import drop_classification as dc


def _tif(path: Path, array=None) -> str:
    """Write a TIFF (noisy 16-bit by default), making the folder.

    :param path: where.
    :param array: the pixels; default a noisy intensity image.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if array is None:
        array = np.random.default_rng(len(str(path))).integers(
            100, 4000, (16, 16)).astype(np.uint16)
    tifffile.imwrite(str(path), np.asarray(array))
    return str(path)


def _labels(value: int = 1) -> np.ndarray:
    """A small label image: two flat objects on background.

    :param value: the first label.
    """
    array = np.zeros((16, 16), np.uint16)
    array[2:6, 2:6] = value
    array[9:14, 8:13] = value + 1
    return array


def _channel_tree(root: Path, channels=("DAPI", "GFP"), fields=3,
                  masks=True) -> Path:
    """``exp/<channel>/field<i>.tif``, each channel with its masks.

    :param root: the parent.
    :param channels: subfolder names.
    :param fields: fields per channel.
    :param masks: write ``masks/`` in each channel folder.
    """
    exp = root / "exp"
    for channel in channels:
        for i in range(1, fields + 1):
            _tif(exp / channel / f"field{i}.tif")
            if masks:
                _tif(exp / channel / "masks" / f"field{i}.tif", _labels(i))
    return exp


# -- case 1: image files only ------------------------------------------------

def test_image_files_open_as_images_in_drop_order(tmp_path):
    b = _tif(tmp_path / "b.tif")
    a = _tif(tmp_path / "a.tif")
    found = dc.classify_drop([b, a])
    assert found.kind == "images"
    assert found.images == [b, a]
    assert found.unrecognised == []


# -- case 2: one folder of images --------------------------------------------

def test_one_folder_of_images_opens_as_it_is(tmp_path):
    _tif(tmp_path / "flat" / "a.tif")
    _tif(tmp_path / "flat" / "masks" / "a.tif", _labels())
    found = dc.classify_drop([tmp_path / "flat"])
    assert found.kind == "folder"
    assert found.folders == [str(tmp_path / "flat")]


# -- case 3: images in subfolders --------------------------------------------

def test_subfolders_named_as_channels_are_channel_like(tmp_path):
    exp = _channel_tree(tmp_path)
    found = dc.classify_drop([exp])
    assert found.kind == "nested"
    assert found.channel_like
    assert found.channel_folders == [str(exp / "DAPI"), str(exp / "GFP")]


def test_subfolders_with_the_same_fields_are_channel_like(tmp_path):
    exp = _channel_tree(tmp_path, channels=("stainA", "stainB"))
    found = dc.classify_drop([exp])
    assert found.kind == "nested" and found.channel_like


def test_channel_words_in_file_names_do_not_hide_matching_fields(tmp_path):
    exp = tmp_path / "exp"
    for folder, word in (("first", "dapi"), ("second", "gfp")):
        for i in (1, 2):
            _tif(exp / folder / f"f{i}_{word}.tif")
    assert dc.classify_drop([exp]).channel_like


def test_subfolders_of_different_fields_are_not_channel_like(tmp_path):
    exp = tmp_path / "exp"
    _tif(exp / "monday" / "x1.tif")
    _tif(exp / "tuesday" / "y7.tif")
    found = dc.classify_drop([exp])
    assert found.kind == "nested"
    assert not found.channel_like


def test_one_subfolder_is_never_channel_like(tmp_path):
    exp = tmp_path / "exp"
    _tif(exp / "DAPI" / "a.tif")
    found = dc.classify_drop([exp])
    assert found.kind == "nested" and not found.channel_like


def test_a_folder_with_top_images_and_nested_ones_is_nested(tmp_path):
    exp = _channel_tree(tmp_path)
    _tif(exp / "top.tif")
    assert dc.classify_drop([exp]).kind == "nested"


def test_masks_and_spacr_folders_do_not_make_a_folder_nested(tmp_path):
    _tif(tmp_path / "flat" / "a.tif")
    _tif(tmp_path / "flat" / "masks" / "a.tif", _labels())
    _tif(tmp_path / "flat" / "orig" / "a.tif")
    _tif(tmp_path / "flat" / "sorted_channels_2" / "x.tif")
    assert dc.classify_drop([tmp_path / "flat"]).kind == "folder"


# -- case 4: several folders -------------------------------------------------

def test_several_image_folders_are_offered_as_channels_in_drop_order(tmp_path):
    exp = _channel_tree(tmp_path)
    found = dc.classify_drop([exp / "GFP", exp / "DAPI"])
    assert found.kind == "folders"
    assert found.folders == [str(exp / "GFP"), str(exp / "DAPI")]
    assert found.channel_folders == found.folders
    assert found.channel_like


def test_a_folder_with_no_top_images_in_a_multi_drop_is_listed(tmp_path):
    exp = _channel_tree(tmp_path)
    found = dc.classify_drop([exp / "DAPI", exp / "GFP", exp])
    assert found.kind == "folders"
    assert any(str(exp) in line and "no images" in line
               for line in found.unrecognised)


def test_folders_and_loose_images_open_as_a_queue(tmp_path):
    exp = _channel_tree(tmp_path)
    loose = _tif(tmp_path / "loose.tif")
    found = dc.classify_drop([exp / "DAPI", loose])
    assert found.kind == "images"
    assert found.images == [loose]
    assert found.folders == [str(exp / "DAPI")]


# -- case 5: images dropped with their masks ---------------------------------

def test_a_masks_folder_dropped_with_images_is_paired(tmp_path):
    a = _tif(tmp_path / "imgs" / "a.tif")
    b = _tif(tmp_path / "imgs" / "b.tif")
    ma = _tif(tmp_path / "drawn_masks" / "a.tif", _labels())
    mb = _tif(tmp_path / "drawn_masks" / "b.tif", _labels())
    found = dc.classify_drop([a, b, tmp_path / "drawn_masks"])
    assert found.kind == "images_with_masks"
    assert found.masks == {a: ma, b: mb}
    assert found.images == [a, b]


def test_mask_named_files_pair_by_stem_and_leftovers_are_listed(tmp_path):
    a = _tif(tmp_path / "a.tif")
    b = _tif(tmp_path / "b.tif")
    ma = _tif(tmp_path / "out" / "a_cp_masks.tif", _labels())
    mb = _tif(tmp_path / "out" / "mask_b.tif", _labels())
    stray = _tif(tmp_path / "out" / "zzz_mask.tif", _labels())
    found = dc.classify_drop([a, b, ma, mb, stray])
    assert found.kind == "images_with_masks"
    assert found.masks == {a: ma, b: mb}
    assert found.unpaired_masks == [stray]


def test_a_folder_of_images_with_a_masks_folder_dropped_beside_it(tmp_path):
    _tif(tmp_path / "imgs" / "a.tif")
    ma = _tif(tmp_path / "labels_masks" / "a.tif", _labels())
    found = dc.classify_drop([tmp_path / "imgs", tmp_path / "labels_masks"])
    assert found.kind == "images_with_masks"
    assert found.masks == {str(tmp_path / "imgs" / "a.tif"): ma}


def test_integer_label_tiffs_are_recognised_by_their_pixels(tmp_path):
    a = _tif(tmp_path / "a.tif")
    b = _tif(tmp_path / "b.tif")
    la = _tif(tmp_path / "other" / "a.tif", _labels())
    found = dc.classify_drop([a, b, la])
    assert found.kind == "images_with_masks"
    assert found.masks == {a: la}
    assert found.images == [a, b]


def test_the_label_test_tells_labels_from_intensities(tmp_path):
    assert dc._looks_like_labels(_tif(tmp_path / "l.tif", _labels()))
    assert not dc._looks_like_labels(_tif(tmp_path / "i.tif"))
    assert not dc._looks_like_labels(
        _tif(tmp_path / "f.tif", np.zeros((8, 8), np.float32)))
    assert not dc._looks_like_labels(str(tmp_path / "missing.tif"))
    png = tmp_path / "p.png"
    png.write_bytes(b"")
    assert not dc._looks_like_labels(str(png))


def test_label_tiffs_alone_are_opened_as_images(tmp_path):
    la = _tif(tmp_path / "a.tif", _labels())
    lb = _tif(tmp_path / "b.tif", _labels(3))
    found = dc.classify_drop([la, lb])
    assert found.kind == "images" and found.images == [la, lb]


def test_masks_nobody_claims_leave_the_images_as_images(tmp_path):
    a = _tif(tmp_path / "a.tif")
    stray = _tif(tmp_path / "zzz_mask.tif", _labels())
    found = dc.classify_drop([a, stray])
    assert found.kind == "images"
    assert found.images == [a]
    assert any("zzz_mask" in line for line in found.unrecognised)


def test_a_masks_folder_alone_opens_the_images_it_belongs_to(tmp_path):
    _tif(tmp_path / "flat" / "a.tif")
    _tif(tmp_path / "flat" / "masks" / "a.tif", _labels())
    found = dc.classify_drop([tmp_path / "flat" / "masks"])
    assert found.kind == "folder"
    assert found.folders == [str(tmp_path / "flat")]
    assert "masks folder" in found.description


# -- case 6: spaCR's own output ----------------------------------------------

def test_a_sorted_channels_folder_opens_its_first_channel(tmp_path):
    dest = tmp_path / "sorted_channels"
    _tif(dest / "C01" / "plate1_A01_T0001F001L01A01Z01C01.tif")
    _tif(dest / "C02" / "plate1_A01_T0001F001L01A01Z01C02.tif")
    found = dc.classify_drop([dest])
    assert found.kind == "spacr_output"
    assert found.open_folder == str(dest / "C01")
    assert "sorted by spaCR" in found.description


def test_a_renamed_sort_is_known_by_its_manifest(tmp_path):
    dest = tmp_path / "my_sort"
    _tif(dest / "C01" / "x.tif")
    (dest / cs.MANIFEST_NAME).write_text("kind\n")
    assert dc.classify_drop([dest]).open_folder == str(dest / "C01")


def test_a_merged_folder_opens_its_images_and_never_the_npy(tmp_path):
    plate = tmp_path / "plate"
    (plate / "merged").mkdir(parents=True)
    np.save(plate / "merged" / "f.npy", np.zeros((4, 4, 2)))
    _tif(plate / "orig" / "f.tif")
    found = dc.classify_drop([plate])
    assert found.kind == "spacr_output"
    assert found.open_folder == str(plate / "orig")
    assert ".npy" in found.description


def test_merged_arrays_without_images_open_nothing(tmp_path):
    plate = tmp_path / "plate"
    (plate / "merged").mkdir(parents=True)
    np.save(plate / "merged" / "f.npy", np.zeros((4, 4, 2)))
    found = dc.classify_drop([plate / "merged"])
    assert found.kind == "spacr_output"
    assert found.open_folder is None
    assert "Measure" in found.description


def test_a_plate_folder_with_images_and_merged_opens_its_images(tmp_path):
    plate = tmp_path / "plate"
    _tif(plate / "a.tif")
    (plate / "merged").mkdir()
    np.save(plate / "merged" / "f.npy", np.zeros((4, 4, 2)))
    assert dc.classify_drop([plate]).open_folder == str(plate)


# -- unrecognised ------------------------------------------------------------

def test_npy_text_and_missing_paths_are_listed_never_opened(tmp_path):
    npy = tmp_path / "x.npy"
    np.save(npy, np.zeros(3))
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    empty = tmp_path / "empty"
    empty.mkdir()
    found = dc.classify_drop([npy, txt, empty, tmp_path / "gone.tif"])
    assert found.kind == "nothing"
    text = "\n".join(found.unrecognised)
    assert ".npy array, not an image" in text
    assert "notes.txt" in text and "not found" in text
    assert "empty" in text
    assert len(found.unrecognised) == 4


def test_an_image_with_an_npy_opens_the_image_and_lists_the_npy(tmp_path):
    a = _tif(tmp_path / "a.tif")
    npy = tmp_path / "x.npy"
    np.save(npy, np.zeros(3))
    found = dc.classify_drop([a, npy])
    assert found.kind == "images" and found.images == [a]
    assert len(found.unrecognised) == 1


def test_the_drop_handler_policy(tmp_path):
    exp = _channel_tree(tmp_path)
    npy = tmp_path / "x.npy"
    np.save(npy, np.zeros(3))
    (tmp_path / "empty").mkdir()
    (tmp_path / "t.txt").write_text("x")
    assert dc._accepts(str(exp))
    assert dc._accepts(str(exp / "DAPI"))
    assert dc._accepts(str(npy))
    assert not dc._accepts(str(tmp_path / "empty"))
    assert not dc._accepts(str(tmp_path / "t.txt"))
    assert not dc._accepts(str(tmp_path / "gone"))


def test_folder_images_are_listed_recursively_without_masks(tmp_path):
    exp = _channel_tree(tmp_path)
    found = dc._folder_images(str(exp))
    assert len(found) == 6
    assert all("masks" not in p for p in found)
    assert dc._folder_images(str(exp), recursive=False) == []


# -- images from other folders in a sort -------------------------------------

def test_an_absolute_name_finds_its_mask_in_its_own_folder(tmp_path):
    exp = _channel_tree(tmp_path)
    image = str(exp / "DAPI" / "field1.tif")
    assert cs.mask_for(str(exp), image) == str(exp / "DAPI" / "masks"
                                               / "field1.tif")
    # An absolute name inside the folder still follows masks_dir.
    other = tmp_path / "elsewhere"
    _tif(other / "field1.tif", _labels())
    assert cs.mask_for(str(exp / "DAPI"), image, str(other)) == str(
        other / "field1.tif")


def test_the_regex_reads_the_file_name_of_an_absolute_path(tmp_path):
    parsed = cs.parse_names(["/x/DAPI/f1.tif"], r"f(?P<fieldID>\d+)\.tif")
    assert parsed[0].matched and parsed[0].groups["fieldID"] == "1"
    assert cs.name_distance("/a/DAPI/f1.tif", "/b/GFP/f1.tif") == 0.0


def test_two_folders_as_two_channels_sort_and_merge(tmp_path):
    exp = _channel_tree(tmp_path)
    channels = {str(exp / ch / f"field{i}.tif"): k + 1
                for k, ch in enumerate(("DAPI", "GFP")) for i in range(1, 4)}
    regex = cs.infer_regex(list(channels), channels)
    assert regex
    report = cs.check_sets(cs.parse_names(list(channels), regex, channels))
    assert report.ok and len(report.complete) == 3
    plan = cs.build_plan(str(exp), report.complete)
    assert plan.ok, plan.problems
    assert plan.mask_roles == {1: "nucleus", 2: "cell"}
    result = cs.apply_plan(plan, log=lambda _t: None)
    dest = Path(result.dest)
    assert dest == exp / "sorted_channels"
    c01 = sorted(os.listdir(dest / "C01"))
    assert "plate1_A01_T0001F001L01A01Z01C01.tif" in c01
    assert len(result.merged) == 3
    merged = np.load(result.merged[0])
    assert merged.shape == (16, 16, 4)
    assert not list((exp / "DAPI").glob("*.tif"))
    manifest = (dest / cs.MANIFEST_NAME).read_text()
    assert str(exp / "DAPI" / "field1.tif") in manifest


def test_an_unclaimed_label_like_tiff_stays_an_image(tmp_path):
    a = _tif(tmp_path / "a.tif")
    z = _tif(tmp_path / "other" / "z.tif", _labels())
    found = dc.classify_drop([a, z])
    assert found.kind == "images" and found.images == [a, z]


def test_masks_folder_names():
    for name in ("masks", "masks_2", "cell_masks", "drawn-mask", "cp_masks"):
        assert dc._is_masks_folder_name(name), name
    for name in ("maskless_run", "test_mask_run", "images", "unmasked"):
        assert not dc._is_masks_folder_name(name), name


# -- "Teach me" inference ----------------------------------------------------

def _oracle(name):
    """Channel 2 for a stem ending in ``c`` (the autophagy image), else 1.

    :param name: a file name.
    """
    return (("channel", 2) if cs.split_extension(name)[0].endswith("c")
            else ("channel", 1))


ENZ_NAMES = [f"Experiment {e}_{s}_{f}{c}.{x}"
             for e, x in ((1, "tif"), (2, "jpg"))
             for s in ("ATG2 KO", "ATG2' KO", "ME49")
             for f in (1, 2, 3) for c in ("", "c")]


def test_teach_learns_the_channel_marker_in_three_answers():
    answers = {}
    for _ in range(10):
        pattern, by_marker, nxt = cs._teach_step(ENZ_NAMES, answers)
        if nxt is None:
            break
        answers[nxt] = _oracle(nxt)
    assert len(answers) == 3
    assert by_marker == {"": ("channel", 1), "c": ("channel", 2)}
    labels = cs._teach_labels(ENZ_NAMES, pattern, by_marker)
    assert labels == {n: _oracle(n) for n in ENZ_NAMES}
    report = cs.check_sets(cs.parse_names(
        ENZ_NAMES, pattern, {n: lab[1] for n, lab in labels.items()}))
    assert report.ok and len(report.complete) == 18


def test_teach_learns_a_marker_in_the_middle_and_a_mask():
    names = [f"img_{m}_{i:02d}.tif" for m in ("DAPI", "GFP", "mask")
             for i in (1, 2, 3)]
    oracle = {"DAPI": ("channel", 1), "GFP": ("channel", 2),
              "mask": ("mask", 1, "nucleus")}
    answers = {}
    for _ in range(12):
        pattern, by_marker, nxt = cs._teach_step(names, answers)
        if nxt is None:
            break
        answers[nxt] = oracle[nxt.split("_")[1]]
    assert set(by_marker) == {"DAPI", "GFP", "mask"}
    labels = cs._teach_labels(names, pattern, by_marker)
    assert all(labels[n] == oracle[n.split("_")[1]] for n in names)


def test_teach_asks_about_a_name_the_regex_does_not_read():
    names = ["a_1.tif", "a_1c.tif", "b_7.png", "b_7C.png"]
    answers = {"a_1.tif": ("channel", 1), "a_1c.tif": ("channel", 2)}
    pattern, _markers, nxt = cs._teach_step(names, answers)
    assert pattern and nxt == "b_7C.png"
    answers[nxt] = ("channel", 2)
    # The regex cannot read that answer yet, so its neighbour is asked next.
    pattern, by_marker, nxt = cs._teach_step(names, answers)
    assert nxt == "b_7.png"
    answers[nxt] = ("channel", 1)
    pattern, by_marker, nxt = cs._teach_step(names, answers)
    assert nxt is None and by_marker["C"] == ("channel", 2)
    assert cs._teach_labels(names, pattern, by_marker) == answers


def test_teach_skip_and_start_and_nothing_to_learn():
    assert cs._teach_step([], {}) == (None, {}, None)
    assert cs._teach_step(["a.tif"], {})[2] == "a.tif"
    assert cs._teach_step(["a.tif", "b.tif"], {"a.tif": "skip"})[2] == "b.tif"
    only = {"a.tif": ("channel", 1), "b.tif": ("channel", 1)}
    assert cs._teach_step(["a.tif", "b.tif"], only) == (None, {}, None)
    clash = {"x_1.tif": ("channel", 1), "x_1c.tif": ("channel", 2),
             "y_1c.tif": ("channel", 1), "y_1.tif": ("channel", 2)}
    assert cs._teach_regex(clash, [".tif"]) == (None, {})


def test_diff_region():
    assert cs._diff_region("a_DAPI_1.tif", "a_GFP_1.tif") == ("DAPI", "GFP", "_1")
    assert cs._diff_region("k_1.tif", "k_1c.tif") == ("", "c", "")
    assert cs._diff_region("a1_x.tif", "b2_y.tif") is None
    assert cs._diff_region("same.tif", "same.tif") is None


def test_original_names_come_from_the_consolidation_manifest(tmp_path):
    manifest = tmp_path / cs.CONSOLIDATION_MANIFEST
    manifest.write_text(
        "original_path,new_filename,status,error\n"
        "/d/enzo/E1/A/1.tif,a.tif,copied,\n"
        "/d/enzo/E2/B/1c.tif,b.tif,copied,\n", encoding="utf-8")
    assert cs._original_names(str(tmp_path)) == {
        "a.tif": "E1_A_1.tif", "b.tif": "E2_B_1c.tif"}
    manifest.write_text(
        "original_path,new_filename,status,error\n"
        "/d/x/1.tif,k.tif,copied,\n", encoding="utf-8")
    assert cs._original_names(str(tmp_path)) == {"k.tif": "1.tif"}
    assert cs._original_names(str(tmp_path / "none")) == {}
    manifest.write_text("x\n", encoding="utf-8")
    assert cs._original_names(str(tmp_path)) == {}


def test_organelle_is_a_mask_role_and_puncta_suggest_it():
    assert "organelle" in cs.MASK_ROLES
    assert cs.default_mask_roles({1: ["cyst_1.tif"], 2: ["puncta_1.tif"]},
                                 [1, 2]) == {1: "cell", 2: "organelle"}
