"""Channel sorting at its edges: unreadable files, odd names, refused plans."""
from __future__ import annotations

import builtins
import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from spacr import channel_sorting as cs


def _tif(path: Path, array) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.asarray(array))
    return path


def _pair_folder(root: Path, n=2, shape=(6, 8), masks=True) -> tuple:
    folder = root / "imgs"
    sets = {}
    for i in range(n):
        a, b = f"s{i + 1}_nuc.tif", f"s{i + 1}_cyto.tif"
        _tif(folder / a, np.full(shape, i + 1, np.uint16))
        _tif(folder / b, np.full(shape, 10 + i, np.uint16))
        if masks:
            _tif(folder / "masks" / a, np.full(shape, 1, np.uint16))
        sets[(("set", f"{i + 1:06d}"),)] = {1: a, 2: b}
    return folder, sets


# --------------------------------------------------------------- file helpers

def test_names_and_listing():
    assert cs.split_extension("a.OME.TIF") == ("a", ".OME.TIF")
    assert cs.split_extension("a.png") == ("a", ".png")
    assert cs.list_folder_images("") == []
    assert cs.list_folder_images("/no/such/folder") == []


def test_plane_conversion_kinds_and_shapes():
    assert cs._plane_conversion(None) is None
    assert cs._plane_conversion((4, 4)) is None
    assert cs._plane_conversion((4, 4, 3)) == "rgb"
    assert cs._plane_conversion((5, 4, 6)) == "zstack"
    assert cs._converted_shape((4, 5), None) == (4, 5)
    assert cs._converted_shape((4, 5, 3), "rgb") == (4, 5)
    assert cs._converted_shape((3, 4, 5), "zstack") == (4, 5)


def test_a_float_rgb_plane_keeps_its_dtype_and_mean():
    rgb = np.zeros((2, 2, 4), np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 3.0, 6.0, 9.0
    out = cs._convert_plane(rgb, "rgb")
    assert out.dtype == np.float32 and np.allclose(out, 6.0)
    stack = np.arange(8).reshape(2, 2, 2)
    assert np.array_equal(cs._convert_plane(stack, "zstack"), stack[1])


def test_unreadable_files_have_no_shape_or_thumbnail(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not an image")
    assert cs.image_shape(str(bad)) is None
    assert cs.thumbnail(str(bad)) is None
    assert cs.mask_thumbnail(str(bad)) is None


def test_image_shape_reads_png_bands_and_drops_singletons(tmp_path):
    grey = tmp_path / "g.png"
    Image.fromarray(np.zeros((5, 7), np.uint8)).save(grey)
    rgb = tmp_path / "c.png"
    Image.fromarray(np.zeros((5, 7, 3), np.uint8)).save(rgb)
    assert cs.image_shape(str(grey)) == (5, 7)
    assert cs.image_shape(str(rgb)) == (5, 7, 3)
    one = _tif(tmp_path / "one.tif", np.zeros((1, 1), np.uint8))
    assert cs.image_shape(str(one)) == (1,)


def test_thumbnail_falls_back_to_tifffile_and_projects_stacks(tmp_path,
                                                              monkeypatch):
    stack = np.zeros((6, 10, 12), np.uint16)
    stack[3, 2:4, 2:4] = 500
    path = _tif(tmp_path / "z.tif", stack)
    real_open = Image.open

    def refuse(*_a, **_k):
        raise OSError("PIL cannot")

    monkeypatch.setattr(Image, "open", refuse)
    thumb = cs.thumbnail(str(path), size=6)
    monkeypatch.setattr(Image, "open", real_open)
    assert thumb.shape == (5, 6) and thumb.dtype == np.uint8
    assert thumb.max() == 255


def test_a_flat_image_thumbnail_is_dark_and_an_empty_one_is_refused(
        tmp_path, monkeypatch):
    flat = _tif(tmp_path / "flat.tif", np.full((8, 8), 7, np.uint16))
    assert cs.thumbnail(str(flat)).max() == 0
    monkeypatch.setattr(cs.np, "asarray",
                        lambda *_a, **_k: np.zeros((0, 4), np.uint8))
    assert cs.thumbnail(str(flat)) is None


def test_a_3d_mask_has_no_thumbnail(tmp_path):
    mask = _tif(tmp_path / "m.tif", np.ones((3, 4, 5), np.uint16))
    assert cs.mask_thumbnail(str(mask)) is None
    flat = _tif(tmp_path / "f.tif", np.eye(4, dtype=np.uint16))
    assert cs.mask_thumbnail(str(flat)).tolist()[0] == [255, 0, 0, 0]


# ----------------------------------------------------------- regex and sets

def test_an_invalid_regex_says_why():
    compiled, error = cs.compile_regex("(")
    assert compiled is None and error


def test_a_set_report_names_every_kind_of_problem():
    parsed = cs.parse_names(
        ["p1_c1.tif", "p1_c2.tif", "p2_c1.tif", "p3_c1.tif", "p3_c1.png",
         "odd.tif", "p4.tif"],
        r"(?P<plateID>p\d)(?:_c(?P<chanID>\d))?\.",
        channels=None)
    for item in parsed:
        if item.name == "p4.tif":
            item.channel = None
    report = cs.check_sets(parsed)
    assert not report.ok
    text = report.summary()
    assert "not matched: odd.tif" in text
    assert "have no channel: p4.tif" in text
    assert "set(s) incomplete: plateID=p2 lacks channel(s)" in text
    assert "duplicate(s): plateID=p3 channel" in text
    assert "p3_c1.tif / p3_c1.png" in text


def test_set_labels():
    assert cs.set_label(None) == ""
    assert cs.set_label(()) == "(one set)"
    assert cs.set_label((("wellID", "A01"), ("fieldID", "2"))) == \
        "wellID=A01 fieldID=2"
    assert cs._few(["a", "b", "c", "d", "e"]) == "a, b, c and 2 more"


def test_token_uniformity_and_character_classes():
    tok = cs._separator_tokens
    assert cs._uniform([tok("a_1"), tok("a_2")])
    assert not cs._uniform([tok("a_1"), tok("a_1_2")])
    assert not cs._uniform([tok("a_1"), tok("b-1")])
    assert not cs._uniform([tok("a_1"), tok("a_x")])
    assert cs._class_for(["12", "3"]) == r"\d+"
    cls = cs._class_for(["a-1", "b+2"])
    assert cls.startswith("[A-Za-z0-9") and "+" in cls


def test_infer_regex_of_nothing_is_none_and_proposer_failure_is_survived(
        monkeypatch):
    assert cs.infer_regex([]) is None
    assert cs.infer_regex(["", ""]) is None
    import spacr.regex_infer as regex_infer

    def broken(_names):
        raise RuntimeError("proposer failed")

    monkeypatch.setattr(regex_infer, "propose", broken)
    names = [f"plate1_well{w}_ch{c}.tif" for w in (1, 2) for c in (1, 2)]
    pattern = cs.infer_regex(names)
    assert pattern is not None
    report = cs.check_sets(cs.parse_names(names, pattern))
    assert report.ok and report.channels == [1, 2]


def test_infer_regex_with_leading_and_trailing_digits_in_the_varying_token():
    names = [f"img{w}x9_{c}.tif" for w in ("1a", "2a") for c in ("dapi", "gfp")]
    pattern = cs.infer_regex(names)
    assert pattern is not None
    assert cs.check_sets(cs.parse_names(names, pattern)).ok


def test_name_distance_of_two_blank_names_is_zero():
    assert cs.name_distance(".tif", "..tif") == 0.0
    assert cs.name_distance("a1.tif", "b2.tif") == 1.0


# ------------------------------------------------------------ detect_sets

def test_detect_sets_with_no_channels_says_so(tmp_path):
    found = cs.detect_sets(str(tmp_path), {})
    assert found.problems == ["No image has a channel yet."]


def test_detect_sets_uses_given_shapes_and_leaves_an_extra_unpaired(tmp_path):
    folder, _sets = _pair_folder(tmp_path, n=2, masks=False)
    channels = {"s1_nuc.tif": 1, "s2_nuc.tif": 1, "s1_cyto.tif": 2,
                "s2_cyto.tif": 2, "s3_nuc.tif": 1}
    shapes = {name: (6, 8) for name in channels}
    found = cs.detect_sets(str(folder), channels, shapes=shapes)
    assert len(found.sets) == 2
    assert found.unpaired == ["s3_nuc.tif"]
    assert "have no partner" in found.problems[0]


def test_assignment_without_scipy_is_greedy(monkeypatch):
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("no scipy")
        return real_import(name, *args, **kwargs)

    assert [len(a) for a in cs._assign(np.zeros((0, 3)))] == [0, 0]
    monkeypatch.setattr(builtins, "__import__", no_scipy)
    rows, cols = cs._assign(np.array([[5.0, 1.0], [0.5, 9.0]]))
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0)]


# -------------------------------------------------------------- naming

def test_wells_read_in_either_spelling():
    assert cs.canonical_well("a1") == "A01"
    assert cs.canonical_well("r02c03") == "B03"
    assert cs.canonical_well("r17c01") is None
    assert cs.canonical_well("Z9") is None
    assert cs.well_name(25) == "B02"


def test_plate_tokens_are_made_safe_and_unique():
    keys = [(("plateID", "p_1"), ("wellID", "A1")),
            (("plateID", "p 1"), ("wellID", "A1")),
            (("plateID", "__"), ("wellID", "A1"))]
    names = cs.name_sets(keys)
    plates = sorted(name.plate for name in names.values())
    assert len(set(plates)) == 3
    assert "plate1" in plates and "p-1" in plates


def test_non_numeric_times_are_numbered_and_numeric_fields_kept():
    keys = [(("wellID", "A1"), ("fieldID", f), ("timeID", t))
            for f, t in (("7", "late"), ("9", "early"))]
    names = cs.name_sets(keys)
    assert {k[2][1]: n.time for k, n in names.items()} == {"early": 1,
                                                          "late": 2}
    assert sorted(n.field for n in names.values()) == [7, 9]


def test_too_many_wells_or_fields_is_refused(monkeypatch):
    monkeypatch.setattr(cs, "WELLS_PER_PLATE", 2)
    with pytest.raises(ValueError, match="384 wells"):
        cs.name_sets([(("wellID", f"w{i}"),) for i in range(3)])
    with pytest.raises(ValueError, match="384 x 999"):
        cs.name_sets([(("id1", str(i)),) for i in range(2 * 999 + 1)])


def test_more_than_999_fields_in_a_well_is_refused():
    with pytest.raises(ValueError, match="999 fields"):
        cs.name_sets([(("wellID", "A1"), ("id1", f"x{i}"))
                      for i in range(1000)])


def test_two_sets_with_one_name_are_refused(monkeypatch):
    keys = [(("wellID", "A1"), ("timeID", "1")),
            (("wellID", "A1"), ("timeID", "01"))]
    with pytest.raises(ValueError, match="would both be named"):
        cs.name_sets(keys)


# -------------------------------------------------------------- planning

def test_default_mask_roles_fill_in_and_run_out():
    roles = cs.default_mask_roles(
        {1: ["x_dapi.tif"], 2: ["y.tif"], 3: ["z_dapi.tif"], 4: ["w.tif"]},
        [1, 2, 3, 4])
    assert roles == {1: "nucleus", 2: "cell", 3: "pathogen"}
    assert cs._word_role(["virus.tif"]) == "pathogen"


def test_unused_folder_is_numbered(tmp_path):
    (tmp_path / "sorted").mkdir()
    (tmp_path / "sorted_2").mkdir()
    assert cs.unused_folder(str(tmp_path), "sorted").endswith("sorted_3")


def test_a_plan_without_sets_or_with_a_taken_dest_is_refused(tmp_path):
    empty = cs.build_plan(str(tmp_path), {})
    assert not empty.ok and "no complete sets" in empty.problems[0]
    folder, sets = _pair_folder(tmp_path)
    taken = tmp_path / "taken"
    taken.mkdir()
    plan = cs.build_plan(str(folder), sets, dest=str(taken))
    assert any("already exists" in p for p in plan.problems)


def test_a_plan_whose_sets_clash_by_name_is_refused(tmp_path):
    keys = {(("wellID", "A1"), ("timeID", "1")): {1: "a.tif"},
            (("wellID", "A1"), ("timeID", "01")): {1: "b.tif"}}
    plan = cs.build_plan(str(tmp_path), keys, check_shapes=False)
    assert any("would both be named" in p for p in plan.problems)
    assert not plan.rows


def test_given_roles_are_filtered_and_doubles_refused(tmp_path):
    folder, sets = _pair_folder(tmp_path)
    plan = cs.build_plan(str(folder), sets,
                         mask_roles={1: "cell", "2": "cell", 9: "nucleus",
                                     3: "none"})
    assert plan.mask_roles == {1: "cell", 2: "cell"}
    assert any("cannot both give the cell mask" in p for p in plan.problems)
    text = plan.summary()
    assert "Mask planes: channel 1 -> cell, channel 2 -> cell" in text
    assert "no mask in channel 2" in text


def test_a_plan_with_no_masks_warns_that_merged_is_skipped(tmp_path):
    folder, sets = _pair_folder(tmp_path, masks=False)
    plan = cs.build_plan(str(folder), sets)
    assert plan.ok and plan.n_sets == 2
    assert any("No channel has masks" in w for w in plan.warnings)


def test_unreadable_3d_and_mismatched_mask_images_are_problems(tmp_path):
    folder, sets = _pair_folder(tmp_path, n=1)
    (folder / "s1_cyto.tif").write_bytes(b"junk")
    _tif(folder / "masks" / "s1_nuc.tif", np.zeros((3, 3), np.uint16))
    plan = cs.build_plan(str(folder), sets)
    text = "\n".join(plan.problems)
    assert "s1_cyto.tif cannot be read." in text
    assert "The mask of s1_nuc.tif is (3, 3)" in text
    _tif(folder / "s1_cyto.tif", np.zeros((2, 3, 5, 5), np.uint16))
    plan = cs.build_plan(str(folder), sets)
    assert any("not a single 2-D plane" in p for p in plan.problems)


def test_a_curation_ledger_travels_with_its_mask(tmp_path):
    folder, sets = _pair_folder(tmp_path, n=1)
    (folder / "masks" / "s1_nuc.tif.curation.json").write_text("{}")
    plan = cs.build_plan(str(folder), sets)
    row = next(r for r in plan.rows if r.channel == 1)
    assert row.source_ledger.endswith("s1_nuc.tif.curation.json")
    assert row.target_ledger == row.target_mask + ".curation.json"


# -------------------------------------------------------------- applying

def test_apply_refuses_a_bad_plan_missing_files_a_taken_dest_and_clashes(
        tmp_path):
    with pytest.raises(ValueError, match="it is empty"):
        cs.apply_plan(cs.SortPlan(folder=str(tmp_path), dest=str(tmp_path)))
    folder, sets = _pair_folder(tmp_path, masks=False)
    plan = cs.build_plan(str(folder), sets)
    (folder / "s2_cyto.tif").unlink()
    with pytest.raises(ValueError, match="Missing before the move"):
        cs.apply_plan(plan, log=lambda _t: None)
    folder, sets = _pair_folder(tmp_path / "b", masks=False)
    plan = cs.build_plan(str(folder), sets)
    Path(plan.dest).mkdir()
    with pytest.raises(ValueError, match="already exists"):
        cs.apply_plan(plan, log=lambda _t: None)
    Path(plan.dest).rmdir()
    plan.rows[1].target_image = plan.rows[0].target_image
    with pytest.raises(ValueError, match="one name"):
        cs.apply_plan(plan, log=lambda _t: None)


def test_a_failed_move_is_recorded_before_it_is_raised(tmp_path, monkeypatch):
    folder, sets = _pair_folder(tmp_path, masks=False)
    plan = cs.build_plan(str(folder), sets)
    calls = []

    def move(source, target):
        calls.append(source)
        if len(calls) == 2:
            raise OSError("device busy")
        Path(target).write_bytes(Path(source).read_bytes())

    monkeypatch.setattr(cs.shutil, "move", move)
    with pytest.raises(OSError, match="device busy"):
        cs.apply_plan(plan, log=lambda _t: None)
    with open(Path(plan.dest) / cs.MANIFEST_NAME, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["status"] for r in rows] == ["moved", "error"]
    assert rows[1]["error"] == "device busy"


def test_progress_every_fifty_moves_and_no_merge_when_asked(tmp_path):
    folder, sets = _pair_folder(tmp_path, n=25, masks=False)
    plan = cs.build_plan(str(folder), sets)
    lines = []
    result = cs.apply_plan(plan, merge=False, log=lines.append)
    assert result.moved == 50 and result.stacks == [] and result.merged == []
    assert "Moved 50 of 50 files…" in lines


def test_a_conversion_of_an_unreadable_image_is_an_oserror(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")
    with pytest.raises(OSError, match="cannot read"):
        cs._write_converted(str(bad), str(tmp_path / "o.tif"), "rgb",
                            str(tmp_path))


def test_a_png_is_converted_and_its_original_kept(tmp_path):
    source = tmp_path / "c.png"
    Image.fromarray(np.full((4, 5, 3), 30, np.uint8)).save(source)
    target = tmp_path / "out" / "c.tif"
    target.parent.mkdir()
    cs._write_converted(str(source), str(target), "rgb", str(tmp_path / "d"))
    assert tifffile.imread(str(target)).shape == (4, 5)
    assert (tmp_path / "d" / "originals" / "c.png").is_file()


def test_merging_without_masks_writes_stacks_only(tmp_path):
    folder, sets = _pair_folder(tmp_path, masks=False)
    plan = cs.build_plan(str(folder), sets)
    lines = []
    result = cs.apply_plan(plan, log=lines.append)
    assert len(result.stacks) == 2 and result.merged == []
    stack = np.load(result.stacks[0])
    assert stack.shape == (6, 8, 2)


def test_a_field_with_an_unreadable_channel_gets_no_stack(tmp_path,
                                                         monkeypatch):
    folder, sets = _pair_folder(tmp_path, n=1, masks=False)
    plan = cs.build_plan(str(folder), sets)
    cs.apply_plan(plan, merge=False, log=lambda _t: None)
    import spacr.io as sio

    monkeypatch.setattr(sio, "load_images_from_paths",
                        lambda paths: {c: [] for c in paths})
    lines = []
    stacks, merged = cs.merge_sorted(plan.dest, {}, log=lines.append)
    assert stacks == [] and merged == []
    assert any("could not be read" in line for line in lines)
