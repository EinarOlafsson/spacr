"""Item 593: consolidate a folder tree, sort images into channels, merge.

Everything here runs on small synthetic files in ``tmp_path``: the
consolidation copy (:mod:`spacr.folder_consolidation`), and the channel
sorting of :mod:`spacr.channel_sorting` -- regex parsing and inference, set
detection by pairing, the Yokogawa names read back through spaCR's OWN
``cellvoyager`` parser, and the ``merged/`` arrays the move-then-merge
writes.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import channel_sorting as cs
from spacr import folder_consolidation as fc


def _tif(path: Path, array) -> Path:
    """Write ``array`` as a TIFF, making the folder.

    :param path: where.
    :param array: the pixels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.asarray(array))
    return path


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def _tree(root: Path) -> Path:
    """A small nested tree: two subfolders, one nested deeper, one masks folder.

    :param root: the parent to build under.
    """
    src = root / "exp"
    for rel in ("nuc/a.tif", "nuc/b.tif", "cell/a.tif", "cell/deep/c.tif",
                "top.tif", "nuc/masks/a.tif", "notes.txt"):
        _tif(src / rel, np.zeros((4, 4), np.uint16)) if rel.endswith(".tif") \
            else (src / rel).write_text("x")
    return src


def test_consolidation_names_copies_after_their_folders_and_numbers_collisions(tmp_path):
    """Folder path joined by ``_``; a second file in a folder gets ``_2``."""
    src = _tree(tmp_path)
    result = fc.consolidate_folder(src, log=lambda _t: None)
    assert result.output == tmp_path / "exp_renamed"
    names = sorted(p.name for p in result.output.iterdir())
    assert names == sorted([
        "exp.tif", "exp.txt", "exp_cell.tif", "exp_cell_deep.tif",
        "exp_nuc.tif", "exp_nuc_2.tif", "exp_nuc_masks.tif",
        fc.MANIFEST_NAME])
    # The originals are untouched: copied, never moved.
    assert (src / "nuc" / "a.tif").is_file()
    with open(result.manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {r["new_filename"]: r["original_path"] for r in rows}
    assert by_name["exp_nuc.tif"].endswith(os.path.join("nuc", "a.tif"))
    assert by_name["exp_nuc_2.tif"].endswith(os.path.join("nuc", "b.tif"))
    assert all(r["status"] == "copied" for r in rows)


def test_consolidation_for_make_masks_keeps_images_and_skips_masks(tmp_path):
    """``extensions`` and ``skip_dirs`` leave out notes and masks folders."""
    src = _tree(tmp_path)
    assert fc.nested_file_count(src, (".tif",), ("masks",)) == (4, 3)
    result = fc.consolidate_folder(src, extensions=(".tif",),
                                   skip_dirs=("masks",), log=lambda _t: None)
    names = sorted(p.name for p in result.output.iterdir())
    assert "exp.txt" not in names and "exp_nuc_masks.tif" not in names
    assert result.copied == 5


def test_consolidation_output_is_a_new_unused_folder(tmp_path):
    """A second run goes to ``_renamed_2``; an existing output is refused."""
    src = _tree(tmp_path)
    first = fc.consolidate_folder(src, log=lambda _t: None)
    second = fc.consolidate_folder(src, log=lambda _t: None)
    assert second.output == tmp_path / "exp_renamed_2"
    with pytest.raises(ValueError):
        fc.consolidate_folder(src, first.output, log=lambda _t: None)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_consolidation_skips_symlinks_and_lists_them(tmp_path):
    """A linked file and a linked folder are not copied, but are recorded."""
    src = _tree(tmp_path)
    os.symlink(src / "nuc" / "a.tif", src / "link.tif")
    os.symlink(src / "cell", src / "linked_dir")
    result = fc.consolidate_folder(src, log=lambda _t: None)
    assert result.skipped_links == 2
    statuses = [row[2] for row in result.rows]
    assert statuses.count("skipped_symlink") == 2
    assert not any("linked_dir" in p.name for p in result.output.iterdir())


def test_long_and_reserved_names_are_made_safe(tmp_path):
    """Windows-forbidden characters, reserved names and length are handled."""
    assert fc.safe_part('a:b*c?') == "a_b_c_"
    used, counters = set(), {}
    reserved = fc.available_filename(tmp_path, "CON", ".tif", used, counters)
    assert reserved.name == "_CON.tif"
    long_name = fc.available_filename(tmp_path, "x" * 400, ".tif", used, counters)
    assert len(long_name.name.encode()) <= 244


def test_the_module_runs_as_a_program(tmp_path):
    """``python -m spacr.folder_consolidation SOURCE`` returns 0."""
    src = _tree(tmp_path)
    assert fc.main([str(src)]) == 0
    assert (tmp_path / "exp_renamed" / fc.MANIFEST_NAME).is_file()


# ---------------------------------------------------------------------------
# Regex parsing and inference
# ---------------------------------------------------------------------------


def test_parse_and_check_find_unmatched_incomplete_and_duplicated_sets():
    """The report names each way the names fail to form sets."""
    names = ["A01_1_w1.tif", "A01_1_w2.tif", "A01_2_w1.tif",
             "A01_3_w1.tif", "A01_3_w1b.tif", "junk.tif"]
    pattern = r"(?P<wellID>[A-Z]\d+)_(?P<fieldID>\d+)_w(?P<chanID>\d)"
    report = cs.check_sets(cs.parse_names(names, pattern))
    assert not report.ok
    assert report.unmatched == ["junk.tif"]
    assert (("wellID", "A01"), ("fieldID", "2")) in report.incomplete
    assert any(len(v) == 2 for v in report.duplicated.values())
    assert len(report.complete) == 1


def test_selection_channels_override_the_regex():
    """A name the user assigned keeps that channel."""
    parsed = cs.parse_names(["a_1.tif", "b_1.tif"], r"(?P<x>[ab])_(?P<fieldID>\d)",
                            {"a_1.tif": 2, "b_1.tif": 1})
    assert [p.channel for p in parsed] == [2, 1]


@pytest.mark.parametrize("names,channel_values", [
    (["nuc_img01.tif", "cell_img01.tif", "nuc_img02.tif", "cell_img02.tif",
      "nuc_img10.tif", "cell_img10.tif"], {"nuc", "cell"}),
    (["exp1_wt_fov01_C1.tif", "exp1_wt_fov01_C2.tif", "exp1_wt_fov02_C1.tif",
      "exp1_wt_fov02_C2.tif"], {"1", "2"}),
    (["img01C1.tif", "img01C2.tif", "img02C1.tif", "img02C2.tif"], {"1", "2"}),
    (["exp_nuc.tif", "exp_nuc_2.tif", "exp_nuc_3.tif", "exp_cell.tif",
      "exp_cell_2.tif", "exp_cell_3.tif"], {"nuc", "cell"}),
])
def test_auto_regex_maps_every_image_to_a_unique_complete_set(names, channel_values):
    """The inferred regex passes the set check, with the right channel token."""
    pattern = cs.infer_regex(names)
    assert pattern is not None
    parsed = cs.parse_names(names, pattern)
    assert cs.check_sets(parsed).ok
    assert {p.groups["chanID"] for p in parsed} == channel_values


def test_auto_regex_with_selected_channels_needs_no_channel_group():
    """When the user assigned channels, only the set identity is inferred."""
    channels = {"stainA_x1.tif": 1, "stainB_x1.tif": 2,
                "stainA_x2.tif": 1, "stainB_x2.tif": 2}
    pattern = cs.infer_regex(list(channels), channels)
    assert pattern is not None
    assert cs.check_sets(cs.parse_names(list(channels), pattern, channels)).ok


def test_auto_regex_prefers_spacrs_own_yokogawa_pattern():
    """Names already in the CellVoyager form keep spaCR's pattern."""
    names = [f"plate1_A01_T0001F00{f}L01A01Z01C0{c}.tif"
             for f in (1, 2) for c in (1, 2)]
    pattern = cs.infer_regex(names)
    assert "(?P<laserID>" in pattern
    names_by_key = cs.name_sets(cs.check_sets(cs.parse_names(names, pattern)).complete)
    assert sorted((n.well, n.field) for n in names_by_key.values()) == [
        ("A01", 1), ("A01", 2)]


# ---------------------------------------------------------------------------
# Yokogawa names round-trip through spaCR's own parser
# ---------------------------------------------------------------------------


def test_yokogawa_names_round_trip_through_the_cellvoyager_regex():
    """Every name built here parses with ``_get_regex('cellvoyager')``."""
    from spacr.utils import _get_regex

    keys = [(("wellID", w), ("fieldID", f)) for w in ("B2", "C10")
            for f in ("1", "7")]
    names = cs.name_sets(keys)
    regex = re.compile(_get_regex("cellvoyager", "tif"))
    for key, place in names.items():
        for channel in (1, 2):
            match = regex.match(cs.yokogawa_name(place, channel))
            assert match is not None
            assert match.group("plateID") == "plate1"
            assert match.group("wellID") == cs.canonical_well(dict(key)["wellID"])
            assert int(match.group("fieldID")) == int(dict(key)["fieldID"])
            assert int(match.group("chanID")) == channel


def test_names_are_enumerated_when_groups_are_not_wells_or_numbers():
    """Free-text wells become A01, A02...; free-text fields are numbered."""
    keys = [(("wellID", "ctrl"), ("fieldID", "left")),
            (("wellID", "ctrl"), ("fieldID", "right")),
            (("wellID", "drug"), ("fieldID", "left"))]
    names = cs.name_sets(keys)
    assert [(n.well, n.field) for n in names.values()] == [
        ("A01", 1), ("A01", 2), ("A02", 1)]


def test_detected_sets_get_a_well_each():
    """Pairing keys become plate1 A01, A02... field 1."""
    keys = [(("set", f"{i:06d}"),) for i in (1, 2, 25)]
    names = cs.name_sets(keys)
    many = cs.name_sets([(("set", f"{i:06d}"),) for i in range(1, 400)])
    assert many[(("set", "000025"),)].well == "B01"
    assert many[(("set", "000385"),)] == cs.SetName("plate2", "A01", 1, 1)
    assert [n.well for n in names.values()] == ["A01", "A02", "A03"]
    assert {n.plate for n in names.values()} == {"plate1"}


# ---------------------------------------------------------------------------
# Set detection, plan, apply and merge
# ---------------------------------------------------------------------------


def _masked_folder(root: Path, shapes=((20, 30), (20, 30), (24, 24))):
    """Nucleus and cell images of three fields, each with its mask.

    Names share nothing that a regex could use, except a scrambled token
    that pairs them by similarity. Field ``i``'s nucleus image holds
    ``10 + i``, its cell image ``100 + i``; masks hold ``i + 1``.

    :param root: where to make the folder.
    :param shapes: one image size per field.
    """
    folder = root / "drawn"
    channels = {}
    for i, shape in enumerate(shapes):
        nuc = f"DAPI stain sample{i + 1}.tif"
        cell = f"sample{i + 1} phalloidin.tif"
        _tif(folder / nuc, np.full(shape, 10 + i, np.uint16))
        _tif(folder / cell, np.full(shape, 100 + i, np.uint16))
        _tif(folder / "masks" / (nuc[:-4] + ".tif"), np.full(shape, i + 1, np.uint16))
        _tif(folder / "masks" / (cell[:-4] + ".tif"), np.full(shape, i + 1, np.uint16))
        (folder / "masks" / (cell[:-4] + ".tif.curation.json")).write_text("{}")
        channels[nuc], channels[cell] = 1, 2
    return folder, channels


def test_detect_sets_pairs_by_name_and_size(tmp_path):
    """Each nucleus image pairs with the cell image of the same field."""
    folder, channels = _masked_folder(tmp_path)
    found = cs.detect_sets(str(folder), channels)
    assert not found.problems
    pairs = sorted(tuple(sorted(m.values())) for m in found.sets.values())
    assert pairs == sorted(
        (f"DAPI stain sample{i}.tif", f"sample{i} phalloidin.tif")
        for i in (1, 2, 3))


def test_detect_sets_refuses_a_pair_of_different_sizes(tmp_path):
    """Size decides over name: a lone odd size stays unpaired."""
    folder, channels = _masked_folder(tmp_path)
    _tif(folder / "sample3 phalloidin.tif", np.zeros((8, 8), np.uint16))
    found = cs.detect_sets(str(folder), channels)
    assert "sample3 phalloidin.tif" in found.unpaired
    assert found.problems


def test_detect_sets_reports_a_mask_of_the_wrong_size(tmp_path):
    """A mask must be its image's size."""
    folder, channels = _masked_folder(tmp_path)
    _tif(folder / "masks" / "sample1 phalloidin.tif", np.zeros((5, 5), np.uint16))
    assert any("mask" in p for p in cs.detect_sets(str(folder), channels).problems)


def test_plan_apply_and_merge_write_the_pipelines_layout(tmp_path):
    """Moves are recorded; merged arrays hold both channels then both masks."""
    folder, channels = _masked_folder(tmp_path)
    found = cs.detect_sets(str(folder), channels)
    plan = cs.build_plan(str(folder), found.sets)
    assert plan.ok, plan.summary()
    assert plan.mask_roles == {1: "nucleus", 2: "cell"}
    # Nothing moves while planning.
    assert (folder / "DAPI stain sample1.tif").is_file()

    result = cs.apply_plan(plan, log=lambda _t: None)
    dest = Path(plan.dest)
    assert dest == folder / "sorted_channels"
    assert not list(folder.glob("*.tif"))
    c1 = sorted(p.name for p in (dest / "C01").glob("*.tif"))
    assert c1 == [f"plate1_A0{i}_T0001F001L01A01Z01C01.tif" for i in (1, 2, 3)]
    assert (dest / "C02" / "masks" /
            "plate1_A01_T0001F001L01A01Z01C02.tif.curation.json").is_file()

    with open(result.manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["kind"] for r in rows} == {"image", "mask", "curation"}
    assert all(r["status"] == "moved" for r in rows)
    assert len(rows) == 6 + 6 + 3

    assert len(result.merged) == 3
    from spacr.crops import MERGED_LAYOUT_SIDECAR
    layout = json.loads((dest / "merged" / MERGED_LAYOUT_SIDECAR).read_text())
    assert layout["mask_plane_order"] == ["cell", "nucleus"]
    for path in result.merged:
        merged = np.load(path)
        stem = os.path.basename(path)
        well = int(stem.split("_")[1][1:])
        i = well - 1
        assert merged.shape[-1] == 4
        assert merged[0, 0, 0] == 10 + i       # channel 1: nucleus stain
        assert merged[0, 0, 1] == 100 + i      # channel 2: cell stain
        assert merged[0, 0, 2] == i + 1        # cell mask
        assert merged[0, 0, 3] == i + 1        # nucleus mask
    shapes = sorted(np.load(p).shape for p in result.merged)
    assert shapes == [(20, 30, 4), (20, 30, 4), (24, 24, 4)]


def test_merged_field_stems_are_the_ones_the_pipeline_would_write(tmp_path):
    """``stack/`` stems come from spaCR's own parser and stem function."""
    from spacr.io import _escaped_field_stem

    folder, channels = _masked_folder(tmp_path, shapes=((16, 16),))
    plan = cs.build_plan(str(folder), cs.detect_sets(str(folder), channels).sets)
    result = cs.apply_plan(plan, log=lambda _t: None)
    assert [os.path.basename(p) for p in result.stacks] == [
        _escaped_field_stem("plate1", "A01", 1, 1) + ".npy"]


def test_a_plan_is_refused_when_images_differ_in_size(tmp_path):
    """Build-time checks stop the move before anything happens."""
    folder, _channels = _masked_folder(tmp_path)
    sets = {(("set", "000001"),): {1: "DAPI stain sample1.tif",
                                    2: "sample3 phalloidin.tif"}}
    plan = cs.build_plan(str(folder), sets)
    assert not plan.ok
    with pytest.raises(ValueError):
        cs.apply_plan(plan)


def test_regex_strategy_end_to_end(tmp_path):
    """Channel from the regex, sets from the regex, wells from the names."""
    folder = tmp_path / "plate"
    for well in ("B02", "B03"):
        for channel in (1, 2):
            _tif(folder / f"{well}_s1_w{channel}.tif",
                 np.full((8, 8), channel, np.uint16))
        _tif(folder / "masks" / f"{well}_s1_w1.tif", np.ones((8, 8), np.uint16))
    names = cs.list_folder_images(str(folder))
    pattern = cs.infer_regex(names)
    report = cs.check_sets(cs.parse_names(names, pattern))
    assert report.ok
    plan = cs.build_plan(str(folder), report.complete)
    assert plan.ok and plan.mask_roles == {1: "cell"}
    result = cs.apply_plan(plan, log=lambda _t: None)
    wells = sorted(os.path.basename(p).split("_")[1] for p in result.merged)
    assert wells == ["B02", "B03"]
    assert np.load(result.merged[0]).shape == (8, 8, 3)
