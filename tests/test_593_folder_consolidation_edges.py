"""Folder consolidation at its edges: names, links, failures and the CLI."""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import pytest

from spacr import folder_consolidation as fc


def _tree(root: Path, files) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_compound_extensions_stay_whole_and_plain_ones_use_the_suffix():
    assert fc.file_extension(Path("a.OME.TIF")) == ".OME.TIF"
    assert fc.file_extension(Path("a.tif")) == ".tif"
    assert fc.file_extension(Path("README")) == ""


def test_a_reserved_windows_name_is_prefixed(tmp_path):
    used, counters = set(), {}
    path = fc.available_filename(tmp_path, "CON", ".tif", used, counters)
    assert path.name == "_CON.tif"
    path = fc.available_filename(tmp_path, "lpt1.x", ".tif", used, counters)
    assert path.name == "_lpt1.x.tif"


def test_collisions_are_numbered_case_insensitively(tmp_path):
    used, counters = set(), {}
    first = fc.available_filename(tmp_path, "exp", ".tif", used, counters)
    second = fc.available_filename(tmp_path, "EXP", ".TIF", used, counters)
    third = fc.available_filename(tmp_path, "exp", ".tif", used, counters)
    assert [first.name, second.name, third.name] == [
        "exp.tif", "EXP_2.TIF", "exp_3.tif"]


def test_an_existing_file_is_skipped_over(tmp_path):
    (tmp_path / "exp.tif").write_text("x")
    path = fc.available_filename(tmp_path, "exp", ".tif", set(), {})
    assert path.name == "exp_2.tif"


def test_a_long_stem_is_cut_and_ends_in_its_hash(tmp_path):
    stem = "é" * 300
    path = fc.available_filename(tmp_path, stem, ".tif", set(), {})
    assert len(path.name.encode("utf-8")) <= 240
    assert path.stem.rsplit("_", 1)[1] and len(path.stem.rsplit("_", 1)[1]) == 12
    other = fc.available_filename(tmp_path, stem + "x", ".tif", set(), {})
    assert other.name != path.name


def test_an_extension_that_leaves_no_room_raises(tmp_path):
    with pytest.raises(ValueError, match="too long"):
        fc.available_filename(tmp_path, "a" * 300, "." + "x" * 230, set(), {})


def test_the_default_output_folder_is_numbered_when_taken(tmp_path):
    source = tmp_path / "exp"
    source.mkdir()
    assert fc.default_output_folder(source) == tmp_path / "exp_renamed"
    (tmp_path / "exp_renamed").mkdir()
    (tmp_path / "exp_renamed_2").mkdir()
    assert fc.default_output_folder(source) == tmp_path / "exp_renamed_3"


def test_nested_file_count_of_a_missing_folder_is_zero(tmp_path):
    assert fc.nested_file_count(tmp_path / "nope") == (0, 0)


def test_nested_file_count_skips_hidden_skipped_and_top_level(tmp_path):
    _tree(tmp_path, {"top.tif": "", "a/1.tif": "", "a/2.png": "",
                     "b/3.TIF": "", ".hidden/4.tif": "",
                     "sorted_channels_2/5.tif": "", "c/notes.txt": ""})
    assert fc.nested_file_count(tmp_path, [".tif"],
                                skip_dirs=["sorted_channels"]) == (2, 2)
    assert fc.nested_file_count(tmp_path) == (5, 4)


def test_consolidate_refuses_a_missing_source_and_an_existing_output(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        fc.consolidate_folder(tmp_path / "nope", tmp_path / "out")
    source = _tree(tmp_path / "src", {"a/1.tif": "x"})
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        fc.consolidate_folder(source, tmp_path / "out")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges")
def test_links_are_skipped_and_listed_and_skip_dirs_are_left_out(tmp_path):
    source = _tree(tmp_path / "src", {"a/1.tif": "1", "skipme/2.tif": "2"})
    outside = _tree(tmp_path / "elsewhere", {"3.tif": "3"})
    os.symlink(outside, source / "linked_dir")
    os.symlink(outside / "3.tif", source / "a" / "link.tif")
    lines = []
    result = fc.consolidate_folder(source, tmp_path / "out",
                                   skip_dirs=["SKIPME"], log=lines.append)
    assert result.copied == 1 and result.skipped_links == 2
    statuses = sorted(row[2] for row in result.rows)
    assert statuses == ["copied", "skipped_symlink", "skipped_symlink"]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "rename_manifest.csv", "src_a.tif"]
    assert any("Symlinks skipped: 2" in line for line in lines)


def test_an_output_inside_the_source_is_not_copied_into_itself(tmp_path):
    source = _tree(tmp_path / "src", {"a/1.tif": "1"})
    result = fc.consolidate_folder(source, source / "out", log=lambda _t: None)
    assert result.copied == 1
    assert [row[1] for row in result.rows] == ["src_a.tif"]


def test_a_failed_copy_is_recorded_and_its_partial_file_removed(
        tmp_path, monkeypatch):
    source = _tree(tmp_path / "src", {"a/1.tif": "1", "a/2.tif": "2"})
    real_copy = shutil.copy2

    def half_copy(src, dst):
        if str(src).endswith("2.tif"):
            Path(dst).write_text("partial")
            raise OSError("disk full")
        return real_copy(src, dst)

    monkeypatch.setattr(fc.shutil, "copy2", half_copy)
    lines = []
    result = fc.consolidate_folder(source, tmp_path / "out", log=lines.append)
    assert (result.copied, result.failed) == (1, 1)
    assert not (tmp_path / "out" / "src_a_2.tif").exists()
    with open(result.manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["original_path", "new_filename", "status", "error"]
    assert [r[2] for r in rows[1:]] == ["copied", "error"]
    assert rows[2][3] == "disk full"
    assert any("Could not copy" in line for line in lines)


def test_an_unreadable_folder_is_a_directory_error(tmp_path, monkeypatch):
    source = _tree(tmp_path / "src", {"a/1.tif": "1"})
    real_walk = os.walk

    def walk(top, **kwargs):
        kwargs["onerror"](PermissionError(13, "denied", str(top / "locked")))
        yield from real_walk(top, **kwargs)

    monkeypatch.setattr(fc.os, "walk", walk)
    result = fc.consolidate_folder(source, tmp_path / "out", log=lambda _t: None)
    assert result.failed == 1 and result.copied == 1
    assert ("directory_error" in [row[2] for row in result.rows])


def test_progress_is_reported_every_hundred_copies(tmp_path):
    source = _tree(tmp_path / "src",
                   {f"a/{i:03d}.tif": "" for i in range(100)})
    lines = []
    result = fc.consolidate_folder(source, tmp_path / "out", log=lines.append)
    assert result.copied == 100
    assert "Copied 100 files..." in lines


def test_the_command_line_copies_and_reports_its_status(tmp_path, capsys):
    source = _tree(tmp_path / "src", {"a/1.tif": "1"})
    assert fc.main([str(source)]) == 0
    assert (tmp_path / "src_renamed" / "src_a.tif").read_text() == "1"
    assert "Copied: 1" in capsys.readouterr().out
    assert fc.main([str(tmp_path / "missing"), str(tmp_path / "o")]) == 1
    assert "Error:" in capsys.readouterr().err


def test_the_command_line_says_what_an_interrupt_left(tmp_path, monkeypatch,
                                                      capsys):
    def interrupted(_source, _output):
        raise KeyboardInterrupt

    monkeypatch.setattr(fc, "copy_and_rename", interrupted)
    assert fc.main([str(tmp_path), str(tmp_path / "o")]) == 130
    assert "Original files were not changed" in capsys.readouterr().err


def test_copy_and_rename_fails_when_any_copy_failed(tmp_path, monkeypatch):
    source = _tree(tmp_path / "src", {"a/1.tif": "1"})

    def broken(_src, _dst):
        raise OSError("no")

    monkeypatch.setattr(fc.shutil, "copy2", broken)
    assert fc.copy_and_rename(source, tmp_path / "out") == 1
