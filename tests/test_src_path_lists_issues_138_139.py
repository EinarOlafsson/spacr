"""A pasted list of plate folders is read as plates, not as one path.

GitHub issues #138 and #139: in Measure, ``src`` was pasted from a notes app
as four plate folders inside brackets, with curly quotes, commas and
newlines. spaCR read the whole text as one folder and stopped with "src does
not exist". These tests pin the reader that every ``src`` route shares
(``spacr.portable_paths._parse_path_list``), the routes themselves (GUI and
API through ``normalize_src_path``, preflight, batch, CLI ``--set``), a
two-plate Measure run, and the error a missing plate produces.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr.portable_paths import (
    _clean_path_item,
    _describe_missing_paths,
    _fold_path_quotes,
    _parse_path_list,
)

PLATES = [f"/Volumes/LouridoLab/jak/screen/plate{n}" for n in range(1, 5)]

ISSUE_138 = ("[\n\u201c" + "\u201d,\n\u201c".join(PLATES) + "\u201d,\n]")

ISSUE_139 = ("\n\u201c" + "\u201d,\n\u201c".join(PLATES) + "\u201d")


def _nothing_exists(_path):
    return False


def parse(value):
    return _parse_path_list(value, _nothing_exists)


@pytest.mark.parametrize("pasted", [ISSUE_138, ISSUE_139])
def test_the_pasted_strings_from_the_issues_are_four_plates(pasted):
    assert parse(pasted) == PLATES


def test_the_issue_strings_without_closing_quotes_still_split():
    pasted = "[\n" + "\n".join("\u201c" + p for p in PLATES) + "\n]"
    assert parse(pasted) == PLATES


@pytest.mark.parametrize("pasted, expected", [
    ("/data/plate1", ["/data/plate1"]),
    ("  /data/plate1  ", ["/data/plate1"]),
    ("\u201c/data/plate1\u201d", ["/data/plate1"]),
    ("\u2018/data/plate1\u2019", ["/data/plate1"]),
    ("[/data/plate1]", ["/data/plate1"]),
    ('["/a", "/b"]', ["/a", "/b"]),
    ("['/a', '/b',]", ["/a", "/b"]),
    ("[\u2018/a\u2019, \u2018/b\u2019]", ["/a", "/b"]),
    ('"/a", "/b"', ["/a", "/b"]),
    ('[\u201c/a\u201d\u201c/b\u201d]', ["/a", "/b"]),
    ("/a\n/b\n\n", ["/a", "/b"]),
    ("/a\r\n/b", ["/a", "/b"]),
    ("/a, /b", ["/a", "/b"]),
    ("/a,/b,/a", ["/a", "/b"]),
    ("[/a, /b]", ["/a", "/b"]),
    ("/data/plate 1, rep2", ["/data/plate 1, rep2"]),
    ("/data/plate (1)", ["/data/plate (1)"]),
    ("/data/[x]", ["/data/[x]"]),
    ("/data/o'brien/plate 1", ["/data/o'brien/plate 1"]),
    ('["/data/o\'brien/a", "/b"]', ["/data/o'brien/a", "/b"]),
    ("s3://bucket/a,s3://bucket/b", ["s3://bucket/a", "s3://bucket/b"]),
    ("./a, ../b", ["./a", "../b"]),
    ("/a\u00a0b\u200b", ["/a b"]),
])
def test_every_accepted_form(pasted, expected):
    assert parse(pasted) == expected


@pytest.mark.parametrize("pasted, expected", [
    (r"C:\data\plate1", [r"C:\data\plate1"]),
    (r"C:\data\plate1, D:\screen\plate2", [r"C:\data\plate1", r"D:\screen\plate2"]),
    ("C:\\data\\plate1\nC:\\data\\plate2", [r"C:\data\plate1", r"C:\data\plate2"]),
    (r'["C:\new\plate1", "C:\\data"]', [r"C:\new\plate1", r"C:\data"]),
    (r"['C:\\data\\p1', 'C:\\data\\p2']", [r"C:\data\p1", r"C:\data\p2"]),
    ('["C:\\\\data\\\\p1"]', [r"C:\data\p1"]),
    ("\\\\srv\\share\\p1\n\\\\srv\\share\\p2", [r"\\srv\share\p1", r"\\srv\share\p2"]),
    (r'"\\\\srv\\sh", "\\\\srv\\sh2"', [r"\\srv\sh", r"\\srv\sh2"]),
    ("\u201cC:\\Users\\me\\plate 1\u201d", [r"C:\Users\me\plate 1"]),
])
def test_windows_paths(pasted, expected):
    assert parse(pasted) == expected


def test_macos_volumes_paths():
    pasted = "/Volumes/Lab Share/plate1\n/Volumes/Lab Share/plate2"
    assert parse(pasted) == ["/Volumes/Lab Share/plate1",
                             "/Volumes/Lab Share/plate2"]


def test_home_is_expanded():
    home = os.path.expanduser("~")
    assert home != "~"
    assert parse("['~/a', '~/b']") == [os.path.expanduser("~/a"),
                                       os.path.expanduser("~/b")]
    assert parse("~/a") == [os.path.expanduser("~/a")]
    assert os.path.normpath(parse("~/a")[0]) == os.path.join(home, "a")


@pytest.mark.parametrize("value, expected", [
    (None, []), ("", []), ("   ", []), ('""', []), ("[]", []),
    ("[1, 2, 3]", ["[1, 2, 3]"]), ("[a, b]", ["[a, b]"]),
    (5, [5]),
])
def test_values_that_name_no_folders(value, expected):
    assert parse(value) == expected


def test_lists_keep_their_entries_and_non_strings(tmp_path):
    assert parse(["/a", "\u201c/b\u201d", 3, tmp_path]) == ["/a", "/b", 3, str(tmp_path)]
    assert parse(("/a", "/a")) == ["/a"]
    assert _parse_path_list(tmp_path) == [str(tmp_path)]


def test_existing_paths_with_commas_spaces_and_brackets_stay_whole(tmp_path):
    odd = tmp_path / "plate 1, rep [2], (final)"
    odd.mkdir()
    assert _parse_path_list(str(odd)) == [str(odd)]
    assert _parse_path_list(f"  {odd}  ") == [str(odd)]
    assert _parse_path_list(f"\u201c{odd}\u201d") == [str(odd)]
    other = tmp_path / "plate2"
    other.mkdir()
    assert _parse_path_list(f"{odd}\n{other}") == [str(odd), str(other)]
    assert _parse_path_list(f"[\u201c{odd}\u201d,\n\u201c{other}\u201d,\n]") == [
        str(odd), str(other)]


def test_a_comma_inside_an_existing_path_is_not_a_separator(tmp_path):
    comma = tmp_path / "a,b"
    comma.mkdir()
    second = tmp_path / "c"
    second.mkdir()
    assert _parse_path_list(f"{comma},{second}") == [str(comma), str(second)]


def test_relative_pieces_split_only_when_every_piece_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "p1").mkdir()
    (tmp_path / "p2").mkdir()
    assert _parse_path_list("p1, p2") == ["p1", "p2"]
    assert _parse_path_list("p1, nope") == ["p1, nope"]


def test_a_raw_path_that_exists_as_typed_wins(tmp_path):
    quoted = tmp_path / "\u201cquoted\u201d"
    quoted.mkdir()
    assert _parse_path_list(str(quoted)) == [str(quoted)]


def test_the_small_helpers():
    assert _fold_path_quotes("\u201ca\u201d \u2018b\u2019\ufeff") == "\"a\" 'b'"
    assert _clean_path_item(" [\u201c/a\u201d], ") == "/a"
    assert _clean_path_item("/data/[x]") == "/data/[x]"
    assert _clean_path_item(" , ") == ""
    assert _clean_path_item("[/a") == "/a"
    assert _clean_path_item("/a]") == "/a"
    assert _clean_path_item("('/a')") == "/a"


def test_normalize_src_path_uses_the_reader():
    from spacr.utils import normalize_src_path

    assert normalize_src_path(ISSUE_138) == PLATES
    assert normalize_src_path("\u201c/data/plate1\u201d") == "/data/plate1"
    assert normalize_src_path("") == ""
    assert normalize_src_path(["\u201c/a\u201d", "/b"]) == ["/a", "/b"]
    assert normalize_src_path("['/a']") == "/a"


def test_preflight_batch_and_cli_read_src_the_same_way():
    from spacr import batch, cli, cli_repro, validate

    settings = {"src": ISSUE_139}
    assert validate._src_values(settings, "measure") == PLATES
    assert validate._src_values({"src": ""}, "measure") == [""]
    assert batch._src_values(settings) == PLATES
    assert batch._src_values({"src": ["/a", 3]}) == ["/a"]
    assert batch._src_values({"src": None}) == []
    assert cli.coerce_value("src", ISSUE_138, "path", {"src": (str, list)}) == PLATES
    assert cli.coerce_value("src", "/data/p1", "path", {"src": (str, list)}) == "/data/p1"
    jobs = cli_repro._workflow_plates(settings)
    assert [job["src"] for job in jobs.values()] == PLATES


def test_missing_paths_say_whether_the_parent_exists(tmp_path):
    typo = str(tmp_path / "plate9")
    unmounted = "/Volumes/NotMounted/screen/plate1"
    lines = _describe_missing_paths([typo, unmounted, str(tmp_path), 7])
    assert len(lines) == 2
    assert typo in lines[0] and "exists" in lines[0] and "typo" in lines[0]
    assert unmounted in lines[1] and "mounted" in lines[1]


def test_measure_error_lists_each_missing_plate_on_its_own_line(tmp_path):
    from spacr.errors import ConfigurationError
    from spacr.measure import measure_crop

    good = tmp_path / "plate1"
    (good / "merged").mkdir(parents=True)
    np.save(good / "merged" / "plate1_A01_F001.npy", np.zeros((4, 4, 2), np.uint16))
    typo = tmp_path / "plate2_typo"
    unmounted = "/Volumes/NotMounted/screen/plate3"
    pasted = f"[\n\u201c{good}\u201d,\n\u201c{typo}\u201d,\n\u201c{unmounted}\u201d,\n]"
    with pytest.raises(ConfigurationError) as caught:
        measure_crop({"src": pasted, "timelapse": False})
    message = str(caught.value)
    lines = message.splitlines()
    assert lines[0] == "Measure cannot start: src does not exist:"
    assert any(str(typo) in line and "typo" in line for line in lines)
    assert any(unmounted in line and "mounted" in line for line in lines)
    assert not any(line.strip().startswith(str(good)) and "(" in line
                   for line in lines)
    assert "src was read as 3 folders:" in message
    assert "holding merged/" in message


def test_measure_merged_folder_names_the_missing_path(tmp_path):
    from spacr.errors import ConfigurationError
    from spacr.measure import _measure_merged_folder

    missing = str(tmp_path / "gone")
    with pytest.raises(ConfigurationError) as caught:
        _measure_merged_folder(missing)
    assert missing in str(caught.value)
    assert "src was read as" not in str(caught.value)


def test_measure_runs_each_pasted_plate_into_its_own_folder(
        tmp_path, synth_masks_multi, rng, capsys):
    from spacr.measure import measure_crop
    from tests.test_measure_crop_core_synth import (
        _build_merged_stack, _settings_for, _write_stack)

    plates = []
    for name in ("plate one", "plate2"):
        plate = tmp_path / name
        _write_stack(plate, _build_merged_stack(synth_masks_multi, rng),
                     name=f"{name.replace(' ', '')}_A01_F001.npy")
        plates.append(plate)
    pasted = "[\n" + ",\n".join(f"\u201c{p}\u201d" for p in plates) + ",\n]"
    settings = _settings_for(plates[0] / "merged", n_jobs=1, save_png=False)
    settings["src"] = pasted
    measure_crop(dict(settings))
    for plate in plates:
        assert (plate / "measurements" / "measurements.db").is_file()
    assert "src names 2 plate folders" in capsys.readouterr().out
