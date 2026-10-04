"""A study of several screens is written as one IDR study over every run.

Two copies of a small screen become screens A and B of one study: the study
file lists both, each screen keeps its own checked package, and a missing
screen file or a changed study file is reported.
"""
from __future__ import annotations

import shutil

import pytest

from spacr import report as rep
from tests.test_archive_package import _form, screen  # noqa: F401


@pytest.fixture
def two_screens(screen, tmp_path):  # noqa: F811
    second = tmp_path / "screen2"
    shutil.copytree(screen, second)
    return screen, second


def test_two_runs_become_one_valid_study_with_two_screens(two_screens, tmp_path):
    first, second = two_screens
    pkg = rep._write_archive_study([first, second], tmp_path / "out",
                                   _form(first))
    assert rep._validate_archive_study(pkg) == []
    study = next((pkg / "idr").glob("*-study.txt")).read_text()
    assert "Study Screens Number\t2" in study
    assert study.count("Comment[IDR Screen Name]") == 2
    assert "/screenB" in study
    assert sorted(p.name.split("-")[-2] for p in (pkg / "idr").glob(
        "*-library.txt")) == ["screenA", "screenB"]


def test_one_run_is_the_single_screen_package(screen, tmp_path):  # noqa: F811
    pkg = rep._write_archive_study([screen], tmp_path / "out", _form(screen))
    assert not (pkg / "study_manifest.json").exists()
    assert rep._validate_archive_study(pkg) == []


def test_a_missing_screen_library_or_changed_study_is_reported(two_screens,
                                                               tmp_path):
    first, second = two_screens
    pkg = rep._write_archive_study([first, second], tmp_path / "out",
                                   _form(first))
    next((pkg / "idr").glob("*-screenB-library.txt")).unlink()
    problems = rep._validate_archive_study(pkg)
    assert any("library file" in p for p in problems)
    assert any(p.startswith("checksums:") for p in problems)


def test_a_run_listed_twice_or_an_inside_destination_is_refused(screen,
                                                                tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="twice"):
        rep._write_archive_study([screen, screen], tmp_path / "out",
                                 _form(screen))
    other = tmp_path / "other"
    shutil.copytree(screen, other)
    with pytest.raises(ValueError, match="outside"):
        rep._write_archive_study([screen, other], screen, _form(screen))


def test_the_command_line_adds_screens_with_screen_src(two_screens, tmp_path,
                                                        capsys):
    import json

    from spacr import cli

    first, second = two_screens
    form = {k: v for k, v in _form(first).items() if isinstance(v, str)}
    code = cli._cmd_archive_package([
        "--src", str(first), "--screen-src", str(second),
        "--out", str(tmp_path / "out"), "--metadata-json", json.dumps(form)])
    assert code == cli.EXIT_OK, capsys.readouterr().err
    pkg = next((tmp_path / "out").iterdir())
    assert (pkg / "study_manifest.json").is_file()
    missing = tmp_path / "missing"
    assert cli._cmd_archive_package([
        "--src", str(first), "--screen-src", str(missing),
        "--out", str(tmp_path / "out2"),
        "--metadata-json", json.dumps(form)]) == cli.EXIT_USAGE
    assert not (tmp_path / "out2").exists()


def test_the_study_has_one_pagetab_over_every_screen(two_screens, tmp_path):
    from spacr.tabular import read_table

    first, second = two_screens
    pkg = rep._write_archive_study([first, second], tmp_path / "out",
                                   _form(first))
    pagetab = next((pkg / "biostudies").glob("*.pagetab.tsv"))
    text = pagetab.read_text()
    assert text.startswith("Submission")
    assert [b.split("\n")[0] for b in text.split("\n\n")].count("Study") == 1
    assert "Screen\tscreenA" in text and "Screen\tscreenB" in text
    files = read_table(pkg / "biostudies" / "file_list.tsv",
                       canonicalise=False, dtype=str)
    assert set(files["Screen"]) == {"screenA", "screenB"}
    assert all(f.startswith(s + "/") for f, s in zip(files["Files"],
                                                      files["Screen"]))
    assert "biostudies/file_list.tsv" in (pkg / "checksums.md5").read_text()

    pagetab.write_text(text.replace("Screen\tscreenB", "Screen\tscreenZ"))
    problems = rep._validate_archive_study(pkg)
    assert any("Screen subsections" in p for p in problems)
    (pkg / "biostudies" / "file_list.tsv").unlink()
    assert any("file_list.tsv is missing" in p
               for p in rep._validate_archive_study(pkg))
