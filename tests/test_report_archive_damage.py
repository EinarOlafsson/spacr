"""How the archive and Zenodo writers of ``spacr.report`` refuse and validate.

The IDR / BioStudies / MIHCSME package and the multi-screen study are read by
archive curators who cannot re-run spaCR, so every way a written package can
be damaged must be named by the validator, and every input the writers cannot
use must be refused with a message that says why.  Each test walks a case
table: one pristine package (or study) is written per test, every case damages
a fresh copy of it, and the validator's full problem list is compared exactly.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from spacr import report as rep
from tests.test_archive_package import _form, screen  # noqa: F401

SLUG = "drugscreen"


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _idr(pkg: Path, suffix: str) -> Path:
    """The single IDR file of ``pkg`` whose name ends with ``suffix``."""
    return next((pkg / "idr").glob(f"*{suffix}"))


def _rewrite_tsv(path: Path, change) -> None:
    """Read a TSV as strings, apply ``change`` to the frame, write it back."""
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    change(frame).to_csv(path, sep="\t", index=False)


def _fresh_copies(pristine: Path, root: Path):
    """Yield a new copy of ``pristine`` per call, under ``root``."""
    counter = iter(range(10_000))

    def copy() -> Path:
        target = root / f"case{next(counter)}" / pristine.name
        shutil.copytree(pristine, target)
        return target

    return copy


def _second_screen(screen: Path, tmp_path: Path) -> list:  # noqa: F811
    """Two run folders: the fixture screen and a copy of it."""
    second = tmp_path / "screen2"
    shutil.copytree(screen, second)
    return [screen, second]


# ---------------------------------------------------------------------------
# _validate_archive_package
# ---------------------------------------------------------------------------

def _no_study_and_plates(pkg):
    _idr(pkg, "-plates.txt").unlink()
    _idr(pkg, "-study.txt").unlink()
    return {}


def _bad_library_rows(pkg):
    def damage(frame):
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame.loc[1, "Well"] = "A1x"
        frame.loc[0, "Characteristics [Organism]"] = ""
        return frame.drop(columns=["Characteristics [Cell Line]"])

    _rewrite_tsv(_idr(pkg, "-library.txt"), damage)
    return {"verify_checksums": False}


def _empty_library(pkg):
    _rewrite_tsv(_idr(pkg, "-library.txt"), lambda f: f.iloc[0:0])
    return {"verify_checksums": False}


def _no_library_or_processed(pkg):
    _idr(pkg, "-library.txt").unlink()
    for path in (pkg / "idr").iterdir():
        if "processed" in path.name:
            path.unlink()
    return {"verify_checksums": False}


def _library_without_plate_and_bad_conditions(pkg):
    _rewrite_tsv(_idr(pkg, "-library.txt"), lambda f: f.drop(columns=["Plate"]))
    pd.DataFrame({"x": ["1"]}).to_csv(pkg / "mihcsme" / "AssayConditions.tsv",
                                      sep="\t", index=False)
    return {"verify_checksums": False}


def _pagetab_not_submission(pkg):
    pagetab = next((pkg / "biostudies").glob("*.pagetab.tsv"))
    text = pagetab.read_text(encoding="utf-8")
    pagetab.write_text(text.replace("Submission", "Notes", 1)
                       .replace("ReleaseDate\t", "ReleaseDate\tsoon ", 1),
                       encoding="utf-8")
    return {"verify_checksums": False}


def _pagetab_missing(pkg):
    next((pkg / "biostudies").glob("*.pagetab.tsv")).unlink()
    return {"verify_checksums": False}


def _file_list_renamed_doubled_and_dangling(pkg):
    path = pkg / "biostudies" / "file_list.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    doubled.loc[len(doubled)] = ["nowhere.tif"] + [""] * (doubled.shape[1] - 1)
    doubled.rename(columns={"Files": "Paths"}).to_csv(path, sep="\t", index=False)
    return {"verify_checksums": False}


def _file_list_empty(pkg):
    path = pkg / "biostudies" / "file_list.tsv"
    pd.read_csv(path, sep="\t", dtype=str).iloc[0:0].to_csv(path, sep="\t",
                                                            index=False)
    return {"verify_checksums": False}


def _file_list_missing(pkg):
    (pkg / "biostudies" / "file_list.tsv").unlink()
    return {"verify_checksums": False}


def _mihcsme_sheet_and_checksums(pkg):
    (pkg / "mihcsme" / "AssayConditions.tsv").unlink()
    sums = pkg / "checksums.md5"
    kept = [line for line in sums.read_text(encoding="utf-8").splitlines()
            if not line.endswith(".tif")]
    sums.write_text("\n".join(kept + ["no-separator-line"]) + "\n",
                    encoding="utf-8")
    return {}


def _checksums_missing(pkg):
    (pkg / "checksums.md5").unlink()
    return {}


def _manifest_missing(pkg):
    (pkg / "archive_manifest.json").unlink()
    return {}


_TIFS = [f"{plate}_{well}_T0001F001L01A01Z01C0{c}.tif"
         for plate, well in (("plate1", "A01"), ("plate1", "B12"), ("plate2", "A01"))
         for c in (1, 2)]

PACKAGE_DAMAGE = [
    ("study and plate list removed", _no_study_and_plates, [
        f"IDR: {SLUG}-study.txt is missing",
        f"IDR: {SLUG}-screenA-plates.txt is missing",
        f"checksums: idr/{SLUG}-screenA-plates.txt does not match",
        f"checksums: idr/{SLUG}-study.txt does not match"]),
    ("bad well, repeated well, no organism", _bad_library_rows, [
        "IDR library: column 'Characteristics [Cell Line]' is missing",
        "IDR library: 1 well(s) not spelled like A01, e.g. 'A1x'",
        "IDR library: a plate and well appear twice",
        "IDR library: a well has no organism"]),
    ("library with no rows", _empty_library, ["IDR library: no wells"]),
    ("library and processed file removed", _no_library_or_processed, [
        f"IDR: library file '{SLUG}-screenA-library.txt' is missing",
        f"IDR: processed file '{SLUG}-screenA-processed.txt' is missing"]),
    ("library without Plate, conditions without wells",
     _library_without_plate_and_bad_conditions, [
         "IDR library: column 'Plate' is missing",
         "MIHCSME AssayConditions: no Plate and Well"]),
    ("PageTab opens with another block", _pagetab_not_submission, [
        "BioStudies: the first block is not Submission",
        "BioStudies: no 'Submission' section",
        "BioStudies: ReleaseDate is not YYYY-MM-DD"]),
    ("PageTab removed", _pagetab_missing,
     [f"BioStudies: {SLUG}.pagetab.tsv is missing"]),
    ("file list renamed, doubled and dangling",
     _file_list_renamed_doubled_and_dangling, [
         "BioStudies file list: first column is not Files",
         "BioStudies file list: a file is listed twice",
         "BioStudies file list: 1 file(s) not found, e.g. nowhere.tif",
         "checksums: no checksum for nowhere.tif"]),
    ("file list with no rows", _file_list_empty,
     ["BioStudies file list: no files"]),
    ("file list removed", _file_list_missing,
     ["BioStudies: file_list.tsv is missing"]),
    ("MIHCSME sheet removed, checksums thinned", _mihcsme_sheet_and_checksums,
     ["MIHCSME: sheet AssayConditions is missing"]
     + [f"checksums: no checksum for {name}" for name in _TIFS]
     + ["checksums: mihcsme/AssayConditions.tsv does not match"]),
    ("checksums removed", _checksums_missing, ["checksums.md5 is missing"]),
]


def test_validate_archive_package_names_every_kind_of_damage(
        screen, tmp_path, subtests):  # noqa: F811
    """Each damaged copy of a valid package gets exactly its own problem lines."""
    pristine = rep._write_archive_package(screen, tmp_path / "out", _form(screen))
    assert rep._validate_archive_package(pristine) == []
    fresh = _fresh_copies(pristine, tmp_path / "copies")

    for case_id, damage, expected in PACKAGE_DAMAGE:
        with subtests.test(msg=case_id):
            pkg = fresh()
            kwargs = damage(pkg)
            assert rep._validate_archive_package(pkg, **kwargs) == expected

    with subtests.test(msg="no archive manifest at all"):
        pkg = fresh()
        _manifest_missing(pkg)
        assert rep._validate_archive_package(pkg) == [
            f"{pkg} has no readable archive_manifest.json"]


# ---------------------------------------------------------------------------
# _write_archive_package
# ---------------------------------------------------------------------------

def test_write_archive_package_refuses_or_degrades_on_odd_inputs(
        screen, tmp_path, monkeypatch, subtests):  # noqa: F811
    """A missing folder is refused; a well-keyed map and no openpyxl still write."""
    with subtests.test(msg="source that is not a folder"):
        with pytest.raises(ValueError, match="^Not a folder: .*none$"):
            rep._write_archive_package(tmp_path / "none", tmp_path / "out", {})

    with subtests.test(msg="plate map without plates, with controls, no openpyxl"):
        (screen / "notes_without_a_well.tif").write_bytes(b"x")
        plate_map = tmp_path / "wells.csv"
        pd.DataFrame({"well": ["A01", "B12"], "control": ["neg", ""],
                      "compound": ["DMSO", "drugX"]}).to_csv(plate_map,
                                                             index=False)
        import spacr.tabular as tabular

        def no_openpyxl(sheets, path):
            raise ImportError("openpyxl")

        monkeypatch.setattr(tabular, "_write_workbook", no_openpyxl)
        said = []
        pkg = rep._write_archive_package(
            screen, tmp_path / "out2", _form(screen, plate_map=str(plate_map)),
            progress=said.append)
        library = pd.read_csv(_idr(pkg, "-library.txt"), sep="\t", dtype=str,
                              keep_default_na=False)
        assert "Control Type" in library.columns
        assert "openpyxl is not installed; wrote the MIHCSME sheets as TSV only" \
            in said
        assert not (pkg / "mihcsme" / "mihcsme.xlsx").exists()


# ---------------------------------------------------------------------------
# _write_archive_study
# ---------------------------------------------------------------------------

def test_write_archive_study_refuses_unusable_source_lists(
        screen, tmp_path, subtests):  # noqa: F811
    """No sources, too many screens and an existing study are refused by name."""
    refusals = [
        ("no run folders", [], "^A study needs at least one run folder\\.$"),
        ("27 screens", [tmp_path / f"s{i}" for i in range(27)],
         "^A study holds at most 26 screens\\.$"),
        ("a run listed twice", [screen, screen],
         "^A run folder is listed twice\\.$"),
    ]
    for case_id, sources, pattern in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=pattern):
                rep._write_archive_study(sources, tmp_path / "s", {})

    sources = _second_screen(screen, tmp_path)
    with subtests.test(msg="study written over an existing one"):
        rep._write_archive_study(sources, tmp_path / "study", _form(screen))
        with pytest.raises(ValueError, match="^Archive package already exists: "):
            rep._write_archive_study(sources, tmp_path / "study", _form(screen))

    with subtests.test(msg="a screen without counts has no processed file"):
        shutil.rmtree(sources[1] / "measurements")
        pkg = rep._write_archive_study(sources, tmp_path / "study2",
                                       _form(screen))
        processed = sorted(p.name for p in (pkg / "idr").glob("*-processed.txt"))
        assert processed == [f"{SLUG}-screenA-processed.txt"]


# ---------------------------------------------------------------------------
# _validate_archive_study
# ---------------------------------------------------------------------------

def _study_manifest_broken(pkg):
    (pkg / "study_manifest.json").write_text("{broken", encoding="utf-8")
    return {}


def _study_file_missing(pkg):
    _idr(pkg, "-study.txt").unlink()
    return {}


def _study_screens_disagree(pkg):
    path = _idr(pkg, "-study.txt")
    text = path.read_text(encoding="utf-8")
    text = text.replace("Study Screens Number\t2", "Study Screens Number\t3")
    first = text.index("Comment[IDR Screen Name]")
    end = text.index("\n", first)
    text = text[:first] + "Comment[IDR Screen Name]\t" + text[end:]
    path.write_text(text, encoding="utf-8")
    return {"verify_checksums": False}


def _study_fields_emptied_and_block_dropped(pkg):
    path = _idr(pkg, "-study.txt")
    lines = path.read_text(encoding="utf-8").splitlines()
    key = rep._IDR_STUDY_REQUIRED[0]
    lines = [f"{key}\t" if line.split("\t", 1)[0] == key else line
             for line in lines]
    last = max(i for i, line in enumerate(lines) if line.startswith("# Screen"))
    tail = next(i for i, line in enumerate(lines)
                if line.startswith("# Ontologies"))
    path.write_text("\n".join(lines[:last] + lines[tail:]) + "\n",
                    encoding="utf-8")
    pagetab = next((pkg / "biostudies").glob("*.pagetab.tsv"))
    _section, keys = next(iter(rep._BIA_REQUIRED.items()))
    text = "\n".join(f"{keys[0]}\t" if line.split("\t", 1)[0] == keys[0]
                     else line
                     for line in pagetab.read_text(encoding="utf-8").splitlines())
    pagetab.write_text(text + "\n", encoding="utf-8")
    return {"verify_checksums": False}


def _study_file_list(change):
    def damage(pkg):
        path = pkg / "biostudies" / "file_list.tsv"
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        change(frame).to_csv(path, sep="\t", index=False)
        return {"verify_checksums": False}

    return damage


def _study_pagetab_notes(pkg):
    next((pkg / "biostudies").glob("*.pagetab.tsv")).write_text(
        "Notes\n", encoding="utf-8")
    return {"verify_checksums": False}


def _study_pagetab_and_checksums_missing(pkg):
    next((pkg / "biostudies").glob("*.pagetab.tsv")).unlink()
    (pkg / "checksums.md5").unlink()
    return {}


def test_validate_archive_study_names_every_kind_of_damage(
        screen, tmp_path, subtests):  # noqa: F811
    """Each damaged copy of a valid two-screen study gets exactly its problems."""
    pristine = rep._write_archive_study(_second_screen(screen, tmp_path),
                                        tmp_path / "study", _form(screen))
    assert rep._validate_archive_study(pristine) == []
    fresh = _fresh_copies(pristine, tmp_path / "copies")
    key = rep._IDR_STUDY_REQUIRED[0]
    section, keys = next(iter(rep._BIA_REQUIRED.items()))

    cases = [
        ("study file removed", _study_file_missing,
         [f"IDR: {SLUG}-study.txt is missing"]),
        ("screen count and names disagree", _study_screens_disagree, [
            "IDR study: 'Study Screens Number' is not 2",
            "IDR study: screen name '' is empty or repeated"]),
        ("required fields emptied, one screen block dropped",
         _study_fields_emptied_and_block_dropped, [
             f"IDR study: '{key}' has no value",
             "IDR study: 1 screen block(s) for 2 screen(s)",
             f"BioStudies {section}: '{keys[0]}' has no value",
             "BioStudies Study: 'Title' has no value"]),
        ("file list with a repeated row",
         _study_file_list(lambda f: pd.concat([f, f.iloc[[0]]])),
         ["BioStudies file list: a file is listed twice"]),
        ("file list of one screen only",
         _study_file_list(lambda f: f[f["Screen"] == f["Screen"].iloc[0]]),
         ["BioStudies file list: not every screen is listed"]),
        ("file list without a Screen column",
         _study_file_list(lambda f: f.rename(columns={"Screen": "Group"})),
         ["BioStudies file list: the first columns are not Files and Screen"]),
        ("PageTab that is only a Notes block", _study_pagetab_notes,
         ["BioStudies: the first block is not Submission"]
         + [f"BioStudies: no '{name}' section" for name in rep._BIA_REQUIRED]
         + ["BioStudies: Screen subsections [] are not "
            "['screenA', 'screenB']"]),
        ("PageTab and checksums removed", _study_pagetab_and_checksums_missing,
         [f"BioStudies: {SLUG}.pagetab.tsv is missing",
          "checksums.md5 is missing"]),
    ]
    for case_id, damage, expected in cases:
        with subtests.test(msg=case_id):
            pkg = fresh()
            problems = rep._validate_archive_study(pkg, **damage(pkg))
            assert problems == expected

    with subtests.test(msg="study manifest that is not JSON"):
        pkg = fresh()
        _study_manifest_broken(pkg)
        assert rep._validate_archive_study(pkg) == [
            f"{pkg / 'study_manifest.json'} is not readable JSON"]

    with subtests.test(msg="a single-screen package is checked as a package"):
        single = rep._write_archive_package(screen, tmp_path / "single",
                                            _form(screen))
        _checksums_missing(single)
        assert rep._validate_archive_study(single) == ["checksums.md5 is missing"]


# ---------------------------------------------------------------------------
# _zenodo_stage
# ---------------------------------------------------------------------------

def test_zenodo_stage_refuses_or_stages_a_bare_folder(tmp_path, monkeypatch,
                                                      subtests):
    """No folder and no free staging name are refused; a bare folder stages its report."""
    src = tmp_path / "run"
    src.mkdir()
    quiet = {"run_dirs": [], "search_journal": False}

    with subtests.test(msg="source that is not a folder"):
        with pytest.raises(ValueError, match="^Not a folder: .*none$"):
            rep._zenodo_stage(tmp_path / "none", tmp_path / "out", {})

    with subtests.test(msg="bare folder, first and retry staging"):
        first, files, _meta = rep._zenodo_stage(
            src, tmp_path / "out", {"title": "t"}, include_masks=True, **quiet)
        assert first == (tmp_path / "out" / "t-zenodo").resolve()
        assert [f.name for f in files] == ["report.html"]
        second, files2, _meta = rep._zenodo_stage(
            src, tmp_path / "out", {"title": "t"}, **quiet)
        assert second.name == "t-zenodo-2"
        assert [f.name for f in files2] == ["report.html"]

    with subtests.test(msg="every staging name already taken"):
        def taken(self, *a, **k):
            raise FileExistsError(str(self))

        monkeypatch.setattr(Path, "mkdir", taken)
        with pytest.raises(ValueError, match="^No unused Zenodo staging folder "
                                             "under .*out2\\.$"):
            rep._zenodo_stage(src, tmp_path / "out2", {"title": "t"}, **quiet)
