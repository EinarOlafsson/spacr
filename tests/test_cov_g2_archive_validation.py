"""Each way an archive package can be damaged is named by the validator."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spacr import report as rep
from tests.test_archive_package import _form, screen  # noqa: F401


@pytest.fixture
def package(screen, tmp_path):  # noqa: F811
    pkg = rep._write_archive_package(screen, tmp_path / "out", _form(screen))
    assert rep._validate_archive_package(pkg) == []
    return pkg


def _idr(pkg, suffix):
    return next((pkg / "idr").glob(f"*{suffix}"))


def _rewrite_tsv(path, change):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    change(frame).to_csv(path, sep="\t", index=False)


def _problems(pkg, **kwargs):
    return "\n".join(rep._validate_archive_package(pkg, **kwargs))


def test_missing_idr_files_are_named(package):
    _idr(package, "-plates.txt").unlink()
    _idr(package, "-study.txt").unlink()
    text = _problems(package)
    assert "study.txt is missing" in text and "plates.txt is missing" in text


def test_a_library_with_bad_wells_repeats_and_no_organism(package):
    library = _idr(package, "-library.txt")

    def damage(frame):
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame.loc[1, "Well"] = "A1x"
        if "Characteristics [Organism]" in frame:
            frame.loc[0, "Characteristics [Organism]"] = ""
        return frame.drop(columns=[c for c in frame.columns
                                   if c == "Characteristics [Cell Line]"])

    _rewrite_tsv(library, damage)
    text = _problems(package, verify_checksums=False)
    assert "not spelled like A01" in text
    assert "appear twice" in text


def test_an_empty_library_has_no_wells(package):
    _rewrite_tsv(_idr(package, "-library.txt"), lambda f: f.iloc[0:0])
    assert "no wells" in _problems(package, verify_checksums=False)


def test_a_missing_library_and_processed_file_are_named(package):
    _idr(package, "-library.txt").unlink()
    processed = [p for p in (package / "idr").iterdir()
                 if "processed" in p.name]
    for path in processed:
        path.unlink()
    text = _problems(package, verify_checksums=False)
    assert "library file" in text


def test_biostudies_damage_is_named(package):
    pagetab = next((package / "biostudies").glob("*.pagetab.tsv"))
    text = pagetab.read_text(encoding="utf-8")
    pagetab.write_text(text.replace("Submission", "Notes", 1)
                       .replace("ReleaseDate\t", "ReleaseDate\tsoon ", 1),
                       encoding="utf-8")
    assert "first block is not Submission" in _problems(package,
                                                       verify_checksums=False)
    pagetab.unlink()
    assert "pagetab.tsv is missing" in _problems(package,
                                                 verify_checksums=False)


def test_the_file_list_damage_is_named(package):
    path = package / "biostudies" / "file_list.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    doubled.loc[len(doubled)] = ["nowhere.tif"] + [""] * (doubled.shape[1] - 1)
    doubled.rename(columns={"Files": "Paths"}).to_csv(path, sep="\t",
                                                      index=False)
    text = _problems(package, verify_checksums=False)
    assert "first column is not Files" in text
    assert "listed twice" in text and "not found" in text
    frame.iloc[0:0].to_csv(path, sep="\t", index=False)
    assert "no files" in _problems(package, verify_checksums=False)
    path.unlink()
    assert "file_list.tsv is missing" in _problems(package,
                                                   verify_checksums=False)


def test_mihcsme_and_checksum_damage_is_named(package):
    sheets = sorted((package / "mihcsme").glob("*.tsv"))
    sheets[0].unlink()
    conditions = package / "mihcsme" / "AssayConditions.tsv"
    if conditions.exists():
        pd.DataFrame({"x": [1]}).to_csv(conditions, sep="\t", index=False)
    sums = package / "checksums.md5"
    lines = sums.read_text().splitlines()
    kept = [line for line in lines if not line.endswith(".tif")]
    sums.write_text("\n".join(kept + ["no-separator-line"]) + "\n")
    text = _problems(package)
    assert "MIHCSME: sheet" in text
    assert "checksums: no checksum for" in text
    sums.unlink()
    assert "checksums.md5 is missing" in _problems(package)


def test_a_study_needs_distinct_sources(tmp_path, screen):  # noqa: F811
    with pytest.raises(ValueError, match="at least one run folder"):
        rep._write_archive_study([], tmp_path / "s", {})
    with pytest.raises(ValueError, match="at most 26"):
        rep._write_archive_study([tmp_path / f"s{i}" for i in range(27)],
                                 tmp_path / "s", {})


def test_a_package_needs_a_folder(tmp_path):
    with pytest.raises(ValueError, match="Not a folder"):
        rep._write_archive_package(tmp_path / "none", tmp_path / "out", {})


def test_unparseable_wells_and_settings_are_skipped(tmp_path):
    assert rep._archive_well("plate1_A00_T0001F001.tif") is None
    settings = tmp_path / "settings"
    settings.mkdir()
    (settings / "gen_mask_settings.csv").write_text("only\n1\n")
    (settings / "measure_crop_settings.csv").write_bytes(b"\xff\xfe\x00bad")
    assert rep._archive_settings(tmp_path) == {}


@pytest.fixture
def study(screen, tmp_path):  # noqa: F811
    import shutil

    second = tmp_path / "screen2"
    shutil.copytree(screen, second)
    pkg = rep._write_archive_study([screen, second], tmp_path / "study",
                                   _form(screen))
    assert rep._validate_archive_study(pkg) == []
    return pkg


def test_a_study_manifest_that_is_not_json_is_reported(study):
    (study / "study_manifest.json").write_text("{broken")
    assert "not readable JSON" in rep._validate_archive_study(study)[0]


def test_a_study_without_its_study_file_is_reported(study):
    next((study / "idr").glob("*-study.txt")).unlink()
    assert any("study.txt is missing" in p
               for p in rep._validate_archive_study(study))


def test_a_study_whose_screen_blocks_disagree_is_reported(study):
    path = next((study / "idr").glob("*-study.txt"))
    text = path.read_text(encoding="utf-8")
    text = text.replace("Study Screens Number\t2", "Study Screens Number\t3")
    first = text.index("Comment[IDR Screen Name]")
    end = text.index("\n", first)
    text = text[:first] + "Comment[IDR Screen Name]\t" + text[end:]
    path.write_text(text, encoding="utf-8")
    problems = "\n".join(rep._validate_archive_study(study,
                                                     verify_checksums=False))
    assert "Study Screens Number" in problems
    assert "empty or repeated" in problems


def test_study_biostudies_damage_is_reported(study):
    pagetab = next((study / "biostudies").glob("*.pagetab.tsv"))
    file_list = study / "biostudies" / "file_list.tsv"
    frame = pd.read_csv(file_list, sep="\t", dtype=str, keep_default_na=False)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(file_list, sep="\t", index=False)
    assert "listed twice" in "\n".join(rep._validate_archive_study(
        study, verify_checksums=False))
    frame[frame["Screen"] == frame["Screen"].iloc[0]].to_csv(
        file_list, sep="\t", index=False)
    assert "not every screen" in "\n".join(rep._validate_archive_study(
        study, verify_checksums=False))
    frame.rename(columns={"Screen": "Group"}).to_csv(file_list, sep="\t",
                                                     index=False)
    assert "first columns are not" in "\n".join(rep._validate_archive_study(
        study, verify_checksums=False))
    pagetab.write_text("Notes\n", encoding="utf-8")
    text = "\n".join(rep._validate_archive_study(study, verify_checksums=False))
    assert "first block is not Submission" in text and "no '" in text
    pagetab.unlink()
    (study / "checksums.md5").unlink()
    text = "\n".join(rep._validate_archive_study(study))
    assert "pagetab.tsv is missing" in text and "checksums.md5 is missing" in text


def test_zenodo_licences_and_empty_trees():
    assert rep._zenodo_license("") == "cc-by-4.0"
    assert rep._zenodo_license("CC0") == "cc0-1.0"
    assert rep._zenodo_zip(Path("/never.zip"), []) is None
    assert rep._zenodo_tree(Path("/no/such/folder"), "x") == []


def test_zenodo_mask_folders_fall_back_to_masks(tmp_path):
    (tmp_path / "masks").mkdir()
    assert rep._zenodo_mask_dirs(tmp_path) == [tmp_path / "masks"]
    assert rep._zenodo_mask_dirs(tmp_path / "none") == []


def test_zenodo_staging_needs_a_folder(tmp_path):
    with pytest.raises(ValueError, match="Not a folder"):
        rep._zenodo_stage(tmp_path / "none", tmp_path / "out", {})


def test_the_zenodo_token_moves_into_a_refusing_or_willing_keyring(monkeypatch):
    from spacr import run_journal as rj

    class _Ring:
        def __init__(self, refuse):
            self.refuse, self.values = refuse, {}

        def set_password(self, service, name, value):
            if self.refuse:
                raise RuntimeError("locked")
            self.values[name] = value

        def get_password(self, service, name):
            if self.refuse:
                raise RuntimeError("locked")
            return self.values.get(name)

        def delete_password(self, service, name):
            raise RuntimeError("absent")

    monkeypatch.setattr(rj, "_notify_keyring", lambda: _Ring(refuse=True))
    assert rep._store_zenodo_token("tok-1") == "file"
    assert rep._load_zenodo_token() == "tok-1"
    willing = _Ring(refuse=False)
    monkeypatch.setattr(rj, "_notify_keyring", lambda: willing)
    assert rep._store_zenodo_token("tok-2") == "keyring"
    assert rep._load_zenodo_token() == "tok-2"
    assert rep._store_zenodo_token("") == "forgotten"
