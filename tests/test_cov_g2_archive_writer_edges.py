"""Archive and Zenodo writers given thin, odd or partly broken run folders."""
from __future__ import annotations

import shutil
import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from spacr import report as rep
from tests.test_archive_package import _form, screen  # noqa: F401


def _settings(folder, rows, name="a.csv"):
    (folder / "settings").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Key": list(rows), "Value": list(rows.values())}).to_csv(
        folder / "settings" / name, index=False)


def test_settings_skip_one_column_and_unreadable_tables_and_keep_values(tmp_path):
    _settings(tmp_path, {"experiment": "exp1"}, "a.csv")
    _settings(tmp_path, {"experiment": ""}, "b.csv")
    (tmp_path / "settings" / "c.csv").write_text("only\n1\n")
    (tmp_path / "settings" / "d.csv").write_bytes(b"\x00\xff\xfe\x00garbage")
    assert rep._archive_settings(tmp_path)["experiment"] == "exp1"


@pytest.mark.parametrize("key, technology", [
    ("grna_csv", "CRISPR screen"), ("viability_plate_map", "compound screen")])
def test_the_technology_follows_the_saved_inputs(tmp_path, key, technology):
    _settings(tmp_path, {key: "/x.csv"})
    assert rep._archive_form_defaults(tmp_path)["technology"] == technology


def test_unreadable_or_unplaced_count_tables_give_no_counts(tmp_path):
    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements" / "a.db").write_bytes(b"not a database" * 100)
    with sqlite3.connect(tmp_path / "measurements" / "b.db") as con:
        pd.DataFrame({"file_name": ["notes.txt"], "cell": [3]}).to_sql(
            "pivoted_counts", con, index=False)
    assert rep._archive_counts(tmp_path) is None


def test_a_plate_map_keyed_by_well_id_is_read(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame({"plateID": ["p1"], "wellID": ["A01"], "drug": ["x"]}).to_csv(
        path, index=False)
    frame = rep._archive_plate_map(path)
    assert list(frame["Well"]) == ["A01"]


def test_the_analysis_text_survives_an_unknown_version(monkeypatch):
    import importlib.metadata

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert "unknown" in rep._archive_analysis_text({})


def test_a_plate_map_without_plates_and_with_controls(screen, tmp_path,  # noqa: F811
                                                       monkeypatch):
    (screen / "notes_without_a_well.tif").write_bytes(b"x")
    plate_map = tmp_path / "wells.csv"
    pd.DataFrame({"well": ["A01", "B12"], "control": ["neg", ""],
                  "compound": ["DMSO", "drugX"]}).to_csv(plate_map, index=False)
    said = []

    def no_openpyxl(sheets, path):
        raise ImportError("openpyxl")

    import spacr.tabular as tabular

    monkeypatch.setattr(tabular, "_write_workbook", no_openpyxl)
    pkg = rep._write_archive_package(screen, tmp_path / "out",
                                     _form(screen, plate_map=str(plate_map)),
                                     progress=said.append)
    library = pd.read_csv(next((pkg / "idr").glob("*-library.txt")), sep="\t",
                          dtype=str, keep_default_na=False)
    assert "Control Type" in library.columns
    assert any("openpyxl is not installed" in line for line in said)


def test_validation_names_libraries_and_conditions_without_wells(screen,  # noqa: F811
                                                                 tmp_path):
    pkg = rep._write_archive_package(screen, tmp_path / "out", _form(screen))
    library = next((pkg / "idr").glob("*-library.txt"))
    frame = pd.read_csv(library, sep="\t", dtype=str, keep_default_na=False)
    frame.drop(columns=["Plate"]).to_csv(library, sep="\t", index=False)
    conditions = pkg / "mihcsme" / "AssayConditions.tsv"
    pd.DataFrame({"x": ["1"]}).to_csv(conditions, sep="\t", index=False)
    text = "\n".join(rep._validate_archive_package(pkg, verify_checksums=False))
    assert "column 'Plate' is missing" in text
    assert "no Plate and Well" in text


def _study(screen, tmp_path):  # noqa: F811
    second = tmp_path / "screen2"
    shutil.copytree(screen, second)
    return [screen, second]


def test_a_study_is_not_written_over_an_existing_one(screen, tmp_path):  # noqa: F811
    sources = _study(screen, tmp_path)
    rep._write_archive_study(sources, tmp_path / "study", _form(screen))
    with pytest.raises(ValueError, match="already exists"):
        rep._write_archive_study(sources, tmp_path / "study", _form(screen))


def test_study_validation_names_empty_fields_and_missing_blocks(screen,  # noqa: F811
                                                                tmp_path):
    pkg = rep._write_archive_study(_study(screen, tmp_path), tmp_path / "study",
                                   _form(screen))
    path = next((pkg / "idr").glob("*-study.txt"))
    lines = path.read_text(encoding="utf-8").splitlines()
    key = rep._IDR_STUDY_REQUIRED[0]
    lines = [f"{key}\t" if line.split("\t", 1)[0] == key else line
             for line in lines]
    last = max(i for i, line in enumerate(lines) if line.startswith("# Screen"))
    tail = next(i for i, line in enumerate(lines)
                if line.startswith("# Ontologies"))
    lines = lines[:last] + lines[tail:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pagetab = next((pkg / "biostudies").glob("*.pagetab.tsv"))
    section, keys = next(iter(rep._BIA_REQUIRED.items()))
    text = pagetab.read_text(encoding="utf-8")
    text = "\n".join(f"{keys[0]}\t" if line.split("\t", 1)[0] == keys[0]
                     else line for line in text.splitlines())
    pagetab.write_text(text + "\n", encoding="utf-8")
    problems = "\n".join(rep._validate_archive_study(pkg, verify_checksums=False))
    assert f"'{key}' has no value" in problems
    assert "screen block(s) for 2 screen(s)" in problems
    assert f"'{keys[0]}' has no value" in problems


def test_an_empty_keyring_answer_falls_back_to_the_file(tmp_path, monkeypatch):
    from spacr import run_journal

    class _Ring:
        def get_password(self, service, name):
            return ""

    monkeypatch.setattr(run_journal, "_notify_keyring", lambda: _Ring())
    monkeypatch.setattr(rep, "_zenodo_token_path", lambda: tmp_path / "t.json")
    assert rep._load_zenodo_token(sandbox=True) == ""


def test_zenodo_metadata_without_keywords_date_or_version(tmp_path, monkeypatch):
    import spacr

    monkeypatch.delattr(spacr, "__version__", raising=False)
    metadata = rep._zenodo_metadata({"title": "t", "keywords": " ; "}, tmp_path)
    assert "keywords" not in metadata and "publication_date" not in metadata
    assert "spaCR unknown" in metadata["notes"]


def test_a_bare_folder_stages_only_its_report(tmp_path):
    src = tmp_path / "run"
    src.mkdir()
    first, files, _meta = rep._zenodo_stage(src, tmp_path / "out", {"title": "t"},
                                            include_masks=True, run_dirs=[],
                                            search_journal=False)
    assert [f.name for f in files] == ["report.html"]
    second, _files, _meta = rep._zenodo_stage(src, tmp_path / "out",
                                              {"title": "t"}, run_dirs=[],
                                              search_journal=False)
    assert second.name == first.name + "-2"


def test_the_zenodo_opener_never_follows_a_redirect():
    opener = rep._zenodo_opener()
    handler = next(h for h in opener.handlers
                   if isinstance(h, urllib.request.HTTPRedirectHandler))
    assert handler.redirect_request(None, None, 302, "", {}, "http://x") is None
    assert Path


def test_a_plate_map_keyed_by_prc_is_read_as_is(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame({"prc": ["plate1_r2_c2"], "drug": ["x"]}).to_csv(path, index=False)
    assert list(rep._archive_plate_map(path)["Well"]) == ["B02"]


def test_authors_without_an_affiliation_carry_none(tmp_path):
    metadata = rep._zenodo_metadata({"title": "t", "authors": "Doe Jane"},
                                    tmp_path)
    assert metadata["creators"] == [{"name": "Doe, Jane"}]


def test_no_free_staging_folder_is_refused(tmp_path, monkeypatch):
    src = tmp_path / "run"
    src.mkdir()

    def taken(self, *a, **k):
        raise FileExistsError(str(self))

    monkeypatch.setattr(Path, "mkdir", taken)
    with pytest.raises(ValueError, match="No unused Zenodo staging folder"):
        rep._zenodo_stage(src, tmp_path / "out", {"title": "t"}, run_dirs=[],
                          search_journal=False)


def test_a_screen_without_counts_has_no_processed_file_to_copy(screen,  # noqa: F811
                                                               tmp_path):
    sources = _study(screen, tmp_path)
    shutil.rmtree(sources[1] / "measurements")
    pkg = rep._write_archive_study(sources, tmp_path / "study", _form(screen))
    assert not list((pkg / "idr").glob("*-screenB-processed.txt"))
