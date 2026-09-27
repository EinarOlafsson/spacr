"""Archive submission packages: MIHCSME/REMBI metadata for IDR and BioStudies.

A small screen (two plates, three wells, saved settings, object counts and a
plate map) is packaged and must pass the template checks; a package with a
missing value, an unknown plate or a changed file must not.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from spacr import report as rep


@pytest.fixture
def screen(tmp_path) -> Path:
    """Two plates of named images, settings, counts and a plate map."""
    src = tmp_path / "screen1"
    (src / "settings").mkdir(parents=True)
    (src / "measurements").mkdir()
    wells = [("plate1", "A01"), ("plate1", "B12"), ("plate2", "A01")]
    for plate, well in wells:
        for channel in (1, 2):
            (src / f"{plate}_{well}_T0001F001L01A01Z01C0{channel}.tif"
             ).write_bytes(f"{plate}{well}{channel}".encode() * 10)
    pd.DataFrame({"Key": ["experiment", "metadata_type", "channels",
                          "nucleus_channel", "cell_channel",
                          "cell_model_name", "magnification"],
                  "Value": ["drugscreen", "cellvoyager", "[0, 1]", "0", "1",
                            "cpsam", "20"]}
                 ).to_csv(src / "settings" / "gen_mask_settings.csv",
                          index=False)
    with sqlite3.connect(src / "measurements" / "measurements.db") as con:
        pd.DataFrame({"file_name": ["plate1_A01_1_1.npy", "plate1_A01_2_1.npy",
                                    "plate2_A01_1_1.npy"],
                      "cell_before_filtration": [10, 12, 7]}
                     ).to_sql("pivoted_counts", con, index=False)
    plate_map = tmp_path / "map.csv"
    pd.DataFrame({"plateID": ["plate1", "plate1", "plate2"],
                  "well": ["A01", "B12", "A01"],
                  "compound": ["DMSO", "drugX", "drugX"],
                  "concentration": [0, 1.5, 3.0]}).to_csv(plate_map,
                                                          index=False)
    return src


def _form(screen: Path, **extra):
    form = rep._archive_form_defaults(screen)
    form.update(description="Host cells treated with drugX.",
                authors="Doe Jane; Roe Rick", email="jane@example.org",
                affiliation="Example Institute", cell_line="HeLa",
                plate_map=str(screen.parent / "map.csv"))
    form.update(extra)
    return form


def test_wells_are_read_from_every_naming():
    assert rep._archive_well("plate1_E01_T0001F001.tif") == ("plate1", "E01")
    assert rep._archive_well("plate1_B3_1_1.npy") == ("plate1", "B03")
    assert rep._archive_well("plate7_r12_c2_f17_o1.png") == ("plate7", "L02")
    assert rep._archive_well("notes.txt") is None
    assert rep._archive_row_letters(27) == "AA"


def test_the_form_is_filled_from_the_settings(screen):
    form = rep._archive_form_defaults(screen)
    assert form["title"] == "drugscreen"
    assert form["microscope"] == "Yokogawa CellVoyager"
    assert form["license"] == "CC BY 4.0"
    assert set(form) == {k for k, _r in rep._ARCHIVE_FORM_FIELDS}


def test_the_example_screen_package_passes_every_template(screen, tmp_path):
    out = tmp_path / "out"
    before = sorted(p.relative_to(screen) for p in screen.rglob("*"))
    pkg = rep._write_archive_package(screen, out, _form(screen))
    assert pkg == out / "drugscreen"
    assert rep._validate_archive_package(pkg) == []

    library = pd.read_csv(pkg / "idr" / "drugscreen-screenA-library.txt",
                          sep="\t", keep_default_na=False)
    assert list(library[["Plate", "Well"]].itertuples(index=False,
                                                        name=None)) == [
        ("plate1", "A01"), ("plate1", "B12"), ("plate2", "A01")]
    assert list(library["Characteristics [compound]"]) == [
        "DMSO", "drugX", "drugX"]
    assert set(library["Term Source 1 Accession"]) == {"NCBITaxon_9606"}
    processed = pd.read_csv(pkg / "idr" / "drugscreen-screenA-processed.txt",
                            sep="\t")
    assert processed["cell before filtration count"].tolist() == [22, 7]

    study = rep._archive_read_kv(pkg / "idr" / "drugscreen-study.txt")
    assert study["Screen Size"][:3] == ["Plates: 2", "5D Images: 3",
                                        "Planes: 6"]
    assert study["Study Person Last Name"] == ["Doe"]
    blocks = dict(rep._archive_read_pagetab(
        pkg / "biostudies" / "drugscreen.pagetab.tsv"))
    assert blocks["Image acquisition"]["Imaging instrument"] == \
        "Yokogawa CellVoyager"
    assert "cpsam" in blocks["Image analysis"]["Image analysis overview"]
    listed = pd.read_csv(pkg / "biostudies" / "file_list.tsv", sep="\t")
    assert len(listed) == 6 and all((screen / f).is_file()
                                    for f in listed["Files"])
    assert (pkg / "mihcsme.xlsx").is_file()
    sheets = pd.read_excel(pkg / "mihcsme.xlsx", sheet_name=None)
    assert list(sheets) == list(rep._MIHCSME_SHEETS)
    assert not any(p.name.startswith(".") for p in pkg.rglob("*"))
    assert sorted(p.relative_to(screen) for p in screen.rglob("*")) == before


def test_copied_images_are_listed_and_checked_inside_the_package(screen,
                                                                 tmp_path):
    pkg = rep._write_archive_package(screen, tmp_path / "out", _form(screen),
                                     copy_images=True)
    assert len(list((pkg / "data").iterdir())) == 6
    assert rep._validate_archive_package(pkg) == []
    (pkg / "data" / "plate2_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"x")
    problems = rep._validate_archive_package(pkg)
    assert problems == [
        "checksums: data/plate2_A01_T0001F001L01A01Z01C01.tif does not match"]


def test_a_package_missing_what_the_templates_need_is_rejected(screen,
                                                                tmp_path):
    pkg = rep._write_archive_package(screen, tmp_path / "out",
                                     _form(screen, description="", email=""))
    problems = rep._validate_archive_package(pkg, verify_checksums=False)
    assert "IDR study: 'Study Description' has no value" in problems
    assert "BioStudies Study: 'Description' has no value" in problems
    assert "BioStudies Author: 'Email' has no value" in problems
    assert "MIHCSME InvestigationInformation: 'Submitter Email' has no value" \
        in problems

    plates = pkg / "idr" / "drugscreen-screenA-plates.txt"
    plates.write_text("Plate\tDirectory\nplate1\t.\n")
    assert any("not on the plate list" in p for p in
               rep._validate_archive_package(pkg, verify_checksums=False))


def test_a_folder_without_images_cannot_be_packaged(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No image files"):
        rep._write_archive_package(tmp_path / "empty", tmp_path / "out", {})
    assert rep._validate_archive_package(tmp_path) == [
        f"{tmp_path} has no readable archive_manifest.json"]


def test_the_manifest_records_the_form_and_source(screen, tmp_path):
    pkg = rep._write_archive_package(screen, tmp_path / "out", _form(screen))
    manifest = json.loads((pkg / "archive_manifest.json").read_text())
    assert manifest["source"] == str(screen.resolve())
    assert manifest["form"]["cell_line"] == "HeLa"
