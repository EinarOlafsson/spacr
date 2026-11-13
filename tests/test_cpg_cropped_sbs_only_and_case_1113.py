"""A well with SBS images but no phenotype images is skipped with a note, and
cycle and channel folders match without regard to case (item 919 follow-up)."""
from __future__ import annotations

import logging
import os

import numpy as np
import pytest
import tifffile

from spacr import cpg_cropped as C
from spacr.ops_engine import run_ops

REAL_PHENO = ["CorrDNA", "CorrER", "CorrMito", "CorrPhalloidin", "CorrWGA"]
REAL_SBS = [f"Cycle{c:02d}_{b}" for c in (1, 2, 12) for b in "ACGT"] + ["Cycle01_DAPI"]


def touch(root, well, channel, site=1, folder=None):
    """An empty site file; the index reads paths only with headers=False."""
    folder = os.path.join(str(root), well, folder or channel)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{channel}_Site_{site}.tiff")
    open(path, "wb").close()
    return path


def real_names_tree(root):
    """CP186A Well1 complete, Well6 with SBS only, as in the S3 listing."""
    for channel in REAL_PHENO + REAL_SBS:
        touch(root, "CP186A_Well1", channel)
    for channel in REAL_SBS:
        touch(root, "CP186A_Well6", channel)
    return root


def test_a_well_with_sbs_but_no_phenotype_images_is_skipped_with_a_warning(tmp_path, caplog):
    root = real_names_tree(tmp_path / "images_corrected_cropped")
    with caplog.at_level(logging.WARNING, logger="spacr.cpg_cropped"):
        index = C.cpg_cropped_index(str(root), headers=False)
    assert set(index["well"]) == {"Well1"}
    assert index.attrs["skipped_wells"] == ["CP186A_Well6"]
    assert any("CP186A_Well6" in record.getMessage() for record in caplog.records)


def test_asking_for_an_sbs_only_well_by_name_raises_naming_missing_phenotype(tmp_path):
    root = real_names_tree(tmp_path / "images_corrected_cropped")
    with pytest.raises(ValueError, match=r"CP186A_Well6.*no phenotype images"):
        C.cpg_cropped_index(str(root), wells=["Well6"], headers=False)
    kept = C.cpg_cropped_index(str(root), wells=["W1"], headers=False)
    assert set(kept["well"]) == {"Well1"}


def test_a_well_with_phenotype_but_no_sbs_images_is_still_indexed(tmp_path):
    root = tmp_path / "images_corrected_cropped"
    for channel in REAL_PHENO:
        touch(root, "CP186A_Well2", channel)
    index = C.cpg_cropped_index(str(root), headers=False)
    assert set(index["modality"]) == {"phenotype"}
    assert index.attrs["skipped_wells"] == []


def test_the_real_listing_names_index_and_complete(tmp_path):
    root = real_names_tree(tmp_path / "images_corrected_cropped")
    index = C.cpg_cropped_index(str(root), headers=False)
    checked = C.check_cpg_complete(index, C.expected_channels(REAL_PHENO, 1))
    assert checked["skipped"] == [] and len(checked["index"]) == len(index)
    assert set(index["plate"]) == {"CP186A"}


def test_mixed_case_cycle_and_channel_folders_index_to_canonical_names(tmp_path):
    root = tmp_path / "images_corrected_cropped"
    for channel in REAL_PHENO:
        touch(root, "CP1_Well1", channel)
    touch(root, "CP1_Well1", "cycle01_a")
    touch(root, "CP1_Well1", "CYCLE01_g")
    touch(root, "CP1_Well1", "Cycle1_t")
    touch(root, "CP1_Well1", "Cycle01_dapi")
    touch(root, "CP1_well2", "CorrDNA")
    index = C.cpg_cropped_index(str(root), headers=False)
    sbs = index[index["modality"] == "sbs"]
    assert sorted(sbs["channel"]) == ["Cycle01_A", "Cycle01_DAPI", "Cycle01_G",
                                      "Cycle01_T"]
    assert set(sbs["cycle"]) == {1} and set(sbs["base"]) == {"A", "DAPI", "G", "T"}
    assert set(index["well"]) == {"Well1", "Well2"}
    assert all(os.path.isfile(path) for path in index["path"])
    missing = C.cpg_missing_files(index[index["well"] == "Well1"],
                                  C.expected_channels(REAL_PHENO, 1, "GTA"))
    assert list(missing["channel"]) == []


def test_a_file_whose_channel_differs_only_in_case_from_its_folder_is_accepted(tmp_path):
    root = tmp_path / "images_corrected_cropped"
    touch(root, "CP1_Well1", "CorrDNA", folder="corrdna")
    index = C.cpg_cropped_index(str(root), headers=False)
    assert list(index["channel"]) == ["corrdna"]


def test_the_engine_run_reports_skipped_wells(tmp_path):
    root = tmp_path / "images_corrected_cropped"
    plane = np.ones((8, 8), dtype=np.uint16)
    for well, channels in (("CP9_Well1", ["CorrDNA", "Cycle01_G", "CYCLE01_T"]),
                           ("CP9_Well6", ["Cycle01_G", "Cycle01_T"])):
        for channel in channels:
            folder = root / well / channel
            folder.mkdir(parents=True)
            tifffile.imwrite(str(folder / f"{channel}_Site_1.tiff"), plane)
    out = run_ops({"ops_layout": "cpg_cropped", "cpg_root": str(root),
                   "dst_root": str(tmp_path / "run"),
                   "cpg_phenotype_channels": ["CorrDNA"], "cpg_sbs_cycles": 1,
                   "cpg_sbs_bases": ["G", "T"], "cpg_sbs_dapi": "", "n_workers": 1})
    assert out["cpg"]["skipped_wells"] == ["CP9_Well6"]
    assert out["cpg"]["sites"] == 1 and out["cpg"]["skipped"] == []
