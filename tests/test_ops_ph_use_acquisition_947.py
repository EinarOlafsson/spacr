"""Phenotype folders that hold two acquisitions sharing a channel (item 947).

Funk plate B's phenotype folder holds two acquisitions, ``DAPI-GFP-A594``
and ``A594-AF750``, and both carry A594. Indexed together, a site has two
A594 files. The refusal names both acquisitions, the shared channel and the
setting that resolves it; ``ops_ph_use_acquisition`` reads one acquisition
alone; a folder with one acquisition is read as before. The tiles here are
empty files: indexing reads names only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spacr import ops_engine                                         # noqa: E402

FIRST, SECOND = "DAPI-GFP-A594", "A594-AF750"
CHANNELS = {FIRST: ("DAPI-GFP", "A594"), SECOND: ("A594", "AF750")}


def _plate(root: Path, layout: str, sites=(0, 1)) -> Path:
    """A phenotype folder: two acquisitions in two folders, in one folder, or one."""
    groups = (FIRST,) if layout == "single" else (FIRST, SECOND)
    for group in groups:
        folder = root / (group if layout == "folders" else "")
        folder.mkdir(parents=True, exist_ok=True)
        for site in sites:
            for token in CHANNELS[group]:
                (folder / f"20X_{group}_A1_{token}_Site-{site}.tif").touch()
    return root


@pytest.mark.parametrize("layout, one, other", [
    ("folders", SECOND, FIRST),
    ("names", SECOND, FIRST),
])
def test_the_refusal_names_both_acquisitions_the_channel_and_the_setting(
        tmp_path, layout, one, other):
    _plate(tmp_path, layout)
    with pytest.raises(ValueError) as raised:
        ops_engine._index_tiles(str(tmp_path), "uncycled")
    message = str(raised.value)
    assert ("holds two acquisitions that both carry channel A594: "
            f"{one} and {other} (well A1, site 0") in message
    assert f"Set ops_ph_use_acquisition to the one to use, for example {one}" in message
    assert "ops_ph_acquisitions" in message


@pytest.mark.parametrize("layout, chosen, tokens, where", [
    ("folders", FIRST, ["A594", "DAPI-GFP"], FIRST),
    ("folders", "a594-af750", ["A594", "AF750"], SECOND),
    ("names", FIRST, ["A594", "DAPI-GFP"], FIRST),
    ("names", SECOND, ["A594", "AF750"], SECOND),
    ("single", None, ["A594", "DAPI-GFP"], FIRST),
    ("single", FIRST, ["A594", "DAPI-GFP"], FIRST),
])
def test_the_setting_reads_one_acquisition_and_one_acquisition_is_unchanged(
        tmp_path, layout, chosen, tokens, where):
    _plate(tmp_path, layout)
    index = ops_engine._index_tiles(str(tmp_path), "uncycled", chosen)
    assert sorted(index["A1"][1]) == [0, 1]
    files = index["A1"][1][0]
    assert sorted(files) == tokens
    assert all(f"_{where}_A1_" in os.path.basename(path) for path in files.values())
    if layout == "single":
        assert index == ops_engine._index_tiles(str(tmp_path), "uncycled")


@pytest.mark.parametrize("layout, extra, refusal", [
    ("folders", {}, "both carry channel A594: A594-AF750 and DAPI-GFP-A594"),
    ("folders", {"ops_ph_use_acquisition": "GFP-only"},
     "ops_ph_use_acquisition is GFP-only, but no tile under .* belongs to it; "
     "the acquisitions there are A594-AF750, DAPI-GFP-A594"),
    ("folders", {"ops_ph_use_acquisition": FIRST,
                 "ops_ph_acquisitions": {FIRST: "DAPI,GFP,A594"}},
     "ops_ph_use_acquisition \\(DAPI-GFP-A594\\) and ops_ph_acquisitions are "
     "both set"),
])
def test_run_ops_refuses_before_reading_a_pixel(tmp_path, layout, extra, refusal):
    phenotype = _plate(tmp_path / "phenotype", layout)
    sequencing = tmp_path / "sequencing"
    sequencing.mkdir()
    (sequencing / "10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-0.tif").touch()
    settings = {"genotype_source": str(sequencing),
                "phenotype_source": str(phenotype),
                "dst_root": str(tmp_path / "out"), "plate": "P1",
                "ops_gpu": False, **extra}
    with pytest.raises(ValueError, match=refusal):
        ops_engine.run_ops(settings, wells=["A1"], phases=("phenotype",))


PLATE_B = Path("/nas_mnt/data/ops/OpticalPooledScreens_data/screenA/"
               "20200202_6W-LaC024B/phenotype/images/input")
PROOF_RUN = Path("/nas_mnt/data/sync/papers/BMCspacr/reanalysis/"
                 "proof_stage1_stage2/ready_check/out/run_reports/B_ph2/"
                 "A1.phenotype.json")


@pytest.mark.skipif(not (PLATE_B / FIRST).is_dir(),
                    reason="Funk et al. 2022 plate B is not mounted")
def test_plate_b_with_the_setting_indexes_what_the_proof_check_placed():
    with pytest.raises(ValueError, match=f"carry channel A594: {SECOND} and {FIRST}"):
        ops_engine._index_tiles(str(PLATE_B), "uncycled")
    chosen = ops_engine._index_tiles(str(PLATE_B), "uncycled", FIRST)
    assert chosen == ops_engine._index_tiles(str(PLATE_B / FIRST), "uncycled")
    if PROOF_RUN.is_file():
        report = json.loads(PROOF_RUN.read_text(encoding="utf-8"))
        assert len(chosen["A1"][1]) == report["report"]["phenotype"]["tiles_found"]
