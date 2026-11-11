"""Item 919: the Cell Painting Gallery cropped-site reader.

A synthetic ``images_corrected_cropped`` tree (two wells, three sites, folders
created in shuffled order) must index from paths alone, stack every plane
exactly as ``tifffile.imread`` returns it in the pinned order, refuse a
missing file by name, refuse a grid mismatch unless resampling is set, refuse
a well folder without ``_Well<N>``, and give identical arrays and index on a
settings rerun.
"""
from __future__ import annotations

import json
import os
import random
import zlib

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import cpg_cropped as C
from spacr.ops_engine import run_ops

PHENO = ["CorrDNA", "CorrPhalloidin", "CorrMito"]
BASES = ["G", "T", "A", "C"]
CYCLES = 3
SHAPE = (24, 20)


def plane_value(well, site, channel, shape=SHAPE):
    """A deterministic uint16 plane unique to (well, site, channel)."""
    seed = zlib.crc32(f"{well}|{site}|{channel}".encode())
    return np.random.default_rng(seed).integers(0, 65535, shape, dtype=np.uint16)


def channel_names(cycles=CYCLES, dapi=True):
    """Every channel folder of a complete synthetic site."""
    names = list(PHENO) + [f"Cycle{c:02d}_{b}" for c in range(1, cycles + 1) for b in BASES]
    return names + (["Cycle01_DAPI"] if dapi else [])


def make_tree(root, wells=("CP1_Well1", "CP1_Well2"), sites=(1, 2, 10), seed=0,
              cycles=CYCLES, shapes=None, skip=()):
    """Write a cropped-site tree, creating folders in a shuffled order."""
    jobs = [(w, ch, s) for w in wells for ch in channel_names(cycles) for s in sites]
    random.Random(seed).shuffle(jobs)
    for well, channel, site in jobs:
        if (well, channel, site) in skip:
            continue
        folder = os.path.join(root, well, channel)
        os.makedirs(folder, exist_ok=True)
        shape = (shapes or {}).get(channel, SHAPE)
        tifffile.imwrite(os.path.join(folder, f"{channel}_Site_{site}.tiff"),
                         plane_value(well, site, channel, shape), compression="zlib")
    return root


def settings_for(root, dst, **extra):
    """Run settings for the synthetic tree."""
    out = {"ops_layout": "cpg_cropped", "cpg_root": str(root), "dst_root": str(dst),
           "cpg_phenotype_channels": PHENO, "cpg_sbs_cycles": CYCLES,
           "cpg_sbs_bases": BASES, "cpg_sbs_dapi": "Cycle01_DAPI", "n_workers": 1}
    out.update(extra)
    return out


def test_well_folder_names_parse_and_refuse():
    assert C.parse_well_folder("CP186A_Well1") == {
        "plate": "CP186A", "well": "Well1", "well_id": "W1", "number": 1}
    assert C.parse_well_folder("A_B_Well12")["plate"] == "A_B"
    with pytest.raises(ValueError, match="_Well<N>"):
        C.parse_well_folder("CP186A-W1")


def test_site_lists_parse():
    assert C.parse_sites(None) is None and C.parse_sites("") is None and C.parse_sites([]) is None
    assert C.parse_sites(3) == [3]
    assert C.parse_sites("1-3, 7,7") == [1, 2, 3, 7]
    assert C.parse_sites([5, "2"]) == [2, 5]


def test_index_from_paths_in_pinned_order(tmp_path):
    root = make_tree(tmp_path / "images_corrected_cropped")
    index = C.cpg_cropped_index(str(root))
    assert list(index.columns) == list(C.INDEX_COLUMNS)
    assert len(index) == 2 * 3 * len(channel_names())
    assert set(index["plate"]) == {"CP1"} and set(index["well_id"]) == {"W1", "W2"}
    one = index[(index["well"] == "Well1") & (index["site"] == 2)]
    assert list(one["modality"].iloc[:len(PHENO)]) == ["phenotype"] * len(PHENO)
    assert set(one.loc[one["modality"] == "sbs", "base"]) == set(BASES) | {"DAPI"}
    assert set(index["shape"]) == {"24x20"} and set(index["dtype"]) == {"uint16"}
    assert sorted(index["site"].unique()) == [1, 2, 10]
    no_headers = C.cpg_cropped_index(str(root), headers=False)
    assert set(no_headers["shape"]) == {""}
    assert len(C.cpg_cropped_index(str(root), wells=["W2"], sites="1-2")) == 2 * len(channel_names())
    assert set(C.cpg_cropped_index(str(root), wells="CP1_Well1")["well"]) == {"Well1"}
    single = C.cpg_cropped_index(str(root / "CP1_Well2"), md5=True)
    assert set(single["well"]) == {"Well2"}
    row = single.iloc[0]
    assert row["md5"] == C.file_md5(row["path"])


def test_shuffled_creation_order_gives_identical_stacks(tmp_path):
    first = make_tree(tmp_path / "a" / "images_corrected_cropped", seed=1)
    second = make_tree(tmp_path / "b" / "images_corrected_cropped", seed=2)
    ia, ib = C.cpg_cropped_index(str(first)), C.cpg_cropped_index(str(second))
    columns = [c for c in C.INDEX_COLUMNS if c not in ("path",)]
    pd.testing.assert_frame_equal(ia[columns], ib[columns])
    for well in ("Well1", "Well2"):
        for site in (1, 2, 10):
            a = C.load_cpg_site(ia, "CP1", well, site, phenotype_channels=PHENO,
                                sbs_cycles=CYCLES, sbs_bases=BASES)
            b = C.load_cpg_site(ib, "CP1", well, site, phenotype_channels=PHENO,
                                sbs_cycles=CYCLES, sbs_bases=BASES)
            for part in ("phenotype", "sbs", "dapi"):
                assert C.array_md5(a[part]) == C.array_md5(b[part])


def test_every_plane_equals_its_file(tmp_path):
    root = make_tree(tmp_path / "images_corrected_cropped")
    index = C.cpg_cropped_index(str(root))
    site = C.load_cpg_site(index, "CP1", "W1", 10, phenotype_channels=PHENO,
                           sbs_cycles=CYCLES, sbs_bases=BASES)
    assert site["phenotype"].shape == SHAPE + (len(PHENO),)
    assert site["phenotype"].dtype == np.uint16
    assert site["sbs"].shape == (CYCLES,) + SHAPE + (4,)
    for k, name in enumerate(PHENO):
        expected = tifffile.imread(site["paths"][name])
        assert C.array_md5(site["phenotype"][..., k]) == C.array_md5(expected)
    for c in range(CYCLES):
        for k, base in enumerate(BASES):
            name = f"Cycle{c + 1:02d}_{base}"
            assert site["sbs_channels"][c][k] == name
            np.testing.assert_array_equal(site["sbs"][c, ..., k],
                                          tifffile.imread(site["paths"][name]))
    np.testing.assert_array_equal(site["dapi"], plane_value("CP1_Well1", 10, "Cycle01_DAPI"))
    reordered = C.load_cpg_site(index, "CP1", "Well1", 10, phenotype_channels=PHENO[::-1],
                                parts=("phenotype",))
    np.testing.assert_array_equal(reordered["phenotype"][..., 0], site["phenotype"][..., -1])
    assert reordered["sbs"] is None


def test_missing_cycle_file_is_named_or_its_site_skipped(tmp_path):
    root = make_tree(tmp_path / "images_corrected_cropped", wells=("CP1_Well1",),
                     cycles=7, skip={("CP1_Well1", "Cycle07_C", 2)})
    index = C.cpg_cropped_index(str(root))
    channels = C.expected_channels(PHENO, 7, BASES, "Cycle01_DAPI")
    with pytest.raises(ValueError, match="Cycle07_C/Cycle07_C_Site_2.tiff"):
        C.check_cpg_complete(index, channels)
    kept = C.check_cpg_complete(index, channels, missing="skip_site")
    assert kept["skipped"] == [("CP1", "Well1", 2)]
    assert sorted(kept["index"]["site"].unique()) == [1, 10]
    with pytest.raises(ValueError, match="cpg_missing"):
        C.check_cpg_complete(index, channels, missing="ignore")
    with pytest.raises(ValueError, match="no file for"):
        C.load_cpg_site(index, "CP1", "Well1", 2, phenotype_channels=PHENO,
                        sbs_cycles=7, sbs_bases=BASES)


def test_many_missing_files_are_counted(tmp_path):
    root = make_tree(tmp_path / "t", wells=("CP1_Well1",), sites=(1,))
    index = C.cpg_cropped_index(str(root))
    channels = C.expected_channels(PHENO, 9, BASES, None)
    with pytest.raises(ValueError, match="and 4 more"):
        C.check_cpg_complete(index, channels)


def test_grid_mismatch_is_refused_unless_resampling(tmp_path):
    shapes = {name: (12, 10) for name in channel_names() if name.startswith("Cycle")}
    root = make_tree(tmp_path / "images_corrected_cropped", wells=("CP1_Well1",),
                     sites=(1,), shapes=shapes)
    index = C.cpg_cropped_index(str(root))
    with pytest.raises(ValueError, match="cpg_sbs_resample"):
        C.load_cpg_site(index, "CP1", "Well1", 1, phenotype_channels=PHENO,
                        sbs_cycles=CYCLES, sbs_bases=BASES)
    for method in ("nearest", "linear"):
        site = C.load_cpg_site(index, "CP1", "Well1", 1, phenotype_channels=PHENO,
                               sbs_cycles=CYCLES, sbs_bases=BASES, resample=method)
        assert site["sbs"].shape == (CYCLES,) + SHAPE + (4,)
        assert site["sbs"].dtype == np.uint16 and site["dapi"].shape == SHAPE
        assert site["resample_factor"] == (2.0, 2.0)
    with pytest.raises(ValueError, match="cpg_sbs_resample must be"):
        C.load_cpg_site(index, "CP1", "Well1", 1, resample="cubic")


def test_planes_of_one_modality_must_agree(tmp_path):
    root = make_tree(tmp_path / "a", wells=("CP1_Well1",), sites=(1,),
                     shapes={"CorrMito": (10, 10)})
    index = C.cpg_cropped_index(str(root))
    with pytest.raises(ValueError, match="painting channels differ"):
        C.load_cpg_site(index, "CP1", "Well1", 1, phenotype_channels=PHENO, parts=("phenotype",))
    root = make_tree(tmp_path / "b", wells=("CP1_Well1",), sites=(1,),
                     shapes={"Cycle02_A": (10, 10)})
    index = C.cpg_cropped_index(str(root))
    with pytest.raises(ValueError, match="sequencing planes differ"):
        C.load_cpg_site(index, "CP1", "Well1", 1, sbs_cycles=CYCLES, sbs_bases=BASES,
                        parts=("sbs",))


def test_bad_trees_get_clear_errors(tmp_path):
    with pytest.raises(ValueError, match="is not a folder"):
        C.cpg_cropped_index(str(tmp_path / "nowhere"))
    loose = tmp_path / "CP186A-W1"
    (loose / "CorrDNA").mkdir(parents=True)
    tifffile.imwrite(loose / "CorrDNA" / "CorrDNA_Site_1.tiff", np.zeros((4, 4), np.uint16))
    with pytest.raises(ValueError, match="CP186A-W1.*_Well<N>"):
        C.cpg_cropped_index(str(loose))
    mixed = tmp_path / "mixed"
    make_tree(mixed, wells=("CP1_Well1",), sites=(1,))
    (mixed / "notes").mkdir()
    with pytest.raises(ValueError, match="not <plate>_Well<N> folders"):
        C.cpg_cropped_index(str(mixed))
    wrong = tmp_path / "wrong" / "CP1_Well1" / "CorrDNA"
    wrong.mkdir(parents=True)
    tifffile.imwrite(wrong / "CorrER_Site_1.tiff", np.zeros((4, 4), np.uint16))
    (wrong / "readme.txt").write_text("x")
    with pytest.raises(ValueError, match="names channel CorrER"):
        C.cpg_cropped_index(str(tmp_path / "wrong"))
    empty = tmp_path / "empty" / "CP1_Well1" / "CorrDNA"
    empty.mkdir(parents=True)
    (empty.parent / "stray.txt").write_text("x")
    with pytest.raises(ValueError, match="no <Channel>_Site_<S>.tiff"):
        C.cpg_cropped_index(str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="cpg_root"):
        C.cpg_settings({})


def test_settings_run_writes_index_channels_and_merged_and_reruns_identically(tmp_path):
    root = make_tree(tmp_path / "images_corrected_cropped")
    dst = tmp_path / "run"
    settings = settings_for(root, dst, cpg_md5=True, n_workers=2)
    out = run_ops(settings)
    assert out["layout"] == "cpg_cropped" and out["phases"] == ["cpg_index", "cpg_merged"]
    report = out["cpg"]
    assert report["files"] == 2 * 3 * len(channel_names()) and report["sites"] == 6
    from spacr.tabular import read_table

    saved = read_table(report["index"], canonicalise=False, report=None)
    assert len(saved) == report["files"] and "md5" in saved.columns
    record = json.loads((dst / "cpg_channels.json").read_text())
    assert record["phenotype_channels"] == PHENO and record["sbs_bases"] == BASES
    assert record["stacking_order"][:len(PHENO)] == PHENO
    layout = json.loads((dst / "merged" / "LAYOUT.json").read_text())
    assert layout["source"] == "cpg_cropped" and layout["masks"] == []
    path = C.merged_path(str(dst), "CP1", "W2", 10)
    with np.load(path) as merged:
        channels = merged["channels"]
        meta = json.loads(str(merged["meta"]))
    assert channels.shape == (len(PHENO),) + SHAPE and channels.dtype == np.uint16
    for k, name in enumerate(PHENO):
        np.testing.assert_array_equal(channels[k], plane_value("CP1_Well2", 10, name))
        assert meta["md5"][name] == C.array_md5(channels[k])
    first = {p: C.array_md5(np.load(p)["channels"])
             for p in sorted(str(x) for x in (dst / "merged").rglob("*.npz"))}
    again = run_ops(settings, phases=["cpg_merged"])
    assert again["cpg"]["merged"]["resumed"] == 6 and again["cpg"]["merged"]["written"] == 0
    rerun = tmp_path / "rerun"
    run_ops(settings_for(root, rerun, cpg_md5=True))
    second = {p.replace(str(rerun), str(dst)): C.array_md5(np.load(p)["channels"])
              for p in sorted(str(x) for x in (rerun / "merged").rglob("*.npz"))}
    assert first == second and len(first) == 6
    assert (dst / "cpg_index.csv").read_bytes() == (rerun / "cpg_index.csv").read_bytes()


def test_skipped_sites_are_listed_in_the_run_folder(tmp_path):
    root = make_tree(tmp_path / "t", wells=("CP1_Well1",),
                     skip={("CP1_Well1", "CorrMito", 2)})
    dst = tmp_path / "run"
    out = run_ops(settings_for(root, dst, cpg_missing="skip_site"), phases=["cpg_index"])
    assert out["cpg"]["skipped"] == [["CP1", "Well1", 2]]
    assert (dst / "cpg_missing.csv").is_file()
    with pytest.raises(ValueError, match="CorrMito_Site_2"):
        run_ops(settings_for(root, tmp_path / "strict"), phases=["cpg_index"])


def test_the_engine_refuses_mixed_layouts_and_phases(tmp_path):
    root = make_tree(tmp_path / "t", wells=("CP1_Well1",), sites=(1,))
    settings = settings_for(root, tmp_path / "run")
    with pytest.raises(ValueError, match="already stitched"):
        run_ops(settings, phases=["cpg_index", "decode"])
    with pytest.raises(ValueError, match="spacer_prefix"):
        run_ops(settings, phases=["prefix_decode"])
    with pytest.raises(ValueError, match="set ops_layout"):
        run_ops({**settings, "ops_layout": "tiles"}, phases=["cpg_index"])
    with pytest.raises(ValueError, match="dst_root"):
        C.run_cpg_phases({**settings, "dst_root": None})
    with pytest.raises(ValueError, match="unknown cropped-site phases"):
        C.run_cpg_phases(settings, ["stitch"])


def test_settings_normalise_strings():
    pinned = C.cpg_settings({"cpg_root": "/x", "cpg_wells": "Well1, W2",
                             "cpg_phenotype_channels": "CorrDNA,CorrER",
                             "cpg_sbs_bases": "g,t,a,c", "cpg_sbs_dapi": "",
                             "cpg_sites": "1-2"})
    assert pinned["cpg_wells"] == ["Well1", "W2"]
    assert pinned["cpg_phenotype_channels"] == ["CorrDNA", "CorrER"]
    assert pinned["cpg_sbs_bases"] == BASES and pinned["cpg_sbs_dapi"] is None
    assert pinned["cpg_sites"] == [1, 2] and pinned["cpg_missing"] == "error"
    assert C.expected_channels(["A"], 1, ["G"], None) == ["A", "Cycle01_G"]


def test_alpha_registration():
    from spacr.settings import ALPHA_FEATURES

    assert "cpg_root" in ALPHA_FEATURES[919]["settings"]
    assert "ops_layout" in ALPHA_FEATURES[919]["settings"]
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["ops_layout"] == "tiles"
    assert OPS_DEFAULTS["cpg_sbs_bases"] == BASES
