"""Item 920: in-situ decode of sgRNA spacer prefixes.

A synthetic library pins the library model (collisions by gene, the cycle
mask, the whitelist distance), the matching rules (exact, one substitution,
one indel only under Levenshtein, equidistant reads ambiguous, unmeasured
bases as wildcards), the per-cell quality classes and the shuffled-whitelist
control; a synthetic cropped-site tree with rendered spots runs the settings
phase end to end and reruns identically.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import ops_prefix as P
from spacr.ops_engine import run_ops
from spacr.tabular import read_table

BASES = ("G", "T", "A", "C")


def library_table(rows):
    """A library frame from (spacer, gene) pairs."""
    return pd.DataFrame(rows, columns=["sgRNA", "gene_symbol"])


SPACERS = [
    ("ACGTACGTACGGTTAACCGG", "GENE1"),
    ("TTTTGGGGCCCCAAAATTTT", "GENE2"),
    ("GATTACAGATTACAGATTAC", "GENE3"),
    ("CCCCCCCCCCAAAAAAAAAA", "GENE4"),
    ("CCCCCCCCCCGGGGGGGGGG", "GENE5"),
    ("AGAGAGAGAGTTTTTTTTTT", "GENE6"),
    ("AGAGAGAGAGCCCCCCCCCC", "GENE6"),
]


def test_positions_follow_length_and_mask():
    assert P.prefix_positions(10) == tuple(range(1, 11))
    assert P.prefix_positions(10, [6], 12) == (1, 2, 3, 4, 5, 7, 8, 9, 10)
    with pytest.raises(ValueError, match="longer than"):
        P.prefix_positions(13, (), 12)
    with pytest.raises(ValueError, match="outside"):
        P.prefix_positions(10, [14], 12)
    with pytest.raises(ValueError, match="every cycle"):
        P.prefix_positions(1, [1], 12)
    with pytest.raises(ValueError, match="at least 1"):
        P.prefix_positions(0)


def test_collisions_by_gene_are_excluded_or_called_at_gene_level():
    lib = P.spacer_prefix_library(library_table(SPACERS))
    assert lib.summary["guides"] == 7 and lib.summary["keys"] == 5
    rows = lib.collisions.set_index("prefix")
    assert rows.loc["CCCCCCCCCC", "kind"] == "different_genes"
    assert rows.loc["AGAGAGAGAG", "kind"] == "same_gene"
    assert not rows["called"].any()
    assert "CCCCCCCCCC" not in lib.callable_keys and "AGAGAGAGAG" not in lib.callable_keys
    gene = P.spacer_prefix_library(library_table(SPACERS), collision="gene_level")
    assert gene.callable_keys["AGAGAGAGAG"] == (
        "AGAGAGAGAGCCCCCCCCCC|AGAGAGAGAGTTTTTTTTTT", "GENE6", "gene")
    assert "CCCCCCCCCC" not in gene.callable_keys
    assert gene.summary["gene_level_keys"] == 1
    assert gene.collisions.set_index("prefix").loc["AGAGAGAGAG", "called"]
    matched = P.match_prefix_reads(["AGAGAGAGAGAA", "CCCCCCCCCCAA"], gene)
    assert list(matched["status"]) == ["exact", "collision"]
    assert matched.loc[0, "gene"] == "GENE6" and matched.loc[0, "level"] == "gene"


def test_duplicate_rows_count_as_one_guide_and_library_errors(tmp_path):
    lib = P.spacer_prefix_library(library_table(SPACERS[:2] + SPACERS[:1]))
    assert lib.summary["guides"] == 2 and lib.summary["library_rows"] == 3
    assert lib.collisions.empty
    path = tmp_path / "Barcodes.csv"
    library_table(SPACERS).to_csv(path, index=False)
    assert P.spacer_prefix_library(str(path)).summary["keys"] == 5
    with pytest.raises(ValueError, match="no column"):
        P.spacer_prefix_library(library_table(SPACERS), gene_col="gene")
    with pytest.raises(ValueError, match="shorter than"):
        P.spacer_prefix_library(library_table([("ACGT", "G1")]))
    with pytest.raises(ValueError, match="ops_prefix_collision"):
        P.spacer_prefix_library(library_table(SPACERS), collision="keep")


def test_reads_match_exactly_after_one_substitution_or_not_at_all():
    lib = P.spacer_prefix_library(library_table(SPACERS))
    read = "ACGTACGTACGG"
    one = "ACGAACGTACGG"
    far = "AAAAAAAAAAAA"
    out = P.match_prefix_reads([read, one, far], lib)
    assert list(out["status"]) == ["exact", "corrected", "out_of_library"]
    assert list(out["distance"]) == [0, 1, -1]
    assert out.loc[1, "guide"] == "ACGTACGTACGGTTAACCGG" and out.loc[1, "gene"] == "GENE1"
    assert P.match_prefix_reads([one], lib, max_distance=0)["status"][0] == "out_of_library"
    two = "ACGAACGTAAGG"
    assert P.match_prefix_reads([two], lib)["status"][0] == "out_of_library"
    deep = P.match_prefix_reads([two], lib, max_distance=2)
    assert deep["status"][0] == "corrected" and deep["distance"][0] == 2
    assert P.match_prefix_reads([far], lib, max_distance=2)["status"][0] == "out_of_library"


def test_one_indel_is_corrected_only_under_levenshtein():
    lib = P.spacer_prefix_library(library_table(SPACERS))
    spacer = SPACERS[2][0]
    skipped = spacer[:3] + spacer[4:11]
    extra = spacer[:4] + "C" + spacer[4:9]
    for read in (skipped, extra):
        assert P.match_prefix_reads([read], lib)["status"][0] == "out_of_library"
        hit = P.match_prefix_reads([read], lib, metric="levenshtein")
        assert hit["status"][0] == "corrected" and hit["gene"][0] == "GENE3"
    with pytest.raises(ValueError, match="at most one"):
        P.match_prefix_reads([skipped], lib, metric="levenshtein", max_distance=2)
    with pytest.raises(ValueError, match="ops_prefix_metric"):
        P.match_prefix_reads([skipped], lib, metric="cosine")
    with pytest.raises(ValueError, match="0 or more"):
        P.match_prefix_reads([skipped], lib, max_distance=-1)


def test_masked_cycle_six_is_removed_and_keys_stay_unique():
    rows = [("ACGTAGGTACGGTTAACCGG", "G1"), ("ACGTATGTACGGTTAACCGG", "G2"),
            ("TTGTAGGTACGGTTAACCGG", "G3")]
    plain = P.spacer_prefix_library(library_table(rows))
    masked = P.spacer_prefix_library(library_table(rows), cycle_mask=[6])
    assert plain.summary["keys"] == 3 and not plain.collisions.size
    assert masked.summary["key_length"] == 9 and masked.summary["cycle_mask"] == [6]
    assert masked.summary["collision_keys_different_genes"] == 1
    third = P.match_prefix_reads(["TTGTANGTACGG", "TTGTACGTACGG"], masked)
    assert list(third["status"]) == ["exact", "exact"]
    assert set(third["gene"]) == {"G3"}
    assert third.loc[0, "key"] == "TTGTAGTAC"


def test_equidistant_reads_are_ambiguous_and_unmeasured_bases_are_wildcards():
    rows = [("AAAAAAAAAAGGGGGGGGGG", "G1"), ("AAAAAAAACCGGGGGGGGGG", "G2")]
    lib = P.spacer_prefix_library(library_table(rows))
    assert lib.summary["min_distance"] == 2
    tie = P.match_prefix_reads(["AAAAAAAACAGG"], lib)
    assert tie["status"][0] == "ambiguous" and tie["prefix"][0] is None
    unknown = P.match_prefix_reads(["AAAAAAAANAGG", "AAAAAAAANNGG", "NNNNNNNNNN", "AAAAAAAAAANN"],
                                   lib)
    assert list(unknown["status"]) == ["exact", "ambiguous", "ambiguous", "exact"]
    short = P.match_prefix_reads(["AAAAAAAAA"], lib)
    assert short["key"][0] == "AAAAAAAAAN" and short["status"][0] == "exact"
    corrected = P.match_prefix_reads(["TAAAAAAANAGG"], lib)
    assert corrected["status"][0] == "corrected" and corrected["gene"][0] == "G1"
    keyed = P.match_prefix_reads(["AAAAAAAACC"], lib, keyed=True)
    assert keyed["gene"][0] == "G2"


def test_whitelist_distance():
    assert P.whitelist_min_distance(["AAAA", "AAAT", "TTTT"]) == {
        "min_distance": 1, "min_distance_capped": False, "pairs_at_distance_1": 1}
    assert P.whitelist_min_distance(["AAAA", "AATT"])["min_distance"] == 2
    assert P.whitelist_min_distance(["AAAA", "TTTT"])["min_distance_capped"]
    assert P.whitelist_min_distance(["AAAA"])["min_distance"] is None


def test_shuffled_control_keeps_size_and_composition():
    lib = P.spacer_prefix_library(library_table(SPACERS))
    control = P.shuffled_library(lib, seed=0)
    again = P.shuffled_library(lib, seed=0)
    other = P.shuffled_library(lib, seed=3)
    assert control.keys == again.keys
    assert len(control.keys) <= len(lib.keys)
    assert sorted("".join(control.keys)) != [] and all(len(k) == 10 for k in control.keys)
    assert control.keys != other.keys or len(control.keys) < 3
    real_letters = sorted("".join(sorted(e)[0] for e in lib.extended.values()))
    shuffled_letters = sorted("".join(sorted(e)[0] for e in control.extended.values()))
    assert len(shuffled_letters) <= len(real_letters)
    assert all(control.callable_keys[k][1] == "control" for k in control.keys)
    hit = P.match_prefix_reads(control.keys[:1], control, metric="levenshtein")
    assert hit["status"][0] == "exact"


def test_cell_classes():
    assert P.classify_cell({}) == "none"
    assert P.classify_cell({"A": (2, 0)}) == "perfect"
    assert P.classify_cell({"A": (3, 1)}, min_reads=3) == "perfect"
    assert P.classify_cell({"A": (1, 0)}) == "great"
    assert P.classify_cell({"A": (0, 3)}) == "great"
    assert P.classify_cell({"A": (3, 0), "B": (1, 0)}) == "great"
    assert P.classify_cell({"A": (2, 0), "B": (1, 1)}) == "ambiguous"
    assert set(P.QUALITY_CLASSES) == {"perfect", "great", "ambiguous", "none"}


def test_cell_calls_table():
    lib = P.spacer_prefix_library(library_table(SPACERS))
    reads = ["ACGTACGTACGG", "ACGTACGTACGG", "TTTTGGGGCCCC", "TTTTGGGGCACC",
             "AAAAAAAAAAAA", "TTTTGGGGCCCC", "GATTACAGATTA", "ACGTACGTACGG"]
    frame = P.match_prefix_reads(reads, lib)
    frame.insert(0, "cell", [1, 1, 2, 2, 2, 3, 3, 0])
    frame.insert(0, "site", 1)
    frame.insert(0, "well", "W1")
    frame.insert(0, "plate", "P")
    cells = P.prefix_cell_calls(frame).set_index("cell")
    assert list(cells.index) == [1, 2, 3]
    assert cells.loc[1, "quality"] == "perfect" and cells.loc[1, "gene"] == "GENE1"
    assert cells.loc[2, "quality"] == "great" and cells.loc[2, "out_of_library"] == 1
    assert cells.loc[2, "top_exact"] == 1 and cells.loc[2, "top_share"] == 1.0
    assert cells.loc[3, "quality"] == "ambiguous" and pd.isna(cells.loc[3, "gene"])
    assert cells.loc[3, "second_reads"] == 1 and cells.loc[3, "barcodes"] == 2
    assert P.prefix_cell_calls(frame[frame["cell"] == 0]).empty
    assert P.prefix_cell_calls(frame.drop(columns="cell")).empty


def test_crosstalk_compensation_modes():
    rng = np.random.default_rng(0)
    n, cycles = 600, 3
    truth = rng.integers(0, 4, (n, cycles))
    pure = np.eye(4, dtype=np.float32)[truth] * 1000 + rng.random((n, cycles, 4)) * 10
    mixing = np.array([[1, 0.6, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.6], [0, 0, 0, 1]], np.float32)
    mixed = pure @ mixing
    for mode in ("per_cycle", "global"):
        out = P.compensate_prefix(mixed, mode)
        assert out.shape == mixed.shape and out.dtype == np.float32
        assert (out.argmax(-1) == truth).mean() > 0.99
    np.testing.assert_array_equal(P.compensate_prefix(mixed, "none"), mixed.astype(np.float32))
    holes = mixed.copy()
    holes[:5, 1] = np.nan
    assert np.isnan(P.compensate_prefix(holes, "per_cycle")[:5, 1]).all()
    assert np.isnan(P.compensate_prefix(holes, "global")[:5, 1]).all()
    assert P.compensate_prefix(np.zeros((0, 3, 4)), "global").shape == (0, 3, 4)
    with pytest.raises(ValueError, match="ops_prefix_crosstalk"):
        P.compensate_prefix(mixed, "median")


CELLS = {1: (8, 8), 2: (8, 40), 3: (40, 8), 4: (40, 40)}
CELL_GUIDES = {1: SPACERS[0][0], 2: SPACERS[1][0], 3: SPACERS[2][0], 4: "TGCATGCATGCATGCATGCA"}


def synthetic_site(cycles=12, size=64):
    """A sequencing stack with three spots per cell and the cell labels."""
    labels = np.zeros((size, size), np.int32)
    stack = np.zeros((cycles, size, size, 4), np.uint16)
    yy, xx = np.mgrid[0:size, 0:size]
    for cell, (y0, x0) in CELLS.items():
        labels[y0:y0 + 16, x0:x0 + 16] = cell
        spacer = CELL_GUIDES[cell]
        for dy, dx in ((4, 4), (4, 11), (11, 7)):
            y, x = y0 + dy, x0 + dx
            spot = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2.0) * 3000
            for c in range(cycles):
                k = BASES.index(spacer[c])
                stack[c, ..., k] = np.maximum(stack[c, ..., k], spot.astype(np.uint16))
    return stack, labels


def test_site_decode_reads_every_cell():
    stack, labels = synthetic_site()
    lib = P.spacer_prefix_library(library_table(SPACERS))
    control = P.shuffled_library(lib)
    reads = P.decode_prefix_site(stack, lib, labels=labels, threshold=10.0, control=control)
    assert len(reads) == 12 and set(reads["cell"]) == {1, 2, 3, 4}
    calls = P.prefix_cell_calls(reads.assign(plate="P", well="W1", site=1)).set_index("cell")
    assert calls.loc[1, "gene"] == "GENE1" and calls.loc[1, "quality"] == "perfect"
    assert calls.loc[2, "gene"] == "GENE2" and calls.loc[3, "gene"] == "GENE3"
    assert calls.loc[4, "quality"] == "none" and calls.loc[4, "out_of_library"] == 3
    fractions = P.read_fractions(reads)
    assert fractions["reads"] == 12 and fractions["frac_exact"] == 0.75
    assert fractions["control_frac_matched"] < fractions["frac_matched"]
    unlabelled = P.decode_prefix_site(stack, lib, threshold=10.0, crosstalk="per_cycle")
    assert (unlabelled["cell"] == 0).all() and (unlabelled["status"] == "exact").sum() == 9
    with pytest.raises(ValueError, match="cell labels are"):
        P.decode_prefix_site(stack, lib, labels=labels[:10], threshold=10.0)
    empty = P.decode_prefix_site(np.zeros_like(stack), lib, threshold=10.0)
    assert empty.empty and P.read_fractions(empty)["frac_matched"] == 0.0


def test_settings_are_checked():
    with pytest.raises(ValueError, match="needs ops_library"):
        P.prefix_settings({})
    chosen = P.prefix_settings({"ops_library": "x.csv", "ops_cycle_mask": "6; 7"})
    assert chosen["ops_cycle_mask"] == [6, 7] and chosen["ops_prefix_crosstalk"] == "none"
    with pytest.raises(ValueError, match="ops_prefix_crosstalk"):
        P.prefix_settings({"ops_library": "x", "ops_prefix_crosstalk": "argmax_median"})
    with pytest.raises(ValueError, match="ops_prefix_metric"):
        P.prefix_settings({"ops_library": "x", "ops_prefix_metric": "edit"})


def test_site_labels_sources(tmp_path):
    labels = np.arange(16, dtype=np.int32).reshape(4, 4)
    np.save(tmp_path / "P_W1_3.npy", labels)
    np.savez(tmp_path / "P_W1_4.npz", cells=labels)
    tifffile.imwrite(tmp_path / "P_Well1_5.tiff", labels)
    for name, site in (("{plate}_{well_id}_{site}.npy", 3), ("{plate}_{well_id}_{site}.npz", 4),
                       ("{plate}_{well}_{site}.tiff", 5)):
        got = P.site_labels(str(tmp_path / name), plate="P", well="Well1", well_id="W1", site=site)
        np.testing.assert_array_equal(got, labels)
    with pytest.raises(ValueError, match="no cell labels"):
        P.site_labels(str(tmp_path / "{site}.npy"), plate="P", well="Well1", well_id="W1", site=9)
    merged = tmp_path / "merged.npz"
    np.savez(merged, channels=labels[None], cells=labels)
    np.testing.assert_array_equal(
        P.site_labels(None, plate="P", well="Well1", well_id="W1", site=1, merged=str(merged)),
        labels)
    np.savez(merged, channels=labels[None])
    assert P.site_labels(None, plate="P", well="Well1", well_id="W1", site=1,
                         merged=str(merged)) is None
    assert P.site_labels(None, plate="P", well="Well1", well_id="W1", site=1) is None


def write_cpg_site(root, site, stack, labels):
    """Write one synthetic site in the cropped layout, its labels beside it."""
    well = root / "images_corrected_cropped" / "CP9_Well1"
    for c in range(stack.shape[0]):
        for k, base in enumerate(BASES):
            name = f"Cycle{c + 1:02d}_{base}"
            (well / name).mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(well / name / f"{name}_Site_{site}.tiff", stack[c, ..., k])
    for name in ("CorrDNA", "Cycle01_DAPI"):
        (well / name).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(well / name / f"{name}_Site_{site}.tiff",
                         np.full(labels.shape, 100, np.uint16))
    (root / "masks").mkdir(exist_ok=True)
    tifffile.imwrite(root / "masks" / f"CP9_Well1_{site}_cells.tiff", labels.astype(np.uint16))


def test_settings_run_decodes_and_reruns_identically(tmp_path):
    stack, labels = synthetic_site()
    for site in (1, 2):
        write_cpg_site(tmp_path, site, stack, labels)
    library = tmp_path / "Barcodes.csv"
    library_table(SPACERS).to_csv(library, index=False)
    settings = {"ops_layout": "cpg_cropped", "ops_library_mode": "spacer_prefix",
                "cpg_root": str(tmp_path / "images_corrected_cropped"),
                "cpg_phenotype_channels": ["CorrDNA"], "ops_library": str(library),
                "ops_prefix_masks": str(tmp_path / "masks" / "{plate}_{well}_{site}_cells.tiff"),
                "ops_read_threshold": 10.0, "n_workers": 1}
    outputs = []
    for name in ("run", "rerun"):
        out = run_ops({**settings, "dst_root": str(tmp_path / name)})
        assert out["phases"] == ["cpg_index", "cpg_merged", "prefix_decode"]
        outputs.append(out["prefix_decode"])
    report = outputs[0]
    assert report["sites"] == 2 and report["cells"] == 8 and report["called_cells"] == 6
    assert report["totals"]["exact"] == 18 and report["totals"]["out_of_library"] == 6
    folder = report["folder"]
    summary = read_table(os.path.join(folder, "site_summary.csv"), canonicalise=False, report=None)
    assert list(summary["cells_perfect"]) == [3, 3] and summary["labels"].all()
    assert (summary["control_frac_matched"] < summary["frac_matched"]).all()
    genes = read_table(os.path.join(folder, "cells_per_gene.csv"), canonicalise=False, report=None)
    assert dict(zip(genes["gene"], genes["cells"])) == {"GENE1": 2, "GENE2": 2, "GENE3": 2}
    classes = json.loads(open(os.path.join(folder, "quality_classes.json")).read())
    assert set(classes["classes"]) == set(P.QUALITY_CLASSES)
    assert classes["settings"]["ops_prefix_length"] == 10
    lib = json.loads(open(os.path.join(folder, "library_summary.json")).read())
    assert lib["collision_keys"] == 2
    collisions = read_table(os.path.join(folder, "library_prefix_collisions.csv"),
                            canonicalise=False, report=None)
    assert set(collisions["kind"]) == {"different_genes", "same_gene"}
    for name in ("reads_CP9_W1.parquet", "cells_CP9_W1.csv", "site_summary.csv",
                 "cells_per_guide.csv"):
        first = read_table(os.path.join(folder, name), canonicalise=False, report=None)
        second = read_table(os.path.join(outputs[1]["folder"], name), canonicalise=False,
                            report=None)
        pd.testing.assert_frame_equal(first, second)
    with pytest.raises(ValueError, match="cpg_sbs_cycles"):
        P.run_prefix_decode({**settings, "dst_root": str(tmp_path / "x"), "cpg_sbs_cycles": 8,
                             "ops_sbs_cycles": 12})
    with pytest.raises(ValueError, match="dst_root"):
        P.run_prefix_decode(settings)


def test_alpha_registration():
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import ALPHA_FEATURES

    assert "ops_prefix_length" in ALPHA_FEATURES[920]["settings"]
    assert OPS_DEFAULTS["ops_library_mode"] == "barcode"
    assert OPS_DEFAULTS["ops_prefix_crosstalk"] == "none"
