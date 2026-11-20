"""Item 961: in situ sequencing modes of the spacer-prefix decode.

Combinatorial barcodes with recombination flags, peak-prioritised calling and
nucleus or cytoplasm read assignment, on synthetic sites whose truth is known;
with every new setting empty the decode writes what it wrote before.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import ops_insitu as I
from spacr import ops_prefix as P
from tests.test_ops_read_mode_952 import _prefix, _same_files
from tests.test_ops_spacer_prefix_decode_920 import BASES, write_cpg_site

#: Construct -> (map bases, cycles 1-4; recomb bases, cycles 5-8; gene).
CONSTRUCTS = {"C1": ("ACGT", "GGTA", "G1"), "C2": ("TTGC", "CATG", "G2"),
              "C3": ("GACA", "TCCA", "G3"), "C4": ("CTAG", "AGTC", "G4")}

#: Cell -> (top-left corner, map construct, recomb construct).
CELLS = {1: ((2, 2), "C1", "C1"), 2: ((2, 30), "C2", "C2"),
         3: ((30, 2), "C1", "C3"), 4: ((30, 30), "C4", "C4")}


def _read(map_c, recomb_c, umi="AC"):
    """The 10 bases a spot of these constructs reads."""
    return CONSTRUCTS[map_c][0] + CONSTRUCTS[recomb_c][1] + umi


def _site(size=64, cycles=10):
    """A synthetic site, its cell and nucleus labels.

    Each cell is 24 px with a 10 px nucleus at its centre. Cells 1-3 hold
    three spots of their constructs; cell 4 holds two nuclear C4 spots and
    three cytoplasmic C2 spots.
    """
    stack = np.zeros((cycles, size, size, 4), np.uint16)
    labels = np.zeros((size, size), np.int32)
    nuclei = np.zeros((size, size), np.int32)
    yy, xx = np.mgrid[0:size, 0:size]

    def spot(y, x, bases, height=3000):
        blob = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2.0) * height
        for c in range(cycles):
            k = BASES.index(bases[c])
            stack[c, ..., k] = np.maximum(stack[c, ..., k], blob.astype(np.uint16))

    for cell, ((y0, x0), map_c, recomb_c) in CELLS.items():
        labels[y0:y0 + 24, x0:x0 + 24] = cell
        nuclei[y0 + 7:y0 + 17, x0 + 7:x0 + 17] = cell
        if cell == 4:
            for dy, dx in ((9, 9), (14, 13)):
                spot(y0 + dy, x0 + dx, _read("C4", "C4"))
            for dy, dx in ((2, 2), (2, 20), (20, 3)):
                spot(y0 + dy, x0 + dx, _read("C2", "C2"))
        else:
            for dy, dx in ((4, 4), (4, 18), (18, 10)):
                spot(y0 + dy, x0 + dx, _read(map_c, recomb_c))
    return stack, labels, nuclei


def _settings(tmp_path, name, **extra):
    """Write the synthetic site, its masks, a construct library and a layout."""
    root = tmp_path / "data"
    if not root.exists():
        stack, labels, nuclei = _site()
        write_cpg_site(root, 1, stack, labels)
        tifffile.imwrite(root / "masks" / "CP9_Well1_1_nuclei.tiff", nuclei.astype(np.uint16))
        pd.DataFrame([{"map_bc": m, "recomb_bc": r, "sgRNA": m + r + "AAAA", "gene_symbol": g}
                      for m, r, g in CONSTRUCTS.values()]).to_csv(tmp_path / "lib.csv",
                                                                  index=False)
        pd.DataFrame([
            {"segment": "map", "start": 1, "end": 4, "role": "map", "column": "map_bc",
             "max_distance": 0},
            {"segment": "recomb", "start": 5, "end": 8, "role": "recomb",
             "column": "recomb_bc", "max_distance": 0},
            {"segment": "umi", "start": 9, "end": 10, "role": "umi", "column": ""},
        ]).to_csv(tmp_path / "layout.csv", index=False)
    return {"ops_layout": "cpg_cropped", "ops_library_mode": "spacer_prefix",
            "cpg_root": str(root / "images_corrected_cropped"), "cpg_sbs_cycles": 10,
            "cpg_phenotype_channels": ["CorrDNA"], "ops_library": str(tmp_path / "lib.csv"),
            "ops_prefix_masks": str(root / "masks" / "{plate}_{well}_{site}_cells.tiff"),
            "ops_prefix_nuclei_masks": str(root / "masks" / "{plate}_{well}_{site}_nuclei.tiff"),
            "ops_prefix_length": 8, "ops_sbs_cycles": 10, "ops_read_threshold": 10.0,
            "ops_prefix_threshold_tuning": "none", "ops_read_mode": "log",
            "ops_prefix_decoy_k": 4, "n_workers": 1, "dst_root": str(tmp_path / name),
            **extra}


def _combinatorial(tmp_path, name, **extra):
    settings = _settings(tmp_path, name, ops_barcode_layout="combinatorial",
                         ops_barcode_layout_file=str(tmp_path / "layout.csv"), **extra)
    out = P.run_prefix_decode(settings)
    cells = pd.read_csv(out["paths"]["cells_CP9_W1"]).set_index("cell")
    reads = pd.read_parquet(out["paths"]["reads_CP9_W1"])
    return out, cells, reads


def test_combinatorial_identities_and_recombination_match_the_truth(tmp_path):
    out, cells, reads = _combinatorial(tmp_path, "combo")
    assert cells.loc[1, "construct"] == "ACGT+GGTA" and cells.loc[1, "gene"] == "G1"
    assert cells.loc[2, "gene"] == "G2" and cells.loc[1, "quality"] == "perfect"
    assert not cells.loc[3, "called"] and cells.loc[3, "recombined"] == 3
    assert reads.loc[reads["cell"] == 3, "recombined"].all()
    assert not reads.loc[reads["cell"].isin([1, 2]), "recombined"].any()
    assert set(reads["umi_seq"]) == {"AC"}
    well = out["report"]["wells"]["CP9_W1"]
    assert well["recombination_pct"] == pytest.approx(100.0 * 3 / reads["recomb_checked"].sum())
    assert well["segments"]["map"]["mapping_pct"] == 100.0
    assert well["shuffle_call_rate_pct"] < well["unique_combination_pct"]
    assert cells.loc[4, "gene"] == "G2"


def test_the_nucleus_compartment_keeps_the_nuclear_amplicon(tmp_path):
    _out, cells, reads = _combinatorial(tmp_path, "nucleus", ops_read_compartment="nucleus")
    assert cells.loc[4, "gene"] == "G4" and cells.loc[4, "reads"] == 2
    assert reads.loc[reads["cell"] == 4, "gene"].eq("G4").all()
    _out, cyto, _reads = _combinatorial(tmp_path, "cyto", ops_read_compartment="cytoplasm")
    assert cyto.loc[4, "gene"] == "G2" and 4 in cyto.index


def test_peak_sort_calls_the_brightest_barcode():
    reads = pd.DataFrame({"plate": "P", "well": "W1", "site": 1, "cell": 1,
                          "status": ["exact"] * 3, "prefix": ["AAA", "AAA", "CCC"],
                          "key": ["AAA", "AAA", "CCC"], "distance": 0,
                          "guide": ["a", "a", "c"], "gene": ["GA", "GA", "GC"],
                          "level": "guide", "quality": [0.5, 0.5, 0.2],
                          "peak": [100.0, 120.0, 900.0]})
    by_count = P._prefix_cell_calls(reads, sort="count").iloc[0]
    by_peak = P._prefix_cell_calls(reads, sort="peak").iloc[0]
    by_quality = P._prefix_cell_calls(reads, sort="quality").iloc[0]
    assert by_count["gene"] == "GA" and by_peak["gene"] == "GC" and by_quality["gene"] == "GA"
    assert by_peak["quality"] == "great"
    assert P._prefix_cell_calls(reads).equals(P.prefix_cell_calls(reads))
    with pytest.raises(ValueError, match="'peak' column"):
        P._prefix_cell_calls(reads.drop(columns="peak"), sort="peak")
    with pytest.raises(ValueError, match="recipe_simple"):
        P._prefix_cell_calls(reads, sort="count", method="recipe_simple")


def test_the_empty_settings_write_what_the_decode_wrote(tmp_path):
    base = _prefix(tmp_path, "base", ops_prefix_decoy_k=4)
    empty = _prefix(tmp_path, "empty", ops_prefix_decoy_k=4, ops_barcode_layout=None,
                    ops_call_sort="", ops_read_compartment="cell", ops_insitu_preset="none",
                    ops_recomb_min_quality=None)
    _same_files(base["folder"], empty["folder"])
    calls = sorted(f for f in os.listdir(base["folder"]) if f.startswith("cells_"))
    assert calls and calls == sorted(
        f for f in os.listdir(empty["folder"]) if f.startswith("cells_"))
    for name in calls:
        assert pd.read_csv(os.path.join(base["folder"], name)).equals(
            pd.read_csv(os.path.join(empty["folder"], name)))


def test_the_t7_preset_and_single_layout_peak_and_nucleus(tmp_path):
    settings = _settings(tmp_path, "t7", ops_insitu_preset="t7", ops_prefix_decoy_k=0,
                         ops_prefix_length=4)
    options = I._insitu_options(settings, P.prefix_settings(settings))
    assert (options["compartment"], options["call_sort"], options["min_reads"]) == \
        ("nucleus", "peak", 1)
    pd.DataFrame({"sgRNA": [m + "AAAA" for m, _r, _g in CONSTRUCTS.values()],
                  "gene_symbol": [g for _m, _r, g in CONSTRUCTS.values()]}).to_csv(
        tmp_path / "single.csv", index=False)
    out = P.run_prefix_decode({**settings, "ops_library": str(tmp_path / "single.csv")})
    cells = pd.read_csv(os.path.join(out["folder"], "cells_CP9_W1.csv")).set_index("cell")
    assert list(cells.index) == [4] and cells.loc[4, "gene"] == "G4"
    assert cells.loc[4, "quality"] == "perfect"
    saved = json.load(open(os.path.join(out["folder"], "quality_classes.json")))
    assert saved["settings"]["ops_insitu_preset"] == "t7"
    assert saved["settings"]["ops_read_compartment"] == "nucleus"


def test_refusals(tmp_path):
    settings = _settings(tmp_path, "bad")
    chosen = P.prefix_settings(settings)
    with pytest.raises(ValueError, match="needs ops_barcode_layout_file"):
        I._insitu_options({**settings, "ops_barcode_layout": "combinatorial"}, chosen)
    with pytest.raises(ValueError, match="ops_read_compartment must be one of"):
        I._insitu_options({**settings, "ops_read_compartment": "membrane"}, chosen)
    with pytest.raises(ValueError, match="per_well_decoy|one read threshold"):
        P.run_prefix_decode({**settings, "ops_barcode_layout": "combinatorial",
                             "ops_barcode_layout_file": str(tmp_path / "layout.csv"),
                             "ops_prefix_threshold_tuning": "per_well_decoy"})
    pd.DataFrame([{"segment": "r", "start": 1, "end": 4, "role": "recomb",
                   "column": "recomb_bc"}]).to_csv(tmp_path / "nomap.csv", index=False)
    with pytest.raises(ValueError, match="at least one 'map' segment"):
        I._load_layout(str(tmp_path / "nomap.csv"))
    with pytest.raises(ValueError, match="nucleus labels"):
        P.run_prefix_decode({**settings, "ops_read_compartment": "nucleus",
                             "ops_prefix_nuclei_masks": None, "ops_prefix_length": 4})


def test_the_new_settings_are_empty_by_default_and_alpha():
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import ALPHA_FEATURES

    keys = ALPHA_FEATURES[961]["settings"]
    assert len(keys) == 6 and all(OPS_DEFAULTS[key] is None for key in keys)
    # The nucleus labels are item 988's setting, which 961 reuses.
    assert "ops_prefix_nuclei_masks" in ALPHA_FEATURES[988]["settings"]
    assert OPS_DEFAULTS["ops_prefix_nuclei_masks"] is None


def test_the_grid_varies_compartment_and_call_sort_as_the_decode_runs_them(tmp_path):
    from spacr import ops_grid_search as G

    pd.DataFrame({"sgRNA": [m + "AAAA" for m, _r, _g in CONSTRUCTS.values()],
                  "gene_symbol": [g for _m, _r, g in CONSTRUCTS.values()]}).to_csv(
        tmp_path / "single.csv", index=False)
    settings = _settings(tmp_path, "grid", ops_prefix_length=4,
                         ops_library=str(tmp_path / "single.csv"),
                         ops_grid_params="ops_read_compartment=cell,nucleus; ops_call_sort=none,peak",
                         ops_grid_tiles="1")
    settings["ops_library"] = str(tmp_path / "single.csv")
    result = G._run_grid_search(settings)
    lines = pd.read_csv(result["csv"])
    pooled = lines[lines["tile"] == "pooled"].set_index("point")
    assert len(pooled) == 5 and bool(pooled.loc[0, "is_default"])
    nuclear = pooled[(pooled["ops_read_compartment"] == "nucleus")
                     & (pooled["ops_call_sort"] == "peak")].index[0]
    decoded = P.run_prefix_decode({**settings, "ops_read_compartment": "nucleus",
                                   "ops_call_sort": "peak",
                                   "dst_root": str(tmp_path / "decode_np")})
    summary = pd.read_csv(os.path.join(decoded["folder"], "site_summary.csv")).iloc[0]
    ours = lines[(lines["point"] == nuclear) & (lines["tile"] != "pooled")].iloc[0]
    for column in ("reads", "cells_in_labels", "cells_real", "cells_decoy"):
        assert int(ours[column]) == int(summary[column]), column
    with pytest.raises(ValueError, match="ops_read_compartment takes"):
        G._parse_grid("ops_read_compartment=membrane")
