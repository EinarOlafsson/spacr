"""Item 923: CORUM within-complex r, STRING AUPRC and PERISCOPE's zero-TPM hit rule.

Synthetic genes with planted feature shifts against non-targeting rows and
zero-TPM genes without effect: planted genes are called, the null stays at
its rate. Complexes with planted correlation score above the random-pair and
shuffled-label controls. The reference loop of the threshold search is
checked against a literal transcription. Validation on the deposited
PERISCOPE A549 tables is recorded in the item's feature file.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from spacr import ops_scoring as S
from spacr.ops_engine import run_ops
from spacr.settings import ALPHA_FEATURES
from spacr.tabular import write_table


def _reference_threshold(scores, fdr_pct, start):
    """PERISCOPE's FDR_thresholds loop, transcribed."""
    total = len(scores)
    allowed = int(total / (100 / fdr_pct))
    c, count, threshold = start, 0, 0
    while count <= allowed:
        count = int((np.asarray(scores) > c).sum())
        threshold = c + 1
        c -= 1
    return threshold


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_threshold_search_is_the_reference_loop(seed):
    scores = np.random.default_rng(seed).poisson(30, 2000)
    for pct in (1, 2, 5):
        assert S.null_threshold(scores, pct / 100, 200) == _reference_threshold(scores, pct, 200)
    with pytest.raises(ValueError):
        S.null_threshold([], 0.05, 200)
    assert S.null_threshold([0, 0, 0], 0.05, 5) == 0


def _guides(seed=0, n_features=40):
    rng = np.random.default_rng(seed)
    features = ([f"Cells_Intensity_Mito_{i}" for i in range(10)] + [f"Nuclei_Texture_DAPI_{i}" for i in range(10)]
                + [f"Cytoplasm_AreaShape_{i}" for i in range(10)] + [f"Cells_Intensity_WGA_{i}" for i in range(10)])
    shifted = {g: rng.choice(n_features, g % 16, replace=False) for g in range(40)}
    rows = []
    for g in range(60):
        name = f"NULL{g}" if g < 40 else f"HIT{g}"
        for _ in range(12):
            values = rng.normal(0, 1, n_features)
            if name.startswith("NULL"):
                values[shifted[g]] += 3.0
            elif g < 50:
                values += 3.0
            else:
                values[:10] += 3.0
            rows.append([name] + list(values))
    for _ in range(200):
        rows.append(["nontargeting"] + list(rng.normal(0, 1, n_features)))
    return pd.DataFrame(rows, columns=["gene"] + features)


def test_planted_genes_are_hits_and_null_genes_are_not():
    u, p = S.mann_whitney_table(_guides())
    assert p.shape == (60, 40) and (u.index == p.index).all()
    zero = [f"NULL{g}" for g in range(40)]
    expressed = [f"HIT{g}" for g in range(40, 60)] + [f"NULL{g}" for g in range(5)]
    called = S.call_hits(p, zero_tpm=zero, expressed=expressed, feature_p=0.001, fdr_levels=[0.05])
    scores = called["scores"]
    whole = set(scores.index[scores["whole_cell_5"]])
    part = set(scores.index[scores["compartment_5"]])
    assert whole == {f"HIT{g}" for g in range(40, 50)}
    assert part == {f"HIT{g}" for g in range(50, 60)}
    assert not scores.loc[[f"NULL{g}" for g in range(5)], "hit_5"].any()
    assert called["summary"]["5"]["per_compartment"]["Mito"] == 10
    assert called["summary"]["null_genes"] == 40
    with pytest.raises(ValueError):
        S.mann_whitney_table(_guides().query("gene != 'nontargeting'"))


def test_counts_drop_the_tally_row_and_use_a_strict_threshold():
    p = pd.DataFrame({"a_Mito": [0.001, 0.0001, 5], "b_DAPI": [0.0, 0.5, 5], "c_mito_dapi": [0.0, 0.0, 5]},
                     index=["G1", "G2", "sig_gene_count"])
    counts = S.significant_feature_counts(p, feature_p=0.001)
    assert list(counts.index) == ["G1", "G2"]
    assert counts.loc["G1"].to_dict() == {"Mito": 1, "ConA": 0, "WGA": 0, "DAPI": 2, "Phalloidin": 0, "Sum": 2}
    assert counts.loc["G2", "Mito"] == 2
    custom = S.compartment_features(["x_ER", "y_er_z", "q"], {"ER": "er"})
    assert custom == {"ER": ["x_ER", "y_er_z"]}


def test_expression_null_from_a_table_and_from_the_summary():
    table = pd.DataFrame({"Gene": ["A", "B", "C", "D"], "A549": [0.0, 1.2, 0.0, 3.0]})
    zero, expressed = S.expression_null(["A", "B", "C", "nontargeting"], table, "A549")
    assert zero == ["A", "C"] and expressed == ["B"]
    zero, expressed = S.expression_null(["A", "B", "C"], {"zero_tpm": ["C"], "expressed_genes": ["A"]})
    assert zero == ["C"] and expressed == ["A"]
    with pytest.raises(ValueError):
        S.expression_null(["A"], table, "HeLa")


def test_reference_overlap_respects_the_cell_rule():
    out = S.overlap_with_reference(["A", "B", "C"], ["B", "C", "D"])
    assert (out["shared"], out["jaccard"], out["recall"]) == (2, 0.5, 2 / 3)
    out = S.overlap_with_reference(["A", "B", "C"], ["B", "C", "D"], cell_counts={"B": 30, "C": 5, "D": 25},
                                   min_cells=20)
    assert (out["hits"], out["reference"], out["shared"]) == (1, 2, 1) and out["restricted"]


def _profiles(seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(60)]
    X = rng.normal(0, 1, (60, 30))
    for start in (0, 10, 20):
        X[start:start + 6] += rng.normal(0, 3, 30)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)])
    frame.insert(0, "Metadata_Foci_Barcode_MatchedTo_GeneCode", genes)
    complexes = [(f"C{k}", [f"G{i}" for i in range(s, s + 6)] + ["NOTSCORED"]) for k, s in enumerate((0, 10, 20))]
    complexes.append(("tiny", ["G40", "X1", "X2", "X3"]))
    string = pd.DataFrame([(f"G{a}", f"G{b}", 950) for s in (0, 10, 20) for a in range(s, s + 6)
                           for b in range(s, s + 6) if a != b] + [("G50", "G51", 450)],
                          columns=["protein1", "protein2", "combined_score"])
    return frame, complexes, string


def test_planted_complexes_beat_both_controls():
    frame, complexes, _ = _profiles()
    rows = S.profile_frame(frame)
    out = S.corum_within_r(rows, complexes, coverage=0.333, seed=0)
    summary = out["summary"].set_index("control")
    assert summary.loc["observed", "median_r"] > 0.5
    assert summary.loc["random_pairs", "median_r"] < 0.2
    assert summary.loc["shuffled_labels", "median_r"] < 0.2
    assert summary.loc["observed", "n_pairs"] == 45
    assert set(out["complexes"]["complex"]) == {"C0", "C1", "C2"}
    reduced = S.pca_rows(rows, 0.9)
    assert reduced.shape[0] == 60 and reduced.shape[1] < 30
    assert S.pca_rows(rows, None) is rows


def test_string_auprc_finds_planted_links():
    frame, _, string = _profiles()
    rows = S.profile_frame(frame)
    auprc, bins = S.string_auprc(rows, string, [0.4, 0.9], seed=0)
    observed = auprc[auprc.control == "observed"].set_index("threshold")
    shuffled = auprc[auprc.control == "shuffled_labels"].set_index("threshold")
    assert observed.loc[0.9, "positives"] == 45 and observed.loc[0.4, "positives"] == 46
    assert observed.loc[0.9, "auprc"] > 0.8 > shuffled.loc[0.9, "auprc"]
    assert observed.loc[0.9, "baseline"] == pytest.approx(45 / (60 * 59 / 2))
    assert bins["links"].sum() == len(string) and list(bins["bin"])[0] == "-1 to -0.66"
    with pytest.raises(ValueError):
        S.string_auprc(pd.DataFrame(np.zeros((S.STRING_MAX_GENES + 1, 2))), string)


def _write_inputs(tmp_path):
    frame, complexes, string = _profiles()
    write_table(frame, str(tmp_path / "genes.csv"), canonicalise=False)
    corum = pd.DataFrame({"ComplexName": [c[0] for c in complexes],
                          "subunits(Gene name)": [";".join(c[1]) for c in complexes]})
    write_table(corum, str(tmp_path / "corum.tsv"), canonicalise=False, sep="\t")
    write_table(string, str(tmp_path / "string.csv"), canonicalise=False)
    guides = _guides()
    write_table(guides, str(tmp_path / "guides.csv"), canonicalise=False)
    expression = pd.DataFrame({"Gene": [f"NULL{g}" for g in range(40)] + [f"HIT{g}" for g in range(40, 60)],
                               "A549": [0.0] * 40 + [2.0] * 20})
    write_table(expression, str(tmp_path / "expr.csv"), canonicalise=False)
    write_table(pd.DataFrame({"Gene": [f"HIT{g}" for g in range(40, 55)]}), str(tmp_path / "published.csv"),
                canonicalise=False)
    write_table(pd.DataFrame({"gene": [f"HIT{g}" for g in range(40, 60)], "cells": [25] * 10 + [5] * 10}),
                str(tmp_path / "cells.csv"), canonicalise=False)
    return {"dst_root": str(tmp_path / "run"),
            "ops_scoring_metrics": ["hits_mw_ztpm", "corum_within_r", "string_auprc"],
            "ops_scoring_inputs": {"ref": {"genes": str(tmp_path / "genes.csv"), "guides": str(tmp_path / "guides.csv"),
                                           "cell_counts": str(tmp_path / "cells.csv")},
                                   "clusters_only": {"clusters": str(tmp_path / "x.csv")}},
            "ops_scoring_corum_file": str(tmp_path / "corum.tsv"),
            "ops_scoring_string_file": str(tmp_path / "string.csv"),
            "ops_scoring_string_thresholds": [0.9],
            "ops_hits_expression_table": str(tmp_path / "expr.csv"), "ops_hits_cell_line": "A549",
            "ops_hits_reference_list": str(tmp_path / "published.csv"), "ops_hits_fdr": [0.05]}


def test_score_phase_runs_the_benchmarks_alone(tmp_path):
    settings = _write_inputs(tmp_path)
    result = run_ops(settings, phases=["score"])
    folder = result["folder"]
    assert "summary_main.csv" not in result["outputs"] and "activity_map.csv" not in result["outputs"]
    summary = pd.read_csv(os.path.join(folder, "benchmark_summary.csv"))
    hits = summary[(summary.metric == "hits_mw_ztpm") & (summary.input == "ref")]
    assert hits.value.tolist() == [20]
    corum = summary[summary.metric == "corum_within_r"].set_index("control")
    assert corum.loc["observed", "value"] > corum.loc["random_pairs", "value"]
    scorable = pd.read_csv(os.path.join(folder, "benchmark_scorable.csv"))
    assert set(scorable.input) == {"clusters_only"} and (scorable.scorable == "not scorable").all()
    with open(os.path.join(folder, "hits_summary.json"), encoding="utf-8") as handle:
        record = json.load(handle)["ref"]
    assert record["source"] == "guides" and record["reference_overlap"]["5"]["restricted"]
    assert record["reference_overlap"]["5"]["recall"] == 1.0 and record["reference_overlap"]["5"]["hits"] == 10
    assert os.path.isfile(os.path.join(folder, "mann_whitney_p_ref.csv.gz"))
    with open(os.path.join(folder, "benchmark_setup.json"), encoding="utf-8") as handle:
        setup = json.load(handle)
    assert len(setup["references"]["corum"]["sha256"]) == 64 and "same functions" in setup["matching"]
    again = run_ops(settings, phases=["score"])
    assert pd.read_csv(os.path.join(again["folder"], "benchmark_summary.csv")).equals(summary)


def test_hits_gene_set_and_deposited_pvalues(tmp_path):
    settings = _write_inputs(tmp_path)
    _, p = S.mann_whitney_table(_guides())
    tally = pd.DataFrame([np.zeros(p.shape[1])], columns=p.columns, index=["sig_gene_count"])
    write_table(pd.concat([p, tally]).rename_axis("Gene").reset_index(), str(tmp_path / "p.csv"),
                canonicalise=False)
    expression = {"zero_tpm": [f"NULL{g}" for g in range(40)], "expressed_genes": [f"HIT{g}" for g in range(40, 60)]}
    with open(tmp_path / "expr.json", "w", encoding="utf-8") as handle:
        json.dump(expression, handle)
    frame, _, _ = _profiles()
    frame["Metadata_Foci_Barcode_MatchedTo_GeneCode"] = [f"HIT{40 + i % 20}" if i < 20 else f"NULL{i}"
                                                         for i in range(60)]
    write_table(frame, str(tmp_path / "genes2.csv"), canonicalise=False)
    settings.update({"ops_scoring_inputs": {"dep": {"pvalues": str(tmp_path / "p.csv"),
                                                    "genes": str(tmp_path / "genes2.csv")},
                                            "nohits": {"genes": str(tmp_path / "genes2.csv")}},
                     "ops_hits_expression_table": str(tmp_path / "expr.json"),
                     "ops_scoring_benchmark_genes": "hits", "ops_hits_reference_list": None,
                     "ops_scoring_metrics": ["hits_mw_ztpm", "corum_within_r"]})
    folder = run_ops(settings, phases=["score"])["folder"]
    summary = pd.read_csv(os.path.join(folder, "benchmark_summary.csv"))
    assert summary[(summary.metric == "corum_within_r")].genes.unique().tolist() == [20]
    scorable = pd.read_csv(os.path.join(folder, "benchmark_scorable.csv"))
    assert (scorable.input == "nohits").any() and (scorable.input == "nohits").sum() == 2
    settings["ops_scoring_benchmark_genes"] = ["HIT40", "HIT41", "HIT42"]
    settings["ops_scoring_metrics"] = ["corum_within_r"]
    folder = run_ops(settings, phases=["score"])["folder"]
    assert pd.read_csv(os.path.join(folder, "benchmark_summary.csv")).genes.unique().tolist() == [3]


def test_benchmarks_say_what_they_need(tmp_path):
    settings = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        run_ops({**settings, "ops_scoring_metrics": ["corum"]}, phases=["score"])
    with pytest.raises(ValueError, match="ops_hits_expression_table"):
        run_ops({**settings, "ops_hits_expression_table": None}, phases=["score"])
    with pytest.raises(ValueError, match="ops_scoring_corum_file"):
        run_ops({**settings, "ops_scoring_metrics": ["corum_within_r"], "ops_scoring_corum_file": None},
                phases=["score"])
    with pytest.raises(ValueError, match="ops_scoring_string_file"):
        run_ops({**settings, "ops_scoring_metrics": ["string_auprc"], "ops_scoring_string_file": None},
                phases=["score"])
    with pytest.raises(ValueError, match="gene column"):
        S.gene_column(pd.DataFrame({"x": [1]}))


def test_registered_as_alpha_923():
    keys = ALPHA_FEATURES[923]["settings"]
    assert "ops_scoring_metrics" in keys and "ops_hits_fdr" in keys
    from spacr.ops_settings import ops_defaults

    defaults = ops_defaults({})
    assert defaults["ops_scoring_metrics"] is None
    assert defaults["ops_hits_fdr"] == [0.05, 0.01] and defaults["ops_hits_feature_p"] == 0.001
