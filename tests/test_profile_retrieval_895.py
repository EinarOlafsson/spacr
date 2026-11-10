"""Item 895: replicate retrieval, compound-gene matching and plate effect of well profiles (CPJUMP1 style)."""
import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as S


def _wells(seed=0, features=12):
    """Three modalities on two plates each: compounds hit genes that CRISPR and ORF also profile."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(6)]
    signature = {g: rng.standard_normal(features) * 3 for g in genes}
    plate_shift = {p: rng.standard_normal(features) * 0.5 for p in ("P1", "P2", "P3", "P4", "P5", "P6")}
    rows = []
    layout = {"compound": ("P1", "P2"), "crispr": ("P3", "P4"), "orf": ("P5", "P6")}
    for modality, plates in layout.items():
        for plate in plates:
            well = 0
            for g in genes:
                for _ in range(2):
                    sample = f"{modality[:2]}_{g}"
                    sign = -1 if modality == "orf" else 1
                    rows.append(dict(plateID=plate, wellID=f"w{well}", modality=modality, broad_sample=sample,
                                     control_type="trt" if g != "G5" else "poscon", gene=g,
                                     target_list=f"{g}|G9" if modality == "compound" else np.nan,
                                     **dict(zip([f"f{j}" for j in range(features)],
                                                sign * signature[g] + plate_shift[plate]
                                                + rng.standard_normal(features)))))
                    well += 1
            for k in range(6):
                rows.append(dict(plateID=plate, wellID=f"w{well}", modality=modality, broad_sample="DMSO",
                                 control_type="negcon", gene=np.nan, target_list=np.nan,
                                 **dict(zip([f"f{j}" for j in range(features)],
                                            plate_shift[plate] + rng.standard_normal(features)))))
                well += 1
            rows.append(dict(plateID=plate, wellID=f"w{well}", modality=modality, broad_sample="EMPTY",
                             control_type="empty", gene=np.nan, target_list=np.nan,
                             **{f"f{j}": 0.0 for j in range(features)}))
    return pd.DataFrame(rows), [f"f{j}" for j in range(features)]


SETTINGS = {"retrieval_perturbation_key": "broad_sample", "retrieval_type_key": "control_type",
            "retrieval_null_size": 500, "retrieval_feature_select": False}


def test_the_whole_benchmark_from_settings(tmp_path):
    wells, features = _wells()
    result = S.profile_retrieval(wells, features, SETTINGS)
    summary = result["summary"]
    assert summary.modality.tolist() == ["compound", "compound", "crispr", "crispr", "orf", "orf"]
    real = summary[summary.label == "real"]
    assert (real.mAP_mean > 0.8).all() and (real.n_perturbations == 6).all() and (real.n_trt == 5).all()
    assert (summary[summary.label != "real"].mAP_mean < real.mAP_mean.min()).all()
    assert set(result["activity"].columns) >= {"broad_sample", "mean_average_precision", "control_type", "modality"}
    matching = result["matching"]
    assert len(matching) == 16 and set(matching.subset) == {"all treatments", "replicable only"}
    crispr = matching[(matching.gene_modality == "crispr") & (matching.similarity == "cosine")
                      & (matching.label == "real") & (matching.subset == "all treatments")].iloc[0]
    assert crispr.mAP_mean == 1.0 and crispr.n_compounds == 5 and crispr.n_genes == 5
    orf_abs = matching[(matching.gene_modality == "orf") & (matching.similarity == "|cosine|")
                       & (matching.label == "real")].iloc[0]
    assert orf_abs.mAP_mean == 1.0
    effect = result["plate_effect"]
    assert effect.stage.tolist() == ["normalised"] * 6 + ["raw"] * 6
    assert (effect.n_negcon_wells == 12).all() and (effect.n_plates == 2).all()
    raw_real = effect[(effect.stage == "raw") & (effect.label == "real")]
    assert (raw_real.plate_R2_top10PC_weighted > 0.1).all()
    written = S.write_profile_retrieval(result, str(tmp_path / "out"))
    assert pd.read_csv(written["matching.csv"]).shape == matching.shape


def test_options_selection_and_refusals():
    wells, features = _wells(seed=1)
    lean = S.profile_retrieval(wells, features, {**SETTINGS, "retrieval_feature_select": True,
                                                 "retrieval_plate_effect": False, "retrieval_matching": None,
                                                 "retrieval_modalities": ["crispr"]})
    assert lean["plate_effect"].empty and lean["matching"].empty and lean["matching_per_compound"].empty
    assert lean["summary"].n_features.iloc[0] <= len(features)
    raw = S.profile_retrieval(wells, features, {**SETTINGS, "retrieval_normalisation": "none",
                                                "retrieval_plate_effect": False, "retrieval_modalities": ["orf"]})
    assert raw["summary"].n_features.iloc[0] == len(features)
    with pytest.raises(ValueError, match="retrieval_matching"):
        S.profile_retrieval(wells, features, {**SETTINGS, "retrieval_matching": "gene_gene"})


def test_matching_with_target_lists_and_without_positives():
    compounds = pd.DataFrame(np.eye(3, 4), index=["c1", "c2", "c3"], columns=list("abcd"))
    genes = pd.DataFrame(np.eye(3, 4), index=["A", "B", "C"], columns=list("abcd"))
    gene = pd.Series({"c1": "A", "c2": "Q", "c3": np.nan})
    targets = pd.Series({"c1": "A", "c2": "B|Q", "c3": "C"})
    summary, table = S.target_matching(compounds, gene, targets, genes, use_target_list=True, null_size=50)
    assert summary["n_compounds"] == 3 and table.broad_sample.tolist() == ["c1", "c2", "c3"]
    summary, _ = S.target_matching(compounds, gene, targets, genes, null_size=50)
    assert summary["n_compounds"] == 1 and summary["mAP_mean"] == 1.0
    shuffled, _ = S.target_matching(compounds, gene, targets, genes, null_size=50, rng=np.random.default_rng(0),
                                    label="control")
    assert shuffled["label"] == "control"


def test_retrieval_without_well_ids_and_shuffle_keeps_controls():
    wells, features = _wells(seed=2)
    one = wells[(wells.modality == "crispr") & (wells.control_type != "empty")].reset_index(drop=True)
    summary, _ = S.replicate_retrieval(one.drop(columns="wellID"), features, null_size=100)
    assert summary["n_perturbations"] == 6
    shuffled = S.shuffle_within_plate(one, "broad_sample", np.random.default_rng(0))
    assert (shuffled[one.control_type == "negcon"].broad_sample == "DMSO").all()
    assert sorted(shuffled.broad_sample) == sorted(one.broad_sample)
