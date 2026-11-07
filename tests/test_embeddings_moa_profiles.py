"""Embeddings aggregate into treatment profiles and score by NSC and NSCB 1-NN.

Synthetic embeddings with a known mechanism-of-action structure: each MoA
has a centre, each compound sits near its MoA's centre at two
concentrations, and wells and cells add noise. With a strong structure the
not-same-compound (NSC) accuracy is high; with random profiles it falls to
the shuffled-label control. A batch shift that pulls compounds of
different MoAs together shows that not-same-compound-or-batch (NSCB) never
picks a neighbour from the same batch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.embeddings as emb


def _cells(*, n_moa=4, compounds_per_moa=3, dims=16, strength=6.0,
           wells=2, cells=5, plates=2, seed=0):
    """Per-cell embeddings for a plate map with DMSO wells on every plate."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_moa, dims)) * strength
    rows = []
    for plate in range(plates):
        plate_shift = rng.normal(size=dims) * 0.5
        for column in range(wells):
            for _ in range(cells):
                rows.append(dict(plateID=f"p{plate}", rowID="A", columnID=f"c{column}",
                                 compound="DMSO", concentration=0.0, moa="DMSO",
                                 batch=f"b{plate}",
                                 vector=plate_shift + rng.normal(size=dims)))
    for m in range(n_moa):
        for c in range(compounds_per_moa):
            compound = f"m{m}c{c}"
            offset = centres[m] + rng.normal(size=dims)
            for dose_index, dose in enumerate((0.1, 1.0)):
                for w in range(wells):
                    plate = (m + c + dose_index) % plates
                    shift = rng.normal(size=dims) * 0.5
                    for _ in range(cells):
                        rows.append(dict(
                            plateID=f"p{plate}", rowID=f"r{m}{c}{dose_index}",
                            columnID=f"c{w}", compound=compound,
                            concentration=dose, moa=f"moa{m}", batch=f"b{plate}",
                            vector=offset + shift + rng.normal(size=dims)))
    frame = pd.DataFrame(rows)
    vectors = np.vstack(frame.pop("vector").to_numpy())
    columns = [f"{emb.EMBEDDING_PREFIX}c0_{index}" for index in range(dims)]
    return pd.concat([frame, pd.DataFrame(vectors, columns=columns)], axis=1)


WELLS = ["plateID", "rowID", "columnID"]


def test_cells_aggregate_to_wells_then_treatments_with_the_right_shapes():
    cells = _cells()
    out = emb.aggregate_profiles(cells, well_columns=WELLS, method="median",
                                 batch_column="batch")
    assert len(out) == 4 * 3 * 2
    features = [c for c in out.columns if c.startswith(emb.EMBEDDING_PREFIX)]
    assert len(features) == 16
    assert list(out.columns[:5]) == ["compound", "concentration", "batch",
                                     "n_wells", "n_cells"]
    assert (out["n_wells"] == 2).all() and (out["n_cells"] == 10).all()
    assert "DMSO" not in set(out["compound"])
    mean = emb.aggregate_profiles(cells, by=["compound"], well_columns=WELLS,
                                  method="mean", drop_controls=False)
    assert len(mean) == 4 * 3 + 1
    one = cells[cells["compound"] == "m0c0"]
    expected = one.groupby(WELLS)[features].mean().mean()
    np.testing.assert_allclose(
        mean.loc[mean["compound"] == "m0c0", features].to_numpy()[0],
        expected.to_numpy())


def test_well_tables_and_table_paths_aggregate_and_batches_join(tmp_path):
    from spacr import tabular

    wells = _cells().groupby(WELLS + ["compound", "concentration", "moa", "batch"],
                             sort=True).mean(numeric_only=True).reset_index()
    path = tabular.write_table(wells, str(tmp_path / "wells.parquet"))
    out = emb.aggregate_profiles(path, by=["compound"], batch_column="batch")
    assert len(out) == 12 and (out["n_wells"] == 4).all()
    assert all(set(value.split("|")) <= {"b0", "b1"} for value in out["batch"])
    assert (out["batch"] == "b0|b1").any()


@pytest.mark.parametrize("how", ["center", "zscore", "robust_zscore", "tvn"])
def test_control_normalisation_centres_the_controls(how):
    cells = _cells(dims=6)
    out = emb.aggregate_profiles(cells, well_columns=WELLS, normalise=how,
                                 plate_column="plateID", method="mean",
                                 drop_controls=False, by=["compound"])
    features = [c for c in out.columns if c.startswith(emb.EMBEDDING_PREFIX)]
    assert len(features) == 6
    dmso = out.loc[out["compound"] == "DMSO", features].to_numpy()[0]
    np.testing.assert_allclose(dmso, 0.0, atol=1e-8)


def test_normalisation_needs_controls_and_names_unknown_options():
    cells = _cells(dims=4)
    with pytest.raises(emb.EmbeddingError, match="control"):
        emb.aggregate_profiles(cells, well_columns=WELLS, normalise="zscore",
                               control="water")
    with pytest.raises(emb.EmbeddingError, match="normalise"):
        emb.aggregate_profiles(cells, normalise="quantile")
    with pytest.raises(emb.EmbeddingError, match="method"):
        emb.aggregate_profiles(cells, method="mode")
    with pytest.raises(emb.EmbeddingError, match="columns not in the table"):
        emb.aggregate_profiles(cells, by=["drug"])


def test_strong_structure_scores_high_and_the_control_rows_are_present():
    profiles = emb.aggregate_profiles(_cells(), well_columns=WELLS,
                                      batch_column="batch")
    labels = _cells()[["compound", "moa"]].drop_duplicates()
    scored = emb.evaluate_moa(profiles, labels=labels, batch_column="batch",
                              n_permutations=50)
    table = scored["table"].set_index(["profiles", "evaluation"])
    assert table.loc[("profiles", "NSC"), "accuracy"] == 1.0
    assert table.loc[("profiles", "NSC"), "n_treatments"] == 24
    low, high = table.loc[("profiles", "NSC"), ["ci_low", "ci_high"]]
    assert 0.8 < low < 1.0 and high == 1.0
    controls = [name for name in scored["table"]["profiles"]
                if name.startswith("control:")]
    assert "control: shuffled MoA labels" in controls
    assert any(name.startswith("control: majority class (") for name in controls)
    assert set(scored["table"]["evaluation"]) == {"NSC", "NSCB"}
    assert table.loc[("control: shuffled MoA labels", "NSC"), "accuracy"] < 0.5
    majority = scored["table"][scored["table"]["profiles"].str.startswith(
        "control: majority")]
    assert (majority["accuracy"] == 0.25).all()
    assert {"MoA", "NSC", "NSCB", "1-NN", "CI"} <= scored["glossary"].keys()
    assert scored["table"].attrs["glossary"] == scored["glossary"]
    predictions = scored["predictions"]
    assert len(predictions) == 24 and predictions["NSC_correct"].all()
    assert (predictions["NSC_neighbour_compound"] != predictions["compound"]).all()


def test_random_profiles_score_at_the_shuffled_label_control():
    rng = np.random.default_rng(3)
    n_moa, per = 6, 10
    profiles = pd.DataFrame({
        "compound": [f"k{i}" for i in range(n_moa * per)],
        "concentration": 1.0,
        "moa": [f"moa{i % n_moa}" for i in range(n_moa * per)],
    })
    accuracies = []
    for repeat in range(20):
        vectors = rng.normal(size=(len(profiles), 32))
        frame = pd.concat([profiles, pd.DataFrame(
            vectors, columns=[f"{emb.EMBEDDING_PREFIX}{i}" for i in range(32)])],
            axis=1)
        table = emb.evaluate_moa(frame, n_permutations=20, seed=repeat)["table"]
        accuracies.append(float(table.loc[table["profiles"] == "profiles",
                                          "accuracy"].iloc[0]))
        shuffled = table.loc[table["profiles"] == "control: shuffled MoA labels"]
        assert len(shuffled) == 1 and shuffled["sd"].iloc[0] > 0
    chance = (per - 1) / (n_moa * per - 1)
    assert abs(np.mean(accuracies) - chance) < 0.06


def test_nscb_never_picks_a_neighbour_from_the_same_batch():
    rng = np.random.default_rng(1)
    rows, vectors = [], []
    batch_shift = {"w1": rng.normal(size=8) * 20, "w2": rng.normal(size=8) * 20}
    centres = {m: rng.normal(size=8) for m in ("a", "b")}
    for batch in ("w1", "w2"):
        for moa in ("a", "b"):
            for k in range(2):
                rows.append(dict(compound=f"{batch}{moa}{k}", concentration=1.0,
                                 moa=moa, batch=batch))
                vectors.append(batch_shift[batch] + centres[moa]
                               + rng.normal(size=8) * 0.1)
    frame = pd.concat([pd.DataFrame(rows), pd.DataFrame(
        np.array(vectors), columns=[f"{emb.EMBEDDING_PREFIX}{i}" for i in range(8)])],
        axis=1)
    scored = emb.evaluate_moa(frame, batch_column="batch", n_permutations=10)
    predictions = scored["predictions"]
    nsc = predictions["NSC_neighbour"].to_numpy()
    nscb = predictions["NSCB_neighbour"].to_numpy()
    batches = predictions["batch"].to_numpy()
    assert (batches[nsc] == batches).all()
    assert (batches[nscb] != batches).all()
    table = scored["table"].set_index(["profiles", "evaluation"])
    assert table.loc[("profiles", "NSC"), "accuracy"] == 1.0
    assert table.loc[("profiles", "NSCB"), "n_solvable"] == 8


def test_nscb_reads_joined_batches_and_skips_treatments_with_no_candidate():
    frame = pd.DataFrame({
        "compound": ["x", "y", "z"], "moa": ["a", "a", "b"],
        "batch": ["w1|w2", "w2", "w3"],
        f"{emb.EMBEDDING_PREFIX}0": [1.0, 1.0, 0.0],
        f"{emb.EMBEDDING_PREFIX}1": [0.0, 0.1, 1.0],
    })
    scored = emb.evaluate_moa(frame, batch_column="batch", n_permutations=0)
    predictions = scored["predictions"]
    assert list(predictions["NSCB_neighbour"]) == [2, 2, 1]
    assert list(predictions["NSC_neighbour"]) == [1, 0, 1]
    table = scored["table"]
    assert "control: shuffled MoA labels" not in set(table["profiles"])
    assert table["profiles"].str.startswith("control: majority class").sum() == 2


def test_reference_profiles_are_scored_beside_the_embeddings():
    profiles = emb.aggregate_profiles(_cells(), well_columns=WELLS)
    labels = _cells()[["compound", "moa"]].drop_duplicates()
    rng = np.random.default_rng(0)
    reference = profiles[["compound", "concentration"]].copy()
    reference["area"] = rng.random(len(reference))
    reference["intensity"] = rng.random(len(reference))
    scored = emb.evaluate_moa(profiles, labels=labels,
                              references={"CellProfiler": reference},
                              n_permutations=5)
    names = list(scored["table"]["profiles"])
    assert names[:2] == ["profiles", "reference: CellProfiler"]
    assert "NSCB" not in scored["glossary"]


def test_evaluation_rejects_unusable_profiles():
    frame = pd.DataFrame({"compound": ["x", "y"], "moa": ["a", "b"],
                          f"{emb.EMBEDDING_PREFIX}0": [0.0, 1.0]})
    with pytest.raises(emb.EmbeddingError, match="zero length"):
        emb.evaluate_moa(frame)
    with pytest.raises(emb.EmbeddingError, match="fewer than two"):
        emb.evaluate_moa(frame.iloc[:1])
    with pytest.raises(emb.EmbeddingError, match="columns not in the profiles"):
        emb.evaluate_moa(frame, batch_column="week")
    with pytest.raises(emb.EmbeddingError, match="confidence"):
        emb.evaluate_moa(frame, confidence=95)
