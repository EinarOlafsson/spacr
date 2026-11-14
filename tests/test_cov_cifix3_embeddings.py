"""Embedding profile, parallel and input checks refuse bad input with a named error.

Each test holds one refusal or fallback that the embedding profile table,
the MoA scorer, the parallel embedder, the multiple-instance bag builder and
the Cell-DINO checkpoint reader promise, so a bad table or setting fails
with a clear message rather than a stack trace.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.embeddings as emb


def _profiles(n=6):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "compound": [f"c{i}" for i in range(n)],
        "moa": ["a", "b"] * (n // 2),
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
    })


def test_input_mean_that_is_not_numbers_is_refused():
    with pytest.raises(emb.EmbeddingError, match="input_mean must be three numbers"):
        emb.EmbeddingSpec(input_mean=("a", "b", "c"))


def test_a_profile_table_must_be_a_frame_or_a_path():
    with pytest.raises(emb.EmbeddingError, match="expected a frame or a table path"):
        emb._profile_table(42)


def test_a_table_path_is_read_into_a_frame(tmp_path):
    path = tmp_path / "t.csv"
    _profiles().to_csv(path, index=False)
    frame = emb._profile_table(str(path))
    assert list(frame.columns) == ["compound", "moa", "x", "y"]
    assert len(frame) == 6


def test_explicit_features_must_exist_and_numeric_ones_must_be_found():
    frame = _profiles()
    assert emb._profile_features(frame, ["x"], []) == ["x"]
    with pytest.raises(emb.EmbeddingError, match="feature columns not in the table"):
        emb._profile_features(frame, ["nope"], [])
    text = frame[["compound", "moa"]]
    with pytest.raises(emb.EmbeddingError, match="no numeric feature columns"):
        emb._profile_features(text, None, ["compound"])


def test_aggregate_profiles_refuses_empty_by_missing_control_and_no_treatments():
    frame = _profiles()
    with pytest.raises(emb.EmbeddingError, match="at least one treatment column"):
        emb.aggregate_profiles(frame, by=[])
    with pytest.raises(emb.EmbeddingError, match="control column 'nope'"):
        emb.aggregate_profiles(frame, by=["compound"], control_column="nope")
    with pytest.raises(emb.EmbeddingError, match="no treatment rows remain"):
        emb.aggregate_profiles(frame, by=["moa"], control=["a", "b"],
                               control_column="moa", drop_controls=True)


def test_the_wilson_interval_of_nothing_is_not_a_number_and_blank_batches_are_empty():
    low, high = emb._wilson(0, 0, 1.96)
    assert np.isnan(low) and np.isnan(high)
    sets = emb._batch_sets([float("nan"), None, "", "b1|b2"])
    assert sets[:3] == [frozenset()] * 3
    assert sets[3] == frozenset({"b1", "b2"})


def test_evaluate_moa_refuses_labels_without_a_shared_column_and_infinite_values():
    frame = _profiles().drop(columns="moa")
    labels = pd.DataFrame({"other": ["z"], "moa": ["a"]})
    with pytest.raises(emb.EmbeddingError, match="share no column"):
        emb.evaluate_moa(frame, labels=labels)
    bad = _profiles()
    bad.loc[0, "x"] = np.inf
    with pytest.raises(emb.EmbeddingError, match="missing or infinite"):
        emb.evaluate_moa(bad)


def test_available_cores_falls_back_to_cpu_count(monkeypatch):
    def broken(_pid):
        raise OSError("no affinity")

    monkeypatch.setattr(emb.os, "sched_getaffinity", broken, raising=False)
    monkeypatch.setattr(emb.os, "cpu_count", lambda: 3)
    assert emb._available_cores() == 3


def test_a_missing_cell_dino_checkpoint_is_reported_missing(tmp_path):
    entry = emb._cell_dino_entry(str(tmp_path / "gone.pt"))
    assert entry["problem"] == "missing"
    assert emb._cell_dino_entry(str(tmp_path))["problem"] == "unreadable"


def test_dino_encoder_refuses_a_missing_checkpoint(tmp_path):
    spec = emb.EmbeddingSpec(backbone=emb._DINO_PREFIX + str(tmp_path / "none.pt"))
    with pytest.raises(emb.EmbeddingError, match="no DINO checkpoint"):
        emb._dino_encoder(spec)


def _bag_frame():
    return pd.DataFrame({
        "well": list("aabbccdd"),
        "label": [0, 0, 0, 0, 1, 1, 1, 1],
        "f1": np.arange(8, dtype=float),
        "f2": np.arange(8, dtype=float) * 2,
    })


def test_mil_bags_default_to_numeric_columns_and_refuse_bad_labels():
    frame = _bag_frame()
    bags, labels, wells, rows, kept = emb._mil_bags(frame, well_column="well",
                                                    label_column="label")
    assert wells == list("abcd") and labels.tolist() == [0, 0, 1, 1]
    assert bags[0].shape == (2, 2)
    with pytest.raises(ValueError, match="no numeric feature columns"):
        emb._mil_bags(frame[["well", "label"]], well_column="well",
                      label_column="label")
    with pytest.raises(ValueError, match="no numeric feature columns"):
        emb._mil_bags(frame, well_column="well", label_column="label",
                      feature_columns=[])
    three = frame.copy()
    three["label"] = [0, 0, 1, 1, 2, 2, 0, 0]
    with pytest.raises(ValueError, match="two classes"):
        emb._mil_bags(three, well_column="well", label_column="label", positive=1)
    few = frame.copy()
    few["label"] = [0, 0, 0, 0, 0, 0, 1, 1]
    with pytest.raises(ValueError, match="at least two wells"):
        emb._mil_bags(few, well_column="well", label_column="label")


def test_a_store_crop_array_must_have_four_dimensions():
    with pytest.raises(emb.EmbeddingError, match="non-empty"):
        emb._store_crops(np.zeros((2, 4, 4), dtype=np.float32), None, "cell", "", "")


def test_moa_scores_shuffle_labels_freely_when_a_compound_has_two_moas():
    frame = pd.DataFrame({
        "compound": ["c0", "c0", "c1", "c1", "c2", "c2"],
        "concentration": [1, 2, 1, 2, 1, 2],
        "moa": ["a", "b", "a", "b", "a", "b"],
        "x": [0.0, 1.0, 0.1, 1.1, 0.2, 1.2],
        "y": [1.0, 0.0, 1.1, 0.1, 1.2, 0.2],
    })
    out = emb.evaluate_moa(frame, n_permutations=5, seed=1)
    assert isinstance(out, dict) and out


def test_a_worker_loads_its_backbone_once_and_reuses_it(monkeypatch):
    spec = emb.EmbeddingSpec(normalize=False)
    monkeypatch.setattr(emb, "_WORKER_ENCODERS", {})
    built = []

    def make(s):
        built.append(1)

        def encoder(array):
            return array.reshape(len(array), -1)[:, :2]

        encoder.in_channels = 3
        return encoder

    monkeypatch.setattr(emb, "_backbone_encoder", make)
    crops = np.random.default_rng(0).random((2, 4, 4, 3)).astype(np.float32)
    values, columns, width = emb._embedding_worker_chunk(crops, spec, None)
    emb._embedding_worker_chunk(crops, spec, None)
    assert values.shape[0] == 2 and built == [1]


def test_a_timm_model_with_a_two_channel_mean_is_refused():
    from types import SimpleNamespace

    timm = SimpleNamespace(data=SimpleNamespace(
        resolve_model_data_config=lambda model: {"mean": (0.5, 0.5), "std": (0.2, 0.2)}))
    spec = emb.EmbeddingSpec(backbone="resnet18")
    with pytest.raises(emb.EmbeddingError, match="2-channel mean"):
        emb._timm_input(spec, timm, object(), None)


def test_a_transformer_dino_network_asks_timm_for_a_dynamic_image_size():
    import sys
    from types import SimpleNamespace

    asked = {}

    def create_model(arch, **kwargs):
        asked.update(kwargs)
        raise RuntimeError("stop")

    fake = SimpleNamespace(create_model=create_model)
    saved = sys.modules.get("timm")
    sys.modules["timm"] = fake
    try:
        with pytest.raises(RuntimeError, match="stop"):
            emb._dino_network("vit_tiny_patch16_224", 3, 32, False, 8,
                              SimpleNamespace(nn=SimpleNamespace()))
    finally:
        if saved is None:
            sys.modules.pop("timm")
        else:
            sys.modules["timm"] = saved
    assert asked["img_size"] == 32 and asked["dynamic_img_size"] is True


def test_stored_embedding_reads_pass_other_errors_on_and_need_a_prcfo_column(monkeypatch, tmp_path):
    from spacr import tabular

    db = tmp_path / "m.db"
    db.write_bytes(b"")

    def other(*a, **k):
        raise ValueError("disk on fire")

    monkeypatch.setattr(tabular, "read_database", other)
    with pytest.raises(ValueError, match="disk on fire"):
        emb._mil_frame_from_db(str(db))
    stored = pd.DataFrame({"prcfo": ["p_A_1_1_1"], "emb_0": [0.1]})
    monkeypatch.setattr(tabular, "read_table",
                        lambda *a, **k: pd.DataFrame({"png_path": ["x.png"]}))
    with pytest.raises(ValueError, match="no prcfo column"):
        emb._mil_frame_from_db(str(db), stored_embeddings=stored)
