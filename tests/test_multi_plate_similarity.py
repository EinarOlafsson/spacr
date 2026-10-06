"""Cross-plate similarity compares only source-compatible feature spaces."""

import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import active_learning as al


@dataclass(frozen=True)
class _Spec:
    backbone: str = "example-encoder"

    def fingerprint(self):
        """Give the synthetic encoder a stable specification identity."""
        return self.backbone


def _plate(tmp_path, name, *, columns=("emb_0", "emb_1"), digest="a" * 64,
           spec=None, provenance=True):
    source = tmp_path / name
    database = source / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True)
    keys = ["shared.png", f"{name}-second.png", f"{name}-third.png"]
    objects = [f"{name}_r1_c1_f1_o{i}" for i in range(1, 4)]
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE png_list (png_path TEXT, prcfo TEXT, annotate INTEGER)")
        db.executemany("INSERT INTO png_list (png_path, prcfo) VALUES (?, ?)",
                       zip(keys, objects))
    result = SimpleNamespace(
        values=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], np.float32),
        columns=columns, spec=spec or _Spec())
    entry = (SimpleNamespace(sha256=digest, key="encoder/example",
                             source="local") if provenance else None)
    al._store_crop_embeddings(str(database), objects, result,
                              encoder_entry=entry)
    return str(database), keys


def test_identical_crop_paths_in_two_databases_keep_their_owners(tmp_path):
    first, first_keys = _plate(tmp_path, "plate1")
    second, second_keys = _plate(tmp_path, "plate2")
    index = al._multi_similarity_index([first, second], feature_kind="embeddings")
    hits = index.like(first, first_keys[0], k=5)
    assert len(index) == 6
    assert (first, first_keys[0]) not in set(zip(hits.db_path, hits.key))
    assert (second, second_keys[0]) in set(zip(hits.db_path, hits.key))
    assert hits.similarity.is_monotonic_decreasing
    assert set(index.columns) == {"emb_0", "emb_1"}
    with pytest.raises(KeyError, match="has no compatible features"):
        index.like(first, "not-a-crop.png")


def test_multi_plate_auto_backend_uses_available_faiss_without_losing_sources(
        tmp_path, monkeypatch):
    import sys
    from tests.test_similarity_search import _fake_faiss

    first, first_keys = _plate(tmp_path, "plate1")
    second, second_keys = _plate(tmp_path, "plate2")
    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss(2))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    accelerated = al._multi_similarity_index([first, second])
    assert accelerated.backend == "faiss-gpu"
    plain = al._multi_similarity_index([first, second], backend="numpy")
    assert plain.backend == "numpy"
    assert set(zip(accelerated.like(first, first_keys[0], 5).db_path,
                   accelerated.like(first, first_keys[0], 5).key)) == set(zip(
                       plain.like(first, first_keys[0], 5).db_path,
                       plain.like(first, first_keys[0], 5).key))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cpu_only = al._multi_similarity_index([first, second])
    assert cpu_only.backend == "faiss"
    assert (second, second_keys[0]) in cpu_only.keys


def test_a_requested_crop_kind_filters_both_plates_without_reassigning_owners(tmp_path):
    first, _ = _plate(tmp_path, "plate1")
    second, _ = _plate(tmp_path, "plate2")
    with sqlite3.connect(second) as db:
        db.execute("UPDATE png_list SET png_path='plate2-second-alternate.png' "
                   "WHERE png_path='plate2-third.png'")
    index = al._multi_similarity_index(
        [first, second], image_type="second", feature_kind="embeddings")
    assert len(index) == 3
    assert {path for path, _ in index.keys} == {first, second}
    with pytest.raises(ValueError, match="No compatible crops"):
        al._multi_similarity_index(
            [first, second], image_type="not-a-crop", feature_kind="embeddings")


def test_a_symlink_to_the_same_database_is_not_another_plate(tmp_path):
    first, _ = _plate(tmp_path, "plate1")
    alias = tmp_path / "same-plate.db"
    alias.symlink_to(first)
    with pytest.raises(ValueError, match="different plate databases"):
        al._multi_similarity_index([first, str(alias)])


def test_invalid_feature_choice_and_disappeared_plate_fail_before_search(tmp_path):
    first, _ = _plate(tmp_path, "plate1")
    missing = str(tmp_path / "gone" / "measurements.db")
    with pytest.raises(ValueError, match="feature kind"):
        al._multi_similarity_index([first, missing], feature_kind="unknown")
    with pytest.raises(ValueError, match="does not exist"):
        al._multi_similarity_index([first, missing])


@pytest.mark.parametrize("difference,message", [
    ("weights", "weights or provenance"),
    ("spec", "different models or settings"),
    ("columns", "feature columns differ"),
    ("legacy", "actual encoder provenance"),
    ("empty_digest", "mixed or missing encoder provenance"),
])
def test_incompatible_embedding_spaces_are_refused(tmp_path, difference, message):
    first, _ = _plate(tmp_path, "plate1")
    options = {}
    if difference == "weights":
        options["digest"] = "b" * 64
    if difference == "spec":
        options["spec"] = _Spec("different-encoder")
    if difference == "columns":
        options["columns"] = ("emb_0", "other_dimension")
    if difference == "legacy":
        options["provenance"] = False
    if difference == "empty_digest":
        options["digest"] = ""
    second, _ = _plate(tmp_path, "plate2", **options)
    with pytest.raises(ValueError, match=message):
        al._multi_similarity_index([first, second], feature_kind="embeddings")


@pytest.mark.parametrize("corruption,message", [
    ("missing_fingerprint", "lack a model fingerprint"),
    ("mixed_fingerprint", "mixed or missing model fingerprints"),
    ("mixed_provenance", "mixed or missing encoder provenance"),
    ("unverified_digest", "verified weights SHA-256"),
    ("unjoined_rows", "join crop rows"),
])
def test_corrupt_stored_plate_cannot_join_a_cross_plate_index(
        tmp_path, corruption, message):
    first, _ = _plate(tmp_path, "plate1")
    second, _ = _plate(tmp_path, "plate2")
    with sqlite3.connect(second) as db:
        if corruption == "missing_fingerprint":
            db.execute("ALTER TABLE crop_embedding DROP COLUMN _embedding_fingerprint")
        elif corruption == "mixed_fingerprint":
            db.execute("UPDATE crop_embedding SET _embedding_fingerprint='other' "
                       "WHERE prcfo='plate2_r1_c1_f1_o1'")
        elif corruption == "mixed_provenance":
            db.execute("UPDATE crop_embedding SET _embedding_encoder_key='other' "
                       "WHERE prcfo='plate2_r1_c1_f1_o1'")
        elif corruption == "unverified_digest":
            db.execute("UPDATE crop_embedding SET _embedding_weights_sha256='guess'")
        else:
            db.execute("UPDATE png_list SET prcfo='not-in-embedding-table'")
    with pytest.raises(ValueError, match=message):
        al._multi_similarity_index([first, second], feature_kind="embeddings")


def test_measurement_choice_does_not_silently_fall_back_to_embeddings(tmp_path):
    first, _ = _plate(tmp_path, "plate1")
    second, _ = _plate(tmp_path, "plate2", digest="b" * 64)
    with pytest.raises(ValueError, match="measurement"):
        al._multi_similarity_index([first, second], feature_kind="measurements")
    with pytest.raises(ValueError, match="provenance"):
        al._multi_similarity_index([first, second], feature_kind="auto")


def test_a_plate_without_saved_embeddings_cannot_join_an_embedding_search(tmp_path):
    from tests.test_cov_active_learning_rounds import _make_project

    first, _ = _plate(tmp_path, "plate1")
    measured = _make_project(tmp_path / "measured", per_well=3,
                             plate="plate2", labelled=False)
    with pytest.raises(ValueError, match="No stored embeddings"):
        al._multi_similarity_index([first, measured["db"]],
                                   feature_kind="embeddings")
    with pytest.raises(ValueError, match="Cannot compare stored embeddings"):
        al._multi_similarity_index([first, measured["db"]],
                                   feature_kind="auto")


@pytest.mark.parametrize("old_digest", ["a" * 64, ""])
def test_one_database_replaces_old_vectors_when_weight_bytes_change(
        tmp_path, old_digest):
    database, _ = _plate(tmp_path, "plate1", digest=old_digest)
    result = SimpleNamespace(
        values=np.asarray([[0.3, 0.7]], np.float32),
        columns=("emb_0", "emb_1"), spec=_Spec())
    entry = SimpleNamespace(sha256="b" * 64, key="encoder/example",
                            source="local")
    al._store_crop_embeddings(database, ["plate1_r1_c1_f1_o1"], result,
                              encoder_entry=entry)
    stored = al._stored_embedding_frame(database)
    assert stored.prcfo.tolist() == ["plate1_r1_c1_f1_o1"]
    assert stored._embedding_weights_sha256.tolist() == ["b" * 64]


def test_matching_actual_model_identity_preserves_previous_crop_rows(tmp_path):
    database, _ = _plate(tmp_path, "plate1")
    result = SimpleNamespace(
        values=np.asarray([[0.3, 0.7]], np.float32),
        columns=("emb_0", "emb_1"), spec=_Spec())
    entry = SimpleNamespace(sha256="a" * 64, key="encoder/example",
                            source="local")
    count = al._store_crop_embeddings(
        database, ["plate1_r1_c1_f1_o1"], result, encoder_entry=entry)
    stored = al._stored_embedding_frame(database)
    assert count == len(stored) == 3
    assert set(stored.prcfo) == {
        f"plate1_r1_c1_f1_o{number}" for number in range(1, 4)}


def test_explicit_measurements_can_compare_plates_with_different_embeddings(tmp_path):
    from tests.test_cov_active_learning_rounds import _make_project

    first = _make_project(tmp_path / "first", per_well=3,
                          plate="plate1", labelled=False)
    second = _make_project(tmp_path / "second", per_well=3,
                           plate="plate2", labelled=False)
    for project, digest in ((first, "a" * 64), (second, "b" * 64)):
        count = len(project["crops"])
        result = SimpleNamespace(
            values=np.arange(count * 2, dtype=np.float32).reshape(count, 2),
            columns=("emb_0", "emb_1"), spec=_Spec())
        entry = SimpleNamespace(sha256=digest, key="encoder/example",
                                source="local")
        al._store_crop_embeddings(
            project["db"], [row["prcfo"] for row in project["crops"]],
            result, encoder_entry=entry)
    with pytest.raises(ValueError, match="weights or provenance"):
        al._multi_similarity_index([first["db"], second["db"]])
    index = al._multi_similarity_index(
        [first["db"], second["db"]], feature_kind="measurements")
    hits = index.like(first["db"], first["crops"][0]["png_path"], k=5)
    assert len(index) == len(first["crops"]) + len(second["crops"])
    assert {first["db"], second["db"]} == {path for path, _ in index.keys}
    assert len(hits) == 5


def test_verified_local_cell_dino_identity_enables_only_compatible_plate_search(
        tmp_path):
    from spacr import embeddings as emb
    import hashlib

    checkpoint = tmp_path / "local-cell-dino.pth"
    content = b"Synthetic local checkpoint identity, not pretrained model weights"
    checkpoint.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), cell_dino_factory="cell_dino_hpa_vitl16",
        checkpoint_path=str(checkpoint), checkpoint_sha256=digest)
    first, first_keys = _plate(tmp_path, "plate1", spec=spec)
    second, second_keys = _plate(tmp_path, "plate2", spec=spec)
    result = emb.EmbeddingResult(
        np.asarray([[1, 0], [0.8, 0.2], [0, 1]], np.float32),
        ("emb_0", "emb_1"), spec, 4, 2)
    entry = emb.encoder_entry(spec)
    assert entry.sha256 == digest and entry.source == "local"
    for name, database in (("plate1", first), ("plate2", second)):
        objects = [f"{name}_r1_c1_f1_o{i}" for i in range(1, 4)]
        al._store_crop_embeddings(database, objects, result, encoder_entry=entry)
    index = al._multi_similarity_index(
        [first, second], feature_kind="embeddings", backend="numpy")
    hits = index.like(first, first_keys[0], k=5)
    assert (second, second_keys[0]) in set(zip(hits.db_path, hits.key))
    checkpoint.write_bytes(b"A different local checkpoint")
    mismatched = emb.encoder_entry(spec)
    assert mismatched.sha256 == ""
    al._store_crop_embeddings(
        second, [f"plate2_r1_c1_f1_o{i}" for i in range(1, 4)],
        result, encoder_entry=mismatched)
    with pytest.raises(ValueError, match="missing encoder provenance"):
        al._multi_similarity_index(
            [first, second], feature_kind="embeddings", backend="numpy")
