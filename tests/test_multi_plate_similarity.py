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


def test_a_symlink_to_the_same_database_is_not_another_plate(tmp_path):
    first, _ = _plate(tmp_path, "plate1")
    alias = tmp_path / "same-plate.db"
    alias.symlink_to(first)
    with pytest.raises(ValueError, match="different plate databases"):
        al._multi_similarity_index([first, str(alias)])


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


def test_measurement_choice_does_not_silently_fall_back_to_embeddings(tmp_path):
    first, _ = _plate(tmp_path, "plate1")
    second, _ = _plate(tmp_path, "plate2", digest="b" * 64)
    with pytest.raises(ValueError, match="measurement"):
        al._multi_similarity_index([first, second], feature_kind="measurements")
    with pytest.raises(ValueError, match="provenance"):
        al._multi_similarity_index([first, second], feature_kind="auto")


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
