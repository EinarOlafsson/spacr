"""UMAP and Activation sweeps of ``spacr.hyperparam``: scoring and persistence.

One table-driven test per unit, each row run as a subtest: the embedding
criteria (``embedding_stability``, ``_cluster_structure``,
``_objective_weights``, ``umap_objective_scores``), the UMAP sweep itself
(``umap_search``: argument checks, then grid / Walk / multi-objective /
clustering / cuML / resume behaviour with a deterministic stand-in
embedder), its checkpoint (``umap_checkpoint_path``, ``_save_array_atomic``,
``_UmapCheckpoint``), the attribution sweep's fit function
(``activation_fit_fn``) on a hand-wired 16 px CPU model, and the sweep
panel's one-gallery-per-panel rule.  No real UMAP fit and no GPU is used.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from spacr import hyperparam as hp
from spacr.hyperparam import SearchSpace, Trial


def _blobs(n_per=12, seed=0):
    """Three well-separated clusters in five dimensions, and their labels."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0, 0, 0, 0], [10.0, 0, 0, 0, 0],
                        [0.0, 10, 0, 0, 0]])
    features = np.vstack([c + rng.normal(scale=0.3, size=(n_per, 5))
                          for c in centres])
    return features, np.repeat([0, 1, 2], n_per)


# ===========================================================================
# embedding_stability
# ===========================================================================

def test_embedding_stability_is_the_shared_neighbour_fraction(subtests):
    """Rotation, scaling and duplicate coordinates keep stability at 1;
    unrelated embeddings share few neighbours (exact fractions); k is capped
    at n-1; fewer than two repeats, short, ragged and non-finite embeddings
    are refused by message."""
    features, _labels = _blobs()
    embedding = features[:, :2]
    angle = np.pi / 3
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    rng0, rng1 = np.random.default_rng(0), np.random.default_rng(1)
    duplicates = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                           [1.0, 1.0]])
    for cid, embeddings, k, expected in [
        ("rotated-and-scaled", [embedding, embedding @ rotation * 5.0], 5, 1.0),
        ("three-identical-repeats", [embedding] * 3, 5, 1.0),
        ("duplicate-coordinates", [duplicates, duplicates.copy()], 2, 1.0),
        ("unrelated-seed-0", [rng0.normal(size=(40, 2)),
                              rng0.normal(size=(40, 2))], 5, 0.14),
        ("unrelated-seed-1", [rng1.normal(size=(40, 2)),
                              rng1.normal(size=(40, 2))], 5, 0.13),
        ("k-capped-at-n-minus-one", [np.arange(10.0).reshape(5, 2),
                                     -np.arange(10.0).reshape(5, 2) ** 2],
         100, 1.0),
        ("k-below-one-is-one", [np.arange(10.0).reshape(5, 2)] * 2, 0, 1.0),
    ]:
        with subtests.test(case=cid):
            assert hp.embedding_stability(
                embeddings, neighbourhood_k=k) == pytest.approx(expected)

    for cid, embeddings, message in [
        ("one-repeat", [np.zeros((5, 2))],
         "Embedding stability requires at least two repeats."),
        ("one-dimensional", [np.zeros(5), np.zeros(5)],
         "Embedding stability needs 2-D embeddings with at least 3 rows."),
        ("two-rows", [np.zeros((2, 2)), np.zeros((2, 2))],
         "Embedding stability needs 2-D embeddings with at least 3 rows."),
        ("ragged", [np.zeros((5, 2)), np.zeros((4, 2))],
         "Every repeated embedding must have the same sample shape."),
        ("nan", [np.zeros((5, 2)), np.full((5, 2), np.nan)],
         "Repeated embeddings contain NaN or infinite values."),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as info:
                hp.embedding_stability(embeddings)
            assert str(info.value) == message


# ===========================================================================
# _cluster_structure
# ===========================================================================

def test_cluster_structure_uses_labels_or_discovers_a_partition(subtests):
    """Usable labels give the supplied-label silhouette; otherwise K-means
    over 2..8 clusters picks the best silhouette; identical points resolve
    no clusters (zero); fewer than 4 unlabelled rows are refused."""
    features, labels = _blobs()
    embedding = features[:, :2]
    silhouette = 0.9528695202054873
    for cid, values, supplied, expected in [
        ("supplied-labels", embedding, labels,
         (silhouette, silhouette, "supplied_labels", 3)),
        ("discovered", embedding, None,
         (silhouette, silhouette, "discovered_kmeans", 3)),
        ("wrong-length-labels-are-ignored", embedding, labels[:5],
         (silhouette, silhouette, "discovered_kmeans", 3)),
        ("one-class-labels-are-ignored", embedding, np.zeros(36),
         (silhouette, silhouette, "discovered_kmeans", 3)),
        ("all-distinct-labels-are-ignored", embedding, np.arange(36),
         (silhouette, silhouette, "discovered_kmeans", 3)),
        ("identical-points", np.zeros((6, 2)), None,
         (0.0, 0.0, "no_resolved_clusters", 1)),
    ]:
        with subtests.test(case=cid):
            got = hp._cluster_structure(values, supplied, seed=0)
            assert got[2:] == expected[2:]
            assert got[:2] == pytest.approx(expected[:2])

    with subtests.test(case="negative-silhouette-normalises-to-zero"):
        crossed = np.array([[0.0, 0], [10, 0], [0.1, 0], [10.1, 0]])
        normalised, raw, method, n = hp._cluster_structure(
            crossed, np.array([0, 0, 1, 1]), seed=0)
        assert raw < 0 and normalised == 0.0
        assert (method, n) == ("supplied_labels", 2)

    with subtests.test(case="three-rows-refused"):
        with pytest.raises(ValueError) as info:
            hp._cluster_structure(np.zeros((3, 2)), None, seed=0)
        assert str(info.value) == (
            "Unsupervised cluster-structure scoring needs at least 4 rows.")


# ===========================================================================
# _objective_weights
# ===========================================================================

def test_objective_weights_merge_over_defaults_and_normalise(subtests):
    """Named weights merge over the 0.4/0.3/0.3 defaults and the sum is
    renormalised to one; unknown, non-numeric, negative, non-finite and
    all-zero weightings are refused by message."""
    for cid, weights, expected in [
        ("defaults", None, {"neighborhood_preservation": 0.4,
                            "stability": 0.3, "cluster_structure": 0.3}),
        ("empty-is-defaults", {}, {"neighborhood_preservation": 0.4,
                                   "stability": 0.3,
                                   "cluster_structure": 0.3}),
        ("one-named", {"stability": 1.0},
         {"neighborhood_preservation": 0.4 / 1.7, "stability": 1.0 / 1.7,
          "cluster_structure": 0.3 / 1.7}),
        ("one-zeroed", {"stability": 0, "cluster_structure": "0.6"},
         {"neighborhood_preservation": 0.4, "stability": 0.0,
          "cluster_structure": 0.6}),
    ]:
        with subtests.test(case=cid):
            assert hp._objective_weights(weights) == pytest.approx(expected)

    for cid, weights, message in [
        ("unknown", {"speed": 1.0},
         "Unknown UMAP objective weight(s): ['speed']. Choose from "
         "['cluster_structure', 'neighborhood_preservation', 'stability']."),
        ("word", {"stability": "lots"},
         "UMAP objective weight 'stability' must be numeric."),
        ("negative", {"stability": -1.0},
         "UMAP objective weight 'stability' must be finite and zero or "
         "greater."),
        ("nan", {"stability": float("nan")},
         "UMAP objective weight 'stability' must be finite and zero or "
         "greater."),
        ("all-zero", {"stability": 0.0, "cluster_structure": 0.0,
                      "neighborhood_preservation": 0.0},
         "At least one UMAP objective weight must be positive."),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as info:
                hp._objective_weights(weights)
            assert str(info.value) == message


# ===========================================================================
# umap_objective_scores
# ===========================================================================

def test_umap_objective_scores_reports_each_objective_and_their_mean(
        subtests):
    """The three objectives, their components, the provenance fields and the
    weighted geometric mean are reported exactly; a collapsed objective
    drags the composite down; one repeat is refused."""
    features, labels = _blobs()
    rng = np.random.default_rng(1)
    repeats = [features[:, :2] + rng.normal(scale=0.01,
                                            size=(len(features), 2))
               for _ in range(3)]

    with subtests.test(case="three-objectives-beside-their-composite"):
        scores = hp.umap_objective_scores(features, repeats, labels=labels,
                                          neighbourhood_k=5, seed=0)
        expected = {
            "neighborhood_preservation": 0.9621658548068284,
            "stability": 0.9722222222222222,
            "cluster_structure": 0.9528465214922672,
            "multi_objective": 0.9623576992572027,
            "trustworthiness": 0.9595899470899472,
            "continuity": 0.9647486772486773,
            "cluster_structure_raw_silhouette": 0.9528465214922672,
            "silhouette": 0.9528465214922672,
            "neighbourhood_k": 5.0,
        }
        assert {k: scores[k] for k in expected} == pytest.approx(expected)
        assert scores["neighborhood_preservation"] == pytest.approx(
            math.sqrt(scores["trustworthiness"] * scores["continuity"]))
        assert (scores["cluster_structure_method"], scores["cluster_counts"],
                scores["stability_repeats"]) == ("supplied_labels",
                                                 [3, 3, 3], 3)
        assert scores["objective_weights"] == pytest.approx(
            {"neighborhood_preservation": 0.4, "stability": 0.3,
             "cluster_structure": 0.3})
        assert scores["multi_objective"] == pytest.approx(math.exp(sum(
            w * math.log(scores[name])
            for name, w in scores["objective_weights"].items())))

    with subtests.test(case="no-labels-discovers-and-has-no-silhouette"):
        scores = hp.umap_objective_scores(features, repeats[:2],
                                          labels=labels[:3],
                                          neighbourhood_k=5, seed=0)
        assert scores["cluster_structure_method"] == "discovered_kmeans"
        assert "silhouette" not in scores

    with subtests.test(case="collapsed-objective-drags-composite-down"):
        scatter = np.random.default_rng(2)
        scattered = [scatter.normal(size=(len(features), 2))
                     for _ in range(2)]
        good = hp.umap_objective_scores(
            features, [features[:, :2], features[:, :2]], neighbourhood_k=5)
        bad = hp.umap_objective_scores(features, scattered, neighbourhood_k=5)
        assert good["multi_objective"] == pytest.approx(0.9703852038640703)
        assert bad["multi_objective"] == pytest.approx(0.3023381907172953)

    with subtests.test(case="one-repeat-refused"):
        with pytest.raises(ValueError) as info:
            hp.umap_objective_scores(features, [features[:, :2]])
        assert str(info.value) == (
            "Multi-objective UMAP search requires at least two stability "
            "repeats.")


# ===========================================================================
# umap_search
# ===========================================================================

def _features(rows=30, cols=4, seed=0):
    return np.random.default_rng(seed).normal(size=(rows, cols))


def _embed(calls=None):
    """Deterministic stand-in for UMAP: the first two columns, shifted."""
    def embed(feats, params):
        if calls is not None:
            calls.append(dict(params))
        values = np.asarray(feats, dtype=float)
        shift = float(params.get("min_dist", 0.1))
        return np.column_stack([values[:, 0] + shift, values[:, 1] - shift])
    return embed


def test_umap_search_refuses_bad_arguments_before_embedding(subtests):
    """Unknown criteria, a display dimensionality other than 2 or 3,
    silhouette without labels, non-numeric or too few stability repeats,
    something that is not a table, fewer than three rows, a word for
    n_neighbors, an unknown backend and a Walk over a grid are refused by
    message."""
    space = SearchSpace({"n_neighbors": [5], "min_dist": [0.1]})
    features = _features()
    for cid, args, kwargs, message in [
        ("unknown-criterion", (features, space), {"metric": "beauty"},
         "Unknown UMAP criterion 'beauty'. Choose one of "),
        ("components-word", (features, space), {"n_components": "many"},
         "UMAP n_components must be 2 or 3."),
        ("components-four", (features, space), {"n_components": 4},
         "UMAP n_components must be 2 or 3."),
        ("components-one", (features, space), {"n_components": 1},
         "UMAP n_components must be 2 or 3."),
        ("silhouette-without-labels", (features, space),
         {"metric": "silhouette"},
         "The 'silhouette' criterion scores how well the embedding separates "
         "labels you already have, so it needs `labels=`."),
        ("repeats-word", (features, space), {"stability_repeats": "a few"},
         "stability_repeats must be a whole number."),
        ("multi-objective-one-repeat", (features, space),
         {"metric": "multi_objective", "stability_repeats": 1},
         "Multi-objective UMAP search needs stability_repeats >= 2."),
        ("not-a-table", (object(), space), {},
         "UMAP features must be a 2-D array-like object."),
        ("two-rows", ([[1.0, 2.0], [3.0, 4.0]], space), {},
         "UMAP hyperparameter search needs at least 3 rows after filtering; "
         "only 2 remain."),
        ("neighbours-as-a-word",
         (features, SearchSpace({"n_neighbors": ["fifteen"]})), {},
         "UMAP n_neighbors values must be whole numbers; got 'fifteen'."),
        ("unknown-backend", (features, space), {"backend": "tpu"},
         "UMAP backend must be 'cpu' or 'cuml'."),
        ("walk-over-a-grid",
         (features, SearchSpace({"n_neighbors": [5, 10], "min_dist": [0.1]})),
         {"adaptive": True},
         "A Walk needs exactly one starting value for every parameter"),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as info:
                hp.umap_search(*args, **kwargs)
            assert str(info.value).startswith(message)


def test_umap_search_scores_walks_clusters_and_resumes(subtests, tmp_path,
                                                       monkeypatch):
    """A grid scores every criterion per trial and clamps n_neighbors to the
    rows; the backend is named honestly (custom / cuml, never a silent CPU
    downgrade); clustering failures stay on their row; a Walk moves along
    the axes the user named; multi-objective fits seeded repeats and
    declares the Pareto objectives; a resumed sweep replays its trials
    without re-embedding; a missing umap-learn returns an empty result."""
    features = _features()

    with subtests.test(case="grid-clamps-neighbours-and-scores-everything"):
        calls = []
        result = hp.umap_search(
            features, SearchSpace({"n_neighbors": [5, 100, 200],
                                   "min_dist": [0.1]}),
            embed_fn=_embed(calls), neighbourhood_k=5, n_components=3)
        assert [c["n_neighbors"] for c in calls] == [5, 29]
        assert {c["n_components"] for c in calls} == {3}
        row = result.trials[0].extra_metrics
        assert (row["criterion"], row["backend"], row["n_components"],
                row["neighbourhood_k"]) == ("trustworthiness", "custom", 3,
                                            5.0)
        assert result.trials[0].score == row["trustworthiness"]
        assert row["embedding"].shape == (30, 2)
        assert result.notes[:4] == [
            hp.UMAP_NO_GROUND_TRUTH,
            f"Criterion 'trustworthiness': "
            f"{hp.UMAP_CRITERIA['trustworthiness']}",
            "Every criterion was computed for every trial, so you can "
            "re-rank the table by a different one and see whether the winner "
            "survives.",
            "Embedding backend: custom. A cuML UMAP is a different map of the "
            "same data, not the same map faster."]
        assert result.notes[4] == (
            "n_neighbors was limited to 2…29 for the 30 available rows; "
            "duplicate effective values were evaluated only once.")

    with subtests.test(case="implicit-neighbours-limited-for-few-rows"):
        calls = []
        result = hp.umap_search(_features(rows=10),
                                SearchSpace({"min_dist": [0.1]}),
                                embed_fn=_embed(calls), neighbourhood_k=3,
                                keep_embeddings=False)
        assert calls == [{"min_dist": 0.1, "n_neighbors": 9,
                          "n_components": 2}]
        assert "embedding" not in result.trials[0].extra_metrics
        assert result.notes[4] == ("UMAP's default n_neighbors was limited "
                                   "from 15 to 9 for the 10 available rows.")

    with subtests.test(case="injected-embedder-is-custom-even-for-cuml"):
        result = hp.umap_search(_features(rows=20, seed=3),
                                SearchSpace({"n_neighbors": [5]}),
                                backend="cuml",
                                embed_fn=lambda values, _p: values[:, :2])
        assert result.best.extra_metrics["backend"] == "custom"

    with subtests.test(case="silhouette-with-labels"):
        blobs, labels = _blobs()
        result = hp.umap_search(blobs, SearchSpace({"n_neighbors": [5]}),
                                metric="silhouette", labels=labels,
                                embed_fn=lambda v, _p: v[:, :2],
                                neighbourhood_k=5)
        assert result.best.score == pytest.approx(0.9528695202054873)

    import spacr.umap_search as umap_search_module

    with subtests.test(case="clustering-failure-stays-on-its-row"):
        def _refuses(*_args, **_kwargs):
            raise RuntimeError("hdbscan is not installed")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(umap_search_module, "walk_clusters", _refuses)
            result = hp.umap_search(features, SearchSpace({"n_neighbors": [5]}),
                                    embed_fn=_embed(),
                                    cluster_during_search=True,
                                    neighbourhood_k=5)
        row = result.trials[0]
        assert row.ok and "embedding" in row.extra_metrics
        assert row.extra_metrics["cluster_error"] == (
            "RuntimeError: hdbscan is not installed")
        assert result.notes[4] == (
            "Each completed embedding also ran an HDBSCAN clustering walk; "
            "the retained labels and cluster score belong to that exact map.")

    with subtests.test(case="clustering-success-keeps-the-walk"):
        class _Row:
            def __init__(self, size):
                self.min_cluster_size = size
                self.labels = np.zeros(30, dtype=int)
                self.silhouette, self.score = 0.42, 0.4
                self.n_clusters, self.noise_fraction = 2, 0.1

        sizes = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(umap_search_module, "walk_clusters",
                       lambda emb, min_cluster_sizes: sizes.append(
                           tuple(min_cluster_sizes)) or [_Row(12), _Row(5)])
            result = hp.umap_search(features, SearchSpace({"n_neighbors": [5]}),
                                    embed_fn=_embed(),
                                    cluster_during_search=True,
                                    cluster_sizes=(5, 12), neighbourhood_k=5)
        extra = result.trials[0].extra_metrics
        assert sizes == [(5, 12)]
        assert (extra["cluster_min_size"], extra["n_clusters"],
                extra["cluster_silhouette"], extra["cluster_score"],
                extra["cluster_noise_fraction"]) == (12, 2, 0.42, 0.4, 0.1)
        assert [r["min_cluster_size"] for r in extra["cluster_walk"]] == [12, 5]

    with subtests.test(case="walk-moves-along-the-named-axes"):
        result = hp.umap_search(
            features, SearchSpace({"n_neighbors": [8], "min_dist": [0.2]}),
            adaptive=True, walk_parameters=("min_dist",),
            walk_steps={"min_dist": 0.05}, n_trials=2, embed_fn=_embed(),
            neighbourhood_k=5)
        assert {t.params["n_neighbors"] for t in result.trials} == {8}
        assert [t.params["min_dist"] for t in result.trials][:2] == [0.15,
                                                                     0.25]

    with subtests.test(case="walk-without-step-override-uses-default"):
        result = hp.umap_search(
            features, SearchSpace({"n_neighbors": [8], "min_dist": [0.2],
                                   "init": ["spectral"]}),
            adaptive=True, walk_parameters=("min_dist", "init"),
            walk_steps={"n_neighbors": 3}, n_trials=1, embed_fn=_embed(),
            neighbourhood_k=5)
        assert [t.params for t in result.trials] == [
            {"n_neighbors": 8, "min_dist": 0.15, "init": "spectral"},
            {"n_neighbors": 8, "min_dist": 0.15, "init": "random"},
            {"n_neighbors": 8, "min_dist": 0.25, "init": "spectral"},
            {"n_neighbors": 8, "min_dist": 0.25, "init": "random"}]

    with subtests.test(case="default-walk-uses-the-umap-pair"):
        calls = []
        result = hp.umap_search(
            features, SearchSpace({"n_neighbors": [8], "min_dist": [0.2]}),
            adaptive=True, n_trials=1, n_neighbors_step=2, min_dist_step=0.1,
            embed_fn=_embed(calls), neighbourhood_k=5)
        assert [(c["n_neighbors"], c["min_dist"]) for c in calls] == [
            (6, 0.1), (6, 0.3), (10, 0.1), (10, 0.3)]
        assert result.objectives == {}

    for cid, adaptive in [("grid", False), ("walk", True)]:
        with subtests.test(case=f"multi-objective-{cid}"):
            calls = []
            result = hp.umap_search(
                features, SearchSpace({"n_neighbors": [8], "min_dist": [0.2]}),
                metric="multi_objective", stability_repeats=2, seed=4,
                adaptive=adaptive, n_trials=1, embed_fn=_embed(calls),
                neighbourhood_k=5, objective_weights={"stability": 1.0})
            assert result.objectives == {name: True
                                         for name in hp.UMAP_OBJECTIVES}
            assert [c["random_state"] for c in calls][:2] == [4, 5]
            assert result.trials[0].extra_metrics["stability"] == 1.0
            assert ("Composite search direction uses normalized weights "
                    "(neighborhood_preservation=0.235, stability=0.588, "
                    "cluster_structure=0.176) and 2 reproducible embedding "
                    "repeats per configuration.") in result.notes

    with subtests.test(case="resume-replays-without-re-embedding"):
        space = SearchSpace({"n_neighbors": [5, 8], "min_dist": [0.1]})
        path = str(tmp_path / "sweep.json")
        hp.umap_search(features, space, embed_fn=_embed(), neighbourhood_k=5,
                       checkpoint_path=path)
        calls, reported = [], []
        result = hp.umap_search(
            features, space, embed_fn=_embed(calls), neighbourhood_k=5,
            checkpoint_path=path, resume=True,
            on_trial=lambda trial, done, total: reported.append((done, total)))
        assert calls == []
        assert reported == [(1, 2), (2, 2)]
        assert [t.params["n_neighbors"] for t in result.trials] == [5, 8]
        assert f"Resumed 2 completed trial(s) from {path}." in result.notes
        stored = json.loads(open(path, encoding="utf-8").read())
        assert stored["meta"] == {"n_trials_completed": 2}

    with subtests.test(case="umap-learn-missing-returns-an-empty-result"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(hp, "umap_available", lambda: (False, "install it"))
            result = hp.umap_search(features, SearchSpace({"n_neighbors": [5]}))
        assert (result.trials, result.best, result.notes) == ([], None,
                                                              ["install it"])

    from spacr import gpu_reduce

    class _DeviceArray:
        def __init__(self, host, copied):
            self._host, self._copied = host, copied

        def get(self):
            self._copied.append(True)
            return self._host

    for cid, available, wrap, actual, outcome in [
        ("cuml-device-array-copied-home", True, True, "cuml", "ok"),
        ("cuml-host-array-passes-through", True, False, "cuml", "ok"),
        ("non-cuml-reducer-fails-the-trial", True, False, "umap-learn",
         "RuntimeError: cuML failed to construct the requested UMAP; the GPU "
         "run was stopped instead of mixing CPU and GPU rows."),
        ("no-cuml-is-refused", False, False, "cuml", "refused"),
    ]:
        with subtests.test(case=cid):
            copied, built = [], []

            class _Reducer:
                def fit_transform(self, feats, copied=copied, wrap=wrap):
                    value = np.ones((len(feats), 2))
                    return _DeviceArray(value, copied) if wrap else value

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(gpu_reduce, "rapids_available",
                           lambda flag=available: flag)
                mp.setattr(gpu_reduce, "make_reducer",
                           lambda *a, out=built, name=actual, **k: out.append(
                               (a, k)) or (_Reducer(), name))
                if outcome == "refused":
                    with pytest.raises(RuntimeError, match="^GPU UMAP was "
                                                           "requested"):
                        hp.umap_search(features,
                                       SearchSpace({"n_neighbors": [5]}),
                                       backend="cuml", seed=7)
                    continue
                result = hp.umap_search(features,
                                        SearchSpace({"n_neighbors": [5]}),
                                        backend="cuml", seed=7,
                                        neighbourhood_k=5)
            trial = result.trials[0]
            assert built == [(("umap",), {"prefer_gpu": True, "n_neighbors": 5,
                                          "n_components": 2,
                                          "random_state": 7})]
            if outcome == "ok":
                assert trial.ok and trial.extra_metrics["backend"] == "cuml"
                assert copied == ([True] if wrap else [])
                assert np.array_equal(trial.extra_metrics["embedding"],
                                      np.ones((30, 2)))
            else:
                assert trial.error == outcome


# ===========================================================================
# umap_checkpoint_path
# ===========================================================================

def test_umap_checkpoint_path_lives_under_the_project(subtests, tmp_path):
    """An explicit path wins as given; otherwise the checkpoint lives in the
    project's results folder however deep ``src`` points, and a missing or
    absent source has none."""
    project = tmp_path / "plate01"
    (project / "measurements").mkdir(parents=True)
    db = project / "measurements" / "measurements.db"
    db.write_bytes(b"")
    expected = str(project / "results" / ".spacr_checkpoints" /
                   "umap_search.json")
    wanted = tmp_path / "elsewhere" / "cp.json"
    for cid, settings, result in [
        ("explicit-path", {"checkpoint_path": str(wanted),
                           "src": str(project)}, str(wanted)),
        ("explicit-home-is-expanded", {"checkpoint_path": "~/cp.json"},
         os.path.join(os.path.expanduser("~"), "cp.json")),
        ("no-source", {}, None),
        ("empty-source-list", {"src": []}, None),
        ("source-list-of-blanks", {"src": [None, ""]}, None),
        ("absent-source", {"src": str(tmp_path / "absent")}, None),
        ("database-in-a-list", {"src": [None, str(db)]}, expected),
        ("measurements-folder", {"src": str(project / "measurements")},
         expected),
        ("project-folder", {"src": str(project)}, expected),
    ]:
        with subtests.test(case=cid):
            assert hp.umap_checkpoint_path(settings) == result
    assert not wanted.parent.exists()


# ===========================================================================
# _save_array_atomic
# ===========================================================================

def test_save_array_atomic_writes_whole_or_leaves_nothing(subtests, tmp_path):
    """The array lands whole in a created folder with no temporary left; a
    failed write removes its temporary and re-raises, and a temporary that
    cannot be removed does not hide the write error."""
    with subtests.test(case="written-whole"):
        target = tmp_path / "deep" / "embedding.npy"
        array = np.arange(6, dtype=float).reshape(3, 2)
        hp._save_array_atomic(target, array)
        assert np.array_equal(np.load(target), array)
        assert [p.name for p in target.parent.iterdir()] == ["embedding.npy"]

    def _explode(stream, arr, allow_pickle=False):
        raise OSError("the disk filled up")

    with subtests.test(case="failed-write-leaves-no-temporary"):
        folder = tmp_path / "failed"
        folder.mkdir()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(np, "save", _explode)
            with pytest.raises(OSError, match="^the disk filled up$"):
                hp._save_array_atomic(folder / "embedding.npy", np.zeros(3))
        assert list(folder.iterdir()) == []

    with subtests.test(case="cleanup-failure-keeps-the-real-error"):
        folder = tmp_path / "stuck"
        folder.mkdir()

        def _cannot_unlink(path):
            raise OSError("permission denied")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(np, "save", _explode)
            mp.setattr("os.unlink", _cannot_unlink)
            with pytest.raises(OSError, match="^the disk filled up$"):
                hp._save_array_atomic(folder / "embedding.npy", np.zeros(3))
        assert [p.name.endswith(".tmp") for p in folder.iterdir()] == [True]


# ===========================================================================
# _UmapCheckpoint
# ===========================================================================

def _checkpoint(tmp_path, *, resume=False, keep_embeddings=True,
                signature="matrix-a"):
    return hp._UmapCheckpoint(str(tmp_path / "umap_search.json"),
                              {"features": signature}, resume,
                              keep_embeddings)


def test_umap_checkpoint_resumes_only_what_it_can_restore(subtests, tmp_path):
    """A recorded trial comes back with its embedding, round and state; a
    vanished or corrupt embedding, a scored trial with no embedding when one
    is wanted, and a hand-edited entry are dropped (recomputed); a
    checkpoint of a different matrix refuses to resume."""
    from spacr.checkpoint import CheckpointMismatch

    with subtests.test(case="trial-comes-back-with-its-embedding"):
        folder = tmp_path / "roundtrip"
        store = _checkpoint(folder)
        embedding = np.arange(8, dtype=float).reshape(4, 2)
        store.record(Trial(params={"n_neighbors": 5}, score=0.8, index=3,
                           duration=1.5,
                           extra_metrics={"embedding": embedding,
                                          "trustworthiness": 0.8}),
                     round_index=2, state={"round": 2})
        store.finish({"round": 2})
        reopened = _checkpoint(folder, resume=True)
        assert reopened.resumed is True and reopened.state == {"round": 2}
        loaded = reopened.load()
        assert list(loaded) == [hp._trial_key({"n_neighbors": 5})]
        (restored, round_index), = loaded.values()
        assert (round_index, restored.params, restored.score,
                restored.duration, restored.index, restored.error) == (
            2, {"n_neighbors": 5}, 0.8, 1.5, 3, None)
        assert np.array_equal(restored.extra_metrics.pop("embedding"),
                              embedding)
        assert restored.extra_metrics == {"trustworthiness": 0.8}

    with subtests.test(case="fresh-store-is-not-resumed"):
        folder = tmp_path / "fresh"
        assert _checkpoint(folder).resumed is False
        assert _checkpoint(folder).load() == {}

    for cid, damage in [("vanished", "unlink"), ("corrupt", "garbage")]:
        with subtests.test(case=f"embedding-{cid}-is-recomputed"):
            folder = tmp_path / cid
            store = _checkpoint(folder)
            store.record(Trial(params={"n_neighbors": 5}, score=0.8, index=0,
                               extra_metrics={"embedding": np.zeros((3, 2))}))
            store.finish()
            for artifact in (folder / "umap_search.json.d").iterdir():
                if damage == "unlink":
                    artifact.unlink()
                else:
                    artifact.write_bytes(b"not an npy file")
            assert _checkpoint(folder, resume=True).load() == {}

    for cid, keep, expected in [("embedding-wanted", True, 0),
                                ("embedding-not-wanted", False, 1)]:
        with subtests.test(case=f"scored-without-embedding-{cid}"):
            folder = tmp_path / cid
            store = _checkpoint(folder, keep_embeddings=keep)
            store.record(Trial(params={"n_neighbors": 5}, score=0.8, index=0))
            store.finish()
            assert len(_checkpoint(folder, resume=True,
                                   keep_embeddings=keep).load()) == expected

    with subtests.test(case="hand-edited-entry-is-skipped"):
        folder = tmp_path / "edited"
        store = _checkpoint(folder, keep_embeddings=False)
        store.record(Trial(params={"n_neighbors": 5}, score=0.8, index=0))
        store.finish()
        path = folder / "umap_search.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["completed"]["hand-edited"] = "not a record"
        path.write_text(json.dumps(document), encoding="utf-8")
        loaded = _checkpoint(folder, resume=True,
                             keep_embeddings=False).load()
        assert list(loaded) == [hp._trial_key({"n_neighbors": 5})]

    with subtests.test(case="different-matrix-is-not-resumed"):
        folder = tmp_path / "mismatch"
        store = _checkpoint(folder, signature="matrix-a")
        store.record(Trial(params={"n_neighbors": 5}, score=0.8, index=0))
        store.finish()
        with pytest.raises(CheckpointMismatch):
            _checkpoint(folder, resume=True, signature="matrix-b")


# ===========================================================================
# activation_fit_fn
# ===========================================================================

@pytest.fixture
def corner_model():
    """A model whose logit depends only on the top-left corner of the image."""
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Corner(nn.Module):
        """Nothing is trained; the weights are set by hand."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1)
            self.head = nn.Linear(1, 2)
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.weight[0, 0, 1, 1] = 1.0
                self.conv.bias.zero_()
                self.head.weight.fill_(8.0)
                self.head.bias.zero_()

        def forward(self, x):
            feature = self.conv(x)
            mask = torch.zeros_like(feature)
            mask[:, :, :3, :3] = 1.0
            return self.head((feature * mask).mean(dim=(2, 3)))

    image = torch.zeros(1, 16, 16)
    image[0, :3, :3] = 2.0
    return hp.ActivationSearchData(model=Corner().eval(), images=[image],
                                   filenames=["synthetic"],
                                   model_type="corner")


def test_activation_fit_fn_scores_every_criterion_per_trial(subtests,
                                                            corner_model):
    """Unknown criteria and an empty image set are refused by message; a
    trial reports deletion and insertion AUC (score = the ranked one), the
    pointing game when masks exist, keeps its first map only on request,
    uses SmoothGrad when samples are asked for, routes through an injected
    attribution, and says why a criterion it could not compute is
    missing."""
    from spacr.attribution import attribute

    for cid, data, kwargs, message in [
        ("unknown-criterion",
         hp.ActivationSearchData(model=object(), images=[object()]),
         {"criterion": "iou"}, "Unknown Activation criterion 'iou'."),
        ("no-images", hp.ActivationSearchData(model=object(), images=[]), {},
         "The Activation search has no images to score on. Point 'dataset' "
         "at a crop tar"),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as info:
                hp.activation_fit_fn(data, **kwargs)
            assert str(info.value).startswith(message)

    with subtests.test(case="injected-attribution"):
        seen = []

        def _attribute(model, image, params):
            seen.append(dict(params))
            return attribute(model, image, "saliency", model_type="corner")

        fit = hp.activation_fit_fn(corner_model, n_steps=3,
                                   run_sanity_check=False,
                                   attribute_fn=_attribute)
        score, extra = fit({"cam_type": "gradcam"})
        assert seen == [{"cam_type": "gradcam"}]
        assert score == extra["deletion_auc"]
        assert (extra["n_images"], extra["n_flat_maps"], extra["criterion"],
                extra["fold_std"], extra["deletion_std"],
                extra["insertion_std"]) == (1, 0, "deletion_auc", 0.0, 0.0,
                                            0.0)
        assert math.isfinite(extra["insertion_auc"])
        assert "attribution" in extra and "pointing_game" not in extra

    for cid, params in [
        ("plain-saliency", {"cam_type": "saliency_image"}),
        ("smoothgrad", {"cam_type": "saliency", "smoothgrad_samples": 2,
                        "smoothgrad_sigma": 0.1}),
    ]:
        with subtests.test(case=f"maps-dropped-{cid}"):
            fit = hp.activation_fit_fn(corner_model, n_steps=3,
                                       criterion="insertion_auc",
                                       run_sanity_check=False,
                                       keep_maps=False)
            score, extra = fit(params)
            assert score == extra["insertion_auc"]
            assert math.isfinite(extra["deletion_auc"])
            assert extra["n_images"] == 1 and "attribution" not in extra

    with subtests.test(case="pointing-game-with-masks"):
        mask = np.zeros((16, 16), dtype=bool)
        mask[:3, :3] = True
        data = hp.ActivationSearchData(model=corner_model.model,
                                       images=corner_model.images,
                                       masks=[mask], model_type="corner")
        fit = hp.activation_fit_fn(data, n_steps=3, criterion="pointing_game",
                                   run_sanity_check=False)
        score, extra = fit({"cam_type": "saliency"})
        assert (score, extra["pointing_hits"], extra["pointing_scored"]) == (
            1.0, 1, 1)

    for cid, criterion, fragment in [
        ("pointing-game-without-masks", "pointing_game",
         "no object masks were available, so the pointing game has no "
         "answer key."),
        ("sanity-gap-with-check-off", "sanity_gap",
         "the randomisation sanity check was disabled for this sweep."),
    ]:
        with subtests.test(case=cid):
            fit = hp.activation_fit_fn(corner_model, n_steps=3,
                                       criterion=criterion,
                                       run_sanity_check=False)
            with pytest.raises(ValueError) as info:
                fit({"cam_type": "saliency"})
            assert str(info.value).startswith(
                f"criterion {criterion!r} could not be computed for this "
                f"trial: {fragment}")


# ===========================================================================
# The sweep panel's gallery (Trial embeddings drawn as small multiples)
# ===========================================================================

def test_the_umap_gallery_is_refreshed_not_duplicated(qtbot):
    """Reopening the gallery after a new result reuses the one window."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.hyperparam import HyperparamPanel

    def _result(n):
        trials = [Trial(params={"n_neighbors": 5 * (i + 1)}, score=0.9,
                        index=i, duration=0.01, extra_metrics={
                            "embedding": np.random.default_rng(i).normal(
                                size=(12, 2))})
                  for i in range(n)]
        return hp.SearchResult(trials=trials, best=trials[0],
                               metric="trustworthiness",
                               space=SearchSpace({"n_neighbors": [
                                   t.params["n_neighbors"] for t in trials]}))

    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    panel._result = _result(1)
    first = panel.open_umap_grid()
    assert first is not None
    qtbot.addWidget(first)
    panel._result = _result(2)
    assert panel.open_umap_grid() is first
