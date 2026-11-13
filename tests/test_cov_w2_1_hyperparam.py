"""Cross-validated search, the estimators it builds, and the report it prints.

The pieces here are the ones a hyperparameter sweep is actually made of: the
fold builder that keeps a well on one side of a split, the leak check that
refuses folds touching the test indices, the estimator ladder the classical-ML
sweep constructs, and the text report a user reads afterwards. Everything is
driven with real numeric data and real scikit-learn estimators -- a fold that
straddles a well is the defect this module exists to prevent, and only a real
split can show it did not happen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import hyperparam as hp
from spacr.hyperparam import SearchSpace

# ---------------------------------------------------------------------------
# Wells, and the matrix a search runs on


def test_a_table_with_no_plate_columns_cannot_be_grouped_by_well():
    """The folds are then ungrouped, and the scores are optimistic."""
    groups, warning = hp._well_groups(pd.DataFrame({"cell_area": [1.0, 2.0]}))

    assert groups is None
    assert "the folds are ungrouped" in warning


def test_well_ids_are_plate_row_and_column_joined():
    """One id per well, so a well lands on one side of every split."""
    frame = pd.DataFrame({"plateID": ["p1", "p1"], "rowID": ["A", "A"],
                          "columnID": ["01", "02"]})

    groups, warning = hp._well_groups(frame)

    assert warning is None
    assert list(groups) == ["p1_A_01", "p1_A_02"]


# ---------------------------------------------------------------------------
# The space itself


@pytest.mark.parametrize("params, message", [
    (["n_neighbors", 5], "must be a mapping"),
    ({}, "Search space is empty"),
    ({"": [1]}, "non-empty strings"),
    ({"n_neighbors": 5}, "must be a list or tuple"),
    ({"n_neighbors": "5"}, "must be a list or tuple"),
    ({"n_neighbors": []}, "empty value list"),
])
def test_a_space_that_cannot_be_searched_says_what_is_wrong(params, message):
    """Every refusal shows the caller the shape it should have written."""
    with pytest.raises(ValueError, match=message):
        SearchSpace(params)


# ---------------------------------------------------------------------------
# The shared runner


def test_a_completed_trial_can_be_replayed_instead_of_refitted():
    """A resumed search must not pay for the trials it already ran."""
    space = SearchSpace({"a": [1, 2]})
    combos = space.grid()
    done = hp.Trial(params=dict(combos[0]), score=0.9, index=99)
    fitted = []

    result = hp._run_trials(lambda p: fitted.append(p) or 0.1, combos, space,
                            "score",
                            prior_trials={hp._trial_key(combos[0]): done})

    assert fitted == [combos[1]]
    assert result.best is done
    assert done.index == 0


# ---------------------------------------------------------------------------
# Result slices


def test_a_result_with_no_successful_trials_has_no_statistics():
    """None, not zero: zero is a score and this has none."""
    stats = hp.SearchResult().score_stats()

    assert stats == {"n": 0, "best": None, "worst": None, "mean": None,
                     "std": None, "spread": None}


# ---------------------------------------------------------------------------
# Persisting a UMAP search


def test_two_equal_arrays_have_one_fingerprint():
    """The signature decides whether a checkpoint may be resumed at all."""
    a = np.arange(12, dtype=float).reshape(3, 4)

    assert hp._array_fingerprint(a) == hp._array_fingerprint(a.copy())
    assert hp._array_fingerprint(a) != hp._array_fingerprint(a + 1)
    assert hp._array_fingerprint(a) != hp._array_fingerprint(
        a.astype(np.float32))


def test_an_object_array_is_fingerprinted_through_its_values():
    """A matrix of strings has no buffer to digest, and still needs an id."""
    words = np.array([["a", "b"], ["c", "d"]], dtype=object)

    assert hp._array_fingerprint(words) == hp._array_fingerprint(words.copy())
    assert hp._array_fingerprint(words) != hp._array_fingerprint(
        np.array([["a", "b"], ["c", "e"]], dtype=object))


# ---------------------------------------------------------------------------
# Resuming a UMAP search from its checkpoint


def _checkpoint(tmp_path, *, resume=False, keep_embeddings=True,
                signature="matrix-a"):
    return hp._UmapCheckpoint(str(tmp_path / "umap_search.json"),
                              {"features": signature}, resume,
                              keep_embeddings)


def test_a_failed_trial_is_remembered_without_an_embedding(tmp_path):
    """It failed; there is nothing to recompute and nothing to store."""
    store = _checkpoint(tmp_path)
    store.record(hp.Trial(params={"n_neighbors": 5}, index=0,
                          error="RuntimeError: no"))
    store.update({"round": 1})
    store.finish()

    (restored, _round), = _checkpoint(tmp_path, resume=True).load().values()

    assert restored.error == "RuntimeError: no"
    assert restored.score is None


# ---------------------------------------------------------------------------
# What the settings panel may offer


def test_the_metric_list_comes_from_the_installed_umap():
    """A metric the installed umap rejects would fail deep inside the run."""
    metrics = hp.umap_metrics()

    assert "euclidean" in metrics
    assert "cosine" in metrics
    assert metrics == tuple(sorted(metrics))


def test_the_panel_still_offers_metrics_without_umap_installed(monkeypatch):
    """A user configuring a run on a laptop and running it elsewhere."""
    import builtins

    real_import = builtins.__import__

    def _no_umap(name, *args, **kwargs):
        if name.startswith("umap"):
            raise ImportError("No module named 'umap'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_umap)

    assert hp.umap_metrics() == hp.UMAP_METRICS


def test_umap_reports_itself_available_when_it_imports():
    """The panel greys the Run button on this answer."""
    available, message = hp.umap_available()

    assert available is True
    assert message == ""


def test_an_unimportable_umap_is_reported_with_the_install_hint(monkeypatch):
    """The message is what the user is shown instead of a traceback."""
    import spacr.utils as utils

    class _Absent:
        def __getattr__(self, name):
            raise ImportError("No module named 'umap'")

    monkeypatch.setattr(utils, "umap", _Absent())

    assert hp.umap_available() == (False, hp.UMAP_MISSING_MESSAGE)


def test_an_incompatible_umap_reports_the_compatibility_problem(monkeypatch):
    """A version clash is a different fix from a missing package."""
    import spacr.utils as utils
    from spacr.utils import OptionalDependencyCompatibilityError

    class _Clashing:
        def __getattr__(self, name):
            raise OptionalDependencyCompatibilityError(
                "umap-learn needs numba < 0.62")

    monkeypatch.setattr(utils, "umap", _Clashing())

    available, message = hp.umap_available()

    assert available is False
    assert "numba" in message


# ---------------------------------------------------------------------------
# The attribution sweep's guards


def test_an_attribution_sweep_has_only_two_modes():
    """A third mode would silently become one of the two."""
    data = hp.ActivationSearchData(model=object(), images=[object()])

    with pytest.raises(ValueError, match="mode must be 'grid' or 'random'"):
        hp.activation_search(data, SearchSpace({"cam_type": ["saliency"]}),
                             mode="bayesian")


# ---------------------------------------------------------------------------
# Scoring an embedding, without fitting one


def _blobs(n_per=12, seed=0):
    """Three well-separated clusters in five dimensions, and their labels."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0, 0, 0, 0], [10.0, 0, 0, 0, 0],
                        [0.0, 10, 0, 0, 0]])
    features = np.vstack([c + rng.normal(scale=0.3, size=(n_per, 5))
                          for c in centres])
    labels = np.repeat([0, 1, 2], n_per)
    return features, labels


def test_a_faithful_embedding_scores_high_on_both_neighbourhood_criteria():
    """Trustworthiness and continuity are the same measure, spaces swapped."""
    features, labels = _blobs()
    embedding = features[:, :2]

    scores = hp._umap_scores(features, embedding, labels, k=15)

    assert scores["trustworthiness"] > 0.9
    assert scores["continuity"] > 0.9
    assert scores["silhouette"] > 0.5
    assert scores["neighbourhood_k"] == float(min(15, (len(features) - 1) // 2))


def test_labels_that_do_not_describe_the_rows_score_no_silhouette():
    """A silhouette over the wrong labels would be a number that means nothing."""
    features, _labels = _blobs()

    assert "silhouette" not in hp._umap_scores(features, features[:, :2],
                                               np.zeros(3), k=5)
    assert "silhouette" not in hp._umap_scores(features, features[:, :2],
                                               np.zeros(len(features)), k=5)


def test_umap_fits_the_embedding_the_search_scores(tmp_path):
    """The default embed function is the one a real sweep uses."""
    features, _labels = _blobs(n_per=8)

    embedding = hp._default_umap_embed(features, {"n_neighbors": 4,
                                                  "min_dist": 0.1}, seed=0)

    assert embedding.shape == (len(features), 2)
    assert np.isfinite(embedding).all()


# ---------------------------------------------------------------------------
# One attribution sweep, end to end on a hand-wired model


@pytest.fixture
def corner_model():
    """A model whose logit depends only on the top-left corner of the image."""
    import torch
    import torch.nn as nn

    class Corner(nn.Module):
        """Nothing is trained; the weights are set by hand."""

        def __init__(self):
            """Wire one identity convolution and a corner-only head."""
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
            """Score the top-left 3x3 corner only."""
            feature = self.conv(x)
            mask = torch.zeros_like(feature)
            mask[:, :, :3, :3] = 1.0
            return self.head((feature * mask).mean(dim=(2, 3)))

    image = torch.zeros(1, 16, 16)
    image[0, :3, :3] = 2.0
    return hp.ActivationSearchData(model=Corner().eval(), images=[image],
                                   filenames=["synthetic"],
                                   model_type="corner")


def test_a_random_attribution_sweep_samples_its_methods(corner_model):
    """`mode='random'` picks configurations instead of running the whole grid."""
    from spacr.attribution import attribute

    result = hp.activation_search(
        corner_model, SearchSpace({"cam_type": ["saliency", "gradcam"]}),
        mode="random", n_trials=1, n_steps=3, run_sanity_check=False,
        attribute_fn=lambda model, image, params: attribute(
            model, image, "saliency", model_type="corner"))

    assert len(result.trials) == 1
    assert result.higher_is_better is False
    assert any("no ground truth" in note for note in result.notes)
    assert any("sanity check was skipped" in note for note in result.notes)
