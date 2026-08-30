"""Five last branches in the run context, the UMAP search and hit review.

Each is a "leave it alone" arc: a log record that already has a run id, a
recipe stored without its column list, a clustering that found nothing to
score, a pair of reviewers with nothing in common. Doing the work anyway would
in each case overwrite or invent something.
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# runctx.install_run_id_logging — arc 284 -> 286, a record already stamped
# ---------------------------------------------------------------------------

def test_a_run_id_set_by_an_earlier_factory_is_not_overwritten():
    """The ``if not getattr(record, "run_id", ""):`` branch not taken.

    The docstring promises this function "chains onto whatever factory is
    already installed instead of replacing it". Chaining is only meaningful if
    the outer link leaves the inner one's work alone -- and a base factory
    that stamps run_id is exactly what a forwarded worker log, or another
    library's factory, looks like. Overwriting it would relabel a child run's
    lines with the parent's id, which is the one thing this stamping exists to
    prevent.
    """
    from spacr import runctx

    runctx.uninstall_run_id_logging()
    original = logging.getLogRecordFactory()

    def base_that_stamps(*args, **kwargs):
        record = original(*args, **kwargs)
        record.run_id = "set-by-the-worker"
        return record

    logging.setLogRecordFactory(base_that_stamps)
    try:
        runctx.install_run_id_logging()
        record = logging.getLogRecordFactory()(
            "spacr.test", logging.INFO, __file__, 1, "m", None, None)
        assert record.run_id == "set-by-the-worker"
    finally:
        runctx.uninstall_run_id_logging()
        logging.setLogRecordFactory(original)


def test_a_record_with_no_run_id_is_stamped_with_a_placeholder():
    """The taken side: never blank, so a log line always says which run."""
    from spacr import runctx

    runctx.install_run_id_logging()
    record = logging.getLogRecordFactory()(
        "spacr.test", logging.INFO, __file__, 1, "m", None, None)

    assert getattr(record, "run_id", "") != ""


# ---------------------------------------------------------------------------
# runctx.seed_everything — arc 691 -> 696, Cellpose not loaded
# ---------------------------------------------------------------------------

def test_a_process_without_cellpose_does_not_claim_to_have_seeded_it(monkeypatch):
    """The ``if "cellpose" in sys.modules:`` branch not taken.

    The entry is recorded so the report does not read as though Cellpose was
    overlooked. Recording it when Cellpose is NOT loaded would be the opposite
    error -- a seeding report claiming coverage of a library the process never
    imported, which is precisely the kind of false assurance the report exists
    to avoid.
    """
    from spacr import runctx

    monkeypatch.delitem(sys.modules, "cellpose", raising=False)
    report = runctx.seed_everything(1234)

    assert not [s for s in report.seeded if "cellpose" in s]
    assert report.seed == 1234


def test_a_process_with_cellpose_records_it_as_seeded_indirectly(monkeypatch):
    """The taken side, with a stand-in module so no heavy import is needed."""
    import types

    from spacr import runctx

    monkeypatch.setitem(sys.modules, "cellpose", types.ModuleType("cellpose"))
    report = runctx.seed_everything(1234)

    assert any("cellpose" in s for s in report.seeded)


# ---------------------------------------------------------------------------
# umap_search.UmapRecipe.from_dict — arc 70 -> 72, a payload with no columns
# ---------------------------------------------------------------------------

def test_a_stored_recipe_without_a_column_list_still_loads():
    """The ``if "columns" in data:`` branch not taken.

    Recipes are read back from run folders written by older versions, where
    the column list was not stored at all. Such a payload must load and keep
    the dataclass default rather than raising -- a KeyError here would make
    every historical run unopenable.
    """
    from spacr.umap_search import UmapRecipe

    recipe = UmapRecipe.from_dict({"n_neighbors": 30, "min_dist": 0.05})

    assert recipe.n_neighbors == 30
    assert recipe.min_dist == 0.05
    assert isinstance(recipe.columns, tuple)


def test_a_stored_column_list_comes_back_as_a_tuple():
    """The taken side: a list on disk becomes the tuple the dataclass wants."""
    from spacr.umap_search import UmapRecipe

    recipe = UmapRecipe.from_dict({"n_neighbors": 15,
                                   "columns": ["area", "perimeter"]})

    assert recipe.columns == ("area", "perimeter")


def test_keys_that_are_not_recipe_fields_are_ignored():
    """The comprehension above, which is what makes a forward-compatible read."""
    from spacr.umap_search import UmapRecipe

    recipe = UmapRecipe.from_dict({"n_neighbors": 15,
                                   "a_field_from_the_future": 1})

    assert recipe.n_neighbors == 15


# ---------------------------------------------------------------------------
# umap_search.walk_clusters — arc 277 -> 282, nothing worth scoring
# ---------------------------------------------------------------------------

def test_a_clustering_with_fewer_than_two_clusters_scores_nan():
    """The silhouette guard not taken.

    A silhouette needs at least two clusters and more points than clusters --
    scikit-learn raises otherwise. NaN is the honest answer for "this
    setting found nothing to compare", and it must not be 0.0, which on a
    silhouette means "clusters that overlap completely" and would rank a
    failed setting alongside a genuinely bad one.
    """
    from spacr.umap_search import walk_clusters

    # Eight identical points: no structure at any setting.
    embedding = np.zeros((8, 2), dtype=float)
    rows = walk_clusters(embedding, min_cluster_sizes=(5,))

    assert len(rows) == 1
    assert np.isnan(rows[0].silhouette)


# ---------------------------------------------------------------------------
# hit_investigation.evaluate_blinded_reviews — arc 239 -> 235, no overlap
# ---------------------------------------------------------------------------

def test_two_reviewers_who_share_no_reviews_contribute_no_kappa():
    """The ``if len(paired) >= 2:`` loop arc that skips.

    Cohen's kappa on fewer than two shared items is not a low agreement, it is
    no measurement. Blinded review assignments routinely leave a pair with
    nothing in common, and folding a degenerate value into the mean would make
    the reported agreement depend on how the work happened to be handed out.
    """
    from spacr.hit_investigation import evaluate_blinded_reviews

    reviews = pd.DataFrame({
        "review_id": ["r1", "r2", "r3", "r4"],
        "reviewer_id": ["a", "a", "b", "b"],     # disjoint review sets
        "reviewer_label": [1, 0, 1, 0],
    })
    key = pd.DataFrame({"review_id": ["r1", "r2", "r3", "r4"],
                        "hit_like_probability": [0.9, 0.1, 0.8, 0.2]})
    metrics = evaluate_blinded_reviews(reviews, key)

    value = metrics.get("mean_pairwise_cohen_kappa")
    assert value is None or np.isnan(value)


def test_reviewers_who_share_reviews_do_contribute_a_kappa():
    """The taken side, so the skip above is visibly a different outcome."""
    from spacr.hit_investigation import evaluate_blinded_reviews

    reviews = pd.DataFrame({
        "review_id": ["r1", "r2", "r3", "r1", "r2", "r3"],
        "reviewer_id": ["a", "a", "a", "b", "b", "b"],
        "reviewer_label": [1, 0, 1, 1, 0, 0],
    })
    key = pd.DataFrame({"review_id": ["r1", "r2", "r3"],
                        "hit_like_probability": [0.9, 0.1, 0.8]})
    metrics = evaluate_blinded_reviews(reviews, key)

    value = metrics.get("mean_pairwise_cohen_kappa")
    assert value is not None and not np.isnan(value)
