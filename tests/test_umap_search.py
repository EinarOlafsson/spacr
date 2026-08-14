"""The Image UMAP search table. Instruction 95.

    "during the search a table is formed with values and scores, i can click
     each row in this table to spawn a that rows umap (3d umap) and then i can
     luster that umap"

The row IS the recipe. A stored row must rebuild exactly the map it scored --
not a map with the same settings, the same map -- or clicking row 7 draws
something row 7's score does not describe, and nobody would notice.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr.umap_search import SearchRow, SearchTable, UmapRecipe, walk_recipes


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------

def test_a_recipe_round_trips_through_json():
    recipe = UmapRecipe(n_neighbors=30, min_dist=0.25, n_components=3,
                        columns=("a", "b"), backend="cuml")
    assert UmapRecipe.from_dict(json.loads(json.dumps(recipe.to_dict()))) == recipe


def test_the_columns_are_part_of_the_recipe():
    """A recipe recording only hyperparameters would rebuild a different map
    the moment the column selection changed, and the score beside it would
    then describe neither."""
    assert UmapRecipe(columns=("a",)) != UmapRecipe(columns=("a", "b"))


def test_the_backend_is_part_of_it_too():
    """cuML's UMAP is a different map of the same data, not the same map
    faster, so two rows differing only in backend are two different maps."""
    assert UmapRecipe(backend="cpu") != UmapRecipe(backend="cuml")


def test_components_are_clamped_to_two_or_three():
    """A UMAP nobody can draw is not a UMAP this module offers."""
    assert UmapRecipe(n_components=1).n_components == 2
    assert UmapRecipe(n_components=9).n_components == 3


def test_3d_is_reported_as_such():
    assert UmapRecipe(n_components=3).is_3d
    assert not UmapRecipe(n_components=2).is_3d


def test_a_recipe_is_frozen():
    with pytest.raises(Exception):
        UmapRecipe().n_neighbors = 99


def test_the_label_says_what_distinguishes_the_row():
    label = UmapRecipe(n_neighbors=30, min_dist=0.5, n_components=3,
                       backend="cuml").label()
    for part in ("30", "0.5", "3D", "cuml"):
        assert part in label


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def _row(n, score=0.5, backend="cpu", labels=None):
    return SearchRow(recipe=UmapRecipe(n_neighbors=n, backend=backend),
                     scores={"score": score},
                     embedding=np.zeros((4, 2)), labels=labels)


def test_rows_keep_the_order_they_were_scored_in():
    table = SearchTable()
    for n in (5, 30, 15):
        table.add(_row(n))
    assert [r.recipe.n_neighbors for r in table] == [5, 30, 15]


def test_the_embedding_travels_with_its_score():
    """Recomputing on click would give a map that merely MATCHES the recipe
    -- and with a non-deterministic backend, not even that."""
    row = _row(5)
    table = SearchTable()
    table.add(row)
    assert table[0].embedding is row.embedding


def test_best_ignores_unscored_rows():
    """NaN is not 'worst'; letting it compare as a number makes the best row
    depend on how NaN happens to sort."""
    table = SearchTable()
    table.add(_row(5, score=float("nan")))
    table.add(_row(30, score=0.2))
    assert table.best().recipe.n_neighbors == 30


def test_a_table_with_nothing_scored_has_no_best():
    table = SearchTable()
    table.add(_row(5, score=float("nan")))
    assert table.best() is None


def test_an_empty_table_has_no_best():
    assert SearchTable().best() is None


def test_mixed_backends_are_reported():
    """A table mixing cuML and umap-learn rows compares two libraries as
    well as the settings the search varied."""
    table = SearchTable()
    table.add(_row(5, backend="cpu"))
    table.add(_row(30, backend="cuml"))
    assert table.backends() == ("cpu", "cuml")
    assert table.mixed_backends()


def test_one_backend_is_not_mixed():
    table = SearchTable()
    table.add(_row(5))
    assert not table.mixed_backends()


def test_clusters_are_counted_without_noise():
    """HDBSCAN marks noise -1, and counting it as a cluster would report a
    partition nobody made."""
    row = _row(5, labels=np.array([0, 0, 1, -1, -1]))
    assert row.cluster_count() == 2


def test_a_row_with_no_labels_has_no_clusters():
    assert _row(5).cluster_count() == 0


def test_saving_drops_the_arrays_but_keeps_the_recipe():
    table = SearchTable()
    table.add(_row(5, labels=np.array([0, 1])))
    saved = table.to_dicts()
    assert saved[0]["recipe"]["n_neighbors"] == 5
    assert saved[0]["clusters"] == 2
    assert "embedding" not in saved[0]
    json.dumps(saved)          # the assertion: it is serialisable


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def test_the_grid_is_known_before_the_first_trial_runs():
    """A progress bar whose denominator arrives at the end is not one."""
    grid = walk_recipes(UmapRecipe(n_neighbors=15), steps=6)
    assert isinstance(grid, list) and len(grid) >= 2


def test_the_default_walk_varies_neighbours():
    """The parameter that changes the SHAPE of a UMAP; min_dist mostly
    changes how tightly it packs."""
    grid = walk_recipes(UmapRecipe(n_neighbors=15), steps=5)
    assert len({r.n_neighbors for r in grid}) == len(grid)
    assert len({r.min_dist for r in grid}) == 1


def test_an_explicit_grid_is_honoured():
    grid = walk_recipes(UmapRecipe(), neighbors=(5, 10),
                        min_dists=(0.0, 0.5), components=(3,))
    assert len(grid) == 4
    assert all(r.n_components == 3 for r in grid)


def test_duplicates_are_dropped():
    """A walk that scores the same recipe twice spends the time and reports
    a second row that adds nothing."""
    grid = walk_recipes(UmapRecipe(), neighbors=(5, 5, 5))
    assert len(grid) == 1


def test_the_walk_keeps_the_columns_and_backend_of_its_base():
    base = UmapRecipe(columns=("a", "b"), backend="cuml")
    for recipe in walk_recipes(base, steps=4):
        assert recipe.columns == ("a", "b")
        assert recipe.backend == "cuml"


def test_a_walk_of_one_step_still_returns_something():
    assert walk_recipes(UmapRecipe(), steps=1)
