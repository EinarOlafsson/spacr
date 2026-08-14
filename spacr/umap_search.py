"""The Image UMAP search: a table of recipes, each of which redraws its map.

Instruction 95, taking starplast's shape.

    "during the search a table is formed with values and scores, i can click
     each row in this table to spawn a that rows umap (3d umap) and then i can
     luster that umap"

THE ROW IS THE RECIPE, and that is the whole design. starplast's
``EmbeddingSpec`` is one frozen record that round-trips, so a stored row can
rebuild EXACTLY the map it scored -- not a map with the same settings, the
same map. Anything else and clicking row 7 draws something that is not what
row 7's score describes, which nobody would notice.

WHICH BACKEND DREW IT IS PART OF THE RECIPE. cuML's UMAP is not umap-learn's:
it is a DIFFERENT MAP of the same data, not the same map faster. A table whose
rows came from both backends is comparing two libraries rather than the
settings the search varied, so ``backend`` is a field and not a footnote.

WHAT IS NOT HERE: Qt. This is the model the panel drives, so the search, the
table and the recipe are testable without a display -- which is the mistake
instruction 52 was reopened for, where the geometry was tested and the
controls were not.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("spacr.umap_search")

__all__ = [
    "UmapRecipe",
    "SearchRow",
    "SearchTable",
    "ClusterWalkRow",
    "cluster_embedding",
    "walk_clusters",
    "walk_recipes",
]


@dataclass(frozen=True)
class UmapRecipe:
    """Everything needed to redraw one embedding, and nothing else.

    Frozen and round-tripping, so a row saved to disk today rebuilds the same
    map tomorrow. ``columns`` is part of it: a recipe that recorded only the
    hyperparameters would rebuild a different map the moment the column
    selection changed, and the score beside it would then describe neither.
    """

    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 2
    metric: str = "euclidean"
    random_state: int = 42
    scale: bool = True
    columns: Tuple[str, ...] = ()
    backend: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "n_components",
                           max(2, min(3, int(self.n_components))))

    @property
    def is_3d(self) -> bool:
        return self.n_components >= 3

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "columns": list(self.columns)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UmapRecipe":
        data = {k: v for k, v in dict(payload).items()
                if k in cls.__dataclass_fields__}
        if "columns" in data:
            data["columns"] = tuple(data["columns"])
        return cls(**data)

    def label(self) -> str:
        """The short description a table cell shows."""
        return (f"n={self.n_neighbors} d={self.min_dist:g} "
                f"{self.n_components}D {self.backend}")


@dataclass
class SearchRow:
    """One trial: its recipe, its scores, and the embedding it produced.

    ``embedding`` is held so clicking the row is instant and, more to the
    point, so it is the SAME array that was scored. Recomputing on click
    would give a map that merely matches the recipe -- and with cuML, or any
    non-deterministic backend, would not even give that.
    """

    recipe: UmapRecipe
    scores: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    note: str = ""

    @property
    def score(self) -> float:
        """The headline number the table sorts on."""
        for key in ("score", "trustworthiness", "silhouette"):
            if key in self.scores:
                return float(self.scores[key])
        return float("nan")

    def cluster_count(self) -> int:
        """How many clusters this row's labels describe, noise excluded."""
        if self.labels is None:
            return 0
        values = np.asarray(self.labels)
        return int(len({int(v) for v in values if int(v) >= 0}))


class SearchTable:
    """The rows a search produced, in the order they were scored.

    Deliberately not a DataFrame: a row owns an ndarray and a recipe, and
    putting those in cells makes every operation on the table a chance to
    lose the pairing between a score and the embedding it describes.
    """

    def __init__(self) -> None:
        self._rows: List[SearchRow] = []

    def add(self, row: SearchRow) -> SearchRow:
        self._rows.append(row)
        return row

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, index: int) -> SearchRow:
        return self._rows[index]

    @property
    def rows(self) -> List[SearchRow]:
        return list(self._rows)

    def best(self) -> Optional[SearchRow]:
        """The highest-scoring row, or None when nothing scored.

        A row whose score is NaN is not "worst", it is UNSCORED, and letting
        it compare as a number would make the best row depend on how NaN
        happens to sort.
        """
        scored = [r for r in self._rows if not np.isnan(r.score)]
        return max(scored, key=lambda r: r.score) if scored else None

    def backends(self) -> Tuple[str, ...]:
        """Which backends drew these rows.

        More than one is worth saying out loud: a table mixing cuML and
        umap-learn rows is comparing two libraries as well as the settings.
        """
        return tuple(sorted({r.recipe.backend for r in self._rows}))

    def mixed_backends(self) -> bool:
        return len(self.backends()) > 1

    def to_dicts(self) -> List[Dict[str, Any]]:
        """The table without its arrays, for saving beside a run."""
        return [{"recipe": r.recipe.to_dict(), "scores": dict(r.scores),
                 "clusters": r.cluster_count(), "note": r.note}
                for r in self._rows]


@dataclass
class ClusterWalkRow:
    """One clustering tried against one fixed embedding.

    The coordinates are deliberately not stored here: a clustering walk
    changes the partition, not the map.  Keeping that distinction explicit
    prevents a cluster button from quietly refitting UMAP and making the row
    the user selected cease to be the row they are looking at.
    """

    min_cluster_size: int
    labels: np.ndarray
    silhouette: float
    n_clusters: int
    noise_fraction: float

    @property
    def score(self) -> float:
        """Ranking score: separation, discounted by unassigned points."""
        if not np.isfinite(self.silhouette):
            return float("-inf")
        return float(self.silhouette) * (1.0 - float(self.noise_fraction))


def _embedding_array(embedding: Any) -> np.ndarray:
    """Validate coordinates at the clustering/viewer boundary."""
    values = np.asarray(embedding, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        raise ValueError(
            "An Image UMAP embedding must have shape (rows, 2) or (rows, 3).")
    if len(values) < 3:
        raise ValueError("Clustering an Image UMAP needs at least 3 points.")
    if not np.isfinite(values).all():
        raise ValueError("The Image UMAP contains NaN or infinite coordinates.")
    return values


def cluster_embedding(
    embedding: Any,
    *,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    """Cluster a fixed 2-D or 3-D embedding with HDBSCAN.

    scikit-learn's implementation is used because it is already a spaCR core
    dependency (spaCR requires a version new enough to provide HDBSCAN). No
    DBSCAN substitution is made: changing the algorithm while keeping the
    HDBSCAN label would make the cluster count beside a map false provenance.
    """
    values = _embedding_array(embedding)
    size = int(min_cluster_size)
    if size < 2:
        raise ValueError("min_cluster_size must be at least 2.")
    if size >= len(values):
        raise ValueError(
            f"min_cluster_size must be smaller than the {len(values)} points.")
    samples = None if min_samples in (None, 0) else int(min_samples)
    if samples is not None and samples < 1:
        raise ValueError("min_samples must be at least 1 when supplied.")
    from sklearn.cluster import HDBSCAN

    estimator = HDBSCAN(
        min_cluster_size=size,
        min_samples=samples,
        store_centers=None,
        copy=True,
    )
    labels = np.asarray(estimator.fit_predict(values), dtype=int)
    if labels.shape != (len(values),):
        raise RuntimeError(
            "HDBSCAN returned one label count that does not match the map.")
    return labels


def walk_clusters(
    embedding: Any,
    *,
    min_cluster_sizes: Sequence[int] = (5, 10, 15, 25, 40),
    min_samples: Optional[int] = None,
) -> List[ClusterWalkRow]:
    """Try HDBSCAN scales on one map and return them best-first.

    This is the clustering half of the Starplast-style walk.  It can run for
    every UMAP trial as that trial arrives, or later against the table row the
    user chose.  Failed/oversized scales are skipped individually; if no scale
    is meaningful the result is empty rather than a fabricated one-cluster
    winner.
    """
    values = _embedding_array(embedding)
    candidates: List[int] = []
    for raw in min_cluster_sizes:
        try:
            size = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cluster-walk sizes must be whole numbers; got {raw!r}.") from exc
        if 2 <= size < len(values) and size not in candidates:
            candidates.append(size)
    if not candidates:
        raise ValueError(
            f"No cluster-walk size is between 2 and {len(values) - 1}.")

    from sklearn.metrics import silhouette_score

    rows: List[ClusterWalkRow] = []
    for size in candidates:
        labels = cluster_embedding(
            values, min_cluster_size=size, min_samples=min_samples)
        cluster_ids = sorted({int(value) for value in labels if value >= 0})
        keep = labels >= 0
        silhouette = float("nan")
        if len(cluster_ids) >= 2 and int(keep.sum()) > len(cluster_ids):
            try:
                silhouette = float(silhouette_score(values[keep], labels[keep]))
            except ValueError:
                silhouette = float("nan")
        rows.append(ClusterWalkRow(
            min_cluster_size=size,
            labels=labels,
            silhouette=silhouette,
            n_clusters=len(cluster_ids),
            noise_fraction=float(np.mean(labels < 0)),
        ))
    rows.sort(key=lambda row: (-row.score, row.min_cluster_size))
    return rows


def walk_recipes(base: UmapRecipe, *, steps: int = 12,
                 neighbors: Sequence[int] = (),
                 min_dists: Sequence[float] = (),
                 components: Sequence[int] = ()) -> List[UmapRecipe]:
    """The recipes a walk would try, worked out before any of them runs.

    Returned as a list rather than yielded one at a time so the panel can say
    how many trials there will be BEFORE the first one starts -- a progress
    bar whose denominator arrives at the end is not a progress bar.

    :param steps: how many to return when no explicit grid is given. The
        default grid walks ``n_neighbors``, which is the parameter that
        actually changes the shape of a UMAP; ``min_dist`` mostly changes how
        tightly it packs.
    """
    if not any((neighbors, min_dists, components)):
        low, high = 5, max(6, int(base.n_neighbors) * 4)
        neighbors = [int(round(v)) for v in
                     np.unique(np.linspace(low, high, max(2, int(steps))))]
    grid: List[UmapRecipe] = []
    for n in (neighbors or [base.n_neighbors]):
        for d in (min_dists or [base.min_dist]):
            for c in (components or [base.n_components]):
                grid.append(replace(base, n_neighbors=int(n),
                                    min_dist=float(d), n_components=int(c)))
    # Deduplicated, keeping order: a walk that scores the same recipe twice
    # spends the time and reports a second row that adds nothing.
    seen, unique = set(), []
    for recipe in grid:
        key = json.dumps(recipe.to_dict(), sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(recipe)
    return unique
