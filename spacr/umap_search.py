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
