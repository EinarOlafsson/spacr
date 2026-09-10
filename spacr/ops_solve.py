"""Absolute tile positions from the pairwise displacements one well produced.

THE THIRD VERB. :mod:`spacr.ops_layout` says which tiles touch,
:mod:`spacr.ops_register` says how far apart a touching pair is, and this
says where every tile ends up. Kept separate from both because
``ops_layout`` is pure geometry that imports only :mod:`math` and should
stay importable anywhere, and ``ops_register``'s whole subject is two tiles
and one shift.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple


def _displacement(value) -> Tuple[float, float]:
    """Read ``(dy, dx)`` from a tuple or from a :class:`Registration`.

    Accepting both keeps the caller free to pass what
    :func:`spacr.ops_register.register_pairs` returns without unpacking it
    first, and keeps this module testable without importing that one.

    :param value: a ``(dy, dx)`` pair, or anything carrying ``.shift``.
    :returns: the displacement as two floats.
    """
    shift = getattr(value, "shift", value)
    dy, dx = shift
    return (float(dy), float(dx))


def solve_placements(edges: Dict[Tuple[int, int], Tuple[float, float]],
                      sites: List[int],
                      ) -> Dict[int, Tuple[float, float]]:
    """Absolute tile positions from the pairwise displacements.

    A LEAST-SQUARES SOLVE, NOT A WALK. Chaining placements from a seed
    gives every tile the accumulated error of whatever path reached it,
    and on a round well the paths are long; solving all the edges at once
    spreads the residual instead and gives one answer no matter which
    tile is called the origin.

    The origin is pinned to the lowest-numbered site of each connected
    component, so a well that registers in two pieces still returns both
    rather than failing -- the caller can see the components in the
    result and say so.

    :param edges: ``(a, b) -> (dy, dx)``, b's position minus a's.
    :param sites: every site to place, including any with no edge.
    :returns: ``site -> (y, x)`` in pixels, one component pinned at the
        origin and the others pinned at their own lowest site.
    """
    import numpy as _np

    order = {site: index for index, site in enumerate(sorted(sites))}
    count = len(order)
    if not count:
        return {}
    # Components first, so each gets exactly one pin. Without that the
    # normal equations are singular for every component after the first.
    parent = list(range(count))

    def find(node: int) -> int:
        """The representative of ``node``'s component, path-compressed.

        :param node: a tile index.
        :returns: the index that stands for its connected component.

        Halving as it walks -- ``parent[node] = parent[parent[node]]`` --
        so a long chain costs its length once rather than on every later
        lookup. The components are what decide where the pins go, and a
        component that is found twice under two names puts two pins in
        one place and leaves another with none.
        """
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (left, right) in edges:
        if left in order and right in order:
            a, b = find(order[left]), find(order[right])
            if a != b:
                parent[a] = b
    pins = {}
    for site, index in sorted(order.items()):
        root = find(index)
        pins.setdefault(root, index)

    rows = len(edges) + len(pins)
    design = _np.zeros((rows, count), dtype=_np.float64)
    target = _np.zeros((rows, 2), dtype=_np.float64)
    for row, ((left, right), value) in enumerate(edges.items()):
        dy, dx = _displacement(value)
        design[row, order[left]] = -1.0
        design[row, order[right]] = 1.0
        target[row] = (dy, dx)
    for offset, index in enumerate(sorted(pins.values())):
        design[len(edges) + offset, index] = 1.0
    solution, *_ = _np.linalg.lstsq(design, target, rcond=None)
    return {site: (float(solution[index, 0]), float(solution[index, 1]))
            for site, index in order.items()}

