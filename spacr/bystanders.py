"""Which uninfected cells are next to an infected one, and which are not.

THE COLUMN THAT DOES NOT EXIST YET. spaCR measures each cell on its own,
and the biology of an intracellular pathogen is not confined to the cell it
is in: an infected cell changes what its neighbours do. Right now an
uninfected cell touching an infected one and an uninfected cell on the far
side of the well are the same row.

TWO CONSEQUENCES, and the second is the one that costs results. A genuine
bystander phenotype cannot be found, because no column separates a
bystander from a distant uninfected cell. And every uninfected control is
quietly diluted: if bystanders are phenotypically shifted and they are
pooled into "uninfected", the control population is a mixture, its variance
is inflated, and the effect size of every infection comparison shrinks
towards nothing. That is a confound in results the package already
produces, not a missing feature.

DISTANCE IS ALREADY SOLVED AND NOTHING HERE RE-IMPLEMENTS IT.
:mod:`spacr.object_distances` computes a Euclidean distance transform per
object type once, at O(image pixels) rather than O(pairs), and this module
reads that field. The work here is a threshold and a label.

THE THRESHOLD IS A LENGTH, NOT A PIXEL COUNT. It is expressed as a
multiple of the measured cell diameter, so it means the same thing on a
20x and a 63x acquisition and on a plate whose cells are simply larger.
A hard-coded micron count would be right for one dataset and silently
wrong for the next.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .object_distances import surface_distance_transform

#: What a cell's neighbourhood is called, in the order a reader meets them.
#:
#: THREE STATES AND NOT TWO. "infected" and "uninfected" is the split the
#: package has; the whole point of this module is that the second of those
#: is two populations. `distal` is the honest name for what "uninfected"
#: used to mean and now does not.
STATUSES: tuple = ("infected", "bystander", "distal")

#: How far an infected cell's influence is assumed to reach, as a multiple
#: of the measured cell diameter.
#:
#: ONE CELL DIAMETER, WHICH IS THE MOST DEFENSIBLE DEFAULT AND NOT A
#: MEASURED ONE. A cell whose surface is within one diameter of an infected
#: cell's surface is, at most, one cell away from it. That is a statement
#: about adjacency rather than about diffusion, and it is the claim this
#: module can actually support: the reach of a secreted effector is a
#: quantity somebody would have to measure per effector, per medium, per
#: time point.
#:
#: It is a SETTING for that reason. The default has to be honest about
#: being a convention; the number that matters is the user's.
DEFAULT_REACH_IN_DIAMETERS: float = 1.0


def reach_from_diameter(cell_diameter: float,
                        diameters: float = DEFAULT_REACH_IN_DIAMETERS
                        ) -> float:
    """The bystander reach in the units ``cell_diameter`` is measured in.

    :param cell_diameter: the plate's measured cell diameter, from
        :mod:`spacr.diameter`.
    :param diameters: how many diameters count as neighbouring.
    :returns: a distance in the same units, never negative.

    Kept as its own function so the conversion is visible in one place and
    a caller can see that the reach is derived rather than chosen.
    """
    try:
        size = float(cell_diameter)
        scale = float(diameters)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(size) or not np.isfinite(scale):
        return 0.0
    return max(0.0, size * scale)


def distance_to_infected(cell_mask, infected_labels: Iterable[int], *,
                         spacing=None) -> pd.DataFrame:
    """How far each cell's surface is from the nearest INFECTED cell.

    :param cell_mask: the cell label image.
    :param infected_labels: labels of the cells carrying a pathogen.
    :param spacing: voxel size, so the distances carry physical units.
    :returns: ``label`` and ``distance_to_infected``, one row per cell.

    ZERO FOR AN INFECTED CELL, ONE PIXEL FOR ITS IMMEDIATE NEIGHBOUR, and
    the distinction is worth stating because the obvious reading is wrong.
    The field is zero on any INFECTED pixel and grows outward, so a cell
    that shares a border with an infected one has its own nearest pixel one
    step away: the minimum over that cell is 1, not 0. Only a cell whose
    own pixels are infected reads 0.

    `surface_distance_transform`'s docstring says "two touching objects
    come out at 0", and that is true of the case it describes -- reading
    the field at a point INSIDE the other object, which is what an
    overlapping or contained object gives. Two disjoint labels that merely
    abut do not overlap, so they do not. A threshold expressed in cell
    diameters is far larger than one pixel either way; the note is here so
    nobody reads a 1 as a bug.

    ``inf`` WHEN THERE IS NOTHING TO BE NEAR. A well with no infected cell
    has no distances, and the honest value is infinity rather than 0 or
    NaN: every cell is arbitrarily far from a population that is not there,
    and infinity is the only value that keeps `distance <= reach` False for
    every finite reach.
    """
    labelled = np.asarray(cell_mask)
    labels = np.unique(labelled)
    labels = labels[labels != 0]
    if not labels.size:
        return pd.DataFrame({"label": np.array([], dtype=labelled.dtype),
                             "distance_to_infected": np.array([], dtype=float)})

    wanted = {int(v) for v in infected_labels}
    infected = np.isin(labelled, list(wanted)) if wanted else np.zeros_like(
        labelled, dtype=bool)
    if not infected.any():
        return pd.DataFrame({
            "label": labels,
            "distance_to_infected": np.full(labels.size, np.inf, dtype=float)})

    field = surface_distance_transform(infected, spacing=spacing)
    # THE MINIMUM OVER THE CELL, NOT THE VALUE AT ITS CENTROID. A large or
    # bent cell can have its centroid far from an infected neighbour while
    # its membrane touches one, and it is the membrane that is exposed.
    # `labeled_comprehension` walks each label once.
    from scipy.ndimage import labeled_comprehension

    smallest = labeled_comprehension(
        field, labelled, labels, np.min, np.float64, np.inf)
    return pd.DataFrame({"label": labels,
                         "distance_to_infected": np.asarray(smallest,
                                                            dtype=float)})


def classify(cell_mask, infected_labels: Iterable[int], *, reach: float,
             spacing=None) -> pd.DataFrame:
    """Label every cell infected, bystander or distal.

    :param cell_mask: the cell label image.
    :param infected_labels: labels of the cells carrying a pathogen.
    :param reach: how close an uninfected cell must be to count as a
        bystander, in the same units as ``spacing``. See
        :func:`reach_from_diameter`.
    :param spacing: voxel size.
    :returns: ``label``, ``distance_to_infected`` and ``neighbourhood``.

    A NON-POSITIVE REACH MAKES EVERY UNINFECTED CELL DISTAL, which is the
    behaviour that lets a user turn the split off without a second setting
    -- and the behaviour a mis-parsed setting falls back to, since it
    cannot invent bystanders.

    THE ACCEPTANCE TEST THIS HAS TO PASS is that a well with no parasites
    reports no bystanders. It follows from `distance_to_infected` being
    infinite there rather than from a special case, which is why that
    function returns infinity.
    """
    frame = distance_to_infected(cell_mask, infected_labels, spacing=spacing)
    wanted = {int(v) for v in infected_labels}
    try:
        limit = float(reach)
    except (TypeError, ValueError):
        limit = 0.0
    if not np.isfinite(limit):
        limit = 0.0

    status = np.full(len(frame), "distal", dtype=object)
    if len(frame):
        is_infected = frame["label"].isin(wanted).to_numpy()
        near = (frame["distance_to_infected"].to_numpy() <= limit) & (limit > 0)
        status[near] = "bystander"
        status[is_infected] = "infected"
    frame["neighbourhood"] = status
    return frame


def counts(frame: pd.DataFrame) -> Dict[str, int]:
    """How many cells landed in each state, including the empty ones.

    :param frame: the output of :func:`classify`.
    :returns: every name in :data:`STATUSES`, zero where absent.

    EVERY KEY ALWAYS PRESENT. A caller comparing wells needs a zero rather
    than a missing key, because "no bystanders in this well" is a result
    and a KeyError is not.
    """
    seen = frame["neighbourhood"].value_counts().to_dict() if len(frame) else {}
    return {name: int(seen.get(name, 0)) for name in STATUSES}
