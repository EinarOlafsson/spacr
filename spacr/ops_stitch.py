"""One well, one cycle, from a folder of tiles to a table of positions.

THE FOURTH VERB, AND THE ONE THAT PRINTS. :mod:`spacr.ops_layout` says
which tiles touch, :mod:`spacr.ops_register` says how far apart a touching
pair is, :mod:`spacr.ops_solve` says where each tile ends up, and this
runs the three of them over a well and reports what happened.

THE OUTPUT IS COORDINATES, NOT PIXELS -- instruction 372's PART 6, the
maintainer's own design. A well of this acquisition is 26,855 x 26,865 px
and one is enough to exhaust a machine; the transform table is a few
kilobytes and every later phase reads through it. A canvas, if anybody
wants one, is rendered from this at whatever downsample suits the screen.

IT PRINTS THREE NUMBERS AND NOT TWO, and that is the whole reason this
module exists rather than three calls at a call site. The first end-to-end
run of this pipeline reported

    333 of 333 placed    residual median 0.2 px    canvas 35,374 x 35,367

and it was WRONG: the driver negated the correlation's shift, the unwrap
then chose the representative one period out, identically in every edge of
a direction, and the well came out one pitch per column too large in both
axes. Every edge still agreed with every other edge, so the residual was
perfect and the count was perfect. A UNIFORM ERROR IS INVISIBLE TO A
RESIDUAL. Only the canvas said otherwise, and only because 21 columns of a
known pitch has an arithmetic answer -- 20 x 1267 + 1480 = 26,820 -- to
check it against.

So :meth:`StitchedWell.summary` gives the count, the residual AND the
canvas in one line, and :attr:`StitchedWell.expected_canvas` carries what
the layout says the canvas should be, because a number with nothing to
compare it to is not a check.

Typical use::

    from spacr.ops_stitch import stitch_well

    well = stitch_well(read_tile, overlap=213, tolerance=8)
    print(well.summary())
    positions = well.placements          # site -> (y, x), in well pixels
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .ops_layout import WellLayout, round_well_layout

LOG = logging.getLogger(__name__)

__all__ = ["StitchedWell", "stitch_well"]


@dataclass
class StitchedWell:
    """What one well's stitch produced, and what it cost.

    :ivar placements: ``site -> (y, x)`` in well pixels, from the solve.
    :ivar edges: ``(a, b) -> Registration`` for every pair the layout
        proposed, ACCEPTED OR NOT. A refused pair is a fact about the
        acquisition and belongs in `ops_geometry`, not in a debug log.
    :ivar tile_shape: ``(height, width)`` of one tile.
    :ivar layout: the well model the pairs came from.
    :ivar overlap: the overlap in pixels the run was told to expect, or
        None when it was not told.
    """

    placements: Dict[int, Tuple[float, float]]
    edges: Dict[Tuple[int, int], object]
    tile_shape: Tuple[int, int]
    layout: WellLayout
    overlap: Optional[int] = None
    _residuals: Optional[np.ndarray] = field(default=None, repr=False)

    # -- what happened -------------------------------------------------
    @property
    def placed(self) -> int:
        """How many tiles got a position."""
        return len(self.placements)

    @property
    def accepted(self) -> int:
        """How many proposed pairs registered."""
        return sum(1 for one in self.edges.values()
                   if getattr(one, "accepted", True))

    @property
    def proposed(self) -> int:
        """How many pairs the layout offered."""
        return len(self.edges)

    @property
    def residuals(self) -> np.ndarray:
        """Per-edge disagreement between the solve and the measurement.

        The distance, in pixels, between where an accepted edge SAID its
        two tiles sit relative to each other and where the solve put
        them. Sub-pixel across a well is what says the geometry closed.
        """
        if self._residuals is None:
            self._residuals = self._measure_residuals()
        return self._residuals

    def _measure_residuals(self) -> np.ndarray:
        found = []
        for (a, b), one in self.edges.items():
            if not getattr(one, "accepted", True):
                continue
            if a not in self.placements or b not in self.placements:
                continue
            dy, dx = getattr(one, "shift", one)
            ay, ax = self.placements[a]
            by, bx = self.placements[b]
            found.append(float(np.hypot((by - ay) - dy, (bx - ax) - dx)))
        return np.asarray(found, dtype=float)

    @property
    def canvas(self) -> Tuple[int, int]:
        """``(height, width)`` the placed tiles span, in pixels."""
        if not self.placements:
            return (0, 0)
        ys = [y for y, _x in self.placements.values()]
        xs = [x for _y, x in self.placements.values()]
        height, width = self.tile_shape
        return (int(round(max(ys) - min(ys) + height)),
                int(round(max(xs) - min(xs) + width)))

    @property
    def expected_canvas(self) -> Tuple[int, int]:
        """What the layout says the canvas should be.

        THE ONLY THING THAT CAUGHT THE UNIFORM ERROR. Columns minus one
        pitches plus one tile, and the same down the tallest column. A
        measured canvas with nothing to compare it to is a number, not a
        check.
        """
        height, width = self.tile_shape
        pitch_y = height - (self.overlap or 0)
        pitch_x = width - (self.overlap or 0)
        columns = sum(1 for one in self.layout.heights if one)
        rows = max(self.layout.heights) if self.layout.heights else 0
        return (int(max(0, rows - 1) * pitch_y + height),
                int(max(0, columns - 1) * pitch_x + width))

    def summary(self) -> str:
        """The three numbers, on one line, because two of them lie alone."""
        residuals = self.residuals
        if residuals.size:
            middle = float(np.median(residuals))
            worst = float(residuals.max())
            residual = f"residual median {middle:.2f} px  max {worst:.2f} px"
        else:
            residual = "residual n/a (no accepted edge placed both tiles)"
        canvas = self.canvas
        expected = self.expected_canvas
        return (f"{self.placed} of {self.layout.site_count} placed  "
                f"{self.accepted}/{self.proposed} edges  {residual}  "
                f"canvas {canvas[0]:,} x {canvas[1]:,} "
                f"(layout says {expected[0]:,} x {expected[1]:,})")

    def canvas_agrees(self, tolerance: float = 0.01) -> bool:
        """Whether the measured canvas matches the layout's arithmetic.

        :param tolerance: allowed fractional difference on either axis.
            One per cent, because the layout's number is exact and the
            stitch's is not: it carries the real stage's skew.
        :returns: True when both axes agree within ``tolerance``.
        """
        measured, expected = self.canvas, self.expected_canvas
        for got, want in zip(measured, expected):
            if want <= 0:
                return False
            if abs(got - want) / want > tolerance:
                return False
        return True


def stitch_well(tiles, layout: Optional[WellLayout] = None, *,
                overlap: Optional[int] = None,
                tolerance: Optional[int] = None,
                skew: Optional[int] = None,
                gpu: bool = True,
                sites: Optional[Sequence[int]] = None,
                **kwargs) -> StitchedWell:
    """Register a well's tiles against each other and solve their positions.

    ONE WELL AT A TIME, AND ONE TILE PAIR AT A TIME WITHIN IT. ``tiles``
    is normally a CALLABLE, because 333 tiles of 1480 px is 2.9 GB and
    nothing here needs two of them resident: the caller decides what to
    keep and what to re-read.

    :param tiles: ``site -> 2-D array``, or a callable taking a site.
    :param layout: the well model. None fits the circle to ``sites`` or,
        failing that, to the measured 333-field well.
    :param overlap: the raster's overlap in pixels. Given, it sets the
        registration's expectation and the layout's canvas arithmetic;
        omitted, both fall back to weaker answers, so pass it.
    :param tolerance: how far ALONG the raster an edge may land from the
        layout's prediction and still be accepted. This is the acceptance
        the real well used -- 624 of 624 -- and without it the peak ratio
        decides, which on a real plate it cannot.
    :param skew: how far ACROSS it may. The stage's skew is real and
        constant -- 9 px on the measured plate -- so it is a separate
        number from the tolerance. None takes
        `spacr.ops_register.SKEW_PX`.
    :param gpu: passed through to the registration.
    :param sites: which sites to place. None uses every site the layout
        holds.
    :param kwargs: passed to :func:`spacr.ops_register.register_edge`.
    :returns: the stitch, its edges and its residuals.
    :raises ValueError: when the well holds no tiles at all, which is a
        caller error rather than an empty result.
    """
    from .ops_register import Registration, register_edge
    from .ops_solve import solve_placements

    if layout is None:
        layout = (round_well_layout(len(sites)) if sites is not None
                  else round_well_layout())
    every_site = list(sites) if sites is not None else list(
        range(layout.site_count))
    if not every_site:
        raise ValueError("a well with no sites cannot be stitched")

    read: Callable = tiles if callable(tiles) else tiles.__getitem__
    first = np.asarray(read(every_site[0]))
    if first.ndim != 2:
        raise ValueError(
            f"a tile is a 2-D field; site {every_site[0]} is {first.shape}. "
            "Pick the DAPI plane before stitching -- the geometry is solved "
            "on one channel and PART 10 says which.")
    tile_shape = (int(first.shape[0]), int(first.shape[1]))

    known = set(every_site)
    edges: Dict[Tuple[int, int], object] = {}
    for a, b, axis in layout.pairs():
        if a not in known or b not in known:
            continue
        try:
            edges[(a, b)] = register_edge(
                read(a), read(b), axis, expected_overlap=overlap,
                tolerance=tolerance, gpu=gpu,
                **({} if skew is None else {"skew": skew}), **kwargs)
        except Exception:                                # noqa: BLE001
            # ONE UNREADABLE TILE IS NOT A FAILED WELL -- and the pair is
            # RECORDED AS REFUSED rather than dropped. A pair that failed
            # is a fact about the acquisition and belongs in
            # `ops_geometry`; leaving it out of the table would make
            # "624 of 624 edges" mean two different things depending on
            # whether a file was readable.
            LOG.debug("could not register the pair %s-%s", a, b,
                      exc_info=True)
            edges[(a, b)] = Registration(dy=0, dx=0, peak_ratio=0.0,
                                         accepted=False, backend="none")

    accepted = {pair: one.shift for pair, one in edges.items()
                if getattr(one, "accepted", True)}
    placements = solve_placements(accepted, every_site)
    return StitchedWell(placements=placements, edges=edges,
                        tile_shape=tile_shape, layout=layout,
                        overlap=overlap)
