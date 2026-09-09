"""Where every tile of a round well is, in closed form.

THE ADJACENCY WAS THE OPEN PROBLEM AND THIS IS THE ANSWER. Instruction 372
spent three prototypes trying to RECONSTRUCT the grid from the links that
register -- a serpentine with a constant stride, then runs delimited by a
failed link, then columns as maximal runs of vertical links. All three
inferred the grid from the very edges that were missing, and all three
failed on the same acquisition: 27 of 333 tiles placed.

The well is ROUND. Each column holds as many fields as fit inside the
circle at that x, so the columns have DIFFERENT HEIGHTS and the index
offset to the next column is that column's height -- which is why the
measured horizontal offsets were +11, +4, -11, -9, -4, -13 and -5 rather
than one number. Fitting the circle takes four parameters and answers
every tile at once:

    columns   21, snaked down (even columns top to bottom)
    circle    radius 10.25 grid units, centre at column 10, row -10
    heights   5, 9, 13, 15, 17, 17, 19, 19, 21, 21, 21, 21, 21,
              19, 19, 17, 17, 15, 13, 9, 5                SUM = 333

IT IS A FACT ABOUT THE ACQUISITION, NOT A GOOD FIT. Seven measured
offsets against four free parameters would only be a fit; the model was
then asked to predict the four neighbours of six sites it had never seen
-- 20, 60, 140, 200, 260, 300 -- and registration confirmed 24 of 24, at
peak/mean 23.6 to 53.1 against control pairs at 9.0 to 15.6. The controls
came back at shift (0,0), which is the no-overlap signature behaving as
it should.

THE MISTAKE THAT COST THE FIRST MODEL FOUR OF SEVEN, recorded because it
is the same class of error as the wrapped shifts in 372's PART 6-A: rows
were indexed from each column's own top. In a round well every column
starts at a different row, so "row 3 of column c" and "row 3 of column
c+1" are not side by side. THE NEIGHBOUR IS AT THE SAME ABSOLUTE GRID
ROW, which is why :meth:`WellLayout.position` returns absolute rows and
:meth:`WellLayout.neighbours` looks them up directly.

WHAT THIS REPLACES. `max_site_gap` and the windowed pair search in
`spacr.spacrops`: 128 candidate pairs per tile, of which four could be
real, giving 993 scored pairs and 994 QC overlays for one well. This
module offers four candidates per tile and every one of the 24 tested is
a real adjacency, so about 640 real edges are scored instead of 993
mostly-imaginary ones.

Typical use::

    from spacr.ops_layout import round_well_layout

    layout = round_well_layout(333)
    for a, b, axis in layout.pairs():
        shift = register(tile(a), tile(b))    # phase correlation, per 372
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple

__all__ = [
    "WellLayout",
    "round_well_layout",
    "MEASURED_WELL",
]

#: The acquisition this model was fitted and then confirmed against:
#: ``(columns, radius, centre column, centre row)`` for well A1 of
#: `screenA/20200202_6W-LaC024A`, 333 sites at 10X.
MEASURED_WELL: Tuple[int, float, int, int] = (21, 10.25, 10, -10)

#: Microns between tile centres at 10X, from the reference implementation's
#: `plate_coordinate`. Measured against a 1267 px pitch, which puts the
#: scale at about 1.01 px/micron with a 213 px overlap on a 1480 px tile.
TILE_PITCH_UM: float = 1280.0

#: ``name -> (column step, row step)``. The four neighbours a tile can
#: have. Vertical steps continue a column; horizontal steps cross to the
#: next one, which is where the varying offset comes from.
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True)
class WellLayout:
    """A round well's tiles, addressed by site index and by grid position.

    :ivar columns: how many columns of fields the acquisition raster has.
    :ivar radius: the well's radius, in grid units (one tile pitch = 1).
    :ivar centre: ``(column, row)`` of the circle's centre. The row is
        negative for the measured well because site 0 is at the TOP and
        rows increase downwards.
    :ivar snake: whether the acquisition snakes -- odd columns collected
        bottom to top. Micro-Manager's HCS plugin does, which is why the
        reference implementation carries a `remap_snake` at all.
    """

    columns: int = MEASURED_WELL[0]
    radius: float = MEASURED_WELL[1]
    centre: Tuple[int, int] = (MEASURED_WELL[2], MEASURED_WELL[3])
    snake: bool = True

    # -- the geometry --------------------------------------------------
    def span(self, column: int) -> Optional[Tuple[int, int]]:
        """``(first row, last row)`` of one column, or None if it is empty.

        :param column: the column index.
        :returns: the inclusive absolute row range inside the circle.
        """
        if not 0 <= column < self.columns:
            return None
        dx = column - self.centre[0]
        remainder = self.radius * self.radius - dx * dx
        if remainder < 0:
            return None
        half = math.floor(math.sqrt(remainder))
        return self.centre[1] - half, self.centre[1] + half

    @property
    def heights(self) -> List[int]:
        """How many fields each column holds, left to right."""
        out = []
        for column in range(self.columns):
            span = self.span(column)
            out.append(0 if span is None else span[1] - span[0] + 1)
        return out

    @property
    def site_count(self) -> int:
        """How many tiles the well holds."""
        return len(_index(self)[0])

    # -- site index <-> grid position ----------------------------------
    def _walk(self) -> Iterator[Tuple[int, int]]:
        """Every ``(column, row)`` in acquisition order."""
        for column in range(self.columns):
            span = self.span(column)
            if span is None:
                continue
            top, bottom = span
            rows = range(top, bottom + 1)
            if self.snake and column % 2:
                rows = range(bottom, top - 1, -1)
            for row in rows:
                yield column, row

    def positions(self) -> List[Tuple[int, int]]:
        """``site -> (column, row)`` for every site, in index order."""
        return list(_index(self)[0])

    def position(self, site: int) -> Tuple[int, int]:
        """The ``(column, row)`` of one site.

        :param site: the site index.
        :returns: the column and the ABSOLUTE grid row -- not the row
            counted from this column's own top. See the module note.
        :raises IndexError: when the well holds no such site.
        """
        places = _index(self)[0]
        if not 0 <= site < len(places):
            raise IndexError(f"site {site} is outside a well of "
                             f"{len(places)}")
        return places[site]

    def site(self, column: int, row: int) -> Optional[int]:
        """The site index at one grid position, or None when it is empty.

        :param column: the column index.
        :param row: the absolute grid row.
        :returns: the site index, or None outside the circle.
        """
        return _index(self)[1].get((column, row))

    # -- adjacency -----------------------------------------------------
    def neighbours(self, site: int) -> Dict[str, int]:
        """The sites physically adjacent to one site.

        FOUR CANDIDATES, NOT 128. Every one of them is a real adjacency:
        this is what replaces `max_site_gap`'s window, which offered a
        band of index neighbours of which at most four could be touching.

        :param site: the site index.
        :returns: ``{direction: site}`` for the neighbours that exist.
        """
        column, row = self.position(site)
        index = _index(self)[1]
        found: Dict[str, int] = {}
        for name, (dcol, drow) in DIRECTIONS.items():
            other = index.get((column + dcol, row + drow))
            if other is not None:
                found[name] = other
        return found

    def pairs(self) -> List[Tuple[int, int, str]]:
        """Every adjacent pair once, as ``(first, second, axis)``.

        ONCE, not twice: registering a pair in both directions doubles the
        work and asks the same question. The axis is "vertical" or
        "horizontal", which the caller wants because a vertical link
        continues a column and a horizontal one crosses a snake turn --
        and because the two carry different expected shifts.

        ORDERED BY GEOMETRY, NOT BY SITE INDEX, and this is the one that
        will not fail loudly. The SECOND tile is always the one BELOW for
        a vertical pair and the one to the RIGHT for a horizontal one,
        because that is what a caller cropping an overlap band has to
        assume -- `register_edge` takes the bottom of the first and the
        top of the second. Ordering by index instead looks identical and
        is wrong on every odd column: the raster snakes, so in a
        bottom-to-top column the tile ABOVE carries the higher index. It
        cost half the edges of a toy well and reported itself as a
        residual of 0.00 px, because each surviving component still
        solved perfectly.

        :returns: the pairs, ordered by their first site.
        """
        index = _index(self)[1]
        seen: List[Tuple[int, int, str]] = []
        for place, site in index.items():
            column, row = place
            # DOWN AND RIGHT ONLY, which is what makes each pair appear
            # once AND puts the tiles in geometric order at the same time.
            for name in ("down", "right"):
                dcol, drow = DIRECTIONS[name]
                other = index.get((column + dcol, row + drow))
                if other is None:
                    continue
                axis = "vertical" if dcol == 0 else "horizontal"
                seen.append((site, other, axis))
        return sorted(seen)

    # -- the reference model -------------------------------------------
    def micron_position(self, site: int,
                        pitch: float = TILE_PITCH_UM) -> Tuple[float, float]:
        """Where the reference implementation says a tile is, in microns.

        The honest score for a stitch is the residual against THIS, not a
        count of how many tiles were placed -- 372's PART 6-A, and the
        reason `plate_coordinate` was read from the source rather than
        described.

        :param site: the site index.
        :param pitch: microns between tile centres.
        :returns: ``(x, y)`` in microns, relative to the circle's centre.
        """
        column, row = self.position(site)
        return ((column - self.centre[0]) * pitch,
                (row - self.centre[1]) * pitch)


@lru_cache(maxsize=32)
def _index(layout: "WellLayout"):
    """``(site -> place, place -> site)`` for one layout, walked once.

    The walk is the same answer every time and the callers ask for it per
    tile, so it is cached on the layout itself -- which is why the class
    is a FROZEN dataclass: it has to be hashable for this, and a layout
    that could be edited under its own cache would answer for the well it
    used to be.

    :param layout: the well.
    :returns: the ordered places, and the reverse map.
    """
    places = tuple(layout._walk())
    return places, {place: site for site, place in enumerate(places)}


def round_well_layout(site_count: int = 333,
                      columns: Optional[int] = None) -> WellLayout:
    """The layout whose circle holds exactly ``site_count`` fields.

    The measured well is returned unchanged for 333, so the confirmed
    model is never re-derived. For any other count the radius is searched
    -- a well imaged at a different magnification or a different plate
    format is the same circle with a different number of fields in it.

    :param site_count: how many tiles the acquisition holds.
    :param columns: the column count, when it is known. Derived from the
        circle otherwise.
    :returns: the layout.
    :raises ValueError: when no circle holds exactly that many fields,
        which is the honest answer -- a count that no round well produces
        means the acquisition is not one, and guessing the nearest would
        place every tile slightly wrong.
    """
    if site_count == 333 and columns in (None, MEASURED_WELL[0]):
        return WellLayout()
    if site_count < 1:
        raise ValueError("a well holds at least one field")
    # A circle of radius r spans 2r + 1 columns, so the radius is bounded
    # by the count itself; step finely enough that no integer span is
    # skipped between one radius and the next.
    step = 0.05
    radius = 0.5
    while radius <= site_count:
        width = 2 * math.floor(radius) + 1
        span = columns or width
        centre = (span // 2, -(span // 2))
        candidate = WellLayout(columns=span, radius=radius, centre=centre)
        if candidate.site_count == site_count:
            return candidate
        if candidate.site_count > site_count:
            break
        radius += step
    raise ValueError(
        f"no round well holds exactly {site_count} fields; the acquisition "
        "is not a circle, so its layout has to be measured rather than fitted")
