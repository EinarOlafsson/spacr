"""Where every tile of a round well is, in closed form.

THE ADJACENCY WAS THE OPEN PROBLEM AND THIS IS THE ANSWER. Three earlier
prototypes tried to RECONSTRUCT the grid from the links that register -- a serpentine with a constant stride, then runs delimited by a
failed link, then columns as maximal runs of vertical links. All three
inferred the grid from the very edges that were missing, and all three
failed on the same acquisition: 27 of 333 tiles placed.

The well is ROUND. Each column holds as many fields as fit inside the
circle at that x, so the columns have DIFFERENT HEIGHTS and the index
offset to the next column is that column's height -- which is why the
measured horizontal offsets were +11, +4, -11, -9, -4, -13 and -5 rather
than one number. Fitting the circle takes four parameters and answers
every tile at once::

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
    :ivar half_tile: half a tile's side, in grid units. A site is kept
        when its whole tile lies inside the circle: at ``(dx, dy)`` from
        the centre, when ``(|dx| + half_tile)**2 + (|dy| + half_tile)**2
        <= radius**2``. Zero keeps every site whose centre is inside, which
        is the rule every layout followed before this field existed.

    The tile has to fit, not its centre. The centre rule holds the
    333-field sequencing well and cannot produce the column heights the
    1,281-field phenotype well was measured at, for any radius. With the
    footprint term both come out as measured, and every column is centred
    on the same row.
    """

    columns: int = MEASURED_WELL[0]
    radius: float = MEASURED_WELL[1]
    centre: Tuple[int, int] = (MEASURED_WELL[2], MEASURED_WELL[3])
    snake: bool = True
    half_tile: float = 0.0

    def span(self, column: int) -> Optional[Tuple[int, int]]:
        """``(first row, last row)`` of one column, or None if it is empty.

        :param column: the column index.
        :returns: the inclusive absolute row range of the sites whose whole
            tile lies inside the circle -- see :attr:`half_tile`.
        """
        if not 0 <= column < self.columns:
            return None
        across = abs(column - self.centre[0]) + self.half_tile
        remainder = self.radius * self.radius - across * across
        if remainder < 0:
            return None
        # At a half tile of zero this is floor(sqrt(R^2 - dx^2)) to the bit:
        # subtracting 0.0 moves no float, so the centre rule is unchanged.
        reach = math.sqrt(remainder) - self.half_tile
        if reach < 0:
            return None
        half = math.floor(reach)
        origin = self.centre[1]
        return origin - half, origin + half

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
            for name in ("down", "right"):
                dcol, drow = DIRECTIONS[name]
                other = index.get((column + dcol, row + drow))
                if other is None:
                    continue
                axis = "vertical" if dcol == 0 else "horizontal"
                seen.append((site, other, axis))
        return sorted(seen)

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


def _reach(radius: float, half_tile: float) -> int:
    """How many columns either side of the centre hold a whole tile.

    :param radius: the circle's radius, in pitches.
    :param half_tile: half a tile's side, in pitches.
    :returns: the largest ``|dx|`` at which a tile fits, or 0 when none
        does.

    At a half tile of zero this is ``floor(radius)`` to the bit, because
    ``sqrt(r * r) == r`` in binary floating point -- which keeps the centre
    rule's column count, and so its column indices, as they were.
    """
    remainder = radius * radius - half_tile * half_tile
    if remainder < 0:
        return 0
    return max(0, math.floor(math.sqrt(remainder) - half_tile))


def _candidate(radius: float, columns: Optional[int],
               half_tile: float) -> WellLayout:
    """The layout one radius gives, centred on its middle column.

    :param radius: the circle's radius, in pitches.
    :param columns: the column count when it is known, else None.
    :param half_tile: half a tile's side, in pitches.
    :returns: the layout, which may hold no field at all.
    """
    span = columns or 2 * _reach(radius, half_tile) + 1
    return WellLayout(columns=span, radius=radius,
                      centre=(span // 2, -(span // 2)), half_tile=half_tile)


def _exact_radius(site_count: int, half_tile: float,
                  above: float) -> Optional[float]:
    """A radius inside the window that holds exactly ``site_count`` fields.

    :param site_count: how many fields the acquisition holds.
    :param half_tile: half a tile's side, in pitches.
    :param above: a radius known to hold more than ``site_count``, which
        bounds the fields worth looking at.
    :returns: the middle of the window of radii holding exactly that many,
        or None when there is no such window -- fields entering the circle
        together, as a symmetric well's do in fours and eights.

    Needed because the search steps the radius by 0.05, and with the
    footprint term the radii holding one count can span a ten-thousandth
    of a pitch (372 PART 14-L: 1,281 fields at a half tile of 0.5005). So
    each grid position's own threshold ``hypot(|dx| + a, |dy| + a)`` is
    listed and the window read off directly.

    THE MIDDLE, NOT AN EDGE. Bisecting onto the edge was tried first and
    found wells no circle holds: at a threshold, the float arithmetic for
    ``(k, 0)`` and ``(0, k)`` can disagree in the last bit, and a radius
    that close keeps two of four mirror-image fields -- a "23-field well".
    A window narrower than float noise is refused for the same reason.
    """
    reach = int(math.ceil(above)) + 1
    entering: Dict[float, int] = {}
    for dx in range(reach + 1):
        for dy in range(dx, reach + 1):
            threshold = math.hypot(dx + half_tile, dy + half_tile)
            if threshold > above:
                continue
            copies = 1 if dx == dy == 0 else 4 if dx == 0 or dx == dy else 8
            entering[threshold] = entering.get(threshold, 0) + copies
    ordered = sorted(entering)
    held = 0
    for index, threshold in enumerate(ordered):
        held += entering[threshold]
        if held > site_count or index + 1 == len(ordered):
            return None
        if held == site_count:
            upper = ordered[index + 1]
            if upper - threshold <= 1e-9 * max(1.0, upper):
                return None
            return (threshold + upper) / 2
    return None


def round_well_layout(site_count: int = 333,
                      columns: Optional[int] = None,
                      half_tile: float = 0.5857) -> WellLayout:
    """The layout whose circle holds exactly ``site_count`` fields.

    The measured well is returned unchanged whenever the search lands on
    its 333 sites, so the confirmed model is never re-derived. For any
    other count the radius is searched -- a well imaged at a different
    magnification or a different plate format is the same circle with a
    different number of fields in it.

    :param site_count: how many tiles the acquisition holds.
    :param columns: the column count, when it is known. Derived from the
        circle otherwise.
    :param half_tile: half a tile's side in tile pitches -- the tile size
        over twice the registered pitch. A field is kept when its whole
        tile lies inside the circle; zero keeps it when its centre does.
        The default is the phenotype acquisition's measured 0.5857. The
        sequencing acquisition measured 0.5840, and every value strictly
        between 0.5 and 2.0 gives both wells -- 333 and 1,281 fields -- the
        column heights they were measured at.
    :returns: the layout.
    :raises ValueError: when no circle holds exactly that many fields,
        which is the honest answer -- a count that no round well produces
        means the acquisition is not one, and guessing the nearest would
        place every tile slightly wrong. Also when ``half_tile`` is
        negative or not a number.

    For one half tile, the count fixes the fields. The number of fields
    inside the circle only grows with the radius, so every radius that
    holds exactly ``site_count`` holds the same ones. The radius the layout
    carries is one of those, not a measurement.
    """
    if site_count < 1:
        raise ValueError("a well holds at least one field")
    half_tile = float(half_tile)
    if not half_tile >= 0.0:
        raise ValueError(
            f"a half tile of {half_tile} pitches is no footprint; it is half "
            "the tile's side over the pitch, so zero or more")
    measured = WellLayout()
    step = 0.05
    radius = 0.5
    while radius <= site_count:
        candidate = _candidate(radius, columns, half_tile)
        held = sum(candidate.heights)
        if held > site_count:
            inside = _exact_radius(site_count, half_tile, radius)
            if inside is None:
                break
            candidate = _candidate(inside, columns, half_tile)
            if sum(candidate.heights) != site_count:
                break
            held = site_count
        if held == site_count:
            if (site_count == measured.site_count
                    and candidate.positions() == measured.positions()):
                return measured
            return candidate
        radius += step
    raise ValueError(
        f"no round well holds exactly {site_count} fields at a half tile of "
        f"{half_tile}; the acquisition is not a circle, so its layout has to "
        "be measured rather than fitted")
