"""Assemble the tiles of one field into the one image the field really is.

spaCR's filename convention --
``plate_well_T####F###L##A##Z##C##`` -- has no tile slot, so a field split
into four tiles produces four images with ONE canonical name and three of
them would be overwritten. Three answers were possible: grow the convention a
tile slot, give each tile its own field number, or put the field back
together. spaCR stitches at import, with the option to turn it off, and that
is the right answer: a stitched field IS one image with one name, so nothing
downstream has to learn what a tile is.

WHAT MAKES THIS HARD IS THAT A TILE INDEX IS NOT A POSITION. ``tile03`` says
an image is the third of some number; it does not say where it goes, how the
tiles are ordered, or how far they overlap. Every one of those has to come
from evidence, and this module is arranged around where the evidence is:

1. THE FILE'S OWN STAGE COORDINATES, when it has them. An OME-TIFF records
   ``PositionX``/``PositionY`` in microns and the pixel size beside them, so
   the layout is not inferred at all -- it is read. Nothing else is as good.
2. THE PIXELS, otherwise. Adjacent tiles overlap, and the overlap is the
   same picture twice, so the right displacement is the one whose implied
   overlap correlates -- scored for EVERY admissible displacement rather
   than picked from a phase-correlation peak, which a small overlap buries.
   This also settles the ORDER -- row-major and serpentine put different
   tiles beside each other, so the arrangement that correlates is the
   arrangement the microscope used.
3. NOTHING, in which case it says so. Blank tiles, single-pixel overlaps and
   images with no shared content produce no correlation to speak of. The
   tiles are then butt-joined in a square-ish grid and the mosaic is marked
   ``assumed`` with its confidence, because a seam in an image a user was
   told about is a different thing from one they were not.

THE FAILURE THIS EXISTS TO AVOID is the one the whole import module was
written against: an answer that looks plausible and is wrong. A mosaic
assembled at the wrong overlap is a field with a duplicated band through it,
every measurement in that band counted twice, and nothing on screen saying
so. So the confidence travels with the result, and a caller that cares can
refuse it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ARRANGEMENTS",
    "Mosaic",
    "Placement",
    "arrangement_of",
    "grid_shape",
    "plan_mosaic",
    "read_stage_positions",
    "stitch_tiles",
]

#: How a microscope may have walked the grid, as ``name -> (row, column)``
#: for tile ``index`` (0-based) in a grid of ``cols`` columns.
#:
#: SERPENTINE IS NOT AN EXOTIC CASE. A stage that returns to the first column
#: at the end of every row wastes the traverse, so many acquisitions snake --
#: and a row-major reading of a snaked field puts the third tile where the
#: fourth belongs, which is a mosaic that is wrong by one tile and looks
#: almost right. The arrangement is chosen by correlation rather than
#: assumed; this table is only the list of what to try.
ARRANGEMENTS: Dict[str, str] = {
    "row_major": "left to right, row by row",
    "serpentine_rows": "left to right, then right to left, row by row",
    "column_major": "top to bottom, column by column",
    "serpentine_columns": "top to bottom, then bottom to top, column by column",
}

#: Fraction of a tile edge below which an overlap is not worth believing.
#: Two tiles sharing four pixels correlate on almost anything.
MIN_OVERLAP_FRACTION = 0.02

#: Fraction above which a claimed overlap is rejected: tiles that overlap by
#: more than this are not a mosaic, they are the same field imaged twice.
MAX_OVERLAP_FRACTION = 0.6

#: Normalised cross-correlation over the overlap, below which a placement is
#: not believed and the grid is butt-joined instead.
MIN_CONFIDENCE = 0.35

@dataclass(frozen=True)
class Placement:
    """Where one tile goes in the mosaic.

    :param tile: the tile's own index, as the filename gave it.
    :param row: its row in the grid.
    :param col: its column in the grid.
    :param y: top edge in mosaic pixels.
    :param x: left edge in mosaic pixels.
    """

    tile: object
    row: int
    col: int
    y: int
    x: int


@dataclass(frozen=True)
class Mosaic:
    """A plan for one field: where every tile goes, and how that was decided.

    :param placements: one per tile, in tile order.
    :param height: mosaic height in pixels.
    :param width: mosaic width in pixels.
    :param rows: rows in the grid.
    :param cols: columns in the grid.
    :param arrangement: which key of :data:`ARRANGEMENTS` was used.
    :param how: ``"single"`` (one tile, nothing to place), ``"stage"``,
        ``"correlated"`` or ``"assumed"`` -- the evidence the placement
        rests on, in descending order of trust.
    :param confidence: mean normalised cross-correlation over the seams, 0
        when nothing was correlated.
    :param overlap: ``(y, x)`` overlap in pixels between neighbours.
    """

    placements: Tuple[Placement, ...]
    height: int
    width: int
    rows: int
    cols: int
    arrangement: str
    how: str
    confidence: float = 0.0
    overlap: Tuple[int, int] = (0, 0)

    @property
    def is_believed(self) -> bool:
        """Whether the placement rests on evidence rather than on a guess."""
        return self.how in ("single", "stage", "correlated")

    def describe(self) -> str:
        """One sentence a user can act on."""
        if self.how == "single":
            return "One tile; nothing to stitch."
        if self.how == "stage":
            return (f"{len(self.placements)} tiles placed from the stage "
                    f"coordinates in the files ({self.rows}x{self.cols}).")
        if self.how == "correlated":
            return (f"{len(self.placements)} tiles stitched "
                    f"{self.rows}x{self.cols}, {ARRANGEMENTS[self.arrangement]}, "
                    f"overlapping {self.overlap[1]}x{self.overlap[0]} px "
                    f"(confidence {self.confidence:.2f}).")
        return (f"{len(self.placements)} tiles butt-joined {self.rows}x"
                f"{self.cols} because nothing in them correlates -- the seams "
                f"are unverified and the field may be wrong by an overlap.")


def grid_shape(count: int) -> Tuple[int, int]:
    """``(rows, cols)`` for ``count`` tiles, as square as it can be.

    THE SHAPE IS A GUESS AND THE ORDER IS NOT. A 6-tile field is 2x3 or 3x2
    and nothing in a tile index says which, so the squarest grid is taken and
    both arrangements are then scored against the pixels -- a 3x2 read as 2x3
    correlates badly, which is how the wrong one is found.
    """
    if count <= 0:
        return (0, 0)
    rows = int(round(count ** 0.5))
    while rows > 1 and count % rows:
        rows -= 1
    return (rows, count // rows)


def arrangement_of(index: int, rows: int, cols: int,
                   arrangement: str) -> Tuple[int, int]:
    """``(row, col)`` for the ``index``-th tile under one arrangement."""
    if arrangement == "row_major":
        return divmod(index, cols)
    if arrangement == "serpentine_rows":
        row, col = divmod(index, cols)
        return (row, col if row % 2 == 0 else cols - 1 - col)
    if arrangement == "column_major":
        col, row = divmod(index, rows)
        return (row, col)
    if arrangement == "serpentine_columns":
        col, row = divmod(index, rows)
        return (row if col % 2 == 0 else rows - 1 - row, col)
    raise ValueError(f"unknown arrangement {arrangement!r}")


# ---------------------------------------------------------------------------
# 1. The file's own stage coordinates
# ---------------------------------------------------------------------------

def read_stage_positions(paths: Sequence) -> Optional[List[Tuple[float, float]]]:
    """``(y, x)`` in PIXELS for each path, or None when any file lacks them.

    ALL OR NOTHING, deliberately. Half a mosaic placed from stage coordinates
    and half from correlation is two coordinate systems in one image, and the
    join between them is exactly where nothing checks.

    Returns None rather than raising for a file that cannot be read: a
    missing position is the ordinary case, not an error, and the caller has
    a second method to fall back to.
    """
    try:
        import tifffile
    except Exception:                                    # pragma: no cover
        return None

    found: List[Tuple[float, float]] = []
    for path in paths:
        try:
            with tifffile.TiffFile(str(path)) as handle:
                position = _position_from(handle)
        except Exception:                                # noqa: BLE001
            return None
        if position is None:
            return None
        found.append(position)
    return found


def _position_from(handle) -> Optional[Tuple[float, float]]:
    """``(y, x)`` in pixels for one open TiffFile, or None.

    OME carries the stage position in microns and the pixel size beside it,
    so the two together give pixels. Either one alone gives nothing usable --
    a position in microns cannot be placed in an image without knowing how
    big a pixel is, and assuming a pixel size is how a mosaic ends up scaled
    by a factor nobody notices.
    """
    metadata = getattr(handle, "ome_metadata", None)
    if not metadata:
        return None
    import re

    def _first(pattern: str) -> Optional[float]:
        """The first number ``pattern`` captures in the OME block, or None.

        The XML is read with a regular expression rather than parsed because
        one attribute is wanted from a document whose schema version varies,
        and a parser that must know the namespace fails on the versions it
        has not been told about.
        """
        found = re.search(pattern, metadata)
        return float(found.group(1)) if found else None

    x_um = _first(r'PositionX="([-\d.eE+]+)"')
    y_um = _first(r'PositionY="([-\d.eE+]+)"')
    px_x = _first(r'PhysicalSizeX="([-\d.eE+]+)"')
    px_y = _first(r'PhysicalSizeY="([-\d.eE+]+)"')
    if None in (x_um, y_um, px_x, px_y) or not px_x or not px_y:
        return None
    return (y_um / px_y, x_um / px_x)


# ---------------------------------------------------------------------------
# 2. The pixels
# ---------------------------------------------------------------------------

def _sliding_ncc(first, second):
    """Zero-mean NCC for every displacement, computed for all at once.

    ``result[d]`` scores the hypothesis "the second tile sits ``d`` pixels to
    the right of the first", over exactly the columns the two would then
    share. Every admissible displacement is scored -- there is no peak
    picking and nothing to miss.

    WHY NOT PHASE CORRELATION, which is what a stitcher usually reaches for:
    its peak height goes with the AREA the two tiles share, and a 6% overlap
    -- an ordinary acquisition setting -- puts the true peak below a dozen
    peaks raised by nothing but noise. Measured on a 64 px tile with a 4 px
    overlap: the correct shift scored 1.00 on the overlap it implies and did
    not appear in the top twelve phase peaks. Scoring the overlap is the test
    that cannot be fooled, so it is the only test used.

    IT IS STILL CHEAP, which is why it can be exhaustive. The numerator is a
    cross-correlation, so one FFT per row gives every displacement together;
    the means and variances of the two windows come from prefix sums along
    the row. The whole thing is O(W log W) per row rather than O(W^2).
    """
    import numpy as np

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    rows, width = a.shape
    size = 1
    while size < 2 * width:
        size *= 2
    cross = np.fft.irfft(np.fft.rfft(a, n=size, axis=1)
                         * np.conj(np.fft.rfft(b, n=size, axis=1)),
                         n=size, axis=1).sum(axis=0)[:width]

    def _prefix(values):
        """Running totals with a leading zero, so ``p[b] - p[a]`` is the sum
        over ``[a, b)`` for every window without a loop over windows."""
        return np.concatenate([[0.0], np.cumsum(values)])

    sum_a = _prefix(a.sum(axis=0))
    sum_aa = _prefix((a * a).sum(axis=0))
    sum_b = _prefix(b.sum(axis=0))
    sum_bb = _prefix((b * b).sum(axis=0))

    displacements = np.arange(width)
    overlap = width - displacements
    count = rows * overlap
    # A contributes its RIGHT-hand columns and B its LEFT-hand ones: that is
    # what "B sits d to the right" means, and getting it the other way round
    # scores every pair against the wrong half of itself.
    total_a = sum_a[width] - sum_a[displacements]
    total_aa = sum_aa[width] - sum_aa[displacements]
    total_b = sum_b[overlap] - sum_b[0]
    total_bb = sum_bb[overlap] - sum_bb[0]

    numerator = cross - total_a * total_b / count
    variance_a = total_aa - total_a ** 2 / count
    variance_b = total_bb - total_b ** 2 / count
    denominator = np.sqrt(np.clip(variance_a, 0, None)
                          * np.clip(variance_b, 0, None))
    scores = np.zeros(width)
    usable = denominator > 0
    scores[usable] = numerator[usable] / denominator[usable]
    return scores


def _score_overlap(first, second, dy: int, dx: int) -> float:
    """Zero-mean NCC of the region two tiles would share at ``(dy, dx)``.

    The full-resolution verdict on a displacement the search proposed from a
    sample of rows. Overlapping regions of one field are the same photons
    twice and correlate near 1; a displacement wrong by a tile lands on
    unrelated pixels and correlates near 0.

    Returns 0 for an overlap too small to mean anything -- four pixels
    correlate on almost any pair of images.
    """
    import numpy as np

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    height, width = a.shape[:2]
    y0, x0 = max(dy, 0), max(dx, 0)
    y1 = min(height, height + dy)
    x1 = min(width, width + dx)
    if y1 - y0 < 2 or x1 - x0 < 2:
        return 0.0
    left = a[y0:y1, x0:x1]
    right = b[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    if left.size < MIN_OVERLAP_FRACTION * height * width:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt((left ** 2).sum() * (right ** 2).sum()))
    if denominator <= 0:
        return 0.0
    return float((left * right).sum() / denominator)


#: Rows (or columns) sampled for the displacement search. A translation
#: along x is as visible in a hundred rows of a tile as in two thousand, and
#: sampling is what keeps a 2048 px tile as quick to place as a 64 px one.
SEARCH_LINES = 128

#: How far a neighbour may be displaced ACROSS the axis it neighbours on.
#: Stage jitter is a few pixels; anything larger is not jitter but the wrong
#: pair, and searching further would find a spurious match eventually.
MAX_JITTER = 32


def _pair_shift(first, second, axis: str) -> Tuple[Tuple[int, int], float]:
    """The best believed ``((dy, dx), score)`` for one adjacent pair.

    ``axis`` is ``"x"`` for a left/right pair and ``"y"`` for a top/bottom
    one. Knowing which is what makes this tractable: neighbours in a mosaic
    are displaced along ONE axis by most of a tile, so the search is a line
    rather than a plane.

    THE PERPENDICULAR OFFSET CANNOT BE LEFT UNTIL AFTERWARDS, which is what
    the first version did. A stage misses its row by a few pixels every
    field, and two tiles compared with their rows three pixels out of step
    correlate on almost nothing -- so the along-axis search run at offset
    zero picks a displacement out of the noise, and the pair is then written
    off as unmeasurable. The offset is therefore part of the search: coarsely
    here, where each candidate costs one FFT pass, and then exactly at full
    resolution around the winner.
    """
    import numpy as np

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if axis == "y":
        a, b = a.T, b.T
    lines, span = a.shape
    step = max(1, lines // SEARCH_LINES)
    low = int(span * (1 - MAX_OVERLAP_FRACTION))
    high = int(span * (1 - MIN_OVERLAP_FRACTION))
    jitter = min(MAX_JITTER, max(4, int(lines * 0.1)))
    coarse = max(1, jitter // 4)
    best_offset, best_displacement, best_coarse = 0, low, -1.0
    for offset in range(-jitter, jitter + 1, coarse):
        top = a[offset:] if offset >= 0 else a[:lines + offset]
        bottom = b[:lines - offset] if offset >= 0 else b[-offset:]
        if top.shape[0] < 4:
            continue
        band = _sliding_ncc(top[::step], bottom[::step])[low:high + 1]
        peak = int(np.argmax(band))
        if band[peak] > best_coarse:
            best_offset, best_displacement = offset, peak + low
            best_coarse = float(band[peak])

    # Exactly, at full resolution, around the coarse winner: the sample of
    # lines cannot see a shift of three of them, and the placement is what
    # this number becomes.
    found, score = (0, 0), 0.0
    for offset in range(best_offset - coarse, best_offset + coarse + 1):
        dy, dx = ((offset, best_displacement) if axis == "x"
                  else (best_displacement, offset))
        candidate = _score_overlap(first, second, dy, dx)
        if candidate > score:
            found, score = (dy, dx), candidate
    return found, score


def _read_tiles(paths: Sequence):
    """Every tile as a 2-D array, or None when one cannot be used.

    A MULTI-PAGE TILE IS REFUSED RATHER THAN FLATTENED. Pages are Z, T or
    channel, and picking the first would silently stitch one plane of a
    stack into a field the caller believes is the whole thing.
    """
    import numpy as np
    import tifffile

    tiles = []
    for path in paths:
        try:
            data = tifffile.imread(str(path))
        except Exception:                                # noqa: BLE001
            return None
        array = np.asarray(data)
        if array.ndim != 2 or array.size == 0:
            return None
        tiles.append(array)
    shapes = {tile.shape for tile in tiles}
    if len(shapes) != 1:
        # TILES OF TWO SIZES ARE NOT A GRID. A mosaic of mixed shapes needs
        # per-tile placement, which needs stage coordinates; without them
        # there is nothing to place them by.
        return None
    return tiles


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def plan_mosaic(paths: Sequence, tiles: Optional[Sequence] = None) -> Optional[Mosaic]:
    """Work out where each tile of one field goes. Reads; writes nothing.

    :param paths: the tile images, in tile order.
    :param tiles: the tile indices, for the record. Defaults to 1..n.
    :returns: the mosaic, or None when the tiles cannot be read at all.
    """
    import numpy as np

    paths = [Path(p) for p in paths]
    names = list(tiles) if tiles is not None else list(range(1, len(paths) + 1))
    if not paths:
        return None
    if len(paths) == 1:
        images = _read_tiles(paths)
        if images is None:
            return None
        height, width = images[0].shape
        # ONE TILE IS NOT A MOSAIC, and calling it a stage placement
        # would claim evidence that was never read. It is placed because
        # there is nowhere else for it to go.
        return Mosaic(placements=(Placement(names[0], 0, 0, 0, 0),),
                      height=height, width=width, rows=1, cols=1,
                      arrangement="row_major", how="single", confidence=1.0)

    images = _read_tiles(paths)
    if images is None:
        return None
    height, width = images[0].shape

    positions = read_stage_positions(paths)
    if positions is not None and len({p for p in positions}) == len(positions):
        return _mosaic_from_positions(names, positions, height, width)

    rows, cols = grid_shape(len(paths))
    best: Optional[Mosaic] = None
    for arrangement in ARRANGEMENTS:
        candidate = _mosaic_by_correlation(images, names, rows, cols,
                                           arrangement, height, width)
        if best is None or candidate.confidence > best.confidence:
            best = candidate
    if rows != cols:
        # THE OTHER WAY ROUND IS A DIFFERENT GRID, not a different order: 6
        # tiles are 2x3 or 3x2 and the squarest shape cannot say which.
        for arrangement in ARRANGEMENTS:
            candidate = _mosaic_by_correlation(images, names, cols, rows,
                                               arrangement, height, width)
            if candidate.confidence > best.confidence:
                best = candidate
    assert best is not None
    if best.confidence < MIN_CONFIDENCE:
        return _butt_joined(names, rows, cols, height, width, best.confidence)
    return best


def _mosaic_from_positions(names, positions, height, width) -> Mosaic:
    """Place tiles at their recorded stage coordinates.

    The grid is read back out of the positions rather than assumed: distinct
    x values are columns and distinct y values are rows, which is what the
    stage actually did.
    """
    ys = sorted({round(y) for y, _x in positions})
    xs = sorted({round(x) for _y, x in positions})
    top, left = min(ys), min(xs)
    placements = []
    for name, (y, x) in zip(names, positions):
        placements.append(Placement(name, ys.index(round(y)), xs.index(round(x)),
                                    int(round(y - top)), int(round(x - left))))
    return Mosaic(
        placements=tuple(placements),
        height=max(p.y for p in placements) + height,
        width=max(p.x for p in placements) + width,
        rows=len(ys), cols=len(xs), arrangement="row_major",
        how="stage", confidence=1.0,
        overlap=(max(0, height - _spacing(ys)), max(0, width - _spacing(xs))))


def _spacing(values: List[int]) -> int:
    """The step between adjacent stage stops, or 0 when there is only one."""
    if len(values) < 2:
        return 0
    return int(round(min(b - a for a, b in zip(values, values[1:]))))


def _mosaic_by_correlation(images, names, rows, cols, arrangement,
                           height, width) -> Mosaic:
    """Score one arrangement by how well its neighbours actually overlap.

    EVERY TILE IS PLACED FROM ITS OWN MEASUREMENT, not from a grid pitch.
    A stage does not step exactly: neighbours sit a few pixels off the row
    they belong to, and a uniform grid puts that error into the seam, where
    it doubles a structure or drops a strip. So the measured displacement of
    each pair is walked out from the first tile -- which is also why a pair
    that could not be measured falls back to the average step rather than
    stopping the mosaic: an approximate placement for one tile is better
    than none for all of them, and the confidence says which happened.
    """
    grid: Dict[Tuple[int, int], int] = {}
    for index in range(len(images)):
        grid[arrangement_of(index, rows, cols, arrangement)] = index

    edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int]] = {}
    scores: List[float] = []
    dx_values: List[int] = []
    dy_values: List[int] = []
    for cell, index in grid.items():
        row, col = cell
        for neighbour, axis in (((row, col + 1), "x"), ((row + 1, col), "y")):
            other = grid.get(neighbour)
            if other is None:
                continue
            (dy, dx), score = _pair_shift(images[index], images[other], axis)
            scores.append(score)
            if score >= MIN_CONFIDENCE:
                edges[(cell, neighbour)] = (dy, dx)
                (dx_values if axis == "x" else dy_values).append(
                    abs(dx) if axis == "x" else abs(dy))

    step_x = int(sum(dx_values) / len(dx_values)) if dx_values else width
    step_y = int(sum(dy_values) / len(dy_values)) if dy_values else height
    confidence = sum(scores) / len(scores) if scores else 0.0

    # Walked out from the first tile along the edges that were believed;
    # anything the walk cannot reach takes the average step, which is the
    # best guess available for a seam that could not be measured.
    origin = (0, 0)
    positions: Dict[Tuple[int, int], Tuple[int, int]] = {origin: (0, 0)}
    frontier = [origin]
    while frontier:
        cell = frontier.pop()
        y, x = positions[cell]
        for (source, target), (dy, dx) in edges.items():
            if source == cell and target not in positions:
                positions[target] = (y + dy, x + dx)
                frontier.append(target)
            elif target == cell and source not in positions:
                positions[source] = (y - dy, x - dx)
                frontier.append(source)
    for cell in grid:
        positions.setdefault(cell, (cell[0] * step_y, cell[1] * step_x))

    top = min(y for y, _x in positions.values())
    left = min(x for _y, x in positions.values())
    placements = tuple(
        Placement(names[index], cell[0], cell[1],
                  positions[cell][0] - top, positions[cell][1] - left)
        for cell, index in sorted(grid.items(), key=lambda kv: kv[1]))
    return Mosaic(
        placements=placements,
        height=max(p.y for p in placements) + height,
        width=max(p.x for p in placements) + width,
        rows=rows, cols=cols, arrangement=arrangement,
        how="correlated", confidence=confidence,
        overlap=(max(0, height - step_y), max(0, width - step_x)))


def _butt_joined(names, rows, cols, height, width, confidence) -> Mosaic:
    """The fallback: tiles side by side, touching, and marked as a guess."""
    placements = tuple(
        Placement(names[index], *divmod(index, cols),
                  (index // cols) * height, (index % cols) * width)
        for index in range(len(names)))
    return Mosaic(placements=placements, height=rows * height,
                  width=cols * width, rows=rows, cols=cols,
                  arrangement="row_major", how="assumed",
                  confidence=confidence, overlap=(0, 0))


# ---------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------

def stitch_tiles(paths: Sequence, tiles: Optional[Sequence] = None,
                 mosaic: Optional[Mosaic] = None):
    """The tiles of one field, assembled into one array.

    OVERLAPS ARE AVERAGED, not overwritten. Where two tiles cover the same
    pixel they are two measurements of it, and taking the last one writes a
    visible step at every seam that a segmenter reads as an edge. Averaging
    also keeps the join honest when the placement is slightly off: the seam
    blurs rather than doubling a structure.

    :param paths: the tile images, in tile order.
    :param tiles: the tile indices, for the record.
    :param mosaic: a plan from :func:`plan_mosaic`; computed when omitted.
    :returns: ``(array, mosaic)``, or ``(None, None)`` when the tiles cannot
        be read.
    """
    import numpy as np

    images = _read_tiles([Path(p) for p in paths])
    if images is None:
        return None, None
    if mosaic is None:
        mosaic = plan_mosaic(paths, tiles)
    # A readable nonempty tile set always receives either a stage-derived,
    # correlated, or explicitly assumed plan.  Keep that contract loud if a
    # future planner change breaks it instead of disguising it as read failure.
    assert mosaic is not None

    dtype = images[0].dtype
    total = np.zeros((mosaic.height, mosaic.width), dtype=np.float64)
    counts = np.zeros((mosaic.height, mosaic.width), dtype=np.float64)
    for placement, image in zip(mosaic.placements, images):
        height, width = image.shape
        total[placement.y:placement.y + height,
              placement.x:placement.x + width] += image
        counts[placement.y:placement.y + height,
               placement.x:placement.x + width] += 1
    counts[counts == 0] = 1
    blended = total / counts
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        blended = np.clip(np.rint(blended), info.min, info.max)
    return blended.astype(dtype), mosaic
