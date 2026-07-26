"""Stitch and align an arbitrary number of image tiles into one canvas.

A 10x10 grid of 2048x2048 uint16 fields is a 20480x20480 canvas: 800 MB
for one channel, 3.2 GB for four. Nothing in this module ever holds that
canvas — or the whole input set — in RAM. That constraint drives every
design decision here:

* :func:`scan_tiles` reads **headers only**. ``.npy`` goes through
  :func:`numpy.lib.format.read_magic` plus the public
  ``read_array_header_*`` pair; TIFF through ``tifffile.TiffFile``'s
  series metadata. A thousand tiles cost a thousand ``seek``\\ s, no
  pixels.
* :func:`estimate_offsets` registers pairs on the **overlap strip only**,
  read out of a memory-mapped tile through the same windowed access
  pattern :class:`spacr.crops.MergedField` uses for on-demand crops. A
  10% overlap on a 2048x2048 tile is a 2048x205 strip — 0.8 MB, not 8 MB.
* :func:`plan_canvas` computes the canvas geometry from the offsets
  *before* anything is allocated, so an impossible stitch is refused on
  arithmetic rather than discovered by the OOM killer.
* :func:`write_stack` fills the output **one horizontal band at a time**.
  The peak working set is one band's float32 accumulator plus one tile
  window, both bounded by ``max_buffer_bytes`` and both independent of
  the canvas size.

The other half of the problem is being honest about what registered and
what did not.

**Offsets are solved globally, never accumulated.** Chaining tile-to-tile
shifts along a row of a 10x10 grid adds every pairwise error together, so
the last tile lands tens of pixels out with nothing in the output saying
so. Instead every overlapping neighbour pair contributes one equation
``p_j - p_i = d_ij`` and the whole set is solved at once as a weighted
least-squares problem over the pair graph (see :func:`solve_positions`).
Redundant edges — the diagonal neighbours of a grid, the ``i``/``i+2``
pairs of a heavily-overlapped row — then *cancel* error instead of
compounding it, and each tile gets a **residual**: how far its solved
position sits from what its own pairs asked for. A large residual is the
signal that a tile did not register, and it is written into the table
rather than averaged away.

**A pair that does not register is reported, not guessed.** Phase
correlation over a blank or near-empty overlap returns a confident-looking
peak that is pure noise. Every pair is therefore scored by the normalised
cross-correlation of the two strips *after* the measured shift is applied;
below ``min_confidence`` the pair is dropped, and a tile left with no
surviving pair keeps its nominal stage position and is marked
``method='nominal'``. A nominally-placed tile is not the same thing as a
registered one and :data:`ALIGN_TABLE` says which is which, per tile.

**Channels share one solution.** Registration runs on
``reference_channel`` alone and the resulting placement is applied to
every channel of that site. Aligning channels independently would shear
the composite — a nucleus mask and its own DAPI channel would no longer
line up — so it is not offered.

**z is not stitched.** Tiles are 2-D ``(H, W)`` or channel-last 3-D
``(H, W, C)``, which is the layout spaCR's ``merged/*.npy`` already uses.
A z-stack must be projected first, or aligned one z-plane at a time by
calling this module per plane; a 4-D array is refused with a message that
says so rather than being silently reinterpreted.

Typical use::

    from spacr import align

    tiles = align.scan_tiles('plate1/tiles', grid=(10, 10), overlap=0.1)
    plan = align.estimate_offsets(tiles)
    print(align.format_plan(plan))
    result = align.write_stack(plan, 'plate1/stitched')
    align.save_coordinates(plan, 'plate1/measurements/measurements.db',
                           canvas=result.canvas, stack_path=result.stack_path)

or in one call from a settings dict, the shape every other spaCR module
takes::

    align.align_folder({'src': 'plate1/tiles', 'dst': 'plate1/stitched',
                        'grid': (10, 10), 'db_path': '.../measurements.db'})

The coordinates table is keyed exactly the way every measurement table in
``measurements.db`` is keyed — ``plateID`` / ``rowID`` / ``columnID`` /
``fieldID`` plus the ``prc`` / ``prcf`` composites, built the way
:func:`spacr.utils._map_wells` builds them — so::

    SELECT c.*, a.y, a.x, a.method, a.confidence, a.residual
    FROM cell AS c
    JOIN align_coordinates AS a USING (plateID, rowID, columnID, fieldID)

puts every measured object next to the stitch quality of the field it came
from. That is the join that lets a suspicious cluster of hits be traced
back to "those three fields fell back to nominal".

This module is deliberately free of torch, cellpose, Qt and TensorFlow:
it is imported by a GUI screen and by header-only scans, and neither
should pay for a deep-learning stack.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import string
from dataclasses import dataclass, field as dc_field
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import numpy as np
import pandas as pd

from .crops import open_merged_field
from .errors import ConfigurationError, RunLedger

__all__ = [
    'ALIGN_TABLE',
    'ALIGN_COLUMNS',
    'METHOD_REGISTRATION',
    'METHOD_NOMINAL',
    'METHOD_SINGLE',
    'METHOD_UNREADABLE',
    'ORDERS',
    'BLEND_MODES',
    'AlignError',
    'Tile',
    'PairResult',
    'Placement',
    'AlignPlan',
    'CanvasSpec',
    'AlignResult',
    'scan_tiles',
    'group_tiles',
    'estimate_offsets',
    'solve_positions',
    'plan_canvas',
    'write_stack',
    'save_coordinates',
    'read_coordinates',
    'format_plan',
    'default_settings',
    'align_folder',
]

#: Table :func:`save_coordinates` writes into ``measurements.db``. Named
#: after ``convert``'s ``conversion_map`` — one provenance table per
#: module, keyed on the same four columns as every measurement table.
ALIGN_TABLE = 'align_coordinates'

#: How a tile got its position.
METHOD_REGISTRATION = 'registration'   #: solved from at least one scored pair
METHOD_NOMINAL = 'nominal'             #: no pair survived; stage position kept
METHOD_SINGLE = 'single'               #: the only tile in the set
METHOD_UNREADABLE = 'unreadable'       #: never placed; see AlignPlan.unplaced

#: Orders :func:`scan_tiles` can lay a flat list of fields out in.
ORDERS = ('row-major', 'column-major', 'snake-row', 'snake-column')

#: Seam handling understood by :func:`write_stack`.
BLEND_MODES = ('feather', 'average', 'none')

#: File suffixes :func:`scan_tiles` picks up.
IMAGE_SUFFIXES = ('.npy', '.tif', '.tiff')

#: Fraction of a tile assumed to overlap its neighbour when nothing says
#: otherwise. Matches the usual Yokogawa / Opera acquisition default.
DEFAULT_OVERLAP = 0.10

#: Normalised cross-correlation below which a pair is not believed.
DEFAULT_MIN_CONFIDENCE = 0.30

#: Sub-pixel refinement factor handed to ``phase_cross_correlation``.
DEFAULT_UPSAMPLE = 10

#: Smallest overlap, in pixels, worth attempting to register.
DEFAULT_MIN_OVERLAP_PX = 16

#: Ceiling on the band accumulator :func:`write_stack` allocates. This,
#: not the canvas, is the module's memory footprint.
#:
#: 64 MB is not a compromise between speed and memory — measured on a
#: 10x10 grid of 2048x2048 uint16 tiles (a 695 MB canvas), a 64 MB budget
#: writes the whole thing in 2.0 s at 244 MB peak RSS, while a 256 MB
#: budget takes 4.7 s at 513 MB. Bigger bands are *slower*: the band no
#: longer fits in cache and every tile still has to be re-read for it.
DEFAULT_MAX_BUFFER_BYTES = 64 << 20

#: Weight given to the "stay at your nominal position" equations in the
#: global solve. Small enough that a connected component's shape comes
#: entirely from its pairs, large enough to fix the gauge and to place a
#: tile that has no pairs at all.
DEFAULT_ANCHOR_WEIGHT = 1e-3

#: Floor on the feather ramp so a pixel covered only by one tile's extreme
#: edge still receives that tile's value instead of a zero.
_WEIGHT_FLOOR = 1e-3

#: Yokogawa CV7000/CV8000 output, the naming spaCR's pipeline expects.
_YOKOGAWA = re.compile(
    r'^(?P<plate>.+)_(?P<well>[A-Za-z]?\d+)_T(?P<t>\d+)F(?P<field>\d+)'
    r'L(?P<l>\d+)A(?P<a>\d+)Z(?P<z>\d+)C(?P<c>\d+)$')

#: The same, without the ``A##`` action id (some exports drop it).
_YOKOGAWA_SHORT = re.compile(
    r'^(?P<plate>.+)_(?P<well>[A-Za-z]?\d+)_T(?P<t>\d+)F(?P<field>\d+)'
    r'L(?P<l>\d+)C(?P<c>\d+)$')

#: spaCR's merged-stack naming: ``<plate>_<well>_<field>.npy``.
_MERGED = re.compile(r'^(?P<plate>[^_]+)_(?P<well>[A-Za-z]?\d+)_(?P<field>\d+)$')

#: Last resort: a trailing integer is the field, a ``_C##`` token the channel.
_TRAILING_FIELD = re.compile(r'(?P<field>\d+)\s*$')
_CHANNEL_TOKEN = re.compile(r'(?i)(?<=[_\-. ])(?:ch|channel|c|w)[_\-]?(\d{1,3})(?=$|[_\-. ])')


class AlignError(RuntimeError):
    """A tile could not be read, or a stitch could not be written."""


# ---------------------------------------------------------------------------
# Join keys — built exactly the way spacr.utils._map_wells builds them
#
# spacr.utils imports torch, and this module is on a GUI/thumbnail path, so
# the eight lines are reimplemented rather than imported. They are pinned
# against the original in tests/test_align.py::test_join_keys_match_utils.
# ---------------------------------------------------------------------------

def _well_ids(well: str) -> Tuple[str, str]:
    """Return ``(rowID, columnID)`` in spaCR's ``r1`` / ``c1`` form.

    ``'B7'`` becomes ``('r2', 'c7')``. A well that does not start with a
    letter is passed through unchanged in both slots, which is what
    :func:`spacr.utils._map_wells` does.
    """
    text = str(well)
    if text[:1].isalpha():
        try:
            row = f'r{string.ascii_uppercase.index(text[0].upper()) + 1}'
            column = f'c{int(text[1:])}'
            return row, column
        except (ValueError, IndexError):
            return text, text
    return text, text


def _join_keys(plate: str, well: str, field: int) -> Dict[str, str]:
    """Return the five spaCR join columns for one field."""
    row_id, column_id = _well_ids(well)
    prc = f'{plate}_{row_id}_{column_id}'
    return {
        'plateID': str(plate),
        'rowID': row_id,
        'columnID': column_id,
        'fieldID': f'f{int(field)}',
        'prc': prc,
        'prcf': f'{prc}_f{int(field)}',
    }


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tile:
    """One site: everything known about a tile without reading its pixels.

    A "site" is a position on the stage, not a file. When each channel is
    its own file — the Yokogawa case — the sibling paths live in
    :attr:`channel_paths` and :attr:`path` is the reference channel's; when
    the channels are planes of one ``(H, W, C)`` array —  spaCR's
    ``merged/*.npy`` — :attr:`channel_paths` is empty and :attr:`path` is
    the only file. Either way there is exactly one :class:`Tile`, and so
    exactly one :class:`Placement`, per stage position.

    :ivar path: absolute path of the reference-channel file.
    :ivar index: position in the list handed to :func:`estimate_offsets`.
        The pair graph and the least-squares solve are indexed by it.
    :ivar plate: plate token parsed from the filename.
    :ivar well: well token parsed from the filename, e.g. ``'B07'``.
    :ivar field: 1-based field id.
    :ivar channel: the channel that drives registration for this site.
    :ivar shape: ``(H, W, C)`` of the assembled site — ``C`` counts the
        sibling files when channels are split across them.
    :ivar dtype: dtype name, e.g. ``'uint16'``.
    :ivar grid_row: nominal row in the acquisition grid.
    :ivar grid_col: nominal column in the acquisition grid.
    :ivar nominal_y: nominal stage position, canvas rows.
    :ivar nominal_x: nominal stage position, canvas columns.
    :ivar channel_paths: one path per channel, or ``()`` when the channels
        live inside :attr:`path`.
    :ivar error: why this tile could not be read; ``''`` when it can.
    """

    path: str
    index: int = 0
    plate: str = ''
    well: str = ''
    field: int = 0
    channel: int = 0
    shape: Tuple[int, ...] = ()
    dtype: str = 'uint16'
    grid_row: int = 0
    grid_col: int = 0
    nominal_y: float = 0.0
    nominal_x: float = 0.0
    channel_paths: Tuple[str, ...] = ()
    error: str = ''

    @property
    def readable(self) -> bool:
        """False when the header could not be read."""
        return not self.error

    @property
    def height(self) -> int:
        """Tile height in pixels, 0 when unknown."""
        return int(self.shape[0]) if len(self.shape) >= 2 else 0

    @property
    def width(self) -> int:
        """Tile width in pixels, 0 when unknown."""
        return int(self.shape[1]) if len(self.shape) >= 2 else 0

    @property
    def n_channels(self) -> int:
        """Number of channels this site contributes to the canvas."""
        return int(self.shape[2]) if len(self.shape) >= 3 else 1

    @property
    def name(self) -> str:
        """Short human label, ``plate_well_field``."""
        return f'{self.plate}_{self.well}_{self.field}'


@dataclass(frozen=True)
class PairResult:
    """One neighbour pair, registered or refused.

    :ivar i: index of the first tile.
    :ivar j: index of the second tile.
    :ivar dy: measured displacement of ``j`` relative to ``i``, rows.
    :ivar dx: measured displacement of ``j`` relative to ``i``, columns.
    :ivar nominal_dy: what the stage positions said the displacement was.
    :ivar nominal_dx: ditto, columns.
    :ivar confidence: normalised cross-correlation of the two overlap
        strips *after* ``(dy, dx)`` is applied. 0.0 when refused.
    :ivar accepted: whether this pair fed the global solve.
    :ivar overlap_px: pixels compared. A pair scored on 200 pixels is
        worth less than one scored on 400 000 and the note records it.
    :ivar note: why a refused pair was refused.
    """

    i: int
    j: int
    dy: float = 0.0
    dx: float = 0.0
    nominal_dy: float = 0.0
    nominal_dx: float = 0.0
    confidence: float = 0.0
    accepted: bool = False
    overlap_px: int = 0
    note: str = ''

    @property
    def drift(self) -> float:
        """Euclidean distance between the measured and nominal displacement."""
        return float(math.hypot(self.dy - self.nominal_dy,
                                self.dx - self.nominal_dx))


@dataclass(frozen=True)
class Placement:
    """Where one tile goes, and how much that position is worth.

    :ivar tile: the :class:`Tile` this places.
    :ivar y: solved row offset, in the global frame. May be negative —
        :func:`plan_canvas` turns the frame into canvas indices.
    :ivar x: solved column offset, in the global frame.
    :ivar confidence: best pair confidence backing this tile; 0.0 for a
        nominal fallback, 1.0 for the single-tile case.
    :ivar method: one of :data:`METHOD_REGISTRATION`,
        :data:`METHOD_NOMINAL`, :data:`METHOD_SINGLE`.
    :ivar note: human explanation, e.g. which pairs were refused.
    :ivar residual: RMS distance, in pixels, between this tile's solved
        position and what its own accepted pairs asked for. 0.0 when the
        tile has no pairs. **This is the number to sort on**: a tile that
        registered against neighbours that disagree shows up here and
        nowhere else.
    :ivar n_pairs: accepted pairs incident on this tile.
    """

    tile: Tile
    y: float = 0.0
    x: float = 0.0
    confidence: float = 0.0
    method: str = METHOD_NOMINAL
    note: str = ''
    residual: float = 0.0
    n_pairs: int = 0


@dataclass
class AlignPlan:
    """The full solution: what goes where, and everything that went wrong.

    :ivar tiles: every tile handed in, readable or not.
    :ivar placements: one per readable tile, in tile order.
    :ivar canvas_shape: ``(H, W, C)`` the stitch would occupy.
    :ivar overlaps: every candidate pair, accepted or refused.
    :ivar warnings: non-fatal problems — mixed dtypes, disconnected
        components, tiles with no overlap at all.
    :ivar unplaced: ``(tile, reason)`` for every tile that could not be
        placed. These are excluded from the canvas, not silently zeroed.
    :ivar origin: ``(y, x)`` of the canvas origin in the global frame.
    :ivar feather: seam ramp width in pixels, derived from the real
        overlaps rather than guessed.
    :ivar dtype: dtype the canvas would be written in.
    :ivar reference_channel: the channel registration was measured on.
    """

    tiles: List[Tile] = dc_field(default_factory=list)
    placements: List[Placement] = dc_field(default_factory=list)
    canvas_shape: Tuple[int, int, int] = (0, 0, 0)
    overlaps: List[PairResult] = dc_field(default_factory=list)
    warnings: List[str] = dc_field(default_factory=list)
    unplaced: List[Tuple[Tile, str]] = dc_field(default_factory=list)
    origin: Tuple[float, float] = (0.0, 0.0)
    feather: int = 1
    dtype: str = 'uint16'
    reference_channel: int = 0

    @property
    def n_registered(self) -> int:
        """Tiles placed by registration."""
        return sum(1 for p in self.placements if p.method == METHOD_REGISTRATION)

    @property
    def n_nominal(self) -> int:
        """Tiles that fell back to their nominal stage position."""
        return sum(1 for p in self.placements if p.method == METHOD_NOMINAL)

    @property
    def accepted_pairs(self) -> List[PairResult]:
        """Pairs that fed the global solve."""
        return [p for p in self.overlaps if p.accepted]

    @property
    def refused_pairs(self) -> List[PairResult]:
        """Pairs that were scored and not believed."""
        return [p for p in self.overlaps if not p.accepted]

    @property
    def max_residual(self) -> float:
        """Worst per-tile residual in the solve."""
        return max((p.residual for p in self.placements), default=0.0)

    @property
    def canvas_bytes(self) -> int:
        """Bytes the written canvas would occupy on disk."""
        h, w, c = self.canvas_shape
        return int(h) * int(w) * int(c) * int(np.dtype(self.dtype).itemsize)

    def nominal_placements(self) -> List[Placement]:
        """Every tile that did *not* register, worst confidence first."""
        return sorted((p for p in self.placements if p.method == METHOD_NOMINAL),
                      key=lambda p: (p.confidence, p.tile.index))


@dataclass(frozen=True)
class CanvasSpec:
    """Canvas geometry, computed before a single byte is allocated.

    :ivar height: canvas rows.
    :ivar width: canvas columns.
    :ivar channels: canvas planes.
    :ivar dtype: numpy dtype name.
    :ivar origin_y: global-frame row that maps to canvas row 0. Negative
        offsets live here rather than being clipped away.
    :ivar origin_x: global-frame column that maps to canvas column 0.
    """

    height: int
    width: int
    channels: int
    dtype: str
    origin_y: float = 0.0
    origin_x: float = 0.0

    @property
    def shape(self) -> Tuple[int, int, int]:
        """``(H, W, C)``."""
        return (int(self.height), int(self.width), int(self.channels))

    @property
    def nbytes(self) -> int:
        """Size of the written ``.npy`` payload, in bytes."""
        return (int(self.height) * int(self.width) * int(self.channels)
                * int(np.dtype(self.dtype).itemsize))

    def canvas_yx(self, y: float, x: float) -> Tuple[int, int]:
        """Map a global-frame position onto integer canvas indices."""
        return (int(round(y - self.origin_y)), int(round(x - self.origin_x)))


@dataclass
class AlignResult:
    """What :func:`write_stack` actually did.

    :ivar plan: the plan that was written.
    :ivar canvas: the geometry it was written at.
    :ivar stack_path: the ``.npy`` written, ``''`` for a dry run.
    :ivar n_written: tiles composited into the canvas.
    :ivar n_skipped: tiles that could not be read at write time. A tile
        whose header parsed but whose pixels do not is caught here, not
        in :func:`estimate_offsets`.
    :ivar peak_buffer_bytes: the largest in-RAM buffer the write
        allocated — band accumulator plus weight plane. Compare it with
        ``canvas.nbytes``: that ratio is the whole point of this module.
    :ivar band_rows: canvas rows held in RAM at once.
    :ivar writer: ``'memmap'`` or ``'stream'``.
    :ivar status: ``RunLedger`` status — ``complete`` / ``partial`` / ``empty``.
    :ivar warnings: anything non-fatal that happened during the write.
    :ivar db_path: database :func:`save_coordinates` wrote to, if any.
    """

    plan: AlignPlan
    canvas: CanvasSpec
    stack_path: str = ''
    n_written: int = 0
    n_skipped: int = 0
    peak_buffer_bytes: int = 0
    band_rows: int = 0
    writer: str = 'memmap'
    status: str = 'empty'
    warnings: List[str] = dc_field(default_factory=list)
    db_path: str = ''

    def summary(self) -> str:
        """One-block human summary of the write."""
        ratio = (self.canvas.nbytes / self.peak_buffer_bytes
                 if self.peak_buffer_bytes else float('inf'))
        lines = [
            f'Stitched {self.n_written} tile(s) into '
            f'{self.canvas.height}x{self.canvas.width}x{self.canvas.channels} '
            f'{self.canvas.dtype} ({_human_bytes(self.canvas.nbytes)}).',
            f'Peak RAM buffer {_human_bytes(self.peak_buffer_bytes)} '
            f'({self.band_rows} row band, {self.writer} writer) — '
            f'{ratio:.0f}x smaller than the canvas.',
        ]
        if self.stack_path:
            lines.append(f'Wrote {self.stack_path}')
        if self.n_skipped:
            lines.append(f'{self.n_skipped} tile(s) skipped — see warnings.')
        lines.extend(f'  ! {w}' for w in self.warnings)
        return '\n'.join(lines)


def _human_bytes(n: Union[int, float]) -> str:
    """Format a byte count for a log line."""
    value = float(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024
    return f'{value:.1f} TB'


# ---------------------------------------------------------------------------
# Header-only readers
# ---------------------------------------------------------------------------

def _npy_header(path: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Return ``(shape, dtype)`` of a ``.npy`` without reading its data.

    Uses the public ``numpy.lib.format`` header readers, so the cost is
    one open and a few hundred bytes regardless of how big the array is.
    """
    with open(path, 'rb') as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise AlignError(
                f'{path}: unsupported .npy format version {version}')
    return tuple(int(v) for v in shape), np.dtype(dtype)


def _tiff_header(path: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Return ``(shape, dtype)`` of a TIFF from its IFD, no pixels read."""
    import tifffile
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        return tuple(int(v) for v in series.shape), np.dtype(series.dtype)


def _read_header(path: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Dispatch to the right header reader for ``path``'s suffix."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == '.npy':
        return _npy_header(path)
    if suffix in ('.tif', '.tiff'):
        return _tiff_header(path)
    raise AlignError(f'{path}: not a .npy or .tif tile')


def _normalise_shape(shape: Tuple[int, ...], path: str) -> Tuple[int, int, int]:
    """Return ``(H, W, C)`` for a 2-D or 3-D tile shape.

    A 3-D array is channel-last unless its *first* axis is both small
    (<= 8) and smaller than its last, in which case it is the ``(C, H, W)``
    a TIFF writer produces. 4-D is refused outright: a z-stack must be
    projected or aligned one plane at a time, and guessing which axis is
    z would put the wrong pixels in the canvas.
    """
    if len(shape) == 2:
        return (int(shape[0]), int(shape[1]), 1)
    if len(shape) == 3:
        if shape[0] <= 8 and shape[0] < shape[-1]:
            return (int(shape[1]), int(shape[2]), int(shape[0]))
        return (int(shape[0]), int(shape[1]), int(shape[2]))
    raise AlignError(
        f'{path}: a tile must be 2-D (H, W) or 3-D (H, W, C); got '
        f'{len(shape)}-D {shape}. z is not stitched — max-project the '
        f'stack, or call align once per z-plane.')


def _channel_last(array):
    """Return a channel-last *view* of a 2-D or 3-D memory-mapped array."""
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim == 3 and array.shape[0] <= 8 and array.shape[0] < array.shape[-1]:
        return np.moveaxis(array, 0, -1)
    return array


# ---------------------------------------------------------------------------
# Windowed tile access
# ---------------------------------------------------------------------------

class _TileReader:
    """Memory-mapped, windowed access to one site's pixels.

    A 3-D ``.npy`` is opened through :func:`spacr.crops.open_merged_field`,
    whose :meth:`~spacr.crops.MergedField.read_window` faults in only the
    requested window one plane at a time and zero-pads anything that runs
    off the edge. That is exactly the access pattern registration and
    band-wise compositing need, so it is reused rather than rewritten.
    2-D ``.npy`` and TIFF get the same contract over a plain memmap.

    The reader is intentionally dumb about caching: a stitch touches each
    tile a bounded number of times (once per band it intersects) and
    caching whole tiles is precisely the thing that would put the canvas
    back in RAM.
    """

    def __init__(self, tile: Tile):
        self.tile = tile
        self._sources: List[Any] = []
        self._fields: List[Any] = []
        self._handles: List[Any] = []
        paths = list(tile.channel_paths) or [tile.path]
        for path in paths:
            suffix = os.path.splitext(path)[1].lower()
            if suffix == '.npy':
                shape, _dtype = _npy_header(path)
                if len(shape) == 3 and not (shape[0] <= 8 and shape[0] < shape[-1]):
                    # (H, W, C) — the merged-stack layout crops.MergedField
                    # was written for.
                    self._fields.append(open_merged_field(path, use_cache=False))
                    self._sources.append(None)
                    continue
                self._fields.append(None)
                self._sources.append(_channel_last(
                    np.load(path, mmap_mode='r', allow_pickle=False)))
            elif suffix in ('.tif', '.tiff'):
                self._fields.append(None)
                self._sources.append(_channel_last(self._open_tiff(path)))
            else:
                raise AlignError(f'{path}: not a .npy or .tif tile')
        self._split_channels = bool(tile.channel_paths)

    def _open_tiff(self, path: str):
        """Memory-map a TIFF, falling back to a full read when it is compressed."""
        import tifffile
        try:
            handle = tifffile.memmap(path, mode='r')
            self._handles.append(handle)
            return handle
        except (ValueError, MemoryError, OSError):
            # Compressed / tiled TIFFs cannot be mapped. Reading one tile
            # whole is bounded by the tile, never by the canvas.
            return tifffile.imread(path)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """``(H, W, C)`` of the assembled site."""
        if self._split_channels:
            first = self._plane_source(0)
            return (int(first.shape[0]), int(first.shape[1]), len(self._sources))
        first = self._plane_source(0)
        return (int(first.shape[0]), int(first.shape[1]), int(first.shape[2]))

    def _plane_source(self, k: int):
        """Return the channel-last array backing group ``k``."""
        field = self._fields[k]
        return field.array if field is not None else self._sources[k]

    def window(self, y0: int, y1: int, x0: int, x1: int,
               channels: Sequence[int], dtype=np.float32) -> np.ndarray:
        """Read ``channels`` over ``[y0:y1, x0:x1]``, zero-padded off the edge.

        Only the requested rectangle is faulted in. The returned array is
        ``(y1 - y0, x1 - x0, len(channels))``.
        """
        out = np.zeros((int(y1 - y0), int(x1 - x0), len(channels)),
                       dtype=np.dtype(dtype))
        for k, channel in enumerate(channels):
            if self._split_channels:
                group, plane = int(channel), 0
                if not 0 <= group < len(self._sources):
                    continue
            else:
                group, plane = 0, int(channel)
            field = self._fields[group]
            if field is not None:
                sub = field.read_window(int(y0), int(y1), int(x0), int(x1),
                                        [plane], dtype=dtype)
                out[:, :, k] = sub[:, :, 0]
                continue
            source = self._sources[group]
            height, width = int(source.shape[0]), int(source.shape[1])
            if plane >= int(source.shape[2]):
                continue
            sy0, sy1 = max(0, int(y0)), min(height, int(y1))
            sx0, sx1 = max(0, int(x0)), min(width, int(x1))
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            dy, dx = sy0 - int(y0), sx0 - int(x0)
            block = np.asarray(source[sy0:sy1, sx0:sx1, plane])
            out[dy:dy + (sy1 - sy0), dx:dx + (sx1 - sx0), k] = \
                block.astype(dtype, copy=False)
        return out

    def close(self) -> None:
        """Unmap everything this reader holds.

        Unmapping, not just forgetting, is the point. Reading a *column*
        strip out of a row-major tile touches one page per row, so a
        2048x205 strip out of a 2048x2048 uint16 tile faults in all 8 MB
        of it. Leave a hundred of those mapped and the registration pass
        has quietly resident the entire input set — the exact failure this
        module exists to avoid. Dropping the mapping returns the pages.
        """
        for group in (self._sources, self._fields):
            for k, item in enumerate(group):
                array = getattr(item, 'array', item)
                handle = getattr(array, '_mmap', None)
                if handle is not None:
                    try:
                        handle.close()
                    except (BufferError, ValueError):
                        pass
                group[k] = None
        self._handles.clear()


class _ReaderCache:
    """At most ``max_open`` tiles mapped at once, least-recently-used evicted.

    Both the registration pass and the band writer walk the tiles in a
    spatially coherent order, so a handful of live readers covers the
    working set: registration needs the current tile and its neighbours,
    the writer needs the tiles that intersect the current band. Anything
    older is unmapped, which is what keeps the resident set proportional
    to ``max_open`` tiles instead of to the whole input folder.
    """

    def __init__(self, max_open: int = 8):
        self.max_open = max(1, int(max_open))
        self._open: "Dict[int, _TileReader]" = {}
        self._order: List[int] = []
        self.opened = 0

    def get(self, tile: Tile) -> _TileReader:
        """Return a reader for ``tile``, opening (and evicting) as needed."""
        key = int(tile.index)
        reader = self._open.get(key)
        if reader is not None:
            self._order.remove(key)
            self._order.append(key)
            return reader
        while len(self._order) >= self.max_open:
            self._open.pop(self._order.pop(0)).close()
        reader = _TileReader(tile)
        self.opened += 1
        self._open[key] = reader
        self._order.append(key)
        return reader

    def close(self) -> None:
        """Unmap every reader still held."""
        for reader in self._open.values():
            reader.close()
        self._open.clear()
        self._order.clear()


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _parse_name(stem: str) -> Dict[str, Any]:
    """Pull plate / well / field / channel out of a filename stem.

    Tries the Yokogawa forms spaCR's pipeline produces, then spaCR's own
    ``<plate>_<well>_<field>`` merged-stack name, then a bare trailing
    integer. Nothing here reads the file.
    """
    for pattern in (_YOKOGAWA, _YOKOGAWA_SHORT):
        match = pattern.match(stem)
        if match:
            return {
                'plate': match.group('plate'),
                'well': match.group('well'),
                'field': int(match.group('field')),
                'channel': int(match.group('c')),
            }
    match = _MERGED.match(stem)
    if match:
        return {
            'plate': match.group('plate'),
            'well': match.group('well'),
            'field': int(match.group('field')),
            'channel': 1,
        }
    channel = 1
    chan_match = _CHANNEL_TOKEN.search(stem)
    if chan_match:
        channel = int(chan_match.group(1))
        stem_wo = _CHANNEL_TOKEN.sub('', stem)
    else:
        stem_wo = stem
    field_match = _TRAILING_FIELD.search(stem_wo)
    field = int(field_match.group('field')) if field_match else 0
    return {'plate': '', 'well': '', 'field': field, 'channel': channel}


def _grid_shape(n: int, grid: Optional[Sequence[int]]) -> Tuple[int, int]:
    """Return ``(rows, cols)`` for ``n`` tiles.

    An explicit ``grid`` wins. Otherwise prefer an exact factorisation
    closest to square (12 fields become 3x4, not 4x4 with 4 holes) and
    fall back to a padded near-square when ``n`` is prime.
    """
    if grid is not None:
        rows, cols = int(grid[0]), int(grid[1])
        if rows <= 0 or cols <= 0:
            raise ConfigurationError(
                f'grid must be two positive integers, got {tuple(grid)!r}')
        if rows * cols < n:
            raise ConfigurationError(
                f'grid {rows}x{cols} has room for {rows * cols} tiles but '
                f'{n} were found. Pass a bigger grid, or leave grid=None '
                f'to infer one.')
        return rows, cols
    if n <= 1:
        return (1, max(1, n))
    best = None
    for rows in range(1, int(math.isqrt(n)) + 1):
        if n % rows == 0:
            best = (rows, n // rows)
    if best is not None and best[0] > 1:
        return best
    side = int(math.ceil(math.sqrt(n)))
    return (int(math.ceil(n / side)), side)


def _grid_position(k: int, rows: int, cols: int, order: str) -> Tuple[int, int]:
    """Return the ``(row, col)`` of the ``k``-th tile under ``order``."""
    if order not in ORDERS:
        raise ConfigurationError(
            f'order must be one of {ORDERS}, got {order!r}')
    if order in ('row-major', 'snake-row'):
        row, col = divmod(k, cols)
        if order == 'snake-row' and row % 2:
            col = cols - 1 - col
        return row, col
    col, row = divmod(k, rows)
    if order == 'snake-column' and col % 2:
        row = rows - 1 - row
    return row, col


def _collect_paths(src: Union[str, os.PathLike, Sequence[Any]],
                   recursive: bool) -> List[str]:
    """Return the sorted image paths under ``src``."""
    if isinstance(src, (list, tuple)):
        return [os.path.abspath(os.fspath(p)) for p in src]
    root = os.path.abspath(os.fspath(src))
    if os.path.isfile(root):
        return [root]
    if not os.path.isdir(root):
        raise ConfigurationError(f'align: source folder does not exist: {root}')
    found: List[str] = []
    if recursive:
        for base, _dirs, names in os.walk(root):
            for name in names:
                if name.lower().endswith(IMAGE_SUFFIXES) and not name.startswith('.'):
                    found.append(os.path.join(base, name))
    else:
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if (os.path.isfile(path) and not name.startswith('.')
                    and name.lower().endswith(IMAGE_SUFFIXES)):
                found.append(path)
    return sorted(found)


def scan_tiles(src: Union[str, os.PathLike, Sequence[Any]],
               *,
               grid: Optional[Sequence[int]] = None,
               overlap: float = DEFAULT_OVERLAP,
               order: str = 'row-major',
               recursive: bool = False,
               positions: Optional[Mapping[int, Sequence[float]]] = None,
               reference_channel: Optional[int] = None) -> List[Tile]:
    """Return one :class:`Tile` per stage position, **without reading pixels**.

    Every file's shape and dtype comes out of its header. A folder of
    1000 tiles is scanned in the time it takes to open 1000 files, which
    is what makes "show me the plan before you write 800 MB" possible.

    Files that share ``(plate, well, field)`` are collapsed into one tile
    with one plane per channel: a Yokogawa well with four channels per
    field produces one :class:`Tile` per field, not four.

    :param src: folder of tiles, a single file, or an explicit list of
        paths (which fixes the order — useful when the filenames carry
        no field number).
    :param grid: ``(rows, cols)`` of the acquisition. Inferred from the
        tile count when omitted.
    :param overlap: nominal neighbour overlap as a fraction of the tile,
        used only to seed the search; registration corrects it.
    :param order: how a flat field sequence maps onto the grid — see
        :data:`ORDERS`. Serpentine acquisitions need ``'snake-row'``.
    :param recursive: walk sub-folders too.
    :param positions: optional ``{field: (y, x)}`` of *known* stage
        positions in pixels. Overrides the grid entirely — pass the
        microscope's real coordinates when you have them.
    :param reference_channel: which channel drives registration. Defaults
        to the lowest channel present.
    :returns: tiles in grid order, each with ``index`` set.
    :raises ConfigurationError: ``src`` does not exist, or ``grid`` is too
        small for the tiles found.
    """
    if not 0.0 <= float(overlap) < 1.0:
        raise ConfigurationError(
            f'overlap must be in [0, 1), got {overlap!r}')
    paths = _collect_paths(src, recursive)
    if not paths:
        raise ConfigurationError(
            f'align: no .npy/.tif tiles found in {src!r}. Point src at the '
            f'folder that holds the tiles, or pass an explicit list.')

    # -- group files into sites -------------------------------------------
    sites: "Dict[Tuple[str, str, int], Dict[str, Any]]" = {}
    order_seen: List[Tuple[str, str, int]] = []
    for position, path in enumerate(paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        meta = _parse_name(stem)
        key = (meta['plate'], meta['well'], meta['field'])
        if key == ('', '', 0):
            key = ('', '', position + 1)
        record = sites.get(key)
        if record is None:
            record = {'paths': {}, 'first': position, 'meta': meta}
            sites[key] = record
            order_seen.append(key)
        record['paths'][int(meta['channel'])] = path

    keys = sorted(order_seen, key=lambda k: (sites[k]['first'],))
    rows, cols = _grid_shape(len(keys), grid)

    tiles: List[Tile] = []
    for k, key in enumerate(keys):
        record = sites[key]
        channels = sorted(record['paths'])
        ref = int(reference_channel) if reference_channel is not None else channels[0]
        if ref not in channels:
            ref = channels[0]
        ref_path = record['paths'][ref]
        plate, well, field = key
        grid_row, grid_col = _grid_position(k, rows, cols, order)

        error = ''
        try:
            raw_shape, dtype = _read_header(ref_path)
            height, width, inner = _normalise_shape(raw_shape, ref_path)
        except Exception as exc:                     # unreadable header
            error = f'{type(exc).__name__}: {exc}'
            height = width = inner = 0
            dtype = np.dtype('uint16')

        if len(channels) > 1:
            n_channels = len(channels)
            channel_paths = tuple(record['paths'][c] for c in channels)
            ref_index = channels.index(ref)
        else:
            n_channels = inner
            channel_paths = ()
            ref_index = 0

        if positions is not None and field in positions:
            nominal_y, nominal_x = (float(positions[field][0]),
                                    float(positions[field][1]))
        else:
            nominal_y = grid_row * height * (1.0 - float(overlap))
            nominal_x = grid_col * width * (1.0 - float(overlap))

        tiles.append(Tile(
            path=ref_path, index=k, plate=plate, well=well,
            field=int(field) if field else k + 1,
            channel=ref_index,
            shape=(height, width, max(1, n_channels)),
            dtype=str(np.dtype(dtype)),
            grid_row=grid_row, grid_col=grid_col,
            nominal_y=nominal_y, nominal_x=nominal_x,
            channel_paths=channel_paths, error=error))
    return tiles


def group_tiles(tiles: Sequence[Tile]) -> "Dict[Tuple[str, str], List[Tile]]":
    """Split tiles into one stitchable set per ``(plate, well)``.

    Fields of different wells are not neighbours and must never be
    registered against each other. Indices are renumbered within each
    group so the returned lists can go straight into
    :func:`estimate_offsets`.
    """
    groups: "Dict[Tuple[str, str], List[Tile]]" = {}
    for tile in tiles:
        groups.setdefault((tile.plate, tile.well), []).append(tile)
    out: "Dict[Tuple[str, str], List[Tile]]" = {}
    for key, members in groups.items():
        out[key] = [
            Tile(**{**vars(t), 'index': i})
            for i, t in enumerate(sorted(members, key=lambda t: t.index))]
    return out


# ---------------------------------------------------------------------------
# Pairwise registration
# ---------------------------------------------------------------------------

def _overlap_windows(tile_a: Tile, tile_b: Tile,
                     dy: float, dx: float) -> Optional[Tuple[Tuple[int, int, int, int],
                                                             Tuple[int, int, int, int]]]:
    """Return the matching overlap rectangles in ``a``'s and ``b``'s frames.

    ``(dy, dx)`` is ``b``'s position relative to ``a``. Returns ``None``
    when the tiles do not overlap at all.
    """
    idy, idx = int(round(dy)), int(round(dx))
    ah, aw = tile_a.height, tile_a.width
    bh, bw = tile_b.height, tile_b.width
    ay0, ay1 = max(0, idy), min(ah, bh + idy)
    ax0, ax1 = max(0, idx), min(aw, bw + idx)
    if ay1 <= ay0 or ax1 <= ax0:
        return None
    return ((ay0, ay1, ax0, ax1),
            (ay0 - idy, ay1 - idy, ax0 - idx, ax1 - idx))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation (Pearson r) of two equal-shaped arrays.

    This — not the phase-correlation peak — is the confidence score. A
    blank overlap has no variance, so ``r`` is undefined and returns 0.0,
    which is exactly the answer that keeps a noise peak out of the solve.
    """
    if a.size == 0 or a.shape != b.shape:
        return 0.0
    af = a.astype(np.float64, copy=False).ravel()
    bf = b.astype(np.float64, copy=False).ravel()
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = float(np.sqrt(float(af @ af) * float(bf @ bf)))
    if denom <= 1e-12:
        return 0.0
    return float(af @ bf) / denom


def _standardise(strip: np.ndarray) -> np.ndarray:
    """Return ``strip`` zero-meaned and scaled to unit variance, in place."""
    out = np.asarray(strip, dtype=np.float32)
    out -= float(out.mean())
    scale = float(out.std())
    if scale > 1e-12:
        out /= scale
    return out


def _register_pair(tile_a: Tile, tile_b: Tile,
                   reader_a: _TileReader, reader_b: _TileReader,
                   *, upsample: int, min_confidence: float,
                   min_overlap_px: int, max_shift: Optional[float],
                   reference_channel: int) -> PairResult:
    """Register ``b`` against ``a`` using only their overlap strips.

    Reads two windows of the nominal overlap — never the tiles — runs
    phase correlation on them, then *checks its own answer* by shifting
    the strips into agreement and scoring them with
    :func:`_ncc`. A peak that does not survive that check is not a
    registration, and the returned pair carries ``accepted=False`` plus a
    note saying why.
    """
    nominal_dy = tile_b.nominal_y - tile_a.nominal_y
    nominal_dx = tile_b.nominal_x - tile_a.nominal_x
    base = PairResult(i=tile_a.index, j=tile_b.index,
                      dy=nominal_dy, dx=nominal_dx,
                      nominal_dy=nominal_dy, nominal_dx=nominal_dx)

    windows = _overlap_windows(tile_a, tile_b, nominal_dy, nominal_dx)
    if windows is None:
        return _replace_pair(base, note='tiles do not overlap at their '
                                        'nominal positions')
    (ay0, ay1, ax0, ax1), (by0, by1, bx0, bx1) = windows
    height, width = ay1 - ay0, ax1 - ax0
    if height < min_overlap_px or width < min_overlap_px:
        return _replace_pair(
            base, overlap_px=height * width,
            note=f'overlap {height}x{width} px is below min_overlap_px='
                 f'{min_overlap_px}')

    strip_a = reader_a.window(ay0, ay1, ax0, ax1, [reference_channel])[:, :, 0]
    strip_b = reader_b.window(by0, by1, bx0, bx1, [reference_channel])[:, :, 0]

    if float(strip_a.std()) <= 1e-9 or float(strip_b.std()) <= 1e-9:
        return _replace_pair(
            base, overlap_px=height * width,
            note='overlap is blank (zero variance) — phase correlation '
                 'would return a noise peak; kept at the nominal position')

    # Zero-mean / unit-variance before the FFT. Removing DC is standard for
    # phase correlation (the DC bin otherwise dominates the normalisation),
    # and it keeps the |F|^2 products inside float32 — raw uint16 intensity
    # over a 2048x205 strip overflows them.
    strip_a = _standardise(strip_a)
    strip_b = _standardise(strip_b)

    from skimage.registration import phase_cross_correlation
    # Two normalisations, because neither is reliable alone.
    #
    # 'phase' whitens the spectrum, which is what makes phase correlation
    # immune to illumination differences between fields — and also what
    # makes it fail on smooth, low-texture images, where whitening
    # amplifies quantisation noise until it outweighs the real signal. On
    # a 64x50 strip of Gaussian-smoothed fluorescence it returns 0 px for
    # a genuine 10 px offset. Plain cross-correlation gets that one right
    # but is pulled around by intensity gradients.
    #
    # So both are tried and each candidate is *scored* on the pixels it
    # implies; the better score wins. One extra FFT over a strip is a
    # rounding error next to being wrong.
    candidates: List[Tuple[float, float]] = []
    for normalization in ('phase', None):
        try:
            shift, _error, _phasediff = phase_cross_correlation(
                strip_a, strip_b, upsample_factor=int(upsample),
                normalization=normalization)
        except Exception:
            continue
        candidate = (float(shift[0]), float(shift[1]))
        if candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        return _replace_pair(base, overlap_px=height * width,
                             note='phase correlation failed on this overlap')

    best_score, best_px, best_shift = -2.0, 0, None
    too_far: List[Tuple[float, float]] = []
    for shift_y, shift_x in candidates:
        if max_shift is not None and math.hypot(shift_y, shift_x) > float(max_shift):
            too_far.append((shift_y, shift_x))
            continue
        # phase_cross_correlation returns the shift that maps the moving
        # image onto the reference, so b's true position is nominal + it.
        score, scored_px = _score_shift(
            tile_a, tile_b, reader_a, reader_b,
            nominal_dy + shift_y, nominal_dx + shift_x,
            reference_channel, min_overlap_px)
        if scored_px > 0 and score > best_score:
            best_score, best_px, best_shift = score, scored_px, (shift_y, shift_x)

    if best_shift is None:
        if too_far:
            shift_y, shift_x = too_far[0]
            return _replace_pair(
                base, overlap_px=height * width,
                note=f'measured shift ({shift_y:+.1f}, {shift_x:+.1f}) px '
                     f'exceeds max_shift={max_shift:g}; kept at the nominal '
                     f'position')
        return _replace_pair(
            base, overlap_px=height * width,
            note='the measured shift leaves no usable overlap to score')

    dy = nominal_dy + best_shift[0]
    dx = nominal_dx + best_shift[1]
    if best_score < float(min_confidence):
        return _replace_pair(
            base, dy=nominal_dy, dx=nominal_dx, confidence=max(0.0, best_score),
            overlap_px=best_px,
            note=f'cross-correlation {best_score:.3f} < min_confidence='
                 f'{min_confidence:g}; kept at the nominal position')
    return PairResult(i=tile_a.index, j=tile_b.index, dy=dy, dx=dx,
                      nominal_dy=nominal_dy, nominal_dx=nominal_dx,
                      confidence=float(best_score), accepted=True,
                      overlap_px=int(best_px), note='')


def _replace_pair(pair: PairResult, **changes: Any) -> PairResult:
    """Return ``pair`` with ``changes`` applied (frozen dataclasses)."""
    data = vars(pair).copy()
    data.update(changes)
    data['accepted'] = False
    return PairResult(**data)


def _score_shift(tile_a: Tile, tile_b: Tile, reader_a: _TileReader,
                 reader_b: _TileReader, dy: float, dx: float,
                 channel: int, min_overlap_px: int) -> Tuple[float, int]:
    """Score a candidate displacement by re-reading the implied overlap.

    Returns ``(ncc, pixels_compared)``. Re-reading is deliberate: the
    strips that phase correlation saw were cut at the *nominal* offset,
    so scoring them there would grade the answer against the question.
    """
    windows = _overlap_windows(tile_a, tile_b, dy, dx)
    if windows is None:
        return 0.0, 0
    (ay0, ay1, ax0, ax1), (by0, by1, bx0, bx1) = windows
    if (ay1 - ay0) < min_overlap_px or (ax1 - ax0) < min_overlap_px:
        return 0.0, 0
    strip_a = reader_a.window(ay0, ay1, ax0, ax1, [channel])[:, :, 0]
    strip_b = reader_b.window(by0, by1, bx0, bx1, [channel])[:, :, 0]
    return _ncc(strip_a, strip_b), int((ay1 - ay0) * (ax1 - ax0))


# ---------------------------------------------------------------------------
# The global solve
# ---------------------------------------------------------------------------

def solve_positions(n_tiles: int,
                    edges: Sequence[Tuple[int, int, float, float, float]],
                    nominal: np.ndarray,
                    *, anchor_weight: float = DEFAULT_ANCHOR_WEIGHT
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve every tile position at once from the pair graph.

    Each edge ``(i, j, dy, dx, w)`` contributes one weighted equation
    ``p_j - p_i = (dy, dx)``; each tile contributes one weak equation
    ``p_i = nominal_i``. Minimising the whole residual at once is what
    stops error accumulating.

    **Why not accumulate.** Walking a row and adding each measured shift
    to the previous position gives ``p_k = sum(d_0..d_k)``, so every
    measurement error is carried forward: ten tiles with 0.3 px errors put
    the last one 3 px out, and nothing in the output says so. Least
    squares over the *same* measurements distributes the error instead —
    and because a real grid has redundant edges (the tile below as well as
    the tile to the right, or the ``i``/``i+2`` pairs of a heavily
    overlapped row), disagreements cancel rather than compound. The
    surviving disagreement is the per-tile residual, which is returned.

    The anchor equations do two jobs: they fix the gauge (a pure pair
    graph determines positions only up to a global translation) and they
    place tiles that ended up in no edge at all — an isolated tile comes
    back at exactly its nominal position rather than at the origin.

    :param n_tiles: number of unknown positions.
    :param edges: ``(i, j, dy, dx, weight)`` per accepted pair.
    :param nominal: ``(n_tiles, 2)`` array of nominal ``(y, x)``.
    :param anchor_weight: weight of the "stay at nominal" equations.
    :returns: ``(positions, residuals, degrees)`` — ``(n, 2)`` solved
        positions, ``(n,)`` RMS per-tile residual in pixels, and ``(n,)``
        count of accepted edges per tile.
    """
    nominal = np.asarray(nominal, dtype=np.float64).reshape(n_tiles, 2)
    if n_tiles <= 0:
        return (np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=int))

    edge_list = [(int(i), int(j), float(dy), float(dx), float(w))
                 for i, j, dy, dx, w in edges]
    n_rows = len(edge_list) + n_tiles
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    rhs = np.zeros((n_rows, 2), dtype=np.float64)

    for r, (i, j, dy, dx, w) in enumerate(edge_list):
        weight = max(float(w), 1e-6)
        rows.extend((r, r))
        cols.extend((j, i))
        vals.extend((weight, -weight))
        rhs[r, 0] = weight * dy
        rhs[r, 1] = weight * dx
    for k in range(n_tiles):
        r = len(edge_list) + k
        rows.append(r)
        cols.append(k)
        vals.append(float(anchor_weight))
        rhs[r] = float(anchor_weight) * nominal[k]

    if n_tiles <= 512:
        design = np.zeros((n_rows, n_tiles), dtype=np.float64)
        for r, c, v in zip(rows, cols, vals):
            design[r, c] += v
        solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    else:
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import lsqr
        design = coo_matrix((vals, (rows, cols)),
                            shape=(n_rows, n_tiles)).tocsr()
        solution = np.column_stack([
            lsqr(design, rhs[:, k], atol=1e-12, btol=1e-12,
                 iter_lim=max(2000, 20 * n_tiles))[0]
            for k in range(2)])

    residual_sum = np.zeros(n_tiles, dtype=np.float64)
    degrees = np.zeros(n_tiles, dtype=int)
    for i, j, dy, dx, _w in edge_list:
        err_y = (solution[j, 0] - solution[i, 0]) - dy
        err_x = (solution[j, 1] - solution[i, 1]) - dx
        err = err_y * err_y + err_x * err_x
        residual_sum[i] += err
        residual_sum[j] += err
        degrees[i] += 1
        degrees[j] += 1
    residuals = np.zeros(n_tiles, dtype=np.float64)
    hit = degrees > 0
    residuals[hit] = np.sqrt(residual_sum[hit] / degrees[hit])
    return solution, residuals, degrees


def _sequential_positions(n_tiles: int,
                          edges: Sequence[Tuple[int, int, float, float, float]],
                          nominal: np.ndarray) -> np.ndarray:
    """Accumulate positions tile by tile — **the thing this module does not do**.

    The classic stitching loop: walk the tiles in acquisition order and
    place each one by adding its measured shift to its immediate
    predecessor's already-assigned position. It uses exactly one edge per
    tile — a spanning tree — and throws every other measurement away.

    That is why it drifts. ``p_k = p_0 + sum(d_0..d_k)`` carries every
    measurement error forward, so a systematic half-pixel bias in the
    overlap (which is what an asymmetric or vignetted overlap produces)
    puts the tenth tile five pixels out, and nothing in the output says
    so, because the algorithm has no second opinion to disagree with.

    Kept and documented as the counter-example that
    ``tests/test_align.py::test_global_solve_beats_sequential_drift``
    measures :func:`solve_positions` against. It is never called by
    :func:`estimate_offsets`.
    """
    nominal = np.asarray(nominal, dtype=np.float64).reshape(n_tiles, 2)
    positions = nominal.copy()
    # incoming[k] = [(j, dy, dx)] meaning "p_k = p_j + (dy, dx)".
    incoming: Dict[int, List[Tuple[int, float, float]]] = {}
    for i, j, dy, dx, _w in edges:
        incoming.setdefault(int(j), []).append((int(i), float(dy), float(dx)))
        incoming.setdefault(int(i), []).append((int(j), -float(dy), -float(dx)))
    placed = {0}
    for k in range(1, n_tiles):
        # The nearest already-placed tile in acquisition order: the
        # predecessor a sequential stitcher would chain from.
        options = [(abs(k - j), j, dy, dx)
                   for j, dy, dx in incoming.get(k, ())
                   if j in placed]
        if not options:
            placed.add(k)                # nothing to chain from; stage position
            continue
        _distance, source, dy, dx = min(options)
        positions[k, 0] = positions[source, 0] + dy
        positions[k, 1] = positions[source, 1] + dx
        placed.add(k)
    return positions


# ---------------------------------------------------------------------------
# estimate_offsets
# ---------------------------------------------------------------------------

def estimate_offsets(tiles: Sequence[Tile],
                     *,
                     reference_channel: int = 0,
                     min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                     min_overlap_px: int = DEFAULT_MIN_OVERLAP_PX,
                     upsample: int = DEFAULT_UPSAMPLE,
                     neighbour_radius: int = 1,
                     max_shift: Optional[float] = None,
                     anchor_weight: float = DEFAULT_ANCHOR_WEIGHT,
                     dtype: Optional[Any] = None,
                     max_open_tiles: int = 8,
                     ledger: Optional[RunLedger] = None) -> AlignPlan:
    """Register every overlapping neighbour pair and solve the whole set at once.

    Memory: the peak allocation is two overlap strips. For a 10% overlap
    on 2048x2048 tiles that is 2 x 0.8 MB, whatever the canvas turns out
    to be.

    :param tiles: from :func:`scan_tiles`, or built by hand.
    :param reference_channel: the plane registration is measured on. Every
        channel of a site then shares that site's single solution —
        registering channels independently would shear the composite.
    :param min_confidence: normalised cross-correlation below which a pair
        is refused and the tile falls back to nominal.
    :param min_overlap_px: overlaps narrower than this in either axis are
        not attempted.
    :param upsample: sub-pixel refinement factor; 10 resolves 0.1 px.
    :param neighbour_radius: Chebyshev grid distance to consider. 1 is the
        eight surrounding tiles. Raise it when the overlap exceeds 50%, so
        the ``i``/``i+2`` pairs become real redundancy in the solve.
    :param max_shift: refuse any pair whose correction exceeds this many
        pixels. Defaults to a quarter of the smaller tile dimension.
    :param anchor_weight: see :func:`solve_positions`.
    :param dtype: force the canvas dtype; defaults to the promotion of
        every tile's dtype.
    :param max_open_tiles: how many tiles may be memory-mapped at once.
        A column strip faults in a whole row-major tile, so leaving every
        tile mapped would make the resident set the size of the input
        folder; 8 covers the neighbourhood being registered.
    :param ledger: optional :class:`spacr.errors.RunLedger` to record
        per-tile success/failure into.
    :returns: an :class:`AlignPlan`; nothing has been allocated or written.
    :raises ConfigurationError: no readable tiles at all.
    """
    tiles = list(tiles)
    plan = AlignPlan(tiles=tiles, reference_channel=int(reference_channel))

    usable: List[Tile] = []
    for tile in tiles:
        if not tile.readable:
            plan.unplaced.append((tile, tile.error))
            if ledger is not None:
                ledger.record_failure(tile.path, stage='scan',
                                      exc=AlignError(tile.error))
            continue
        if tile.height <= 0 or tile.width <= 0:
            reason = f'header reports an empty shape {tile.shape}'
            plan.unplaced.append((tile, reason))
            if ledger is not None:
                ledger.record_failure(tile.path, stage='scan',
                                      exc=AlignError(reason))
            continue
        usable.append(tile)

    if not usable:
        raise ConfigurationError(
            'align: none of the {} tile(s) could be read. First problem: {}'
            .format(len(tiles),
                    plan.unplaced[0][1] if plan.unplaced else 'no tiles given'))

    # -- consistency ------------------------------------------------------
    dtypes = sorted({t.dtype for t in usable})
    if len(dtypes) > 1:
        plan.warnings.append(
            f'tiles have mixed dtypes ({", ".join(dtypes)}); the canvas will '
            f'be written as {np.result_type(*[np.dtype(d) for d in dtypes])}')
    shapes = sorted({(t.height, t.width) for t in usable})
    if len(shapes) > 1:
        plan.warnings.append(
            f'tiles have {len(shapes)} different shapes '
            f'({", ".join(f"{h}x{w}" for h, w in shapes)}); each is placed at '
            f'its own size and the canvas is sized to fit them all')
    channel_counts = sorted({t.n_channels for t in usable})
    if len(channel_counts) > 1:
        plan.warnings.append(
            f'tiles have different channel counts ({channel_counts}); the '
            f'canvas gets {max(channel_counts)} planes and short tiles '
            f'contribute nothing to the missing ones')
    plan.dtype = str(np.dtype(dtype) if dtype is not None
                     else np.result_type(*[np.dtype(d) for d in dtypes]))

    smallest = min(min(t.height, t.width) for t in usable)
    if max_shift is None:
        max_shift = max(4.0, 0.25 * smallest)

    # -- single tile ------------------------------------------------------
    if len(usable) == 1:
        only = usable[0]
        plan.placements = [Placement(
            tile=only, y=only.nominal_y, x=only.nominal_x, confidence=1.0,
            method=METHOD_SINGLE,
            note='only one tile — nothing to register against')]
        plan.warnings.append(
            'one tile only: the stitch is a copy, placed at its nominal '
            'position')
        plan.feather = 1
        _finalise_geometry(plan)
        if ledger is not None:
            ledger.record_success(only.path, stage='align')
        return plan

    # -- candidate pairs --------------------------------------------------
    candidates: List[Tuple[Tile, Tile]] = []
    for a_pos, tile_a in enumerate(usable):
        for tile_b in usable[a_pos + 1:]:
            if max(abs(tile_a.grid_row - tile_b.grid_row),
                   abs(tile_a.grid_col - tile_b.grid_col)) > int(neighbour_radius):
                continue
            if _overlap_windows(tile_a, tile_b,
                                tile_b.nominal_y - tile_a.nominal_y,
                                tile_b.nominal_x - tile_a.nominal_x) is None:
                continue
            candidates.append((tile_a, tile_b))

    cache = _ReaderCache(max_open=int(max_open_tiles))
    try:
        for tile_a, tile_b in candidates:
            try:
                pair = _register_pair(
                    tile_a, tile_b, cache.get(tile_a), cache.get(tile_b),
                    upsample=int(upsample), min_confidence=float(min_confidence),
                    min_overlap_px=int(min_overlap_px), max_shift=max_shift,
                    reference_channel=int(reference_channel))
            except Exception as exc:
                pair = PairResult(
                    i=tile_a.index, j=tile_b.index,
                    dy=tile_b.nominal_y - tile_a.nominal_y,
                    dx=tile_b.nominal_x - tile_a.nominal_x,
                    nominal_dy=tile_b.nominal_y - tile_a.nominal_y,
                    nominal_dx=tile_b.nominal_x - tile_a.nominal_x,
                    note=f'{type(exc).__name__}: {exc}')
            plan.overlaps.append(pair)
    finally:
        cache.close()

    if not candidates:
        plan.warnings.append(
            'no two tiles overlap at their nominal positions — every tile is '
            'placed by stage position alone. Check the grid, the order and '
            'the overlap fraction.')

    # -- global solve -----------------------------------------------------
    slot = {tile.index: k for k, tile in enumerate(usable)}
    nominal = np.array([[t.nominal_y, t.nominal_x] for t in usable],
                       dtype=np.float64)
    edges = [(slot[p.i], slot[p.j], p.dy, p.dx, p.confidence)
             for p in plan.overlaps if p.accepted]
    positions, residuals, degrees = solve_positions(
        len(usable), edges, nominal, anchor_weight=float(anchor_weight))

    best_conf = np.zeros(len(usable), dtype=np.float64)
    refused: Dict[int, List[str]] = {}
    for pair in plan.overlaps:
        if pair.accepted:
            for idx in (slot[pair.i], slot[pair.j]):
                best_conf[idx] = max(best_conf[idx], pair.confidence)
        else:
            for idx in (pair.i, pair.j):
                refused.setdefault(idx, []).append(
                    f'{pair.i}-{pair.j}: {pair.note}')

    placements: List[Placement] = []
    for k, tile in enumerate(usable):
        if degrees[k] > 0:
            method = METHOD_REGISTRATION
            note = ''
        else:
            method = METHOD_NOMINAL
            reasons = refused.get(tile.index, [])
            note = ('; '.join(reasons[:3]) if reasons
                    else 'no overlapping neighbour to register against')
        placements.append(Placement(
            tile=tile, y=float(positions[k, 0]), x=float(positions[k, 1]),
            confidence=float(best_conf[k]), method=method, note=note,
            residual=float(residuals[k]), n_pairs=int(degrees[k])))
        if ledger is not None:
            if method == METHOD_REGISTRATION:
                ledger.record_success(tile.path, stage='align')
            else:
                ledger.record_failure(
                    tile.path, stage='align',
                    exc=AlignError(f'placed by nominal position: {note}'))
    plan.placements = placements

    if plan.n_nominal:
        plan.warnings.append(
            f'{plan.n_nominal} of {len(usable)} tile(s) did not register and '
            f'are placed at their nominal stage position — they are marked '
            f"method='{METHOD_NOMINAL}' in the plan and in the database.")

    n_components = _count_components(len(usable), edges)
    if n_components > 1:
        plan.warnings.append(
            f'the pair graph has {n_components} disconnected component(s); '
            f'components are positioned relative to each other by stage '
            f'coordinates only')

    plan.feather = _feather_width(plan, usable)
    _finalise_geometry(plan)
    return plan


def _count_components(n_tiles: int,
                      edges: Sequence[Tuple[int, int, float, float, float]]) -> int:
    """Number of connected components in the accepted-pair graph."""
    parent = list(range(n_tiles))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, *_rest in edges:
        ra, rb = find(int(i)), find(int(j))
        if ra != rb:
            parent[ra] = rb
    return len({find(k) for k in range(n_tiles)})


def _feather_width(plan: AlignPlan, tiles: Sequence[Tile]) -> int:
    """Pick the seam ramp width from the overlaps that actually exist.

    The ramp should span the overlap: narrower leaves a visible edge just
    inside it, wider pushes the blend into pixels only one tile covers.
    """
    widths: List[int] = []
    by_index = {t.index: t for t in tiles}
    for pair in plan.overlaps:
        tile_a, tile_b = by_index.get(pair.i), by_index.get(pair.j)
        if tile_a is None or tile_b is None:
            continue
        windows = _overlap_windows(tile_a, tile_b, pair.dy, pair.dx)
        if windows is None:
            continue
        (ay0, ay1, ax0, ax1), _ = windows
        span = min(ay1 - ay0, ax1 - ax0)
        if span > 0:
            widths.append(int(span))
    smallest = min(min(t.height, t.width) for t in tiles)
    if not widths:
        return 1
    return int(max(1, min(int(np.median(widths)), smallest // 2)))


def _finalise_geometry(plan: AlignPlan) -> None:
    """Fill ``plan.canvas_shape`` and ``plan.origin`` from the placements."""
    spec = plan_canvas(plan.placements, dtype=plan.dtype)
    plan.canvas_shape = spec.shape
    plan.origin = (spec.origin_y, spec.origin_x)


# ---------------------------------------------------------------------------
# Canvas geometry
# ---------------------------------------------------------------------------

def plan_canvas(placements: Sequence[Placement],
                *, dtype: Optional[Any] = None,
                channels: Optional[int] = None) -> CanvasSpec:
    """Compute the canvas geometry from the offsets, allocating nothing.

    Negative offsets are the normal case — the solve fixes the gauge to
    the nominal centroid, so a tile can easily land above or left of the
    origin. The canvas is sized to the bounding box of every placed tile
    and :attr:`CanvasSpec.origin_y` / :attr:`~CanvasSpec.origin_x` carry
    the mapping back to the global frame, so nothing is ever clipped.

    :param placements: from :attr:`AlignPlan.placements`.
    :param dtype: canvas dtype; defaults to promoting every tile's.
    :param channels: canvas planes; defaults to the widest tile.
    :returns: a :class:`CanvasSpec`. ``height``/``width`` are 0 for an
        empty placement list.
    """
    placements = list(placements)
    if not placements:
        return CanvasSpec(height=0, width=0, channels=int(channels or 0),
                          dtype=str(np.dtype(dtype or 'uint16')))
    top = min(p.y for p in placements)
    left = min(p.x for p in placements)
    bottom = max(p.y + p.tile.height for p in placements)
    right = max(p.x + p.tile.width for p in placements)
    # Snap away solver noise before flooring: a solved 0.0 that came back
    # as -3e-13 would otherwise cost the canvas a whole row of padding.
    origin_y = math.floor(round(top, 6))
    origin_x = math.floor(round(left, 6))
    height = int(math.ceil(round(bottom - origin_y, 6)))
    width = int(math.ceil(round(right - origin_x, 6)))
    if dtype is None:
        dtype = np.result_type(*[np.dtype(p.tile.dtype) for p in placements])
    if channels is None:
        channels = max(p.tile.n_channels for p in placements)
    return CanvasSpec(height=max(0, height), width=max(0, width),
                      channels=int(channels), dtype=str(np.dtype(dtype)),
                      origin_y=float(origin_y), origin_x=float(origin_x))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _band_bytes_per_row(spec: CanvasSpec) -> int:
    """Bytes of RAM one canvas row of band costs.

    A float32 accumulator (``4 * channels``), a float32 weight plane
    (``4``) and the cast output band (``itemsize * channels``). All three
    are live at once at the moment the band is written, so all three are
    counted — a budget that ignores the output band understates the peak
    by a third.
    """
    return int(spec.width) * (4 * int(spec.channels) + 4
                              + int(np.dtype(spec.dtype).itemsize)
                              * int(spec.channels))


def _band_rows(spec: CanvasSpec, max_buffer_bytes: int,
               requested: Optional[int]) -> int:
    """Rows of canvas held in RAM at once.

    Solving :func:`_band_bytes_per_row` for ``rows`` under
    ``max_buffer_bytes`` is what makes the footprint independent of the
    canvas height.
    """
    if spec.height <= 0 or spec.width <= 0:
        return 1
    if requested is not None:
        return int(max(1, min(int(requested), spec.height)))
    rows = int(max_buffer_bytes) // max(1, _band_bytes_per_row(spec))
    return int(max(1, min(rows if rows > 0 else 1, spec.height)))


def _ramp(start: int, stop: int, extent: int, feather: int) -> np.ndarray:
    """Linear feather weight for tile-local indices ``[start, stop)``.

    Rises from :data:`_WEIGHT_FLOOR` at the tile edge to 1.0 ``feather``
    pixels in, and mirrors at the far edge. The floor matters: a pixel
    covered only by one tile's outermost row must still take that tile's
    value, not a zero.
    """
    index = np.arange(int(start), int(stop), dtype=np.float32)
    distance = np.minimum(index + 0.5, float(extent) - 0.5 - index)
    if feather <= 0:
        return np.ones_like(distance)
    return np.clip(distance / float(feather), _WEIGHT_FLOOR, 1.0)


def _cast_to(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a float accumulator to the canvas dtype, rounding and clipping.

    Rounds and clips **in place**. Written the obvious way —
    ``np.clip(np.rint(x), lo, hi).astype(dtype)`` — this allocates two more
    band-sized float32 temporaries, so a 256 MB band costs 768 MB of peak.
    That is a lot of memory to spend on an expression that does not need
    any of it.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        np.rint(values, out=values)
        np.clip(values, float(info.min), float(info.max), out=values)
    return values.astype(dtype)


def _drop_pages(array) -> None:
    """Best-effort: tell the kernel the written pages are no longer needed.

    A shared file mapping keeps dirty pages resident until they are
    flushed *and* dropped. Without this the memmap writer's RSS climbs to
    the canvas size even though only one band is ever live, which would
    make the whole memory argument untrue in the only measurement that
    matters. Silently a no-op where madvise is unavailable.
    """
    handle = getattr(array, '_mmap', None)
    advise = getattr(handle, 'madvise', None)
    import mmap as _mmap
    flag = getattr(_mmap, 'MADV_DONTNEED', None)
    if advise is None or flag is None:
        return
    try:
        advise(flag)
    except (OSError, ValueError):
        pass


class _StreamCanvas:
    """Sequential band writer for a ``.npy``, with no mapping at all.

    :func:`numpy.lib.format.open_memmap` creates the file (correct header,
    correct length) and is then closed; the bands go in through a plain
    buffered file handle at the data offset. Because ``.npy`` is
    C-contiguous row-major and bands are written top to bottom, "seek to
    ``offset + y0 * row_bytes`` and write" is exactly right — and the
    process never maps a byte of the canvas, so peak RSS is the band and
    nothing else.
    """

    def __init__(self, path: str, spec: CanvasSpec):
        self.path = path
        self.spec = spec
        array = np.lib.format.open_memmap(
            path, mode='w+', dtype=np.dtype(spec.dtype), shape=spec.shape)
        array.flush()
        del array
        self._row_bytes = (int(spec.width) * int(spec.channels)
                           * int(np.dtype(spec.dtype).itemsize))
        self._offset = os.path.getsize(path) - spec.nbytes
        self._handle = open(path, 'r+b')

    def write_band(self, y0: int, block: np.ndarray) -> None:
        """Write ``block`` at canvas row ``y0``.

        Handed to ``write`` as a buffer, not via ``tobytes()``: the latter
        would copy the whole band again on its way out.
        """
        self._handle.seek(self._offset + int(y0) * self._row_bytes)
        self._handle.write(np.ascontiguousarray(block).data)

    def close(self) -> None:
        """Flush and fsync, then close."""
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()


class _MemmapCanvas:
    """Band writer backed by :func:`numpy.lib.format.open_memmap`.

    Each band is assigned, flushed and then dropped from the process's
    resident set (see :func:`_drop_pages`), so the mapping stays a
    file-backed window rather than becoming an 800 MB anonymous
    allocation by another name.
    """

    def __init__(self, path: str, spec: CanvasSpec):
        self.path = path
        self.spec = spec
        self.array = np.lib.format.open_memmap(
            path, mode='w+', dtype=np.dtype(spec.dtype), shape=spec.shape)

    def write_band(self, y0: int, block: np.ndarray) -> None:
        """Write ``block`` at canvas row ``y0`` and release its pages."""
        self.array[int(y0):int(y0) + block.shape[0]] = block
        self.array.flush()
        _drop_pages(self.array)

    def close(self) -> None:
        """Flush and release the mapping."""
        self.array.flush()
        self.array = None


def write_stack(plan: AlignPlan, dst: Union[str, os.PathLike],
                *,
                blend: str = 'feather',
                band_rows: Optional[int] = None,
                max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
                feather: Optional[int] = None,
                dtype: Optional[Any] = None,
                subpixel: bool = False,
                writer: str = 'stream',
                max_open_tiles: int = 8,
                overwrite: bool = False,
                name: Optional[str] = None,
                dry_run: bool = False,
                ledger: Optional[RunLedger] = None) -> AlignResult:
    """Composite the plan into one ``.npy``, one horizontal band at a time.

    The canvas is never an in-RAM array. The output file is created at
    full size up front (so a disk that cannot hold it fails immediately,
    not at 90%), then filled band by band; the only heap allocations are
    the band's float32 accumulator, its weight plane, and one tile window.
    All three are bounded by ``max_buffer_bytes`` and none of them scales
    with the canvas.

    Seams are **feathered**: each tile's contribution is weighted by a
    linear ramp that rises from the tile edge inward over
    :attr:`AlignPlan.feather` pixels, and the band is divided by the total
    weight. A hard cut would leave a straight, high-contrast edge exactly
    where two fields meet — and a straight edge is the single most
    reliable thing a downstream segmentation will find and call an object
    boundary. Feathering costs one extra float32 plane per band and makes
    the join a gradient instead. ``blend='average'`` weights every
    contributor equally and ``blend='none'`` is last-writer-wins, both
    kept mainly so the difference can be measured.

    :param plan: from :func:`estimate_offsets`.
    :param dst: output ``.npy``, or a folder to write into.
    :param blend: one of :data:`BLEND_MODES`.
    :param band_rows: force the band height; normally derived from
        ``max_buffer_bytes``.
    :param max_buffer_bytes: ceiling on the RAM the write may use for the
        band. This is the knob; see :data:`DEFAULT_MAX_BUFFER_BYTES` for
        why the default is small.
    :param feather: override the ramp width in pixels.
    :param dtype: override the canvas dtype.
    :param subpixel: resample each tile by its fractional offset before
        compositing. Off by default: it costs an interpolation per tile
        window, and the sub-pixel part is recorded in the table either
        way.
    :param writer: ``'stream'`` (default; never maps the canvas) or
        ``'memmap'`` (``open_memmap`` plus per-band flush and drop).
    :param max_open_tiles: how many tiles may be mapped at once. The
        tiles intersecting one band are the working set; older mappings
        are dropped so the input folder never becomes resident.
    :param overwrite: replace an existing output. False refuses.
    :param name: output filename when ``dst`` is a folder.
    :param dry_run: compute the geometry and the band plan, write nothing.
    :param ledger: optional :class:`spacr.errors.RunLedger`.
    :returns: an :class:`AlignResult`, whose
        :attr:`~AlignResult.peak_buffer_bytes` is the measured allocation
        ceiling, not an estimate.
    :raises ConfigurationError: bad ``blend``/``writer``, or an existing
        output without ``overwrite``.
    """
    if blend not in BLEND_MODES:
        raise ConfigurationError(
            f'blend must be one of {BLEND_MODES}, got {blend!r}')
    if writer not in ('stream', 'memmap'):
        raise ConfigurationError(
            f"writer must be 'stream' or 'memmap', got {writer!r}")

    placements = [p for p in plan.placements if p.tile.readable]
    spec = plan_canvas(placements, dtype=dtype or plan.dtype)
    result = AlignResult(plan=plan, canvas=spec, writer=writer)
    if not placements or spec.height <= 0 or spec.width <= 0:
        result.warnings.append('nothing to write: no placed tiles')
        return result

    rows = _band_rows(spec, int(max_buffer_bytes), band_rows)
    result.band_rows = rows
    result.peak_buffer_bytes = int(rows) * _band_bytes_per_row(spec)

    if dry_run:
        result.status = 'empty'
        result.warnings.append('dry_run: geometry computed, nothing written')
        return result

    dst = os.fspath(dst)
    if os.path.isdir(dst) or not dst.lower().endswith('.npy'):
        first = placements[0].tile
        stem = name or (f'{first.plate}_{first.well}_stitched.npy'
                        if first.plate or first.well else 'stitched.npy')
        os.makedirs(dst, exist_ok=True)
        out_path = os.path.join(dst, stem)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(dst)) or '.', exist_ok=True)
        out_path = dst
    if os.path.exists(out_path) and not overwrite:
        raise ConfigurationError(
            f'{out_path} already exists — pass overwrite=True to replace it.')

    ramp_width = int(feather) if feather is not None else int(plan.feather)
    out_dtype = np.dtype(spec.dtype)

    # Integer canvas positions, plus the sub-pixel remainder.
    boxes: List[Tuple[Placement, int, int, float, float]] = []
    for placement in placements:
        cy, cx = spec.canvas_yx(placement.y, placement.x)
        boxes.append((placement, cy, cx,
                      float(placement.y - spec.origin_y - cy),
                      float(placement.x - spec.origin_x - cx)))

    canvas = (_StreamCanvas(out_path, spec) if writer == 'stream'
              else _MemmapCanvas(out_path, spec))
    cache = _ReaderCache(max_open=int(max_open_tiles))
    contributed: set = set()
    failed: set = set()
    try:
        for y0 in range(0, spec.height, rows):
            y1 = min(spec.height, y0 + rows)
            band_h = y1 - y0
            acc = np.zeros((band_h, spec.width, spec.channels), dtype=np.float32)
            wsum = np.zeros((band_h, spec.width), dtype=np.float32)
            for placement, cy, cx, fy, fx in boxes:
                tile = placement.tile
                th, tw = tile.height, tile.width
                if cy + th <= y0 or cy >= y1 or cx + tw <= 0 or cx >= spec.width:
                    continue
                ty0, ty1 = max(0, y0 - cy), min(th, y1 - cy)
                tx0, tx1 = max(0, -cx), min(tw, spec.width - cx)
                if ty1 <= ty0 or tx1 <= tx0:
                    continue
                if tile.index in failed:
                    continue
                try:
                    reader = cache.get(tile)
                except Exception as exc:
                    failed.add(tile.index)
                    result.n_skipped += 1
                    result.warnings.append(
                        f'{tile.path}: {type(exc).__name__}: {exc}')
                    if ledger is not None:
                        ledger.record_failure(tile.path, stage='write',
                                              exc=exc)
                    continue
                margin = 2 if subpixel and (fy or fx) else 0
                try:
                    block = reader.window(
                        ty0 - margin, ty1 + margin, tx0 - margin, tx1 + margin,
                        list(range(min(tile.n_channels, spec.channels))))
                except Exception as exc:
                    failed.add(tile.index)
                    result.n_skipped += 1
                    result.warnings.append(
                        f'{tile.path}: {type(exc).__name__}: {exc}')
                    if ledger is not None:
                        ledger.record_failure(tile.path, stage='write', exc=exc)
                    continue
                if margin:
                    from scipy.ndimage import shift as _shift
                    block = _shift(block, (fy, fx, 0), order=1, mode='nearest')
                    block = block[margin:margin + (ty1 - ty0),
                                  margin:margin + (tx1 - tx0)]
                if blend == 'feather':
                    weight = (_ramp(ty0, ty1, th, ramp_width)[:, None]
                              * _ramp(tx0, tx1, tw, ramp_width)[None, :])
                else:
                    weight = np.ones((ty1 - ty0, tx1 - tx0), dtype=np.float32)
                dy0, dx0 = cy + ty0 - y0, cx + tx0
                dy1, dx1 = dy0 + (ty1 - ty0), dx0 + (tx1 - tx0)
                if blend == 'none':
                    acc[dy0:dy1, dx0:dx1, :block.shape[2]] = block
                    wsum[dy0:dy1, dx0:dx1] = 1.0
                else:
                    acc[dy0:dy1, dx0:dx1, :block.shape[2]] += \
                        block * weight[:, :, None]
                    wsum[dy0:dy1, dx0:dx1] += weight
                contributed.add(tile.index)
            np.maximum(wsum, 1e-6, out=wsum)
            acc /= wsum[:, :, None]
            canvas.write_band(y0, _cast_to(acc, out_dtype))
            del acc, wsum
    finally:
        canvas.close()
        cache.close()

    result.stack_path = out_path
    result.n_written = len(contributed)
    result.status = ('complete' if result.n_skipped == 0 else 'partial')
    if contributed and result.n_written < len(placements) and not failed:
        result.warnings.append(
            f'{len(placements) - result.n_written} placed tile(s) fell '
            f'entirely outside the canvas and contributed no pixels')
    if ledger is not None:
        for placement in placements:
            if placement.tile.index in contributed:
                ledger.record_success(placement.tile.path, stage='write')
        try:
            ledger.stamp(out_path)
        except Exception as exc:                     # stamping must not fail a run
            result.warnings.append(f'could not stamp {out_path}: {exc}')
    return result


# ---------------------------------------------------------------------------
# The coordinates table
# ---------------------------------------------------------------------------

#: Columns of :data:`ALIGN_TABLE`, in order.
ALIGN_COLUMNS: Tuple[str, ...] = (
    'plateID', 'rowID', 'columnID', 'fieldID', 'prc', 'prcf',
    'plate', 'well', 'field', 'tile_index',
    'source', 'source_channels', 'stack_path',
    'y', 'x', 'canvas_y', 'canvas_x', 'subpixel_y', 'subpixel_x',
    'nominal_y', 'nominal_x', 'grid_row', 'grid_col',
    'method', 'confidence', 'residual', 'n_pairs',
    'tile_height', 'tile_width', 'tile_channels', 'tile_dtype',
    'canvas_height', 'canvas_width', 'canvas_channels', 'canvas_dtype',
    'origin_y', 'origin_x', 'reference_channel', 'note',
)


def _coordinate_rows(plan: AlignPlan, canvas: Optional[CanvasSpec],
                     stack_path: str) -> List[Dict[str, Any]]:
    """Render one plan as :data:`ALIGN_COLUMNS` rows."""
    spec = canvas or plan_canvas(plan.placements, dtype=plan.dtype)
    rows: List[Dict[str, Any]] = []
    for placement in plan.placements:
        tile = placement.tile
        cy, cx = spec.canvas_yx(placement.y, placement.x)
        row = _join_keys(tile.plate, tile.well, tile.field)
        row.update({
            'plate': tile.plate,
            'well': tile.well,
            'field': int(tile.field),
            'tile_index': int(tile.index),
            'source': tile.path,
            'source_channels': json.dumps(list(tile.channel_paths)),
            'stack_path': stack_path,
            'y': float(placement.y),
            'x': float(placement.x),
            'canvas_y': int(cy),
            'canvas_x': int(cx),
            'subpixel_y': float(placement.y - spec.origin_y - cy),
            'subpixel_x': float(placement.x - spec.origin_x - cx),
            'nominal_y': float(tile.nominal_y),
            'nominal_x': float(tile.nominal_x),
            'grid_row': int(tile.grid_row),
            'grid_col': int(tile.grid_col),
            'method': str(placement.method),
            'confidence': float(placement.confidence),
            'residual': float(placement.residual),
            'n_pairs': int(placement.n_pairs),
            'tile_height': int(tile.height),
            'tile_width': int(tile.width),
            'tile_channels': int(tile.n_channels),
            'tile_dtype': str(tile.dtype),
            'canvas_height': int(spec.height),
            'canvas_width': int(spec.width),
            'canvas_channels': int(spec.channels),
            'canvas_dtype': str(spec.dtype),
            'origin_y': float(spec.origin_y),
            'origin_x': float(spec.origin_x),
            'reference_channel': int(plan.reference_channel),
            'note': str(placement.note),
        })
        rows.append(row)
    for tile, reason in plan.unplaced:
        row = _join_keys(tile.plate, tile.well, tile.field)
        row.update({
            'plate': tile.plate, 'well': tile.well, 'field': int(tile.field),
            'tile_index': int(tile.index), 'source': tile.path,
            'source_channels': json.dumps(list(tile.channel_paths)),
            'stack_path': '', 'y': float('nan'), 'x': float('nan'),
            'canvas_y': -1, 'canvas_x': -1,
            'subpixel_y': 0.0, 'subpixel_x': 0.0,
            'nominal_y': float(tile.nominal_y), 'nominal_x': float(tile.nominal_x),
            'grid_row': int(tile.grid_row), 'grid_col': int(tile.grid_col),
            'method': METHOD_UNREADABLE, 'confidence': 0.0,
            'residual': float('nan'), 'n_pairs': 0,
            'tile_height': int(tile.height), 'tile_width': int(tile.width),
            'tile_channels': int(tile.n_channels), 'tile_dtype': str(tile.dtype),
            'canvas_height': int(spec.height), 'canvas_width': int(spec.width),
            'canvas_channels': int(spec.channels), 'canvas_dtype': str(spec.dtype),
            'origin_y': float(spec.origin_y), 'origin_x': float(spec.origin_x),
            'reference_channel': int(plan.reference_channel),
            'note': str(reason),
        })
        rows.append(row)
    return rows


def save_coordinates(plan: Union[AlignPlan, Iterable[AlignPlan]],
                     db_path: Union[str, os.PathLike],
                     *,
                     table: str = ALIGN_TABLE,
                     canvas: Optional[CanvasSpec] = None,
                     stack_path: str = '',
                     if_exists: str = 'replace') -> int:
    """Write one row per tile into ``measurements.db``.

    The point of this table is that a stitch is a *claim*: "field 47 of
    B07 sits at canvas row 3641, column 8192". Downstream, a cluster of
    odd measurements in one corner of a well is either biology or a tile
    that silently fell back to its nominal position, and only this table
    can tell the two apart. So every row carries ``method``,
    ``confidence`` and ``residual`` next to the coordinates.

    It is keyed exactly the way ``conversion_map`` and every measurement
    table are keyed, so it joins on the same four columns::

        SELECT c.*, a.canvas_y, a.canvas_x, a.method, a.residual
        FROM cell AS c
        JOIN align_coordinates AS a
          ON  c.plateID  = a.plateID
          AND c.rowID    = a.rowID
          AND c.columnID = a.columnID
          AND c.fieldID  = a.fieldID

    or on the single ``prcf`` column, which is indexed. The same join
    works against ``nucleus`` / ``pathogen`` / ``cytoplasm`` / ``png_list``
    and against ``conversion_map`` itself, which chains a stitched
    coordinate all the way back to the original vendor file.

    Tiles that could never be read are written too, with
    ``method='unreadable'`` and NULL coordinates — a missing field must be
    visible in the table, not absent from it.

    :param plan: one :class:`AlignPlan` or several (one per well).
    :param db_path: SQLite database; created if missing.
    :param table: table name.
    :param canvas: the geometry actually written, when it differs from
        the plan's.
    :param stack_path: the ``.npy`` these coordinates index into.
    :param if_exists: ``'replace'`` (default), ``'append'`` or ``'fail'``.
    :returns: number of rows written.
    """
    plans = [plan] if isinstance(plan, AlignPlan) else list(plan)
    rows: List[Dict[str, Any]] = []
    for one in plans:
        rows.extend(_coordinate_rows(one, canvas, stack_path))
    frame = pd.DataFrame(rows, columns=list(ALIGN_COLUMNS))

    parent = os.path.dirname(os.path.abspath(os.fspath(db_path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        frame.to_sql(table, connection, if_exists=if_exists, index=False)
        connection.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{table}_prcf ON {table} (prcf)')
        connection.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{table}_keys ON {table} '
            f'(plateID, rowID, columnID, fieldID)')
        connection.commit()
    finally:
        connection.close()
    return int(len(frame))


def read_coordinates(db_path: Union[str, os.PathLike],
                     *, table: str = ALIGN_TABLE,
                     plate: Optional[str] = None,
                     well: Optional[str] = None) -> pd.DataFrame:
    """Read :data:`ALIGN_TABLE` back out, closing the loop.

    :param db_path: the database :func:`save_coordinates` wrote.
    :param table: table name.
    :param plate: optionally restrict to one ``plateID``.
    :param well: optionally restrict to one well, given as ``'B07'`` or as
        an already-mapped ``rowID``/``columnID`` pair joined by ``_``.
    :returns: the table as a DataFrame, in tile order.
    :raises ConfigurationError: the database or the table is missing.
    """
    path = os.fspath(db_path)
    if not os.path.isfile(path):
        raise ConfigurationError(f'align: no such database: {path}')
    connection = sqlite3.connect(path, timeout=30)
    try:
        present = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)).fetchone()
        if not present:
            raise ConfigurationError(
                f'{path} has no {table!r} table — run align (or '
                f'save_coordinates) against this database first.')
        query = f'SELECT * FROM {table}'
        clauses: List[str] = []
        params: List[Any] = []
        if plate is not None:
            clauses.append('plateID = ?')
            params.append(str(plate))
        if well is not None:
            row_id, column_id = _well_ids(str(well))
            clauses.append('(well = ? OR (rowID = ? AND columnID = ?))')
            params.extend([str(well), row_id, column_id])
        if clauses:
            query += ' WHERE ' + ' AND '.join(clauses)
        frame = pd.read_sql_query(query, connection, params=params)
    finally:
        connection.close()
    if 'tile_index' in frame.columns:
        frame = frame.sort_values(['plateID', 'well', 'tile_index'],
                                  kind='stable').reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_plan(plan: AlignPlan, *, max_rows: int = 12) -> str:
    """Render a plan as the block a user should read before writing 800 MB.

    Leads with what is *wrong* — nominal fallbacks, refused pairs, the
    worst residual — because that is the part a stitch summary usually
    buries.
    """
    height, width, channels = plan.canvas_shape
    lines = [
        f'Align plan: {len(plan.placements)} tile(s) placed, '
        f'{len(plan.unplaced)} unplaced.',
        f'  canvas          {height} x {width} x {channels} {plan.dtype} '
        f'({_human_bytes(plan.canvas_bytes)} on disk)',
        f'  origin          y={plan.origin[0]:.2f}, x={plan.origin[1]:.2f}',
        f'  registered      {plan.n_registered}',
        f'  nominal         {plan.n_nominal}',
        f'  pairs           {len(plan.accepted_pairs)} accepted / '
        f'{len(plan.overlaps)} scored',
        f'  worst residual  {plan.max_residual:.2f} px',
        f'  feather         {plan.feather} px',
        f'  reference chan  {plan.reference_channel}',
    ]
    fallbacks = plan.nominal_placements()
    if fallbacks:
        lines.append(f'  -- placed by stage position only ({len(fallbacks)}) --')
        for placement in fallbacks[:max_rows]:
            lines.append(
                f'     {placement.tile.name:<28} '
                f'y={placement.y:9.2f} x={placement.x:9.2f}  {placement.note}')
        if len(fallbacks) > max_rows:
            lines.append(f'     … and {len(fallbacks) - max_rows} more')
    worst = sorted((p for p in plan.placements if p.n_pairs),
                   key=lambda p: -p.residual)[:max_rows]
    if worst and worst[0].residual > 0:
        lines.append('  -- largest residuals --')
        for placement in worst:
            lines.append(
                f'     {placement.tile.name:<28} '
                f'residual={placement.residual:7.2f} px  '
                f'conf={placement.confidence:.3f}  pairs={placement.n_pairs}')
    for tile, reason in plan.unplaced[:max_rows]:
        lines.append(f'  !! unplaced {tile.path}: {reason}')
    for warning in plan.warnings:
        lines.append(f'  ! {warning}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Settings entry point
# ---------------------------------------------------------------------------

def default_settings(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return the settings :func:`align_folder` understands, with defaults.

    Shaped like every other ``spacr`` settings factory — pass a partial
    dict, get it back filled in — so the CLI and the Qt bridge can build a
    panel from it without special-casing this module.

    :param settings: partial settings; keys given here win.
    """
    resolved: Dict[str, Any] = {
        'src': None,
        'dst': None,
        'db_path': None,
        'grid': None,
        'overlap': DEFAULT_OVERLAP,
        'order': 'row-major',
        'recursive': False,
        'reference_channel': 0,
        'min_confidence': DEFAULT_MIN_CONFIDENCE,
        'neighbour_radius': 1,
        'upsample': DEFAULT_UPSAMPLE,
        'min_overlap_px': DEFAULT_MIN_OVERLAP_PX,
        'max_shift': None,
        'blend': 'feather',
        'feather': None,
        'subpixel': False,
        'writer': 'stream',
        'max_buffer_bytes': DEFAULT_MAX_BUFFER_BYTES,
        'band_rows': None,
        'save_stack': True,
        'overwrite': False,
        'preview_only': False,
        'group_by_well': True,
    }
    resolved.update(dict(settings or {}))
    return resolved


def align_folder(settings: Optional[Mapping[str, Any]] = None,
                 **overrides: Any) -> List[AlignResult]:
    """Scan, plan, optionally write and optionally record, in one call.

    Always prints the plan before writing anything, so even a headless run
    leaves the "these four tiles fell back to nominal" block in the log
    where a surprised user can find it.

    :param settings: see :func:`default_settings`.
    :param overrides: keyword form of the same keys; these win.
    :returns: one :class:`AlignResult` per stitched group (one per well
        when ``group_by_well``).
    :raises ConfigurationError: no ``src``, or no readable tiles.
    """
    resolved = default_settings(settings)
    resolved.update(overrides)

    src = resolved.get('src')
    if not src:
        raise ConfigurationError(
            "align_folder needs a 'src' folder of tiles to stitch.")

    tiles = scan_tiles(src,
                       grid=resolved.get('grid'),
                       overlap=float(resolved.get('overlap') or 0.0),
                       order=str(resolved.get('order') or 'row-major'),
                       recursive=bool(resolved.get('recursive')),
                       reference_channel=None)
    groups = (group_tiles(tiles) if resolved.get('group_by_well')
              else {('', ''): tiles})

    dst = resolved.get('dst')
    if dst is None and not resolved.get('preview_only'):
        base = src if isinstance(src, (str, os.PathLike)) else '.'
        dst = os.path.normpath(os.fspath(base)) + '_stitched'

    ledger = RunLedger('align')
    results: List[AlignResult] = []
    first_write = True
    for key, members in sorted(groups.items()):
        plan = estimate_offsets(
            members,
            reference_channel=int(resolved.get('reference_channel') or 0),
            min_confidence=float(resolved.get('min_confidence')),
            min_overlap_px=int(resolved.get('min_overlap_px')),
            upsample=int(resolved.get('upsample')),
            neighbour_radius=int(resolved.get('neighbour_radius')),
            max_shift=resolved.get('max_shift'),
            ledger=ledger)
        label = '_'.join(part for part in key if part) or os.path.basename(str(src))
        print(f'--- {label} ---')
        print(format_plan(plan))

        if resolved.get('preview_only') or not resolved.get('save_stack'):
            spec = plan_canvas(plan.placements, dtype=plan.dtype)
            results.append(AlignResult(plan=plan, canvas=spec,
                                       band_rows=_band_rows(
                                           spec,
                                           int(resolved['max_buffer_bytes']),
                                           resolved.get('band_rows')),
                                       warnings=['preview only — nothing written']))
        else:
            result = write_stack(
                plan, dst,
                blend=str(resolved.get('blend') or 'feather'),
                band_rows=resolved.get('band_rows'),
                max_buffer_bytes=int(resolved['max_buffer_bytes']),
                feather=resolved.get('feather'),
                subpixel=bool(resolved.get('subpixel')),
                writer=str(resolved.get('writer') or 'stream'),
                overwrite=bool(resolved.get('overwrite')),
                name=(f'{label}_stitched.npy' if label else None))
            print(result.summary())
            results.append(result)

        db_path = resolved.get('db_path')
        if db_path:
            written = save_coordinates(
                plan, db_path, canvas=results[-1].canvas,
                stack_path=results[-1].stack_path,
                if_exists='replace' if first_write else 'append')
            results[-1].db_path = str(db_path)
            first_write = False
            print(f'Wrote {written} {ALIGN_TABLE} row(s) into {db_path}.')

    ledger.finalize()
    return results
