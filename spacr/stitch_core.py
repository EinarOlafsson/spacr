"""One registration, solve and composition core for every tile stitch (item 936).

spaCR stitches tiles in two places that matter: the general Align & Stitch
module (:mod:`spacr.align`) and the optical pooled screening chain
(:mod:`spacr.ops_register`, :mod:`spacr.ops_solve`, :mod:`spacr.ops_stitch`,
:mod:`spacr.ops_compose`). The OPS chain is the one proven against reference
data: 624 of 624 edges on well A1 of Funk et al. 2022, a deterministic solve,
feathered seams that score like an unstitched tile (item 931), float32
registration at half the CPU time (the S1 stitch speed card). This module is
the one place both callers take that code from:

* the registration: :func:`phase_correlate`, :func:`unwrap`,
  :func:`available_backends`, :func:`register_edge`, the float32 transform
  engine :class:`FFTEngine` (item 932), and :func:`register_strip_pair`, which
  adds the NCC check and sub-pixel refinement Align & Stitch needs on top of
  the same strip registration;
* the solve: :func:`solve_positions`, one least squares with two gauges,
  ``'lowest_site'`` (OPS) and ``'anchor'`` (Align & Stitch), on one BLAS
  thread;
* the composition: :func:`windows_over`, :func:`compose_window` and
  :func:`feather_weights`, the per-tile weights both the OPS window writer
  and the Align & Stitch band writer use;
* the placement seed (item 940): :func:`write_placement_seed`,
  :func:`load_placement_seed`, :func:`validate_placement_seed`,
  :func:`seed_to_placements` and :func:`ops_seed_from_db`, a saved file of
  whole-well tile positions a later subset run composes with instead of
  solving.

The OPS functions are the same objects as in their modules
(``stitch_core.phase_correlate is ops_register.phase_correlate``), so every
OPS import, test and reference keeps working and nothing is implemented twice.
They are attributes of this module but are documented, and listed in
``__all__``, only where they are defined.
No Qt and no torch are imported here; torch and CuPy are imported only inside
the GPU backends of :func:`phase_correlate`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ops_compose import ComposeError, Window, compose_window, overlap_gain, windows_over
from .ops_cycles import FFT_BACKENDS, FFT_PRECISIONS
from .ops_cycles import _FFTEngine as FFTEngine
from .ops_register import (MIN_PEAK_RATIO, SKEW_PX, STRIP_FRACTION, Registration,
                           _edge_band, _read_peak, _register_strips, available_backends,
                           phase_correlate, register_edge, register_pairs, unwrap)
from .ops_solve import solve_placements
from .ops_stitch import StitchedWell, stitch_well

LOG = logging.getLogger(__name__)

__all__ = [
    "ACCEPTANCES", "DEFAULT_ANCHOR_WEIGHT", "DEFAULT_MIN_NCC", "GAUGES",
    "OPS_TO_ALIGN_KEYS", "PLACEMENT_SCHEMA", "PLACEMENT_SCHEMA_VERSION",
    "SEED_MODES", "StripRegistration", "feather_weights",
    "load_placement_seed", "ops_seed_from_db", "register_strip_pair",
    "seed_to_placements", "solve_positions", "validate_placement_seed",
    "write_placement_seed",
]

#: The acceptance rules :func:`register_strip_pair` offers. ``'peak_ratio'``
#: is the OPS rule (the layout prediction within a tolerance when one is
#: given, else the peak ratio); ``'ncc'`` is Align & Stitch's
#: normalised cross-correlation of the shifted strips; ``'both'`` asks for
#: both, the default the maintainer chose for general users.
ACCEPTANCES = ("peak_ratio", "ncc", "both")

#: The gauges :func:`solve_positions` fixes the solution with.
GAUGES = ("lowest_site", "anchor")

#: The default NCC threshold, Align & Stitch's ``min_confidence``.
DEFAULT_MIN_NCC = 0.30

#: The weight of the "stay at nominal" equations under ``gauge='anchor'``,
#: Align & Stitch's ``DEFAULT_ANCHOR_WEIGHT``.
DEFAULT_ANCHOR_WEIGHT = 1e-3

#: The floor of the linear ramp weights, Align & Stitch's ``_WEIGHT_FLOOR``.
_WEIGHT_FLOOR = 1e-3

#: Which OPS setting each new Align & Stitch key mirrors. A test keeps it
#: complete: every key Align & Stitch gained for the core engine is listed,
#: with the OPS setting or constant that gives the same behaviour.
OPS_TO_ALIGN_KEYS: Dict[str, str] = {
    "ops_registration_precision": "registration_precision",
    "ops_registration_backend": "registration_backend",
    "ops_raster_overlap": "expected_overlap_px",
    "ops_compose_blend": "blend",
    "ops_window_overlap": "compose_window_overlap",
    "ops_placement_seed": "align_placement_seed",
    "MIN_PEAK_RATIO": "min_peak_ratio",
    "STRIP_FRACTION": "strip_fraction",
    "SKEW_PX": "skew_px",
    "_STITCH_TOLERANCE": "tolerance_px",
    "solve_placements": "solve_gauge",
    "peak_ratio acceptance": "acceptance",
    "windows_over": "compose",
    "_WINDOW": "compose_window_px",
}

#: The ``schema`` value of a placement file (item 940).
PLACEMENT_SCHEMA = "spacr.ops_placement"

#: The placement file version this spaCR writes and the newest it reads.
PLACEMENT_SCHEMA_VERSION = 1

#: The acquisitions a placement file may describe.
_SEED_ACQUISITIONS = ("sbs", "phenotype", "align")

#: The keys every placement file carries.
_SEED_REQUIRED = ("schema", "schema_version", "plate", "well", "acquisition",
                  "tile_shape", "tiles", "origin", "canvas")

#: The placement file modes: the run uses exactly the file's tiles, or any
#: non-empty subset of them.
SEED_MODES = ("subset", "strict")


@dataclass(frozen=True)
class StripRegistration:
    """One pair's registration by :func:`register_strip_pair`.

    :ivar dy: rows from the first tile's origin to the second's; a whole
        number unless sub-pixel refinement ran.
    :ivar dx: the same across columns.
    :ivar peak_ratio: the correlation peak against the surface's mean.
    :ivar ncc: the normalised cross-correlation of the two strips where
        they overlap under the measured shift, or None when not computed.
    :ivar accepted: whether the pair passed the acceptance rule.
    :ivar backend: which array module computed the surface.
    :ivar overlap_px: how many strip pixels overlap under the shift.
    :ivar note: why a pair was refused, empty when it was accepted.
    """

    dy: float
    dx: float
    peak_ratio: float
    ncc: Optional[float]
    accepted: bool
    backend: str
    overlap_px: int = 0
    note: str = ""

    @property
    def shift(self) -> Tuple[float, float]:
        """``(dy, dx)``, what :func:`solve_positions` reads."""
        return self.dy, self.dx


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equal-shaped arrays; 0.0 when undefined.

    :param a: the first array.
    :param b: the second, the same shape.
    :returns: r in ``[-1, 1]``, or 0.0 for empty or flat input.
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


def _strip_overlap(shape: Tuple[int, int], dy: int, dx: int):
    """The overlapping rectangles of two same-shaped strips offset by a shift.

    :param shape: the strips' ``(rows, columns)``.
    :param dy: rows from the first strip's origin to the second's.
    :param dx: the same across columns.
    :returns: ``((y0, y1, x0, x1) in the first, (y0, y1, x0, x1) in the
        second)``, or None when they do not overlap.
    """
    height, width = shape
    y0, y1 = max(0, dy), min(height, height + dy)
    x0, x1 = max(0, dx), min(width, width + dx)
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, y1, x0, x1), (y0 - dy, y1 - dy, x0 - dx, x1 - dx)


def _subpixel(strip_first: np.ndarray, strip_second: np.ndarray,
              shift: Tuple[int, int], upsample: int) -> Tuple[float, float]:
    """Refine an integer strip shift to ``1 / upsample`` of a pixel.

    The phase-correlation surface is evaluated on a grid of ``1 / upsample``
    pixels over a 1.5 px region centred on the integer peak by a matrix
    Fourier transform of the normalised cross-power spectrum
    (Guizar-Sicairos et al. 2008), the method and region
    ``skimage.registration.phase_cross_correlation`` uses with
    ``normalization='phase'``, in NumPy float64. The integer peak lies on
    that grid, so the refined shift is never further than 0.75 px from it.

    :param strip_first: the first strip.
    :param strip_second: the second strip, the same shape.
    :param shift: the integer strip shift, already unwrapped.
    :param upsample: the refinement factor, above 1.
    :returns: the refined ``(dy, dx)``.
    """
    a = np.fft.fft2(np.asarray(strip_first, dtype=np.float64))
    b = np.fft.fft2(np.asarray(strip_second, dtype=np.float64))
    product = a * np.conj(b)
    magnitude = np.abs(product)
    magnitude[magnitude == 0] = 1
    product /= magnitude
    rows, cols = product.shape
    step = 1.0 / float(upsample)
    span = int(math.ceil(1.5 * upsample)) // 2
    offsets = np.arange(-span, span + 1) * step
    uy = shift[0] + offsets
    ux = shift[1] + offsets
    ky = np.fft.fftfreq(rows) * rows
    kx = np.fft.fftfreq(cols) * cols
    kernel_y = np.exp(2j * np.pi * np.outer(uy, ky) / rows)
    kernel_x = np.exp(2j * np.pi * np.outer(kx, ux) / cols)
    surface = np.real(kernel_y @ product @ kernel_x)
    best_y, best_x = np.unravel_index(int(np.argmax(surface)), surface.shape)
    return float(uy[best_y]), float(ux[best_x])


def register_strip_pair(first: np.ndarray, second: np.ndarray, axis: str, *,
                        extent: Optional[int] = None,
                        strip_fraction: Optional[float] = STRIP_FRACTION,
                        expected_overlap: Optional[int] = None,
                        tolerance: Optional[int] = None,
                        skew: Optional[int] = SKEW_PX,
                        acceptance: str = "peak_ratio",
                        min_peak_ratio: float = MIN_PEAK_RATIO,
                        min_ncc: float = DEFAULT_MIN_NCC,
                        upsample: int = 1,
                        precision: str = "float64",
                        backend: Optional[str] = None,
                        gpu: bool = False) -> StripRegistration:
    """Register two tiles that touch along one edge, on their overlap strips.

    The OPS edge registration (:func:`register_edge`) with the two checks
    Align & Stitch adds. The integer shift and the ``'peak_ratio'``
    acceptance are exactly :func:`register_edge`'s, so OPS settings
    (``acceptance='peak_ratio'``, ``strip_fraction=0.30``, ``upsample=1``)
    give the same edges as :func:`spacr.ops_stitch.stitch_well`.

    :param first: the tile the edge is measured against, 2-D; or, when
        ``extent`` is given, its strip already cut (the last rows for
        ``'vertical'``, the last columns for ``'horizontal'``).
    :param second: its neighbour below (``'vertical'``) or to the right
        (``'horizontal'``); or its first rows or columns when ``extent`` is
        given.
    :param axis: ``'vertical'`` or ``'horizontal'``.
    :param extent: the tiles' length across the edge when ``first`` and
        ``second`` are strips; None when they are whole tiles.
    :param strip_fraction: the share of the tile a strip takes when whole
        tiles are given; 0.30 is the OPS value, None takes the nominal
        overlap (``expected_overlap``).
    :param expected_overlap: the overlap in pixels the layout predicts.
    :param tolerance: how far along the edge the shift may land from the
        prediction (``'peak_ratio'`` and ``'both'``); None judges by the
        peak ratio instead.
    :param skew: how far across the edge it may land.
    :param acceptance: one of :data:`ACCEPTANCES`.
    :param min_peak_ratio: the peak ratio below which a pair is refused when
        no tolerance is given.
    :param min_ncc: the NCC below which ``'ncc'`` and ``'both'`` refuse.
    :param upsample: 1 keeps the integer shift; above 1 refines it to
        ``1 / upsample`` px (Align & Stitch uses 10).
    :param precision: ``'float64'`` or ``'float32'``, as
        :func:`phase_correlate`.
    :param backend: force a backend of :func:`phase_correlate`.
    :param gpu: allow the GPU backends.
    :returns: the registration, accepted or not, in tile coordinates.
    :raises ValueError: for an unknown axis or acceptance, tiles of two
        shapes, or strips given without ``extent``.
    """
    if axis not in ("vertical", "horizontal"):
        raise ValueError(f"axis is 'vertical' or 'horizontal', not {axis!r}")
    if acceptance not in ACCEPTANCES:
        raise ValueError(f"acceptance must be one of {list(ACCEPTANCES)}; "
                         f"got {acceptance!r}")
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(
            f"an edge is between two tiles of one shape, got {first.shape} "
            f"and {second.shape}")
    down = axis == "vertical"
    if extent is None:
        extent = first.shape[0] if down else first.shape[1]
        if strip_fraction is None:
            if not expected_overlap:
                raise ValueError("strip_fraction=None needs expected_overlap, "
                                 "the nominal overlap the strip is cut to")
            band = max(8, min(extent, int(expected_overlap)))
        else:
            band = _edge_band(extent, strip_fraction)
        if down:
            first, second = first[-band:, :], second[:band, :]
        else:
            first, second = first[:, -band:], second[:, :band]
    band = first.shape[0] if down else first.shape[1]
    measured = _register_strips(
        first, second, axis, int(extent), expected_overlap=expected_overlap,
        tolerance=tolerance, skew=skew, min_peak_ratio=min_peak_ratio,
        precision=precision, backend=backend, gpu=gpu)
    lead = int(extent) - band
    strip_dy = measured.dy - (lead if down else 0)
    strip_dx = measured.dx - (0 if down else lead)
    dy, dx = float(measured.dy), float(measured.dx)
    if int(upsample) > 1:
        fine_y, fine_x = _subpixel(first, second, (strip_dy, strip_dx),
                                   int(upsample))
        dy = fine_y + (lead if down else 0)
        dx = fine_x + (0 if down else lead)
    ncc = None
    overlap_px = 0
    windows = _strip_overlap(first.shape, int(strip_dy), int(strip_dx))
    if windows is not None:
        (ay0, ay1, ax0, ax1), (by0, by1, bx0, bx1) = windows
        overlap_px = int((ay1 - ay0) * (ax1 - ax0))
    if acceptance != "peak_ratio":
        ncc = 0.0
        if windows is not None:
            ncc = _ncc(first[ay0:ay1, ax0:ax1], second[by0:by1, bx0:bx1])
    by_ratio = bool(measured.accepted)
    by_ncc = ncc is not None and ncc >= float(min_ncc)
    accepted = {"peak_ratio": by_ratio, "ncc": by_ncc,
                "both": by_ratio and by_ncc}[acceptance]
    note = ""
    if not accepted:
        reasons = []
        if acceptance in ("peak_ratio", "both") and not by_ratio:
            reasons.append(f"peak ratio {measured.peak_ratio:.2f} or the "
                           f"layout check refused the shift")
        if acceptance in ("ncc", "both") and not by_ncc:
            reasons.append(f"cross-correlation {ncc:.3f} < {min_ncc:g}")
        note = "; ".join(reasons)
    return StripRegistration(dy=dy, dx=dx, peak_ratio=float(measured.peak_ratio),
                             ncc=ncc, accepted=accepted,
                             backend=measured.backend, overlap_px=overlap_px,
                             note=note)


def _blas_limit(threads: Optional[int]):
    """A context that holds BLAS to ``threads`` threads, or does nothing.

    :param threads: the thread count, or None to leave BLAS as it is.
    :returns: a context manager.
    """
    import contextlib

    if threads is None:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=int(threads), user_api="blas")


def solve_positions(edges: Any,
                    sites: Sequence[Any], *,
                    gauge: str = "lowest_site",
                    weights: Optional[Mapping[Tuple[Any, Any], float]] = None,
                    anchors: Optional[Mapping[Any, Tuple[float, float]]] = None,
                    anchor_weight: float = DEFAULT_ANCHOR_WEIGHT,
                    blas_threads: Optional[int] = 1
                    ) -> Dict[Any, Tuple[float, float]]:
    """Absolute tile positions from the pairwise displacements, in one solve.

    Every accepted edge ``(a, b) -> (dy, dx)`` is one equation
    ``p_b - p_a = (dy, dx)`` and all of them are solved at once by least
    squares, so measurement error is spread over the graph rather than
    accumulated along a path. The gauge -- a pure pair graph fixes positions
    only up to a translation -- is fixed one of two ways:

    * ``'lowest_site'`` (OPS): the lowest site of each connected component
      is pinned at the origin, unweighted, exactly
      :func:`spacr.ops_solve.solve_placements`;
    * ``'anchor'`` (Align & Stitch): every site gets a weak equation
      ``p = anchor`` of weight ``anchor_weight`` and each edge its own
      weight, exactly ``spacr.align.solve_positions``; a tile in no edge
      lands on its anchor.

    The solve runs on ``blas_threads`` BLAS threads, one by default, so two
    runs give byte-identical positions whatever the machine's thread count.

    :param edges: ``(a, b) -> (dy, dx)`` or anything carrying ``.shift``,
        b's position minus a's, for the accepted pairs. Under ``'anchor'``
        it may also be a sequence of ``(a, b, dy, dx)`` or
        ``(a, b, dy, dx, weight)`` rows, which may repeat a pair, as
        Align & Stitch's edge lists do.
    :param sites: every site to place.
    :param gauge: one of :data:`GAUGES`.
    :param weights: ``(a, b) -> weight`` under ``'anchor'``; a missing pair
        weighs 1.0, and weights are floored at 1e-6.
    :param anchors: ``site -> (y, x)`` under ``'anchor'``, one per site.
    :param anchor_weight: the weight of the anchor equations.
    :param blas_threads: the BLAS thread count during the solve; None leaves
        it as it is, which only the legacy Align & Stitch engine asks for.
    :returns: ``site -> (y, x)``.
    :raises ValueError: for an unknown gauge, weights or anchors under
        ``'lowest_site'``, or a site without an anchor under ``'anchor'``.
    """
    if gauge not in GAUGES:
        raise ValueError(f"gauge must be one of {list(GAUGES)}; got {gauge!r}")
    if gauge == "lowest_site":
        if weights or anchors:
            raise ValueError("gauge='lowest_site' is unweighted and has no "
                             "anchors; use gauge='anchor' for those")
        return solve_placements(dict(edges), list(sites))
    sites = list(sites)
    count = len(sites)
    if not count:
        return {}
    anchors = dict(anchors or {})
    missing = [site for site in sites if site not in anchors]
    if missing:
        raise ValueError(f"gauge='anchor' needs an anchor for every site; "
                         f"{len(missing)} have none, first {missing[:5]}")
    slot = {site: index for index, site in enumerate(sites)}
    edge_list = []
    if isinstance(edges, Mapping):
        for (left, right), value in edges.items():
            shift = getattr(value, "shift", value)
            weight = 1.0 if weights is None else weights.get((left, right), 1.0)
            edge_list.append((slot[left], slot[right], float(shift[0]),
                              float(shift[1]), float(weight)))
    else:
        for row in edges:
            left, right, dy, dx = row[:4]
            weight = row[4] if len(row) > 4 else (
                1.0 if weights is None else weights.get((left, right), 1.0))
            edge_list.append((slot[left], slot[right], float(dy), float(dx),
                              float(weight)))
    nominal = np.array([anchors[site] for site in sites],
                       dtype=np.float64).reshape(count, 2)
    n_rows = len(edge_list) + count
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
    for k in range(count):
        r = len(edge_list) + k
        rows.append(r)
        cols.append(k)
        vals.append(float(anchor_weight))
        rhs[r] = float(anchor_weight) * nominal[k]
    with _blas_limit(blas_threads):
        if count <= 512:
            design = np.zeros((n_rows, count), dtype=np.float64)
            for r, c, v in zip(rows, cols, vals):
                design[r, c] += v
            solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        else:
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import lsqr
            design = coo_matrix((vals, (rows, cols)),
                                shape=(n_rows, count)).tocsr()
            solution = np.column_stack([
                lsqr(design, rhs[:, k], atol=1e-12, btol=1e-12,
                     iter_lim=max(2000, 20 * count))[0]
                for k in range(2)])
    return {site: (float(solution[slot[site], 0]), float(solution[slot[site], 1]))
            for site in sites}


def _ramp(start: int, stop: int, extent: int, feather: int) -> np.ndarray:
    """Linear feather weight for tile-local indices ``[start, stop)``.

    Rises from :data:`_WEIGHT_FLOOR` at the tile edge to 1.0 ``feather``
    pixels in, and mirrors at the far edge. The floor matters: a pixel
    covered only by one tile's outermost row must still take that tile's
    value, not a zero.

    :param start: the first tile-local index.
    :param stop: one past the last.
    :param extent: the tile's length on this axis.
    :param feather: the ramp width in pixels; 0 or less gives ones.
    :returns: float32 weights, one per index.
    """
    index = np.arange(int(start), int(stop), dtype=np.float32)
    distance = np.minimum(index + 0.5, float(extent) - 0.5 - index)
    if feather <= 0:
        return np.ones_like(distance)
    return np.clip(distance / float(feather), _WEIGHT_FLOOR, 1.0)


def feather_weights(tile_shape: Tuple[int, int], rows: Tuple[int, int],
                    cols: Tuple[int, int], *, mode: str = "distance",
                    width: Optional[int] = None) -> np.ndarray:
    """The blending weights of one tile over a rectangle of that tile.

    Both writers take their weights from here: the OPS window composer
    (:func:`compose_window`, ``blend='feather'``) and the Align & Stitch
    band writer (``spacr.align.write_stack``).

    :param tile_shape: the tile's ``(height, width)``.
    :param rows: ``(first, one past last)`` tile-local rows.
    :param cols: the same for columns.
    :param mode: ``'distance'`` (item 931): each pixel weighs its distance in
        pixels to the tile's nearest edge, plus one, in float64, over the
        whole tile; ``'ramp'``: Align & Stitch's linear ramp, rising from
        1e-3 at the edge to 1 at ``width`` pixels in, in float32.
    :param width: the ramp width for ``'ramp'``.
    :returns: a ``(rows, cols)`` weight array.
    :raises ValueError: for an unknown mode, or ``'ramp'`` without a width.
    """
    height, length = int(tile_shape[0]), int(tile_shape[1])
    if mode == "distance":
        ys = np.arange(int(rows[0]), int(rows[1]))
        xs = np.arange(int(cols[0]), int(cols[1]))
        return np.minimum.outer(np.minimum(ys, height - 1 - ys),
                                np.minimum(xs, length - 1 - xs)
                                ).astype(np.float64) + 1.0
    if mode == "ramp":
        if width is None:
            raise ValueError("mode='ramp' needs the ramp width")
        return (_ramp(rows[0], rows[1], height, int(width))[:, None]
                * _ramp(cols[0], cols[1], length, int(width))[None, :])
    raise ValueError(f"mode must be 'distance' or 'ramp'; got {mode!r}")


def _tr(text: str, **values: Any) -> str:
    """``spacr.qt.i18n.tr`` imported only when a message is raised.

    :param text: the English message.
    :param values: its placeholders.
    :returns: the translated, formatted message.
    """
    from .qt.i18n import tr

    return tr(text, **values)


def _sha256_of(values: Iterable[Any]) -> str:
    """The sha256 of the JSON of ``values``, sorted keys, no spaces.

    :param values: anything JSON can write.
    :returns: the hex digest.
    """
    text = json.dumps(list(values), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _well_key(well: Any) -> Tuple[str, str]:
    """A well's ``(rowID, columnID)``, so ``A1`` and ``a01`` compare equal.

    :param well: a well name.
    :returns: the parsed pair, or the stripped upper-case name twice when it
        does not parse.
    """
    from .schema import parse_well

    try:
        return parse_well(str(well).strip())
    except Exception:                                    # noqa: BLE001
        text = str(well).strip().upper()
        return text, text


def _source_records(tile_files: Mapping[int, Tuple[str, int]],
                    sites: Iterable[int]) -> str:
    """The sha256 over ``(site, file name, byte size)`` of ``sites``.

    :param tile_files: ``site -> (file name, byte size)``.
    :param sites: the sites to include, in any order.
    :returns: the hex digest; the pixels are not read.
    """
    return _sha256_of([int(site), str(tile_files[site][0]), int(tile_files[site][1])]
                      for site in sorted(int(s) for s in sites)
                      if site in tile_files)


def write_placement_seed(path: str, *, plate: str, well: str,
                         placements: Mapping[int, Tuple[float, float]],
                         tile_shape: Tuple[int, int],
                         origin: Tuple[float, float],
                         canvas: Tuple[int, int],
                         acquisition: str = "sbs",
                         cycle: Optional[int] = None,
                         dtype: Optional[str] = None,
                         n_tiles_in_well: Optional[int] = None,
                         raster_overlap: Optional[int] = None,
                         tile_files: Optional[Mapping[int, Tuple[str, int]]] = None,
                         tile_groups: Optional[Sequence[Mapping[str, Any]]] = None,
                         settings_hash: Optional[str] = None,
                         extra: Optional[Mapping[str, Any]] = None,
                         created: Optional[str] = None) -> str:
    """Write one well's tile positions as a placement file (item 940).

    The file is ``<plate>_<well>.ops_placement.json``: the well's
    ``ops_geometry`` rows plus provenance, so a later run on a subset of the
    well's fields composes them at the positions this run solved instead of
    solving again. It is written to a temporary file and renamed, keys
    sorted, floats as ``repr`` so reading it back gives the same numbers.

    :param path: the file to write, or a folder to write
        ``<plate>_<well>.ops_placement.json`` into.
    :param plate: the plate name.
    :param well: the well name.
    :param placements: ``site -> (y, x)`` in whole-well pixels, relative to
        the lowest placed tile, as ``ops_geometry`` stores them.
    :param tile_shape: ``(height, width)`` of one tile.
    :param origin: ``ops_geometry``'s ``(origin_y, origin_x)``.
    :param canvas: ``(height, width)`` of the composed well.
    :param acquisition: ``'sbs'``, ``'phenotype'`` or ``'align'``.
    :param cycle: the reference cycle the positions were solved on.
    :param dtype: the tiles' pixel type, ``'uint16'`` for example.
    :param n_tiles_in_well: the layout's tile count (highest site + 1).
    :param raster_overlap: the raster overlap in pixels the run used.
    :param tile_files: ``site -> (file name, byte size)`` of the planes the
        positions were solved on; stored per tile and hashed, never the
        pixels.
    :param tile_groups: per tile group ``{"id", "matrix", "n_nuclei"}`` for
        a phenotype file (item 930).
    :param settings_hash: the hash of the settings that decide placement.
    :param extra: further top-level keys kept as they are.
    :param created: the creation time to record; None records now. Two
        writes of the same positions with the same ``created`` give the same
        bytes.
    :returns: the path written.
    :raises ValueError: for an unknown acquisition or no placements.
    """
    from datetime import datetime, timezone

    from ._version import __version__

    if acquisition not in _SEED_ACQUISITIONS:
        raise ValueError(f"acquisition must be one of {list(_SEED_ACQUISITIONS)}; "
                         f"got {acquisition!r}")
    if not placements:
        raise ValueError("a placement file needs at least one tile")
    if os.path.isdir(path):
        path = os.path.join(path, f"{plate}_{well}.ops_placement.json")
    tiles = []
    for site in sorted(int(s) for s in placements):
        y, x = placements[site]
        entry: Dict[str, Any] = {"site": site, "y": float(y),
                                 "x": float(x)}
        if tile_files and site in tile_files:
            entry["file"] = str(tile_files[site][0])
            entry["bytes"] = int(tile_files[site][1])
        tiles.append(entry)
    try:
        from .run_journal import _git_hash

        git_hash = _git_hash()
    except Exception:                                    # noqa: BLE001
        git_hash = None
    seed: Dict[str, Any] = dict(extra or {})
    seed.update({
        "schema": PLACEMENT_SCHEMA,
        "schema_version": PLACEMENT_SCHEMA_VERSION,
        "plate": str(plate), "well": str(well),
        "acquisition": acquisition, "cycle": cycle,
        "tile_shape": [int(tile_shape[0]), int(tile_shape[1])],
        "dtype": None if dtype is None else str(dtype),
        "n_tiles_in_well": None if n_tiles_in_well is None else int(n_tiles_in_well),
        "tiles": tiles,
        "origin": [float(origin[0]), float(origin[1])],
        "canvas": [int(canvas[0]), int(canvas[1])],
        "raster_overlap": None if raster_overlap is None else int(raster_overlap),
        "provenance": {
            "spacr_version": __version__, "git_hash": git_hash,
            "settings_hash": settings_hash,
            "tile_ids_sha256": _sha256_of(sorted(int(s) for s in placements)),
            "source_files_sha256": (_source_records(tile_files, placements)
                                    if tile_files else None),
            "created": created or datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
        },
    })
    if tile_groups is not None:
        seed["tile_groups"] = [dict(group) for group in tile_groups]
    text = json.dumps(seed, sort_keys=True, indent=1)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    temporary = f"{path}.part{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    os.replace(temporary, path)
    return path


def _upgrade_seed(seed: Dict[str, Any], version: int) -> Dict[str, Any]:
    """Bring an older placement file up to :data:`PLACEMENT_SCHEMA_VERSION`.

    Version 1 is the first, so nothing is upgraded yet; a later version adds
    one step here per version, oldest first.

    :param seed: the file's content.
    :param version: its ``schema_version``.
    :returns: the content in the current version.
    """
    seed["schema_version"] = PLACEMENT_SCHEMA_VERSION
    return seed


def load_placement_seed(source: Any) -> Dict[str, Any]:
    """Read a placement file, check its schema and bring it to this version.

    :param source: a path to a ``.ops_placement.json`` file, or the
        mapping :func:`ops_seed_from_db` returned (taken as it is).
    :returns: the file's content, with ``"path"`` and ``"sha256"`` of the
        file added (None for a mapping).
    :raises ValueError: for a missing, unreadable, truncated or corrupt
        file, another schema, a missing key, or a version newer than this
        spaCR reads.
    """
    if isinstance(source, Mapping):
        seed = dict(source)
        seed.setdefault("path", None)
        seed.setdefault("sha256", None)
        path, digest = seed["path"], seed["sha256"]
    else:
        path = os.fspath(source)
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
        except OSError as failure:
            raise ValueError(_tr(
                "placement file {path} cannot be read: {reason}; check the "
                "ops_placement_seed setting", path=path, reason=failure)) from failure
        digest = hashlib.sha256(raw).hexdigest()
        try:
            seed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as failure:
            raise ValueError(_tr(
                "placement file {path} is not valid JSON ({reason}); it may be "
                "truncated or corrupt. Write it again with "
                "ops_placement_seed_export on a full run", path=path,
                reason=failure)) from failure
        if not isinstance(seed, dict):
            raise ValueError(_tr(
                "placement file {path} is not a placement file: it holds no "
                "JSON object", path=path))
    if seed.get("schema") != PLACEMENT_SCHEMA:
        raise ValueError(_tr(
            "{path} is not a placement file: its schema is {schema}, not "
            "{expected}", path=path, schema=seed.get("schema"),
            expected=PLACEMENT_SCHEMA))
    try:
        version = int(seed.get("schema_version"))
    except (TypeError, ValueError):
        raise ValueError(_tr(
            "placement file {path} has no readable schema_version",
            path=path)) from None
    if version > PLACEMENT_SCHEMA_VERSION:
        raise ValueError(_tr(
            "placement file version {version} is newer than this spaCR "
            "({supported}); update spaCR", version=version,
            supported=PLACEMENT_SCHEMA_VERSION))
    missing = [key for key in _SEED_REQUIRED if key not in seed]
    if missing:
        raise ValueError(_tr(
            "placement file {path} is missing {keys}; write it again with "
            "ops_placement_seed_export on a full run", path=path,
            keys=", ".join(missing)))
    if version < PLACEMENT_SCHEMA_VERSION:
        seed = _upgrade_seed(seed, version)
    seed["path"], seed["sha256"] = path, digest
    return seed


def seed_to_placements(seed: Mapping[str, Any],
                       sites: Optional[Iterable[int]] = None
                       ) -> Dict[int, Tuple[float, float]]:
    """The whole-well positions a placement file holds.

    :param seed: from :func:`load_placement_seed`.
    :param sites: which sites to return; None returns every tile of the file.
    :returns: ``site -> (y, x)`` in the file's whole-well frame, the same
        floats the file stores.
    """
    found = {int(tile["site"]): (float(tile["y"]), float(tile["x"]))
             for tile in seed["tiles"]}
    if sites is None:
        return found
    return {int(site): found[int(site)] for site in sites if int(site) in found}


def validate_placement_seed(seed: Mapping[str, Any], *, plate: str, well: str,
                            sites: Iterable[int],
                            mode: str = "subset",
                            n_tiles_in_well: Optional[int] = None,
                            tile_shape: Optional[Tuple[int, int]] = None,
                            dtype: Optional[str] = None,
                            acquisition: Optional[str] = None,
                            cycle: Optional[int] = None,
                            tile_files: Optional[Mapping[int, Tuple[str, int]]] = None,
                            align_model: Optional[str] = None) -> List[str]:
    """Check that a placement file belongs to this run, before any pixel.

    The checks, in order: the plate and well (case and zero padding
    ignored); the tile IDs (``'strict'``: the run's sites equal the file's,
    ``'subset'``: every site of the run is in the file); the layout count
    (highest site + 1); the tile shape and pixel type; the acquisition and
    cycle; ``ops_align_model='tile_affine'`` with no tile groups in the
    file; the source files (changed files are a warning under ``'subset'``
    and an error under ``'strict'``); and a file written by another spaCR
    version (a warning). A value given as None is not checked.

    :param seed: from :func:`load_placement_seed`.
    :param plate: the run's plate.
    :param well: the run's well.
    :param sites: the sites this run uses.
    :param mode: ``'subset'`` or ``'strict'``.
    :param n_tiles_in_well: the run's layout count.
    :param tile_shape: the run's tile ``(height, width)``.
    :param dtype: the run's tile pixel type.
    :param acquisition: the run's acquisition kind.
    :param cycle: the run's reference cycle.
    :param tile_files: ``site -> (file name, byte size)`` of this run.
    :param align_model: the run's ``ops_align_model`` for a phenotype file.
    :returns: the warnings, possibly none.
    :raises ValueError: naming the file, the well and the values, for every
        failed check that is not a warning.
    """
    from ._version import __version__

    path = seed.get("path") or "the placement seed"
    if mode not in SEED_MODES:
        raise ValueError(_tr(
            "ops_placement_seed_mode must be one of {modes}; got {value}",
            modes=list(SEED_MODES), value=mode))
    if (str(seed.get("plate")) != str(plate)
            or _well_key(seed.get("well")) != _well_key(well)):
        raise ValueError(_tr(
            "placement file {path} is for plate {seed_plate} well "
            "{seed_well}, not plate {plate} well {well}; choose the file of "
            "this well", path=path, seed_plate=seed.get("plate"),
            seed_well=seed.get("well"), plate=plate, well=well))
    used = sorted({int(site) for site in sites})
    if not used:
        raise ValueError(_tr(
            "well {well} has no tiles to place from {path}", well=well,
            path=path))
    listed = {int(tile["site"]) for tile in seed["tiles"]}
    absent = [site for site in used if site not in listed]
    extra = sorted(listed - set(used))
    if mode == "strict" and (absent or extra):
        raise ValueError(_tr(
            "placement file {path} (well {well}) lists {listed} tiles and this "
            "run has {used}; ops_placement_seed_mode 'strict' needs the same "
            "tiles. Missing from the file: {missing} ({n_missing}); not in "
            "this run: {extra} ({n_extra}). Use 'subset' for a part of the "
            "well", path=path, well=well, listed=len(listed), used=len(used),
            missing=absent[:10], n_missing=len(absent), extra=extra[:10],
            n_extra=len(extra)))
    if absent:
        raise ValueError(_tr(
            "placement file {path} (well {well}) has no position for sites "
            "{missing} ({n_missing} of the {used} this run uses); write the "
            "file from a run that placed them", path=path, well=well,
            missing=absent[:10], n_missing=len(absent), used=len(used)))
    wanted = seed.get("n_tiles_in_well")
    if n_tiles_in_well is not None and wanted is not None and (
            int(wanted) != int(n_tiles_in_well) if mode == "strict"
            else int(n_tiles_in_well) > int(wanted)):
        raise ValueError(_tr(
            "placement file {path} (well {well}) was written for a layout of "
            "{expected} tiles and this run's layout has {found} (highest site "
            "+ 1); include the well's last site or use the file of this "
            "layout", path=path, well=well, expected=wanted,
            found=n_tiles_in_well))
    if tile_shape is not None and [int(v) for v in tile_shape] != \
            [int(v) for v in seed["tile_shape"]]:
        raise ValueError(_tr(
            "placement file {path} (well {well}) is for {expected} px tiles "
            "and this run's tiles are {found}", path=path, well=well,
            expected=list(seed["tile_shape"]), found=list(tile_shape)))
    if dtype is not None and seed.get("dtype") is not None \
            and str(dtype) != str(seed["dtype"]):
        raise ValueError(_tr(
            "placement file {path} (well {well}) is for {expected} tiles and "
            "this run's tiles are {found}", path=path, well=well,
            expected=seed["dtype"], found=dtype))
    if acquisition is not None and seed.get("acquisition") != acquisition:
        raise ValueError(_tr(
            "placement file {path} (well {well}) describes the {expected} "
            "acquisition, not {found}", path=path, well=well,
            expected=seed.get("acquisition"), found=acquisition))
    if cycle is not None and seed.get("cycle") is not None \
            and int(seed["cycle"]) != int(cycle):
        raise ValueError(_tr(
            "placement file {path} (well {well}) was solved on cycle "
            "{expected}, and this run's reference cycle is {found}",
            path=path, well=well, expected=seed["cycle"], found=cycle))
    if align_model == "tile_affine" and seed.get("acquisition") == "phenotype" \
            and not seed.get("tile_groups"):
        raise ValueError(_tr(
            "placement file {path} (well {well}) holds no tile groups, and "
            "ops_align_model 'tile_affine' cannot fit them on a subset; use "
            "a file written with tile groups or ops_align_model "
            "'similarity'", path=path, well=well))
    warnings: List[str] = []
    stored = {int(tile["site"]): (tile.get("file"), tile.get("bytes"))
              for tile in seed["tiles"] if tile.get("file") is not None}
    if tile_files and stored:
        changed = [site for site in used if site in stored and site in tile_files
                   and (str(stored[site][0]), int(stored[site][1]))
                   != (str(tile_files[site][0]), int(tile_files[site][1]))]
        if changed:
            message = _tr(
                "the source files of sites {sites} ({count}) changed since "
                "placement file {path} was written (name or size differ)",
                sites=changed[:10], count=len(changed), path=path)
            if mode == "strict":
                raise ValueError(message)
            warnings.append(message)
    version = (seed.get("provenance") or {}).get("spacr_version")
    if version and str(version) != str(__version__):
        warnings.append(_tr(
            "placement file {path} was written by spaCR {written}; this is "
            "spaCR {current}. The positions are used as written",
            path=path, written=version, current=__version__))
    for message in warnings:
        LOG.warning("%s", message)
    return warnings


def ops_seed_from_db(db: str, plate: str, well: str, *,
                     cycle: Optional[int] = None) -> Dict[str, Any]:
    """A placement seed read straight from a finished ``measurements.db``.

    The same content :func:`write_placement_seed` writes, built from the
    well's ``ops_geometry`` rows, so a user with a finished database needs
    no JSON file. The pixel type, the layout count and the source files are
    not in the database and are left unset, so those checks are skipped.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle: the reference cycle; None takes the rows' cycle.
    :returns: the seed, as :func:`load_placement_seed` returns it.
    :raises ValueError: when the well has no ``ops_geometry`` rows.
    """
    from .ops_engine import _compose_canvas, _well_rows

    frame = _well_rows(db, "ops_geometry", plate, well)
    if frame is None or frame.empty:
        raise ValueError(_tr(
            "well {well} has no ops_geometry rows in {db}; run the stitch "
            "phase first", well=well, db=db))
    placements = {int(row.site): (float(row.y), float(row.x))
                  for row in frame.itertuples()}
    shape = (int(frame["tile_height"].iloc[0]), int(frame["tile_width"].iloc[0]))
    seed = {
        "schema": PLACEMENT_SCHEMA, "schema_version": PLACEMENT_SCHEMA_VERSION,
        "plate": str(plate), "well": str(well), "acquisition": "sbs",
        "cycle": int(frame["cycle"].iloc[0]) if cycle is None else int(cycle),
        "tile_shape": list(shape), "dtype": None, "n_tiles_in_well": None,
        "tiles": [{"site": site, "y": y, "x": x}
                  for site, (y, x) in sorted(placements.items())],
        "origin": [float(frame["origin_y"].iloc[0]),
                   float(frame["origin_x"].iloc[0])],
        "canvas": list(_compose_canvas(placements, shape)),
        "raster_overlap": None,
        "provenance": {"source": f"ops_geometry in {db}",
                       "tile_ids_sha256": _sha256_of(sorted(placements))},
        "path": f"{db}#ops_geometry", "sha256": None,
    }
    return seed
