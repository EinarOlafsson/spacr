"""Rescues for a magic-wand flood that runs away.

A flood fill from one click is the fastest way to take a whole object,
and it has one failure mode that matters: the object touches something
bright that is not it -- a debris streak, a saturated membrane seam, the
rim of a well -- and the flood walks out along that seam and swallows the
field. The tolerance that takes the object correctly and the tolerance
that escapes are often the same number, so "lower the tolerance" is not
an answer: it shrinks the object as well.

Three independent rescues are offered here, in the order they run. Each
catches the runaway at a different point, and each can be turned off:

1. **Directional runaway detection and trimming** —
   :func:`trim_directional_runaway` measures flood width along scanlines
   extending from the seed. A sustained abrupt expansion marks a leak, and
   pixels beyond that position are removed. Detection at this stage enables
   the following two refinements.
2. **Intensity-constrained reflooding** — :func:`wand_region` performs a
   binary search for the highest tested tolerance that does not trigger
   runaway detection. The resulting connected region replaces the
   straight-line directional cut.
3. **Gradient-based boundary refinement** —
   :func:`taper_region_to_intensity` applies a watershed to a smoothed
   intensity gradient within a configurable band, moving the provisional
   boundary toward a nearby image edge.

Separately, :func:`cap_region_from_seed` bounds a flood that is simply too
big, keeping the pixels *geodesically* nearest the click so the result is
a bounded piece of the thing that was clicked rather than an arbitrary
prefix of a scan order.

Ported from the standalone curation tool
(``plaque_assay_model/tools/curate_masks_qt.py``), where these were built
against crystal-violet plaque scans. The failure is not specific to that
stain: a nucleus touching a bright fibre and a plaque touching a well rim
are the same flood escaping down the same kind of seam.

Nothing here imports Qt, so the geometry is testable without a GUI.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np

#: Every rescue off, and the pieces that are always on set to the values
#: the standalone tool shipped. :func:`wand_region` fills missing keys
#: from here, so a caller may pass only what it wants to change.
RESCUE_DEFAULTS: Dict[str, object] = {
    "trim_runaway": True,
    "runaway_ratio": 2.0,
    "runaway_warmup": 12,
    "runaway_min_base": 8,
    "runaway_confirm": 2,
    "intensity_border": True,
    "intensity_steps": 8,
    "gradient_taper": True,
    "gradient_sigma": 2.0,
    "gradient_margin": 8,
    "gradient_erode": 3,
    "salvage_over_cap": True,
}


def flood_region(image: np.ndarray, seed_x: int, seed_y: int,
                 tolerance: float) -> np.ndarray:
    """Boolean flood from ``(seed_x, seed_y)``, uncapped.

    A pixel joins when its distance from the seed's value is at most
    ``tolerance`` -- absolute difference on a grey image, Euclidean
    distance across channels on a colour one -- and it is reachable from
    the seed through four-connected steps. Both rules are
    :func:`spacr.qt.mask_engine.magic_wand`'s, so the region this returns
    is the region that wand would fill given no pixel budget.

    Uncapped on purpose: the runaway detector has to see how far the leak
    went to recognise it as one. A flood truncated at the budget looks
    like a large compact object, which is exactly what a leak is not.
    """
    from skimage.segmentation import flood as _sk_flood

    values = np.asarray(image)
    height, width = values.shape[:2]
    if not (0 <= seed_y < height and 0 <= seed_x < width):
        return np.zeros((height, width), dtype=bool)
    values = values.astype(np.float32)
    seed_value = values[seed_y, seed_x]
    # Flooding the *distance from the seed* rather than the image itself
    # collapses grey and multi-channel into one code path: the seed sits at
    # distance 0, so a tolerance band around it is exactly the wand's rule.
    if values.ndim == 2:
        distance = np.abs(values - seed_value)
    else:
        distance = np.linalg.norm(values - seed_value, axis=-1).astype(np.float32)
    # connectivity=1 is the four-neighbourhood the BFS wand steps through.
    return _sk_flood(distance, (int(seed_y), int(seed_x)),
                     connectivity=1, tolerance=float(max(0.0, tolerance)))


def trim_directional_runaway(region: np.ndarray, seed_yx: Tuple[int, int],
                             ratio: float = 2.0, warmup: int = 12,
                             min_baseline: int = 8,
                             confirm: int = 2) -> Tuple[np.ndarray, Dict[str, int]]:
    """Cut a flood where it suddenly widens, and say where it was cut.

    Walking up, down, left and right from the clicked scanline, the flood's
    width is a profile. An object's own profile changes gradually; a leak
    into a seam appears as a step. A leak is called when ``confirm``
    consecutive scanlines are all at least ``ratio`` times wider than the
    widest scanline established strictly *before* them, and everything from
    that scanline outward is removed.

    The three guards exist because a naive step detector fires on the
    object itself:

    * ``warmup`` ignores the scanlines nearest the click, where a
      one-pixel-wide start doubling to two pixels is a ratio of 2.0 and
      means nothing;
    * ``min_baseline`` refuses to judge until the object has reached a real
      width, for the same reason;
    * ``confirm`` requires the expansion to persist, so a single noisy row
      cannot cut the object in half.

    :returns: ``(trimmed, cuts)``. ``cuts`` maps each direction that leaked
        to the image coordinate the cut was made at, and is empty when
        nothing leaked -- which is how a caller knows the flood was clean.
    """
    region = np.asarray(region, dtype=bool)
    y, x = int(seed_yx[0]), int(seed_yx[1])
    height, width = region.shape
    if not (0 <= y < height and 0 <= x < width):
        return region.copy(), {}

    profiles = {
        "up": region[:y + 1].sum(axis=1)[::-1],
        "down": region[y:].sum(axis=1),
        "left": region[:, :x + 1].sum(axis=0)[::-1],
        "right": region[:, x:].sum(axis=0),
    }
    warmup = max(1, int(warmup))
    confirm = max(1, int(confirm))
    ratio = max(1.01, float(ratio))
    min_baseline = max(1, int(min_baseline))

    offsets: Dict[str, int] = {}
    for direction, profile in profiles.items():
        profile = np.asarray(profile, dtype=float)
        for i in range(warmup, len(profile) - confirm + 1):
            # The baseline is the established width BEFORE the candidate.
            # Including the candidate would let a leak raise the bar it has
            # to clear and hide itself.
            baseline = float(profile[:i].max(initial=0))
            if baseline < min_baseline:
                continue
            if np.all(profile[i:i + confirm] >= ratio * baseline):
                offsets[direction] = i
                break

    out = region.copy()
    cuts: Dict[str, int] = {}
    if "up" in offsets:
        cut = y - offsets["up"]
        out[:cut + 1, :] = False
        cuts["up"] = cut
    if "down" in offsets:
        cut = y + offsets["down"]
        out[cut:, :] = False
        cuts["down"] = cut
    if "left" in offsets:
        cut = x - offsets["left"]
        out[:, :cut + 1] = False
        cuts["left"] = cut
    if "right" in offsets:
        cut = x + offsets["right"]
        out[:, cut:] = False
        cuts["right"] = cut
    return out, cuts


def cap_region_from_seed(region: np.ndarray, seed_yx: Tuple[int, int],
                         max_pixels: int) -> np.ndarray:
    """Keep at most ``max_pixels`` of ``region``, nearest the click.

    Nearest is measured *through the region* -- breadth-first growth that
    may only step on flooded pixels -- so the kept piece cannot jump a gap
    to a bright patch that merely happens to be close in a straight line.
    Eight-connected, because the piece being salvaged is a shape to keep
    whole, not a flood to grow.

    Returns ``region`` unchanged when it already fits, and an empty mask
    when the seed is not inside it.
    """
    region = np.asarray(region, dtype=bool)
    y, x = int(seed_yx[0]), int(seed_yx[1])
    height, width = region.shape
    limit = max(1, int(max_pixels))
    if int(region.sum()) <= limit:
        return region.copy()
    if not (0 <= y < height and 0 <= x < width and region[y, x]):
        return np.zeros_like(region)

    kept = np.zeros_like(region)
    queued = np.zeros_like(region)
    queued[y, x] = True
    queue = deque([(y, x)])
    taken = 0
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1))
    while queue and taken < limit:
        cy, cx = queue.popleft()
        kept[cy, cx] = True
        taken += 1
        for dy, dx in neighbours:
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < height and 0 <= nx < width and region[ny, nx]
                    and not queued[ny, nx]):
                queued[ny, nx] = True
                queue.append((ny, nx))
    return kept


def taper_region_to_intensity(image: np.ndarray, flooded_region: np.ndarray,
                              provisional: np.ndarray,
                              seed_yx: Tuple[int, int], sigma: float = 2.0,
                              margin: int = 8,
                              foreground_erode: int = 3) -> np.ndarray:
    """Move a geometric edge onto the nearest real intensity edge.

    A directional cut is a straight line and a geodesic cap is a circular
    arc; objects are neither. ``provisional`` is whichever of those the
    earlier rescues produced. Its interior, inset by ``foreground_erode``,
    is marked as certainly object; the part of ``flooded_region`` that was
    thrown away, at least ``margin`` deep, is marked as certainly not. A
    watershed on the ``sigma``-smoothed intensity gradient decides the band
    between them, so the final boundary follows an intensity change -- up
    or down -- instead of the cut.

    The result never leaves the original flood, and always keeps the
    connected piece the click is in. If the discarded part is thinner than
    ``margin`` there is no room for a band, so its deepest quarter is used
    rather than giving up and leaving the straight edge.
    """
    from scipy.ndimage import (binary_erosion, distance_transform_edt,
                               gaussian_filter, label)
    from skimage.filters import sobel
    from skimage.segmentation import watershed

    grey = np.asarray(image, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    flooded = np.asarray(flooded_region, dtype=bool)
    provisional = np.asarray(provisional, dtype=bool) & flooded
    y, x = int(seed_yx[0]), int(seed_yx[1])
    height, width = flooded.shape
    if not (0 <= y < height and 0 <= x < width and provisional[y, x]):
        return provisional.copy()

    inset = max(0, int(foreground_erode))
    foreground = (binary_erosion(provisional, iterations=inset)
                  if inset else provisional.copy())
    # Erosion can erase a small object entirely. The click is trusted, so a
    # few pixels around it are always foreground -- enough to seed the
    # watershed without dictating the shape of its answer.
    yy, xx = np.ogrid[:height, :width]
    foreground |= (((xx - x) ** 2 + (yy - y) ** 2 <= 4) & provisional)

    removed = flooded & ~provisional
    if not removed.any():
        return provisional.copy()
    depth = distance_transform_edt(removed)
    background = removed & (depth >= max(1, int(margin)))
    if not background.any():
        cutoff = max(1.0, float(depth.max()) * 0.75)
        background = removed & (depth >= cutoff)
    if not background.any() or not foreground.any():
        return provisional.copy()

    markers = np.zeros(flooded.shape, dtype=np.uint8)
    markers[foreground] = 1
    markers[background] = 2
    gradient = sobel(gaussian_filter(grey, sigma=max(0.0, float(sigma))))
    labels = watershed(gradient, markers=markers, mask=flooded)
    result = labels == 1
    components, _ = label(result, structure=np.ones((3, 3), dtype=np.uint8))
    wanted = int(components[y, x])
    return components == wanted if wanted else provisional.copy()


def wand_region(image: np.ndarray, seed_x: int, seed_y: int,
                tolerance: float, max_pixels: int = 100_000,
                **settings) -> Tuple[np.ndarray, Dict[str, object]]:
    """Flood from one click and apply whichever rescues are switched on.

    Runs the three rescues in order -- detect the runaway, replace the
    straight cut with an intensity border, taper what is left onto the
    local gradient -- and then applies the pixel budget. Every step is
    optional and each is inert on a flood that did not run away: with no
    leak detected, this returns exactly :func:`flood_region`'s answer.

    ``settings`` accepts the keys of :data:`RESCUE_DEFAULTS`; anything
    missing takes the default.

    :returns: ``(region, report)``. ``report`` names what happened --
        ``cuts`` (the directions that leaked), ``intensity_border`` and the
        ``refined_tolerance`` it settled on, ``tapered``, ``capped``, and
        the pixel counts before and after -- so the caller can tell the
        user why the wand took what it took, and write it in the ledger.
    """
    s = dict(RESCUE_DEFAULTS)
    s.update({k: v for k, v in settings.items() if k in RESCUE_DEFAULTS})

    region = flood_region(image, seed_x, seed_y, tolerance)
    initial = region.copy()
    raw_n = int(region.sum())
    report: Dict[str, object] = {
        "flooded_px": raw_n, "kept_px": raw_n, "cuts": [],
        "intensity_border": False, "refined_tolerance": float(tolerance),
        "tapered": False, "capped": False, "rejected": False,
    }
    if not raw_n:
        return region, report

    seed = (int(seed_y), int(seed_x))
    grey = np.asarray(image, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    cuts: Dict[str, int] = {}

    if s["trim_runaway"]:
        detector = dict(ratio=s["runaway_ratio"], warmup=s["runaway_warmup"],
                        min_baseline=s["runaway_min_base"],
                        confirm=s["runaway_confirm"])
        straight, cuts = trim_directional_runaway(region, seed, **detector)
        if cuts and s["intensity_border"]:
            # The detector proved this tolerance escapes. Rather than keep
            # the half-plane cut, bisect for the highest tolerance whose
            # whole flood stays put: that boundary is drawn by the image.
            lo, hi = 0.0, float(tolerance)
            best, best_tol = None, 0.0
            for _ in range(max(1, int(s["intensity_steps"]))):
                mid = (lo + hi) / 2.0
                candidate = flood_region(image, seed_x, seed_y, mid)
                _, candidate_cuts = trim_directional_runaway(
                    candidate, seed, **detector)
                if candidate_cuts:
                    hi = mid
                else:
                    best, best_tol = candidate, mid
                    lo = mid
            if best is not None and int(best.sum()) > 0:
                region = best
                report["intensity_border"] = True
                report["refined_tolerance"] = float(best_tol)
            else:
                region = straight
        elif cuts:
            region = straight
        if cuts and s["gradient_taper"]:
            tapered = taper_region_to_intensity(
                grey, initial, region, seed, sigma=s["gradient_sigma"],
                margin=s["gradient_margin"],
                foreground_erode=s["gradient_erode"])
            if tapered.any():
                region = tapered
                report["tapered"] = True
    report["cuts"] = sorted(cuts)

    limit = max(1, int(max_pixels))
    n = int(region.sum())
    if n > limit:
        if not s["salvage_over_cap"]:
            # Refusing is a real answer: an object truncated at a budget is
            # not the object, and a user who is tuning tolerance needs to
            # see that the budget was the thing that stopped the flood.
            report.update(rejected=True, capped=True, kept_px=0)
            return np.zeros_like(region), report
        over_cap = region
        bounded = cap_region_from_seed(over_cap, seed, limit)
        region = bounded
        if s["gradient_taper"]:
            # A geodesic cap ends in an arc. Give it an intensity edge too,
            # narrowing the band until the tapered result still fits the
            # budget it was capped to.
            wide = max(1, int(s["gradient_margin"]))
            for band in sorted({wide, max(1, wide // 2), 1}, reverse=True):
                tapered = taper_region_to_intensity(
                    grey, over_cap, bounded, seed, sigma=s["gradient_sigma"],
                    margin=band, foreground_erode=s["gradient_erode"])
                if 0 < int(tapered.sum()) <= limit:
                    region = tapered
                    report["tapered"] = True
                    break
        report["capped"] = True
    report["kept_px"] = int(region.sum())
    return region, report


def magic_wand(image: np.ndarray, mask: np.ndarray, seed_x: int, seed_y: int,
               tolerance: float, max_pixels: int = 100_000,
               action: str = "add",
               **settings) -> Tuple[np.ndarray, Dict[str, object]]:
    """:func:`wand_region`, written into a copy of ``mask``.

    Writes 255 where the region landed for ``action="add"`` and 0 for
    ``action="erase"``, matching
    :func:`spacr.qt.mask_engine.magic_wand`, and returns the report beside
    the new mask. A rejected flood leaves the mask untouched.
    """
    if mask is None or image is None:
        # Same report shape as a real click, so a caller writing it into a
        # ledger does not have to special-case the nothing-to-do path.
        return mask, {"flooded_px": 0, "kept_px": 0, "cuts": [],
                      "intensity_border": False, "refined_tolerance": 0.0,
                      "tapered": False, "capped": False, "rejected": True}
    region, report = wand_region(image, seed_x, seed_y, tolerance,
                                 max_pixels, **settings)
    out = mask.copy()
    if region.any():
        out[region] = 255 if action == "add" else 0
    return out, report
