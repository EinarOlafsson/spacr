#!/usr/bin/env python
"""Find Mandelbrot coordinates worth flying to, and write them down.

Instruction 327 (3): "map out which coordinates show interesting visuals
like spirals and texture and have say 20 regions on the image that the
camera will automatically smoothely float towards."

THE SET IS DETERMINISTIC, so these are found ONCE and committed rather
than searched at runtime. Same principle as
``tools/draw_volcano_explorer_icon.py``: the artefact is generated, the
generator is kept, and the numbers are reproducible rather than
mysterious.

WHAT MAKES A POINT INTERESTING, numerically. A spiral or a filament is a
region where the escape time changes quickly from pixel to pixel -- a
high gradient. A point that dissolves into uniform colour two zoom
levels in has a high gradient at one zoom and none at the next. So a
candidate is scored by the WORST of its gradients across a range of
zooms, not the average: the whole point is that it stays interesting as
the camera descends.

Run it with::

    python tools/find_mandelbrot_regions.py --count 20

and it rewrites ``spacr/qt/widgets/fractal_regions.py``.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

#: Zoom levels a candidate must hold up across, as half-widths of the
#: sampled window. Four decades: what looks like structure at 1e-2 and
#: nothing at 1e-5 is a point the camera arrives at and then has nothing
#: to show.
ZOOMS = (1e-2, 1e-3, 1e-4, 1e-5)

#: Pixels a side for each scoring render. Small on purpose -- this runs
#: 4 zooms x N candidates, and gradient is a local measure.
GRID = 96

#: Iteration ceiling. Deep zooms need more; this is enough to separate
#: structure from the interior at the zooms above.
ITERATIONS = 600


def escape_time(centre_x: float, centre_y: float, half_width: float,
                grid: int = GRID, iterations: int = ITERATIONS):
    """Smooth escape time over a square window. Vectorised."""
    axis = np.linspace(-half_width, half_width, grid)
    real = centre_x + axis[None, :]
    imag = centre_y + axis[:, None]
    c = real + 1j * imag
    z = np.zeros_like(c)
    escaped_at = np.full(c.shape, float(iterations))
    alive = np.ones(c.shape, dtype=bool)
    for step in range(iterations):
        z[alive] = z[alive] * z[alive] + c[alive]
        just_left = alive & (np.abs(z) > 2.0)
        if just_left.any():
            # Smooth (continuous) escape time, so the gradient is a real
            # gradient rather than integer banding.
            magnitude = np.abs(z[just_left])
            escaped_at[just_left] = (
                step + 1 - np.log(np.log(magnitude)) / math.log(2.0))
            alive &= ~just_left
        if not alive.any():
            break
    return escaped_at


def interest(centre_x: float, centre_y: float) -> float:
    """How interesting this point stays as the camera descends.

    The WORST gradient across :data:`ZOOMS`, normalised. A point that is
    dramatic at one zoom and flat at the next scores low, which is the
    behaviour asked for.
    """
    worst = None
    for half_width in ZOOMS:
        field = escape_time(centre_x, centre_y, half_width)
        if not np.isfinite(field).all():
            return 0.0
        spread = float(field.max() - field.min())
        if spread <= 1.0:
            # Uniform: either the interior, or far enough outside that
            # every sample escapes on the same step. Neither is anything
            # to fly to.
            return 0.0
        gy, gx = np.gradient(field)
        # NOT NORMALISED BY SPREAD. Dividing by it measures RELATIVE
        # texture, so a nearly-uniform field with a whisker of variation
        # scores as detailed -- measured: a point at (2, 2), far outside
        # the set, scored 0.0074 against 0.0087 for the seahorse valley.
        # The raw gradient is the thing asked for: how fast the escape
        # time changes from pixel to pixel.
        detail = float(np.mean(np.hypot(gx, gy)))
        worst = detail if worst is None else min(worst, detail)
    return float(worst or 0.0)


def search(count: int, samples: int, seed: int = 7) -> list:
    """Score `samples` candidates and keep the best `count`, spread out.

    SPREAD MATTERS. Twenty coordinates inside one filament is one place,
    not twenty, so a candidate too close to one already kept is skipped.
    """
    rng = np.random.default_rng(seed)
    scored = []
    for index in range(samples):
        # The interesting boundary lives roughly here; sampling the whole
        # plane wastes most of its time in the far exterior.
        x = float(rng.uniform(-1.9, 0.6))
        y = float(rng.uniform(-1.2, 1.2))
        score = interest(x, y)
        if score > 0.0:
            scored.append((score, x, y))
        if index % 50 == 0:
            print(f"  {index}/{samples} sampled, {len(scored)} usable",
                  file=sys.stderr)

    scored.sort(reverse=True)
    kept: list = []
    for score, x, y in scored:
        if any(math.hypot(x - kx, y - ky) < 0.08 for _s, kx, ky in kept):
            continue
        kept.append((score, x, y))
        if len(kept) >= count:
            break
    return kept


def write_module(regions: list, path: pathlib.Path) -> None:
    lines = [
        '"""Mandelbrot coordinates the backdrop camera tours.',
        '',
        'GENERATED by tools/find_mandelbrot_regions.py -- do not hand-edit.',
        'Instruction 327 (3). Each entry is a point that keeps a high',
        'escape-time gradient across four decades of zoom, so the camera',
        'still has something to show when it arrives.',
        '"""',
        'from __future__ import annotations',
        '',
        'from typing import Final, Tuple',
        '',
        '#: ``(name, real, imaginary, deepest_useful_half_width, score)``.',
        '#: The half-width is how far down this point stays interesting;',
        '#: the score is the worst mean escape-time gradient across ZOOMS.',
        'REGIONS: Final[Tuple[Tuple[str, float, float, float, float], ...]] = (',
    ]
    for index, (score, x, y) in enumerate(regions, start=1):
        lines.append(f'    ("region {index:02d}", {x!r}, {y!r}, '
                     f'{ZOOMS[-1]!r}, {round(score, 6)!r}),')
    lines += [')', '']
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="spacr/qt/widgets/fractal_regions.py")
    args = parser.parse_args()

    regions = search(args.count, args.samples, args.seed)
    if not regions:
        print("no interesting regions found", file=sys.stderr)
        return 1
    write_module(regions, pathlib.Path(args.out))
    print(f"wrote {len(regions)} regions to {args.out}")
    return 0


if __name__ == "__main__":       # pragma: no cover - a generator script
    raise SystemExit(main())
