#!/usr/bin/env python3
"""Generate the lighter-weight spaCR logo from its preserved master.

The master artwork is never edited. Thick connected strokes receive a larger
inward reduction than the already-fine grid and small motifs. This makes the
change remain visible after GitHub scales the 3334 px artwork to README size
without erasing the smallest details.

Run::

    python packaging/generate_spacr_logo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "spacr" / "resources" / "icons"
SOURCE = ICON_DIR / "source" / "logo_spacr_original.png"
OUTPUTS = (
    ICON_DIR / "logo_spacr.png",
    ROOT / "docs" / "source" / "_static" / "logo_spacr.png",
    ROOT / "docs" / "source" / "_extra" / "tutorials" / "logo_spacr.png",
)
BASE_RADIUS = 10
LARGE_STROKE_RADIUS = 20
ELONGATED_STROKE_RADIUS = 14
HALO_ALPHA = 60


def _disk(radius: int) -> np.ndarray:
    axis = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return (xx * xx + yy * yy) <= radius * radius


def _erode_component(
    alpha: np.ndarray,
    component: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Erode one component in a padded crop and return a full-size layer."""
    output = np.zeros_like(alpha)
    rows, columns = np.where(component)
    if not len(rows):
        return output
    padding = radius + 2
    top = max(0, int(rows.min()) - padding)
    bottom = min(alpha.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(alpha.shape[1], int(columns.max()) + padding + 1)
    crop = np.where(
        component[top:bottom, left:right],
        alpha[top:bottom, left:right],
        0,
    ).astype(np.uint8)
    if radius:
        crop = ndimage.grey_erosion(crop, footprint=_disk(radius))
    output[top:bottom, left:right] = crop
    return output


def build_logo() -> Image.Image:
    """Return the canonical logo with visibly lighter major strokes."""
    image = Image.open(SOURCE).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    base = ndimage.grey_erosion(alpha, footprint=_disk(BASE_RADIUS))
    thinned = np.minimum(base, HALO_ALPHA)

    labels, count = ndimage.label(alpha >= 128)
    for component_id in range(1, count + 1):
        component = labels == component_id
        rows, columns = np.where(component)
        area = int(component.sum())
        height = int(rows.max() - rows.min() + 1)
        width = int(columns.max() - columns.min() + 1)
        extent = area / (height * width)
        aspect = max(height, width) / min(height, width)
        if extent > 0.7:
            radius = 0  # filled dots are shapes, not strokes
        elif area > 100_000:
            radius = LARGE_STROKE_RADIUS
        elif aspect > 3:
            radius = ELONGATED_STROKE_RADIUS
        else:
            radius = BASE_RADIUS
        thinned = np.maximum(
            thinned,
            _erode_component(alpha, component, radius),
        )
    image.putalpha(Image.fromarray(thinned.astype(np.uint8)))
    return image


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"missing preserved logo master: {SOURCE}")
    logo = build_logo()
    for target in OUTPUTS:
        target.parent.mkdir(parents=True, exist_ok=True)
        logo.save(target, "PNG", optimize=True)
        print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual artwork regeneration
    raise SystemExit(main())
