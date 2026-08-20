#!/usr/bin/env python3
"""Generate the lighter-weight spaCR logo from its preserved master.

The master artwork is never edited. A circular alpha erosion moves every edge
inward by ten pixels at the 3334 px source resolution, reducing each stroke by
roughly twenty pixels while preserving the original geometry and white ink.

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
EROSION_RADIUS = 10


def build_logo() -> Image.Image:
    """Return the canonical logo with a uniformly lighter line weight."""
    image = Image.open(SOURCE).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    axis = np.arange(-EROSION_RADIUS, EROSION_RADIUS + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    footprint = (xx * xx + yy * yy) <= EROSION_RADIUS * EROSION_RADIUS
    thinned = ndimage.grey_erosion(alpha, footprint=footprint)
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
