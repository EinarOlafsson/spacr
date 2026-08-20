#!/usr/bin/env python3
"""Build the README installer buttons from the supplied platform artwork.

The source marks live in ``spacr/resources/icons/platforms/source`` and are
kept separate from the generated buttons. Regenerating the buttons therefore
cannot overwrite the originals.

All four buttons use the same treatment:

* a square 512 px ``#2B2F3A`` tile;
* a restrained 32 px corner radius (6.25% of the side);
* a white, centered mark whose longest dimension is 80% of the tile;
* no text inside or below any button.

The supplied files do not share an alpha convention. Linux is black artwork
on transparency, Windows is white artwork over a baked dark checkerboard, and
the Apple file is a JPEG with a baked light checkerboard. The extraction
functions below deliberately handle those three measured cases and always
paint the resulting mask white.

Run::

    python packaging/generate_platform_icons.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "spacr" / "resources" / "icons"
OUT_DIR = ICON_DIR / "platforms"
SOURCE_DIR = OUT_DIR / "source"

CANVAS = 512
OUTPUT_HEIGHT = 600
SLATE = (43, 47, 58, 255)  # #2B2F3A
WHITE = (255, 255, 255, 255)
CORNER_RADIUS = 32
MARK_FRACTION = 0.80
MARK_SIZE = round(CANVAS * MARK_FRACTION)

SOURCE_FILES = {
    "linux": SOURCE_DIR / "linux.png",
    "macos": SOURCE_DIR / "macos.jpg",
    "windows": SOURCE_DIR / "windows.png",
}
LEGACY_LOGO = ICON_DIR / "logo_spacr.png"


def _alpha_mask(image: Image.Image) -> Image.Image:
    """Return the existing alpha channel, including its antialiased edge."""
    return image.convert("RGBA").getchannel("A")


def _bright_mask(image: Image.Image, threshold: int = 245) -> Image.Image:
    """Extract white artwork from an opaque dark background."""
    grey = np.asarray(image.convert("L"))
    return Image.fromarray(np.where(grey >= threshold, 255, 0).astype("uint8"))


def _apple_mask(image: Image.Image) -> Image.Image:
    """Recover the white Apple silhouette from its baked 20 px checkerboard.

    The source JPEG alternates 255 and 237 every 20 pixels. On the darker
    squares the white mark is unambiguous. Closing those measured 20 px gaps
    reconstructs the adjoining light-square portions without tracing or
    redrawing the supplied silhouette.
    """
    grey = np.asarray(image.convert("L"))
    height, width = grey.shape
    yy, xx = np.indices((height, width))
    dark_checker_square = ((xx // 20 + yy // 20) % 2) == 1
    seeds = dark_checker_square & (grey >= 246)
    mask = Image.fromarray(np.where(seeds, 255, 0).astype("uint8"))
    # A 25 px close bridges one 20 px checker cell. A light blur restores a
    # clean antialiased edge after the binary background removal.
    mask = mask.filter(ImageFilter.MaxFilter(25))
    mask = mask.filter(ImageFilter.MinFilter(25))
    mask = mask.filter(ImageFilter.GaussianBlur(8.0))
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
    return mask.filter(ImageFilter.GaussianBlur(1.5))


def source_mask(name: str) -> Image.Image:
    """Return the supplied mark as a white-ready luminance mask."""
    path = SOURCE_FILES[name]
    image = Image.open(path)
    if name == "linux":
        return _alpha_mask(image)
    if name == "macos":
        return _apple_mask(image)
    return _bright_mask(image)


def _trim_and_fit(mask: Image.Image, size: int) -> Image.Image:
    """Trim transparent margins and fit the longest side to ``size``."""
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("source artwork contains no visible mark")
    cropped = mask.crop(bounds)
    scale = size / max(cropped.size)
    fitted = cropped.resize(
        (
            max(1, round(cropped.width * scale)),
            max(1, round(cropped.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    return fitted


def _tile() -> Image.Image:
    image = Image.new("RGBA", (CANVAS, OUTPUT_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, CANVAS - 1, CANVAS - 1),
        radius=CORNER_RADIUS,
        fill=SLATE,
    )
    return image


def _paste_white(
    image: Image.Image,
    mask: Image.Image,
    *,
    y: int | None = None,
) -> None:
    x = (CANVAS - mask.width) // 2
    top = (CANVAS - mask.height) // 2 if y is None else y
    image.paste(WHITE, (x, top), mask)


def render_platform(name: str) -> Image.Image:
    """Render one centered, text-free platform mark at 80% scale."""
    image = _tile()
    _paste_white(image, _trim_and_fit(source_mask(name), MARK_SIZE))
    return image


def render_legacy() -> Image.Image:
    """Render the spaCR mark at 80%, without an embedded caption."""
    image = _tile()
    logo = _trim_and_fit(_alpha_mask(Image.open(LEGACY_LOGO)), MARK_SIZE)
    _paste_white(image, logo)
    return image


def main() -> int:
    required = (*SOURCE_FILES.values(), LEGACY_LOGO)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing icon source files: " + ", ".join(missing)
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered = {
        name: render_platform(name)
        for name in ("linux", "macos", "windows")
    }
    rendered["legacy"] = render_legacy()
    for name, image in rendered.items():
        target = OUT_DIR / f"{name}.png"
        image.save(target, "PNG", optimize=True)
        print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual artwork regeneration
    raise SystemExit(main())
