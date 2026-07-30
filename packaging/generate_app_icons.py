"""Generate the platform application icons from the canonical spaCR mark."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "spacr" / "resources" / "icons"
SOURCE = ICON_DIR / "logo_spacr.png"
PNG_OUTPUT = ICON_DIR / "app_icon.png"
ICO_OUTPUT = ICON_DIR / "app_icon.ico"

CANVAS_SIZE = 1024
BACKGROUND = (0, 55, 55, 255)
CORNER_RADIUS = 188
LOGO_SIZE = 720


def build_icon() -> Image.Image:
    """Return the exact-color rounded application icon."""
    source = Image.open(SOURCE).convert("RGBA")
    alpha_bounds = source.getchannel("A").getbbox()
    if alpha_bounds is None:
        raise ValueError(f"{SOURCE} does not contain a visible logo")
    source = source.crop(alpha_bounds)
    source.thumbnail((LOGO_SIZE, LOGO_SIZE), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    mask = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, CANVAS_SIZE - 1, CANVAS_SIZE - 1),
        radius=CORNER_RADIUS,
        fill=255,
    )
    background = Image.new("RGBA", canvas.size, BACKGROUND)
    canvas.paste(background, mask=mask)

    position = (
        (CANVAS_SIZE - source.width) // 2,
        (CANVAS_SIZE - source.height) // 2,
    )
    canvas.alpha_composite(source, position)
    return canvas


def main() -> None:
    icon = build_icon()
    icon.save(PNG_OUTPUT, optimize=True)
    icon.save(
        ICO_OUTPUT,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"Generated {PNG_OUTPUT.relative_to(ROOT)}")
    print(f"Generated {ICO_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
