#!/usr/bin/env python3
"""Generate the README workflow catalog and rounded resource buttons.

The application catalog is read from :data:`spacr.qt.app.APPS`, the same
registry used by the home screen. This keeps the README picture from drifting
as applications are added or moved between sections. The resource buttons use
the supplied artwork in ``spacr/resources/icons/databanks`` without replacing
the source files.

Run::

    python packaging/generate_readme_visuals.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ICON_DIR = ROOT / "spacr" / "resources" / "icons"
DATABANK_DIR = ICON_DIR / "databanks"
FONT_DIR = ROOT / "spacr" / "resources" / "font" / "open_sans" / "static"

SLATE = (43, 47, 58, 255)  # #2B2F3A
PAGE = (24, 27, 34, 255)
WHITE = (255, 255, 255, 255)
MUTED = (196, 203, 216, 255)
ACCENT = (74, 158, 255, 255)

BUTTON_SIZE = 512
BUTTON_RADIUS = 32
BUTTON_MARK = round(BUTTON_SIZE * 0.80)

RESOURCE_SOURCES = {
    "biostudies": DATABANK_DIR / "bioimages.jpg",
    "biorxiv": DATABANK_DIR / "biorxiv.jpeg",
    "huggingface": DATABANK_DIR / "huggingface.png",
    "ncbi": DATABANK_DIR / "ncbi.png",
    "spacrpower": ICON_DIR / "logo_spacr.png",
}

APP_ICON_OVERRIDES = {
    "analyze_plaques": "plaque.png",
    "train_cellpose": "cellpose_masks.png",
    "agreement": "annotate.png",
    "plate_view": "map_barcodes.png",
    "model_compare": "mask.png",
    "model_zoo": "download.png",
    "classify_merged": "classify.png",
    "volcano_explorer": "activation.png",
    "parameter_sweep": "regression.png",
    "explain_cv": "classifier_evaluation.png",
    "investigate_hit": "hit_list.png",
}

MAIN_PIPELINE = (
    ("mask", "Mask"),
    ("measure", "Measure"),
    ("annotate", "Annotate"),
    ("classify_merged", "Classify"),
    ("map_barcodes", "Map Barcodes"),
    ("regression", "Regression"),
)
SECTION_ORDER = (
    "Core",
    "Data",
    "Segmentation models",
    "Results & QC",
    "Explore",
    "Toxoplasma",
    "Design",
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "OpenSans-Bold.ttf" if bold else "OpenSans-Regular.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def _trim(image: Image.Image) -> Image.Image:
    bounds = image.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("artwork contains no visible pixels")
    return image.crop(bounds)


def _fit(image: Image.Image, longest: int) -> Image.Image:
    image = _trim(image.convert("RGBA"))
    scale = longest / max(image.size)
    return image.resize(
        (max(1, round(image.width * scale)),
         max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _remove_flat_background(image: Image.Image, *, light: bool) -> Image.Image:
    """Remove a JPEG's white or black field while retaining antialiasing."""
    rgba = image.convert("RGBA")
    pixels = []
    for red, green, blue, _alpha in rgba.getdata():
        if light:
            alpha = max(0, min(255, max(255 - red, 255 - green, 255 - blue) * 3))
        else:
            alpha = max(0, min(255, max(red, green, blue) * 3))
        pixels.append((red, green, blue, alpha))
    rgba.putdata(pixels)
    return rgba


def _resource_art(name: str) -> Image.Image:
    image = Image.open(RESOURCE_SOURCES[name]).convert("RGBA")
    if name == "biostudies":
        image = _remove_flat_background(image, light=False)
    elif name == "biorxiv":
        image = _remove_flat_background(image, light=True)
        recolored = []
        for red, green, blue, alpha in image.getdata():
            if max(red, green, blue) - min(red, green, blue) < 45 and red < 190:
                red = green = blue = 255
            recolored.append((red, green, blue, alpha))
        image.putdata(recolored)
    elif name == "ncbi":
        # The supplied PNG has a baked white/grey transparency checkerboard.
        # NCBI's mark is blue, so colour saturation isolates it without
        # tracing or redrawing the supplied artwork.
        isolated = []
        for red, green, blue, _alpha in image.getdata():
            chroma = max(red, green, blue) - min(red, green, blue)
            alpha = max(0, min(255, (chroma - 8) * 3))
            isolated.append((red, green, blue, alpha))
        image.putdata(isolated)
    elif image.getchannel("A").getextrema() == (255, 255):
        image = _remove_flat_background(image, light=True)
    return image


def render_resource_button(name: str) -> Image.Image:
    button = Image.new("RGBA", (BUTTON_SIZE, BUTTON_SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(button).rounded_rectangle(
        (0, 0, BUTTON_SIZE - 1, BUTTON_SIZE - 1),
        radius=BUTTON_RADIUS,
        fill=SLATE,
    )
    art = _fit(_resource_art(name), BUTTON_MARK)
    button.alpha_composite(
        art,
        ((BUTTON_SIZE - art.width) // 2, (BUTTON_SIZE - art.height) // 2),
    )
    return button


def _app_icon(key: str, size: int) -> Image.Image:
    filename = APP_ICON_OVERRIDES.get(key, f"{key}.png")
    path = ICON_DIR / filename
    if not path.is_file():
        path = ICON_DIR / "run.png"
    source = _fit(Image.open(path).convert("RGBA"), size)
    # Home-screen icons are monochrome masks that Qt re-inks for the theme.
    alpha = source.getchannel("A")
    white = Image.new("RGBA", source.size, WHITE)
    white.putalpha(alpha)
    return white


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill=WHITE,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2 - bounds[0],
         (top + bottom - height) / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
    )


def _draw_tile(
    image: Image.Image,
    xy: tuple[int, int],
    key: str,
    label: str,
    *,
    width: int,
    height: int,
    icon_size: int,
) -> None:
    x, y = xy
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (x, y, x + width - 1, y + height - 1),
        radius=18,
        fill=SLATE,
    )
    icon = _app_icon(key, icon_size)
    image.alpha_composite(icon, (x + (width - icon.width) // 2, y + 18))
    _centered_text(
        draw,
        (x + 8, y + height - 49, x + width - 8, y + height - 10),
        label,
        _font(22, bold=True),
    )


def _registry() -> list[tuple[str, str, str, str]]:
    from spacr.qt.app import APPS

    return list(APPS)


def render_app_catalog() -> Image.Image:
    apps = _registry()
    pipeline_keys = {key for key, _label in MAIN_PIPELINE}
    grouped = {
        section: [(key, label) for key, label, _desc, actual in apps
                  if actual == section and key not in pipeline_keys]
        for section in SECTION_ORDER
    }

    canvas_width = 1240
    margin = 50
    tile_width = 266
    tile_height = 164
    column_gap = 24
    row_gap = 20
    section_gap = 44
    heading_height = 52
    flow_tile_width = 164
    flow_tile_height = 170
    flow_gap = 34

    content_height = 60 + heading_height + flow_tile_height + section_gap
    for section in SECTION_ORDER:
        items = grouped[section]
        if not items:
            continue
        rows = (len(items) + 3) // 4
        content_height += heading_height + rows * tile_height
        content_height += max(0, rows - 1) * row_gap + section_gap
    canvas = Image.new("RGBA", (canvas_width, content_height + 20), PAGE)
    draw = ImageDraw.Draw(canvas)

    _centered_text(
        draw,
        (margin, 20, canvas_width - margin, 76),
        "The spaCR workflow",
        _font(34, bold=True),
    )
    y = 90
    total_flow_width = len(MAIN_PIPELINE) * flow_tile_width + (
        len(MAIN_PIPELINE) - 1) * flow_gap
    x = (canvas_width - total_flow_width) // 2
    for index, (key, label) in enumerate(MAIN_PIPELINE):
        _draw_tile(
            canvas, (x, y), key, label,
            width=flow_tile_width, height=flow_tile_height, icon_size=94,
        )
        if index < len(MAIN_PIPELINE) - 1:
            arrow_x = x + flow_tile_width + 5
            arrow_y = y + flow_tile_height // 2
            draw.line(
                (arrow_x, arrow_y, arrow_x + flow_gap - 12, arrow_y),
                fill=ACCENT,
                width=5,
            )
            draw.polygon(
                ((arrow_x + flow_gap - 12, arrow_y - 9),
                 (arrow_x + flow_gap - 2, arrow_y),
                 (arrow_x + flow_gap - 12, arrow_y + 9)),
                fill=ACCENT,
            )
        x += flow_tile_width + flow_gap

    y += flow_tile_height + section_gap
    for section in SECTION_ORDER:
        items = grouped[section]
        if not items:
            continue
        title = "More core tools" if section == "Core" else section
        draw.text((margin + 8, y), title, font=_font(29, bold=True), fill=WHITE)
        y += heading_height
        for index, (key, label) in enumerate(items):
            row, column = divmod(index, 4)
            x = margin + column * (tile_width + column_gap)
            tile_y = y + row * (tile_height + row_gap)
            _draw_tile(
                canvas, (x, tile_y), key, label,
                width=tile_width, height=tile_height, icon_size=92,
            )
        rows = (len(items) + 3) // 4
        y += rows * tile_height + max(0, rows - 1) * row_gap + section_gap

    return canvas.crop((0, 0, canvas_width, y - section_gap + 30))


def main() -> int:
    missing = [str(path) for path in RESOURCE_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing resource artwork: " + ", ".join(missing))
    for name in RESOURCE_SOURCES:
        target = DATABANK_DIR / f"{name}_button.png"
        render_resource_button(name).save(target, "PNG", optimize=True)
        print(target.relative_to(ROOT))
    target = ICON_DIR / "workflow_home_apps.png"
    render_app_catalog().save(target, "PNG", optimize=True)
    print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual artwork regeneration
    raise SystemExit(main())
