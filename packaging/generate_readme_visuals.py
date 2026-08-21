#!/usr/bin/env python3
"""Generate linked README/API workflow tiles and rounded resource buttons.

The application catalog and API destinations come from the same registries
used by the home screen and its API-info links. This keeps both documentation
surfaces from drifting as applications are added or moved. Resource buttons
use the supplied artwork without replacing the source files.

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
WORKFLOW_DIR = ICON_DIR / "workflow"
APP_WORKFLOW_DIR = WORKFLOW_DIR / "apps"
FONT_DIR = ROOT / "spacr" / "resources" / "font" / "open_sans" / "static"
DOC_WORKFLOW = ROOT / "docs" / "source" / "_generated" / "workflow_grid.rst"
DOC_WORKFLOW_DIR = ROOT / "docs" / "source" / "_static" / "workflow"
README_PATHS = (
    ROOT / "README.rst",
    *(ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"),
)
WORKFLOW_BEGIN = ".. spacr-workflow-begin"
WORKFLOW_END = ".. spacr-workflow-end"

RESOURCE_SLATE = (43, 47, 58, 255)  # #2B2F3A
WORKFLOW_TILE = (13, 14, 16, 255)  # GUI dark-theme surface, #0D0E10
WHITE = (255, 255, 255, 255)
WORKFLOW_RIM = (255, 255, 255, 96)

BUTTON_SIZE = 512
BUTTON_RADIUS = 32
BUTTON_MARK = round(BUTTON_SIZE * 0.80)
README_LOGO_SIZE = (920, 380)
README_LOGO_MARK = 340
# Percentage widths keep the linked images responsive on both GitHub and the
# Sphinx API. The rows leave a little rounding headroom so the browser never
# moves the last tile onto a new line.
PIPELINE_DISPLAY_PERCENT = 14.5
ARROW_DISPLAY_PERCENT = 2.5
APP_DISPLAY_PERCENT = 19.8
APP_TILE_PADDING = 16
PIPELINE_DISPLAY_WIDTH = f"{PIPELINE_DISPLAY_PERCENT}%"
ARROW_DISPLAY_WIDTH = f"{ARROW_DISPLAY_PERCENT}%"
APP_DISPLAY_WIDTH = f"{APP_DISPLAY_PERCENT}%"

# Match the arrow canvas aspect ratio to its displayed width relative to a
# square pipeline tile. Even renderers that ignore RST's ``:align: middle``
# therefore show the arrow glyph halfway up the neighbouring tiles.
ARROW_CANVAS_WIDTH = 100
ARROW_CANVAS_HEIGHT = round(
    ARROW_CANVAS_WIDTH
    * PIPELINE_DISPLAY_PERCENT
    / ARROW_DISPLAY_PERCENT
)

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


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / "OpenSans-Light.ttf"), size)


def _tile_font(size: int) -> ImageFont.FreeTypeFont:
    """Use the same Open Sans Regular weight as GUI module names."""
    return ImageFont.truetype(str(FONT_DIR / "OpenSans-Regular.ttf"), size)


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
        fill=RESOURCE_SLATE,
    )
    art = _fit(_resource_art(name), BUTTON_MARK)
    button.alpha_composite(
        art,
        ((BUTTON_SIZE - art.width) // 2, (BUTTON_SIZE - art.height) // 2),
    )
    return button


def render_readme_logo() -> Image.Image:
    """Place the canonical logo in a full-width transparent README canvas."""
    canvas = Image.new("RGBA", README_LOGO_SIZE, (0, 0, 0, 0))
    logo = _fit(Image.open(ICON_DIR / "logo_spacr.png"), README_LOGO_MARK)
    canvas.alpha_composite(
        logo,
        ((canvas.width - logo.width) // 2, (canvas.height - logo.height) // 2),
    )
    return canvas


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


def _fitted_tile_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    width: int,
    preferred: int = 54,
    minimum: int = 36,
) -> ImageFont.FreeTypeFont:
    """Return the largest GUI-weight font that keeps a full tile name."""
    for size in range(preferred, minimum - 1, -1):
        font = _tile_font(size)
        bounds = draw.textbbox((0, 0), text, font=font)
        if bounds[2] - bounds[0] <= width:
            return font
    return _tile_font(minimum)


def _render_workflow_tile(key: str, label: str) -> Image.Image:
    """Render the square documentation counterpart of a GUI ``AppTile``."""
    size = 512
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    draw.rounded_rectangle(
        (2, 2, size - 3, size - 3),
        radius=32,
        fill=WORKFLOW_TILE,
        outline=WORKFLOW_RIM,
        width=3,
    )
    icon = _app_icon(key, 220)
    tile.alpha_composite(icon, ((size - icon.width) // 2, 78))
    font = _fitted_tile_font(draw, label, width=size - 40)
    _centered_text(draw, (20, 350, size - 20, 470), label, font)
    return tile


def render_pipeline_tile(key: str, label: str) -> Image.Image:
    """Render one linked workflow step as a standalone square tile."""
    return _render_workflow_tile(key, label)


def render_app_tile(key: str, label: str) -> Image.Image:
    """Render a non-pipeline tile with an even transparent gutter."""
    tile = _render_workflow_tile(key, label)
    inner_size = tile.width - 2 * APP_TILE_PADDING
    tile = tile.resize((inner_size, inner_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    canvas.alpha_composite(tile, (APP_TILE_PADDING, APP_TILE_PADDING))
    return canvas


def render_pipeline_arrow() -> Image.Image:
    """Render U+2192 midway up a tile-height transparent inline asset."""
    arrow = Image.new(
        "RGBA",
        (ARROW_CANVAS_WIDTH, ARROW_CANVAS_HEIGHT),
        (0, 0, 0, 0),
    )
    # Open Sans deliberately has no arrow glyph. DejaVu Sans is present in
    # the Linux documentation/build environments and contains the real
    # U+2192 glyph, avoiding both a hand-drawn approximation and a tofu box.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 128)
    except OSError as exc:
        raise RuntimeError(
            "DejaVu Sans is required to render the U+2192 workflow arrow"
        ) from exc
    probe = ImageDraw.Draw(arrow)
    bounds = probe.textbbox((0, 0), "\u2192", font=font)
    glyph = Image.new(
        "RGBA",
        (bounds[2] - bounds[0], bounds[3] - bounds[1]),
        (0, 0, 0, 0),
    )
    ImageDraw.Draw(glyph).text(
        (-bounds[0], -bounds[1]), "\u2192", font=font, fill=WHITE
    )
    glyph = _fit(glyph, 82)
    arrow.alpha_composite(
        glyph,
        (
            (arrow.width - glyph.width) // 2,
            (arrow.height - glyph.height) // 2,
        ),
    )
    return arrow


def _inline_image_row(names: list[str]) -> str:
    """Join RST substitutions without browser whitespace between images."""
    return r"\ ".join(names)


def _registry() -> list[tuple[str, str, str, str]]:
    from spacr.qt.app import APPS

    return list(APPS)


def _api_urls() -> dict[str, str]:
    """Return the home-screen API destinations used by the running app."""
    from spacr.qt.screens.settings_model import _APP_API_MODULE

    base = "https://einarolafsson.github.io/spacr/api/spacr"
    return {
        key: f"{base}/{_APP_API_MODULE[key].strip('/')}/index.html"
        for key, _label, _desc, _section in _registry()
    }


def _grouped_apps() -> dict[str, list[tuple[str, str]]]:
    apps = _registry()
    pipeline_keys = {key for key, _label in MAIN_PIPELINE}
    return {
        section: [(key, label) for key, label, _desc, actual in apps
                  if actual == section and key not in pipeline_keys]
        for section in SECTION_ORDER
    }


def _readme_workflow(icon_prefix: str) -> str:
    grouped = _grouped_apps()
    urls = _api_urls()
    pipeline_names = {key: f"Workflow_{key}" for key, _label in MAIN_PIPELINE}
    top = []
    for index, (key, _label) in enumerate(MAIN_PIPELINE):
        if index:
            top.append("|Workflow_arrow|")
        top.append(f"|{pipeline_names[key]}|")
    lines = [_inline_image_row(top), ""]
    for key, label in MAIN_PIPELINE:
        lines.extend([
            f".. |{pipeline_names[key]}| image:: {icon_prefix}/workflow/{key}.png",
            f"   :width: {PIPELINE_DISPLAY_WIDTH}",
            f"   :alt: Open the {label} API",
            f"   :target: {urls[key]}",
            "   :align: middle",
        ])
    lines.extend([
        f".. |Workflow_arrow| image:: {icon_prefix}/workflow/arrow.png",
        f"   :width: {ARROW_DISPLAY_WIDTH}",
        "   :align: middle",
        "",
    ])

    definitions: list[str] = []
    for section in SECTION_ORDER:
        items = grouped[section]
        if not items:
            continue
        title = "More core tools" if section == "Core" else section
        lines.extend([f"**{title}**", ""])
        for start in range(0, len(items), 5):
            row = items[start:start + 5]
            lines.extend([
                _inline_image_row([f"|App_{key}|" for key, _ in row]),
                "",
            ])
        for key, label in items:
            definitions.extend([
                f".. |App_{key}| image:: {icon_prefix}/workflow/apps/{key}.png",
                f"   :width: {APP_DISPLAY_WIDTH}",
                f"   :alt: Open the {label} API",
                f"   :target: {urls[key]}",
                "   :align: middle",
            ])
    return "\n".join([*lines, *definitions]).rstrip()


def _documentation_workflow() -> str:
    grouped = _grouped_apps()
    urls = _api_urls()
    pipeline_names = {
        key: f"DocWorkflow_{key}" for key, _label in MAIN_PIPELINE
    }
    top = []
    for index, (key, _label) in enumerate(MAIN_PIPELINE):
        if index:
            top.append("|DocWorkflow_arrow|")
        top.append(f"|{pipeline_names[key]}|")
    lines = [
        "Core workflow",
        "~~~~~~~~~~~~~",
        "",
        _inline_image_row(top),
        "",
    ]
    definitions: list[str] = []
    for key, label in MAIN_PIPELINE:
        definitions.extend([
            f".. |{pipeline_names[key]}| image:: /_static/workflow/{key}.png",
            f"   :width: {PIPELINE_DISPLAY_WIDTH}",
            f"   :alt: Open the {label} API",
            f"   :target: {urls[key]}",
            "   :align: middle",
        ])
    definitions.extend([
        ".. |DocWorkflow_arrow| image:: /_static/workflow/arrow.png",
        f"   :width: {ARROW_DISPLAY_WIDTH}",
        "   :align: middle",
        "",
    ])
    lines.extend(["Other applications", "~~~~~~~~~~~~~~~~~~", ""])
    for section in SECTION_ORDER:
        items = grouped[section]
        if not items:
            continue
        title = "More core tools" if section == "Core" else section
        lines.extend([
            title,
            "^" * len(title),
            "",
        ])
        for start in range(0, len(items), 5):
            row = items[start:start + 5]
            lines.extend([
                _inline_image_row(
                    [f"|DocApp_{key}|" for key, _label in row]
                ),
                "",
            ])
        for key, label in items:
            definitions.extend([
                ".. |DocApp_"
                f"{key}| image:: /_static/workflow/"
                f"apps/{key}.png",
                f"   :width: {APP_DISPLAY_WIDTH}",
                f"   :alt: Open the {label} API",
                f"   :target: {urls[key]}",
                "   :align: middle",
            ])
    return "\n".join([*lines, *definitions]).rstrip() + "\n"


def _replace_workflow_block(path: Path, markup: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.find(WORKFLOW_BEGIN)
    end = text.find(WORKFLOW_END)
    if start < 0 or end <= start:
        raise ValueError(f"{path} is missing its workflow markers")
    end += len(WORKFLOW_END)
    replacement = f"{WORKFLOW_BEGIN}\n\n{markup}\n\n{WORKFLOW_END}"
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def main() -> int:
    missing = [str(path) for path in RESOURCE_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing resource artwork: " + ", ".join(missing))
    for name in RESOURCE_SOURCES:
        target = DATABANK_DIR / f"{name}_button.png"
        render_resource_button(name).save(target, "PNG", optimize=True)
        print(target.relative_to(ROOT))
    target = ICON_DIR / "logo_spacr_readme.png"
    render_readme_logo().save(target, "PNG", optimize=True)
    print(target.relative_to(ROOT))
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    DOC_WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    for key, label in MAIN_PIPELINE:
        image = render_pipeline_tile(key, label)
        target = WORKFLOW_DIR / f"{key}.png"
        image.save(target, "PNG", optimize=True)
        image.save(DOC_WORKFLOW_DIR / f"{key}.png", "PNG", optimize=True)
        print(target.relative_to(ROOT))
    target = WORKFLOW_DIR / "arrow.png"
    arrow = render_pipeline_arrow()
    arrow.save(target, "PNG", optimize=True)
    arrow.save(DOC_WORKFLOW_DIR / "arrow.png", "PNG", optimize=True)
    print(target.relative_to(ROOT))
    APP_WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    doc_app_dir = DOC_WORKFLOW_DIR / "apps"
    doc_app_dir.mkdir(parents=True, exist_ok=True)
    pipeline_keys = {item[0] for item in MAIN_PIPELINE}
    for key, label, _description, _section in _registry():
        if key in pipeline_keys:
            continue
        image = render_app_tile(key, label)
        target = APP_WORKFLOW_DIR / f"{key}.png"
        image.save(target, "PNG", optimize=True)
        image.save(doc_app_dir / f"{key}.png", "PNG", optimize=True)
        print(target.relative_to(ROOT))
    for readme in README_PATHS:
        prefix = (
            "../../../spacr/resources/icons"
            if readme.parent.name == "readme"
            else "spacr/resources/icons"
        )
        _replace_workflow_block(readme, _readme_workflow(prefix))
        print(readme.relative_to(ROOT))
    DOC_WORKFLOW.parent.mkdir(parents=True, exist_ok=True)
    DOC_WORKFLOW.write_text(_documentation_workflow(), encoding="utf-8")
    print(DOC_WORKFLOW.relative_to(ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual artwork regeneration
    raise SystemExit(main())
