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
import re
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools.readme_i18n import localize_workflow_markup  # noqa: E402

ICON_DIR = ROOT / "spacr" / "resources" / "icons"
DATABANK_DIR = ICON_DIR / "databanks"
WORKFLOW_DIR = ICON_DIR / "workflow"
APP_WORKFLOW_DIR = WORKFLOW_DIR / "apps"
FONT_DIR = ROOT / "spacr" / "resources" / "font" / "open_sans" / "static"
DOC_WORKFLOW = ROOT / "docs" / "source" / "_generated" / "workflow_grid.rst"
DOC_FOLDS = ROOT / "docs" / "source" / "_generated" / "folded_modules.rst"
HARDWARE_TABLE = ROOT / "docs" / "source" / "_generated" / "hardware_table.rst"
MODEL_ZOO_TABLE = (ROOT / "docs" / "source" / "_generated"
                   / "model_zoo_table.rst")
MODEL_ZOO_SECTIONS = (ROOT / "docs" / "source" / "_generated"
                      / "model_zoo_sections.rst")
DOC_WORKFLOW_DIR = ROOT / "docs" / "source" / "_static" / "workflow"
README_PATHS = (
    ROOT / "README.rst",
    *(ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"),
)
LANGUAGE_PICKER_PAGE = ROOT / "docs" / "i18n" / "readme" / "README.md"
LANGUAGE_PICKER_BEGIN = ".. spacr-language-picker-begin"
LANGUAGE_PICKER_END = ".. spacr-language-picker-end"
WORKFLOW_BEGIN = ".. spacr-workflow-begin"
WORKFLOW_END = ".. spacr-workflow-end"
MODEL_ZOO_BEGIN = ".. spacr-model-zoo-begin"
MODEL_ZOO_END = ".. spacr-model-zoo-end"
HARDWARE_BEGIN = ".. spacr-hardware-begin"
HARDWARE_END = ".. spacr-hardware-end"
INSTALLER_BEGIN = ".. spacr-installer-links-begin"
INSTALLER_END = ".. spacr-installer-links-end"
INSTALLER_SUBSTITUTIONS = (
    "InstallerWindows", "InstallerMacOS", "InstallerLinux", "InstallerLegacy",
)
DATA_SUBSTITUTIONS = (
    "DataBioStudies", "DataHuggingFace", "DataNCBI", "DataSpaCRPower",
    "DataBioRxiv",
)
INSTALLER_ROW = (
    "|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|"
)
DATA_ROW = (
    "|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| "
    "|DataBioRxiv|"
)
_IMAGE_SUBSTITUTION_RE = re.compile(
    r"(?m)^\.\. \|(?P<name>[^|\n]+)\| image::[^\n]*"
    r"(?:\n   [^\n]*)*(?:\n)?"
)

RESOURCE_SLATE = (43, 47, 58, 255)  # #2B2F3A
WORKFLOW_TILE = (13, 14, 16, 255)  # GUI dark-theme surface, #0D0E10
WHITE = (255, 255, 255, 255)
WORKFLOW_RIM = (255, 255, 255, 96)

BUTTON_SIZE = 512
BUTTON_RADIUS = 32
BUTTON_MARK = round(BUTTON_SIZE * 0.80)
README_LOGO_SIZE = (920, 380)
README_LOGO_MARK = 340
#: Corner radius of the logo's panel, in proportion with BUTTON_RADIUS on
#: a canvas this much larger than a button.
README_LOGO_RADIUS = 56

#: The API documentation's logo: a SQUARE, because Sphinx themes put it in
#: a sidebar slot sized for one, and dark teal rather than the workflow
#: tile's near-black so it reads as a mark rather than a hole in the page.
DOCS_LOGO_SIZE = 512
DOCS_LOGO_RADIUS = 96
DOCS_LOGO_MARK = 340
DOCS_LOGO_TEAL = (16, 52, 58, 255)  # #10343A
# ONE GRID OF IDENTICAL TILES, asked for on 2026-09-02: "make them all
# the same size and present them on an evenly spaced grid with 6 modules
# per row".
#
# WHAT THIS REPLACED, so it is not rebuilt by accident. The modules used
# to be drawn as two different things: a six-wide pipeline STRIP joined by
# arrow glyphs, then three named bands ("Data", "Tools", "Assays") of
# smaller tiles underneath. That made three separate problems, and they
# were one problem:
#
#   * the two kinds of tile could not be the same size, because the strip
#     had to fit six buttons AND five arrows in the width the bands used
#     for six buttons, so the arrows came out of the buttons;
#   * the bands were 6, 5 and 4 wide, so three of four rows ended short
#     and the grid never read as a grid;
#   * the band titles duplicated the Home screen's own grouping, and went
#     stale every time Home was restructured.
#
# Deleting the arrows removed the width difference, which is what let
# every tile become the same size, which is what makes an even grid
# possible. The three changes are one change.
GRID_COLUMNS = 6
#: The transparent margin around each tile inside its square canvas. It is
#: what puts an even gutter BETWEEN neighbours: RST inline images are
#: joined with a zero-width ``\ `` separator, so tiles touch unless their
#: own canvases hold the gap. Two margins meet between any two tiles, so
#: the visible gutter is twice this.
TILE_PADDING = 16
TILE_SIZE = BUTTON_SIZE - 2 * TILE_PADDING
#: Six of these per row. 6 x 16 = 96%, and the four points of headroom are
#: deliberate: a browser that rounds each percentage up must still not push
#: the sixth tile onto a line of its own.
TILE_DISPLAY_PERCENT = 16.0
TILE_DISPLAY_WIDTH = f"{TILE_DISPLAY_PERCENT}%"
#: The underline that turns a section name into a heading over its band.
#:
#: ``^`` because RST binds a level to an underline CHARACTER by order of
#: first appearance in the file, and this one block is included into three
#: documents whose heading ladders were each decided elsewhere. README.rst
#: introduces ``=``, ``~`` then ``-`` in that order, so ``~`` there is a
#: LEVEL ABOVE ``-`` and a band heading written with it would break out of
#: the "spaCR modules" section it belongs to. ``^`` appears in none of the
#: three, so wherever the block lands it is introduced immediately under
#: the heading above it and becomes that heading's child.
SECTION_HEADING_CHAR = "^"

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
def _home_layout() -> "tuple[tuple[str, ...], dict[str, tuple[str, ...]]]":
    """``(sections, {section: keys})`` for the tiles Home draws, from Home.

    READ, NOT RESTATED. This file used to carry its own copy of both --
    a SECTION_ORDER tuple and an APP_ORDER table naming every key by
    hand -- with a comment explaining that the order had to be explicit
    because late self-registration made registry order depend on import
    order. That reasoning was sound and the copy was still wrong within a
    day of Home changing: the README kept advertising seven sections and
    thirty-eight tiles after Home became four and twenty-one, and every
    module folded onto a host masthead was still offered as a place to
    start.

    ``SECTION_TILE_ORDER`` solves the problem the copy existed for. It is
    a literal table in ``spacr.qt.app``, so it is as reproducible across
    processes as a literal here -- and it is the one the GUI itself draws
    from, so the two cannot disagree.

    The Core pipeline keys are dropped: they are drawn as the workflow
    strip above the grid, and a second tile for each would say the same
    thing twice.
    """
    from spacr.qt.app import SECTION_ORDER, SECTION_TILE_ORDER

    pipeline = {key for key, _label in MAIN_PIPELINE}
    grouped = {
        section: tuple(key for key in SECTION_TILE_ORDER.get(section, ())
                       if key not in pipeline)
        for section in SECTION_ORDER
    }
    # A section whose every member is in the pipeline draws no grid row,
    # and an empty heading is worse than no heading.
    sections = tuple(s for s in SECTION_ORDER if grouped[s])
    return sections, grouped


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
    """Draw the logo on a dark rounded panel.

    THE MARK IS WHITE. On a transparent canvas that is invisible against
    GitHub's light theme -- the README opened to a blank space where the
    logo should be, for every reader not using dark mode.

    A light and a dark variant swapped by ``prefers-color-scheme`` is the
    usual answer and is not available here: GitHub renders README.rst
    through docutils, which does not pass raw HTML, so there is no
    ``<picture>`` element to switch on. One image has to work in both
    themes, which means it has to bring its own background.

    The panel is the workflow tile's own surface and rim at the button
    corner radius, so the logo reads as the first element of the same
    design system as the module buttons directly beneath it.
    """
    canvas = Image.new("RGBA", README_LOGO_SIZE, (0, 0, 0, 0))
    panel = Image.new("RGBA", README_LOGO_SIZE, (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle(
        (0, 0, README_LOGO_SIZE[0] - 1, README_LOGO_SIZE[1] - 1),
        radius=README_LOGO_RADIUS, fill=WORKFLOW_TILE, outline=WORKFLOW_RIM,
        width=2,
    )
    canvas.alpha_composite(panel)
    logo = _fit(Image.open(ICON_DIR / "logo_spacr.png"), README_LOGO_MARK)
    canvas.alpha_composite(
        logo,
        ((canvas.width - logo.width) // 2, (canvas.height - logo.height) // 2),
    )
    return canvas


def render_docs_logo() -> Image.Image:
    """The white mark on a dark teal rounded square, for the API docs.

    Same problem as the README logo and the same reason it cannot be
    solved with a theme swap: the mark is pure white, Sphinx serves one
    ``html_logo`` for both colour schemes, and against the light theme's
    white sidebar a white logo is a blank space.

    Square rather than the README's wide canvas because a Sphinx sidebar
    gives the logo a square slot, and teal rather than the workflow
    tile's near-black so it reads as a mark on a light page rather than a
    hole punched in it.
    """
    canvas = Image.new("RGBA", (DOCS_LOGO_SIZE, DOCS_LOGO_SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).rounded_rectangle(
        (0, 0, DOCS_LOGO_SIZE - 1, DOCS_LOGO_SIZE - 1),
        radius=DOCS_LOGO_RADIUS, fill=DOCS_LOGO_TEAL,
    )
    logo = _fit(Image.open(ICON_DIR / "logo_spacr.png"), DOCS_LOGO_MARK)
    canvas.alpha_composite(
        logo,
        ((DOCS_LOGO_SIZE - logo.width) // 2,
         (DOCS_LOGO_SIZE - logo.height) // 2),
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


def render_module_tile(key: str, label: str) -> Image.Image:
    """Render one module as a linked grid tile.

    EVERY tile goes through here -- the six pipeline modules and the
    fifteen others alike -- which is the whole point: identical size is
    not something the two paths have to be kept in agreement about, it is
    something there is only one path to produce.

    The drawn button is ``TILE_SIZE`` centred in a ``BUTTON_SIZE`` canvas.
    The canvas is square and constant so the RST percentage width means
    the same thing for every tile, and the margin inside it is what
    separates neighbours on the page.
    """
    tile = _render_workflow_tile(key, label)
    tile = tile.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (BUTTON_SIZE, BUTTON_SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(tile, (TILE_PADDING, TILE_PADDING))
    return canvas


def _inline_image_row(names: list[str]) -> str:
    """Join RST substitutions without browser whitespace between images."""
    return r"\ ".join(names)


def _registry() -> list[tuple[str, str, str, str]]:
    """Every REGISTERED app, folded ones included.

    Used where the question is "does this module exist" -- the API
    destination table, for one, because a folded module still has an API
    page and a button that opens it.
    """
    from spacr.qt import register_self_registering_modules
    from spacr.qt.app import APPS

    # ``spacr.qt.run`` performs this registration immediately before launch.
    # Documentation must describe the tiles users actually see after launch,
    # not the smaller import-time snapshot of the registry.
    register_self_registering_modules()
    return list(APPS)


def _tiled_registry() -> list[tuple[str, str, str, str]]:
    """Only the rows that draw a TILE on Home.

    The distinction that instruction 318 introduced and this file missed:
    a module folded onto a host's masthead keeps its registry row, its
    screen and its API page, and loses only its tile. The README grid
    advertises places to START, so it draws these; the API documents what
    EXISTS, so it keeps all of them.
    """
    from spacr.qt.app import tiled_apps

    _registry()
    return list(tiled_apps())


def _api_urls() -> dict[str, str]:
    """Return the home-screen API destinations used by the running app.

    THROUGH `api_docs_url`, WHICH IS WHAT THE APP ITSELF CALLS, rather than
    by rebuilding the URL from `_APP_API_MODULE` here. This function used to
    format its own string, so the README and the running GUI agreed only for
    as long as nobody added a case to the resolver -- and instruction 366
    added one: the six tiles that share three module pages now carry an
    anchor to the entry point that answers for them, which a second
    formatter would silently drop and send four toxoplasma assays back to
    the same paragraph.

    `key` is left empty deliberately. That is the module-level question --
    "what is this module" -- and it is the only one a tile asks.
    """
    from spacr.qt.screens.settings_model import api_docs_url

    return {
        key: api_docs_url(key)
        for key, _label, _desc, _section in _registry()
    }


def _grouped_apps() -> dict[str, list[tuple[str, str]]]:
    apps = _tiled_registry()
    section_order, app_order = _home_layout()
    pipeline_keys = {key for key, _label in MAIN_PIPELINE}
    by_key = {key: (label, section) for key, label, _desc, section in apps}
    expected = {
        key for keys in app_order.values() for key in keys
    }
    actual = set(by_key) - pipeline_keys
    if actual != expected:
        missing = sorted(expected - actual)
        additional = sorted(actual - expected)
        raise ValueError(
            "documented application order does not match the Home registry: "
            f"missing={missing}, additional={additional}"
        )

    expected_order = tuple(
        key for section in section_order for key in app_order[section]
    )
    actual_order = tuple(
        key for key, _label, _desc, _section in apps
        if key not in pipeline_keys
    )
    if actual_order != expected_order:
        raise ValueError(
            "documented application order does not match the Home registry: "
            f"expected={expected_order}, actual={actual_order}"
        )

    grouped = {}
    for section in section_order:
        keys = app_order[section]
        refiled = [key for key in keys if by_key[key][1] != section]
        if refiled:
            raise ValueError(
                f"documented applications are no longer in {section!r}: "
                f"{refiled}"
            )
        grouped[section] = [(key, by_key[key][0]) for key in keys]
    return grouped


#: The module whose Help menu opens `feature_dict`, which registers its
#: own entry from `spacr/qt/widgets/feature_dictionary.py` rather than from
#: app.py's central table. Named here so the reachability report is
#: complete; without it the Feature Dictionary looks stranded when it is
#: simply registered somewhere else.
_HELP_FROM_ITS_OWN_WIDGET = ("feature_dict",)


#: One row per hardware configuration the README describes.
#:
#: ``(label, maturity, Accelerator kwargs)``. The kwargs are what the
#: resolver would have found on that machine, and `capabilities()` is
#: then asked the real question with that answer in place -- the same
#: trick `tests/test_the_accelerator_resolver.py` uses to exercise 19
#: backends on a machine that has one.
#:
#: MATURITY IS A SEPARATE CLAIM FROM SUPPORT, and it is why this table
#: has three colours rather than two. "The code exists" and "somebody ran
#: it on that hardware" are different things:
#:
#:   stable  CUDA. Years of use, and the resolver's own tests assert
#:           instruction 319 left it unchanged.
#:   beta    everything else. Metal on an Intel Mac was MEASURED -- 444.5 s
#:           to 3.2 s for one 256x256 Cellpose image -- but on one machine,
#:           one day old. ROCm, XPU and Apple Silicon Metal are implemented
#:           and measured on NOTHING, because no such hardware was here.
#:
#: The CPU row is `stable`, deliberately. Everything works without a GPU;
#: it is only slower. Marking it otherwise would be the most damaging
#: thing this table could say.
HARDWARE_ROWS = (
    ("NVIDIA (CUDA)", "stable",
     dict(kind="cuda", device="cuda", label="NVIDIA")),
    ("AMD on Linux (ROCm)", "beta",
     dict(kind="rocm", device="cuda", label="AMD")),
    ("AMD in an Intel Mac (Metal)", "beta",
     dict(kind="mps", device="mps", label="AMD", float64=False)),
    ("Apple Silicon (Metal)", "beta",
     dict(kind="mps", device="mps", label="Apple", float64=False)),
    ("Intel Arc/Xe (XPU)", "beta",
     dict(kind="xpu", device="xpu", label="Intel", float64=False)),
    ("No GPU", "stable",
     dict(kind="cpu", device="cpu", label="CPU")),
)

#: The tasks the table has a column for, matched against the START of
#: what `capabilities()` returns so a detail sentence can be reworded
#: without silently emptying a column.
HARDWARE_TASKS = (
    ("Cellpose 4", "Segmentation"),
    ("Torch", "Model training"),
    ("UMAP / clustering", "UMAP"),
)

#: Circles, not colours. An RST colour role emits `<span class="green">`
#: with NO inline style, and GitHub applies no custom CSS to a README --
#: so coloured text renders in the body colour and a legend describing it
#: describes nothing the reader can see. Tested before it was written.
#: These render identically on GitHub, PyPI, Sphinx and a plain editor.
GREEN, PURPLE, RED = "\U0001F7E2", "\U0001F7E3", "\U0001F534"


def _cell(accelerated: bool, maturity: str, row_has_a_gpu: bool) -> str:
    """One table cell: the mark, then what it runs on.

    THREE STATES, AND THE THIRD IS THE ONE WORTH GETTING RIGHT.

    A machine WITH a GPU that cannot accelerate a task is RED, and the
    legend says "CPU support only" rather than "not supported" -- changed
    2026-09-01, because the task DOES run, on the processor. UMAP on
    Metal is the case: cuML ships for CUDA only, so there is nothing to
    wait for and nothing missing except the acceleration.

    A machine with NO GPU is GREEN on every row. Its cells say CPU
    because that is what they use, and marking them red would say spaCR
    does not support running without a GPU. That is false, and it is the
    most damaging claim this table could make -- every task runs, every
    result is identical, only the clock changes.
    """
    if not row_has_a_gpu:
        return f"{GREEN} CPU"
    if not accelerated:
        return f"{RED} CPU"
    return f"{GREEN if maturity == 'stable' else PURPLE} GPU"


def _capabilities_for(**kwargs) -> "dict[str, bool]":
    """``{task: accelerated}`` as `capabilities()` answers for one machine.

    The REAL function, with the resolver's cache holding the accelerator
    that machine would have found. Asking the real function is the point:
    a second table of per-task facts here would be free to drift from the
    one the setup screen and `spacr-doctor` both render.
    """
    from spacr import accelerator

    previous = accelerator._CACHED
    try:
        accelerator._CACHED = accelerator.Accelerator(**kwargs)
        return {task: bool(ok) for task, ok, _detail
                in accelerator.capabilities()}
    finally:
        accelerator._CACHED = previous


def _hardware_table() -> str:
    """The README's hardware table, derived rather than typed.

    A ``list-table`` rather than an RST simple table: the cells carry
    emoji, which are one character to ``len()`` and two columns wide in
    most renderers, so the ``===`` rules of a simple table would be
    aligned to the wrong width and the table would not parse. A
    list-table needs no alignment at all.
    """
    header = ["Hardware"] + [name for name, _prefix in HARDWARE_TASKS]
    rows = []
    for label, maturity, kwargs in HARDWARE_ROWS:
        answers = _capabilities_for(**kwargs)
        has_gpu = kwargs.get("kind") != "cpu"
        cells = []
        for _name, prefix in HARDWARE_TASKS:
            matched = [ok for task, ok in answers.items()
                       if task.startswith(prefix)]
            if not matched:
                raise ValueError(
                    f"no capability row starts with {prefix!r}; the table "
                    "and spacr.accelerator.capabilities() have diverged")
            cells.append(_cell(matched[0], maturity, has_gpu))
        rows.append([label] + cells)

    lines = [".. list-table::", "   :header-rows: 1", "   :widths: 32 18 18 22",
             ""]
    for row in [header] + rows:
        lines.append(f"   * - {row[0]}")
        for cell in row[1:]:
            lines.append(f"     - {cell}")
    lines += [
        "",
        f"{GREEN} supported (stable) \u2003 {PURPLE} implemented (beta) "
        f"\u2003 {RED} CPU support only",
        "",
    ]
    return "\n".join(lines)


def _model_zoo_rows() -> "list[tuple[str, str, str, str]]":
    """The published models as ``(key, architecture, dataset, versus stock)``.

    READ FROM THE SHIPPED LITERAL, never restated. ``BUNDLED_REMOTE_MODELS``
    in ``spacr/model_zoo.py`` is the only list of what spaCR publishes, and a
    second hand-written copy in the README is a copy that goes stale --
    which is exactly the fault part 1 of instruction 362 spent a day
    removing from the module grid.

    THE LITERAL RATHER THAN ``catalogue()``, since 2026-09-02. The three
    fields this table wants are short, table-shaped restatements written for
    it -- ``architecture``, ``dataset``, ``versus_stock`` -- and
    ``ModelEntry`` maps a fixed set of keys, so they would be dropped on the
    way through. Reading the literal also does what ``catalogue()`` had to
    be argued into doing: THE GENERATED README MUST NOT DEPEND ON THE
    MACHINE THAT GENERATED IT, and the literal cannot pick up this
    checkout's bundled models, a plugin's entries, or a catalogue named by
    an environment variable. The guards those needed are simply not
    reachable from here.

    Importing :mod:`spacr.model_zoo` is cheap: no torch, no Qt, no network.
    """
    from spacr.model_zoo import BUNDLED_REMOTE_MODELS

    rows = []
    for entry in BUNDLED_REMOTE_MODELS:
        rows.append((
            str(entry.get("key") or entry.get("name") or ""),
            str(entry.get("architecture") or "").strip().rstrip("."),
            str(entry.get("dataset") or "").strip().rstrip("."),
            str(entry.get("versus_stock") or "").strip().rstrip("."),
        ))
    return rows


def _model_zoo_sections() -> str:
    """One API section per published model, with its Hugging Face link.

    Instruction 370 asks for "a API section for each model in the model zoo",
    linking to Hugging Face. Measured 2026-09-03: the generated
    `model_zoo_table.rst` was INCLUDED BY NOTHING -- so the API had no model
    zoo page at all, and neither the table nor a per-model section reached a
    reader. Both are emitted here and the page that includes them is
    `docs/source/model_zoo.rst`.

    THE FULL PROSE LIVES HERE, NOT IN THE TABLE. The table is deliberately
    three short columns (see `_model_zoo_table`); this is where a reader who
    clicked through gets `trained_on` in full, every note, the checksum and
    the place to fetch it from -- which is the split 370 asks for between the
    surface with the least room and the one with the most.
    """
    from spacr.model_zoo import BUNDLED_REMOTE_MODELS

    lines: list[str] = []
    for entry in BUNDLED_REMOTE_MODELS:
        key = str(entry.get("key") or entry.get("name") or "")
        if not key:
            continue
        title = str(entry.get("display_name") or key)
        lines += [title, "-" * len(title), ""]

        architecture = str(entry.get("architecture") or "").strip()
        if architecture:
            lines += [f"**Architecture.** {architecture}", ""]

        trained_on = str(entry.get("trained_on") or "").strip()
        if trained_on:
            lines += [f"**Trained on.** {trained_on}", ""]

        versus = str(entry.get("versus_stock") or "").strip()
        if versus:
            lines += [f"**Measured.** {versus}", ""]

        for note in entry.get("notes") or ():
            note = str(note).strip()
            if note:
                lines += [f"* {note}"]
        if entry.get("notes"):
            lines += [""]

        repo = str(entry.get("repo_id") or "").strip()
        if repo:
            kind = str(entry.get("repo_type") or "model")
            where = "datasets/" if kind == "dataset" else ""
            lines += [
                f"Published as `{repo} "
                f"<https://huggingface.co/{where}{repo}>`_, as "
                f"``{entry.get('name')}``.",
                "",
            ]

        digest = str(entry.get("sha256") or "").strip()
        if digest:
            # THE CHECKSUM IS PUBLISHED BECAUSE FETCH REFUSES WITHOUT ONE.
            # An entry with no digest is declined rather than installed --
            # a truncated or substituted checkpoint cannot be told from the
            # real one -- so the digest is part of what makes the model
            # usable, not a footnote.
            lines += [f"SHA-256 ``{digest}``.", ""]

    return "\n".join(lines).rstrip() + "\n"


def _model_zoo_table() -> str:
    """The README's model-zoo table, generated from the catalogue."""
    rows = _model_zoo_rows()
    if not rows:  # pragma: no cover - the catalogue is never empty
        return ""
    # THREE COLUMNS, ASKED FOR ON 2026-09-02: "this new table for the model
    # zoo contains way to much information. just state the model name and
    # architecture, training dataset (staining + number of images from n
    # datasets), and performance on hold out data compared to stock model".
    # The full `trained_on` prose and both `notes` are still published -- on
    # the Model Zoo screen, and in the scorecard instruction 370 describes.
    # This table is the index, not the record.
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 24 34 42",
        "",
        "   * - Model",
        "     - Training data",
        "     - Hold-out performance",
    ]
    for key, architecture, dataset, versus in rows:
        lines.extend([
            f"   * - ``{key}``",
            f"       ({architecture})",
            f"     - {dataset}",
            f"     - {versus}",
        ])
    return "\n".join(lines)


#: "Languages" written in each translated README's own language.
#:
#: A DELIBERATE MIRROR of ``LANGUAGE_PICKER_LABELS`` in
#: ``tools/build_documentation_i18n.py``. Two generators write this label:
#: that one when a translated README is rebuilt from the English source, this
#: one when the picker block is regenerated, and if the two ever disagree a
#: regeneration would quietly put the English word "Languages:" back at the
#: top of the Swedish page. It is copied rather than imported because that
#: module imports ``build_i18n_catalogs`` by bare name and is therefore only
#: importable with ``tools/`` on ``sys.path`` -- which a packaging script
#: should not be arranging. ``tests/test_the_language_picker_is_a_dropdown.py``
#: fails if the two tables drift apart.
LANGUAGE_PICKER_LABELS = {
    "en": "Languages",
    "sv": "Språk",
    "de": "Sprachen",
    "es": "Idiomas",
    "zh_CN": "语言",
    "pt": "Idiomas",
    "hi": "भाषाएँ",
    "ko": "언어",
    "is": "Tungumál",
    "fr": "Langues",
}

#: The locales a fluent speaker actually read, recorded by instruction 316.
#:
#: The other six -- es, zh_CN, pt, hi, ko, fr -- are machine drafts nobody
#: who speaks them has checked, and instruction 357's fourth guideline
#: forbids implying otherwise. A picker is exactly where that difference has
#: to be visible: a menu of ten languages with nothing said about them is
#: itself the claim that all ten are equally good.
SPOT_CHECKED_LOCALES = ("sv", "de", "is")

#: The caret is what makes the link read as a control rather than as one more
#: link in a row. It is a plain character, not markup: GitHub gives a README
#: no CSS of its own, so a real menu affordance cannot be drawn here.
PICKER_CARET = "\u25be"
PICKER_GLOBE = "\U0001f310"


def _languages() -> list:
    """The ten shipped languages, from the registry the application uses.

    ``spacr.qt.i18n.LANGUAGES`` is the only list of what spaCR ships, and it
    already carries both spellings of each name. Retyping the native names
    here is how a picker ends up offering a language the Preferences dialog
    does not have, or spelling it differently from the way the application
    does.
    """
    from spacr.qt.i18n import LANGUAGES

    return list(LANGUAGES)


def _language_picker_line(code: str) -> str:
    """The one-line language control for one README.

    ONE LINK, NOT TEN. The nine side-by-side links this replaces were the
    first thing under the title on every page, and the request was for a
    menu instead. The link shows the language you are reading now, which is
    the label a language switcher carries everywhere else, and the leading
    word is localized so a reader who cannot read the link text still meets
    the word "language" in their own.

    ``Languages:`` must stay the first token of the English line:
    ``translatable_blocks`` in ``tools/build_documentation_i18n.py`` keys on
    exactly that prefix to hold the picker out of the translation model, and
    a picker that goes through a model comes back with its RST delimiters
    rearranged.
    """
    label = LANGUAGE_PICKER_LABELS[code]
    native = {language.code: language.native_name
              for language in _languages()}[code]
    # The English README sits at the repository root and the translated ones
    # sit beside the picker page, so the same page has two relative paths.
    target = ("docs/i18n/readme/README.md" if code == "en" else "README.md")
    return f"{label}: `{PICKER_GLOBE} {native} {PICKER_CARET} <{target}>`_"


def _language_picker_page() -> str:
    """The picker itself: a ``<details>`` menu on a Markdown page.

    WHY THE MENU IS NOT ON THE FRONT PAGE, measured on 2026-09-02 rather
    than assumed. GitHub renders ``README.rst`` through docutils with
    github/markup's settings, and those set ``raw_enabled=False``:

    * ``.. raw:: html`` is refused. Rendered locally with those exact
      settings it emits a "raw directive disabled" system message, and on
      github.com the block is printed as a literal ``<pre>`` of escaped
      HTML -- checked on a real rendered page, dask/dask's
      ``docs/source/index.rst``, fetched through the contents API with
      ``Accept: application/vnd.github.html``, which is the renderer the
      site itself uses.
    * ``<details>`` typed straight into the RST is escaped to a visible
      ``&lt;details&gt;``, because to docutils it is not HTML at all, only
      text with angle brackets in it.

    So a collapsible menu cannot exist in a reStructuredText README, and no
    JavaScript runs in one either. It CAN exist in Markdown: posting this
    page's exact shape to ``api.github.com/markdown`` with ``mode=gfm``
    returns ``<details open="">`` with the table nested inside it, so what is
    written here is GitHub's own rendering rather than a hope about it.

    THE MENU IS ``open``. This page has nothing else on it, and a reader who
    followed a link to reach the languages should not have to click a second
    time to see them. Collapsing it is still theirs to do.

    The status column is not decoration: nine of these are machine drafts,
    three of those were sampled by a fluent speaker, and instruction 316
    requires the difference to be visible wherever the translations are
    offered.
    """
    english_status = "Source text. Every other README is translated from it."
    rows = []
    for language in _languages():
        target = ("../../../README.rst" if language.code == "en"
                  else f"README.{language.code}.rst")
        name = f"[{language.native_name}]({target})"
        if language.native_name != language.english_name:
            name = f"{name} ({language.english_name})"
        if language.code == "en":
            status = english_status
        elif language.code in SPOT_CHECKED_LOCALES:
            status = (
                "Machine draft. A fluent speaker read a sample and their "
                "corrections are kept."
            )
        else:
            status = "Machine draft. No fluent-speaker review."
        rows.append(f"| {name} | {status} |")
    return "\n".join([
        "<!-- Generated by packaging/generate_readme_visuals.py.",
        "     Edit that generator, not this file. -->",
        "",
        "# spaCR in your language",
        "",
        "<details open>",
        f"<summary><b>{PICKER_GLOBE} Choose a language</b></summary>",
        "",
        "| Language | Translation |",
        "| --- | --- |",
        *rows,
        "",
        "</details>",
        "",
        "Every spaCR README links here, so a reader who lands in the wrong",
        "language is one click from the right one.",
        "",
        "English is the source text: where a translation and the English",
        "README disagree, the English one is right. The models that drafted",
        "the translations and their licenses are listed in",
        "[TRANSLATION_MODELS.md](../TRANSLATION_MODELS.md), and the measured",
        "per-locale coverage in [COVERAGE.md](../COVERAGE.md).",
        "",
        "The front page carries a single link to this page rather than the",
        "menu itself because `README.rst` is reStructuredText: GitHub renders",
        "it with raw HTML disabled, so `<details>` is printed as text there",
        "and only renders here, in Markdown.",
        "",
    ])


def _write_the_language_picker(path) -> bool:
    """Replace the marked language-picker block in one README.

    Between markers for the reason every other generated block here is: a
    regeneration has to be able to REPLACE what it wrote last time, and
    finding that by content is how a generator ends up with two pickers in
    one file.
    """
    text = path.read_text(encoding="utf-8")
    start = text.find(LANGUAGE_PICKER_BEGIN)
    end = text.find(LANGUAGE_PICKER_END)
    if start < 0 or end < 0:
        return False
    end += len(LANGUAGE_PICKER_END)
    match = re.fullmatch(r"README\.(?P<language>[^.]+)\.rst", path.name)
    code = match.group("language") if match else "en"
    line = _language_picker_line(code)
    block = (f"{LANGUAGE_PICKER_BEGIN}\n\n{line}\n\n"
             f"{LANGUAGE_PICKER_END}")
    updated = text[:start] + block + text[end:]
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return True


def _write_the_model_zoo_table(path) -> bool:
    """Replace the marked model-zoo block in ``path``.

    Between markers for the same reason the hardware table is: a
    regeneration has to be able to REPLACE what it wrote last time, and
    locating that by content is how a generator ends up with two copies of
    its own output in one file.
    """
    text = path.read_text(encoding="utf-8")
    start = text.find(MODEL_ZOO_BEGIN)
    end = text.find(MODEL_ZOO_END)
    if start < 0 or end < 0:
        return False
    end += len(MODEL_ZOO_END)
    block = f"{MODEL_ZOO_BEGIN}\n\n{_model_zoo_table()}\n\n{MODEL_ZOO_END}"
    updated = text[:start] + block + text[end:]
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return True


def _write_the_hardware_table(path) -> bool:
    """Replace the marked block in ``path`` with the generated table.

    Between markers rather than appended, for the reason the workflow
    grid is: a regeneration must be able to REPLACE what it wrote last
    time, and finding that by content is how a generator ends up with
    two copies of its own output in the file.
    """
    text = path.read_text(encoding="utf-8")
    start = text.find(HARDWARE_BEGIN)
    end = text.find(HARDWARE_END)
    if start < 0 or end < 0:
        return False
    end += len(HARDWARE_END)
    block = f"{HARDWARE_BEGIN}\n\n{_hardware_table()}\n{HARDWARE_END}"
    updated = text[:start] + block + text[end:]
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return True


def _fold_hosts() -> "dict[str, str]":
    """``{folded key: host key}`` for every module opened from a masthead.

    ASKED OF `spacr.qt.app.folded_children`, WHICH IS THE RESOLVER THE DOCK
    AND THE MENU BAR USE. This function used to do its own walk over
    `map_barcodes.FOLD_HOST_MODULES` -- a different table, with the same
    name as the one in `fold_strip`, mapping host keys to screen module
    names -- and it was wrong in a way that only showed on this page:

      * it missed MAKE MASKS entirely, because Make Masks spells its list
        `FOLD_ORDER` rather than `FOLDED_APPS`. Six modules -- Cellpose
        Workbench, Mask the whole folder, Model Compare, Model Zoo, Curate
        and Napari Bridge -- were folded in the application and absent from
        the documentation of what is folded, which is the exact failure
        instruction 374 describes;
      * it was a FOURTH copy of the fold tables, after the hosts
        themselves, `fold_strip.folded_modules` and `folded_children`,
        and 374 asks for it not to be.

    `folded_children` handles both spellings and both host lists
    (`fold_strip.FOLD_HOST_MODULES` plus `app._EXTRA_FOLD_HOSTS`), never
    raises, and is what draws the nesting in the running window.

    `fold_strip.folded_modules` IS THEN ASKED FOR WHAT IT ALONE KNOWS, and
    the reason is a gap in the application rather than a preference here.
    The two shipped resolvers do not cover the same set:

      * `folded_children` keys its answer by HOST KEY, which it reads from
        the host module's `APP_KEY` or `HOST_KEY`. `annotate` declares
        neither, so its one fold -- Annotator Agreement -- falls out of
        that answer, and out of the dock with it;
      * `folded_modules` keys its answer by host MODULE, so it keeps
        Annotator Agreement, but its own docstring records that it walks
        eight hosts while the dock draws eleven -- the seven modules folded
        onto Graph Builder, QC Dashboard and Database Browser are missing
        from it.

    Neither is a superset of the other, so the union is the only complete
    answer available without writing a fifth table. A module name is turned
    into a host key only when it IS a registered key, so this cannot invent
    a host. Giving `spacr/qt/screens/annotate.py` an `APP_KEY` would make
    the second half unnecessary -- and would fix the dock at the same time
    -- but that is `spacr/` source and not this item's to change.
    """
    from spacr.qt.app import APPS, folded_children
    from spacr.qt.widgets.fold_strip import folded_modules

    hosts = {key: host
             for host, keys in folded_children().items()
             for key in keys}
    registered = {row[0] for row in APPS}
    for key, entry in folded_modules().items():
        host = str(entry[3]).rsplit(".", 1)[-1]
        if key not in hosts and host in registered:
            hosts[key] = host
    return hosts


def _documentation_folds() -> str:
    """The API page's "reached from" reference.

    EVERY MODULE KEEPS ITS API PAGE, including the folded ones: a folded
    module still has a screen, a settings model and a headless entry
    point, and its page is what a scripting user reads. What changes is
    how it is DESCRIBED -- a button on its host, not a place to start.

    The workflow grid above it draws only the twenty-one tiled modules, so
    without this table a reader could not reach the other twenty-three
    from the API index at all.
    """
    from spacr.qt.app import APPS, _HELP_MODULES
    from spacr.qt.screens.settings_model import api_docs_url

    names = {key: label for key, label, _desc, _section in APPS}
    urls = _api_urls()
    hosts = _fold_hosts()
    help_keys = {key for key, *_rest in _HELP_MODULES}

    # THE SAME RESOLVER THE BUTTONS USE. `fold_description` reads the
    # registry first and the declared catalogue second, so a module with
    # no registry row still gets its real name -- title-casing the key
    # gives "Explain Cv" and "Pca", which are not what anything calls
    # them.
    from spacr.qt.screens.map_barcodes import fold_description

    # What `api_docs_url` answers for a key it has never heard of: the API
    # index itself. Fourteen folded modules used to come out of this page
    # as bare text, because `_api_urls` is keyed on the REGISTRY and a
    # module folded hard enough loses its registry row. Asking the
    # resolver directly finds the page for all but five, and this sentinel
    # is how those five are told apart from a real destination -- linking
    # them to the index would point the reader back at the page they are
    # already reading.
    unresolved = api_docs_url("no_such_module_key")

    def _link(key: str) -> str:
        label = names.get(key) or fold_description(key)[0]
        if not label:
            label = key.replace("_", " ").title()
        target = urls.get(key) or api_docs_url(key)
        if not target or target == unresolved:
            return label
        return f"`{label} <{target}>`_"

    lines = [
        "Modules reached from another screen",
        "-----------------------------------",
        "",
        "These do not have a tile on the home screen. Each one answers a",
        "question about a run its host produced rather than starting a run",
        "of its own, so it opens as a page beside that host's settings,",
        "already pointed at the same project.",
        "",
        "They are not second-class: each is shipped, translated and",
        "documented like any other module, and the ones that are pipelines",
        "still run headlessly under ``spacr-run``. Every module below can",
        "also be reached from the command palette, which is the only route",
        "that covers all of them.",
        "",
        "Opened from a host's masthead",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
    ]
    by_host: "dict[str, list[str]]" = {}
    for key, host in sorted(hosts.items()):
        by_host.setdefault(host, []).append(key)
    for host in sorted(by_host, key=lambda h: names.get(h, h)):
        opened = ", ".join(_link(k) for k in sorted(by_host[host]))
        lines.append(f"* **{names.get(host, host)}** opens {opened}")
    lines.extend([
        "",
        "Opened from the Help menu",
        "~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
        "These inspect or administer work that already exists, rather than",
        "belonging behind any one module.",
        "",
    ])
    for key in sorted(help_keys | set(_HELP_FROM_ITS_OWN_WIDGET)):
        lines.append(f"* {_link(key)}")
    lines.append("")
    return "\n".join(lines)


def _grid_sections() -> "list[tuple[str, str, list[tuple[str, str, str]]]]":
    """The grid as Home groups it: ``(section, note, tiles)`` per band.

    THE HEADINGS ARE READ, NOT WRITTEN. Instruction 374 asks the API
    homepage to show the module structure the rest of spaCR shows, and
    the objection that removed the old band titles on 2026-09-02 was
    never that a grid should be flat -- it was that those titles were a
    SECOND COPY of Home's grouping, typed here, going stale every time
    Home was restructured. `_grouped_apps` closed that hole: it reads
    `SECTION_ORDER` and `SECTION_TILE_ORDER` out of `spacr.qt.app` and
    RAISES when the registry and the documented order disagree. So the
    grouping is now the registry's own answer, and a heading over each
    band cannot say anything Home does not.

    ``note`` is Home's own tab description for that section, from
    `SECTION_NOTES`, for the same reason: the API homepage carries no
    prose at all around the grid, and a bare four-word heading does not
    tell a reader which six modules are the pipeline. One sentence per
    band does, and it is the sentence the running GUI shows.

    CORE IS THE PIPELINE BAND. `_home_layout` drops the six pipeline
    keys from every section because they used to be drawn as a separate
    arrow-joined strip; they are ordinary tiles now, and
    `SECTION_TILE_ORDER["Core"]` is exactly :data:`MAIN_PIPELINE` in
    exactly that order, so putting them back under their own heading
    restores the section without reordering a single tile.

    ``image_path`` is relative to the icon root because the two groups
    still live in different folders: the pipeline tiles are also used
    elsewhere as module artwork, so they keep their place at the top
    level. Nothing about the tiles themselves differs any more -- only
    where they are stored.
    """
    from spacr.qt.app import SECTION_CORE, SECTION_NOTES

    grouped = _grouped_apps()
    bands = [(
        SECTION_CORE,
        [(key, label, f"workflow/{key}.png") for key, label in MAIN_PIPELINE],
    )]
    bands.extend(
        (section, [(key, label, f"workflow/apps/{key}.png")
                   for key, label in items])
        for section, items in grouped.items()
    )
    return [(section, SECTION_NOTES[section], tiles)
            for section, tiles in bands if tiles]


def _module_grid() -> "list[tuple[str, str, str]]":
    """Every module tile in one flat order: ``(key, label, image_path)``.

    THE PIPELINE FIRST, then the other modules in the order Home lists
    them. The six come first because they are the order a screen is
    actually analysed in and a reader scanning left-to-right from the top
    meets them in that order -- the arrows that used to say so were
    decoration on a sequence the layout already carries.

    This is :func:`_grid_sections` flattened, so the order the headings
    group and the order the artwork is rendered in cannot drift apart.
    """
    return [tile for _section, _note, tiles in _grid_sections()
            for tile in tiles]


def _grid_lines(names: "list[str]") -> "list[str]":
    """The grid as an RST LINE BLOCK, one line per row.

    MEASURED ON THE REAL PAGE, 2026-09-02: with each row as its own
    paragraph, the gap between rows was 2.5 to 3 times the gap between
    columns. The horizontal gutter is deliberate and controlled -- two tile
    canvases meet, so it is exactly ``2 * TILE_PADDING``. The vertical one
    was not controlled at all: it was GitHub's paragraph margin, about 16px,
    stacked on top of the same tile padding, and nothing in this repository
    sets it.

    A line block removes the paragraph entirely. Each row becomes one line
    inside a single block, so what separates two rows is the line box rather
    than a margin, and the only space left between them is the transparent
    padding the tiles already carry -- which is the same padding that makes
    the horizontal gap. The two gutters are then the same measurement rather
    than two numbers that happen to be close.

    SIX PER ROW STAYS EXPLICIT. The other way to drop the margins is to let
    one flowing paragraph wrap, and then the row length is the browser's
    decision: a narrow viewport or a different font size gives five or seven.
    A line block keeps the row length ours.
    """
    return [
        *(f"| {_inline_image_row(row)}"
          for row in _grid_rows(names)),
        "",
    ]


def _grid_rows(names: "list[str]") -> "list[list[str]]":
    """Split substitution names into rows of :data:`GRID_COLUMNS`."""
    return [names[start:start + GRID_COLUMNS]
            for start in range(0, len(names), GRID_COLUMNS)]


def _grid_markup(
    name_prefix: str,
    image_prefix: str,
    *,
    alt_template: str = "Open the {module} API",
) -> str:
    """The module grid, grouped by section, for ONE surface.

    THE ONE PLACE THE GRID IS BUILT. The README and the API homepage draw
    the same grid and used to build it in two nearly identical functions
    that differed in a substitution prefix and an asset root. Two builders
    is how a heading, a row width or a target lands on one page and not the
    other, which is the drift instruction 374 was filed about; one builder
    with two arguments cannot do that.

    :param name_prefix: substitution namespace. The README uses ``Module``
        and the Sphinx page ``DocModule`` because the two files are
        rendered by different toolchains and the tests tell them apart by
        it.
    :param image_prefix: what the paths from :func:`_grid_sections` hang
        off -- a repository-relative icon root for the README, ``/_static``
        for Sphinx, which copies the same PNGs.
    :param alt_template: accessibility text; localized per language for
        the translated READMEs.

    The heading underline is :data:`SECTION_HEADING_CHAR` on BOTH surfaces,
    which is not cosmetic. RST assigns heading levels by order of first
    appearance, and ``^`` is unused in README.rst and in both Sphinx index
    pages, so it is introduced directly beneath whatever heading the grid
    is included under and becomes that heading's child in all three. A
    character either file already uses would have bound to a level chosen
    somewhere else in that file.
    """
    urls = _api_urls()
    lines: list[str] = []
    definitions: list[str] = []
    for section, note, tiles in _grid_sections():
        names = {key: f"{name_prefix}_{key}" for key, _label, _path in tiles}
        lines.extend([
            section,
            SECTION_HEADING_CHAR * len(section),
            "",
            # Wrapped, because these land in README.rst beside prose a
            # human wrapped at the same width and a one-line paragraph
            # would be the only 130-column line in the file. RST joins the
            # lines back into one paragraph, so nothing about the rendered
            # page depends on where they break.
            # `break_on_hyphens=False`: the default splits
            # "multi-plate" across two lines, and RST rejoins them
            # with a space -- "multi- plate" on the rendered page.
            *textwrap.wrap(note, width=76, break_on_hyphens=False),
            "",
        ])
        lines.extend(_grid_lines([f"|{names[key]}|" for key, _l, _p in tiles]))
        for key, label, image in tiles:
            definitions.extend([
                f".. |{names[key]}| image:: {image_prefix}/{image}",
                f"   :width: {TILE_DISPLAY_WIDTH}",
                f"   :alt: {alt_template.format(module=label)}",
                f"   :target: {urls[key]}",
                # MIDDLE, and it MUST be. These are SUBSTITUTION definitions
                # used inline in a paragraph, and docutils accepts only
                # top/middle/bottom there -- "left" is a block-image value
                # and raises "not a valid value for the align option within
                # a substitution definition".
                #
                # The failure mode is why this comment is long: the
                # directive errors, the substitution is never defined, and
                # GitHub renders the reference as its alt text. The whole
                # grid turns into a column of blue links, which is what
                # happened when this was set to "left" on 2026-08-31.
                # Left-alignment comes from the rows being left-anchored
                # paragraphs and every tile sharing one canvas -- not from
                # this option.
                "   :align: middle",
            ])
    return "\n".join([*lines, *definitions]).rstrip()


def _readme_workflow(
    icon_prefix: str,
    *,
    alt_template: str = "Open the {module} API",
) -> str:
    """The README's module grid.

    The block's own title lives in the README ("spaCR modules"); the
    section headings inside it come from Home through
    :func:`_grid_sections`.
    """
    return _grid_markup("Module", icon_prefix, alt_template=alt_template)


def _documentation_workflow() -> str:
    """The same grid for the Sphinx pages, with its own asset paths."""
    heading = ["spaCR modules", "~~~~~~~~~~~~~", ""]
    return "\n".join(heading) + "\n" + _grid_markup(
        "DocModule", "/_static") + "\n"


def _replace_workflow_block(path: Path, markup: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.find(WORKFLOW_BEGIN)
    end = text.find(WORKFLOW_END)
    if start < 0 or end <= start:
        raise ValueError(f"{path} is missing its workflow markers")
    end += len(WORKFLOW_END)
    replacement = f"{WORKFLOW_BEGIN}\n\n{markup}\n\n{WORKFLOW_END}"
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def _normalize_linked_resource_blocks(path: Path) -> None:
    """Keep installer and dataset definitions outside the workflow block.

    Translation preserves the substitutions but may move a definition across
    a generated marker. A later workflow or release regeneration would then
    delete it. Gather each definition by identity and put it back inside the
    block that owns it before replacing either generated surface.
    """
    text = path.read_text(encoding="utf-8")
    wanted = set(INSTALLER_SUBSTITUTIONS + DATA_SUBSTITUTIONS)
    definitions: dict[str, str] = {}
    for match in _IMAGE_SUBSTITUTION_RE.finditer(text):
        name = match.group("name")
        if name not in wanted:
            continue
        if name in definitions:
            raise ValueError(f"{path} defines |{name}| more than once")
        definitions[name] = match.group(0).rstrip()
    missing = sorted(wanted - set(definitions))
    if missing:
        raise ValueError(f"{path} is missing linked image definitions: {missing}")

    text = _IMAGE_SUBSTITUTION_RE.sub(
        lambda match: "" if match.group("name") in wanted else match.group(0),
        text,
    )
    start = text.find(INSTALLER_BEGIN)
    end = text.find(INSTALLER_END, start + len(INSTALLER_BEGIN))
    if start < 0 or end <= start:
        raise ValueError(f"{path} is missing its installer-link markers")
    end += len(INSTALLER_END)
    installer_definitions = "\n".join(
        definitions[name] for name in INSTALLER_SUBSTITUTIONS
    )
    installer = (
        f"{INSTALLER_BEGIN}\n\n{INSTALLER_ROW}\n\n"
        f"{installer_definitions}\n\n{INSTALLER_END}"
    )
    text = text[:start] + installer + text[end:]

    rows = list(re.finditer(
        r"(?m)^\|DataBioStudies\|[^\n]*$\n*", text,
    ))
    if len(rows) != 1:
        raise ValueError(f"{path} must use the linked dataset row exactly once")
    data_definitions = "\n".join(
        definitions[name] for name in DATA_SUBSTITUTIONS
    )
    row = rows[0]
    text = (
        text[:row.start()] + DATA_ROW + "\n\n" + data_definitions
        + "\n\n" + text[row.end():]
    )
    path.write_text(text, encoding="utf-8")


def _remove_stale_app_assets(current_keys: set[str]) -> list[Path]:
    """Delete generated app tiles whose registry row no longer exists."""
    removed = []
    for directory in (APP_WORKFLOW_DIR, DOC_WORKFLOW_DIR / "apps"):
        for path in directory.glob("*.png"):
            if path.stem not in current_keys:
                path.unlink()
                removed.append(path)
    return removed


def _workflow_markup_for_readme(path: Path, icon_prefix: str) -> str:
    """Return canonical or reviewed localized markup for one README."""
    markup = _readme_workflow(icon_prefix)
    match = re.fullmatch(r"README\.(?P<language>[^.]+)\.rst", path.name)
    if match:
        return localize_workflow_markup(markup, match.group("language"))
    return markup


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
    # The API docs' own logo, written straight into _static so Sphinx
    # picks it up without a copy step that could put the bare white mark
    # back.
    docs_logo = (ROOT / "docs" / "source" / "_static"
                 / "logo_spacr_docs.png")
    docs_logo.parent.mkdir(parents=True, exist_ok=True)
    render_docs_logo().save(docs_logo, "PNG", optimize=True)
    print(target.relative_to(ROOT))
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    DOC_WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    APP_WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    doc_app_dir = DOC_WORKFLOW_DIR / "apps"
    doc_app_dir.mkdir(parents=True, exist_ok=True)
    # ONE LOOP OVER ONE ORDER. The pipeline tiles and the rest are drawn
    # by the same call now, so there is no second loop that could drift
    # from the first -- which is how they came to be different sizes.
    for key, label, image in _module_grid():
        target = WORKFLOW_DIR.parent / image
        render_module_tile(key, label).save(target, "PNG", optimize=True)
        (DOC_WORKFLOW_DIR.parent / image).write_bytes(target.read_bytes())
        print(target.relative_to(ROOT))
    app_keys = {key for key, _label, image in _module_grid()
                if image.startswith("workflow/apps/")}
    for stale in _remove_stale_app_assets(app_keys):
        print(f"removed {stale.relative_to(ROOT)}")
    # The pipeline arrow is gone with the strip it joined. Delete the
    # asset rather than leaving it: an unreferenced PNG in the resource
    # tree is the kind of thing a later change quietly starts using again.
    for stale_arrow in (WORKFLOW_DIR / "arrow.png",
                        DOC_WORKFLOW_DIR / "arrow.png"):
        if stale_arrow.is_file():
            stale_arrow.unlink()
            print(f"removed {stale_arrow.relative_to(ROOT)}")
    for readme in README_PATHS:
        _normalize_linked_resource_blocks(readme)
        prefix = (
            "../../../spacr/resources/icons"
            if readme.parent.name == "readme"
            else "spacr/resources/icons"
        )
        _replace_workflow_block(
            readme, _workflow_markup_for_readme(readme, prefix)
        )
        print(readme.relative_to(ROOT))
    DOC_WORKFLOW.parent.mkdir(parents=True, exist_ok=True)
    DOC_WORKFLOW.write_text(_documentation_workflow(), encoding="utf-8")
    print(DOC_WORKFLOW.relative_to(ROOT))
    DOC_FOLDS.write_text(_documentation_folds(), encoding="utf-8")
    HARDWARE_TABLE.write_text(_hardware_table(), encoding="utf-8")
    print(HARDWARE_TABLE.relative_to(ROOT))
    MODEL_ZOO_TABLE.write_text(_model_zoo_table() + "\n", encoding="utf-8")
    MODEL_ZOO_SECTIONS.write_text(_model_zoo_sections(),
                                  encoding="utf-8")
    print(MODEL_ZOO_TABLE.relative_to(ROOT))
    # The picker page before the pickers that point at it, so a fresh
    # checkout never has ten links to a file that is not there yet.
    LANGUAGE_PICKER_PAGE.parent.mkdir(parents=True, exist_ok=True)
    LANGUAGE_PICKER_PAGE.write_text(_language_picker_page(), encoding="utf-8")
    print(LANGUAGE_PICKER_PAGE.relative_to(ROOT))
    for readme_path in README_PATHS:
        if _write_the_hardware_table(readme_path):
            print(readme_path.relative_to(ROOT))
        if _write_the_model_zoo_table(readme_path):
            print(readme_path.relative_to(ROOT))
        if _write_the_language_picker(readme_path):
            print(readme_path.relative_to(ROOT))
    print(DOC_FOLDS.relative_to(ROOT))
    print(DOC_WORKFLOW.relative_to(ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual artwork regeneration
    raise SystemExit(main())
