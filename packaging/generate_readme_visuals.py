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
DOC_WORKFLOW_DIR = ROOT / "docs" / "source" / "_static" / "workflow"
README_PATHS = (
    ROOT / "README.rst",
    *(ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"),
)
WORKFLOW_BEGIN = ".. spacr-workflow-begin"
WORKFLOW_END = ".. spacr-workflow-end"
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
# Percentage widths keep the linked images responsive on both GitHub and the
# Sphinx API. The rows leave a little rounding headroom so the browser never
# moves the last tile onto a new line.
PIPELINE_DISPLAY_PERCENT = 14.5
ARROW_DISPLAY_PERCENT = 2.5
# SIX canvases span the same 99.5% width as the core row, so the largest
# section fits on one line: Home's bands are 6, 6, 5 and 4 since the
# 2026-08-31 restructure, and a band that wraps reads as two groups.
APP_COLUMNS = 6
# Rounded DOWN to three places. The exact quotient is 16.5833...%, and
# emitting that repeating tail into every one of ~150 image directives
# across ten READMEs is noise; rounding up could total more than 100% and
# wrap the last tile, which is the one thing this width exists to prevent.
APP_DISPLAY_PERCENT = round(
    (6 * PIPELINE_DISPLAY_PERCENT + 5 * ARROW_DISPLAY_PERCENT)
    / APP_COLUMNS - 0.0005, 3)
APP_TILE_PADDING = 16
APP_TILE_SIZE = BUTTON_SIZE - 2 * APP_TILE_PADDING
# EVERY TILE DRAWN AT THE SAME OFFSET IN ITS CANVAS -- asked for as "make
# all buttons the same size and aligned to the left".
#
# This used to distribute each tile's transparent gutter across the row,
# left to right, so a FULL row met both edges of the core row above it
# with one constant gap. The cost was that a button's position depended
# on which column it landed in, so the same module moved within its
# canvas when the row above it changed length -- and with bands of 6, 6,
# 5 and 4, three of the four rows are short. A constant offset draws
# every button identically and anchors every row, full or not, to the
# left edge.
APP_COLUMN_STEP = 0
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
    """Render a smaller tile positioned for a five-column linked row."""
    tile = _render_workflow_tile(key, label)
    tile = tile.resize((APP_TILE_SIZE, APP_TILE_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    canvas.alpha_composite(
        tile,
        (_app_column(key) * APP_COLUMN_STEP, APP_TILE_PADDING),
    )
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
    """Return the home-screen API destinations used by the running app."""
    from spacr.qt.screens.settings_model import _APP_API_MODULE

    base = "https://einarolafsson.github.io/spacr/api/spacr"
    return {
        key: f"{base}/{_APP_API_MODULE[key].strip('/')}/index.html"
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

    A machine WITH a GPU that cannot accelerate a task is RED -- that is
    what "not supported" means here, and UMAP on Metal is the case: cuML
    ships for CUDA only, so there is nothing to wait for.

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
        f"\u2003 {RED} not supported",
        "",
        "Every cell is generated from ``spacr.accelerator.capabilities()``",
        "with that backend's probe faked, so this table, the first setup",
        "screen and ``spacr-doctor`` cannot disagree.",
        "",
        "**No GPU is supported, not broken.** Every task runs on a CPU and",
        "every result is identical; only the wall clock changes. On the",
        "machine these were measured on, one 256x256 Cellpose image took",
        "444.5 s on the CPU and 3.2 s on its Radeon.",
        "",
        "*Beta* means implemented and dispatched to, but exercised on one",
        "machine or none. CUDA is the only configuration with years behind",
        "it.",
        "",
    ]
    return "\n".join(lines)


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

    Walked from the same table the application walks --
    `map_barcodes.FOLD_HOST_MODULES` plus each host module's
    `FOLDED_APPS` -- so the documentation cannot claim a fold the GUI does
    not install, or miss one it does.
    """
    from importlib import import_module

    from spacr.qt.screens.map_barcodes import FOLD_HOST_MODULES

    hosts = {}
    for host_key, module_name in FOLD_HOST_MODULES.items():
        module = import_module(f"spacr.qt.screens.{module_name}")
        for key in getattr(module, "FOLDED_APPS", ()):
            hosts[key] = host_key
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

    def _link(key: str) -> str:
        label = names.get(key) or fold_description(key)[0]
        if not label:
            label = key.replace("_", " ").title()
        target = urls.get(key)
        return f"`{label} <{target}>`_" if target else label

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


def _app_column(key: str) -> int:
    """Return the zero-based column occupied by an application tile."""
    for items in _grouped_apps().values():
        for index, (candidate, _label) in enumerate(items):
            if candidate == key:
                return index % APP_COLUMNS
    raise KeyError(f"unknown non-pipeline application: {key}")


def _readme_workflow(
    icon_prefix: str,
    *,
    alt_template: str = "Open the {module} API",
) -> str:
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
            f"   :alt: {alt_template.format(module=label)}",
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
    for section in _home_layout()[0]:
        items = grouped[section]
        if not items:
            continue
        title = "More core tools" if section == "Core" else section
        lines.extend([f"**{title}**", ""])
        for start in range(0, len(items), APP_COLUMNS):
            row = items[start:start + APP_COLUMNS]
            lines.extend([
                _inline_image_row([f"|App_{key}|" for key, _ in row]),
                "",
            ])
        for key, label in items:
            definitions.extend([
                f".. |App_{key}| image:: {icon_prefix}/workflow/apps/{key}.png",
                f"   :width: {APP_DISPLAY_WIDTH}",
                f"   :alt: {alt_template.format(module=label)}",
                f"   :target: {urls[key]}",
                # MIDDLE, and it MUST be. These are SUBSTITUTION
                # definitions used inline in a paragraph, and docutils
                # accepts only top/middle/bottom there -- "left" is a
                # block-image value and raises "not a valid value for the
                # align option within a substitution definition".
                #
                # The failure mode is why this comment is long: the
                # directive errors, the substitution is never defined,
                # and GitHub renders the reference as its alt text. The
                # whole grid turns into a column of blue links, which is
                # what happened when this was set to "left" on
                # 2026-08-31. Left-alignment comes from the rows being
                # left-anchored paragraphs and every tile sharing one
                # offset -- not from this option.
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
    for section in _home_layout()[0]:
        items = grouped[section]
        if not items:
            continue
        title = "More core tools" if section == "Core" else section
        lines.extend([
            title,
            "^" * len(title),
            "",
        ])
        for start in range(0, len(items), APP_COLUMNS):
            row = items[start:start + APP_COLUMNS]
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
    # TILED rows. Rendering a tile image for a folded module would write a
    # PNG nothing references, and `_app_column` refuses the key outright --
    # which is how this was caught rather than shipped as dead artwork.
    app_rows = [
        row for row in _tiled_registry() if row[0] not in pipeline_keys
    ]
    for key, label, _description, _section in app_rows:
        image = render_app_tile(key, label)
        target = APP_WORKFLOW_DIR / f"{key}.png"
        image.save(target, "PNG", optimize=True)
        image.save(doc_app_dir / f"{key}.png", "PNG", optimize=True)
        print(target.relative_to(ROOT))
    for stale in _remove_stale_app_assets({row[0] for row in app_rows}):
        print(f"removed {stale.relative_to(ROOT)}")
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
    for readme_path in README_PATHS:
        if _write_the_hardware_table(readme_path):
            print(readme_path.relative_to(ROOT))
    print(DOC_FOLDS.relative_to(ROOT))
    print(DOC_WORKFLOW.relative_to(ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual artwork regeneration
    raise SystemExit(main())
