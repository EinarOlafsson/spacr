"""User-facing README and documentation typography contracts."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.rst"
DOCS_CONF = ROOT / "docs" / "source" / "conf.py"
DOCS_CSS = ROOT / "docs" / "source" / "_static" / "custom.css"
LOCALIZATION = ROOT / "docs" / "source" / "localization.rst"
SETTING_ANIMATIONS = ROOT / "docs" / "source" / "setting_animations.rst"
FEATURES = ROOT / "docs" / "source" / "features.rst"
INSTALLER_GUIDE = ROOT / "docs" / "source" / "installer_guide.rst"
DOCS_INDEX = ROOT / "docs" / "source" / "index.rst"
AUTOAPI_INDEX = ROOT / "docs" / "source" / "_autoapi_templates" / "index.rst"
DOC_WORKFLOW = ROOT / "docs" / "source" / "_generated" / "workflow_grid.rst"
WORKFLOW_DIR = ROOT / "spacr" / "resources" / "icons" / "workflow"
APP_WORKFLOW_DIR = WORKFLOW_DIR / "apps"
DOC_APP_WORKFLOW_DIR = (
    ROOT / "docs" / "source" / "_static" / "workflow" / "apps"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_uses_an_explicit_supported_python_badge():
    text = _read(README)
    assert "Python-3.9%E2%80%933.14" in text
    assert ":alt: Python 3.9 through 3.14" in text


def test_readme_keeps_the_feature_catalog_curated_and_points_to_detail():
    text = _read(README)
    features = _read(FEATURES)

    # RENAMED on 2026-09-01. "What you can do" said nothing the six
    # module names underneath it did not already say; "Core workflow" is
    # what the section actually is, and it now sits directly above the
    # module grid rather than above the installers.
    assert "Core workflow\n-------------" in text
    assert "What you can do" not in text
    assert "The primary workflow comprises six modules" in text
    assert "docs/source/features.rst" in text
    # Image substitutions carry accessibility text and targets but are not
    # visible prose. Remove their directive blocks, and every GENERATED
    # block, before enforcing the README's editorial ceiling.
    #
    # EVERY generated block, not just the workflow one. This ceiling
    # polices editorial sprawl -- prose an author chose to write -- and a
    # generated table's length is chosen by the data instead: the hardware
    # table grows when a GPU backend is added and the model zoo table grows
    # when a model is published. Counting them meant publishing a fourth
    # model would fail an editorial test, and the only way to pass it would
    # be to delete an explanation somewhere else in the file. That is the
    # same conflict the code-block carve-out below already records, so it
    # is resolved the same way: measure the thing the test was written to
    # measure.
    generated = (
        (".. spacr-workflow-begin", ".. spacr-workflow-end"),
        (".. spacr-hardware-begin", ".. spacr-hardware-end"),
        (".. spacr-model-zoo-begin", ".. spacr-model-zoo-end"),
    )
    kept = text
    for begin, end in generated:
        head, marker, rest = kept.partition(begin)
        assert marker, f"the README has lost its {begin} marker"
        _, marker, tail = rest.partition(end)
        assert marker, f"{begin} is not closed by {end}"
        kept = head + tail
    before_workflow, after_workflow = kept, ""
    visible_prose = re.sub(
        r"(?m)^\.\. \|[^|\n]+\| image::[^\n]*(?:\n   [^\n]*)*",
        "",
        before_workflow + after_workflow,
    )
    # Code blocks are NOT prose, and counting them here put this ceiling in
    # direct conflict with test_the_readme_documents_every_command, which
    # requires every console-script entry point to be named in the README.
    # The command reference that satisfies it is ~420 words of command
    # lines; measuring those against an editorial ceiling meant the README
    # could only document more commands by deleting explanation, which is
    # the opposite of what this test is for.
    #
    # So the ceiling now measures the thing it was written to police --
    # explanatory prose -- and the blocks get their own bound below, so
    # excluding them is not an escape hatch for sprawl.
    literal_block = (r"(?ms)^(?:\.\. code-block::[^\n]*\n(?:[ \t]*\n)*)?"
                     r"(?:^[ \t]{3,}[^\n]*\n|^[ \t]*\n(?=[ \t]{3,}\S))+")
    prose_only = re.sub(literal_block, "", visible_prose)
    assert len(prose_only.split()) < 1800, (
        f"README prose is {len(prose_only.split())} words; trim it or move "
        f"detail into docs/source/features.rst")
    blocks = len(visible_prose.split()) - len(prose_only.split())
    assert blocks < 600, (
        f"{blocks} words of code blocks -- the command reference is meant to "
        f"be a reference, not a manual; long examples belong in the docs")
    for heading in (
        "Core screen workflow",
        "Planning, quality control and exploration",
        "Reproducibility and interoperability",
        "Maturity labels",
    ):
        assert heading in features


def test_readme_uses_branch_safe_documentation_links():
    text = _read(README)
    for page in ("installer_guide", "python_api", "features"):
        assert f"einarolafsson.github.io/spacr/{page}.html" not in text
        assert f"docs/source/{page}.rst" in text


def test_every_workflow_button_tracks_the_home_screen_registry_and_api():
    from PIL import Image, ImageChops

    path = ROOT / "packaging" / "generate_readme_visuals.py"
    spec = importlib.util.spec_from_file_location("spacr_readme_visuals", path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    text = _read(README)
    docs = _read(DOC_WORKFLOW)
    registry = generator._registry()
    urls = generator._api_urls()
    pipeline = dict(generator.MAIN_PIPELINE)

    # Pinned deliberately: a new Home app needs a generated tile and an API
    # destination before this count advances.
    assert len(registry) == 44
    assert set(urls) == {key for key, _label, _description, _section in registry}
    # The generator no longer keeps its OWN copy of the section order. It
    # reads spacr.qt.app's SECTION_ORDER/SECTION_TILE_ORDER through
    # _home_layout(), which is the whole point -- the README figure and the
    # Home screen cannot disagree about what is in which section if there
    # is only one list. Ask the generator for the layout it will actually
    # draw rather than for module attributes that no longer exist.
    section_order, app_order = generator._home_layout()
    assert "Segmentation models" not in section_order
    # TILES are drawn for what Home offers as a place to START. Instruction
    # 318 folded 23 modules onto host mastheads: each KEEPS its registry
    # row, its screen and its API page, and loses only its tile. So the
    # figure is built from _tiled_registry(), while _registry() -- which is
    # what the API documents -- stays larger.
    #
    # Comparing the drawn layout against the full registry was the old
    # behaviour and is now wrong by 23 entries; it would demand a tile
    # image for every folded module, and none exists.
    tiled = generator._tiled_registry()
    assert len(tiled) < len(registry), (
        "if every registry row draws a tile then the fold strip has been "
        "lost; instruction 318 is what makes these two counts differ")
    assert tuple(
        key
        for section in section_order
        for key in app_order[section]
    ) == tuple(key for key, *_rest in tiled if key not in pipeline)

    for key, label, _description, _section in tiled:
        # ONE renderer for every tile since 2026-09-02. The two groups
        # still live in different folders -- the pipeline artwork is used
        # elsewhere too -- but nothing about the tiles differs.
        folder = "workflow" if key in pipeline else "workflow/apps"
        relative = f"spacr/resources/icons/{folder}/{key}.png"
        committed = Image.open(ROOT / relative).convert("RGBA")
        rendered = generator.render_module_tile(key, label).convert("RGBA")
        assert ImageChops.difference(committed, rendered).getbbox() is None
        assert relative in text
        assert urls[key] in text
        docs_relative = relative.replace(
            "spacr/resources/icons/workflow", "/_static/workflow"
        )
        assert docs_relative in docs
        assert urls[key] in docs

    # A folded module has no tile, but it must still be REACHABLE: the API
    # destination is the only thing left pointing at it, so losing that
    # would strand the module entirely.
    for key, _label, _description, _section in registry:
        assert key in urls, f"{key} has no API destination"
    assert not (APP_WORKFLOW_DIR / "report.png").exists(), (
        "a folded module has grown a tile image; either it was unfolded "
        "and belongs in the layout, or the asset is stale")

    # Assets exist for TILED apps only, for the same reason -- a folded
    # module draws nothing, so committing a PNG for it would be dead
    # weight that no page references.
    expected_app_assets = {
        f"{key}.png"
        for key, _label, _description, _section in tiled
        if key not in pipeline
    }
    assert {path.name for path in APP_WORKFLOW_DIR.glob("*.png")} == (
        expected_app_assets
    )
    assert {path.name for path in DOC_APP_WORKFLOW_DIR.glob("*.png")} == (
        expected_app_assets
    )

    assert "flow_chart_v3" not in text
    assert "The spaCR workflow" not in text
    assert "Select a tile to open that\nmodule's API page" in text
    assert "_generated/workflow_grid.rst" in _read(DOCS_INDEX)
    assert "../_generated/workflow_grid.rst" in _read(AUTOAPI_INDEX)


def test_every_module_is_one_tile_of_one_size_in_one_grid():
    """The README's module grid: identical tiles, six a row, no arrows.

    Asked for on 2026-09-02: "make them all the same size and present them
    on an evenly spaced grid with 6 modules per row", then "remove the
    titles and arrows for the modules and the title should be spaCR
    modules".

    The three asks are one contract and this test states it as one. The
    tiles could not be the same size while the pipeline strip carried
    arrows -- the strip had to fit six buttons and five arrows in the
    width the bands used for six buttons -- so "no arrows" is what makes
    "same size" reachable, and "same size" is what makes an even grid
    possible.
    """
    from PIL import Image, ImageChops

    path = ROOT / "packaging" / "generate_readme_visuals.py"
    spec = importlib.util.spec_from_file_location("spacr_readme_visuals", path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    text = _read(README)
    grid = generator._module_grid()

    # THE PIPELINE FIRST, then Home's order. The arrows used to say the
    # six were a sequence; the position now does.
    assert [key for key, _label, _image in grid][:len(generator.MAIN_PIPELINE)] == [
        key for key, _label in generator.MAIN_PIPELINE
    ]
    assert len({key for key, _l, _i in grid}) == len(grid), "a module tiled twice"

    # ONE RENDERER, so identical size is not a thing two paths have to be
    # kept in agreement about.
    for key, label, image in grid:
        committed = Image.open(ROOT / "spacr" / "resources" / "icons" / image)
        committed = committed.convert("RGBA")
        rendered = generator.render_module_tile(key, label).convert("RGBA")
        assert ImageChops.difference(committed, rendered).getbbox() is None, (
            f"{image} on disk is not what the generator draws")
        assert committed.size == (generator.BUTTON_SIZE, generator.BUTTON_SIZE)
        # Every tile occupies the same box in its canvas, so every tile
        # draws at the same size and every row anchors to the same left
        # edge whether it is full or short.
        bounds = committed.getchannel("A").getbbox()
        assert bounds is not None
        pad = generator.TILE_PADDING
        assert bounds[0] >= pad and bounds[1] >= pad
        assert bounds[2] <= committed.width - pad
        assert bounds[3] <= committed.height - pad
        assert image in text
        assert generator._api_urls()[key] in text

    assert str(generator._tile_font(22).path).endswith("OpenSans-Regular.ttf")

    # The grid's own markup, without the prose around it. Several of the
    # checks below have to be scoped to it: "**Tools**" is also ordinary
    # prose further down ("Make Masks appears under **Tools**"), and a
    # whole-file search would call that a heading.
    block = text.partition(generator.WORKFLOW_BEGIN)[2]
    block = block.partition(generator.WORKFLOW_END)[0]
    assert block.strip(), "the workflow markers bracket nothing"

    # THE ARROWS ARE GONE, asset and markup both. An unreferenced PNG left
    # in the resource tree is what a later change quietly starts using.
    assert not (WORKFLOW_DIR / "arrow.png").exists()
    assert not (ROOT / "docs" / "source" / "_static" / "workflow"
                / "arrow.png").exists()
    assert "arrow" not in block.lower()
    assert not hasattr(generator, "render_pipeline_arrow")

    # THE BAND TITLES ARE GONE. They restated Home's own grouping and went
    # stale every time Home was restructured.
    for gone in ("**Data**", "**Tools**", "**Assays**", "**More core tools**"):
        assert gone not in block, f"{gone} is still a heading over the grid"
    assert "spaCR modules\n-------------" in text

    # A LINE BLOCK, so every row starts "| ". Measured on the real GitHub
    # page on 2026-09-02: with each row as its own PARAGRAPH the gap between
    # rows was 2.5 to 3 times the gap between columns, because the horizontal
    # gutter is two tile canvases meeting and the vertical one was GitHub's
    # paragraph margin stacked on top of the same padding. A line block has
    # no paragraph margin, so both gutters become the same measurement.
    rows = [line for line in text.splitlines()
            if line.startswith("| |Module_")]
    assert rows, "the grid emitted no rows"
    assert not any(line.startswith("|Module_") for line in text.splitlines()), (
        "a grid row is a bare paragraph again; its bottom margin is what made "
        "the vertical gutter three times the horizontal one")
    assert sum(line.count("|Module_") for line in rows) == len(grid)
    # SIX PER ROW, and the last row is the only short one.
    assert all(line.count("|Module_") == generator.GRID_COLUMNS
               for line in rows[:-1])
    assert 1 <= rows[-1].count("|Module_") <= generator.GRID_COLUMNS
    # Zero-width separators: one fewer than the tiles they join, so the
    # browser puts no whitespace between neighbours and the gutter is
    # exactly the two canvas margins that meet.
    assert all(line.count(r"\ ") == line.count("|Module_") - 1
               for line in rows)

    # The row must not wrap. Six tiles at the declared width have to leave
    # headroom, because a browser that rounds each percentage up must
    # still not push the sixth onto a line of its own.
    row_width = generator.GRID_COLUMNS * generator.TILE_DISPLAY_PERCENT
    assert row_width < 100, "a full row wraps its last tile"

    # The gutter is made entirely by each tile's own padding, so it has to
    # be a real gap.
    visible_gap = generator.BUTTON_SIZE - generator.TILE_SIZE
    assert visible_gap == 2 * generator.TILE_PADDING
    assert visible_gap > 0

    # No leftover artwork for a module that is no longer tiled, and none
    # of the old two-sizes machinery still around to be picked back up.
    tiled = {image for _key, _label, image in grid}
    for stray in APP_WORKFLOW_DIR.glob("*.png"):
        assert f"workflow/apps/{stray.name}" in tiled, (
            f"{stray.name} is artwork nothing references")
    for gone in ("render_pipeline_tile", "render_app_tile", "_app_column",
                 "APP_COLUMNS", "APP_COLUMN_STEP", "APP_DISPLAY_PERCENT",
                 "PIPELINE_DISPLAY_PERCENT", "ARROW_DISPLAY_PERCENT",
                 "APP_TILE_SIZE", "APP_TILE_PADDING"):
        assert not hasattr(generator, gone), (
            f"{gone} survived the single-grid rewrite; two sizes can come "
            f"back the moment there are two ways to ask for one")


def test_the_grid_order_does_not_depend_on_qt_import_order():
    """Late screen registration cannot reorder the committed grid.

    The grid used to be checked by asking each tile which COLUMN it was
    drawn at, because the column decided where the button sat inside its
    canvas. Every tile now draws at one offset, so the column is not a
    property of the artwork any more -- the ORDER is the thing import
    order could still disturb, and it is what is pinned here.
    """
    script = r"""
import importlib.util
import json
import os

preimport = os.environ.get("SPACR_README_PREIMPORT")
if preimport:
    __import__(preimport)
spec = importlib.util.spec_from_file_location(
    "spacr_readme_visuals", "packaging/generate_readme_visuals.py"
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
order = [key for key, _label, _image in generator._module_grid()]
print("SPACR_COLUMNS=" + json.dumps(order))
"""

    def grid_order(preimport: str) -> "list[str]":
        env = dict(os.environ)
        env["SPACR_README_PREIMPORT"] = preimport
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        line = next(
            item for item in result.stdout.splitlines()
            if item.startswith("SPACR_COLUMNS=")
        )
        return json.loads(line.partition("=")[2])

    clean = grid_order("")
    settings_first = grid_order("spacr.qt.screens.settings_model")
    assert clean == settings_first


def test_the_readme_lists_every_published_model_with_its_limits():
    """The model zoo section is generated from the catalogue, not typed.

    Asked for on 2026-09-02: "add the uploaded modules in the current model
    zoo to readme in a model zoo section".

    GENERATED, because a hand-written table is a second copy of the
    catalogue and a second copy goes stale -- which is the fault the module
    grid above spent a day having removed. Publishing a fourth model must
    make this test fail until the generator is re-run.
    """
    import importlib.util

    path = ROOT / "packaging" / "generate_readme_visuals.py"
    spec = importlib.util.spec_from_file_location("spacr_readme_visuals", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    text = _read(README)
    block = text.partition(generator.MODEL_ZOO_BEGIN)[2]
    block = block.partition(generator.MODEL_ZOO_END)[0]
    assert block.strip(), "the README has no model zoo block"
    assert block.strip() == generator._model_zoo_table().strip(), (
        "the README's model zoo table is not what the catalogue produces; "
        "re-run packaging/generate_readme_visuals.py")

    rows = generator._model_zoo_rows()
    assert rows, "the catalogue published nothing"
    for key, name, trained_on, limits in rows:
        assert f"``{key}``" in block, f"{key} is published but not in the README"
        assert name in block
        # THE LIMITS TOO, not only the headline number. Every entry's notes
        # carry the caveat that says what the model is not for -- "accuracy
        # falls sharply above IoU 0.8", "locates WELLS, not plaques" -- and
        # a table that prints only the good number is the sort of claim
        # instruction 316 exists to prevent.
        assert limits, f"{key} publishes no notes at all"
        assert limits in block, f"{key}'s stated limits are missing"
        assert trained_on in block, f"{key} does not say what it was trained on"

    # The table must not depend on the machine that generated it: a
    # developer with a local checkpoint installed, a plugin, or the
    # catalogue environment variable set must all produce the same README.
    import os

    from spacr.model_zoo import CATALOGUE_ENV_VAR

    os.environ[CATALOGUE_ENV_VAR] = "/nonexistent/catalogue.json"
    try:
        assert generator._model_zoo_rows() == rows, (
            "the generated table changed with the environment; the README "
            "would differ depending on who ran the generator")
    finally:
        os.environ.pop(CATALOGUE_ENV_VAR, None)


def test_installer_guide_is_distinct_from_the_version_archive():
    readme = _read(README)
    guide = _read(INSTALLER_GUIDE)
    index = _read(DOCS_INDEX)

    assert "docs/source/installer_guide.rst" in readme
    assert ".. _installer-guide:" in guide
    for heading in (
        "Desktop installers",
        "Updating",
        "Uninstalling",
        "Offline installation",
        "Troubleshooting",
    ):
        assert heading in guide
    assert ":target: docs/source/installers.rst" in readme
    assert "   installer_guide" in index
    assert "   installers" in index


def test_installer_guide_gives_old_desktop_builds_the_pipless_escape():
    """The broken updater cannot be the only route to its own repair."""
    guide = _read(INSTALLER_GUIDE)

    assert "No module named pip" in guide
    assert "Windows 1.5.0.4" in guide
    assert "administrator access nor a reinstall" in guide
    assert (
        "~/.local/share/spacr/bootstrap/uv pip install --upgrade --python "
        "~/.local/share/spacr/venv/bin/python spacr"
    ) in guide
    assert (
        '"$HOME/Library/Application Support/SpaCR/bootstrap/uv" pip install '
        '--upgrade --python "$HOME/Library/Application Support/SpaCR/venv/'
        'bin/python" spacr'
    ) in guide
    assert (
        '"$env:LOCALAPPDATA\\SpaCR\\bootstrap\\uv.exe" pip install '
        '--upgrade --python "$env:LOCALAPPDATA\\SpaCR\\venv\\Scripts\\'
        'python.exe" spacr'
    ) in guide


def test_reference_resources_are_linked_rounded_buttons():
    from PIL import Image

    text = _read(README)
    for name in ("biostudies", "huggingface", "ncbi", "spacrpower", "biorxiv"):
        relative = f"spacr/resources/icons/databanks/{name}_button.png"
        assert relative in text
        image = Image.open(ROOT / relative).convert("RGBA")
        assert image.size == (512, 512)
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((256, 0)) == (43, 47, 58, 255)

    for old_text_link in (
        "Full microscopy dataset:",
        "Testing dataset:",
        "Sequencing data:",
        "Power analysis: spaCRPower",
    ):
        assert old_text_link not in text


def test_readme_contains_only_user_facing_installation_copy():
    text = _read(README)
    for creator_note in (
        "one-time reviewed onboarding",
        "one-time maintainer procedure",
        "rewritten automatically whenever",
        "These links are rewritten automatically",
        "Project data model",
        "A typical project contains:",
    ):
        assert creator_note not in text


def test_language_support_is_a_documented_output_safe_feature():
    readme = _read(README)
    guide = _read(LOCALIZATION)
    index = _read(DOCS_INDEX)

    assert "Language & translation" in readme
    assert "The interface supports ten languages" in readme
    assert "localization.rst#contextual-help" in readme
    assert re.search(r"AI and\s+LIVE", readme)
    assert re.search(r"scientific\s+output remains canonical English", readme)

    assert guide.startswith("Language & translation\n")
    assert "Language <localization>" in index
    assert "What is translated" in guide
    assert "Contextual help" in guide
    assert "Raw worker stdout, logs, tracebacks" in guide
    assert "User chat messages" in guide
    assert "append_notice" in guide


def test_setting_animations_are_wired_into_readme_and_docs():
    readme = _read(README)
    gallery = _read(SETTING_ANIMATIONS)

    assert "Animated setting guidance" in readme
    assert "Setting animation registry" in readme
    assert "setting_animations.html" in readme
    assert "Setting animation gallery" in gallery
    assert gallery.count(".. _setting-animation-") == 94
    assert gallery.count(".. image:: ../../spacr/resources/") == 94
    assert ":mod:`spacr.setting_animations`" in gallery


def test_documentation_uses_bundled_open_sans_at_all_weights():
    css = _read(DOCS_CSS)
    conf = _read(DOCS_CONF)

    for filename in (
        "OpenSans-Light.ttf",
        "OpenSans-Regular.ttf",
        "OpenSans-SemiBold.ttf",
    ):
        assert filename in css
        assert (DOCS_CSS.parent / "fonts" / filename).is_file()
    assert css.count('font-family: "Open Sans"') == 3
    assert '--font-stack: "Open Sans"' in css
    assert conf.count('"Open Sans", ui-sans-serif') == 2
