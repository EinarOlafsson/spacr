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

    assert "What you can do\n---------------" in text
    assert "The primary workflow comprises six modules" in text
    assert "docs/source/features.rst" in text
    # Image substitutions carry accessibility text and targets but are not
    # visible prose. Remove their directive blocks as well as the generated
    # workflow before enforcing the README's editorial ceiling.
    before_workflow, _, rest = text.partition(".. spacr-workflow-begin")
    _, _, after_workflow = rest.partition(".. spacr-workflow-end")
    visible_prose = re.sub(
        r"(?m)^\.\. \|[^|\n]+\| image::[^\n]*(?:\n   [^\n]*)*",
        "",
        before_workflow + after_workflow,
    )
    assert len(visible_prose.split()) < 1800
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
    assert "Segmentation models" not in generator.SECTION_ORDER
    assert tuple(
        key
        for section in generator.SECTION_ORDER
        for key in generator.APP_ORDER[section]
    ) == tuple(key for key, *_rest in registry if key not in pipeline)
    for key, label, _description, _section in registry:
        if key in pipeline:
            relative = f"spacr/resources/icons/workflow/{key}.png"
            committed = Image.open(WORKFLOW_DIR / f"{key}.png").convert("RGBA")
            rendered = generator.render_pipeline_tile(key, label).convert("RGBA")
        else:
            relative = f"spacr/resources/icons/workflow/apps/{key}.png"
            committed = Image.open(APP_WORKFLOW_DIR / f"{key}.png").convert("RGBA")
            rendered = generator.render_app_tile(key, label).convert("RGBA")
        assert ImageChops.difference(committed, rendered).getbbox() is None
        assert relative in text
        assert urls[key] in text
        docs_relative = relative.replace(
            "spacr/resources/icons/workflow", "/_static/workflow"
        )
        assert docs_relative in docs
        assert urls[key] in docs

    expected_app_assets = {
        f"{key}.png"
        for key, _label, _description, _section in registry
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
    assert "Select a workflow module to open its API page" in text
    assert "_generated/workflow_grid.rst" in _read(DOCS_INDEX)
    assert "../_generated/workflow_grid.rst" in _read(AUTOAPI_INDEX)


def test_workflow_modules_are_dark_linked_tiles_with_separate_white_arrows():
    from PIL import Image, ImageChops

    path = ROOT / "packaging" / "generate_readme_visuals.py"
    spec = importlib.util.spec_from_file_location("spacr_readme_visuals", path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    text = _read(README)
    for key, label in generator.MAIN_PIPELINE:
        committed = Image.open(WORKFLOW_DIR / f"{key}.png").convert("RGBA")
        rendered = generator.render_pipeline_tile(key, label).convert("RGBA")
        assert ImageChops.difference(committed, rendered).getbbox() is None
        assert committed.getpixel(
            (committed.width // 2, 3)
        ) == generator.WORKFLOW_RIM
        assert f"workflow/{key}.png" in text
        assert generator._api_urls()[key] in text

    arrow = Image.open(WORKFLOW_DIR / "arrow.png").convert("RGBA")
    assert ImageChops.difference(
        arrow, generator.render_pipeline_arrow().convert("RGBA")
    ).getbbox() is None
    assert arrow.getpixel(
        (arrow.width // 2, arrow.height // 2)
    ) == generator.WHITE
    assert arrow.size == (
        generator.ARROW_CANVAS_WIDTH,
        generator.ARROW_CANVAS_HEIGHT,
    )
    assert arrow.getchannel("A").getbbox() is not None
    assert str(generator._tile_font(22).path).endswith("OpenSans-Regular.ttf")
    for path in APP_WORKFLOW_DIR.glob("*.png"):
        app = Image.open(path).convert("RGBA")
        assert app.size == (512, 512)
        bounds = app.getchannel("A").getbbox()
        assert bounds is not None
        key = path.stem
        left = generator._app_column(key) * generator.APP_COLUMN_STEP
        assert bounds[0] >= left
        assert bounds[1] >= generator.APP_TILE_PADDING
        assert bounds[2] <= left + generator.APP_TILE_SIZE
        assert bounds[3] <= app.height - generator.APP_TILE_PADDING

    workflow_row = next(
        line for line in text.splitlines()
        if line.startswith("|Workflow_mask|")
    )
    assert workflow_row.count("|Workflow_") == 11
    assert workflow_row.count("|Workflow_arrow|") == 5
    assert workflow_row.count(r"\ ") == 10
    app_rows = [
        line for line in text.splitlines() if line.startswith("|App_")
    ]
    assert app_rows
    assert max(line.count("|App_") for line in app_rows) == generator.APP_COLUMNS
    assert all(
        line.count(r"\ ") == line.count("|App_") - 1
        for line in app_rows
    )
    # Percent widths and zero-width RST separators keep the declared number
    # of tiles on each row at every normal documentation viewport width. Five
    # secondary canvases meet both core-row edges; the visible buttons remain
    # smaller and have one constant gap. Partial rows start at the left edge.
    top_width = (
        6 * generator.PIPELINE_DISPLAY_PERCENT
        + 5 * generator.ARROW_DISPLAY_PERCENT
    )
    app_width = generator.APP_COLUMNS * generator.APP_DISPLAY_PERCENT
    assert top_width < 100
    assert app_width == top_width
    assert generator.APP_COLUMN_STEP * (generator.APP_COLUMNS - 1) == (
        2 * generator.APP_TILE_PADDING
    )
    visible_gap = (
        generator.BUTTON_SIZE
        + generator.APP_COLUMN_STEP
        - generator.APP_TILE_SIZE
    )
    assert visible_gap > 0
    first_left = 0
    last_right = (
        (generator.APP_COLUMNS - 1) * generator.BUTTON_SIZE
        + (generator.APP_COLUMNS - 1) * generator.APP_COLUMN_STEP
        + generator.APP_TILE_SIZE
    )
    assert first_left == 0
    assert last_right == generator.APP_COLUMNS * generator.BUTTON_SIZE
    for items in generator._grouped_apps().values():
        for start in range(0, len(items), generator.APP_COLUMNS):
            assert generator._app_column(items[start][0]) == 0
    assert (
        generator.ARROW_CANVAS_WIDTH / generator.ARROW_CANVAS_HEIGHT
        == generator.ARROW_DISPLAY_PERCENT
        / generator.PIPELINE_DISPLAY_PERCENT
    )


def test_workflow_asset_columns_do_not_depend_on_qt_import_order():
    """Late screen registration cannot move committed README artwork."""
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
columns = {
    key: generator._app_column(key)
    for items in generator._grouped_apps().values()
    for key, _label in items
}
print("SPACR_COLUMNS=" + json.dumps(columns, sort_keys=True))
"""

    def columns(preimport: str) -> dict[str, int]:
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

    clean = columns("")
    settings_first = columns("spacr.qt.screens.settings_model")
    assert clean == settings_first


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
