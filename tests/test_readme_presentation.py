"""User-facing README and documentation typography contracts."""

from __future__ import annotations

import importlib.util
import re
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
    assert "Most screens follow six modules" in text
    assert "docs/source/features.rst" in text
    # Image substitutions carry accessibility text but are not visible prose.
    before_workflow, _, rest = text.partition(".. spacr-workflow-begin")
    _, _, after_workflow = rest.partition(".. spacr-workflow-end")
    assert len((before_workflow + after_workflow).split()) < 1800
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

    assert len(registry) == 58
    assert set(urls) == {key for key, _label, _description, _section in registry}
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
    assert arrow.size == (112, 112)
    assert arrow.getchannel("A").getbbox() is not None
    assert str(generator._tile_font(22).path).endswith("OpenSans-Regular.ttf")
    for path in APP_WORKFLOW_DIR.glob("*.png"):
        assert Image.open(path).size == (512, 512)

    workflow_row = next(
        line for line in text.splitlines()
        if line.startswith("|Workflow_mask|")
    )
    assert workflow_row.count("|Workflow_") == 11
    assert workflow_row.count("|Workflow_arrow|") == 5
    app_rows = [
        line for line in text.splitlines() if line.startswith("|App_")
    ]
    assert app_rows
    assert max(line.count("|App_") for line in app_rows) == 5
    # With normal inline-image spacing, both rows occupy the same width.
    top_width = (
        6 * generator.PIPELINE_DISPLAY_WIDTH
        + 5 * generator.ARROW_DISPLAY_WIDTH
        + 10 * 4
    )
    app_width = 5 * generator.APP_DISPLAY_WIDTH + 4 * 4
    assert abs(top_width - app_width) <= 1


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
