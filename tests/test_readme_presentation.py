"""User-facing README and documentation typography contracts."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.rst"
DOCS_CONF = ROOT / "docs" / "source" / "conf.py"
DOCS_CSS = ROOT / "docs" / "source" / "_static" / "custom.css"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_uses_an_explicit_supported_python_badge():
    text = _read(README)
    assert "Python-3.9%E2%80%933.14" in text
    assert ":alt: Python 3.9 through 3.14" in text


def test_feature_catalog_is_one_equal_width_curated_table():
    text = _read(README)
    features = text[text.index("Features\n--------"):text.index("\nData\n----")]

    assert features.count(".. list-table::") == 1
    assert ":widths: 25 25 25 25" in features
    for heading in (
        "**Image analysis**",
        "**AI and phenotyping**",
        "**Sequencing and screen analysis**",
    ):
        assert heading in features

    rows = re.findall(r"^   \* - ", features, flags=re.MULTILINE)
    # One column-header row, three category rows, and 24 feature rows.
    assert len(rows) == 28


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
