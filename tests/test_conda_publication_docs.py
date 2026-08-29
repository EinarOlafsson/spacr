"""The published conda package is linked and described as an install route.

The repository used to expose an unused badge for its staging recipe.  A
recipe link looks plausible after publication but sends users to source files
instead of the package they can install.  These contracts keep the live
package, the direct Conda command and the distinct PyPI-in-a-Conda-environment
route aligned across the English and localized GitHub pages.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
README = ROOT / "README.rst"
INSTALLER_GUIDE = ROOT / "docs" / "source" / "installer_guide.rst"
LOCALIZED = sorted((ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"))
ALL_READMES = [README, *LOCALIZED]

PACKAGE_URL = "https://anaconda.org/conda-forge/spacr"
BADGE_URL = f"{PACKAGE_URL}/badges/version.svg"
CONDA_COMMAND = "conda install conda-forge::spacr"
PIP_COMMAND = 'python -m pip install "spacr[qt]"'

LOCALIZED_BADGE_ALT = {
    "de": "conda-forge-Version",
    "es": "Versión en conda-forge",
    "fr": "Version conda-forge",
    "hi": "conda-forge संस्करण",
    "is": "conda-forge-útgáfa",
    "ko": "conda-forge 버전",
    "pt": "Versão no conda-forge",
    "sv": "conda-forge-version",
    "zh_CN": "conda-forge 版本",
}


@pytest.mark.parametrize("path", ALL_READMES, ids=lambda path: path.name)
def test_live_conda_badge_is_visible_and_targets_the_package(path: Path):
    text = path.read_text(encoding="utf-8")
    first_line = text.splitlines()[0]

    assert "|Conda|" in first_line
    assert f".. |Conda| image:: {BADGE_URL}" in text
    assert f"   :target: {PACKAGE_URL}" in text
    assert "CondaRecipe" not in text
    assert "conda--forge-recipe" not in text
    assert "/tree/main/conda-forge/recipe" not in text


@pytest.mark.parametrize("path", ALL_READMES, ids=lambda path: path.name)
def test_conda_and_pypi_are_separate_install_routes(path: Path):
    text = path.read_text(encoding="utf-8")

    assert text.count(CONDA_COMMAND) == 1
    assert text.count(PIP_COMMAND) == 1
    assert text.index(CONDA_COMMAND) < text.index(PIP_COMMAND)


@pytest.mark.parametrize("path", LOCALIZED, ids=lambda path: path.name)
def test_conda_badge_alt_text_is_localized(path: Path):
    language = path.name.removeprefix("README.").removesuffix(".rst")
    text_after = path.read_text(encoding="utf-8").partition(
        f".. |Conda| image:: {BADGE_URL}"
    )[2]
    assert text_after
    assert (
        f"   :alt: {LOCALIZED_BADGE_ALT[language]}"
        in text_after.partition("\n\n")[0]
    )


def test_english_readme_says_which_installer_owns_each_route():
    text = README.read_text(encoding="utf-8")

    assert "official conda-forge package installs spaCR" in text
    assert "For the PyPI release, install spaCR with pip inside a Conda" in text


def test_installer_guide_covers_install_update_and_removal_by_source():
    text = INSTALLER_GUIDE.read_text(encoding="utf-8")

    for value in (
        PACKAGE_URL,
        CONDA_COMMAND,
        "conda update conda-forge::spacr",
        "conda install conda-forge::spacr=1.5.0.4",
        "conda remove spacr",
        PIP_COMMAND,
        "python -m pip uninstall spacr",
    ):
        assert value in text
    assert "**conda-forge:**" in text
    assert "**PyPI:**" in text
    assert "**pip or conda:**" not in text


def test_new_readme_prose_has_source_bound_review_in_every_locale():
    from build_documentation_i18n import REVIEWED_README_EVIDENCE_BLOCKS

    sources = (
        "The official conda-forge package installs spaCR and its desktop "
        "dependencies into the active environment:",
        "For the PyPI release, install spaCR with pip inside a Conda "
        "environment. Python 3.12 has the widest choice of optional "
        "scientific packages:",
    )
    assert set(LOCALIZED_BADGE_ALT) == {
        "de", "es", "fr", "hi", "is", "ko", "pt", "sv", "zh_CN"
    }
    for source in sources:
        translations = REVIEWED_README_EVIDENCE_BLOCKS[source]
        assert set(translations) == set(LOCALIZED_BADGE_ALT)
        for language, translation in translations.items():
            localized = (
                ROOT / "docs" / "i18n" / "readme"
                / f"README.{language}.rst"
            ).read_text(encoding="utf-8")
            assert translation in localized
