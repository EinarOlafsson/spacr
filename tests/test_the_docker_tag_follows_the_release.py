"""The README's Docker commands name the image of the current release.

Asked for on 2026-10-10: "where is the docker version of spacr? it should be
on the readme page for spacr". The Docker section sat below PyPI and
conda-forge and still named 1.5.1.0 three releases later, because nothing
moved its tag. ``packaging/release.py`` now rewrites the tag on every page
that names one, both on ``bump`` and on ``sync-published``; these tests hold
the pages to setup.py and the helper to its contract.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

from tests.readme_source_words import readme_words

ROOT = Path(__file__).resolve().parents[1]


def _release_module():
    spec = importlib.util.spec_from_file_location(
        "spacr_release_helper_docker", ROOT / "packaging" / "release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_docker_page_names_the_setup_py_version():
    release = _release_module()
    version = release.read_version(ROOT / "setup.py")
    for page in release.docker_tag_pages(ROOT):
        shown = release.docker_tag_versions(page.read_text(encoding="utf-8"))
        assert shown == {version}, (
            f"{page.relative_to(ROOT)} names {sorted(shown)}, setup.py "
            f"declares {version}; run packaging/release.py bump or "
            f"sync-published.")


def test_the_readme_offers_docker_beside_the_installers_and_pip():
    readme = readme_words()
    assert "|Docker|" in readme.splitlines()[0]
    install = readme.index("Install spaCR\n~~~~")
    desktop = readme.index("Desktop application\n---", install)
    routes = readme[install:desktop]
    for route in ("**Desktop application**", "**pip**", "**conda-forge**",
                  "**Docker**"):
        assert route in routes, route
    assert re.search(r"docker run --rm ghcr\.io/einarolafsson/spacr:"
                     r"[\d.]+ spacr-run --list", routes)
    assert re.search(r"docker run --rm --gpus all ghcr\.io/einarolafsson/"
                     r"spacr:[\d.]+-cuda12\.4 spacr-run --list", routes)


def test_translated_prose_names_no_image_version():
    """Tags live in code blocks, so a bump never stales a reviewed record."""
    from_tools = ROOT / "tools"
    import sys
    sys.path.insert(0, str(from_tools))
    try:
        from build_documentation_i18n import translatable_blocks
    finally:
        sys.path.remove(str(from_tools))
    blocks, _layout = translatable_blocks(readme_words())
    assert not [block for block in blocks
                if re.search(r"spacr:\d", block)]


def _tree(tmp_path: Path) -> Path:
    release = _release_module()
    for page in (Path("setup.py"), Path("CITATION.cff"),
                 Path("spacr") / "_version.py",
                 *(path.relative_to(ROOT)
                   for path in release.docker_tag_pages(ROOT))):
        target = tmp_path / page
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / page, target)
    return tmp_path


def test_bump_moves_every_docker_tag_and_keeps_the_suffixes(tmp_path):
    release = _release_module()
    root = _tree(tmp_path)
    current = release.read_version(root / "setup.py")
    major, minor, patch, build = (current.split(".") + ["0"])[:4]
    new = f"{major}.{minor}.{patch}.{int(build) + 1}"
    guide = root / "docs" / "source" / "installer_guide.rst"
    cuda_before = guide.read_text(encoding="utf-8").count(
        f"spacr:{current}-cuda12.4")

    assert release.bump_release(
        root / "setup.py", root / "CITATION.cff", new,
        package_version_path=root / "spacr" / "_version.py",
        release_date="2026-10-10") == new

    for page in release.docker_tag_pages(root):
        assert release.docker_tag_versions(
            page.read_text(encoding="utf-8")) == {new}, page
    assert cuda_before and guide.read_text(encoding="utf-8").count(
        f"spacr:{new}-cuda12.4") == cuda_before


def test_a_page_that_lost_its_tag_stops_the_sync_before_any_write(tmp_path):
    release = _release_module()
    root = _tree(tmp_path)
    readme = root / "README.rst"
    before = readme.read_text(encoding="utf-8")
    apptainer = root / "packaging" / "apptainer" / "README.md"
    apptainer.write_text("no image named here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="names no"):
        release.sync_docker_tags(root, "9.9.9.9")
    assert readme.read_text(encoding="utf-8") == before
