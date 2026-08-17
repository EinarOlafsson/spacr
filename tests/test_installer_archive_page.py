"""Instruction 123: every installer version stays reachable, on one page.

Nothing is being deleted -- ``release.yml`` uploads with ``--clobber``, which
only overwrites assets *within one tag*, and every release still carries its
assets. What disappears is the LINK: ``packaging/release.py collect`` rewrites
the README's three installer URLs to the version being released, so the moment
a new version ships there is no path anywhere in the project to the previous
one. The archive page is that path.

These tests are about the two ways such a page rots:

* it stops being generated, and a release adds no row -- so the generator is
  driven against a fabricated release list, which is what a release looks like
  before it happens;
* it links to files that are not there -- so the committed page's links are
  checked against the live releases, in the one test here that needs network.

The version-pinning claim the page makes is checked too. An online installer
downloads the package at install time; if it resolved "the latest spacr"
rather than the version it was built for, an archived 1.4.9.9 installer would
hand over the current release while displaying the old version number, and an
archive whose entire purpose is getting an old version back would be lying.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "source" / "installers.rst"
README = ROOT / "README.rst"


def _release_helper():
    """Import ``packaging/release.py`` by path.

    ``packaging`` is also the name of an installed distribution that
    ``release.py`` itself imports (``packaging.version``), so it cannot be
    imported as a package from here without shadowing it.
    """
    spec = importlib.util.spec_from_file_location(
        "spacr_release_helper", ROOT / "packaging" / "release.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: The published assets keep the spelling they were published under, which is
#: why ``tests/test_the_name_is_spaCR.py`` exempts ``releases/download/``
#: lines -- and why this template is one physical line.
ASSET_URL = "https://github.com/EinarOlafsson/spacr/releases/download/v{version}/SpaCR-{version}-{suffix}"  # noqa: E501


def _release(tag, *suffixes):
    version = tag.lstrip("v")
    urls = [ASSET_URL.format(version=version, suffix=suffix)
            for suffix in suffixes]
    return {
        "tag_name": tag,
        "assets": [{"name": url.rsplit("/", 1)[-1],
                    "browser_download_url": url} for url in urls],
    }


ALL_THREE = ("Linux-x86_64-Online.run", "macOS-Universal-Online.pkg",
             "Windows-Online-Setup.exe")


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------

def test_a_new_release_adds_a_row_without_anyone_editing_the_page():
    """The failure that produced this request was a hand-maintained list.

    A table of download links typed by hand is a table that is wrong one
    release later, so the only thing that may decide the rows is the release
    list itself.
    """
    R = _release_helper()
    before = R.render_installer_index(
        [_release("v1.5.0.4", *ALL_THREE), _release("v1.5.0.1", *ALL_THREE)],
        "1.5.0.4")
    after = R.render_installer_index(
        [_release("v1.5.0.5", *ALL_THREE),
         _release("v1.5.0.4", *ALL_THREE),
         _release("v1.5.0.1", *ALL_THREE)],
        "1.5.0.5")

    assert "1.5.0.5" not in before
    assert after.count("/download/v1.5.0.5/") == 3
    # and the row that used to be current is still there, no longer marked
    assert "1.5.0.4 (current)" in before
    assert "1.5.0.4 (current)" not in after
    assert "1.5.0.5 (current)" in after
    assert "1.5.0.4" in after


def test_the_table_is_newest_first_whatever_order_github_answers_in():
    """GitHub returns releases by creation date, which is not version order
    once a patch is published out of band."""
    R = _release_helper()
    rows = R.installer_index_rows([
        _release("v1.4.9.9", *ALL_THREE),
        _release("v1.5.0.4", *ALL_THREE),
        _release("v1.5.0.1", *ALL_THREE),
    ])
    assert [version for version, _ in rows] == ["1.5.0.4", "1.5.0.1", "1.4.9.9"]


def test_a_release_that_shipped_no_installer_is_not_given_a_row():
    """1.3.5, 1.3.6 and 1.4.9.8 predate the packaging work.

    Verified against the live release list rather than assumed: they carry
    wheels and sdists and no native installers at all. A row of three empty
    cells on a page whose whole subject is installers only invites the click
    that finds nothing.
    """
    R = _release_helper()
    rows = R.installer_index_rows([
        _release("v1.5.0.4", *ALL_THREE),
        {"tag_name": "v1.4.9.8", "assets": [
            {"name": "spacr-1.4.9.8-py3-none-any.whl",
             "browser_download_url": "https://example.invalid/w.whl"}]},
        {"tag_name": "v1.3.5", "assets": []},
    ])
    assert [version for version, _ in rows] == ["1.5.0.4"]


def test_a_platform_missing_from_one_release_gets_an_empty_cell():
    """Per release, not per history: a release that shipped two of the three
    keeps its row, and the third column is blank rather than a dead link."""
    R = _release_helper()
    page = R.render_installer_index(
        [_release("v2.0.0", "Linux-x86_64-Online.run",
                  "Windows-Online-Setup.exe")],
        "2.0.0")

    row = [line for line in page.splitlines()
           if line.startswith("   * - 2.0.0")]
    assert row, page
    body = page.split("   * - 2.0.0", 1)[1]
    cells = [line.strip() for line in body.splitlines() if line.startswith("     - ")]
    assert len(cells) == 3
    assert cells[0].startswith("- `.run <")     # Linux
    assert cells[1] == "-"                      # macOS: empty, not a link
    assert cells[2].startswith("- `.exe <")     # Windows
    assert ".pkg" not in page


def test_a_release_that_ships_installers_under_an_unreadable_tag_is_refused():
    """Dropping it would be a missing row on a page nobody re-reads."""
    R = _release_helper()
    with pytest.raises(ValueError, match="not a version"):
        R.installer_index_rows([_release("nightly", *ALL_THREE)])


def test_the_page_renders_without_a_docutils_warning():
    """Every row reuses the same three link labels.

    A *named* target repeated with a different URL is a duplicate-target
    error, so the rows must use anonymous references. The page is only ever
    seen after Sphinx has rendered it, and the docs build does not fail on
    warnings, so a broken table would publish looking like a broken table.
    """
    import io

    from docutils.core import publish_doctree

    R = _release_helper()
    page = R.render_installer_index(
        [_release("v1.5.0.4", *ALL_THREE),
         _release("v1.5.0.1", *ALL_THREE),
         _release("v1.4.9.9", "Linux-x86_64-Online.run")],
        "1.5.0.4")

    warnings = io.StringIO()
    publish_doctree(page, settings_overrides={
        "report_level": 2, "halt_level": 5, "warning_stream": warnings,
        "file_insertion_enabled": False})
    assert warnings.getvalue() == ""


def test_the_column_headers_are_the_readme_icons_and_not_a_second_copy():
    """One glyph set, one place it is drawn -- a second copy is a second
    thing to restyle, and the two would drift."""
    R = _release_helper()
    page = R.render_installer_index([_release("v1.5.0.4", *ALL_THREE)], "1.5.0.4")

    for _key, stem, _alt in R.README_ICONS:
        assert f"{R.README_ICON_ROOT}/{stem}.png" in page
    assert page.index("|Linux|") < page.index("|MacOS|") < page.index("|Windows|")


# ---------------------------------------------------------------------------
# the committed page
# ---------------------------------------------------------------------------

def _committed_rows():
    """``[(version, [cell, cell, cell])]`` read back out of the page."""
    rows = []
    for block in PAGE.read_text(encoding="utf-8").split("   * - ")[1:]:
        lines = block.splitlines()
        version = lines[0].replace("(current)", "").strip()
        cells = [line[len("     - "):].strip() for line in lines[1:]
                 if line.startswith("     - ")]
        rows.append((version, cells, "(current)" in lines[0]))
    return rows[1:]          # the first block is the header row


def test_the_committed_page_marks_the_version_setup_py_ships():
    """Exactly one row is the current one, and it is not decided by hand.

    Without this the page reads as a list of *old* versions and the reader
    cannot tell which one the README's icons point at.
    """
    R = _release_helper()
    current = R.read_version(ROOT / "setup.py")
    marked = [version for version, _cells, is_current in _committed_rows()
              if is_current]
    assert marked == [current]


def test_every_link_in_a_row_belongs_to_that_row_s_version():
    """The failure this page exists to prevent, one level in.

    A row whose cells point at a different release is worse than no archive:
    the reader believes they have downloaded 1.4.9.9 and has 1.5.0.4.
    """
    rows = _committed_rows()
    assert rows, "the committed page has no rows"
    for version, cells, _ in rows:
        assert len(cells) == 3, (version, cells)
        for cell in cells:
            if not cell:
                continue
            url = re.search(r"<(\S+)>", cell).group(1)
            assert f"/download/v{version}/" in url, (version, url)
            assert f"-{version}-" in url, (version, url)


def test_the_committed_page_is_newest_first():
    """Newest first was the shape asked for, and a page that drifts out of
    order reads as a page nobody regenerates."""
    from packaging.version import Version

    versions = [Version(version) for version, _c, _m in _committed_rows()]
    assert versions == sorted(versions, reverse=True)
    assert len(set(versions)) == len(versions)


def test_the_page_is_in_the_toctree_and_the_readme_points_at_it():
    """A page nothing links to is a page nobody reaches. The README's icons
    keep pointing at the latest; this is where someone goes for anything
    else, so the README is where the pointer belongs."""
    index = (ROOT / "docs" / "source" / "index.rst").read_text(encoding="utf-8")
    assert re.search(r"^\s+installers$", index, re.MULTILINE)
    readme = README.read_text(encoding="utf-8")
    assert "installers.html" in readme


@pytest.mark.parametrize("script,pattern", [
    ("install_spacr_unix.sh", r'PACKAGE_SPEC="spacr\[\$DEFAULT_EXTRAS\]==\$DEFAULT_SPACR_VERSION"'),
    ("install_spacr_windows.ps1", r'\$PackageSpec = "spacr\[\$DefaultExtras\]==\$Version"'),
])
def test_an_archived_installer_installs_its_own_version(script, pattern):
    """The claim the page makes, checked in the templates that make it.

    An online installer downloads the package at install time. Resolving "the
    latest spacr" instead would mean an archived 1.4.9.9 installer installs
    the current release while displaying 1.4.9.9 -- and the user believes they
    have reproduced an old run and has not. Both templates pin the version
    they were built for.
    """
    text = (ROOT / "packaging" / "online" / script).read_text(encoding="utf-8")
    assert re.search(pattern, text), f"{script} no longer pins its version"


@pytest.mark.network
def test_every_link_on_the_page_is_live():
    """A link table that is not verified is a link table with dead entries.

    Skips rather than fails when GitHub is unreachable, which is what the
    ``network`` marker means here.
    """
    import urllib.error
    import urllib.request

    urls = re.findall(r"<(https://github\.com/[^>]+)>", PAGE.read_text(encoding="utf-8"))
    assert len(urls) >= 3, "the page has no download links to check"

    for url in urls:
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                assert response.status == 200, f"{url} answered {response.status}"
        except urllib.error.HTTPError as error:          # a real dead link
            pytest.fail(f"{url} answered {error.code}")
        except OSError as error:                          # no network
            pytest.skip(f"GitHub unreachable: {error}")
