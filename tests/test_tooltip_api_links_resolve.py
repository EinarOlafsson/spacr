"""Every tooltip's API link must point at a page that exists.

A tooltip's link is the only route from a setting to its documentation, and
it is followed at exactly the moment the user is stuck. A dead one is worse
than no link: it costs a click and answers nothing.

Checked against the docs tree in this repository rather than over the
network -- a test that needs the internet is a test that fails on a plane
and gets deleted.
"""

import os
import re
import urllib.parse
from pathlib import Path

import pytest

#: Where the built site is. ``SPACR_DOCS_ROOT`` exists because sphinx builds
#: into ``docs/_build/html`` while this file used to look only in ``docs/`` --
#: so the resolve test below could not pass ANYWHERE, including the docs CI
#: job that is the one place a fresh build exists. A permanently-skipped test
#: counts as coverage while proving nothing (instruction 112).
DOCS = Path(os.environ.get("SPACR_DOCS_ROOT")
            or Path(__file__).resolve().parent.parent / "docs")

#: The prefix every generated link is built on.
SITE = "https://einarolafsson.github.io/spacr/"


def _tooltip_links():
    """(app_key, setting_key, url) for every setting of every module."""
    from spacr.qt.screens.settings_model import (
        resolve_default_settings, format_tooltip,
    )
    from spacr.qt.app import APPS

    seen = []
    for entry in APPS:
        app_key = entry[0] if isinstance(entry, (tuple, list)) else entry
        try:
            settings = resolve_default_settings(app_key)
        except Exception:
            continue
        for key in settings:
            html = format_tooltip("", app_key, key)
            for url in re.findall(r'href="([^"]+)"', html or ""):
                seen.append((app_key, key, url))
    return seen


@pytest.fixture(scope="module")
def links():
    pytest.importorskip("PySide6")
    found = _tooltip_links()
    assert found, "no tooltip produced a link at all — the test is not looking"
    return found


def test_every_link_is_on_the_documentation_site(links):
    off_site = {url for _a, _k, url in links if not url.startswith(SITE)}
    assert not off_site, f"tooltip links leaving the docs site: {sorted(off_site)}"


@pytest.mark.skipif(
    not os.environ.get("SPACR_DOCS_BUILT"),
    reason=(
        "needs a FRESH docs build. The api pages are generated and "
        "untracked, so whatever is on disk is whoever's last local build -- "
        "one held 4 module pages for 124 modules. Checking against that "
        "proves nothing about the published site and fails for everyone. "
        "The docs CI job now runs this against its own fresh build; locally, "
        "build the docs and run with SPACR_DOCS_BUILT=1 and "
        "SPACR_DOCS_ROOT=docs/_build/html."
    ))
def test_every_link_resolves_to_a_page_that_exists(links):
    """The point of the whole file.

    A wrong module name in a link produces a URL that looks perfectly
    reasonable and 404s, and nothing else in the suite would notice.

    Opt-in for the reason in the skip message above. The SHAPE checks below
    run always and catch the errors that do not need a built site -- a link
    off the documentation domain, or a setting with no link at all.
    """
    missing = {}
    for app_key, key, url in links:
        rel = urllib.parse.urlparse(url).path.lstrip("/")
        # The site serves this repository's `docs/` at /spacr/.
        if rel.startswith("spacr/"):
            rel = rel[len("spacr/"):]
        target = DOCS / rel
        if not target.exists():
            missing.setdefault(str(target.relative_to(DOCS)), set()).add(
                f"{app_key}.{key}")
    assert not missing, (
        "tooltip links point at pages that do not exist:\n"
        + "\n".join(f"  {page}  <- {sorted(keys)[:3]}"
                    for page, keys in sorted(missing.items()))
    )


def test_a_setting_with_no_documentation_still_gets_a_usable_link(links):
    """Every setting gets SOME link. A settings row with no route to the
    docs is the case this test exists to make impossible."""
    from spacr.qt.screens.settings_model import format_tooltip
    html = format_tooltip("", "mask", "a_key_that_does_not_exist")
    assert "href=" in html
