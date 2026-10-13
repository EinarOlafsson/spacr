"""The built documentation pages show no reST markers as text.

``tests/test_docstring_inline_markup_renders.py`` reads every docstring and
asks docutils whether its markup would render. This file asks the other
question: on the pages the reader actually gets, is any reST marker left
standing in the prose? The two answer different things -- one reads the
source, one reads the build, which also carries the ``.rst`` pages, the
autoapi templates and whatever Sphinx does between them -- and item 63 wrote
this scan twice in agent scratch directories before it was kept here.

HOW IT DECIDES. Each page is parsed with the standard library's HTML parser;
everything inside ``<pre>``, ``<code>``, ``<tt>``, ``<samp>``, ``<kbd>``,
``<script>`` and ``<style>`` is dropped, because a code sample that shows
backticks is not a leak. What remains is prose, and a leak is:

    a double-backtick run   ``SELECT *`` inside a strong span, a literal
                            glued to a letter (``lru_cache``s), or an
                            UNCLOSED one that docutils refused to open --
                            the scan must not demand a closing pair, which
                            is exactly the mistake the second rewrite fixed
                            (item 63, 2026-09-19, install_hint_for);
    a role marker's opening :class:`x` or :func:`x` rendered as text,
                            closed or not;
    a stray field marker    :raises SpecError: or :param top: part-way
                            through a line, which loses a documented field.

The control test runs always. The scan of the real build needs a FRESH build
-- ``docs/_build/html`` is untracked and whatever is on disk is whoever's
last local build -- so it runs only with ``SPACR_DOCS_BUILT=1``, the same
convention as ``tests/test_tooltip_api_links_resolve.py``.
"""
from __future__ import annotations

import html.parser
import os
import re
from pathlib import Path

import pytest

BUILT_DOCS = Path(__file__).resolve().parent.parent / "docs" / "_build" / "html"

#: Elements whose text is code, not prose, and may show markers legitimately.
CODE_TAGS = frozenset({"pre", "code", "tt", "samp", "kbd", "script", "style"})

#: Void elements never get an end tag, so they must not open a skip region.
VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "source", "track", "wbr",
})

FIELD_NAMES = (
    "param", "parameter", "arg", "argument", "key", "keyword", "type",
    "raises", "raise", "except", "exception", "returns", "return", "rtype",
    "yields", "yield", "ytype", "var", "ivar", "cvar", "vartype", "meta",
)

LEAK_PATTERNS = (
    ("double-backtick", re.compile(r"``")),
    ("role-marker", re.compile(r":(?:[a-z]+:)?[a-z][a-z-]*:`")),
    ("stray-field", re.compile(
        r"(?<![\w:]):(?:" + "|".join(FIELD_NAMES) + r")(?:\s+[\w.\[\], ]+)?:(?=\s)"
    )),
)


class _ProseText(html.parser.HTMLParser):
    """Collect the text of a page that sits outside every code element."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in CODE_TAGS and tag not in VOID_TAGS:
            self._depth += 1
        elif tag in ("p", "div", "li", "dd", "dt", "td", "th", "br",
                     "h1", "h2", "h3", "h4", "h5", "h6", "section"):
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in CODE_TAGS and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            self.chunks.append(data)


def markup_leaks(page_html: str) -> list[tuple[str, str]]:
    """Return ``(kind, line)`` for every reST marker left in a page's prose."""
    parser = _ProseText()
    parser.feed(page_html)
    parser.close()
    found = []
    for line in "".join(parser.chunks).splitlines():
        for kind, pattern in LEAK_PATTERNS:
            if pattern.search(line):
                found.append((kind, line.strip()[:160]))
    return found


CONTROL_PAGE = """
<html><head><style>p::before { content: "``"; }</style>
<script>var s = ":class:`x`";</script></head><body>
<p><strong>Never ``SELECT *`` the whole table.</strong></p>
<p>Wells ``plate1``..``plate4`` are read in order.</p>
<p>Cached with ``lru_cache``s per call.</p>
<p>joined with ``   # or <code class="docutils literal">, which a POSIX
shell reads</code> as a comment.</p>
<p><strong>See :class:`spacr.io.Reader` first.</strong></p>
<p>The fitted top. :param covariance_ok: whether the fit converged.</p>
<p>Returns a spec. :raises SpecError: when the column is unknown.</p>
<div class="highlight"><pre>x = "``literal``"  # :func:`shown`</pre></div>
<p>Call <code class="docutils literal"><span class="pre">run()</span></code>
with <tt>``quoted``</tt> and <kbd>:role:`k`</kbd>.</p>
<p>Ratio 1:2 at 10:30, see note: it is fine.</p>
</body></html>
"""


def test_the_scan_finds_every_shape_and_skips_code():
    """Seven planted leaks, and not one hit from code, script or plain colons."""
    leaks = markup_leaks(CONTROL_PAGE)
    kinds = [kind for kind, _line in leaks]
    assert len(leaks) == 7, leaks
    assert kinds.count("double-backtick") == 4
    assert kinds.count("role-marker") == 1
    assert kinds.count("stray-field") == 2
    assert not any("shown" in line or "run()" in line or "1:2" in line
                   for _kind, line in leaks)


def test_a_clean_page_is_clean():
    page = ("<p>Use <code class='docutils literal'>``x``</code> and "
            "<a href='#'>spacr.io.Reader</a>.</p>")
    assert markup_leaks(page) == []


@pytest.mark.skipif(
    not os.environ.get("SPACR_DOCS_BUILT"),
    reason=(
        "needs a FRESH docs build: docs/_build/html is untracked, so what is "
        "on disk is whoever's last local build. Build the docs, then run with "
        "SPACR_DOCS_BUILT=1."
    ))
def test_no_built_page_shows_rest_markup_as_text():
    pages = sorted(BUILT_DOCS.rglob("*.html"))
    assert len(pages) > 100, f"only {len(pages)} pages under {BUILT_DOCS}"
    leaking = {}
    for page in pages:
        leaks = markup_leaks(page.read_text(encoding="utf-8", errors="replace"))
        if leaks:
            leaking[str(page.relative_to(BUILT_DOCS))] = leaks
    assert not leaking, (
        f"{sum(map(len, leaking.values()))} leaks on {len(leaking)} pages:\n"
        + "\n".join(f"  {page}: {leaks[:3]}"
                    for page, leaks in sorted(leaking.items())[:20])
    )
