#!/usr/bin/env python
"""Bundle the GitHub release notes into ``spacr/resources/release_notes.json``.

WHY BUNDLED. Home's News panel draws this file, and Home is the first thing a
user sees. A panel whose CONTENT came from api.github.com would depend on the
network being up, on a rate limit shared with every other spaCR install behind
the same NAT, and on a token the user does not have; offline it would show
nothing at all. So the bundle is the offline source of truth and the panel is
complete before anything touches a socket.

THE WHEEL CANNOT CONTAIN ITS OWN RELEASE NOTE, and this tool will not pretend
otherwise. The note for version X is written when the GitHub release for X is
published, and by then the wheel for X has been built, uploaded and made
immutable on PyPI. Nothing run before tagging can put X's note into X. What
``.github/workflows/release.yml`` does is run this tool AFTER
``gh release create`` and commit the result to main and nightly, so the
resource is current from that moment on and the NEXT wheel carries X. A user
on X therefore has a bundled list that stops at X-1, which is exactly the
staleness reported on 2026-09-24 ("im on 1.5.1.0 and the news only goes to
1.5.0.7"). The gap is closed at runtime instead: Home's News panel asks the
releases API once a day, on a worker thread, after the page is drawn, and
merges anything newer in front of the bundle -- see
:class:`spacr.qt.widgets.home.NewsPanel` and
:func:`spacr.updater.fetch_release_notes`.

RUN THIS AFTER THE RELEASE IS PUBLISHED, which the release workflow now does
for you. By hand, to refresh the resource on a working branch:

    python tools/build_release_notes.py

It calls ``gh`` (already required for the release workflow) and rewrites the
resource in place. ``--check`` exits non-zero when the bundled file is stale
instead of writing, which is what CI wants.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "EinarOlafsson/spacr"
TARGET = (Path(__file__).resolve().parent.parent
          / "spacr" / "resources" / "release_notes.json")

#: How many releases to bundle. All of them, in practice -- the file is a few
#: KB and "all of the other ones should be scrollable" (2026-09-03) means the
#: panel wants the whole history, not a window onto it.
PER_PAGE = 100


def fetch() -> list:
    """Ask GitHub for every release, newest first."""
    raw = subprocess.run(
        ["gh", "api", f"repos/{REPO}/releases?per_page={PER_PAGE}",
         "--jq", "[.[] | {tag: .tag_name, name: .name, "
                 "published: .published_at, url: .html_url, body: .body, "
                 "prerelease: .prerelease, draft: .draft}]"],
        check=True, capture_output=True, text=True).stdout
    return json.loads(raw)


def _links(body: str) -> list:
    """Every URL in ``body``, in order, de-duplicated.

    Pulled out at BUILD time rather than parsed in the panel. The bodies are
    GitHub-flavoured markdown and the panel renders a small subset of HTML;
    finding the links here means the panel never has to be a markdown parser,
    and a body whose formatting is unusual costs a link rather than a
    traceback in the dashboard.
    """
    seen, out = set(), []
    for url in re.findall(r"https?://[^\s<>)\]\"']+", body or ""):
        url = url.rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def build() -> dict:
    """The resource, ready to write."""
    entries = []
    for release in fetch():
        if release.get("draft"):
            # A draft is not released. Bundling one would announce a version
            # nobody can install.
            continue
        body = (release.get("body") or "").strip()
        entries.append({
            "tag": release.get("tag") or "",
            "name": (release.get("name") or release.get("tag") or "").strip(),
            "published": (release.get("published") or "")[:10],
            "url": release.get("url") or "",
            "body": body,
            "links": _links(body),
            "prerelease": bool(release.get("prerelease")),
        })
    return {"repo": REPO, "releases": entries}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the bundled file is out of date")
    args = parser.parse_args(argv)

    fresh = build()
    text = json.dumps(fresh, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} is stale -- run "
                  f"`python tools/build_release_notes.py`", file=sys.stderr)
            return 1
        print(f"{TARGET}: up to date "
              f"({len(fresh['releases'])} releases)")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text)
    print(f"wrote {TARGET} ({len(fresh['releases'])} releases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
