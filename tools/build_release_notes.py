#!/usr/bin/env python
"""Bundle the GitHub release notes into ``spacr/resources/release_notes.json``.

WHY BUNDLED AND NOT FETCHED. Home's News panel draws this file, and Home is
the first thing a user sees. A panel that fetched from api.github.com would
make the dashboard's content depend on the network being up, on a rate limit
shared with every other spaCR install behind the same NAT, and on a token the
user does not have; offline it would show nothing at all. The notes also
belong to the release: what shipped in 1.5.0.4 does not change after 1.5.0.4
is out, so a file in the wheel is the honest representation and a live fetch
would only ever tell the user about releases they do not have.

RUN THIS BEFORE TAGGING, after the release notes are written on GitHub:

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
