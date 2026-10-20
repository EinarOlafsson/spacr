#!/usr/bin/env python3
"""Keep item 709's open-issue block and Status line in step with GitHub.

Item 709 ("resolve all open GitHub issues") is never finished for good: a
new issue reopens it. This tool rewrites two parts of the item file and
leaves every other byte alone:

* the ``Status:`` line, which reads ``OPEN — N open GitHub issues`` while any
  issue is open and ``DONE — no open GitHub issues (reopens automatically)``
  when none is;
* the lines between ``--- OPEN ISSUES (auto) ---`` and
  ``--- END OPEN ISSUES ---``, one ``#N title (author, YYYY-MM-DD)`` per issue
  in ascending number order, or ``(none)``.

Issues come from a JSON list, either ``gh issue list --json
number,title,author,createdAt`` output (``--json FILE``, ``-`` for stdin) or
the REST API (``--repo OWNER/REPO`` with ``GITHUB_TOKEN``). Pull requests,
which the REST issues endpoint also returns, are dropped. Running it twice
changes nothing.

    python tools/sync_open_issues_item.py --repo EinarOlafsson/spacr
    gh issue list --state open --json number,title,author,createdAt \\
        | python tools/sync_open_issues_item.py --json -

The workflow ``.github/workflows/issues-item.yml`` runs it on every issue
event and commits the file to nightly when it changed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEM = ROOT / "features" / "new" / "709_resolve_all_open_github_issues.txt"

BEGIN = "--- OPEN ISSUES (auto) ---"
END = "--- END OPEN ISSUES ---"
STATUS_OPEN = "OPEN — {n} open GitHub issue{s}"
STATUS_DONE = "DONE — no open GitHub issues (reopens automatically)"

_STATUS = re.compile(r"^(Status:[ \t]*)(.*)$", re.MULTILINE)


def normalise(raw):
    """``[(number, title, author, date)]`` from gh or REST issue JSON."""
    issues = []
    for entry in raw:
        if entry.get("pull_request") or entry.get("isPullRequest"):
            continue
        state = str(entry.get("state", "open")).lower()
        if state not in ("open",):
            continue
        author = entry.get("author") or entry.get("user") or {}
        login = author.get("login", "") if isinstance(author, dict) else str(author)
        created = entry.get("createdAt") or entry.get("created_at") or ""
        title = " ".join(str(entry.get("title", "")).split())
        issues.append((int(entry["number"]), title, login or "unknown", created[:10]))
    return sorted(set(issues))


def status_for(count):
    if count == 0:
        return STATUS_DONE
    return STATUS_OPEN.format(n=count, s="" if count == 1 else "s")


def block_lines(issues):
    if not issues:
        return ["(none)"]
    return [f"#{n} {title} ({author}, {date})" if date else f"#{n} {title} ({author})"
            for n, title, author, date in issues]


def rewrite(text, issues):
    """Return ``text`` with the Status line and the issue block replaced."""
    if not _STATUS.search(text):
        raise ValueError("the item file has no 'Status:' line")
    start = text.find(BEGIN)
    end = text.find(END, start + len(BEGIN)) if start >= 0 else -1
    if start < 0 or end < 0:
        raise ValueError(f"the item file has no '{BEGIN}' ... '{END}' block")
    body = "\n".join(block_lines(issues))
    text = text[:start] + BEGIN + "\n" + body + "\n" + text[end:]
    return _STATUS.sub(lambda m: m.group(1) + status_for(len(issues)), text, count=1)


def fetch(repo, token):
    """All open issues of ``repo`` from the REST API, pull requests included."""
    out, page = [], 1
    while True:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/issues"
            f"?state=open&per_page=100&page={page}",
            headers={"Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28",
                     **({"Authorization": f"Bearer {token}"} if token else {})})
        with urllib.request.urlopen(request, timeout=60) as response:
            batch = json.load(response)
        out.extend(batch)
        if len(batch) < 100:
            return out
        page += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", help="issue JSON file, or - for stdin")
    source.add_argument("--repo", help="OWNER/REPO to read with GITHUB_TOKEN")
    parser.add_argument("--item", type=Path, default=ITEM)
    args = parser.parse_args(argv)

    if args.json:
        handle = sys.stdin if args.json == "-" else open(args.json, encoding="utf-8")
        with handle:
            raw = json.load(handle)
    else:
        raw = fetch(args.repo, os.environ.get("GITHUB_TOKEN", ""))

    issues = normalise(raw)
    before = args.item.read_text(encoding="utf-8")
    after = rewrite(before, issues)
    if after != before:
        args.item.write_text(after, encoding="utf-8")
    print(f"{status_for(len(issues))}; {'updated' if after != before else 'unchanged'} "
          f"{args.item.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
