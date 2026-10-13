"""Set the version on the bio.tools entry, from a release.

bio.tools does NOT accept PATCH -- it answers 405 Method Not Allowed -- so a
field cannot be changed on its own. The only write is a PUT of the whole
entry.

That makes the read the important part. The entry is fetched from bio.tools
immediately before the write and only its version is changed, so anything an
ELIXIR curator has altered on their side is carried straight back. Sending a
copy stored in this repository would silently revert their work on every
release.

Two shapes in the API's own output are rejected by its input validation, and
both have to be removed before the PUT:

* server-managed fields (``additionDate``, ``owner``, ``validated`` and the
  rest), which are not writable;
* ``null`` and ``[]``, which GET returns for every unset optional field and
  PUT refuses with "This field may not be null".

Usage::

    BIOTOOLS_TOKEN=... python tools/biotools_set_version.py --version 1.5.1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Sequence

ENTRY = "https://bio.tools/api/tool/spacr/?format=json"

#: Set by the server; present in GET output and refused by PUT.
SERVER_MANAGED = (
    "additionDate", "lastUpdate", "owner", "editPermission", "validated",
    "homepage_status", "confidence_flag", "elixir_badge",
)


def strip_empty(value: Any) -> Any:
    """Drop every ``None`` and every empty list, recursively.

    GET fills unset optional fields with null and unset lists with [], and
    PUT rejects both. Stripping them is what makes a fetched entry writable.
    """
    if isinstance(value, dict):
        return {key: strip_empty(inner) for key, inner in value.items()
                if inner is not None and inner != []}
    if isinstance(value, list):
        return [strip_empty(item) for item in value]
    return value


def with_version(entry: Dict[str, Any], version: str) -> Dict[str, Any]:
    """Return ``entry`` ready to PUT, carrying ``version`` and nothing else new.

    :param entry: the entry as bio.tools just returned it.
    :param version: the version to publish.
    :returns: the same entry, cleaned for PUT, with its version replaced.
    """
    body = {key: value for key, value in entry.items()
            if key not in SERVER_MANAGED}
    body["version"] = [version]
    return strip_empty(body)


def fetch(url: str = ENTRY) -> Dict[str, Any]:
    """Read the current entry."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def put(body: Dict[str, Any], token: str) -> int:
    """Write the entry back. Returns the HTTP status."""
    request = urllib.request.Request(
        ENTRY, method="PUT",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Token {token}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        print(f"bio.tools answered HTTP {error.code}: {detail}", file=sys.stderr)
        return error.code


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``tools/biotools_set_version.py`` entry point."""
    parser = argparse.ArgumentParser(
        description="Publish a version to the bio.tools entry for spaCR.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and print what would be sent, and stop.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    entry = fetch()
    if entry.get("version") == [args.version]:
        print(f"bio.tools already reports {args.version}; no write made.")
        return 0

    body = with_version(entry, args.version)
    if args.dry_run:
        print(json.dumps(body, indent=1)[:4000])
        return 0

    token = os.environ.get("BIOTOOLS_TOKEN", "")
    if not token:
        print("error: BIOTOOLS_TOKEN is not set", file=sys.stderr)
        return 2

    status = put(body, token)
    if status in (401, 403):
        print("::error::bio.tools rejected the token; replace BIOTOOLS_TOKEN.",
              file=sys.stderr)
        return 1

    # The POST that created this entry answered 500 after writing the row, so
    # a status is never the evidence. Re-reading is.
    live = fetch()
    if live.get("version") != [args.version]:
        print(f"::error::bio.tools reports {live.get('version')}, "
              f"not {args.version}.", file=sys.stderr)
        return 1
    print(f"bio.tools now reports {args.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
