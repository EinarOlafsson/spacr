"""Set the software version (P348) on a Wikidata item, from a release.

The Wikidata item for spaCR carries ``P348 software version identifier``
qualified by ``P577 publication date``. Nothing was updating it, so the item
would have gone on claiming whatever version was current on the day it was
created.

Used by ``.github/workflows/wikidata.yml`` after PyPI has confirmed a
release. Standard library only -- a release job should not have to install a
MediaWiki client to change one string.

Authentication is a bot password from ``Special:BotPasswords``, supplied as
``WIKIDATA_BOT_USER`` and ``WIKIDATA_BOT_PASSWORD``. A bot password is scoped
and separately revocable, which an account password is not.

The claim is REPLACED in place rather than removed and recreated, so the
statement keeps its id, its references and its rank. Only the version string
and the date qualifier move.

Usage::

    python tools/wikidata_set_version.py --item Q123456 --version 1.5.1.0
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import http.cookiejar
import json
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Sequence

API = "https://www.wikidata.org/w/api.php"

#: Wikidata's Gregorian calendar item, required on every time datavalue.
GREGORIAN = "http://www.wikidata.org/entity/Q1985727"

#: ``precision`` 11 is "day"; anything coarser would drop the day from the
#: date and a release is a dated event.
DAY_PRECISION = 11


def time_datavalue(date: _datetime.date) -> Dict[str, Any]:
    """Build the ``P577`` time datavalue for ``date``."""
    return {
        "value": {
            "time": date.strftime("+%Y-%m-%dT00:00:00Z"),
            "timezone": 0,
            "before": 0,
            "after": 0,
            "precision": DAY_PRECISION,
            "calendarmodel": GREGORIAN,
        },
        "type": "time",
    }


def build_claim(version: str, date: _datetime.date,
                existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the ``P348`` claim to submit.

    :param version: the version string, for example ``1.5.1.0``.
    :param date: the release date, used as the ``P577`` qualifier.
    :param existing: the claim already on the item, if there is one. Its id,
        rank and references are preserved -- only the version and the date
        change -- so an editor's later work on the statement survives a
        release.
    :returns: a claim structure accepted by ``wbsetclaim``.
    """
    claim: Dict[str, Any] = json.loads(json.dumps(existing)) if existing else {
        "type": "statement",
        "rank": "normal",
    }
    claim["mainsnak"] = {
        "snaktype": "value",
        "property": "P348",
        "datatype": "string",
        "datavalue": {"value": version, "type": "string"},
    }
    qualifiers = claim.setdefault("qualifiers", {})
    qualifiers["P577"] = [{
        "snaktype": "value",
        "property": "P577",
        "datatype": "time",
        "datavalue": time_datavalue(date),
    }]
    order = claim.setdefault("qualifiers-order", [])
    if "P577" not in order:
        order.append("P577")
    return claim


def claim_is_current(claim: Dict[str, Any], version: str) -> bool:
    """Whether ``claim`` already states ``version``.

    A release that changes nothing should make no edit at all: a no-op edit
    still appears in the item's history and in every watchlist following it.
    """
    try:
        return claim["mainsnak"]["datavalue"]["value"] == version
    except (KeyError, TypeError):
        return False


class Session:
    """A logged-in MediaWiki API session."""

    def __init__(self) -> None:
        jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar))
        self._opener.addheaders = [
            ("User-Agent", "spacr-release-bot/1.0 "
                           "(https://github.com/EinarOlafsson/spacr)")]

    def _call(self, method: str, **params: Any) -> Dict[str, Any]:
        params.setdefault("format", "json")
        data = urllib.parse.urlencode(params).encode("utf-8")
        if method == "GET":
            request = urllib.request.Request(f"{API}?{data.decode('utf-8')}")
        else:
            request = urllib.request.Request(API, data=data)
        with self._opener.open(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if "error" in payload:
            raise RuntimeError(f"Wikidata API error: {payload['error']}")
        return payload

    def login(self, user: str, password: str) -> None:
        """Log in with a bot password."""
        tokens = self._call("GET", action="query", meta="tokens", type="login")
        result = self._call(
            "POST", action="login", lgname=user, lgpassword=password,
            lgtoken=tokens["query"]["tokens"]["logintoken"])
        status = result.get("login", {}).get("result")
        if status != "Success":
            raise RuntimeError(f"Wikidata login failed: {status}")

    def csrf_token(self) -> str:
        """Fetch a CSRF token for an edit."""
        tokens = self._call("GET", action="query", meta="tokens", type="csrf")
        return tokens["query"]["tokens"]["csrftoken"]

    def existing_claim(self, item: str) -> Optional[Dict[str, Any]]:
        """Return the item's current ``P348`` claim, or None."""
        payload = self._call(
            "GET", action="wbgetclaims", entity=item, property="P348")
        claims = payload.get("claims", {}).get("P348") or []
        return claims[0] if claims else None

    def set_claim(self, claim: Dict[str, Any], token: str,
                  summary: str) -> Dict[str, Any]:
        """Submit ``claim`` through ``wbsetclaim``."""
        return self._call(
            "POST", action="wbsetclaim", claim=json.dumps(claim),
            token=token, summary=summary, bot=1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``tools/wikidata_set_version.py`` entry point."""
    parser = argparse.ArgumentParser(
        description="Set P348 software version on a Wikidata item.")
    parser.add_argument("--item", required=True,
                        help="Item id, for example Q123456.")
    parser.add_argument("--version", required=True,
                        help="Version string, for example 1.5.1.0.")
    parser.add_argument("--date", default=None,
                        help="Release date as YYYY-MM-DD. Default: today.")
    parser.add_argument("--user", default=None,
                        help="Bot user. Default: WIKIDATA_BOT_USER.")
    parser.add_argument("--password", default=None,
                        help="Bot password. Default: WIKIDATA_BOT_PASSWORD.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the claim that would be sent, and stop.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    import os

    user = args.user or os.environ.get("WIKIDATA_BOT_USER", "")
    password = args.password or os.environ.get("WIKIDATA_BOT_PASSWORD", "")
    date = (_datetime.date.fromisoformat(args.date) if args.date
            else _datetime.date.today())

    if args.dry_run:
        print(json.dumps(build_claim(args.version, date), indent=2))
        return 0

    if not user or not password:
        print("error: WIKIDATA_BOT_USER and WIKIDATA_BOT_PASSWORD are required",
              file=sys.stderr)
        return 2

    session = Session()
    session.login(user, password)
    existing = session.existing_claim(args.item)
    if existing is not None and claim_is_current(existing, args.version):
        print(f"{args.item} already reports {args.version}; no edit made.")
        return 0

    claim = build_claim(args.version, date, existing)
    session.set_claim(
        claim, session.csrf_token(),
        summary=f"spaCR {args.version} released")
    print(f"{args.item} now reports {args.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
