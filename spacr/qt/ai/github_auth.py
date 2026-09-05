"""GitHub authentication + direct issue creation for spaCR.

Lets users file approved issues WITHOUT a browser round-trip. A token is
resolved from the ``GITHUB_TOKEN`` / ``GH_TOKEN`` environment or the GitHub
CLI (``gh auth token``). spaCR does not capture or persist credentials; the
official CLI owns interactive login and its platform credential store.

When a token is available, :func:`create_issue` POSTs straight to the GitHub
REST API and returns the created issue's URL. When none is available the caller
falls back to opening the pre-filled ``issues/new`` URL in the browser.

Public API::

    github_auth.is_authenticated()      -> bool
    github_auth.auth_source()            -> "token" | "env" | "gh" | None
    github_auth.create_issue(repo, title, body, labels) -> (ok, url_or_error)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

from PySide6.QtCore import QSettings

_ORG = "spacr"
_APP = "qt"
_KEY_TOKEN = "github/pat"
_EPHEMERAL_TOKEN = ""

# The HTTP seam used by every GitHub API request.  Production leaves this at
# the real stdlib transport.  Offline tests replace the seam itself, which is
# materially different from setting an environment variable that says a real
# write is allowed: the replacement is process-local, cannot be inherited by a
# subprocess, and any late teardown call still lands in the fake transport.
_REAL_HTTP_OPEN = urllib.request.urlopen
_HTTP_OPEN = _REAL_HTTP_OPEN


def _settings() -> QSettings:
    """Open spaCR's ``QSettings``.

    :returns: the settings store.
    """
    return QSettings(_ORG, _APP)


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------

def get_stored_token() -> str:
    """Return a process-only token, after erasing insecure legacy storage."""
    settings = _settings()
    if settings.contains(_KEY_TOKEN):
        settings.remove(_KEY_TOKEN)
        settings.sync()
    return _EPHEMERAL_TOKEN


def set_stored_token(token: str) -> None:
    """Set a process-only compatibility token; never persist it.

    The GUI no longer exposes this function. It remains for API callers and
    offline transport tests that need to inject a credential for one process.
    Interactive users authenticate with ``gh auth login``.
    """
    global _EPHEMERAL_TOKEN
    token = (token or "").strip()
    _EPHEMERAL_TOKEN = token
    _settings().remove(_KEY_TOKEN)


def _env_token() -> str:
    """Read a GitHub token from the environment.

    :returns: the token, or ``""`` when none is set -- checked before the
        stored one so a CI run can override what a developer saved.
    """
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return ""


def _gh_cli_token() -> str:
    """Return a token from the GitHub CLI (`gh auth token`), or empty string."""
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True,
                             text=True, timeout=8)
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return ""


def resolve_token() -> Tuple[str, Optional[str]]:
    """Return ``(token, source)`` — source is 'token' | 'env' | 'gh' | None."""
    # Calling this also erases a token left by an older spaCR build. The
    # process-only value exists for API injection, never installer/UI login.
    tok = get_stored_token()
    if tok:
        return tok, "token"
    tok = _env_token()
    if tok:
        return tok, "env"
    tok = _gh_cli_token()
    if tok:
        return tok, "gh"
    return "", None


def is_authenticated() -> bool:
    """True iff a GitHub token is available from any source."""
    return bool(resolve_token()[0])


def auth_source() -> Optional[str]:
    """Where the active token comes from — 'token' | 'env' | 'gh' | None."""
    return resolve_token()[1]


# ---------------------------------------------------------------------------
# Issue creation
# ---------------------------------------------------------------------------

#: Fallback net for any rendering of the credential we didn't anticipate.
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s'\"]{8,}")

REDACTED = "<REDACTED>"


def _scrub(message: str, token: str) -> str:
    """Remove ``token`` from a message that is about to be shown or logged.

    Some failures echo the outgoing request back at us — most concretely
    ``http.client.putheader`` raises
    ``ValueError: Invalid header value b'Bearer ghp_…'`` when a stored PAT
    contains an embedded newline (a routine copy-paste artefact from a
    wrapped terminal). Without this scrub that string lands verbatim in
    the UI status line and in the log.

    Both the literal token and its backslash-escaped (``repr``) rendering
    are removed, then any surviving ``Bearer …`` value as a backstop.
    """
    if token:
        escaped = token.encode("unicode_escape").decode("ascii", "replace")
        for form in (token, escaped):
            if form and form in message:
                message = message.replace(form, REDACTED)
    return _BEARER_RE.sub(lambda m: m.group(1) + REDACTED, message)


def _refuse_writes_under_test() -> Optional[str]:
    """Return a reason to refuse, or None when writing is allowed.

    spaCR's issue reporter posts to the REAL tracker whenever a token is
    resolvable, and on a developer machine the ``gh`` CLI supplies one. So any
    test that reaches a write path without mocking files a live issue --
    which is how ``[auto 54a0e8] [mask] Error: boom`` (#75) arrived on the
    public tracker from a test fixture.

    ``PYTEST_CURRENT_TEST`` is not sufficient: pytest removes it between
    phases and at session teardown.  The root conftest therefore installs
    ``SPACR_PYTEST_SESSION`` before collection; ordinary subprocesses inherit
    it.  There is deliberately no environment-variable escape hatch.
    """
    import os

    if (os.environ.get("SPACR_PYTEST_SESSION") == "1"
            or "PYTEST_CURRENT_TEST" in os.environ):
        return "refusing GitHub network access from inside a test run"
    return None


def _transport_refusal() -> Optional[str]:
    """Refuse real GitHub transport in tests, while admitting a fake seam.

    Offline transport tests replace :data:`_HTTP_OPEN`, so exercising request
    construction remains possible without an escape hatch.  A subprocess
    imports this module afresh and therefore gets :data:`_REAL_HTTP_OPEN`; a
    fixture teardown restores it.  Both cases remain refused for the lifetime
    of the pytest session.
    """
    reason = _refuse_writes_under_test()
    if reason and _HTTP_OPEN is _REAL_HTTP_OPEN:
        return reason
    return None


def find_issue_by_fingerprint(repo: str, fingerprint: str
                              ) -> Tuple[bool, Optional[dict]]:
    """Find an open GitHub issue containing a diagnostic fingerprint.

    Parameters
    ----------
    repo : str
        Repository slug in ``owner/name`` form.
    fingerprint : str
        Short diagnostic hash stored in the issue body.

    Returns
    -------
    search_completed : bool
        ``True`` when GitHub returned a search result, including no matches.
        ``False`` when authentication is unavailable or the search fails.
    issue : dict or None
        First matching issue, or ``None`` when no issue was found or the
        search could not be completed.
    """
    if _transport_refusal():
        return False, None

    token, _src = resolve_token()
    if not token:
        return False, None

    query = urllib.parse.quote(
        f'repo:{repo} is:issue is:open in:body "{fingerprint}"')
    req = urllib.request.Request(
        f"https://api.github.com/search/issues?q={query}&per_page=1",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "spacr",
        },
    )
    try:
        with _HTTP_OPEN(req, timeout=20) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        items = info.get("items") or []
        return True, (items[0] if items else None)
    except Exception:
        return False, None


def comment_on_issue(repo: str, number: int, body: str) -> Tuple[bool, str]:
    """Add a comment to an existing issue.

    :param repo: ``owner/name`` slug.
    :param number: the issue number.
    :param body: markdown comment body.
    :returns: ``(True, comment_html_url)`` on success, else ``(False, error)``.
    """
    refusal = _transport_refusal()
    if refusal:
        return False, refusal

    token, _src = resolve_token()
    if not token:
        return False, "Not signed in to GitHub (no token available)."

    data = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{number}/comments",
        data=data, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "spacr",
            "Content-Type": "application/json",
        },
    )
    try:
        with _HTTP_OPEN(req, timeout=20) as resp:
            info = json.loads(resp.read().decode("utf-8"))
            return True, info.get("html_url", "")
    except Exception as exc:
        return False, _scrub(str(exc), token)


def create_issue(repo: str, title: str, body: str,
                 labels: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Create a GitHub issue directly via the REST API.

    :param repo: ``owner/name`` slug.
    :param title: issue title.
    :param body: markdown body.
    :param labels: labels to attach (created lazily by GitHub if new).
    :returns: ``(True, issue_html_url)`` on success, else ``(False, error)``.
    """
    refusal = _transport_refusal()
    if refusal:
        return False, refusal

    token, _src = resolve_token()
    if not token:
        return False, "Not signed in to GitHub (no token available)."

    payload = {"title": title, "body": body}
    if labels:
        payload["labels"] = list(labels)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=data, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "spacr",
            "Content-Type": "application/json",
        },
    )
    try:
        with _HTTP_OPEN(req, timeout=20) as resp:
            body_out = resp.read().decode("utf-8")
            info = json.loads(body_out)
            return True, info.get("html_url", "")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("message", "")
        except Exception:
            pass
        return False, _scrub(
            f"GitHub API error {e.code}: {detail or e.reason}", token)
    except Exception as e:  # noqa: BLE001 — surface any network/parse failure
        return False, _scrub(f"Failed to reach GitHub: {e}", token)
