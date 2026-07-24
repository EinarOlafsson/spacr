"""GitHub authentication + direct issue creation for spaCR.

Lets users file auto-issues WITHOUT a browser round-trip. A token is resolved
from, in order:

1. A Personal Access Token the user stored in spaCR (Settings → GitHub).
2. The ``GITHUB_TOKEN`` / ``GH_TOKEN`` environment variable.
3. The GitHub CLI (``gh auth token``), if ``gh`` is installed + logged in.

When a token is available, :func:`create_issue` POSTs straight to the GitHub
REST API and returns the created issue's URL. When none is available the caller
falls back to opening the pre-filled ``issues/new`` URL in the browser.

Public API::

    github_auth.is_authenticated()      -> bool
    github_auth.auth_source()            -> "token" | "env" | "gh" | None
    github_auth.get_stored_token()       -> str
    github_auth.set_stored_token(tok)    -> None
    github_auth.create_issue(repo, title, body, labels) -> (ok, url_or_error)
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from typing import List, Optional, Tuple

from PySide6.QtCore import QSettings

_ORG = "spacr"
_APP = "qt"
_KEY_TOKEN = "github/pat"


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------

def get_stored_token() -> str:
    """Return the user-stored Personal Access Token (or empty string)."""
    return str(_settings().value(_KEY_TOKEN, "") or "")


def set_stored_token(token: str) -> None:
    """Persist (or clear, with '') a Personal Access Token."""
    token = (token or "").strip()
    if token:
        _settings().setValue(_KEY_TOKEN, token)
    else:
        _settings().remove(_KEY_TOKEN)


def _env_token() -> str:
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

def create_issue(repo: str, title: str, body: str,
                 labels: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Create a GitHub issue directly via the REST API.

    :param repo: ``owner/name`` slug.
    :param title: issue title.
    :param body: markdown body.
    :param labels: labels to attach (created lazily by GitHub if new).
    :returns: ``(True, issue_html_url)`` on success, else ``(False, error)``.
    """
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
        with urllib.request.urlopen(req, timeout=20) as resp:
            body_out = resp.read().decode("utf-8")
            info = json.loads(body_out)
            return True, info.get("html_url", "")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("message", "")
        except Exception:
            pass
        return False, f"GitHub API error {e.code}: {detail or e.reason}"
    except Exception as e:  # noqa: BLE001 — surface any network/parse failure
        return False, f"Failed to reach GitHub: {e}"
