"""
Opt-in error reporting → pre-filled GitHub issue.

When the user turns on "File errors as GitHub issues" in the AI
Settings tab, the "Explain error" flow gains a second button:
"File as GitHub issue". Clicking it:

1. Builds a sanitized report from the current traceback + active app
   + settings + spacr / python / OS versions + tail of the log file.
2. URL-encodes the report into GitHub's `issues/new?title=…&body=…`
   query params.
3. Opens the user's default browser at that URL. GitHub uses the
   user's existing browser session — no token, no OAuth, no server
   round-trip. The user reviews and clicks Submit themselves.

Everything is deliberately kept client-side and one-click-away from
posting so users see exactly what leaves their machine before it
does.
"""
from __future__ import annotations

import hashlib
import os
import platform
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = "EinarOlafsson/spacr"
ISSUE_LABEL = "auto-filed"
LOG_TAIL_LINES = 50
MAX_URL_LEN = 7500   # GitHub caps the pre-filled issue URL at ~8 KB


# ---------------------------------------------------------------------------
# Sanitisation
# ---------------------------------------------------------------------------

def sanitize_path(s: str) -> str:
    """Replace absolute paths pointing inside ``$HOME`` with ``~/``.

    Also collapses any string that looks like an on-disk ``*.db`` path
    down to ``<DB>`` so lab / patient / experiment identifiers embedded
    in a filename don't leak.

    :param s: arbitrary text.
    :returns: text with home-relative paths abbreviated and DB paths
        redacted.
    """
    home = str(Path.home())
    s = s.replace(home, "~")
    # Redact any `.db` path suffix even if not under $HOME
    s = re.sub(r"[/\\][^\s'\"]+\.db\b", "<DB>", s)
    return s


def sanitize_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``settings`` with paths + DB names sanitized.

    :param settings: any pipeline settings dict.
    :returns: sanitized copy safe to include in a public issue.
    """
    out: Dict[str, Any] = {}
    for k, v in (settings or {}).items():
        if isinstance(v, str):
            out[k] = sanitize_path(v)
        elif isinstance(v, list):
            out[k] = [sanitize_path(x) if isinstance(x, str) else x
                      for x in v]
        else:
            out[k] = v
    return out


def sanitize_traceback(tb: str) -> str:
    """Sanitise a full traceback string via :func:`sanitize_path`."""
    return sanitize_path(tb or "")


def _traceback_hash(tb: str) -> str:
    """Short deterministic hash of a traceback for dedup coalescing.

    :returns: first 6 hex chars of sha256(sanitized traceback lines
        that start with ``File`` — filters out random line-numbers
        and stack noise so the same error dedupes cleanly).
    """
    lines = [ln for ln in tb.splitlines()
             if ln.strip().startswith(("File", "  File"))
             or ln.strip().startswith(("Error", "Exception"))]
    key = "\n".join(lines) or tb
    return hashlib.sha256(key.encode()).hexdigest()[:6]


# ---------------------------------------------------------------------------
# Log tail
# ---------------------------------------------------------------------------

def log_tail(n_lines: int = LOG_TAIL_LINES,
              log_path: Optional[Path] = None) -> str:
    """Return the last ``n_lines`` of ``~/.spacr/logs/spacr.log`` (or
    a custom path), sanitized.

    :param n_lines: how many trailing lines to include.
    :param log_path: override for the log file path.
    :returns: sanitised last-N-lines block or ``""`` if the file is
        absent or unreadable.
    """
    if log_path is None:
        try:
            from ..logging_util import log_path as _lp
            log_path = _lp()
        except Exception:
            return ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return ""
    return sanitize_path("".join(lines[-n_lines:]))


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _env_lines() -> List[str]:
    """Return lines describing the current spacr / python / OS env."""
    try:
        from spacr.version import __version__ as _spacr_ver
    except Exception:
        _spacr_ver = "unknown"
    return [
        f"- **spaCR**: {_spacr_ver}",
        f"- **Python**: {sys.version.split()[0]}",
        f"- **Platform**: {platform.platform()}",
        f"- **PySide6**: {_optional_version('PySide6')}",
        f"- **torch**: {_optional_version('torch')}",
        f"- **cellpose**: {_optional_version('cellpose')}",
    ]


def _optional_version(pkg: str) -> str:
    try:
        from importlib.metadata import version as _v
        return _v(pkg)
    except Exception:
        return "not installed"


def build_report(
    traceback_text: str,
    active_app: str = "",
    settings: Optional[Dict[str, Any]] = None,
    include_log_tail: bool = True,
) -> Dict[str, str]:
    """Build a ``(title, body)`` pair for a pre-filled GitHub issue.

    :param traceback_text: full traceback text (as caught by
        :func:`traceback.format_exc`).
    :param active_app: id of the app the user was in when the error
        fired (``"mask"`` / ``"measure"`` / …).
    :param settings: the pipeline settings dict in play, if any.
        Sanitised before inclusion.
    :param include_log_tail: also attach the last N log lines.
    :returns: dict with keys ``title`` and ``body`` ready to be
        URL-encoded onto ``issues/new``.
    """
    tb_clean = sanitize_traceback(traceback_text)
    tb_hash = _traceback_hash(tb_clean)

    # First non-empty error-type-looking line for the title
    err_line = ""
    for ln in reversed(tb_clean.splitlines()):
        if ln.strip() and not ln.startswith(" "):
            err_line = ln.strip()
            break
    err_line = err_line[:80] or "Runtime error"

    app_tag = f"[{active_app}] " if active_app else ""
    title = f"[auto {tb_hash}] {app_tag}{err_line}"[:120]

    body_parts: List[str] = []
    body_parts.append(
        "> Auto-filed from the spaCR AI Console. "
        f"Traceback fingerprint: `{tb_hash}`. "
        f"Active app: `{active_app or 'unknown'}`."
    )
    body_parts.append("")
    body_parts.append("### Traceback")
    body_parts.append("```")
    body_parts.append(tb_clean.strip())
    body_parts.append("```")
    body_parts.append("")
    body_parts.append("### Environment")
    body_parts.extend(_env_lines())
    body_parts.append("")

    if settings:
        clean_settings = sanitize_settings(settings)
        body_parts.append("<details><summary>Pipeline settings</summary>")
        body_parts.append("")
        body_parts.append("```")
        for k, v in clean_settings.items():
            body_parts.append(f"{k} = {v!r}")
        body_parts.append("```")
        body_parts.append("</details>")
        body_parts.append("")

    if include_log_tail:
        tail = log_tail()
        if tail:
            body_parts.append("<details><summary>Recent log lines</summary>")
            body_parts.append("")
            body_parts.append("```")
            body_parts.append(tail.strip())
            body_parts.append("```")
            body_parts.append("</details>")

    return {"title": title, "body": "\n".join(body_parts)}


# ---------------------------------------------------------------------------
# GitHub URL + browser opener
# ---------------------------------------------------------------------------

def issue_url(title: str, body: str, label: str = ISSUE_LABEL,
               repo: str = REPO) -> str:
    """Build the ``https://github.com/<repo>/issues/new?…`` URL.

    The URL is truncated to ~7.5 KB so it fits GitHub's parser limit;
    an ellipsis + note is appended to the body when we clip.

    :param title: URL-encodable issue title.
    :param body: markdown body; may be truncated.
    :param label: label to attach (created lazily by GitHub if it
        doesn't already exist).
    :param repo: ``owner/name`` slug.
    :returns: fully-quoted ``https://github.com/…`` URL.
    """
    # Reserve room for the fixed URL scaffolding + title
    scaffold_len = (
        len(f"https://github.com/{repo}/issues/new?labels={label}&title=&body=")
        + len(urllib.parse.quote(title))
    )
    if scaffold_len + len(urllib.parse.quote(body)) > MAX_URL_LEN:
        # Trim body — keep the traceback (most valuable), drop
        # subsequent details blocks.
        head_len = MAX_URL_LEN - scaffold_len - 80
        body = body[:head_len].rstrip()
        body += (
            "\n\n_[report truncated to fit GitHub URL limit — "
            "the full log lives at ~/.spacr/logs/spacr.log]_"
        )
    q = urllib.parse.urlencode({
        "labels": label,
        "title":  title,
        "body":   body,
    }, quote_via=urllib.parse.quote)
    return f"https://github.com/{repo}/issues/new?{q}"


def open_issue_in_browser(url: str) -> bool:
    """Open ``url`` in the user's default browser.

    :returns: ``True`` if webbrowser accepted the request, else False.
    """
    import webbrowser
    try:
        return webbrowser.open(url, new=2)
    except Exception:
        return False


def file_issue(
    traceback_text: str,
    active_app: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """End-to-end helper: build report, build URL, open browser, return URL.

    :param traceback_text: full traceback text.
    :param active_app: id of the app the user was in.
    :param settings: pipeline settings dict in play.
    :returns: the constructed ``https://github.com/…`` URL — useful for
        tests and for logging what was opened.
    """
    report = build_report(traceback_text, active_app=active_app,
                            settings=settings)
    # If the user is signed in to GitHub (stored token / env / gh CLI), create
    # the issue directly via the API — no browser needed. Otherwise fall back to
    # opening the pre-filled issues/new URL in the browser.
    try:
        from . import github_auth
        if github_auth.is_authenticated():
            ok, result = github_auth.create_issue(
                REPO, report["title"], report["body"], labels=[ISSUE_LABEL])
            if ok and result:
                return result   # the created issue's html_url
    except Exception:
        pass
    url = issue_url(report["title"], report["body"])
    open_issue_in_browser(url)
    return url
