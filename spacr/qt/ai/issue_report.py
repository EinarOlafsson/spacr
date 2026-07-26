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

#: Placeholder substituted for anything that looks like a credential.
REDACTED = "<REDACTED>"

#: Vendor-specific credential shapes. Matched anywhere in the text —
#: a traceback, a settings value or a log line can all carry one.
_TOKEN_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{8,}"),
)

#: ``Authorization: Bearer <token>`` — keep the scheme, drop the secret.
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._\-]{8,}")

#: ``api_key = 'xxx'`` / ``GITHUB_TOKEN: xxx`` style assignments.
_ASSIGN_RE = re.compile(
    r"(?i)"
    r"([\"']?\b[A-Za-z0-9_\-]*"
    r"(?:api[_-]?key|secret|passwd|password|token|credential)"
    r"[A-Za-z0-9_\-]*\b[\"']?\s*[=:]\s*)"
    r"([\"']?)"
    r"([^\s,;'\"}\)]{6,})"
    r"\2"
)

#: Settings keys whose *value* is dropped wholesale regardless of shape.
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|secret|passwd|password|token|credential)"
)


def redact_secrets(s: str) -> str:
    """Strip anything that looks like an API key / access token.

    The issue body is posted to a PUBLIC GitHub repo, so a token that
    survived into a traceback, a settings value or a log line would be
    leaked to the world (and, for GitHub PATs, instantly revoked).

    :param s: arbitrary text.
    :returns: the same text with credential-shaped substrings replaced
        by :data:`REDACTED`.
    """
    if not s:
        return s
    for pat in _TOKEN_PATTERNS:
        s = pat.sub(REDACTED, s)
    s = _BEARER_RE.sub(lambda m: m.group(1) + REDACTED, s)
    s = _ASSIGN_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}{m.group(2)}", s)
    return s


def sanitize_path(s: str) -> str:
    """Replace absolute paths pointing inside ``$HOME`` with ``~/``.

    Also collapses any string that looks like an on-disk ``*.db`` path
    down to ``<DB>`` so lab / patient / experiment identifiers embedded
    in a filename don't leak, and redacts credential-shaped substrings
    via :func:`redact_secrets`.

    :param s: arbitrary text.
    :returns: text with home-relative paths abbreviated and DB paths +
        secrets redacted.
    """
    home = str(Path.home())
    s = s.replace(home, "~")
    # Redact any `.db` path suffix even if not under $HOME
    s = re.sub(r"[/\\][^\s'\"]+\.db\b", "<DB>", s)
    return redact_secrets(s)


def sanitize_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``settings`` with paths + DB names sanitized.

    Values whose *key* names a credential (``api_key``, ``GITHUB_TOKEN``,
    ``password``, …) are dropped entirely — the key name is enough of a
    hint that the value must never reach a public issue.

    :param settings: any pipeline settings dict.
    :returns: sanitized copy safe to include in a public issue.
    """
    out: Dict[str, Any] = {}
    for k, v in (settings or {}).items():
        if isinstance(k, str) and _SECRET_KEY_RE.search(k):
            out[k] = REDACTED
        elif isinstance(v, str):
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


#: ``, line 123,`` inside a traceback frame — volatile, stripped before hashing.
_LINENO_RE = re.compile(r",\s*line\s+\d+\s*,")


def _traceback_hash(tb: str) -> str:
    """Short deterministic fingerprint of a traceback, for dedup coalescing.

    The key is built from the call stack (file + function, with the
    volatile line NUMBERS removed) plus the exception TYPE. That gives
    the two properties dedup needs:

    * the same bug still fingerprints the same after an unrelated edit
      shifts the line numbers above it, and
    * two genuinely different exceptions raised from the same frame get
      different fingerprints instead of being merged into one issue.

    The exception *message* is deliberately excluded — it routinely
    embeds a filename or a plate id, which would fork the fingerprint on
    every run.

    :returns: first 6 hex chars of sha256 over that key.
    """
    lines: List[str] = []
    for ln in tb.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("File "):
            lines.append(_LINENO_RE.sub(",", stripped))
        elif not ln.startswith((" ", "\t")):
            if stripped.startswith("Traceback"):
                continue
            # "ValueError: channels must be a list" -> "ValueError"
            lines.append(stripped.split(":", 1)[0])
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
