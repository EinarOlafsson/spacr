"""
Opt-in error reporting → pre-filled GitHub issue.

When the user enables public issue reporting during installation or later in
Preferences, the error flow gains a "File as GitHub issue" action. Clicking
it:

1. Builds a sanitized report from the current traceback + active app
   + settings + spacr / python / OS versions + tail of the log file.
2. Shows the exact title and body in an editable preview, with filenames
   stripped by default.
3. Submits only after the report-specific Send click. An authenticated
   official ``gh`` session can post through the API; otherwise spaCR opens a
   pre-filled GitHub form in the browser for the user to submit there.

spaCR never stores a durable GitHub token itself. Everything stays client-side
until the explicit preview action, so the user sees exactly what leaves the
machine before it does.
"""
from __future__ import annotations

import hashlib
import platform
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = "EinarOlafsson/spacr"
ISSUE_LABEL = "auto-filed"
LOG_TAIL_LINES = 50

#: Lines kept in the log file saved beside a report. Larger than
#: :data:`LOG_TAIL_LINES` because this one is not going into a URL.
LOG_BUNDLE_LINES = 2000
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


def strip_report_paths(text: str) -> str:
    """Remove file/folder names from an already sanitised report.

    The ordinary sanitizer abbreviates the home directory so a traceback is
    still useful. Public reports default to the stricter form: traceback file
    fields and remaining absolute path-like tokens become ``<PATH>``. The
    preview lets the user restore the useful names before sending.
    """
    value = str(text or "")
    value = re.sub(r'(?m)(\bFile\s+)["\'][^"\']+["\']', r'\1"<PATH>"', value)
    value = re.sub(r"(?<![\w~])(?:[A-Za-z]:[\\/]|/)[^\s'\"`]+", "<PATH>", value)
    value = re.sub(r"(?<!\w)~[/\\][^\s'\"`]+", "<PATH>", value)
    return value


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


def log_bundle_dir() -> Path:
    """Where a report's log copy is written."""
    return Path.home() / ".spacr" / "reports"


def save_log_bundle(fingerprint: str,
                    log_path: Optional[Path] = None,
                    n_lines: int = LOG_BUNDLE_LINES) -> Optional[Path]:
    """Write the log tail to a file beside the report and return its path.

    The public issue names this path instead of carrying the log itself.
    More lines are kept here than would ever have gone in an issue --
    once the log is not being pasted into a URL there is no length to
    stay under, and whoever reads the report wants the whole run, not a
    keyhole.

    :param fingerprint: the traceback hash, so one report's log is easy
        to match to the issue that names it.
    :param log_path: override for the log file path.
    :param n_lines: how many trailing lines to keep.
    :returns: the path written, or ``None`` if there was nothing to write
        or the write failed -- a report must still be filable on a
        read-only home directory.
    """
    tail = log_tail(n_lines=n_lines, log_path=log_path)
    if not tail.strip():
        return None
    try:
        folder = log_bundle_dir()
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"log-{fingerprint}.txt"
        target.write_text(tail, encoding="utf-8")
    except Exception:
        return None
    return target


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
    """Report an optional package's version for the issue body.

    :param pkg: the distribution name.
    :returns: its version, or a marker saying it is not installed -- which
        is itself worth knowing in a bug report, since half of spaCR's
        failures are a missing extra.
    """
    try:
        from importlib.metadata import version as _v
        return _v(pkg)
    except Exception:
        return "not installed"


#: How much of spaCR AI's analysis goes into an issue.
#:
#: It sits between the traceback and the environment, and `issue_url` trims
#: the TAIL of the body to fit GitHub's URL limit -- so an unbounded analysis
#: would push the environment, settings and log-bundle path out of the report
#: entirely. Four thousand characters is several screens of prose, which is
#: more than any useful diagnosis needs.
AI_ANALYSIS_MAX_CHARS = 4000


def build_report(
    traceback_text: str,
    active_app: str = "",
    settings: Optional[Dict[str, Any]] = None,
    include_log_tail: bool = True,
    ai_response: str = "",
) -> Dict[str, str]:
    """Build a ``(title, body)`` pair for a pre-filled GitHub issue.

    :param traceback_text: full traceback text (as caught by
        :func:`traceback.format_exc`).
    :param active_app: id of the app the user was in when the error
        fired (``"mask"`` / ``"measure"`` / …).
    :param settings: the pipeline settings dict in play, if any.
        Sanitised before inclusion.
    :param include_log_tail: also attach the last N log lines.
    :param ai_response: spaCR AI's analysis of this same error, when the AI
        is switched on and has already answered. Sanitised and length-capped
        like everything else here, and clearly marked as machine-generated:
        it is a lead for whoever reads the report, not a finding.
    :returns: dict with keys ``title``, ``body`` and ``fingerprint``,
        ready to be
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

    # AFTER THE TRACEBACK, BEFORE THE ENVIRONMENT. When the AI is on it has
    # usually already diagnosed the crash by the time the user files, and that
    # analysis is the most useful thing in the report after the traceback
    # itself -- it is what a reader would otherwise spend the first hour
    # reproducing. It goes below the traceback because `issue_url` trims the
    # tail, and the traceback must survive that trim.
    #
    # MARKED AS MACHINE-GENERATED, and folded shut. It is a lead, not a
    # finding: the analysis in the session this was written for was right
    # about the cause and wrong about the fix, in a way that would have
    # changed behaviour silently for every run that left a field blank.
    analysis = sanitize_path(str(ai_response or "")).strip()
    if analysis:
        if len(analysis) > AI_ANALYSIS_MAX_CHARS:
            analysis = (analysis[:AI_ANALYSIS_MAX_CHARS].rstrip()
                        + "\n\n… (analysis truncated)")
        body_parts.append(
            "<details><summary>spaCR AI's analysis of this error"
            "</summary>")
        body_parts.append("")
        body_parts.append(
            "Generated by spaCR AI from the traceback above, unreviewed. "
            "Treat it as a lead rather than a diagnosis.")
        body_parts.append("")
        body_parts.append(analysis)
        body_parts.append("</details>")
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
        # THE LOG DOES NOT GO IN THE ISSUE. An issue on the public tracker
        # is world-readable and permanent, and a log line carries whatever
        # the run happened to be about -- a gene name, a plate barcode, a
        # collaborator's folder, the name of an unpublished screen. None of
        # that is credential-shaped, so no redaction pass catches it, and
        # the person filing the bug has no way to know it is there.
        #
        # So the log is written BESIDE the report instead: a file on the
        # user's own disk, whose path the issue names. The maintainer can
        # ask for it, and the user decides then, having read it.
        saved = save_log_bundle(tb_hash)
        if saved is not None:
            body_parts.append("<details><summary>Log</summary>")
            body_parts.append("")
            body_parts.append(
                "The log is NOT attached: it can carry sample names, plate "
                "barcodes and folder names, and this issue is public.")
            body_parts.append("")
            # Through the same sanitiser as everything else: the bundle
            # lives under the user's home, and the home path carries their
            # account name.
            body_parts.append(f"It was saved on the reporter's machine at "
                              f"`{sanitize_path(str(saved))}`.")
            body_parts.append("")
            body_parts.append(
                "If you need it, ask -- and read it before sending it.")
            body_parts.append("</details>")

    # `fingerprint` is returned, not just embedded in the body, so the
    # caller can look for an existing issue carrying it before opening a
    # new one. Without that the hash was written and never read.
    return {"title": title, "body": "\n".join(body_parts),
            "fingerprint": tb_hash}


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
        #
        # Measured against the ENCODED length, not the raw one. This used to
        # slice `body[:head_len]` with head_len computed from the URL budget
        # in raw characters, which is a different unit: quoting expands, and
        # a traceback is mostly newlines at three characters each (`%0A`).
        # A realistic crash report came out at 11,924 characters against a
        # 7,500 limit AFTER "truncation", and GitHub answers an over-long
        # issues/new with a page that reads "page not found" -- which is the
        # 404 users were getting.
        note = (
            "\n\n_[report truncated to fit GitHub URL limit — "
            "the full log lives at ~/.spacr/logs/spacr.log]_"
        )
        budget = MAX_URL_LEN - scaffold_len - len(urllib.parse.quote(note))
        # Shrink until the QUOTED body fits. Halving converges in a few
        # passes for any expansion ratio, where a fixed guess cannot: the
        # ratio is 1x for plain ASCII and 3x for newline-dense text, and the
        # body that matters most here is the newline-dense one.
        head = body
        while head and len(urllib.parse.quote(head)) > budget:
            head = head[:max(1, int(len(head) * 0.8))]
        body = head.rstrip() + note
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


def submit_report(report: Dict[str, str]) -> str:
    """Submit one payload the user has already approved in the preview."""
    # If the user is signed in to GitHub (stored token / env / gh CLI), create
    # the issue directly via the API — no browser needed. Otherwise fall back to
    # opening the pre-filled issues/new URL in the browser.
    try:
        from . import github_auth
        # This check uses the module instance resolved NOW.  A broad batch once
        # left this module holding a different instance from the one a test had
        # patched; its process-wide allow flag then sent four real comments to
        # issue #114.  A real transport is refused before credential discovery,
        # while an explicitly substituted offline seam can exercise the flow.
        refusal = github_auth._transport_refusal()
        if refusal:
            return refusal
        if github_auth.is_authenticated():
            # DEDUPE BY FINGERPRINT FIRST. `_traceback_hash` exists so the
            # same bug hashes the same across runs and machines, and nothing
            # consumed it: one crash produced one issue per occurrence -- ten
            # in a single day on 2026-08-11 (#79-#81, #84-#90), which buries
            # the reports that matter.
            #
            # A hit gets a COMMENT rather than a new issue, because the
            # second occurrence is still information: it says the bug is
            # reproducible and carries that run's environment.
            #
            # `searched` is distinguished from "found nothing" deliberately.
            # If the search could not run we still file, because losing a
            # crash report is worse than filing a duplicate.
            searched, existing = github_auth.find_issue_by_fingerprint(
                REPO, report["fingerprint"])
            if searched and existing:
                number = existing.get("number")
                url = existing.get("html_url", "")
                ok, _ = github_auth.comment_on_issue(
                    REPO, number,
                    "Seen again.\n\n" + report["body"])
                if ok:
                    return url
            ok, result = github_auth.create_issue(
                REPO, report["title"], report["body"], labels=[ISSUE_LABEL])
            if ok and result:
                return result   # the created issue's html_url
    except Exception:
        pass
    url = issue_url(report["title"], report["body"])
    open_issue_in_browser(url)
    return url


def file_issue(
    traceback_text: str,
    active_app: str = "",
    settings: Optional[Dict[str, Any]] = None,
    *,
    include_log_tail: bool = True,
    ai_response: str = "",
) -> str:
    """Legacy end-to-end helper retained for API callers and tests.

    The GUI does not call this directly: it builds the payload, displays an
    editable preview, then passes the approved mapping to
    :func:`submit_report`. Headless callers invoking this function are the
    report-specific affirmative action themselves.
    """
    report = build_report(
        traceback_text,
        active_app=active_app,
        settings=settings,
        include_log_tail=include_log_tail,
        ai_response=ai_response,
    )
    return submit_report(report)
