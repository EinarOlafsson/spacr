"""
Error reports filed as issues on the public spaCR GitHub repository.

A failed run is reported in one of two ways, chosen by the "One-click issue
filing" preference (:func:`spacr.qt.preferences.get_issue_prompt_mode`):

* ``'always'``, the default: :func:`file_without_review` files the report as
  soon as the run has failed. It sends :func:`public_report`, the same
  redaction the preview applies by default, and it never opens a browser.
  Filing needs a GitHub sign-in (the ``gh`` CLI or ``GITHUB_TOKEN``). Without
  one, nothing is sent.
* ``'ask'``: the report opens in an editable preview, and
  :func:`submit_report` sends it only after the Send click. Without a
  sign-in it opens a pre-filled GitHub form in the browser.

Both paths build the report with :func:`build_report`, and both look for an
open issue carrying the same traceback fingerprint before they open a new
one. spaCR never stores a durable GitHub token itself.
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
MAX_URL_LEN = 7500



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


#: Placeholder for this computer's login name.
USER_PLACEHOLDER = "<USER>"

#: Placeholder for this computer's network name.
HOST_PLACEHOLDER = "<HOST>"

#: Login and host names that identify nobody, and are left in place.
_GENERIC_NAMES = frozenset({
    "root", "user", "users", "admin", "administrator", "runner", "ubuntu",
    "jovyan", "vagrant", "guest", "test", "spacr", "python", "home",
    "localhost", "localdomain", "local", "default", "docker", "codespace",
    "codespaces",
})


def _identity_words() -> List[tuple]:
    """This computer's login and host names, each with its placeholder.

    The login name comes from :func:`getpass.getuser` and from the name of
    the home folder. The host name comes from :func:`socket.gethostname` and
    :func:`platform.node`, both in full and as their first dotted label.
    Both calls read local state and do not touch the network. Names shorter
    than three characters, and the generic names in :data:`_GENERIC_NAMES`,
    are left out.

    :returns: ``[(name, placeholder)]``, longest name first, so a host name
        that contains the login name is replaced whole.
    """
    users, hosts = set(), set()
    try:
        import getpass

        users.add(getpass.getuser())
    except Exception:                                        # noqa: BLE001
        pass
    try:
        users.add(Path.home().name)
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import socket

        hosts.add(socket.gethostname())
    except Exception:                                        # noqa: BLE001
        pass
    try:
        hosts.add(platform.node())
    except Exception:                                        # noqa: BLE001
        pass
    hosts |= {name.split(".", 1)[0] for name in list(hosts) if name}
    words = []
    for names, placeholder in ((hosts, HOST_PLACEHOLDER),
                               (users, USER_PLACEHOLDER)):
        for name in names:
            name = str(name or "").strip()
            if len(name) < 3 or name.lower() in _GENERIC_NAMES:
                continue
            words.append((name, placeholder))
    words.sort(key=lambda pair: len(pair[0]), reverse=True)
    return words


def redact_identity(s: str) -> str:
    """Replace this computer's login and host names with placeholders.

    A home folder is already shortened to ``~`` by :func:`sanitize_path`. The
    login name can still appear elsewhere: another folder named after the
    user, a permission error, an ``owner`` setting. The host name can appear
    in a network path or a connection error. Each is replaced wherever it
    stands as a whole word, in any letter case.

    :param s: arbitrary text.
    :returns: the text with :data:`USER_PLACEHOLDER` and
        :data:`HOST_PLACEHOLDER` in place of those names.
    """
    if not s:
        return s
    for name, placeholder in _identity_words():
        s = re.sub(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
                   placeholder, s, flags=re.IGNORECASE)
    return s


def sanitize_path(s: str) -> str:
    """Replace absolute paths pointing inside ``$HOME`` with ``~/``.

    Also collapses any string that looks like an on-disk ``*.db`` path
    down to ``<DB>`` so lab / patient / experiment identifiers embedded
    in a filename don't leak, replaces this computer's login and host names
    through :func:`redact_identity`, and redacts credential-shaped substrings
    via :func:`redact_secrets`.

    :param s: arbitrary text.
    :returns: text with home-relative paths abbreviated and DB paths,
        login and host names and secrets redacted.
    """
    home = str(Path.home())
    s = s.replace(home, "~")
    s = re.sub(r"[/\\][^\s'\"]+\.db\b", "<DB>", s)
    s = redact_identity(s)
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

    A slash straight after ``<`` starts a closing tag, not a path. The
    report's collapsible sections end in ``</summary>`` and ``</details>``,
    and issue #121 was filed with both turned into ``<<PATH>``, so every
    section after the first one stayed open.
    """
    value = str(text or "")
    value = re.sub(r'(?m)(\bFile\s+)["\'][^"\']+["\']', r'\1"<PATH>"', value)
    value = re.sub(r"(?<![\w~<])(?:[A-Za-z]:[\\/]|/)[^\s'\"`]+", "<PATH>",
                   value)
    value = re.sub(r"(?<!\w)~[/\\][^\s'\"`]+", "<PATH>", value)
    return value


def public_report(report: Dict[str, str]) -> Dict[str, str]:
    """The report as it is sent when nobody reviews it first.

    The preview opens with "Remove file and folder names" switched on, so
    what a reviewer sends by default is the body after
    :func:`strip_report_paths`. A report filed automatically gets that same
    body. The title is stripped as well: it quotes the exception line, and
    that line can carry a file name.

    :param report: a report from :func:`build_report`.
    :returns: ``title``, ``body`` and ``fingerprint``, ready to post.
    """
    return {
        "title": strip_report_paths(str(report.get("title", ""))),
        "body": strip_report_paths(str(report.get("body", ""))),
        "fingerprint": str(report.get("fingerprint", "")),
    }


#: ``, line 123,`` inside a traceback frame — volatile, stripped before hashing.
_LINENO_RE = re.compile(r",\s*line\s+\d+\s*,")

#: A traceback frame line: the quoted file, then the rest of the line.
_FRAME_RE = re.compile(r'^File\s+"(?P<path>[^"]*)"(?P<rest>.*)$')


def _frame_key(frame: str) -> str:
    """One frame line as the fingerprint sees it.

    The file is cut to its last two path components, which is the module
    and the package it sits in. Everything above them depends on the
    machine: the home folder, the Python version, the name of the conda
    environment, the operating system's separator. With the whole path in
    the key, the same crash on two computers had two fingerprints, so the
    open-issue search could never find the other computer's report.

    :param frame: a stripped ``File "...", line N, in f`` line.
    :returns: the line with the path shortened and the line number removed.
    """
    match = _FRAME_RE.match(frame)
    if match is not None:
        parts = [p for p in re.split(r"[\\/]+", match.group("path")) if p]
        where = "/".join(parts[-2:]) if parts else match.group("path")
        frame = f'File "{where}"{match.group("rest")}'
    return _LINENO_RE.sub(",", frame)


def _traceback_hash(tb: str) -> str:
    """Short deterministic fingerprint of a traceback, for dedup coalescing.

    The key is built from the call stack (file + function, with the
    volatile line NUMBERS removed) plus the exception TYPE. That gives
    the three properties dedup needs:

    * the same bug still fingerprints the same after an unrelated edit
      shifts the line numbers above it,
    * the same bug fingerprints the same on another computer, because each
      file is named by its last two path components (:func:`_frame_key`),
      and
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
            lines.append(_frame_key(stripped))
        elif not ln.startswith((" ", "\t")):
            if stripped.startswith("Traceback"):
                continue
            lines.append(stripped.split(":", 1)[0])
    key = "\n".join(lines) or tb
    return hashlib.sha256(key.encode()).hexdigest()[:6]


def fingerprint_of(traceback_text: str) -> str:
    """The fingerprint :func:`build_report` gives this traceback.

    :param traceback_text: the raw traceback.
    :returns: six hex characters, without building the report. The
        automatic filer checks this against what it has filed before, so a
        repeated crash costs no log copy and no network call.
    """
    return _traceback_hash(sanitize_traceback(traceback_text))



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
        saved = save_log_bundle(tb_hash)
        if saved is not None:
            body_parts.append("<details><summary>Log</summary>")
            body_parts.append("")
            body_parts.append(
                "The log is NOT attached: it can carry sample names, plate "
                "barcodes and folder names, and this issue is public.")
            body_parts.append("")
            body_parts.append(f"It was saved on the reporter's machine at "
                              f"`{sanitize_path(str(saved))}`.")
            body_parts.append("")
            body_parts.append(
                "If you need it, ask -- and read it before sending it.")
            body_parts.append("</details>")

    return {"title": title, "body": "\n".join(body_parts),
            "fingerprint": tb_hash}



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
    scaffold_len = (
        len(f"https://github.com/{repo}/issues/new?labels={label}&title=&body=")
        + len(urllib.parse.quote(title))
    )
    if scaffold_len + len(urllib.parse.quote(body)) > MAX_URL_LEN:
        note = (
            "\n\n_[report truncated to fit GitHub URL limit — "
            "the full log lives at ~/.spacr/logs/spacr.log]_"
        )
        budget = MAX_URL_LEN - scaffold_len - len(urllib.parse.quote(note))
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


#: What :func:`file_without_review` reports back, one of these.
FILED = "filed"
SEEN_AGAIN = "seen_again"
SIGNED_OUT = "signed_out"
REFUSED = "refused"
FAILED = "failed"


def _post_report(report: Dict[str, str]) -> Dict[str, str]:
    """Post a report through the GitHub API, onto an open duplicate if any.

    Shared by both ways of filing. The open-issue search is by the
    fingerprint, which :func:`build_report` writes into every body, so a
    crash that already has an open issue gets a "Seen again" comment there
    instead of a second issue. A search that could not run does not stop
    the report.

    :param report: ``title``, ``body`` and ``fingerprint``.
    :returns: ``{"status": FILED or SEEN_AGAIN, "url": ...}``, or
        ``{"status": FAILED, "detail": ...}``.
    """
    from . import github_auth

    searched, existing = github_auth.find_issue_by_fingerprint(
        REPO, report["fingerprint"])
    if searched and existing:
        ok, _ = github_auth.comment_on_issue(
            REPO, existing.get("number"),
            "Seen again.\n\n" + report["body"])
        if ok:
            return {"status": SEEN_AGAIN,
                    "url": str(existing.get("html_url", "") or "")}
    ok, result = github_auth.create_issue(
        REPO, report["title"], report["body"], labels=[ISSUE_LABEL])
    if ok and result:
        return {"status": FILED, "url": str(result)}
    return {"status": FAILED, "detail": str(result or "no issue came back")}


def submit_report(report: Dict[str, str]) -> str:
    """Submit one payload the user has already approved in the preview."""
    try:
        from . import github_auth
        refusal = github_auth._transport_refusal()
        if refusal:
            return refusal
        if github_auth.is_authenticated():
            posted = _post_report(report)
            if posted.get("url"):
                return posted["url"]
    except Exception:
        pass
    url = issue_url(report["title"], report["body"])
    open_issue_in_browser(url)
    return url


def file_without_review(report: Dict[str, str]) -> Dict[str, str]:
    """File a report automatically, for issue reporting set to 'always'.

    Runs on a worker thread: resolving the sign-in can run ``gh auth
    token``, and posting waits on api.github.com.

    It never opens a browser. The browser form is a prompt the user
    answers, and 'always' is the choice not to be prompted. So without a
    GitHub sign-in it files nothing and says so.

    :param report: the payload to post, already passed through
        :func:`public_report`.
    :returns: ``{"status": ..., "url": ..., "detail": ...}`` with a status
        of :data:`FILED`, :data:`SEEN_AGAIN` (a comment on the open issue
        with the same fingerprint), :data:`SIGNED_OUT`, :data:`REFUSED`
        (inside a test run) or :data:`FAILED`. Never raises.
    """
    try:
        from . import github_auth

        refusal = github_auth._transport_refusal()
        if refusal:
            return {"status": REFUSED, "detail": refusal}
        if not github_auth.is_authenticated():
            return {"status": SIGNED_OUT}
        return _post_report(report)
    except Exception as exc:                                 # noqa: BLE001
        return {"status": FAILED, "detail": f"{type(exc).__name__}: {exc}"}


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
