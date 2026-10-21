"""
Error reports filed as issues on the public spaCR GitHub repository.

These helpers build reports for automatic filing with a GitHub sign-in or submission
after an editable preview. Callers decide whether and when to report; anonymous relay
reporting is handled separately.

* ``'always'``: the caller can pass the report produced by :func:`public_report` to :func:`file_without_review`
  for automatic filing without a browser. This requires a GitHub sign-in; without one,
  nothing is sent. Eligibility checks, duplicate-memory checks and any wait for an AI
  answer are handled by the caller.
* ``'ask'``: the user can open an editable preview and approve it before
  :func:`submit_report` is called. Without a sign-in, or if API posting fails,
  submission opens a pre-filled GitHub form in the browser.

Both paths can use :func:`build_report`. With a GitHub sign-in, posting searches for an
open issue with the same traceback fingerprint and attempts to comment there. A failed
search or comment can lead to a new issue instead. This module does not persist GitHub
tokens.
"""
from __future__ import annotations

import hashlib
import platform
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional
from ...logging_util import _spacr_home

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
    """Replace substrings matched by the supported credential patterns.

    Apply the case-sensitive vendor patterns in ``_TOKEN_PATTERNS``, then the
    case-insensitive ``_BEARER_RE`` and ``_ASSIGN_RE`` patterns. Vendor patterns use
    fixed prefixes, character sets and minimum lengths. The Bearer pattern requires at
    least ``8`` token characters, and the assignment pattern at least ``6`` value
    characters. Bearer replacement preserves the matched scheme and whitespace;
    assignment replacement preserves its prefix and quote delimiter. Unrecognized or
    too-short text can remain. This function does not post reports, check credentials
    with a service or revoke tokens.

    :param s: arbitrary text.
    :returns: The input after the supported pattern substitutions, with matched
        credential text replaced by :data:`REDACTED`.
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
#:
#: The second group is the one a lab machine is actually called, and it is
#: here for a second reason: every one of these words also occurs inside
#: ordinary report text. A machine named ``gpu`` turned ``use_gpu = True``
#: in the settings block into ``use_<HOST> = True``, because ``_`` is a word
#: boundary to :func:`redact_identity`. Keeping the name in is the honest
#: trade — it says nothing about who the user is, and the alternative is a
#: report that rewrites words it was never meant to touch.
_GENERIC_NAMES = frozenset({
    "root", "user", "users", "admin", "administrator", "runner", "ubuntu",
    "jovyan", "vagrant", "guest", "test", "spacr", "python", "home",
    "localhost", "localdomain", "local", "default", "docker", "codespace",
    "codespaces",
    "gpu", "cpu", "bio", "lab", "data", "server", "desktop", "laptop",
    "workstation", "node", "host", "main", "dev", "build", "linux", "mac",
    "windows", "win", "imaging", "microscope", "compute", "cluster", "nas",
})


def _identity_words() -> List[tuple]:
    """This computer's login and host names, each with its placeholder.

    The login name comes from :func:`getpass.getuser` and from the name of
    the home folder. The host name comes from :func:`socket.gethostname` and
    :func:`platform.node`, both in full and as their first dotted label.
    Both calls read local state and do not touch the network. Names shorter
    than three characters, names that are all digits (the first label of a
    dotted-quad address is not a name, and ``192`` would otherwise be
    replaced wherever it stood), and the generic names in
    :data:`_GENERIC_NAMES` are left out.

    A name that is both the login and the host name — which is what a
    workstation named after its user gives — is listed once, as the login
    name, rather than twice with the host entry winning.

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
    users = {str(name or "").strip() for name in users}
    hosts = {str(name or "").strip() for name in hosts} - users
    words = []
    for names, placeholder in ((hosts, HOST_PLACEHOLDER),
                               (users, USER_PLACEHOLDER)):
        for name in names:
            if (len(name) < 3 or name.isdigit()
                    or name.lower() in _GENERIC_NAMES):
                continue
            words.append((name, placeholder))
    words.sort(key=lambda pair: len(pair[0]), reverse=True)
    return words


def redact_identity(s: str) -> str:
    """Replace eligible local login and host name matches with placeholders.

    Candidate names come from ``_identity_words``; names shorter than three characters,
    all-digit names and entries in ``_GENERIC_NAMES`` are omitted. Matches are
    case-insensitive and blocked if either adjacent character matches ``[A-Za-z0-9]``
    under ``re.IGNORECASE``; underscores do not block them. Home-string substitution is
    handled separately by :func:`sanitize_path`.

    :param s: arbitrary text.
    :returns: The input with eligible matches replaced by :data:`USER_PLACEHOLDER` or
        :data:`HOST_PLACEHOLDER`; excluded names and other unmatched identifiers are
        unchanged.
    """
    if not s:
        return s
    for name, placeholder in _identity_words():
        s = re.sub(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
                   placeholder, s, flags=re.IGNORECASE)
    return s


def sanitize_path(s: str) -> str:
    """Apply literal home-string replacement and pattern-based redaction.

    Replace exact, case-sensitive occurrences of ``str(Path.home())`` with ``~``
    wherever they occur; no path boundaries are checked. Replace substrings matching the
    case-sensitive database-path regex with ``<DB>``: a slash or backslash, one or more
    characters other than whitespace or single/double quotes, then ``.db`` and a regex
    word boundary. Finally apply :func:`redact_identity` and :func:`redact_secrets`.
    These substitutions do not guarantee removal of every identifier or credential.

    :param s: arbitrary text.
    :returns: The text after the literal home-string replacement and the database,
        identity and credential-pattern substitutions.
    """
    home = str(Path.home())
    s = s.replace(home, "~")
    s = re.sub(r"[/\\][^\s'\"]+\.db\b", "<DB>", s)
    s = redact_identity(s)
    return redact_secrets(s)


def sanitize_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow settings copy with selected strings sanitized.

    Top-level values whose string key names a credential (``api_key``, ``GITHUB_TOKEN``,
    ``password``, …) are replaced by :data:`REDACTED`; their keys remain. Other string
    values and direct string elements of lists pass through :func:`sanitize_path`.
    Nested mappings and other values are not recursively sanitized.

    :param settings: any pipeline settings dict.
    :returns: a shallow copy with the redactions described above; arbitrary nested data
        is not guaranteed safe for a public issue.
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
    """Sanitise a full traceback string via :func:`sanitize_path`.

    :param tb: the traceback text; ``None`` or an empty string gives ``""``.
    """
    return sanitize_path(tb or "")


#: Where a path can begin: a drive letter, a UNC or scheme-relative pair of
#: separators, a home-relative ``~/``, or a bare root slash.
_PATH_ROOT = r"(?:[A-Za-z]:[\\/]|[\\/]{2}|~[\\/]|/)"

#: One path component: anything up to the next separator, quote or space.
_PATH_SEG = r"[^\s'\"`\\/]+"

#: A traceback's ``File "..."`` field, whatever the file is called.
_FILE_FIELD_RE = re.compile(r'(?m)(\bFile\s+)["\'][^"\']+["\']')

#: A quoted value that IS a path, to its closing quote or the end of the
#: line. Everything that quotes a path quotes the whole of it: ``repr`` of a
#: settings value, the file name in an :class:`OSError` message, the log
#: copy's backticked path. The end-of-line branch is for the title, which
#: :func:`build_report` cuts to 80 characters and so can carry an opening
#: quote whose closing one was cut off.
_QUOTED_PATH_RE = re.compile(
    r"(?m)(?P<q>['\"`])" + _PATH_ROOT + r"[^'\"`\n]*(?P<end>(?P=q)|$)"
)

#: A word after the LAST component of an unquoted path that is taken as
#: part of that component: one that starts with a digit (``plate 1``,
#: ``Patient 042``) or a capital (``Lab Drive``, ``plate A``), or that ends
#: in a file extension (``plate a.tif``). Prose after a path on the same
#: line is lower case -- ``failed``, ``not found``, ``does not exist`` -- so
#: those words stay readable.
_PATH_TAIL_WORD = (r"(?:[0-9A-Z][^\s'\"`\\/]*"
                   r"|[^\s'\"`\\/]*\.[A-Za-z][A-Za-z0-9]{0,4}(?![\w.]))")

#: An unquoted path. A component may contain spaces when a separator
#: follows it, which is what makes ``/Volumes/Lab Drive/x`` and
#: ``C:\Program Files\spaCR\x`` one token rather than three. The last
#: component, which no separator follows, runs on over the words
#: :data:`_PATH_TAIL_WORD` accepts.
_BARE_PATH_RE = re.compile(
    r"(?<![\w~<])" + _PATH_ROOT + r"(?=[^\s'\"`\\/])"
    r"(?:" + _PATH_SEG + r"(?:[ \t]" + _PATH_SEG + r")*[\\/])*"
    r"(?:" + _PATH_SEG + r"(?:[ \t]" + _PATH_TAIL_WORD + r")*)?"
)


def strip_report_paths(text: str) -> str:
    """Replace recognized path-like text in an already sanitised report.

    The ordinary sanitizer abbreviates the home directory so a traceback is still
    useful. This helper then applies patterns for traceback file fields, quoted path
    values and remaining absolute path-like tokens, replacing recognized paths with
    ``<PATH>``. Callers use it for the stricter public-report form; it does not call
    :func:`sanitize_path`, display a preview or send a report. Unrecognized names can
    remain.

    A path can contain a space, and a rule that stops a path at the first
    whitespace publishes the rest of it. ``/Volumes/Lab Drive/Patient 042``
    came out as ``<PATH> Drive/Patient 042``, and ``C:\\Users\\anna\\OneDrive -
    Karolinska Institutet\\Screens`` kept the institution and the
    screen — the two operating systems whose own folders have spaces in
    them. A UNC path was not a path at all: nothing here started at
    ``\\\\``. The LAST component of an unquoted path has no separator after
    it to prove a space is inside the path rather than after it, so it runs
    on only over words shaped like a name -- a number, a capitalised word, a
    file name -- and ``\\\\LAB-NAS\\screens\\plate 1`` is replaced whole while
    ``opening /mnt/data/x.tif failed`` keeps its ``failed``. A lower-case
    word ending such a component still survives it; a quoted path is taken
    whole, and the title and every ``repr``-ed settings value are quoted.

    A slash straight after ``<`` starts a closing tag, not a path. The
    report's collapsible sections end in ``</summary>`` and ``</details>``,
    and issue #121 was filed with both turned into ``<<PATH>``, so every
    section after the first one stayed open.

    :param text: a report title or body, already through
        :func:`sanitize_path`.
    :returns: the same text with path-shaped tokens replaced by ``<PATH>``.
    """
    value = str(text or "")
    value = _FILE_FIELD_RE.sub(r'\1"<PATH>"', value)
    value = _QUOTED_PATH_RE.sub(
        lambda m: f"{m.group('q')}<PATH>{m.group('end')}", value)
    value = _BARE_PATH_RE.sub("<PATH>", value)
    return value


def public_report(report: Dict[str, str]) -> Dict[str, str]:
    """Prepare the path-stripped title and body of a report.

    The preview defaults to removing file and folder names, and automatic reporting uses
    this helper to prepare that form. Apply :func:`strip_report_paths` to the title and
    body after converting their values to strings; copy the fingerprint as a string.
    Missing fields default to empty strings, and other fields are omitted. This helper
    does not send the report or check user approval.

    :param report: a report from :func:`build_report`.
    :returns: A new mapping containing the string fields ``title``, ``body`` and
        ``fingerprint``. Path removal is limited to the patterns supported by
        :func:`strip_report_paths`.
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
    :returns: six hex characters, without building the report, copying a log or making a
        network call. Callers can use this key to check their remembered reports before
        doing that work; this function does not itself remember or suppress reports.
    """
    return _traceback_hash(sanitize_traceback(traceback_text))



def log_tail(n_lines: int = LOG_TAIL_LINES,
              log_path: Optional[Path] = None) -> str:
    """Read selected lines from the supplied or configured log file.

    :param n_lines: Number of trailing lines; zero selects all lines, and ``-k`` skips
        the first ``k`` lines.
    :param log_path: override for the log file path.
    :returns: Selected lines processed by :func:`sanitize_path`, or ``""`` if resolving
        the configured path or reading the file fails. The entire file is read as UTF-8
        with invalid bytes replaced before selecting lines.
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
    return _spacr_home() / "reports"


def save_log_bundle(fingerprint: str,
                    log_path: Optional[Path] = None,
                    n_lines: int = LOG_BUNDLE_LINES) -> Optional[Path]:
    """Write the log tail to a file beside the report and return its path.

    The report builder can include this local path instead of the log contents; stricter
    public redaction may replace the path. The file is not constrained by the issue URL
    length limit.

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
    """Build a report mapping containing a title, body and traceback fingerprint.

    :param traceback_text: full traceback text (as caught by
        :func:`traceback.format_exc`).
    :param active_app: id of the app the user was in when the error
        fired (``"mask"`` / ``"measure"`` / …).
    :param settings: the pipeline settings dict in play, if any.
        Sanitised before inclusion.
    :param include_log_tail: attempt to save a sanitized local log tail and include its
        local path in the body. The log contents are not attached to the public issue.
    :param ai_response: spaCR AI's analysis of this same error, when available. It
        passes through :func:`sanitize_path` and is shortened after
        :data:`AI_ANALYSIS_MAX_CHARS` characters, with a truncation note added. It is
        marked as unreviewed machine-generated advice rather than a diagnosis.
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

    If the estimated URL-encoded length exceeds :data:`MAX_URL_LEN`, shorten the body
    and append a truncation note. The title, repository and label are not shortened, so
    unusually large values can still exceed that limit.

    :param title: URL-encodable issue title.
    :param body: markdown body; may be truncated.
    :param label: label requested in the pre-filled form; this helper does not create
        it.
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

    :param url: the address to open, typically from :func:`issue_url`; it is
        opened in a new tab where the browser supports one.
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
    """Submit an approved payload through the API, with browser-form fallback.

    :param report: The approved payload with ``title`` and ``body`` keys, plus
        ``fingerprint`` for the duplicate search when a token is available. Callers
        prepare any required redaction before calling. If API posting does not yield a
        URL, construct a pre-filled issue URL and request that the browser open it. This
        helper does not display an approval preview or apply :func:`public_report`.
    :returns: The API result URL, the pre-filled ``issues/new`` URL, or a transport
        refusal message. Returning the browser-form URL does not mean the browser
        accepted it or an issue was filed. Errors constructing that fallback URL can
        propagate.
    """
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
    """Attempt automatic API reporting without opening a browser.

    Runs synchronously on the calling thread: resolving a token can run ``gh auth
    token``, and GitHub API calls can block while waiting for responses. Callers choose
    where to run it.

    This helper is used for issue reporting set to 'always', but does not enforce that
    setting. A transport refusal produces :data:`REFUSED`, and no available token
    produces :data:`SIGNED_OUT`. Otherwise attempt an API post or a comment on an
    existing issue; no browser form is opened.

    :param report: the payload to post, already passed through
        :func:`public_report`.
    :returns: A mapping with ``status`` and optional ``url`` or ``detail`` fields.
        Status is :data:`FILED`, :data:`SEEN_AGAIN` (a successful comment on an existing
        issue with the same fingerprint), :data:`SIGNED_OUT`, :data:`REFUSED` or
        :data:`FAILED`. Caught :class:`Exception` instances are converted to
        :data:`FAILED` with an exception description.
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

    This function builds a report and immediately passes it to :func:`submit_report`; it
    does not show a preview or apply :func:`public_report`. GUI callers handle preview
    approval or automatic-filing decisions separately. A headless caller is responsible
    for deciding whether this report may be submitted.

    :param traceback_text: full traceback text, passed to
        :func:`build_report`, which sanitises it and derives the issue title
        and fingerprint from it.
    """
    report = build_report(
        traceback_text,
        active_app=active_app,
        settings=settings,
        include_log_tail=include_log_tail,
        ai_response=ai_response,
    )
    return submit_report(report)
