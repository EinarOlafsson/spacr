"""Create an attachable diagnostic bundle with ``spacr-crashreport``.

This module produces one ``.zip`` with the logs, settings, versions and
diagnostics a bug report needs, for attaching to a GitHub issue, and it is
assembled out of the pieces spaCR already keeps rather than collected a second
time:

* :mod:`spacr.doctor` runs 17 checks over the installation — the running
  checkout, duplicate installs, the GPU, the Cellpose version, the project
  database, the settings — and every non-``PASS`` row already carries the fix.
* :mod:`spacr.runctx` writes a per-run JSONL log keyed by run id, so "show me
  everything from the run that produced this" has an answer. The bundle carries
  that run's log verbatim, not a filtered summary of it.
* :mod:`spacr.errors` stamps each run's status onto the artifact it wrote, so
  the report can say whether the last run *finished* — which is a different
  question from whether it crashed just now, and the more useful one when a
  user reports numbers rather than a traceback.

Design rules:

**Collectors do not raise.** Every collector runs through :func:`_collect`,
which turns a failure into a manifest entry naming what could not be gathered
and why. Catching ``Exception`` is acceptable here because the failure is
recorded in the output: the manifest is part of the bundle, so a missing
section is visible to the report recipient instead of silently absent.

**Bounded size.** ``spacr.log`` rotates but a single run can still write
hundreds of megabytes. The bundle takes the *tail* of it, capped by
:data:`MAX_LOG_BYTES`, and the manifest records how much was dropped.

**Nothing secret.** Environment variables are included because ``SPACR_*``,
``CUDA_*`` and ``PATH`` explain a large fraction of "works here, not there" —
but a value whose *name* looks like a credential is replaced with
``"<redacted>"`` and listed by name in the manifest, so you can see what was
withheld and the report recipient can see that something was. Absolute paths are
kept: they name the checkout, the plate and the database, and a report with
them stripped cannot be acted on.

**Deterministic layout.** File names and ordering remain stable so repeated
reports are quick to inspect.

Anonymous reports
-----------------
The bundle above stays on this computer. What may leave it is a separate,
much smaller thing: :func:`public_summary`, an allowlisted record of the
versions, the operating-system family, the GPU class, the module, the
exception type, a scrubbed one-line message and the call stack as file
basenames, function names and line numbers. No log, setting, environment
variable or path is in it. :func:`submit_public_summary` posts it to spaCR's
relay (:data:`REPORT_RELAY_URL`), which files it as a public issue on
EinarOlafsson/spacr, so a user needs no GitHub account. The switch is
:func:`anonymous_reports_enabled`, on unless the user turns it off; with it
off, or with :data:`NO_NETWORK_ENV` set, nothing is sent. A report that
cannot be sent waits in a small queue (:data:`QUEUE_LIMIT` reports,
:data:`QUEUE_RETENTION_S` seconds) and is tried again later.
:func:`diagnose_remotely` sends the same summary to the relay's hosted model
for advice.

Usage
-----
.. code-block:: bash

    python -m spacr.crashreport                     # last run, current folder
    python -m spacr.crashreport --run-id 3f9c1a2b   # one particular run
    python -m spacr.crashreport --db plate1/measurements/measurements.db \\
        --settings measure.csv --app measure -o ~/spacr-bug.zip

.. code-block:: python

    from spacr.crashreport import report_exception, install_excepthook

    install_excepthook()          # any unhandled crash writes a bundle
    try:
        measure_crop(settings)
    except Exception as exc:      # noqa: BLE001 - reporting, then re-raising
        print(report_exception(exc, settings=settings))
        raise
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import platform
import re
import sys
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MAX_LOG_BYTES",
    "MAX_RUN_LOG_BYTES",
    "REDACT_TOKENS",
    "REPORTED_ENVIRONMENT",
    "CrashReport",
    "collect",
    "find_last_run_id",
    "install_excepthook",
    "report_exception",
    "write_crash_report",
    "REPORT_RELAY_URL",
    "REPORT_RELAY_ENV",
    "NO_NETWORK_ENV",
    "PUBLIC_SUMMARY_KEYS",
    "RelayRefused",
    "public_summary",
    "scrub_public_text",
    "scrub_public_lines",
    "public_problem_lines",
    "explanation_signature",
    "explanation_key",
    "saved_explanation",
    "save_explanation",
    "normalise_cause",
    "split_cause",
    "relay_url",
    "network_allowed",
    "anonymous_reports_enabled",
    "set_anonymous_reports_enabled",
    "anonymous_report_notice_due",
    "mark_anonymous_report_notice_shown",
    "install_id",
    "queued_summaries",
    "submit_public_summary",
    "submit_public_summary_in_background",
    "flush_report_queue",
    "diagnose_remotely",
    "fetch_allowance",
    "next_reset_utc",
    "ai_summary_prompt",
    "classify_failure",
    "error_report_mode",
    "set_error_report_mode",
    "mode_explains",
    "mode_sends",
    "mode_code_context",
    "example_public_summary",
    "build_parser",
    "main",
]

#: Bytes of ``spacr.log`` kept, counted from the end. Two megabytes is about
#: 20k lines, which reaches back past any plausible cause while staying inside
#: every issue tracker's attachment limit.
MAX_LOG_BYTES = 2 * 1024 * 1024

#: Bytes of the per-run JSONL kept, counted from the end. Larger than the main
#: log's share because this one is already scoped to the run that failed, which
#: makes it the most valuable thing in the bundle.
MAX_RUN_LOG_BYTES = 8 * 1024 * 1024

#: Substrings that make an environment variable's *value* a credential. Matched
#: case-insensitively against the name, never against the value: a heuristic
#: over values would redact a plate path containing the word "key" and leave a
#: token whose name nobody thought of.
REDACT_TOKENS: Tuple[str, ...] = (
    "SECRET", "TOKEN", "PASSWORD", "PASSWD", "APIKEY", "API_KEY", "ACCESS_KEY",
    "PRIVATE_KEY", "CREDENTIAL", "AUTH", "SESSION", "COOKIE", "SIGNATURE",
)

#: Environment variable name prefixes worth reporting. Everything else is
#: dropped rather than redacted: a full ``os.environ`` is mostly the user's
#: shell and is where accidental disclosure comes from.
REPORTED_ENVIRONMENT: Tuple[str, ...] = (
    "SPACR_", "CUDA_", "NVIDIA_", "PYTORCH_", "OMP_", "MKL_", "QT_",
    "CONDA_", "VIRTUAL_ENV", "PYTHON", "PATH", "LD_LIBRARY_PATH", "DISPLAY",
    "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "HOME", "TMPDIR", "MPLBACKEND",
)

#: Distributions whose versions decide most spaCR bug reports.
REPORTED_PACKAGES: Tuple[str, ...] = (
    "spacr", "numpy", "pandas", "scipy", "scikit-image", "scikit-learn",
    "torch", "torchvision", "cellpose", "opencv-python-headless", "opencv-python",
    "matplotlib", "statsmodels", "PySide6", "tifffile", "zarr", "umap-learn",
    "shap", "anndata", "biopython", "ttkbootstrap",
)


@dataclass
class CrashReport:
    """Collected report sections and manifest, before serialisation.

    Kept as data rather than written straight to a zip so that a caller — a
    test, the Qt layer, a support script — can inspect or re-render it without
    a temporary file, and so that :func:`write_crash_report` has nothing in it
    but serialisation.

    :param created_utc: when the report was made, ISO-8601.
    :param run_id: the run it is about, empty when none could be identified.
    :param sections: file name to text, exactly as it will appear in the zip.
    :param manifest: what was collected, what was not, and why. Every entry
        that failed carries the exception text; every entry that was truncated
        carries the byte counts.
    """

    created_utc: str
    run_id: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def problems(self) -> List[str]:
        """Names of the sections that *failed* while being gathered.

        Distinct from :attr:`omitted`: a section that had nothing to collect is
        not a failure, and the two are listed separately.

        :returns: section names, in collection order. Empty on a clean report.
        """
        return [name for name, entry in self.manifest.get("sections", {}).items()
                if entry.get("status") == "failed"]

    @property
    def omitted(self) -> List[str]:
        """Names of the sections that had nothing to collect.

        No settings file was named, the project has no database, the run wrote
        no warnings. Each is a fact about the invocation rather than a failure,
        and each is still listed, because the report recipient must be able to tell
        "there were no warnings" from "the warnings were not gathered".

        :returns: section names, in collection order.
        """
        return [name for name, entry in self.manifest.get("sections", {}).items()
                if entry.get("status") == "empty"]

    def summary(self) -> str:
        """One short block naming the run, the versions and what is missing.

        This is also ``summary.txt``, the first file in the bundle, because a
        report recipient opening a zip should not have to guess which file to read
        first.

        :returns: the summary text.
        """
        lines = [
            "spaCR crash report",
            f"created      {self.created_utc}",
            f"run id       {self.run_id or '(none identified)'}",
            f"spacr        {self.manifest.get('spacr_version', 'unknown')}",
            f"python       {platform.python_version()}",
            f"platform     {platform.platform()}",
        ]
        doctor = self.manifest.get("doctor_summary")
        if doctor:
            counts = ", ".join(f"{k} {v}" for k, v in sorted(doctor.items()) if v)
            lines.append(f"doctor       {counts or 'no rows'}")
        contents = ", ".join(sorted(self.sections))
        lines.append(f"contents     {contents}")
        if self.problems:
            lines.append("")
            lines.append("FAILED to gather (see manifest.json for the error):")
            lines.extend(f"  - {name}" for name in self.problems)
        if self.omitted:
            lines.append("")
            lines.append("Nothing to gather (not a failure):")
            lines.extend(f"  - {name}" for name in self.omitted)
        return "\n".join(lines) + "\n"


def _utcnow() -> str:
    """Return the current UTC instant as ISO-8601, seconds resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _collect(report: CrashReport, name: str,
             gather: Callable[[], Optional[str]]) -> None:
    """Run one collector and record what happened, whatever happens.

    The whole module's failure policy in one function. ``gather`` returns the
    section text, or ``None`` for "there was nothing to collect", and may raise
    anything at all -- a locked database, a permission error, an optional
    dependency that fails to import, a bug in a collector. None of those may
    stop the report, and none of them may vanish either, so the exception text
    lands in the manifest under the section it belongs to.

    :param report: the report being built; mutated in place.
    :param name: the file name this section will have in the bundle.
    :param gather: the collector.
    """
    entry: Dict[str, Any] = {"status": "ok"}
    try:
        text = gather()
    except BaseException as exc:            # noqa: BLE001 - see the docstring
        if isinstance(exc, KeyboardInterrupt):
            raise
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
    else:
        if text is None:
            entry["status"] = "empty"
        else:
            report.sections[name] = text
            entry["bytes"] = len(text.encode("utf-8", "replace"))
    report.manifest.setdefault("sections", {})[name] = entry


def _tail(path: Path, limit: int) -> Tuple[str, Dict[str, Any]]:
    """Return the last ``limit`` bytes of ``path`` and what that cost.

    Reads from the end rather than loading the file, because the file this is
    used on is the one that grows without bound.

    :param path: file to read.
    :param limit: maximum bytes to keep.
    :returns: ``(text, facts)``, where ``facts`` records the total size and how
        much was dropped, so a truncated section is never mistaken for a short
        one.
    """
    total = path.stat().st_size
    with open(path, "rb") as handle:
        if total > limit:
            handle.seek(total - limit)
            handle.readline()
        payload = handle.read()
    text = payload.decode("utf-8", "replace")
    facts = {"total_bytes": total, "kept_bytes": len(payload)}
    if total > len(payload):
        facts["dropped_bytes"] = total - len(payload)
        facts["truncated"] = True
    return text, facts


def _redacted_environment() -> Dict[str, str]:
    """Return the reportable environment, with credential-shaped names hidden.

    :returns: variable name to value, with a redacted value replaced by
        ``'<redacted>'``.
    """
    out: Dict[str, str] = {}
    for name, value in sorted(os.environ.items()):
        if not any(name.startswith(prefix) or name == prefix
                   for prefix in REPORTED_ENVIRONMENT):
            continue
        upper = name.upper()
        if any(token in upper for token in REDACT_TOKENS):
            out[name] = "<redacted>"
        else:
            out[name] = value
    return out


def _package_versions() -> Dict[str, str]:
    """Return installed versions of the packages that decide bug reports.

    :returns: distribution name to version, ``'not installed'`` for the ones
        that are absent -- which is itself the answer in about a third of the
        reports this module exists for.
    """
    from importlib.metadata import PackageNotFoundError, version

    out: Dict[str, str] = {}
    for name in REPORTED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
        except Exception as exc:            # noqa: BLE001 - reporting only
            out[name] = f"unreadable: {exc}"
    return out


def _gpu_facts() -> Dict[str, Any]:
    """Return what torch says about the GPU, without allocating on it.

    Deliberately does not probe: :mod:`spacr.doctor` does that, under its own
    ``--probe-gpu``, and a crash report that allocated GPU memory could fail
    for a reason that has nothing to do with the crash being reported.

    :returns: a dict that always has ``torch``; the rest depends on what could
        be read.
    """
    facts: Dict[str, Any] = {}
    try:
        import torch
    except Exception as exc:                # noqa: BLE001 - reporting only
        return {"torch": f"not importable: {exc}"}
    facts["torch"] = getattr(torch, "__version__", "unknown")
    facts["cuda_built"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:                # noqa: BLE001 - reporting only
        facts["cuda_available"] = f"unreadable: {exc}"
        return facts
    facts["cuda_available"] = available
    if not available:
        return facts
    try:
        facts["devices"] = [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "capability": ".".join(str(p) for p in
                                    torch.cuda.get_device_capability(i))}
            for i in range(torch.cuda.device_count())]
    except Exception as exc:                # noqa: BLE001 - reporting only
        facts["devices"] = f"unreadable: {exc}"
    return facts


def _spacr_version() -> str:
    """Return the running spaCR's version string, or ``'unknown'``."""
    try:
        import spacr

        return str(getattr(spacr, "__version__", "") or "unknown")
    except Exception:                       # noqa: BLE001 - reporting only
        return "unknown"


def _spacr_location() -> str:
    """Return the directory ``import spacr`` actually resolves to.

    The single most valuable line in a spaCR bug report: this repository is
    checked out more than once on most developer machines, and an editable
    install points at exactly one of them.
    """
    try:
        import spacr

        return str(Path(getattr(spacr, "__file__", "")).resolve().parent)
    except Exception as exc:                # noqa: BLE001 - reporting only
        return f"unreadable: {exc}"


def find_last_run_id(problems: Optional[List[str]] = None) -> str:
    """Return the id of the most recent run this machine logged.

    The run ids live one JSONL file per run under
    :func:`spacr.runctx.runs_log_dir`, so "the last run" is the newest file
    there. An active run wins over it: inside a ``run_context`` the crash being
    reported is *this* run, not the one before it.

    :param problems: optional list that anything which went wrong on the way to
        the answer is appended to. It distinguishes "no run was found" from
        "the run directory could not be read". :func:`collect` passes one and
        puts it in the manifest.
    :returns: the run id, or ``''`` when none could be identified. Never raises
        -- a missing log directory is an ordinary answer here.
    """
    notes = problems if problems is not None else []
    try:
        from .runctx import current_run_id, runs_log_dir
    except Exception as exc:                # noqa: BLE001 - reporting only
        notes.append(f"spacr.runctx could not be imported: {exc}")
        return ""
    try:
        active = current_run_id()
    except Exception as exc:                # noqa: BLE001 - reporting only
        notes.append(f"the active run id could not be read: {exc}")
        active = ""
    if active and active != "no-run":
        return str(active)
    try:
        folder = Path(runs_log_dir())
        candidates = sorted(folder.glob("*.jsonl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception as exc:                # noqa: BLE001 - reporting only
        notes.append(f"the run log directory could not be listed: {exc}")
        return ""
    if not candidates:
        notes.append(f"no run has logged anything under {folder}")
        return ""
    return candidates[0].stem


def _doctor_sections(report: CrashReport, checkout: Optional[Path],
                     db: Optional[Path], settings: Optional[Path],
                     app: str, probe_gpu: bool) -> None:
    """Run :mod:`spacr.doctor` once and add both of its renderings.

    Add human-readable text and structured JSON from the same ``run_checks``
    result so the two representations cannot disagree.

    ``run_checks`` itself goes *inside* :func:`_collect` rather than beside it.
    It already turns a check that raises into an ``ERROR`` row, but importing
    :mod:`spacr.doctor`, building its :class:`~spacr.doctor.Context` and
    resolving the checkout are all outside that guarantee — and a report is
    exactly the situation in which an import fails.
    """
    held: Dict[str, Any] = {}

    def gather_text() -> str:
        """Run Doctor, retain its rows and summary, and return formatted text."""
        from dataclasses import asdict

        from . import doctor

        context = doctor.Context(
            checkout=Path(checkout) if checkout else Path.cwd(),
            db=Path(db) if db else None,
            settings=Path(settings) if settings else None,
            app=str(app or ""),
            probe_gpu=bool(probe_gpu),
        )
        results = doctor.run_checks(context)
        held["rows"] = [asdict(r) for r in results]
        report.manifest["doctor_summary"] = doctor.summarize(results)
        return doctor.format_report(results)

    def gather_json() -> str:
        """Return retained Doctor rows as JSON, or raise if collection failed."""
        if "rows" not in held:
            raise RuntimeError(
                "the doctor checks did not run; see the doctor.txt entry in "
                "this manifest for the error that stopped them")
        return json.dumps(held["rows"], indent=2) + "\n"

    _collect(report, "doctor.txt", gather_text)
    _collect(report, "doctor.json", gather_json)


def _run_log_section(report: CrashReport, run_id: str) -> None:
    """Add the per-run JSONL for ``run_id``, tail-capped and recorded."""

    def gather() -> Optional[str]:
        """Return the capped run-log tail and record its path and size facts."""
        from .runctx import run_log_path

        path = Path(run_log_path(run_id))
        if not path.is_file():
            return None
        text, facts = _tail(path, MAX_RUN_LOG_BYTES)
        report.manifest.setdefault("run_log", {}).update(
            {"path": str(path), **facts})
        return text

    _collect(report, f"run-{run_id}.jsonl", gather)


def _run_summary_section(report: CrashReport, run_id: str) -> None:
    """Add the run's warnings and errors, read back through :mod:`spacr.runctx`.

    The raw JSONL is already in the bundle; this is the same run read at
    ``WARNING`` and above through :func:`spacr.runctx.read_run_log`, because
    the first question anyone asks of a crash report is "what went wrong", and
    scrolling a JSONL to find out is a tax on the person helping.
    """

    def gather() -> Optional[str]:
        """Return WARNING-and-higher run records as readable text, if any."""
        from .runctx import read_run_log

        records = read_run_log(run_id, level="WARNING")
        if not records:
            return None
        lines = []
        for record in records:
            lines.append(f"{record.get('utc', '')} {record.get('level', ''):8} "
                         f"{record.get('logger', '')} — {record.get('message', '')}")
            if record.get("traceback"):
                lines.append(str(record["traceback"]).rstrip())
        return "\n".join(lines) + "\n"

    _collect(report, "run-problems.txt", gather)


def _settings_section(report: CrashReport, settings: Optional[Path],
                      values: Optional[Mapping[str, Any]]) -> None:
    """Add the settings, from an explicit mapping or from the file named.

    A crash without its settings is unreproducible, and "the defaults" is
    almost never what was run.
    """

    def gather() -> Optional[str]:
        """Return explicit or recorded settings as JSON, or ``None`` if absent."""
        if values is not None:
            return json.dumps(_jsonable(dict(values)), indent=2,
                              sort_keys=True) + "\n"
        if settings is None:
            return None
        path = Path(settings)
        if not path.is_file():
            report.manifest.setdefault("settings", {})["missing"] = str(path)
            return None
        from .resume import read_recorded_settings

        recorded = read_recorded_settings(str(path))
        report.manifest.setdefault("settings", {})["source"] = str(path)
        return json.dumps(_jsonable(dict(recorded)), indent=2,
                          sort_keys=True) + "\n"

    _collect(report, "settings.json", gather)


def _run_status_section(report: CrashReport, db: Optional[Path]) -> None:
    """Add :func:`spacr.errors.read_run_status` for the project database.

    Whether the last run *finished* is a different question from whether
    something crashed just now, and it is the one that matters when a user
    reports numbers rather than a traceback: a run that stopped two thirds of
    the way through produces a plate that looks exactly like a complete one.
    """

    def gather() -> Optional[str]:
        """Return project run-status rows as JSON when a database provides them."""
        if db is None:
            return None
        path = Path(db)
        if not path.is_file():
            return None
        from .errors import read_run_status

        records = read_run_status(path)
        if not records:
            return None
        return json.dumps(_jsonable(records), indent=2) + "\n"

    _collect(report, "run-status.json", gather)


def _main_log_section(report: CrashReport) -> None:
    """Add the tail of the rotating ``spacr.log``."""

    def gather() -> Optional[str]:
        """Return the capped main-log tail and record its path and size facts."""
        from .logging_util import log_path

        path = Path(log_path())
        if not path.is_file():
            return None
        text, facts = _tail(path, MAX_LOG_BYTES)
        report.manifest.setdefault("main_log", {}).update(
            {"path": str(path), **facts})
        return text

    _collect(report, "spacr.log", gather)


def _jsonable(value: Any) -> Any:
    """Return ``value`` with anything ``json`` cannot serialise turned to text.

    A settings dict routinely holds ``Path``, ``numpy`` scalars and a torch
    device. Refusing to serialise them would drop the whole section; showing
    their ``repr`` keeps every key the user actually set.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def collect(run_id: Optional[str] = None, *,
            db: Optional[os.PathLike] = None,
            settings: Optional[os.PathLike] = None,
            settings_values: Optional[Mapping[str, Any]] = None,
            checkout: Optional[os.PathLike] = None,
            app: str = "",
            exception: Optional[BaseException] = None,
            note: str = "",
            probe_gpu: bool = False) -> CrashReport:
    """Collect the crash-report sections into a :class:`CrashReport`.

    Composed from :mod:`spacr.doctor`, :mod:`spacr.runctx`, :mod:`spacr.errors`
    and :mod:`spacr.logging_util` rather than collected again, so the report
    cannot disagree with what those tools say when run directly.

    :param run_id: the run to include. Defaults to :func:`find_last_run_id`,
        which prefers an active ``run_context`` over the newest log on disk.
    :param db: project database, e.g. ``plate1/measurements/measurements.db``.
        Adds the doctor's database checks and the run-status stamps.
    :param settings: settings csv/json that was run.
    :param settings_values: the settings as a mapping, when the caller has them
        in memory. Takes precedence over ``settings``.
    :param checkout: directory the user believes they are editing; defaults to
        the current one; the report states whether the running spaCR is
        imported from it.
    :param app: app key for the settings file (``mask``, ``measure``, ...).
    :param exception: the exception being reported, if any. Its traceback goes
        in verbatim.
    :param note: free text from the user — what they were doing.
    :param probe_gpu: let the doctor allocate on the GPU to prove it works.
        Off by default: a probe that fails would add a failure that is not the
        one being reported.
    :returns: the assembled :class:`CrashReport`. Never raises; a section that
        could not be gathered is named in :attr:`CrashReport.problems`.
    """
    run_notes: List[str] = []
    resolved_run = str(run_id) if run_id else find_last_run_id(run_notes)
    report = CrashReport(created_utc=_utcnow(), run_id=resolved_run)
    if run_notes:
        report.manifest["run_id_notes"] = run_notes
    report.manifest.update({
        "spacr_version": _spacr_version(),
        "spacr_location": _spacr_location(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "max_log_bytes": MAX_LOG_BYTES,
        "max_run_log_bytes": MAX_RUN_LOG_BYTES,
    })

    if exception is not None:
        _collect(report, "traceback.txt", lambda: "".join(
            traceback.format_exception(type(exception), exception,
                                       exception.__traceback__)))
    if note:
        _collect(report, "note.txt", lambda: str(note).rstrip() + "\n")

    _collect(report, "versions.json", lambda: json.dumps({
        "spacr": _spacr_version(),
        "spacr_location": _spacr_location(),
        "python": platform.python_version(),
        "python_full": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
        "gpu": _gpu_facts(),
    }, indent=2, sort_keys=True) + "\n")

    def environment() -> str:
        """Return redacted environment JSON and record every redacted name."""
        redacted = _redacted_environment()
        report.manifest["redacted_environment"] = sorted(
            name for name, value in redacted.items() if value == "<redacted>")
        return json.dumps(redacted, indent=2, sort_keys=True) + "\n"

    _collect(report, "environment.json", environment)

    _doctor_sections(report, Path(checkout) if checkout else None,
                     Path(db) if db else None,
                     Path(settings) if settings else None, app, probe_gpu)
    _settings_section(report, Path(settings) if settings else None,
                      settings_values)
    _run_status_section(report, Path(db) if db else None)
    if resolved_run:
        _run_log_section(report, resolved_run)
        _run_summary_section(report, resolved_run)
    else:
        report.manifest.setdefault("sections", {})["run log"] = {
            "status": "empty",
            "error": "no run id could be identified; pass --run-id, or run "
                     "inside spacr.runctx.run_context so one is minted",
        }
    _main_log_section(report)
    return report


def write_crash_report(destination: Optional[os.PathLike] = None,
                       **kwargs: Any) -> str:
    """Gather a report and write it as one zip.

    :param destination: where to write. A directory gets
        ``spacr-crashreport-<run id or timestamp>.zip`` inside it; anything
        else is used as the file name. Defaults to ``<log
        dir>/spacr-crashreport-<...>.zip``, the spaCR log directory.
    :param kwargs: passed straight to :func:`collect`.
    :returns: the absolute path of the file written.
    :raises OSError: only when the destination itself cannot be written, This
        failure is raised rather than recorded, so the caller is never given
        the path of a report that is not on disk.

    Example:
        .. code-block:: python

            from spacr.crashreport import write_crash_report
            path = write_crash_report(db='plate1/measurements/measurements.db')
            print(f'attach {path} to the issue')
    """
    report = collect(**kwargs)
    target = _resolve_destination(destination, report)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.txt", report.summary())
        for name in sorted(report.sections):
            archive.writestr(name, report.sections[name])
        archive.writestr("manifest.json",
                         json.dumps(_jsonable(report.manifest), indent=2,
                                    sort_keys=True) + "\n")
    target.write_bytes(payload.getvalue())
    return str(target.resolve())


def _resolve_destination(destination: Optional[os.PathLike],
                         report: CrashReport) -> Path:
    """Turn the ``destination`` argument into the file to write."""
    stamp = report.run_id or report.created_utc.replace(":", "").replace("-", "")
    name = f"spacr-crashreport-{stamp}.zip"
    if destination is None:
        from .logging_util import log_dir

        return Path(log_dir()) / name
    target = Path(destination).expanduser()
    if target.is_dir():
        return target / name
    return target


def report_exception(exception: BaseException,
                     destination: Optional[os.PathLike] = None,
                     **kwargs: Any) -> str:
    """Write a crash report about ``exception`` and return where it went.

    The call to make from an ``except`` block that is about to re-raise.

    :param exception: the exception being reported.
    :param destination: as :func:`write_crash_report`.
    :param kwargs: passed to :func:`collect`.
    :returns: the path written, or ``''`` when even writing the file failed --
        which is reported on stderr rather than raised, because this is called
        from a failure path and must never replace the user's exception with
        one of its own.
    """
    try:
        return write_crash_report(destination, exception=exception, **kwargs)
    except Exception as exc:                # noqa: BLE001 - see the docstring
        print(f"spaCR could not write a crash report: {exc}", file=sys.stderr)
        return ""


def install_excepthook(destination: Optional[os.PathLike] = None,
                       **kwargs: Any) -> Callable:
    """Write a crash report on any unhandled exception, then behave as before.

    Chains rather than replaces: the previous ``sys.excepthook`` still runs, so
    the traceback the user is used to seeing still appears, with one line after
    it saying where the bundle is. With affirmative saved consent and automatic
    reporting enabled, the crash's :func:`public_summary` is processed on a
    background thread under the configured AI and reporting policy; the bundle
    is never sent.

    :param destination: as :func:`write_crash_report`.
    :param kwargs: passed to :func:`collect`.
    :returns: the hook that was installed, so a caller can restore
        ``sys.excepthook`` to what it was.
    """
    previous = sys.excepthook

    def hook(kind, value, tb) -> None:
        """Report, then delegate to the hook that was installed before."""
        if not issubclass(kind, KeyboardInterrupt):
            path = report_exception(value, destination, **kwargs)
            _report_anonymously(value, str(kwargs.get("app", "") or ""))
            if path:
                print(f"\nspaCR wrote a crash report to {path}\n"
                      f"Attach it to a bug report — it holds the log, the "
                      f"settings, the versions and this run.", file=sys.stderr)
        previous(kind, value, tb)

    sys.excepthook = hook
    return hook


#: Where anonymous reports and hosted diagnoses are posted. The relay holds
#: the GitHub App key and files the issue; nothing secret ships with spaCR.
REPORT_RELAY_URL = "https://spacr-first-responder.einar-olafsson.workers.dev"

#: Environment variable that points reports at another relay, for testing a
#: relay under development. Only ``https://`` addresses and a local
#: ``http://127.0.0.1`` or ``http://localhost`` one are accepted.
REPORT_RELAY_ENV = "SPACR_REPORT_RELAY"

#: Environment variable that forbids every report and hosted diagnosis.
#: Any value other than empty, ``0``, ``false``, ``no`` or ``off`` sets it.
NO_NETWORK_ENV = "SPACR_NO_NETWORK"

#: The summary's schema version, sent as ``schema_version``.
PUBLIC_SUMMARY_SCHEMA = 1

#: Every key a summary may carry, in the order it is built. ``user_note``,
#: ``dry_run``, ``ai_summary``, ``summarize`` and ``kind`` are present only
#: when given. A frame may also carry ``code``, its source line, but only for
#: a frame inside the spaCR package and only when code context is chosen.
PUBLIC_SUMMARY_KEYS: Tuple[str, ...] = (
    "schema_version", "install_id", "spacr_version", "os", "python", "qt",
    "torch", "gpu", "module", "exception_type", "message", "frames",
    "user_note", "dry_run", "ai_summary", "summarize", "kind", "screen",
    "error_text", "chain", "settings", "inputs", "environment", "run",
    "log_tail", "cause",
)

#: The relay refuses a body larger than this many bytes.
MAX_SUMMARY_BYTES = 32 * 1024

#: The longest scrubbed exception message sent, in characters.
MAX_MESSAGE_CHARS = 300

#: The longest message of a user error, which lists every settings problem.
MAX_USER_ERROR_MESSAGE_CHARS = 800

#: The longest scrubbed console error text sent with a report.
MAX_ERROR_TEXT_CHARS = 1500

#: The longest screen title sent with a report.
MAX_SCREEN_CHARS = 60

#: Public spaCR addresses an AI answer may keep through the scrubber.
PUBLIC_DOCS_URLS: Tuple[str, ...] = (
    "https://einarolafsson.github.io/spacr",
    "https://github.com/EinarOlafsson/spacr",
)

#: The most call-stack frames sent; the innermost ones are kept.
MAX_FRAMES = 40

#: The longest note a user may add to a report, in characters.
MAX_NOTE_CHARS = 1000

#: The longest question sent with a hosted diagnosis, in characters.
MAX_QUESTION_CHARS = 2000

#: The longest AI-written summary a report carries, in characters.
MAX_AI_SUMMARY_CHARS = 2000

#: The length the AI is asked to keep its report summary under.
AI_SUMMARY_TARGET_CHARS = 1200

#: Seconds the user's own AI may take to write a report summary before the
#: report is filed without one.
AI_SUMMARY_TIMEOUT_S = 30.0

#: The most reports waiting to be sent; the oldest go first.
QUEUE_LIMIT = 20

#: Seconds a waiting report is kept before it is deleted unsent: 7 days.
QUEUE_RETENTION_S = 7 * 24 * 3600

#: Seconds a report may take to post before it waits for a later try.
SEND_TIMEOUT_S = 8.0

#: Seconds a hosted diagnosis may take; a model needs longer than a post.
DIAGNOSE_TIMEOUT_S = 60.0

#: Whether a profile that never chose sends anonymised reports.
DEFAULT_ANONYMOUS_REPORTS = True

#: The longest source line a frame's ``code`` carries, in characters.
MAX_CODE_CHARS = 160

#: A failure that is spaCR's own fault: an unexpected exception.
KIND_FAULT = "fault"

#: A failure the user can fix: a settings, validation or settings-path error.
KIND_USER_ERROR = "user_error"

#: The "AI Error report" choices, ``(key, label)``, in the order offered.
ERROR_REPORT_MODES: Tuple[Tuple[str, str], ...] = (
    ("all", "Summarize and send all errors"),
    ("all_code", "Summarize and send all errors, with code context"),
    ("faults", "Summarize all errors, send only faults"),
    ("summarize", "Summarize all errors, send nothing"),
    ("send_all", "Send all errors, no AI summary"),
    ("send_faults", "Send only faults, no AI summary"),
    ("off", "Off"),
)

#: The choice for a profile that never made one.
DEFAULT_ERROR_REPORT_MODE = "all"

#: The fixed question a failure is explained with.
EXPLAIN_QUESTION = (
    "Explain the likely cause of this spaCR error in one or two sentences, "
    "say where in spaCR it happens, then give a short fix the user can "
    "apply. Do not include any personal data.")

#: Anonymised reports the relay accepts from one install per UTC day.
REPORTS_PER_DAY = 5

#: Hosted-AI questions per install per UTC day answered by the best model.
AI_BEST_PER_DAY = 5

#: Further questions per install per UTC day answered by the lighter model.
AI_LIGHT_PER_DAY = 10

#: Outcome of a report the relay accepted.
REPORT_SENT = "sent"

#: Outcome of a report kept to try again: no network, a server error, a
#: timeout, or the relay's daily quota.
REPORT_QUEUED = "queued"

#: Outcome of a report the relay rejected as malformed; it is not retried.
REPORT_DROPPED = "dropped"

#: Outcome when reports are off or the network is forbidden; nothing is sent.
REPORT_DISABLED = "disabled"

_STATE_LOCK = threading.RLock()
_SEND_LOCK = threading.Lock()

_GPU_CLASSES = ("nvidia", "amd", "apple", "none", "unknown")

_FILE_EXTENSIONS = (
    "tif", "tiff", "png", "jpg", "jpeg", "bmp", "gif", "czi", "nd2", "lif",
    "lsm", "oib", "oif", "ims", "vsi", "ome", "zarr", "h5", "hdf5", "npy",
    "npz", "csv", "tsv", "xlsx", "xls", "json", "yaml", "yml", "txt", "log",
    "db", "sqlite", "sqlite3", "pkl", "pickle", "pt", "pth", "onnx", "ckpt",
    "mat", "py", "ipynb", "zip", "tar", "gz", "mp4", "avi", "mov", "pdf",
    "docx", "pptx", "svg", "parquet", "feather", "dat", "bin", "raw", "fcs",
)

_TOP_LEVEL_DOMAINS = (
    "com", "org", "net", "edu", "gov", "mil", "int", "io", "ai", "dev",
    "app", "cloud", "local", "lan", "internal", "intra", "corp", "home",
    "uk", "us", "ca", "de", "fr", "se", "is", "no", "dk", "fi", "nl", "be",
    "ch", "at", "it", "es", "pt", "jp", "cn", "kr", "in", "au", "nz", "br",
    "eu", "ru", "pl", "cz",
)

_URL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>`]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{12}\b")
_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{16,}\b")
_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9+/_\-])(?=[A-Za-z0-9+/_\-]*[0-9])"
    r"(?=[A-Za-z0-9+/_\-]*[a-z])(?=[A-Za-z0-9+/_\-]*[A-Z])"
    r"[A-Za-z0-9+/_\-]{24,}={0,2}")
_IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6_RE = re.compile(
    r"(?<![\w:])(?=[0-9A-Fa-f:]*::|(?:[0-9A-Fa-f]{1,4}:){4})"
    r"[0-9A-Fa-f:]{6,}(?![\w:])")
_HOST_RE = re.compile(
    r"(?i)\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+(?:"
    + "|".join(_TOP_LEVEL_DOMAINS) + r")\b(?!\.\w)")
_WINDOWS_PATH_RE = re.compile(r"(?<![\w<])[A-Za-z]:[\\/][^\s'\"<>|]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\s'\"<>|]+")
_ROOTED_WINDOWS_PATH_RE = re.compile(r"(?<![\w<.:\\/])\\[^\s'\"<>\\|][^\s'\"<>|]*")
_POSIX_PATH_RE = re.compile(r"(?<![\w<.:/])~?/[^\s'\"<>/][^\s'\"<>]*")
_FILE_NAME_RE = re.compile(
    r"(?i)(?<![\w<])[\w\-.]*\w\.(?:" + "|".join(_FILE_EXTENSIONS)
    + r")(?![\w])")
_FRAME_LINE_RE = re.compile(
    r'^\s*File\s+"(?P<file>[^"]*)",\s*line\s+(?P<line>\d+)'
    r'(?:,\s*in\s+(?P<function>.+?))?\s*$')
_EXCEPTION_LINE_RE = re.compile(
    r"^(?P<type>[A-Za-z_][\w.]*)(?::\s?(?P<message>.*))?$")
_SAFE_TOKEN_RE = re.compile(r"[^\w.\-<>]")


class RelayRefused(RuntimeError):
    """The relay did not answer a hosted diagnosis.

    :param reason: ``"quota"`` when a free daily limit is used up,
        ``"network"`` when the relay could not be reached or the network is
        forbidden, and ``"error"`` for any other refusal.
    :param detail: what the relay or the network said, for the log.
    :param scope: for a quota refusal, whose limit it was: ``"install"``
        (this install's questions), ``"global"`` (spaCR's shared free
        capacity) or ``"ip"``.
    :param resets_at: when the limit resets, as an aware UTC datetime;
        defaults to the next 00:00 UTC.

    The values are kept on the exception as ``reason``, ``scope`` and
    ``resets_at``.
    """

    def __init__(self, reason: str, detail: str = "", scope: str = "",
                 resets_at: Optional[datetime] = None):
        """Keep the reason, scope and reset time, and the detail as the message."""
        self.reason = str(reason or "error")
        self.scope = str(scope or "")
        self.resets_at = resets_at or next_reset_utc()
        super().__init__(str(detail or self.reason))


def next_reset_utc(now: Optional[datetime] = None) -> datetime:
    """Return the next 00:00 UTC, when the relay's daily allowances reset.

    :param now: the moment to count from; defaults to now.
    :returns: an aware UTC datetime.
    """
    from datetime import timedelta

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    midnight = now.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(days=1)


def _parse_reset(value: Any) -> Optional[datetime]:
    """Read the relay's ``resets_at`` (ISO-8601 or epoch seconds) as UTC."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), timezone.utc)
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _count(value: Any) -> int:
    """Return an allowance count as a non-negative int, ``0`` if unreadable."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def relay_url() -> str:
    """Return the relay address reports are posted to, without a trailing slash.

    :data:`REPORT_RELAY_ENV` overrides :data:`REPORT_RELAY_URL` when it holds
    an ``https://`` address, or an ``http://`` one on this computer.

    :returns: the base address.
    """
    override = os.environ.get(REPORT_RELAY_ENV, "").strip().rstrip("/")
    if override.startswith("https://") or re.match(
            r"^http://(?:127\.0\.0\.1|localhost)(?::\d+)?(?:/|$)", override):
        return override
    return REPORT_RELAY_URL


def network_allowed() -> bool:
    """Whether a report or a hosted diagnosis may use the network at all.

    :returns: ``False`` when :data:`NO_NETWORK_ENV` has a nonempty value other than
        ``0``, ``false``, ``no`` or ``off`` (ignoring case and surrounding
        whitespace), or spaCR runs in safe mode; ``True`` otherwise.
    """
    flag = os.environ.get(NO_NETWORK_ENV, "").strip().lower()
    if flag and flag not in ("0", "false", "no", "off"):
        return False
    from .logging_util import _in_safe_mode

    return not _in_safe_mode()


def _reports_dir() -> Path:
    """Return the folder anonymous-report state and the queue live in."""
    from .logging_util import _spacr_home

    return _spacr_home() / "anonymous_reports"


def _queue_dir() -> Path:
    """Return the folder of reports waiting to be sent."""
    return _reports_dir() / "queue"


def _read_state() -> Dict[str, Any]:
    """Return the saved switch, notice flag and install id, or ``{}``."""
    try:
        data = json.loads((_reports_dir() / "state.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(state: Mapping[str, Any]) -> None:
    """Save the state atomically, so a crash never leaves half a file."""
    folder = _reports_dir()
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"state.{os.getpid()}.{threading.get_ident()}.tmp"
    temporary.write_text(json.dumps(dict(state), indent=2), encoding="utf-8")
    os.replace(temporary, folder / "state.json")
    globals()["_ERROR_POLICY_CACHE"] = dict(state)


def _update_state(**values: Any) -> None:
    """Merge ``values`` into the saved state."""
    with _STATE_LOCK:
        state = _read_state()
        state.update(values)
        _write_state(state)


def error_report_mode() -> str:
    """Return the "AI Error report" choice, a key of :data:`ERROR_REPORT_MODES`.

    A profile that only ever turned the older on/off switch off reads as
    ``"off"``.

    :returns: the saved key, or :data:`DEFAULT_ERROR_REPORT_MODE`.
    """
    state = _read_state()
    mode = state.get("mode")
    if mode in dict(ERROR_REPORT_MODES):
        return str(mode)
    if state.get("enabled") is False:
        return "off"
    return DEFAULT_ERROR_REPORT_MODE


def set_error_report_mode(mode: str) -> None:
    """Save the "AI Error report" choice.

    A choice that sends nothing also deletes every report still waiting, so
    nothing queued before the change is sent after it.

    :param mode: a key of :data:`ERROR_REPORT_MODES`.
    :raises ValueError: for any other value.
    """
    if mode not in dict(ERROR_REPORT_MODES):
        raise ValueError(f"unknown error-report mode: {mode!r}")
    _update_state(mode=mode, enabled=mode_sends(KIND_USER_ERROR, mode)
                  or mode_sends(KIND_FAULT, mode))
    if not (mode_sends(KIND_FAULT, mode) or mode_sends(KIND_USER_ERROR, mode)):
        with _STATE_LOCK:
            for path in _queue_files():
                path.unlink(missing_ok=True)


def mode_explains(mode: Optional[str] = None) -> bool:
    """Whether a choice has the AI explain every failure in the console.

    :param mode: a key of :data:`ERROR_REPORT_MODES`; defaults to the saved
        one.
    :returns: ``True`` for the four "Summarize" choices.
    """
    mode = error_report_mode() if mode is None else mode
    return mode in ("all", "all_code", "faults", "summarize")


def mode_sends(kind: str, mode: Optional[str] = None) -> bool:
    """Whether a choice files a failure of ``kind`` as an issue.

    :param kind: :data:`KIND_FAULT` or :data:`KIND_USER_ERROR`.
    :param mode: a key of :data:`ERROR_REPORT_MODES`; defaults to the saved
        one.
    :returns: whether it is sent.
    """
    mode = error_report_mode() if mode is None else mode
    if mode in ("all", "all_code", "send_all"):
        return True
    if mode in ("faults", "send_faults"):
        return kind == KIND_FAULT
    return False


def mode_code_context(mode: Optional[str] = None) -> bool:
    """Whether a choice adds each spaCR frame's source line to a report.

    :param mode: a key of :data:`ERROR_REPORT_MODES`; defaults to the saved
        one.
    :returns: ``True`` only for ``"all_code"``.
    """
    mode = error_report_mode() if mode is None else mode
    return mode == "all_code"


def anonymous_reports_enabled() -> bool:
    """Whether any failure is sent as an anonymised report.

    :returns: ``False`` for the choices that send nothing.
    """
    mode = error_report_mode()
    return mode_sends(KIND_FAULT, mode) or mode_sends(KIND_USER_ERROR, mode)


def set_anonymous_reports_enabled(enabled: bool) -> None:
    """Turn reports on or off, as the first-run notice's switch does.

    On restores the default choice when the current one sends nothing;
    off is ``"off"``.

    :param enabled: whether reports may be sent.
    """
    if not enabled:
        set_error_report_mode("off")
    elif not anonymous_reports_enabled():
        set_error_report_mode(DEFAULT_ERROR_REPORT_MODE)


def _user_error_names() -> set:
    """Names of every exception class that is the user's to fix."""
    from .errors import ConfigurationError

    names = {"ConfigurationError"}
    pending = [ConfigurationError]
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            if sub.__name__ not in names:
                names.add(sub.__name__)
                pending.append(sub)
    return names


def classify_failure(exception: Optional[BaseException] = None,
                     traceback_text: str = "") -> str:
    """Return whether a failure is a fault or a user error.

    A :class:`spacr.errors.ConfigurationError`, or any of its subclasses, is
    a user error: a setting, a validation or a missing input the user can
    fix. Anything else is a fault.

    :param exception: the exception, when at hand.
    :param traceback_text: a formatted traceback otherwise.
    :returns: :data:`KIND_USER_ERROR` or :data:`KIND_FAULT`.
    """
    try:
        from .errors import ConfigurationError

        if isinstance(exception, ConfigurationError):
            return KIND_USER_ERROR
        if exception is None and traceback_text:
            kind, _message, _frames = _parse_traceback_text(traceback_text)
            if kind in _user_error_names():
                return KIND_USER_ERROR
    except Exception:
        pass
    return KIND_FAULT


def anonymous_report_notice_due() -> bool:
    """Whether the one-time notice about anonymised reports is still to show.

    :returns: ``True`` until :func:`mark_anonymous_report_notice_shown`.
    """
    return not bool(_read_state().get("notice_shown", False))


def mark_anonymous_report_notice_shown() -> None:
    """Record that the notice about anonymised reports has been shown."""
    _update_state(notice_shown=True)


def install_id() -> str:
    """Return this profile's random install id, creating it on first use.

    The id is a random UUID, not derived from the computer, the user or the
    network. The relay uses it only to share its free allowance fairly.

    :returns: the id as a string.
    """
    with _STATE_LOCK:
        state = _read_state()
        value = str(state.get("install_id", "") or "")
        if not _UUID_RE.fullmatch(value):
            value = str(uuid.uuid4())
            state["install_id"] = value
            try:
                _write_state(state)
            except OSError:
                pass
        return value


def _home_variants() -> List[str]:
    """Return this user's home folder as text, in each separator style."""
    try:
        home = str(Path.home())
    except Exception:
        return []
    if len(home) < 4:
        return []
    variants = {home, home.replace("\\", "/"), home.replace("/", "\\")}
    return sorted(variants, key=len, reverse=True)


def _identity_scrub(text: str) -> str:
    """Replace this computer's login and host names with placeholders."""
    try:
        from .qt.ai.issue_report import redact_identity

        return redact_identity(text)
    except Exception:
        pass
    names = []
    try:
        import getpass

        names.append((getpass.getuser(), "<USER>"))
    except Exception:
        pass
    try:
        import socket

        names.append((socket.gethostname().split(".", 1)[0], "<HOST>"))
    except Exception:
        pass
    for name, placeholder in names:
        if name and len(name) >= 3 and not name.isdigit():
            text = re.sub(r"(?<![A-Za-z0-9])" + re.escape(name)
                          + r"(?![A-Za-z0-9])", placeholder, text,
                          flags=re.IGNORECASE)
    return text


def scrub_public_text(text: Any, limit: int = MAX_MESSAGE_CHARS,
                      keep_docs_urls: bool = False) -> str:
    """Remove anything personal from a short text that will be published.

    Absolute paths (POSIX, Windows and network shares) become ``<PATH>``,
    file names with a data or code extension ``<FILE>``, e-mail addresses
    ``<EMAIL>``, web addresses ``<URL>``, IP addresses ``<IP>``, host names
    ``<HOST>``, this computer's login name ``<USER>``, and anything shaped
    like a key, token or long hexadecimal or base64 string ``<TOKEN>``. The
    redaction helpers of :mod:`spacr.qt.ai.issue_report` run first, and these
    stricter rules after them.

    :param text: the text; anything else is turned into text first.
    :param limit: the longest result, in characters; a longer one is cut and
        ends in ``…``.
    :param keep_docs_urls: keep the public spaCR addresses in
        :data:`PUBLIC_DOCS_URLS`, for an AI answer that links the
        documentation; every other address still becomes ``<URL>``.
    :returns: the scrubbed text on one line.
    """
    value = str(text if text is not None else "")
    kept: List[str] = []
    if keep_docs_urls:
        def _hold(match: "re.Match") -> str:
            """Keep a public spaCR address aside; any other becomes <URL>."""
            url = match.group(0)
            if not any(url == base or url.startswith(base + "/")
                       or url.startswith(base + "#")
                       for base in PUBLIC_DOCS_URLS):
                return "<URL>"
            trail = ""
            while url and url[-1] in ".,;:!?)]*":
                trail = url[-1] + trail
                url = url[:-1]
            kept.append(url)
            return f"\x00{len(kept) - 1}\x00{trail}"
        value = _URL_RE.sub(_hold, value)
    value = _URL_RE.sub("<URL>", value)
    value = _EMAIL_RE.sub("<EMAIL>", value)
    try:
        from .qt.ai.issue_report import redact_secrets, strip_report_paths

        value = strip_report_paths(redact_secrets(value))
    except Exception:
        pass
    for home in _home_variants():
        value = re.sub(re.escape(home) + r"[^\s'\"<>]*", "<PATH>", value)
    value = _UNC_PATH_RE.sub("<PATH>", value)
    value = _WINDOWS_PATH_RE.sub("<PATH>", value)
    value = _ROOTED_WINDOWS_PATH_RE.sub("<PATH>", value)
    value = _POSIX_PATH_RE.sub("<PATH>", value)
    value = _UUID_RE.sub("<TOKEN>", value)
    value = _HEX_RE.sub("<TOKEN>", value)
    value = _KEY_RE.sub("<TOKEN>", value)
    value = _IPV4_RE.sub("<IP>", value)
    value = _IPV6_RE.sub("<IP>", value)
    value = _EMAIL_RE.sub("<EMAIL>", value)
    value = _HOST_RE.sub("<HOST>", value)
    value = _FILE_NAME_RE.sub("<FILE>", value)
    value = _identity_scrub(value)
    value = " ".join(value.split())
    for index, url in enumerate(kept):
        value = value.replace(f"\x00{index}\x00", url)
    value = value.replace("\x00", "")
    limit = max(1, int(limit))
    if len(value) > limit:
        value = value[:limit - 1].rstrip() + "…"
    return value


def _safe_token(text: Any, limit: int = 80) -> str:
    """Return a file or function name with only name-like characters kept."""
    value = _SAFE_TOKEN_RE.sub("", str(text or ""))
    value = _identity_scrub(value)
    return value[:limit]


def _library_roots() -> List[str]:
    """Return the folders whose files are spaCR's or an installed library's."""
    roots = {str(Path(__file__).resolve().parent)}
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix):
        if prefix:
            roots.add(str(Path(prefix)))
    return [root.replace("\\", "/").rstrip("/").lower() + "/"
            for root in roots if root]


def _public_frame_file(path: Any) -> str:
    """Return a frame's file as it is published: a basename, or a stand-in.

    A file of spaCR, of Python or of an installed library is named by its
    basename. Any other file is the user's own script, whose name can say
    what they work on, and is published as ``<user-code>``.
    """
    raw = str(path or "")
    if raw.startswith("<") and raw.endswith(">"):
        return raw if re.fullmatch(r"<[\w .\-]{1,40}>", raw) else "<string>"
    normal = raw.replace("\\", "/").lower()
    library = ("/site-packages/" in normal or "/dist-packages/" in normal
               or re.search(r"/lib/python\d", normal) is not None
               or any(normal.startswith(root) for root in _library_roots()))
    if not library:
        return "<user-code>"
    name = re.split(r"[\\/]+", raw)[-1]
    return _safe_token(name) or "<unknown>"


def _line_number(value: Any) -> int:
    """Return a frame's line number as a non-negative int, ``0`` if unreadable."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _is_spacr_file(path: Any) -> bool:
    """Whether a frame's file is part of the spaCR package itself."""
    normal = str(path or "").replace("\\", "/").lower()
    root = str(Path(__file__).resolve().parent).replace("\\", "/").lower()
    return (normal.startswith(root.rstrip("/") + "/")
            or re.search(r"/(?:site|dist)-packages/spacr/", normal) is not None)


def _frame(file: Any, function: Any, line: Any,
           code: Any = None) -> Dict[str, Any]:
    """Return one published frame: ``{file, function, line}``.

    ``code``, the frame's source line, is kept only for a file inside the
    spaCR package, scrubbed and cut to :data:`MAX_CODE_CHARS`; a user's own
    code never leaves the computer.
    """
    out: Dict[str, Any] = {"file": _public_frame_file(file),
                           "function": _safe_token(function) or "<unknown>",
                           "line": _line_number(line)}
    if code and _is_spacr_file(file):
        text = scrub_public_text(str(code).strip(), MAX_CODE_CHARS)
        if text:
            out["code"] = text
    return out


def _parse_traceback_text(text: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Read the last exception of a formatted traceback.

    :returns: ``(exception type, raw message, frames)``. Only the frames of
        the last ``Traceback`` block are kept, which belong to the exception
        that was finally raised.
    """
    lines = str(text or "").splitlines()
    start = 0
    for index, line in enumerate(lines):
        if line.strip().startswith("Traceback (most recent call last)"):
            start = index
    raw: List[List[Any]] = []
    tail: List[str] = []
    for line in lines[start:]:
        match = _FRAME_LINE_RE.match(line)
        if match is not None:
            raw.append([match.group("file"), match.group("function") or "",
                        match.group("line"), None])
            tail = []
            continue
        if line.startswith((" ", "\t")) or not line.strip():
            if tail and line.strip():
                tail.append(line.strip())
            elif raw and not tail and raw[-1][3] is None and line.strip():
                raw[-1][3] = line.strip()
            continue
        if line.strip().startswith("Traceback"):
            continue
        tail = [line.strip()]
    frames = [_frame(*entry) for entry in raw]
    if not tail:
        stripped = [line.strip() for line in lines if line.strip()]
        tail = stripped[-1:] if stripped else []
    if not tail:
        return "", "", frames
    head = tail[0]
    match = _EXCEPTION_LINE_RE.match(head)
    if match is None:
        return "", " ".join(tail), frames
    kind = match.group("type").rsplit(".", 1)[-1]
    message = " ".join([match.group("message") or ""] + tail[1:]).strip()
    return kind, message, frames


def _os_family() -> str:
    """Return ``linux``, ``windows``, ``macos`` or ``other``."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith(("win", "cygwin", "msys")):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "other"


def _distribution_version(name: str) -> str:
    """Return an installed distribution's version, or ``''``."""
    try:
        from importlib.metadata import version

        return _safe_token(version(name), 40)
    except Exception:
        return ""


def _gpu_class() -> str:
    """Return the kind of GPU spaCR can use, without importing torch.

    Torch is asked only when it is already imported. Otherwise an Apple
    Silicon Mac is ``apple``, a computer with ``nvidia-smi`` is ``nvidia``,
    and anything else is ``none``.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                hip = getattr(getattr(torch, "version", None), "hip", None)
                return "amd" if hip else "nvidia"
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and mps.is_available():
                return "apple"
            return "none"
        except Exception:
            return "unknown"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "apple"
    import shutil

    return "nvidia" if shutil.which("nvidia-smi") else "none"


def scrub_public_lines(text: Any, limit: int = MAX_AI_SUMMARY_CHARS,
                       keep_docs_urls: bool = False) -> str:
    """Scrub a text like :func:`scrub_public_text`, keeping its line breaks.

    Each line is scrubbed on its own, so a Markdown answer keeps its headings
    and numbered steps on lines of their own; runs of blank lines become one.

    :param text: the text; anything else is turned into text first.
    :param limit: the longest result, in characters; a longer one is cut and
        ends in ``…``.
    :param keep_docs_urls: as for :func:`scrub_public_text`.
    :returns: the scrubbed text, lines joined by ``\\n``.
    """
    raw = str(text if text is not None else "").replace("\r\n", "\n")
    lines: List[str] = []
    for line in raw.replace("\r", "\n").split("\n"):
        clean = scrub_public_text(line, max(1, len(line) + 1),
                                  keep_docs_urls) if line.strip() else ""
        if clean or (lines and lines[-1]):
            lines.append(clean)
    while lines and not lines[-1]:
        lines.pop()
    value = "\n".join(lines)
    limit = max(1, int(limit))
    if len(value) > limit:
        value = value[:limit - 1].rstrip() + "…"
    return value


_PATH_MENTION_RE = re.compile(
    r"\s*(?::\s*)?(?:['\"]?<(?:PATH|FILE)>['\"]?)(?:\s*\(\+\d+ more\))?")


def public_problem_lines(problems: Sequence[Any],
                         limit: int = MAX_USER_ERROR_MESSAGE_CHARS) -> List[str]:
    """Scrub settings problems into short lines that may be published.

    Paths and file names are taken out rather than replaced, so
    "src does not exist: /data/plate1" is sent as "src does not exist".

    :param problems: :class:`spacr.validate.Problem` objects, or texts.
    :param limit: the longest line, in characters.
    :returns: one line per problem, ``"setting: message"`` when the problem
        names a setting; duplicates and empty lines are dropped.
    """
    out: List[str] = []
    for problem in problems or ():
        setting = str(getattr(problem, "setting", "") or "")
        message = str(getattr(problem, "message", problem) or "")
        text = scrub_public_text(message, max(1, len(message) + 1))
        text = _PATH_MENTION_RE.sub("", text).replace("( ", "(").strip()
        if setting:
            name = _safe_token(setting, 60)
            text = f"{name}: {text}" if name and not text.startswith(
                name) else text
        text = " ".join(text.split())
        if len(text) > limit:
            text = text[:limit - 1].rstrip() + "…"
        if text and text not in out:
            out.append(text)
    return out


_SCREEN_RE = re.compile(r"[^A-Za-z0-9 &()+.,'\-]")


#: The most settings a report lists.
MAX_SETTINGS = 150

#: The longest setting value kept as text; longer ones become ``<text>``.
MAX_SETTING_TEXT = 40

#: The most exceptions of a chain a report lists, the raised one excluded.
MAX_CHAIN = 5

#: The longest console log excerpt sent, in characters, from its end.
MAX_LOG_TAIL_CHARS = 3000

#: The most console lines a report's log excerpt holds.
MAX_LOG_TAIL_LINES = 40

#: File extensions counted as images when a report describes the input.
_IMAGE_EXTENSIONS = (
    "tif", "tiff", "png", "jpg", "jpeg", "bmp", "czi", "nd2", "lif", "npy",
    "npz", "ims", "oib", "oif", "lsm", "dv", "zarr",
)

_NAME_LIKE_KEYS = re.compile(
    r"(?i)(name|experiment|plate|user|host|mail|title|label|note|comment|"
    r"prefix|suffix|tag|project|sample|well|author|owner|desc|token|key|"
    r"secret|pass|url|regex|pattern)")
_PATH_LIKE_KEYS = re.compile(
    r"(?i)(^src$|path|dir|folder|file|dst|dest|root|_db$|database|output|"
    r"input|location|mount)")
_PLAIN_VALUE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.+\-]{0,39}$")
_SETTING_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_MARKER_RE = re.compile(
    r"<(?:path: exists=(?:yes|no)|text|list: n=\d+|mapping: n=\d+|"
    r"[A-Za-z_]+)>")
_SAFE_FACT_RE = re.compile(r"[^A-Za-z0-9 ._()+\-]")


def _looks_like_a_path(value: str) -> bool:
    """Whether a setting's text names a file or folder."""
    return bool(re.search(r"[\\/]|^~|^[A-Za-z]:", value)) or bool(
        _FILE_NAME_RE.search(value))


def _path_marker(value: str) -> str:
    """``<path: exists=yes|no>`` for a path, without the path itself."""
    exists = os.path.exists(os.path.expanduser(value))
    return f"<path: exists={'yes' if exists else 'no'}>"


def _public_setting_value(key: str, value: Any, depth: int = 0) -> Any:
    """One setting's value as a report may carry it.

    Numbers, booleans and ``None`` stay. Text stays only when it is a short
    plain word (an option or a model name) under a key that does not hold
    names; a path becomes ``<path: exists=yes|no>`` and other text
    ``<text>``. Short lists are kept item by item.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") \
            else str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or _MARKER_RE.fullmatch(text):
            return text
        if _PATH_LIKE_KEYS.search(key) or _looks_like_a_path(text):
            return _path_marker(text)
        if (_NAME_LIKE_KEYS.search(key) or not _PLAIN_VALUE_RE.match(text)
                or scrub_public_text(text, 200) != text):
            return "<text>"
        return text
    if isinstance(value, (list, tuple)) and depth == 0:
        if len(value) > 16:
            return f"<list: n={len(value)}>"
        return [_public_setting_value(key, item, 1) for item in value]
    if isinstance(value, Mapping):
        return f"<mapping: n={len(value)}>"
    if isinstance(value, (list, tuple)):
        return f"<list: n={len(value)}>"
    return f"<{type(value).__name__}>"


def public_settings(settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The settings a run used, with nothing personal left in them.

    Setting names are kept as they are; values go through
    :func:`_public_setting_value`, so ``src`` reads ``<path: exists=no>``
    when the folder is missing, without the folder's name. At most
    :data:`MAX_SETTINGS` settings, in name order.

    :param settings: the run's settings.
    :returns: ``{name: value}``.
    """
    out: Dict[str, Any] = {}
    for key in sorted(str(k) for k in (settings or {}).keys()):
        if len(out) >= MAX_SETTINGS:
            break
        if not _SETTING_NAME_RE.match(key):
            continue
        out[key] = _public_setting_value(key, (settings or {}).get(key))
    return out


def _first_source(settings: Optional[Mapping[str, Any]]) -> str:
    """The first source folder named in a run's settings, or ``''``."""
    for key in ("src", "source", "images", "image_folder", "input_folder"):
        value = (settings or {}).get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        if isinstance(value, str) and value.strip():
            return os.path.expanduser(value.strip())
    return ""


def _image_shape(path: str) -> Tuple[List[int], str]:
    """Shape and dtype of one image, read from its header only."""
    lower = path.lower()
    if lower.endswith((".tif", ".tiff")):
        import tifffile

        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            return [int(n) for n in series.shape], str(series.dtype)
    if lower.endswith(".npy"):
        import numpy

        array = numpy.load(path, mmap_mode="r")
        return [int(n) for n in array.shape], str(array.dtype)
    from PIL import Image

    with Image.open(path) as image:
        bands = len(image.getbands())
        shape = [image.height, image.width] + ([bands] if bands > 1 else [])
        return shape, str(image.mode)


def public_input_facts(settings: Optional[Mapping[str, Any]],
                       limit: int = 5000) -> Dict[str, Any]:
    """What the run's source folder holds, without any name in it.

    Counts the image files directly in the folder (at most ``limit``
    entries are looked at), the extensions seen, and the shape, dtype and
    likely channel count of the first image, read from its header.

    :param settings: the run's settings.
    :param limit: the most folder entries looked at.
    :returns: ``{"images", "extensions", "shape", "dtype", "channels"}``,
        only the keys that could be found; ``{}`` without a readable folder.
    """
    source = _first_source(settings)
    if not source or not os.path.isdir(source):
        return {}
    images: List[str] = []
    extensions: Dict[str, int] = {}
    try:
        with os.scandir(source) as entries:
            for index, entry in enumerate(entries):
                if index >= limit:
                    break
                ext = entry.name.rsplit(".", 1)[-1].lower() \
                    if "." in entry.name else ""
                if ext in _IMAGE_EXTENSIONS and entry.is_file():
                    images.append(entry.path)
                    extensions[ext] = extensions.get(ext, 0) + 1
    except OSError:
        return {}
    facts: Dict[str, Any] = {"images": len(images),
                             "extensions": sorted(extensions)[:10]}
    if images:
        try:
            shape, dtype = _image_shape(sorted(images)[0])
            facts["shape"] = shape[:6]
            facts["dtype"] = _safe_token(dtype, 16)
            if len(shape) == 2:
                facts["channels"] = 1
            elif len(shape) >= 3:
                facts["channels"] = int(min(shape[-1], shape[0])
                                        if min(shape[-1], shape[0]) <= 16
                                        else 1)
        except Exception:
            pass
    return facts


def _install_type() -> str:
    """How spaCR was installed: editable, conda, installer, pip or unknown."""
    if getattr(sys, "frozen", False):
        return "installer"
    try:
        from importlib.metadata import distribution

        direct = distribution("spacr").read_text("direct_url.json") or ""
        if '"editable": true' in direct.replace("'", '"').lower():
            return "editable"
    except Exception:
        return "unknown"
    if os.path.isdir(os.path.join(sys.prefix, "conda-meta")):
        return "conda"
    return "pip"


def _safe_fact(text: Any, limit: int = 60) -> str:
    """A hardware or OS name with only plain characters kept."""
    return _identity_scrub(_SAFE_FACT_RE.sub("", str(text or "")))[:limit]


def public_environment() -> Dict[str, Any]:
    """The computer and libraries a report was made on, without any name.

    OS version, CPU architecture and cores, memory rounded to whole GB, the
    GPU's model and memory when torch already knows it, CUDA or MPS, the
    versions of spaCR's main libraries and how spaCR was installed.

    :returns: the facts that could be read.
    """
    env: Dict[str, Any] = {
        "os_version": _safe_fact(
            platform.mac_ver()[0] if sys.platform == "darwin"
            else platform.release(), 40),
        "arch": _safe_fact(platform.machine(), 20),
        "cores": int(os.cpu_count() or 0),
        "install": _install_type(),
    }
    try:
        import psutil

        env["ram_gb"] = int(round(psutil.virtual_memory().total / 2 ** 30))
    except Exception:
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            env["cuda"] = bool(torch.cuda.is_available())
            if env["cuda"]:
                props = torch.cuda.get_device_properties(0)
                env["gpu_model"] = _safe_fact(props.name, 60)
                env["vram_gb"] = int(round(props.total_memory / 2 ** 30))
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            env["mps"] = bool(mps is not None and mps.is_available())
        except Exception:
            pass
    versions = {}
    for key, dist in (("torch", "torch"), ("cellpose", "cellpose"),
                      ("numpy", "numpy"), ("scikit_image", "scikit-image"),
                      ("pyside6", "PySide6")):
        found = _distribution_version(dist)
        if found:
            versions[key] = found
    qtcore = sys.modules.get("PySide6.QtCore")
    if qtcore is not None:
        try:
            versions["qt"] = _safe_token(qtcore.qVersion(), 40)
        except Exception:
            pass
    env["versions"] = versions
    return env


_CHAIN_SPLIT_RE = re.compile(
    r"^(?:The above exception was the direct cause of the following "
    r"exception:|During handling of the above exception, another exception "
    r"occurred:)\s*$", re.M)


def _exception_chain(exception: Optional[BaseException] = None,
                     traceback_text: str = "") -> List[Dict[str, str]]:
    """The exceptions that led to the raised one, oldest first, scrubbed."""
    found: List[Tuple[str, str]] = []
    if exception is not None:
        seen = {id(exception)}
        node = exception.__cause__ or exception.__context__
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            found.insert(0, (type(node).__name__, str(node)))
            node = node.__cause__ or node.__context__
    elif traceback_text:
        for block in _CHAIN_SPLIT_RE.split(str(traceback_text))[:-1]:
            kind, message, _frames = _parse_traceback_text(block)
            if kind:
                found.append((kind, message))
    return [{"type": _safe_token(kind, 100),
             "message": scrub_public_text(message, MAX_MESSAGE_CHARS)}
            for kind, message in found[-MAX_CHAIN:] if kind]


def public_log_tail(text: str, lines: int = MAX_LOG_TAIL_LINES,
                    limit: int = MAX_LOG_TAIL_CHARS) -> str:
    """The console's last lines before a failure, scrubbed and cut.

    :param text: the console text.
    :param lines: the most lines kept, from the end.
    :param limit: the most characters kept, from the end.
    :returns: the scrubbed excerpt.
    """
    kept = [line for line in str(text or "").splitlines() if line.strip()]
    value = scrub_public_lines("\n".join(kept[-lines:]), 10 ** 6)
    if len(value) > limit:
        value = "…" + value[-(limit - 1):].split("\n", 1)[-1]
    return value


def _public_run(run: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The run's stage, seconds before failure and preview flag."""
    out: Dict[str, Any] = {}
    run = run or {}
    stage = _safe_token(run.get("stage", ""), 60)
    if stage:
        out["stage"] = stage
    try:
        elapsed = float(run.get("elapsed_s"))
        if elapsed >= 0:
            out["elapsed_s"] = round(elapsed, 1)
    except (TypeError, ValueError):
        pass
    if "dry_run" in run:
        out["dry_run"] = bool(run.get("dry_run"))
    return out


def _clean_environment(env: Any) -> Dict[str, Any]:
    """An environment block reduced to its allowlisted keys and types."""
    if not isinstance(env, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in ("os_version", "arch", "gpu_model"):
        if env.get(key):
            out[key] = _safe_fact(env[key], 60 if key == "gpu_model" else 40)
    for key in ("cores", "ram_gb", "vram_gb"):
        if isinstance(env.get(key), int) and not isinstance(env[key], bool):
            out[key] = max(0, min(int(env[key]), 100000))
    for key in ("cuda", "mps"):
        if isinstance(env.get(key), bool):
            out[key] = env[key]
    if env.get("install") in ("pip", "conda", "installer", "editable",
                              "unknown"):
        out["install"] = env["install"]
    versions = env.get("versions")
    if isinstance(versions, Mapping):
        out["versions"] = {
            key: _safe_token(versions[key], 40)
            for key in ("torch", "cellpose", "numpy", "scikit_image",
                        "pyside6", "qt") if versions.get(key)}
    return out


def _clean_inputs(facts: Any) -> Dict[str, Any]:
    """Input facts reduced to counts, extensions, shape and dtype."""
    if not isinstance(facts, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in ("images", "channels"):
        if isinstance(facts.get(key), int) and not isinstance(facts[key],
                                                              bool):
            out[key] = max(0, int(facts[key]))
    extensions = facts.get("extensions")
    if isinstance(extensions, (list, tuple)):
        out["extensions"] = [e for e in (str(x).lower() for x in extensions)
                             if e in _IMAGE_EXTENSIONS][:10]
    shape = facts.get("shape")
    if isinstance(shape, (list, tuple)):
        out["shape"] = [int(n) for n in shape[:6]
                        if isinstance(n, int) and not isinstance(n, bool)]
    if facts.get("dtype"):
        out["dtype"] = _safe_token(facts["dtype"], 16)
    return out


def _clean_settings(settings: Any) -> Dict[str, Any]:
    """A settings block re-checked: names, and values as public_settings."""
    if not isinstance(settings, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key, value in settings.items():
        key = str(key)
        if len(out) >= MAX_SETTINGS or not _SETTING_NAME_RE.match(key):
            continue
        out[key] = _public_setting_value(key, value)
    return out



def _validated_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``summary`` with only allowlisted keys, types and sizes.

    Applied to every summary before it is queued or posted, so a summary
    built elsewhere, or edited on disk, still leaves in the published shape.
    The body is kept under :data:`MAX_SUMMARY_BYTES` by dropping the
    outermost frames first.
    """
    source = dict(summary or {})
    out: Dict[str, Any] = {
        "schema_version": PUBLIC_SUMMARY_SCHEMA,
        "install_id": str(source.get("install_id", "") or ""),
        "spacr_version": _safe_token(source.get("spacr_version", ""), 40),
        "os": source.get("os") if source.get("os") in (
            "linux", "windows", "macos", "other") else "other",
        "python": _safe_token(source.get("python", ""), 40),
        "qt": _safe_token(source.get("qt", ""), 40),
        "torch": _safe_token(source.get("torch", ""), 40),
        "gpu": source.get("gpu") if source.get("gpu") in _GPU_CLASSES
        else "unknown",
        "module": re.sub(r"[^a-z0-9_]", "",
                         str(source.get("module", "") or "").lower())[:40],
        "exception_type": _safe_token(source.get("exception_type", ""), 100),
        "message": scrub_public_text(source.get("message", ""),
                                     MAX_MESSAGE_CHARS),
        "frames": [],
    }
    if source.get("kind") == KIND_USER_ERROR:
        out["message"] = scrub_public_lines(source.get("message", ""),
                                            MAX_USER_ERROR_MESSAGE_CHARS)
    if not _UUID_RE.fullmatch(out["install_id"]):
        out["install_id"] = install_id()
    frames = source.get("frames") or []
    if isinstance(frames, (list, tuple)):
        for frame in list(frames)[-MAX_FRAMES:]:
            if isinstance(frame, Mapping):
                kept = {
                    "file": _safe_token(frame.get("file", ""), 80)
                    or "<unknown>",
                    "function": _safe_token(frame.get("function", ""), 80)
                    or "<unknown>",
                    "line": _line_number(frame.get("line", 0)),
                }
                code = frame.get("code")
                if code and kept["file"] != "<user-code>":
                    kept["code"] = scrub_public_text(code, MAX_CODE_CHARS)
                out["frames"].append(kept)
    note = scrub_public_text(source.get("user_note", ""), MAX_NOTE_CHARS) \
        if source.get("user_note") else ""
    if note:
        out["user_note"] = note
    if source.get("dry_run") is True:
        out["dry_run"] = True
    summary_text = scrub_public_lines(source.get("ai_summary", ""),
                                      MAX_AI_SUMMARY_CHARS, True) \
        if source.get("ai_summary") else ""
    if summary_text:
        out["ai_summary"] = summary_text
    if isinstance(source.get("summarize"), bool):
        out["summarize"] = source["summarize"]
    if source.get("kind") in (KIND_FAULT, KIND_USER_ERROR):
        out["kind"] = source["kind"]
    screen = " ".join(_SCREEN_RE.sub("", str(source.get("screen", "") or ""))
                      .split())[:MAX_SCREEN_CHARS].strip()
    if screen and screen[0].isalnum():
        out["screen"] = screen
    error_text = scrub_public_lines(source.get("error_text", ""),
                                    MAX_ERROR_TEXT_CHARS) \
        if source.get("error_text") else ""
    if error_text:
        out["error_text"] = error_text
    chain = source.get("chain")
    if isinstance(chain, (list, tuple)) and chain:
        out["chain"] = [
            {"type": _safe_token(item.get("type", ""), 100),
             "message": scrub_public_text(item.get("message", ""),
                                          MAX_MESSAGE_CHARS)}
            for item in list(chain)[-MAX_CHAIN:]
            if isinstance(item, Mapping) and _safe_token(
                item.get("type", ""), 100)]
        if not out["chain"]:
            out.pop("chain")
    for key, clean in (("settings", _clean_settings),
                       ("inputs", _clean_inputs),
                       ("environment", _clean_environment),
                       ("run", _public_run)):
        value = clean(source.get(key)) if source.get(key) else {}
        if value:
            out[key] = value
    cause = normalise_cause(source.get("cause", ""))
    if cause:
        out["cause"] = cause
    log_tail = public_log_tail(source.get("log_tail", "")) \
        if source.get("log_tail") else ""
    if log_tail:
        out["log_tail"] = log_tail
    for key in ("log_tail", "settings"):
        if len(json.dumps(out).encode("utf-8")) > MAX_SUMMARY_BYTES:
            out.pop(key, None)
    while (len(json.dumps(out).encode("utf-8")) > MAX_SUMMARY_BYTES
           and out["frames"]):
        out["frames"].pop(0)
    return out


def public_summary(exception: Optional[BaseException] = None, *,
                   traceback_text: str = "", module: str = "",
                   user_note: str = "", dry_run: bool = False,
                   kind: str = "", code_context: bool = False,
                   screen: str = "", error_text: str = "",
                   message: str = "",
                   settings: Optional[Mapping[str, Any]] = None,
                   log_text: str = "",
                   run: Optional[Mapping[str, Any]] = None,
                   diagnostics: bool = False
                   ) -> Dict[str, Any]:
    """Build the anonymised summary of an error that may be published.

    Only the keys in :data:`PUBLIC_SUMMARY_KEYS` are present. The message is
    passed through :func:`scrub_public_text` and cut to
    :data:`MAX_MESSAGE_CHARS`; the frames are at most :data:`MAX_FRAMES`
    ``{file, function, line}`` records with the file as a basename, or
    ``<user-code>`` for a file outside spaCR, Python and its libraries.
    No log, setting, environment variable or path is read into it.

    :param exception: the exception, with its traceback.
    :param traceback_text: a formatted traceback, used when no exception
        object is at hand (a failed pipeline run reports one as text).
    :param module: the spaCR module key the error happened in, such as
        ``mask`` or ``measure``.
    :param user_note: what the user adds, scrubbed and cut to
        :data:`MAX_NOTE_CHARS`; left out when empty.
    :param dry_run: ask the relay to validate without filing; left out when
        false.
    :param kind: :data:`KIND_FAULT` or :data:`KIND_USER_ERROR`; worked out
        by :func:`classify_failure` when empty.
    :param code_context: add each spaCR frame's source line as ``code``.
        Frames of any other file never carry code.
    :param screen: the screen's title as the user sees it, such as
        ``Mask``; at most :data:`MAX_SCREEN_CHARS` characters.
    :param error_text: the error as the console showed it, scrubbed with
        its line breaks kept and cut to :data:`MAX_ERROR_TEXT_CHARS`.
    :param message: the message to send in place of the exception's own,
        such as every problem a refused run found; a user error keeps its
        line breaks and may be :data:`MAX_USER_ERROR_MESSAGE_CHARS` long.
    :param settings: the run's settings; their names and public values
        (:func:`public_settings`) and the source folder's input facts
        (:func:`public_input_facts`) are added.
    :param log_text: the console text before the failure; its last lines
        are added, scrubbed (:func:`public_log_tail`).
    :param run: ``{"stage", "elapsed_s", "dry_run"}`` of the failed run.
    :param diagnostics: add the environment block
        (:func:`public_environment`) and the exception chain.
    :returns: the summary, ready for :func:`submit_public_summary`.
    """
    failure_kind = kind or classify_failure(exception, traceback_text)
    override = message
    kind, message, frames = "", "", []
    if exception is not None:
        kind = type(exception).__name__
        message = str(exception)
        frames = [_frame(item.filename, item.name, item.lineno,
                         item.line if code_context else None)
                  for item in traceback.extract_tb(exception.__traceback__)]
    elif traceback_text:
        kind, message, frames = _parse_traceback_text(traceback_text)
        if not code_context:
            for frame in frames:
                frame.pop("code", None)
    return _validated_summary({
        "install_id": install_id(),
        "spacr_version": _spacr_version(),
        "os": _os_family(),
        "python": platform.python_version(),
        "qt": _distribution_version("PySide6"),
        "torch": _distribution_version("torch"),
        "gpu": _gpu_class(),
        "module": module,
        "exception_type": kind,
        "message": override or message,
        "frames": frames,
        "user_note": user_note,
        "dry_run": bool(dry_run),
        "kind": failure_kind,
        "screen": screen,
        "error_text": error_text,
        "chain": _exception_chain(exception, traceback_text)
        if diagnostics else [],
        "settings": public_settings(settings) if settings else {},
        "inputs": public_input_facts(settings) if settings else {},
        "environment": public_environment() if diagnostics else {},
        "run": run or {},
        "log_tail": log_text,
    })


def ai_summary_prompt(summary: Mapping[str, Any]) -> str:
    """The fixed prompt that asks the user's own AI to summarise a report.

    The AI sees only the anonymised summary, never the traceback, log or
    settings, and what it writes is scrubbed again before it is sent.

    :param summary: a summary from :func:`public_summary`.
    :returns: the prompt.
    """
    shown = {key: value for key, value in _validated_summary(summary).items()
             if key not in ("install_id", "ai_summary", "summarize",
                            "dry_run", "user_note")}
    return (
        "Below is an anonymised spaCR error report as JSON. Summarise the "
        "likely cause of the error and where in spaCR it happens (module, "
        "file and function), then give a short fix the user can apply. Use "
        f"plain prose, at most {AI_SUMMARY_TARGET_CHARS} characters. Do not "
        "include any personal data: no paths, file names, user or host "
        "names, e-mail or IP addresses, or tokens. End with one last line "
        "of the form CAUSE: <kebab-case-slug>, at most 5 lowercase words "
        "naming the root cause, such as CAUSE: no-object-channel-set.\n\n"
        + json.dumps(shown, indent=2))


def example_public_summary() -> Dict[str, Any]:
    """Build a summary from a made-up error, to show what a report holds.

    The error's message names a plate file under the home folder, an
    e-mail address and a token, so the example shows each one removed.

    :returns: a summary built exactly as a real one is.
    """
    home = _home_variants()[:1] or ["/home/someone"]
    planted = (f"could not open {home[0]}/plates/patient_042.tif for "
               f"someone@example.org with key ghp_0123456789abcdefABCDEF")
    settings = {"src": f"{home[0]}/plates/patient_042", "cell_channel": 0,
                "nucleus_channel": None, "cell_diameter": 30.5,
                "save": True, "cellpose_model": "cyto3",
                "experiment": "patient_042",
                "notes": "ask someone@example.org at 10.1.2.3"}
    log = (f"[INFO] reading {home[0]}/plates/patient_042/A01.tif on "
           f"labhost42.example.org (10.1.2.3)\n[ERROR] {planted}")
    try:
        try:
            raise OSError(f"disk error on {home[0]}/plates")
        except OSError as cause:
            raise FileNotFoundError(planted) from cause
    except FileNotFoundError as exc:
        return public_summary(
            exc, module="mask", screen="Mask", settings=settings,
            log_text=log, error_text=f"FileNotFoundError: {planted}",
            run={"stage": "preprocess_generate_masks", "elapsed_s": 3.2,
                 "dry_run": False},
            diagnostics=True)


def _queue_files() -> List[Path]:
    """Return the waiting reports, oldest first."""
    try:
        return sorted(_queue_dir().glob("*.json"))
    except OSError:
        return []


def _prune_queue(now: Optional[float] = None) -> List[Path]:
    """Delete expired, unreadable and surplus reports; return those kept."""
    now = time.time() if now is None else now
    kept: List[Path] = []
    for path in _queue_files():
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            queued_at = float(entry.get("queued_at", 0))
            if not isinstance(entry.get("summary"), dict):
                raise ValueError("no summary")
        except (OSError, ValueError, TypeError, AttributeError):
            path.unlink(missing_ok=True)
            continue
        if now - queued_at > QUEUE_RETENTION_S:
            path.unlink(missing_ok=True)
            continue
        kept.append(path)
    while len(kept) > QUEUE_LIMIT:
        kept.pop(0).unlink(missing_ok=True)
    return kept


def _enqueue(summary: Mapping[str, Any], *, automatic=False) -> Optional[Path]:
    """Write a summary to the queue and return its file.

    :returns: the file, or ``None`` when it could not be written.
    """
    with _STATE_LOCK:
        try:
            folder = _queue_dir()
            folder.mkdir(parents=True, exist_ok=True)
            now = time.time()
            path = folder / f"{int(now * 1000):015d}-{uuid.uuid4().hex[:8]}.json"
            record = {"queued_at": now, "summary": dict(summary)}
            if automatic:
                record.update(automatic=True, active_pid=os.getpid())
                _ERROR_ACTIVE_QUEUES.add(path)
            path.write_text(json.dumps(record), encoding="utf-8")
        except OSError:
            return None
        _prune_queue(now)
        return path if path.exists() else None


def queued_summaries() -> List[Dict[str, Any]]:
    """Return the reports waiting to be sent, oldest first.

    Expired and surplus reports are deleted first.

    :returns: the summaries.
    """
    out: List[Dict[str, Any]] = []
    with _STATE_LOCK:
        for path in _prune_queue():
            try:
                out.append(json.loads(path.read_text(encoding="utf-8"))
                           ["summary"])
            except (OSError, ValueError, KeyError):
                continue
    return out


def _open_url(request: Any, timeout: float) -> Any:
    """Open ``request`` with :func:`urllib.request.urlopen`; tests replace this."""
    import urllib.request

    return urllib.request.urlopen(request, timeout=timeout)


def _post_json(route: str, payload: Optional[Mapping[str, Any]],
               timeout: float) -> Tuple[int, Dict[str, Any]]:
    """POST ``payload`` to the relay, or GET when it is ``None``.

    :returns: the HTTP status and the JSON answer.
    :raises OSError: when the relay cannot be reached or does not answer in
        time.
    """
    import urllib.error
    import urllib.request

    headers = {"User-Agent": f"spaCR/{_safe_token(_spacr_version(), 40)}"}
    if payload is None:
        request = urllib.request.Request(relay_url() + route, method="GET",
                                         headers=headers)
    else:
        headers["Content-Type"] = "application/json"
        body = json.dumps(dict(payload),
                          separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            relay_url() + route, data=body, method="POST", headers=headers)
    try:
        with _open_url(request, timeout) as response:
            status = int(getattr(response, "status", 0)
                         or response.getcode())
            raw = response.read(64 * 1024)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            raw = exc.read(64 * 1024)
        except Exception:
            raw = b""
    try:
        data = json.loads(raw.decode("utf-8", "replace") or "{}")
    except ValueError:
        data = {}
    return status, data if isinstance(data, dict) else {}


def _send_one(summary: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
    """Post one summary to ``/report`` and say what became of it."""
    import http.client

    try:
        status, data = _post_json("/report", summary, timeout)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        return {"status": REPORT_QUEUED,
                "detail": f"{type(exc).__name__}: {exc}"}
    if status == 200:
        try:
            number = int(data.get("issue_number", 0) or 0)
        except (TypeError, ValueError):
            number = 0
        source = str(data.get("ai_summary_source", "") or "none")
        outcome = {"status": REPORT_SENT, "issue_number": number,
                   "issue_url": str(data.get("issue_url", "") or ""),
                   "deduplicated": bool(data.get("deduplicated", False)),
                   "ai_summary_source": source if source in (
                       "client", "relay", "none") else "none"}
        if source == "relay" and data.get("ai_summary"):
            text = scrub_public_lines(str(data.get("ai_summary", "")),
                                      MAX_AI_SUMMARY_CHARS, True)
            if text:
                outcome["ai_summary"] = text
        allowance = data.get("allowance")
        if isinstance(allowance, Mapping):
            outcome["allowance"] = {
                "reports_left": _count(allowance.get("reports_left")),
                "resets_at": (_parse_reset(allowance.get("resets_at"))
                              or next_reset_utc()).isoformat()}
        return outcome
    if status == 429:
        resets = next_reset_utc()
        _update_state(reports_paused_until=resets.timestamp())
        return {"status": REPORT_QUEUED, "detail": "quota", "quota": True,
                "scope": str(data.get("scope", "") or ""),
                "resets_at": resets.isoformat()}
    if 400 <= status < 500:
        return {"status": REPORT_DROPPED,
                "detail": str(data.get("error", "") or f"HTTP {status}")}
    return {"status": REPORT_QUEUED, "detail": f"HTTP {status}"}


def _sending_allowed() -> bool:
    """Whether reports are on and the network may be used."""
    try:
        return anonymous_reports_enabled() and network_allowed()
    except Exception:
        return False


def _reports_paused_until() -> Optional[datetime]:
    """When a quota refusal lets reports be posted again, or ``None``."""
    try:
        until = float(_read_state().get("reports_paused_until", 0) or 0)
    except (TypeError, ValueError):
        return None
    if until <= time.time():
        return None
    return datetime.fromtimestamp(until, timezone.utc)


def _send_queued(path: Optional[Path], summary: Mapping[str, Any],
                 timeout: float, *, allowed=None) -> Dict[str, Any]:
    """Send a queued summary, deleting it unless it is to be tried again.

    While a quota refusal holds (to the next 00:00 UTC) nothing is
    posted: the summary stays queued and the outcome says so.
    """
    paused = _reports_paused_until()
    if paused is not None:
        return {"status": REPORT_QUEUED, "detail": "quota", "quota": True,
                "scope": "install", "resets_at": paused.isoformat()}
    with _SEND_LOCK:
        if allowed is not None and not allowed():
            if path is not None:
                path.unlink(missing_ok=True)
            return {"status": REPORT_DISABLED}
        outcome = _send_one(summary, timeout)
        if path is not None and outcome["status"] in (REPORT_SENT,
                                                      REPORT_DROPPED):
            path.unlink(missing_ok=True)
    if outcome["status"] == REPORT_SENT:
        flush_report_queue(timeout)
    return outcome


def submit_public_summary(summary: Mapping[str, Any], *,
                          timeout: float = SEND_TIMEOUT_S) -> Dict[str, Any]:
    """Send an anonymised summary to the relay, or queue it for later.

    Blocks for up to ``timeout`` seconds, so call it off the GUI thread, or
    use :func:`submit_public_summary_in_background`. With reports off, or
    the network forbidden, nothing is sent and nothing is stored.

    :param summary: a summary from :func:`public_summary`.
    :param timeout: seconds the post may take.
    :returns: ``{"status": ...}`` with :data:`REPORT_SENT` (and
        ``issue_number``, ``issue_url``, ``deduplicated``),
        :data:`REPORT_QUEUED` (kept to try again), :data:`REPORT_DROPPED`
        (rejected, not retried) or :data:`REPORT_DISABLED`.
    """
    if not _sending_allowed():
        return {"status": REPORT_DISABLED}
    clean = _validated_summary(summary)
    path = _enqueue(clean)
    return _send_queued(path, clean, timeout)


def submit_public_summary_in_background(
        summary: Mapping[str, Any],
        on_done: Optional[Callable[[Dict[str, Any]], None]] = None,
        ) -> Optional[threading.Thread]:
    """Queue a summary now and send it on a daemon thread.

    The summary is on disk before this returns, so a process that ends at
    once still sends it the next time spaCR starts.

    :param summary: a summary from :func:`public_summary`.
    :param on_done: called with the outcome :func:`submit_public_summary`
        describes, on the sending thread.
    :returns: the thread, or ``None`` when reports are off and nothing was
        queued.
    """
    if not _sending_allowed():
        if on_done is not None:
            on_done({"status": REPORT_DISABLED})
        return None
    clean = _validated_summary(summary)
    path = _enqueue(clean)

    def _run() -> None:
        """Send the queued summary and hand the outcome to ``on_done``."""
        try:
            outcome = _send_queued(path, clean, SEND_TIMEOUT_S)
        except Exception as exc:
            outcome = {"status": REPORT_QUEUED,
                       "detail": f"{type(exc).__name__}: {exc}"}
        if on_done is not None:
            try:
                on_done(outcome)
            except Exception:
                pass

    thread = threading.Thread(target=_run, name="spacr-anonymous-report",
                              daemon=True)
    thread.start()
    return thread


def flush_report_queue(timeout: float = SEND_TIMEOUT_S) -> Dict[str, int]:
    """Try to send every waiting report, oldest first.

    Stops at the first report that has to wait again (no network, a server
    error or the relay's quota), so an offline computer makes one attempt,
    not twenty. Does nothing while another send is running.

    :param timeout: seconds each post may take.
    :returns: ``{"sent": n, "dropped": n, "kept": n}``.
    """
    counts = {"sent": 0, "dropped": 0, "kept": 0}
    if not _sending_allowed():
        return counts
    if _reports_paused_until() is not None:
        counts["kept"] = len(_queue_files())
        return counts
    if not _SEND_LOCK.acquire(blocking=False):
        counts["kept"] = len(_queue_files())
        return counts
    try:
        with _STATE_LOCK:
            waiting = _prune_queue()
        for index, path in enumerate(waiting):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                summary = record["summary"]
            except (OSError, ValueError, KeyError):
                path.unlink(missing_ok=True)
                continue
            if (record.get("active_pid") == os.getpid()
                    and path in _ERROR_ACTIVE_QUEUES):
                counts["kept"] += 1
                continue
            if record.get("automatic") and not _automatic_error_policy(
                    summary.get("kind", KIND_FAULT))[2]:
                path.unlink(missing_ok=True)
                counts["dropped"] += 1
                continue
            if record.get("automatic") and not mode_code_context(
                    _ERROR_POLICY_CACHE.get("mode")):
                for frame in summary.get("frames", []):
                    frame.pop("code", None)
            outcome = _send_one(_validated_summary(summary), timeout)
            if outcome["status"] == REPORT_SENT:
                counts["sent"] += 1
                path.unlink(missing_ok=True)
            elif outcome["status"] == REPORT_DROPPED:
                counts["dropped"] += 1
                path.unlink(missing_ok=True)
            else:
                counts["kept"] = len(waiting) - index
                break
    finally:
        _SEND_LOCK.release()
    return counts


def diagnose_remotely(summary: Mapping[str, Any], question: str = "", *,
                      timeout: float = DIAGNOSE_TIMEOUT_S) -> Dict[str, str]:
    """Ask the relay's hosted model about an error, with no account.

    Sends the anonymised summary and the scrubbed question, cut to
    :data:`MAX_QUESTION_CHARS`, to ``/diagnose``. The answer is generated
    advice, not a checked fix. Blocks, so call it off the GUI thread.

    :param summary: a summary from :func:`public_summary`.
    :param question: what the user asks; may be empty.
    :param timeout: seconds the model may take.
    :returns: ``{"advice": text, "model": model name}``.
    :raises RelayRefused: with ``reason`` ``"quota"`` when the free daily
        limit is used up, ``"network"`` when the relay cannot be reached or
        the network is forbidden, and ``"error"`` otherwise.
    """
    import http.client

    if not network_allowed():
        raise RelayRefused("network", f"{NO_NETWORK_ENV} or safe mode "
                                      f"forbids the network")
    payload = _validated_summary(summary)
    for key in ("user_note", "dry_run", "error_text", "chain", "settings",
                "inputs", "environment", "run", "log_tail"):
        payload.pop(key, None)
    asked = scrub_public_text(question, MAX_QUESTION_CHARS) if question else ""
    if asked:
        payload["question"] = asked
    try:
        status, data = _post_json("/diagnose", payload, timeout)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise RelayRefused("network", f"{type(exc).__name__}: {exc}") from exc
    if status == 429:
        raise RelayRefused("quota", str(data.get("refused", "") or "quota"),
                           scope=str(data.get("scope", "") or "install"),
                           resets_at=_parse_reset(data.get("resets_at")))
    advice = str(data.get("advice", "") or "").strip()
    if status != 200 or not advice:
        raise RelayRefused("error", str(data.get("error", "")
                                        or f"HTTP {status}"))
    answer: Dict[str, Any] = {
        "advice": advice, "model": str(data.get("model", "") or ""),
        "tier": "light" if data.get("tier") == "light" else "best",
        "cached": data.get("cached") is True,
        "cause": normalise_cause(data.get("cause", ""))}
    if isinstance(data.get("allowance"), Mapping):
        answer["allowance"] = _normal_allowance(
            {"ai": data["allowance"],
             "resets_at": data["allowance"].get("resets_at")})
    return answer


_CAUSE_LINE_RE = re.compile(
    r"^\s*[*_`>#-]*\s*(?:CAUSE\s*[:=]\s*[*_`]*\s*(.*?)"
    r"|(?i:cause)\s*[:=]\s*[*_`]*\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+))"
    r"\s*[*_`]*\s*$")


def normalise_cause(raw: Any) -> str:
    """A likely-cause key as the relay keeps it: kebab-case, 5 words, 40 chars.

    :param raw: the key as the AI wrote it.
    :returns: the slug, or ``""``.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", str(raw or "").lower()) if w]
    slug = "-".join(words[:5])
    return slug[:40].rstrip("-")


def split_cause(text: str) -> Tuple[str, str]:
    """Take the ``CAUSE: <slug>`` line off an AI answer.

    :param text: the answer.
    :returns: ``(answer without the line, slug or "")``.
    """
    kept, cause = [], ""
    for line in str(text or "").splitlines():
        match = _CAUSE_LINE_RE.match(line)
        if match:
            cause = normalise_cause(match.group(1) or match.group(2)) or cause
            continue
        kept.append(line)
    return "\n".join(kept).strip(), cause


#: Explanations kept on this computer; the oldest go first past this many.
MAX_SAVED_EXPLANATIONS = 200

#: Days a saved explanation is reused, as the relay's own cache does.
SAVED_EXPLANATION_DAYS = 30

_SAVED_LOCK = threading.Lock()


def _normalized_problems(message: Any) -> List[str]:
    """A user error's problem list as the relay's signature reads it.

    The lines beginning with a hyphen and a space, lower case, with
    placeholders, paths and numbers taken out and sorted, so the same
    problems in another order or on another plate give the same signature.
    """
    out: List[str] = []
    for line in str(message or "").splitlines():
        line = line.strip()
        if not re.match(r"^[-*•]\s+\S", line):
            continue
        line = re.sub(r"^[-*•]\s+", "", line).lower()
        line = re.sub(r"<[^<>]*>", " ", line)
        line = re.sub(r"\S*[\\/]\S*", " ", line)
        line = re.sub(r"[0-9]+", "", line)
        line = " ".join(line.split())
        if line:
            out.append(line)
    return sorted(out)


def explanation_signature(summary: Mapping[str, Any]) -> str:
    """The error's signature, computed as the relay computes it.

    The exception type, the module and the innermost five frames
    (``file:function``); a user error adds its normalised problem list.
    The relay keys its saved explanations, and its issues, by the same
    value.

    :param summary: a summary from :func:`public_summary`.
    :returns: the SHA-256 hex digest.
    """
    parts = [str(summary.get("exception_type", "") or ""),
             str(summary.get("module", "") or "")]
    for frame in list(summary.get("frames") or [])[-5:]:
        parts.append(f"{frame.get('file', '')}:{frame.get('function', '')}")
    if summary.get("kind") == KIND_USER_ERROR:
        parts.extend(_normalized_problems(summary.get("message", "")))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def explanation_key(traceback_text: str, module: str = "", kind: str = "",
                    message: str = "") -> str:
    """:func:`explanation_signature` of a failure, without the environment.

    Reads only the traceback, so it is cheap enough for the GUI thread and
    gives what the full :func:`public_summary` of the same failure gives.

    :param traceback_text: the formatted traceback.
    :param module: the spaCR module key.
    :param kind: :data:`KIND_FAULT` or :data:`KIND_USER_ERROR`.
    :param message: the message sent in place of the exception's own.
    :returns: the signature.
    """
    exception_type, parsed, frames = _parse_traceback_text(traceback_text)
    return explanation_signature(_validated_summary({
        "install_id": "00000000-0000-4000-8000-000000000000",
        "module": module, "exception_type": exception_type,
        "message": message or parsed, "frames": frames,
        "kind": kind or classify_failure(None, traceback_text)}))


def _spacr_minor() -> str:
    """spaCR's ``major.minor``, which a saved explanation must match."""
    match = re.match(r"^(\d+)\.(\d+)", _spacr_version())
    return f"{match.group(1)}.{match.group(2)}" if match else _spacr_version()


def _saved_path() -> Path:
    """The file saved explanations live in, in the spaCR home."""
    return _reports_dir() / "explanations.json"


def _read_saved() -> Dict[str, Any]:
    """The saved explanations, ``{signature: entry}``, or ``{}``."""
    try:
        data = json.loads(_saved_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def saved_explanation(signature: str, now: Optional[float] = None
                      ) -> Optional[Dict[str, Any]]:
    """The explanation saved on this computer for an error, if still good.

    Good means written for this spaCR ``major.minor`` and less than
    :data:`SAVED_EXPLANATION_DAYS` days ago.

    :param signature: from :func:`explanation_signature`.
    :param now: the time, in seconds since the epoch; defaults to now.
    :returns: ``{"text", "tier", "model"}``, or ``None``.
    """
    entry = _read_saved().get(str(signature or ""))
    if not isinstance(entry, dict) or not entry.get("text"):
        return None
    now = time.time() if now is None else now
    try:
        age = now - float(entry.get("created", 0))
    except (TypeError, ValueError):
        return None
    if age >= SAVED_EXPLANATION_DAYS * 86400 or entry.get(
            "minor") != _spacr_minor():
        return None
    return {"text": str(entry["text"]), "tier": str(entry.get("tier", "")),
            "model": str(entry.get("model", "")),
            "cause": normalise_cause(entry.get("cause", ""))}


def _created(entry: Mapping[str, Any]) -> float:
    """When a saved answer was written; ``0`` for a damaged entry."""
    try:
        return float(entry.get("created", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def save_explanation(signature: str, text: str, tier: str = "",
                     model: str = "", now: Optional[float] = None,
                     cause: str = "") -> None:
    """Keep an explanation on this computer for the next identical error.

    At most :data:`MAX_SAVED_EXPLANATIONS` are kept, oldest dropped first,
    and expired ones are dropped on every write. A failed write is ignored.

    :param signature: from :func:`explanation_signature`.
    :param text: the scrubbed explanation.
    :param tier: ``"best"``, ``"light"`` or ``""``.
    :param model: the model that wrote it, when known.
    :param now: the time, in seconds since the epoch; defaults to now.
    :param cause: the likely-cause key the AI gave, if any.
    """
    if not signature or not text:
        return
    now = time.time() if now is None else now
    with _SAVED_LOCK:
        saved = {key: entry for key, entry in _read_saved().items()
                 if isinstance(entry, dict)
                 and now - _created(entry) < SAVED_EXPLANATION_DAYS * 86400}
        saved[str(signature)] = {"text": str(text), "tier": str(tier or ""),
                                 "model": str(model or ""), "created": now,
                                 "minor": _spacr_minor(),
                                 "cause": normalise_cause(cause)}
        newest = sorted(saved.items(), key=lambda item: _created(item[1])
                        )[-MAX_SAVED_EXPLANATIONS:]
        try:
            folder = _reports_dir()
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / (f"explanations.{os.getpid()}."
                                  f"{threading.get_ident()}.tmp")
            temporary.write_text(json.dumps(dict(newest)), encoding="utf-8")
            os.replace(temporary, _saved_path())
        except OSError:
            return


#: The most characters of question and pasted context an answer is saved for.
QUESTION_CACHE_CONTEXT_MAX = 500


def normalise_question(text: Any) -> str:
    """A question as both caches compare it.

    Lower case, placeholders such as ``<PATH>`` removed, and every run of
    whitespace or punctuation folded into one space, the way the relay's
    ``normaliseQuestion`` does it.

    :param text: the question.
    :returns: the normalised question.
    """
    text = re.sub(r"<[^<>]*>", " ", str(text or "").lower())
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def question_key(question: str, place: str = "",
                 context_chars: int = 0) -> str:
    """The key a question's saved answer is kept under, or ``""``.

    SHA-256 of the scrubbed, normalised question, the place it was asked
    (screen or module) and spaCR's ``major.minor``, prefixed ``q:``. A
    question with more than :data:`QUESTION_CACHE_CONTEXT_MAX` characters
    of question and pasted context is never saved.

    :param question: what the user asked.
    :param place: the screen or module it was asked on.
    :param context_chars: characters of console context, code or traceback
        pasted with it.
    :returns: the key, or ``""`` when the answer must not be saved.
    """
    question = str(question or "")
    if len(question) + max(0, int(context_chars)) > QUESTION_CACHE_CONTEXT_MAX:
        return ""
    normal = normalise_question(scrub_public_text(question, MAX_QUESTION_CHARS))
    if not normal:
        return ""
    digest = hashlib.sha256(
        f"{normal}\n{place}\n{_spacr_minor()}".encode("utf-8")).hexdigest()
    return f"q:{digest}"


def _normal_allowance(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the relay's allowance in one fixed shape, counts as ints."""
    ai = data.get("ai") if isinstance(data.get("ai"), Mapping) else {}
    pool = str(data.get("pool", "") or "best")
    out_ai: Dict[str, Any] = {"best_left": _count(ai.get("best_left")),
                              "light_left": _count(ai.get("light_left"))}
    for key in ("neurons_left", "share_neurons", "active_users"):
        if ai.get(key) is not None:
            out_ai[key] = _count(ai.get(key))
    if ai.get("share_pct") is not None:
        try:
            out_ai["share_pct"] = max(0.0, float(ai.get("share_pct")))
        except (TypeError, ValueError):
            pass
    if ai.get("tier_now") in ("best", "light"):
        out_ai["tier_now"] = ai.get("tier_now")
    out = {
        "ai": out_ai,
        "reports_left": _count(data.get("reports_left", REPORTS_PER_DAY)),
        "pool": pool if pool in ("best", "light", "exhausted") else "best",
        "resets_at": (_parse_reset(data.get("resets_at"))
                      or next_reset_utc()).isoformat(),
    }
    if data.get("pool_pct_left") is not None:
        try:
            out["pool_pct_left"] = max(0.0, float(data.get("pool_pct_left")))
        except (TypeError, ValueError):
            pass
    return out


def fetch_allowance(timeout: float = SEND_TIMEOUT_S
                    ) -> Optional[Dict[str, Any]]:
    """Ask the relay how much of today's free allowance this install has left.

    A GET of ``/allowance?install_id=...``; it uses up nothing. Blocks, so
    call it off the GUI thread.

    :param timeout: seconds the request may take.
    :returns: ``{"ai": {"best_left", "light_left"}, "reports_left",
        "pool", "resets_at"}`` with ``pool`` one of ``best``, ``light`` or
        ``exhausted`` and ``resets_at`` an ISO-8601 UTC time; ``None`` when
        the network is forbidden or the relay gives no answer.
    """
    import http.client
    import urllib.parse

    if not network_allowed():
        return None
    query = urllib.parse.urlencode({"install_id": install_id()})
    try:
        status, data = _post_json(f"/allowance?{query}", None, timeout)
    except (OSError, http.client.HTTPException, ValueError):
        return None
    if status != 200 or not data:
        return None
    return _normal_allowance(data)


_REPORTING_TERMS_VERSION = "4.4"
_ERROR_DISPATCH_LOCK = threading.RLock()
_ERROR_DISPATCH_SEEN = {}
_ERROR_DISPATCH_LOCAL = threading.local()
_ERROR_RESULT_LISTENERS = []
_ERROR_POLICY_CACHE = {}
_ERROR_DISPATCH_LIMIT = threading.BoundedSemaphore(QUEUE_LIMIT)
_ERROR_ACTIVE_QUEUES = set()
_ERROR_OCCURRENCE_COUNTER = itertools.count()
_ERROR_OCCURRENCE_POLICIES = {}


def _automatic_error_policy(kind):
    """Read only the affirmed, mirrored reporting policy; never import Qt."""
    state = _ERROR_POLICY_CACHE
    saved = state.get("affirmed_policy")
    policy = dict(saved) if isinstance(saved, Mapping) else {}
    accepted = policy.get("accepted_at")
    consent = (policy.get("version") == _REPORTING_TERMS_VERSION
               and isinstance(accepted, str)
               and accepted.endswith(("Z", "+00:00"))
               and _parse_reset(accepted) is not None)
    mode = state.get("mode", "off" if state.get("enabled") is False else DEFAULT_ERROR_REPORT_MODE)
    allowed = consent and network_allowed()
    explain = bool(allowed and mode_explains(mode) and policy.get("explain")
                   and policy.get("provider") != "none")
    send = bool(allowed and mode_sends(kind, mode) and policy.get("auto_file")
                and policy.get("prompt") == "always")
    policy["_mode"] = mode
    return policy, explain, send


def _deliver_error_result(outcome):
    """Notify weak GUI listeners, or print a sanitized answer to the terminal."""
    for ref in reversed(tuple(_ERROR_RESULT_LISTENERS)):
        listener = ref()
        if listener is not None:
            try:
                if listener(outcome):
                    return
            except Exception:
                pass
    _write_error_result(outcome)


def _write_error_result(outcome):
    """Write sanitized results to the original terminal outside captured streams."""
    text = str((outcome.get("explanation") or {}).get("text", ""))
    url = str(outcome.get("issue_url", ""))
    if not (text or url):
        return
    active = getattr(_ERROR_DISPATCH_LOCAL, "active", False)
    _ERROR_DISPATCH_LOCAL.active = True
    try:
        stream = sys.__stderr__ or sys.stderr
        if stream is not None:
            stream.write("\n" + "\n".join(v for v in (text, url) if v) + "\n")
            stream.flush()
    except (OSError, ValueError, RuntimeError):
        pass
    finally:
        _ERROR_DISPATCH_LOCAL.active = active



def _new_error_occurrence(exception=None, *, event_key=""):
    """Create a producer identity and freeze its policy without retaining errors."""
    known = getattr(exception, "_spacr_error_occurrence", "") if exception is not None else ""
    traceback_id = id(exception.__traceback__) if exception is not None else None
    if known and getattr(exception, "_spacr_error_traceback_id", None) == traceback_id:
        return known
    event_id = str(event_key or f"error-{next(_ERROR_OCCURRENCE_COUNTER)}")
    with _ERROR_DISPATCH_LOCK:
        if event_id not in _ERROR_OCCURRENCE_POLICIES:
            _ERROR_OCCURRENCE_POLICIES[event_id] = _automatic_error_policy(KIND_FAULT)
        if len(_ERROR_OCCURRENCE_POLICIES) > 512:
            _ERROR_OCCURRENCE_POLICIES.pop(next(iter(_ERROR_OCCURRENCE_POLICIES)))
    if exception is not None:
        try:
            exception._spacr_error_occurrence = event_id
            exception._spacr_error_traceback_id = traceback_id
        except Exception:
            pass
    return event_id


def _error_occurrence(event_id):
    """Forward a producer's policy; an expired or unknown transport cannot opt in."""
    event_id = event_id or _new_error_occurrence()
    with _ERROR_DISPATCH_LOCK:
        captured = _ERROR_OCCURRENCE_POLICIES.get(
            event_id, ({"_mode": "off"}, False, False))
        return event_id, captured


def _dispatch_error(text="", *, exception=None, module="", traceback_text="",
                    summary=None, event_id="", origin="direct"):
    """Schedule bounded error processing without file I/O on the emitting thread."""
    if getattr(_ERROR_DISPATCH_LOCAL, "active", False):
        return None
    event_id, (captured, explain, send) = _error_occurrence(
        event_id or (_new_error_occurrence(exception) if exception is not None else ""))
    if not (explain or send) or not _ERROR_DISPATCH_LIMIT.acquire(blocking=False):
        return None
    with _ERROR_DISPATCH_LOCK:
        now = time.monotonic()
        for key, when in list(_ERROR_DISPATCH_SEEN.items()):
            if now - when > QUEUE_RETENTION_S:
                _ERROR_DISPATCH_SEEN.pop(key, None)
        if event_id in _ERROR_DISPATCH_SEEN:
            _ERROR_DISPATCH_LIMIT.release()
            return None
        if len(_ERROR_DISPATCH_SEEN) >= 256:
            _ERROR_DISPATCH_SEEN.pop(next(iter(_ERROR_DISPATCH_SEEN)))
        _ERROR_DISPATCH_SEEN[event_id] = now

    def work():
        """Sanitize, deduplicate, explain and publish only on this background thread."""
        _ERROR_DISPATCH_LOCAL.active = True
        queued = None
        try:
            kind = classify_failure(exception, traceback_text or text)
            original_send = send and mode_sends(kind, captured["_mode"])
            current, now_explain, now_send = _automatic_error_policy(kind)
            if not ((explain and now_explain) or (original_send and now_send)):
                return
            clean = summary or public_summary(
                exception, traceback_text=traceback_text or text,
                error_text=text, kind=kind, module=module or "background",
                code_context=(mode_code_context(captured["_mode"])
                              and mode_code_context(current["_mode"])))
            if not clean.get("exception_type"):
                clean["exception_type"] = "LoggedError"
            if not clean.get("module"):
                clean["module"] = "background"
            if not clean.get("message"):
                clean["message"] = scrub_public_text(text, MAX_MESSAGE_CHARS)
            if original_send and now_send:
                queued = _enqueue(dict(clean, summarize=False), automatic=True)
            outcome = {}
            current, now_explain, now_send = _automatic_error_policy(kind)
            selected = str(captured.get("provider", ""))
            do_explain = explain and now_explain and current.get("provider", "") == selected
            if do_explain:
                from .qt.ai.providers import list_providers, HostedProvider, explain_failure
                candidates = [HostedProvider()] + [type(p)() for p in list_providers()]
                provider = next((p for p in candidates if p.name == selected), None)
                if not selected:
                    provider = candidates[0]
                if provider is not None:
                    provider._reporting_args = list(captured.get("provider_args") or [])
                outcome["explanation"] = explain_failure(clean, provider)
            payload = dict(clean, summarize=False)
            advice = (outcome.get("explanation") or {}).get("text", "")
            if advice:
                payload["ai_summary"] = advice
                cause = (outcome.get("explanation") or {}).get("cause")
                if cause:
                    payload["cause"] = cause
            payload = _validated_summary(payload)

            def allowed():
                """Recheck revocation under the actual relay send lock."""
                latest, may_explain, may_send = _automatic_error_policy(kind)
                if not mode_code_context(latest["_mode"]):
                    for frame in payload.get("frames", []):
                        frame.pop("code", None)
                if not may_explain or latest.get("provider", "") != selected:
                    payload.pop("ai_summary", None)
                    payload.pop("cause", None)
                return original_send and may_send

            if original_send:
                outcome.update(_send_queued(queued, payload, SEND_TIMEOUT_S, allowed=allowed))
                outcome["ai_summary_attached"] = bool(payload.get("ai_summary"))
            latest, may_explain, may_send = _automatic_error_policy(kind)
            if not may_explain or latest.get("provider", "") != selected:
                outcome.pop("explanation", None)
                do_explain = False
            outcome["explained"] = do_explain
            outcome["kind"] = kind
            _deliver_error_result(outcome)
        except Exception:
            pass
        finally:
            if queued is not None:
                _ERROR_ACTIVE_QUEUES.discard(queued)
            _ERROR_DISPATCH_LOCAL.active = False
            _ERROR_DISPATCH_LIMIT.release()

    thread = threading.Thread(target=work, name="spacr-error-report", daemon=True)
    thread._spacr_error_occurrence = event_id
    try:
        thread.start()
    except Exception:
        _ERROR_DISPATCH_LIMIT.release()
        with _ERROR_DISPATCH_LOCK:
            _ERROR_DISPATCH_SEEN.pop(event_id, None)
        return None
    return thread


def _capture_error_record(record):
    """Preserve logging severity and exception context before console formatting."""
    import logging
    if record.levelno < logging.ERROR or record.name.startswith((
            "spacr.crashreport", "spacr.qt.ai")):
        return
    text = record.getMessage()
    exc = record.exc_info[1] if record.exc_info else None
    tb = str(record.exc_text or "") if exc else ""
    event_id = getattr(record, "_spacr_error_occurrence", "") or _new_error_occurrence(exc)
    record._spacr_error_occurrence = event_id
    _dispatch_error(text, exception=exc, module=record.name, traceback_text=tb,
                    event_id=event_id, origin="log")


def _capture_error_line(text, *, origin="terminal"):
    """Recognize an explicitly tagged terminal ERROR, never incidental prose."""
    if getattr(_ERROR_DISPATCH_LOCAL, "active", False):
        return
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_globals.get("__name__") == "logging":
            return
        frame = frame.f_back
    stamp = r"(?:(?:\d{4}-\d\d-\d\d[ T])?\d\d:\d\d:\d\d(?:[.,]\d+)?(?:Z|[+-]\d\d:\d\d)?\s+)?"
    for line in str(text).splitlines(keepends=True):
        buffers = getattr(_ERROR_DISPATCH_LOCAL, "traceback_buffers", {})
        pending = buffers.get(origin, "")
        if line.startswith("Traceback (most recent call last)"):
            pending = ""
        if pending or line.startswith("Traceback (most recent call last)"):
            pending = (pending + line)[-65536:]
            buffers[origin] = pending
            _ERROR_DISPATCH_LOCAL.traceback_buffers = buffers
            if not line[:1].isspace() and _EXCEPTION_LINE_RE.match(line.strip()):
                buffers[origin] = ""
                _dispatch_error(pending, traceback_text=pending, origin=origin)
        elif re.match(r"^\s*" + stamp + r"(?:\[ERROR\]|ERROR(?:\s|:))", line):
            _dispatch_error(line, origin=origin)


def _consume_native_crash(path):
    """Consume only a new bounded fatal section after startup, never in a signal."""
    try:
        path = Path(path)
        state = _read_state()
        key = "native_fatal_ack"
        size = path.stat().st_size
        policy = state.get("affirmed_policy") or {}
        if (policy.get("version") != _REPORTING_TERMS_VERSION
                or _parse_reset(policy.get("accepted_at")) is None):
            return None
        old = state.get(key) or {}
        offset = int(old.get("offset", 0)) if old.get("path") == str(path) else 0
        if time.time() - path.stat().st_mtime > QUEUE_RETENTION_S:
            return None
        if size < offset:
            offset = 0
        with path.open("rb") as handle:
            handle.seek(max(offset, size - MAX_LOG_BYTES))
            raw = handle.read(MAX_LOG_BYTES).decode("utf-8", "replace")
        _update_state(**{key: {"path": str(path), "offset": size}})
        fatals = list(re.finditer(r"Fatal Python error: ([^\n]+)", raw))
        if fatals:
            fatal = fatals[-1]
            message = "Native process termination: " + fatal.group(1)
            summary = public_summary(message=message, kind=KIND_FAULT)
            summary["exception_type"] = "NativeCrash"
            summary["module"] = "native"
            observed = raw[fatal.end():].split("\n\n", 1)[-1]
            frames = re.findall(r'  File "([^"\n]+)", line (\d+) in ([^\n]+)', observed)
            summary["frames"] = [_frame(file, function, line)
                                 for file, line, function in frames[:MAX_FRAMES]]
            record = f"{path.stat().st_ino}:{offset}:{size}"
            event_id = _new_error_occurrence(event_key=record)
            return _dispatch_error(message, summary=summary, event_id=event_id)
    except (OSError, ValueError, TypeError):
        pass
    return None


_NATIVE_CAPTURE_HANDLE = None
_NATIVE_EXCEPTION_HOOK = None


def _install_native_error_capture(path=None):
    """Prepare faulthandler output; consume old records only in normal startup."""
    global _NATIVE_CAPTURE_HANDLE, _NATIVE_EXCEPTION_HOOK
    import faulthandler
    from .logging_util import log_dir
    path = Path(path) if path is not None else log_dir() / "spacr-crash.log"
    _consume_native_crash(path)
    if _NATIVE_CAPTURE_HANDLE is not None:
        return str(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            old_size = path.stat().st_size
            with path.open("rb") as previous:
                previous.seek(old_size - MAX_LOG_BYTES)
                tail = previous.read(MAX_LOG_BYTES)
            ack = _read_state().get("native_fatal_ack") or {}
            old_offset = int(ack.get("offset", 0)) if ack.get("path") == str(path) else 0
            path.write_bytes(tail)
            _update_state(native_fatal_ack={
                "path": str(path), "offset": max(0, old_offset - old_size + len(tail))})
        handle = path.open("a", buffering=1, encoding="utf-8")
        handle.write(f"\n=== spaCR started (pid {os.getpid()}) ===\n")
        faulthandler.enable(file=handle, all_threads=True)
        _NATIVE_CAPTURE_HANDLE = handle
        if _NATIVE_EXCEPTION_HOOK is None:
            _NATIVE_EXCEPTION_HOOK = install_excepthook()
        return str(path)
    except OSError:
        return ""


def _report_anonymously(exception: BaseException, app: str = "") -> None:
    """Queue and send an anonymised summary of an unhandled exception.

    Never raises: it runs on the way out of a crash.
    """
    _dispatch_error(str(exception), exception=exception, module=app)


def build_parser() -> argparse.ArgumentParser:
    """Return the ``spacr-crashreport`` argument parser.

    :returns: the parser, built separately so a test can exercise the argument
        surface without running a collection.
    """
    parser = argparse.ArgumentParser(
        prog="spacr-crashreport",
        description="Bundle the log, settings, versions and last run into one "
                    "file you can attach to a bug report.")
    parser.add_argument("-o", "--output", default=None,
                        help="file or directory to write the zip to "
                             "(default: the spaCR log directory)")
    parser.add_argument("--run-id", default=None,
                        help="run to include (default: the most recent)")
    parser.add_argument("--db", default=None,
                        help="project database, e.g. "
                             "plate1/measurements/measurements.db")
    parser.add_argument("--settings", default=None,
                        help="settings csv/json that was run")
    parser.add_argument("--app", default="",
                        help="app the settings are for (mask, measure, ...)")
    parser.add_argument("--checkout", default=None,
                        help="checkout you believe you are running "
                             "(default: the current directory)")
    parser.add_argument("--note", default="",
                        help="what you were doing when it went wrong")
    parser.add_argument("--probe-gpu", action="store_true",
                        help="let the doctor allocate on the GPU to prove it "
                             "works; off by default so a probe failure cannot "
                             "be mistaken for the crash")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point. ``python -m spacr.crashreport``.

    :param argv: arguments, defaulting to ``sys.argv[1:]``.
    :returns: ``0`` when the bundle was written, ``1`` when it could not be —
        and never anything else; a section that could not be gathered does not
        change the exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        path = write_crash_report(
            args.output, run_id=args.run_id, db=args.db,
            settings=args.settings, checkout=args.checkout, app=args.app,
            note=args.note, probe_gpu=args.probe_gpu)
    except Exception as exc:                # noqa: BLE001 - CLI boundary
        print(f"spaCR could not write a crash report: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {path}")
    print("Attach it to your bug report. It holds the doctor's checks, the "
          "installed versions, the settings, this run's log and the tail of "
          "spacr.log. Open summary.txt first; manifest.json says what could "
          "not be gathered and why.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
