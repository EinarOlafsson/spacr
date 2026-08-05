"""``spacr-crashreport`` — everything a maintainer needs, in one attachable file.

A bug report that says "it crashed" costs a round trip to ask for the log, a
second one for the settings, a third for the versions, and by then the user has
re-run the pipeline and the evidence is gone. This module produces one ``.zip``
the user can drag into a GitHub issue, and it is assembled out of the pieces
spaCR already keeps rather than collected a second time:

* :mod:`spacr.doctor` runs 17 checks over the installation — the running
  checkout, duplicate installs, the GPU, the Cellpose version, the project
  database, the settings — and every non-``PASS`` row already carries the fix.
  A crash report that did not include it would be asking the maintainer to
  re-derive what one command already knows.
* :mod:`spacr.runctx` writes a per-run JSONL log keyed by run id, so "show me
  everything from the run that produced this" has an answer. The bundle carries
  that run's log verbatim, not a filtered summary of it.
* :mod:`spacr.errors` stamps each run's status onto the artifact it wrote, so
  the report can say whether the last run *finished* — which is a different
  question from whether it crashed just now, and the more useful one when a
  user reports numbers rather than a traceback.

Design rules, all of them load-bearing:

**Nothing here may raise.** A crash reporter that crashes while reporting a
crash destroys the only evidence there was. Every collector runs through
:func:`_collect`, which turns a failure into a manifest entry naming what could
not be gathered and why. This is the one place in the codebase where catching
``Exception`` and carrying on is the correct behaviour rather than a swallowed
error, and the reason it is correct is that the failure is *recorded in the
output* — the manifest is part of the bundle, so a missing section is visible
to the maintainer instead of silently absent.

**Bounded size.** ``spacr.log`` rotates but a single run can still write
hundreds of megabytes. The bundle takes the *tail* of it, capped by
:data:`MAX_LOG_BYTES`, and the manifest records how much was dropped. An
attachment nobody can upload is not evidence.

**Nothing secret.** Environment variables are included because ``SPACR_*``,
``CUDA_*`` and ``PATH`` explain a large fraction of "works here, not there" —
but a value whose *name* looks like a credential is replaced with
``"<redacted>"`` and listed by name in the manifest, so the user can see what
was withheld and the maintainer can see that something was. Absolute paths are
kept: they name the checkout, the plate and the database, and a report with
them stripped cannot be acted on.

**Deterministic layout.** Same file names, same order, every time, so a
maintainer opening their tenth report knows where to look.

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
import io
import json
import os
import platform
import sys
import traceback
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
    """Everything gathered, before it is written anywhere.

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

        Distinct from :attr:`omitted`, and the distinction matters: a section
        that had nothing to collect is an ordinary answer, and listing the two
        together under one alarming heading would train a reader to skip both.

        :returns: section names, in collection order. Empty on a clean report.
        """
        return [name for name, entry in self.manifest.get("sections", {}).items()
                if entry.get("status") == "failed"]

    @property
    def omitted(self) -> List[str]:
        """Names of the sections that had nothing to collect.

        No settings file was named, the project has no database, the run wrote
        no warnings. Each is a fact about the invocation rather than a failure,
        and each is still listed, because a maintainer must be able to tell
        "there were no warnings" from "the warnings were not gathered".

        :returns: section names, in collection order.
        """
        return [name for name, entry in self.manifest.get("sections", {}).items()
                if entry.get("status") == "empty"]

    def summary(self) -> str:
        """One short block naming the run, the versions and what is missing.

        This is also ``summary.txt``, the first file in the bundle, because a
        maintainer opening a zip should not have to guess which file to read
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
        # BaseException rather than Exception: a collector that trips a
        # MemoryError or a RecursionError must still leave a report behind,
        # and re-raising here would lose every section gathered before it.
        # KeyboardInterrupt is re-raised, because a user pressing Ctrl-C
        # while a report is being written means stop, not "record that".
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
            # The seek lands mid-line; drop the partial first line rather than
            # present half a message as a whole one.
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
        the answer is appended to. The reason this exists rather than a bare
        ``return ''``: "no run was found" and "the run directory could not be
        read" are different answers, and a report that cannot tell them apart
        sends the maintainer looking for a run that was there all along.
        :func:`collect` passes one and puts it in the manifest.
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

    Both, on purpose: the text is what a human reads in the issue thread, and
    the JSON is what a maintainer greps across a hundred reports. They come
    from one ``run_checks`` call so they cannot disagree.

    ``run_checks`` itself goes *inside* :func:`_collect` rather than beside it.
    It already turns a check that raises into an ``ERROR`` row, but importing
    :mod:`spacr.doctor`, building its :class:`~spacr.doctor.Context` and
    resolving the checkout are all outside that guarantee — and a report is
    exactly the situation in which an import fails.
    """
    held: Dict[str, Any] = {}

    def gather_text() -> str:
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
    """Gather everything worth attaching to a bug report.

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
        the current one, which is what makes "am I running this code" answerable.
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
        # Inside the collector, not beside it. Everything in this function
        # that runs outside _collect is a way for the whole bundle to be lost,
        # and two of them were found here by the tests that say so.
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
        else is used as the file name. Defaults to
        ``<log dir>/spacr-crashreport-<...>.zip``, which exists and is
        writable on every machine spaCR has ever logged on.
    :param kwargs: passed straight to :func:`collect`.
    :returns: the absolute path of the file written.
    :raises OSError: only when the destination itself cannot be written, which
        is the one failure this function must not hide -- a report the caller
        is told about but which is not on disk is worse than an error.

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
        # summary first, manifest last: the two files a reader wants at the
        # top and the bottom of the listing.
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
    it saying where the bundle is.

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
            if path:
                print(f"\nspaCR wrote a crash report to {path}\n"
                      f"Attach it to a bug report — it holds the log, the "
                      f"settings, the versions and this run.", file=sys.stderr)
        previous(kind, value, tb)

    sys.excepthook = hook
    return hook


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
        and never anything else, because a crash reporter that exits non-zero
        for a section it could not gather would train people to ignore it.
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


if __name__ == "__main__":               # pragma: no cover - module entry
    raise SystemExit(main())
