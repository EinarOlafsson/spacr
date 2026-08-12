"""One-click, shareable report for a finished spaCR run.

Why this exists
---------------
A spaCR run leaves behind a plate folder: a SQLite database, a ``qc``
folder, a ``results`` tree of PDFs and CSVs, a ``settings`` folder, and —
somewhere in ``~/.spacr/runs`` — a journal entry recording the exact
settings and package versions that produced all of it. Every piece of that
is legible to somebody with spaCR installed and a terminal. None of it is
legible to the collaborator who asks "so did it work?".

This module walks that folder, gathers what is *actually there*, and writes
a single self-contained HTML file (and, optionally, a PDF) that answers the
question without spaCR, without Python and without the original machine.

Three rules govern what it does, because a report that breaks any of them
is worse than no report:

**A missing section is stated, never omitted.**
    If segmentation QC was never run, the report contains a "Segmentation
    QC" heading that says *not run*. A reader cannot distinguish "clean"
    from "never checked" if unchecked things silently vanish, and that is
    exactly how a partial run gets forwarded as though it were complete.

**Failure goes at the top.**
    :mod:`spacr.errors` stamps every artifact with a ``run_status``
    recording how many items failed. If anything failed — or if nothing was
    stamped at all, so completeness is simply unknown — that is the first
    thing in the document, not an appendix entry.

**Nothing is recomputed.**
    Every number here was produced by the pipeline and read back off disk.
    The one exception is deliberate and narrow: the segmentation scorecard
    CSV is re-fed to :func:`spacr.seg_qc.summarize_qc`, the same function
    that printed the verdict at run time, so the plate verdict in the report
    is the verdict spaCR gave — not a second opinion. No p-value, no effect
    size and no aggregate statistic is invented by this module. Where a
    check exists but has to be run on demand (plate edge effects,
    inter-annotator κ), the report says so and names the tool.

What it reads
-------------
===================================  =============================================
source                               section
===================================  =============================================
``spacr.errors`` run-status stamps   Run status (stamps on ``*.db`` and
                                     ``*.run_status.json`` sidecars)
``~/.spacr/runs`` journal            Provenance + versions
``<src>/qc/segmentation_qc_*``       Segmentation QC
``<src>/qc/`` layout exports         Plate QC / edge effects
``<src>/results``, ``<src>/figure``  Key figures
``<src>/results/**/*.csv``           Statistics
``<src>/settings/*.csv``, journal    Settings
``<src>/measurements/*.db``          Appendix (feature dictionary, annotations)
===================================  =============================================

Usage::

    from spacr.report import build_report
    build_report('/data/plate1', '/tmp/plate1_report.html')
    build_report('/data/plate1', '/tmp/reports', fmt='both')

Or from the GUI: **Tools → Report**.

Output formats
--------------
The HTML is the real deliverable: one file, no external requests, images
base64-embedded, CSS in a ``<style>`` block, no JavaScript at all. It opens
from a USB stick on a machine with no network.

The PDF is composed with :class:`matplotlib.backends.backend_pdf.PdfPages`
— a monospace transcription of the same content plus one page per embedded
figure. spaCR has no HTML-to-PDF engine among its dependencies and this
module refuses to add one, so the PDF is a faithful *summary*, not a
rendering of the HTML. Send the HTML when you can.
"""
from __future__ import annotations

import base64
import csv
import html as _html
import io
import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

__all__ = [
    "Figure",
    "Table",
    "Section",
    "Report",
    "collect_report",
    "render_html",
    "render_text",
    "write_html",
    "write_pdf",
    "build_report",
    "pdf_page_count",
    "SECTION_KEYS",
    "STATUS_OK",
    "STATUS_MISSING",
    "STATUS_PROBLEM",
    "DEFAULT_MAX_FIGURES",
    "DEFAULT_MAX_FIGURE_PX",
    "DEFAULT_MAX_TABLE_ROWS",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Figures embedded before the rest are listed by name only. Every embedded
#: PNG lands in the file as base64, which is 4/3 of its byte size, so the
#: cap is what keeps a shareable report shareable over email.
DEFAULT_MAX_FIGURES = 20

#: Longest edge, in pixels, a figure is downscaled to before embedding.
DEFAULT_MAX_FIGURE_PX = 1400

#: Rows shown from any single CSV preview.
DEFAULT_MAX_TABLE_ROWS = 25

#: Rows shown from a settings CSV before the rest are counted.
DEFAULT_MAX_SETTINGS_ROWS = 500

#: Feature-dictionary rows shown in the appendix.
DEFAULT_MAX_DICT_ROWS = 60

#: Recent journal runs scanned when looking for the ones that made ``src``.
DEFAULT_JOURNAL_LIMIT = 200

#: Hard ceiling on directory entries visited by any single walk. A plate
#: folder can hold a million PNG crops; the report must not read them all
#: to tell you the run failed.
WALK_BUDGET = 40000

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_PROBLEM = "problem"

#: Section keys, in the order a reader needs them.
SECTION_KEYS: Tuple[str, ...] = (
    "run_status",
    "provenance",
    "segmentation_qc",
    "plate_qc",
    "figures",
    "statistics",
    "settings",
    "appendix",
)

SECTION_TITLES: Dict[str, str] = {
    "run_status": "Run status",
    "provenance": "What was run, and when",
    "segmentation_qc": "Segmentation QC",
    "plate_qc": "Plate QC and edge effects",
    "figures": "Key figures",
    "statistics": "Statistics and result tables",
    "settings": "Settings",
    "appendix": "Appendix",
}

#: Image suffixes that can be embedded in the HTML.
RASTER_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

#: Figure suffixes spaCR writes that cannot be embedded without a
#: rasteriser. ``spacr.io._save_figure`` and ``spacr.timelapse.save_figure``
#: both write PDF, so this is the common case, not the exotic one.
VECTOR_SUFFIXES = (".pdf", ".svg", ".eps", ".ps")

DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")

#: Folders under ``src`` that hold figures and result tables.
RESULT_DIRS: Tuple[str, ...] = ("results", "figure", "figures", "plots", "plot")

#: Folders that hold bulk pixel data rather than results. Never descended
#: into, and reported as one line each in the file inventory.
BULK_DIRS = frozenset({
    "orig", "stack", "masks", "merged", "norm_channel_stack", "datasets",
    "data", "train", "test", "pngs", "png_images", "movies", "tiff", "tifs",
    "cell_mask_stack", "nucleus_mask_stack", "pathogen_mask_stack",
    "organelle_mask_stack", "__pycache__", ".git", ".ipynb_checkpoints",
})

#: Overall report statuses.
_STATUS_LABELS = {
    "complete": "Complete — every stamped step processed every item.",
    "partial": "PARTIAL — items failed. The numbers below describe a subset.",
    "failed": "FAILED — a journalled run raised before it finished.",
    "unknown": "Unknown — nothing in this folder records whether the run "
               "completed. Treat the results as unverified.",
    "empty": "Nothing to report — no spaCR output was found here.",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _esc(value: Any) -> str:
    """HTML-escape ``value``.

    Every scrap of user-derived text — well names, file names, settings
    values, tracebacks — passes through here before it reaches the page. A
    field called ``A01<script>`` must render as text, not run.

    :param value: anything; ``None`` becomes an empty string.
    :returns: an HTML-safe string.
    """
    if value is None:
        return ""
    return _html.escape(str(value), quote=True)


def _fmt_bytes(n: Any) -> str:
    """Render a byte count as ``1.4 MB``."""
    try:
        size = float(n)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _fmt_time(value: Any) -> str:
    """Render an ISO timestamp or epoch as a readable UTC string."""
    if value in (None, ""):
        return "-"
    text = str(value)
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            stamp = datetime.fromtimestamp(float(text), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return text
    return stamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_elapsed(value: Any) -> str:
    """Render seconds as ``3 m 12 s``."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "-"
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 90:
        return f"{minutes:.0f} m {rest:.0f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:.0f} h {minutes:.0f} m"


def _same_path(a: Any, b: Any) -> bool:
    """True when two path-ish values point at the same place."""
    try:
        return os.path.realpath(str(a)) == os.path.realpath(str(b))
    except (TypeError, ValueError, OSError):
        return False


def _iter_dir_files(root: Path, budget: int = WALK_BUDGET,
                    recurse: bool = True) -> Tuple[List[Path], bool]:
    """List files under ``root``, bounded.

    Never descends into :data:`BULK_DIRS`; a plate's ``orig`` folder holds
    the raw images, not the results, and walking it can take minutes.

    :param root: directory to walk.
    :param budget: hard ceiling on files collected.
    :param recurse: descend into subdirectories.
    :returns: ``(files, truncated)`` — ``truncated`` is True when the
        budget stopped the walk before it finished.
    """
    files: List[Path] = []
    if not root.is_dir():
        return files, False
    stack = [root]
    while stack:
        current = stack.pop(0)
        try:
            entries = sorted(os.scandir(current), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if len(files) >= budget:
                return files, True
            try:
                if entry.is_dir(follow_symlinks=False):
                    if recurse and entry.name not in BULK_DIRS:
                        stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files.append(Path(entry.path))
            except OSError:
                continue
    return files, False


def _dir_stats(root: Path, budget: int = WALK_BUDGET) -> Tuple[int, int, bool]:
    """Return ``(n_files, total_bytes, truncated)`` for a directory tree.

    Descends everywhere, including bulk folders — the point of the file
    inventory is to say how big the plate is.
    """
    n_files = 0
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if n_files >= budget:
                return n_files, total, True
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    n_files += 1
                    total += entry.stat().st_size
            except OSError:
                continue
    return n_files, total, False


def _read_csv_head(path: Path, max_rows: int,
                   max_bytes: int = 8 << 20) -> Tuple[List[str], List[List[str]], int]:
    """Read the header and first ``max_rows`` data rows of a CSV.

    Uses :mod:`csv` rather than pandas: the report must survive a
    half-written result file, and a ragged row is a formatting problem,
    not an exception.

    A file that is damaged rather than merely ragged — an embedded NUL
    from a half-flushed write, a cell longer than
    :func:`csv.field_size_limit` — makes :mod:`csv` itself raise. That is
    caught here and the scan stops: the rows parsed before the damage are
    still worth showing, and losing the entire report to one corrupt CSV
    is the opposite of what this function is for.

    :param path: CSV file.
    :param max_rows: rows to keep.
    :param max_bytes: stop counting rows past this many bytes read.
    :returns: ``(columns, rows, n_total_rows)``. ``n_total_rows`` is the
        number of data rows seen, which equals the file's row count unless
        ``max_bytes`` cut the scan short, or the file is damaged past some
        row.
    """
    columns: List[str] = []
    rows: List[List[str]] = []
    n_total = 0
    read = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            for i, row in enumerate(reader):
                if i == 0:
                    columns = [str(c) for c in row]
                    continue
                # Python's CSV reader no longer rejects embedded NUL bytes on
                # every supported version. Treat them as the documented
                # half-written-file boundary before counting or displaying
                # the corrupt row.
                if any("\x00" in cell for cell in row):
                    break
                n_total += 1
                read += sum(len(c) for c in row) + len(row)
                if len(rows) < max_rows:
                    rows.append([str(c) for c in row])
                elif read > max_bytes:
                    break
        except csv.Error:
            pass
    return columns, rows, n_total


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Figure:
    """One figure found under ``src``.

    :ivar path: where the figure lives on disk.
    :ivar title: caption shown above it.
    :ivar mime: MIME type of :attr:`data`.
    :ivar data: the bytes embedded in the HTML, or ``None`` when the figure
        was found but not embedded.
    :ivar reason: why it was not embedded, when it was not.
    :ivar n_bytes: size of the file on disk.
    """
    path: Path
    title: str = ""
    mime: str = "image/png"
    data: Optional[bytes] = None
    reason: str = ""
    n_bytes: int = 0

    @property
    def embedded(self) -> bool:
        """True when the bytes are in the report."""
        return bool(self.data)

    def data_uri(self) -> str:
        """Return the ``data:`` URI for :attr:`data`.

        :raises ValueError: when the figure was not embedded.
        """
        if not self.data:
            raise ValueError(f"{self.path} was not embedded: {self.reason}")
        return f"data:{self.mime};base64," + base64.b64encode(self.data).decode("ascii")


@dataclass
class Table:
    """A rectangle of already-stringified cells.

    :ivar columns: header row.
    :ivar rows: body rows, already truncated to what will be shown.
    :ivar caption: one line above the table.
    :ivar n_total_rows: rows the source had, so truncation can be stated.
    """
    columns: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    caption: str = ""
    n_total_rows: int = 0

    def __post_init__(self) -> None:
        if not self.n_total_rows:
            self.n_total_rows = len(self.rows)

    @property
    def n_omitted(self) -> int:
        """Rows the source had that are not shown."""
        return max(0, int(self.n_total_rows) - len(self.rows))


@dataclass
class Section:
    """One chapter of the report.

    A section is always present, even when its evidence is not: a section
    whose :attr:`status` is :data:`STATUS_MISSING` renders with its heading
    and a sentence explaining what was looked for and not found.

    :ivar title: heading text.
    :ivar body_html: pre-escaped HTML for the section body.
    :ivar figures: embedded (or listed) figures.
    :ivar table: the section's primary table, if it has one.
    :ivar notes: caveats rendered as a bullet list under the body.
    :ivar key: stable id, one of :data:`SECTION_KEYS`.
    :ivar status: :data:`STATUS_OK`, :data:`STATUS_MISSING` or
        :data:`STATUS_PROBLEM`.
    :ivar text_lines: plain-text rendering used for the PDF, so the PDF is
        not a re-parse of the HTML.
    """
    title: str
    body_html: str = ""
    figures: List[Figure] = field(default_factory=list)
    table: Optional[Table] = None
    notes: List[str] = field(default_factory=list)
    key: str = ""
    status: str = STATUS_OK
    text_lines: List[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """True when the section's evidence exists on disk."""
        return self.status != STATUS_MISSING


@dataclass
class Report:
    """Everything :func:`collect_report` gathered.

    :ivar src: the run folder the report describes.
    :ivar title: document title.
    :ivar generated_utc: ISO timestamp of collection.
    :ivar sections: chapters in reading order, one per :data:`SECTION_KEYS`.
    :ivar status: ``complete`` / ``partial`` / ``failed`` / ``unknown`` /
        ``empty``.
    :ivar status_detail: one line expanding on :attr:`status`.
    :ivar spacr_version: version of the spaCR that wrote the report.
    :ivar n_figures_found: figures discovered under ``src``.
    :ivar n_figures_embedded: figures actually in the file.
    """
    src: Path
    title: str = "spaCR report"
    generated_utc: str = ""
    sections: List[Section] = field(default_factory=list)
    status: str = "unknown"
    status_detail: str = ""
    spacr_version: str = ""
    n_figures_found: int = 0
    n_figures_embedded: int = 0

    def section(self, key: str) -> Optional[Section]:
        """Return the section with ``key``, or None."""
        for sec in self.sections:
            if sec.key == key:
                return sec
        return None

    @property
    def missing_sections(self) -> List[str]:
        """Keys of sections whose evidence was not found."""
        return [s.key for s in self.sections if s.status == STATUS_MISSING]

    @property
    def found_sections(self) -> List[str]:
        """Keys of sections that have something to show."""
        return [s.key for s in self.sections if s.status != STATUS_MISSING]

    @property
    def has_failures(self) -> bool:
        """True when something in this run is known to have failed."""
        return self.status in ("partial", "failed")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _find_artifacts(src: Path) -> Dict[str, List[Path]]:
    """Locate the files the report reads, once, so no collector re-walks.

    :param src: the plate folder.
    :returns: dict with keys ``databases``, ``sidecars``, ``qc_csv``,
        ``qc_flags``, ``layout_csv``, ``settings_csv``, ``result_csv``,
        ``raster``, ``vector``, ``truncated``.
    """
    found: Dict[str, Any] = {
        "databases": [], "sidecars": [], "qc_csv": [], "qc_flags": [],
        "layout_csv": [], "settings_csv": [], "result_csv": [],
        "raster": [], "vector": [], "truncated": False,
    }
    if not src.is_dir():
        return found

    # Top level + the folders that hold results. measurements/ holds the
    # database; qc/ the scorecards; settings/ the settings CSVs.
    roots: List[Tuple[Path, bool]] = [(src, False)]
    for name in ("measurements", "qc", "settings", "model") + RESULT_DIRS:
        child = src / name
        if child.is_dir():
            roots.append((child, True))

    seen: set = set()
    for root, recurse in roots:
        files, truncated = _iter_dir_files(root, recurse=recurse)
        found["truncated"] = found["truncated"] or truncated
        for path in files:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            suffix = path.suffix.lower()
            name = path.name
            if suffix in DB_SUFFIXES:
                found["databases"].append(path)
            elif name.endswith(".run_status.json"):
                found["sidecars"].append(path)
            elif suffix in RASTER_SUFFIXES:
                found["raster"].append(path)
            elif suffix in VECTOR_SUFFIXES:
                found["vector"].append(path)
            elif suffix == ".csv":
                parent = path.parent.name
                if parent == "qc" and name.startswith("segmentation_qc_"):
                    found["qc_csv"].append(path)
                elif parent == "qc":
                    found["layout_csv"].append(path)
                elif parent == "settings":
                    found["settings_csv"].append(path)
                else:
                    found["result_csv"].append(path)
            elif suffix == ".json" and path.parent.name == "qc":
                found["qc_flags"].append(path)

    for key in ("databases", "sidecars", "qc_csv", "qc_flags", "layout_csv",
                "settings_csv", "result_csv", "raster", "vector"):
        found[key] = sorted(found[key])
    return found


def _load_journal_runs(src: Path, run_dirs: Optional[Sequence[Any]],
                       search_journal: bool,
                       journal_limit: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return the journalled runs whose ``src`` setting points at ``src``.

    :param src: the plate folder.
    :param run_dirs: explicit run folders to read instead of scanning the
        journal. Tests pass this; the GUI does not.
    :param search_journal: when no ``run_dirs`` are given, scan
        ``~/.spacr/runs``.
    :param journal_limit: how many recent runs to consider.
    :returns: ``(records, problems)``. Records are newest-first dicts with
        ``dir``, ``manifest``, ``settings``, ``app_key``, ``status``,
        ``start_utc``, ``elapsed_s``.
    """
    problems: List[str] = []
    records: List[Dict[str, Any]] = []
    try:
        from . import run_journal as journal
    except Exception as exc:
        return records, [f"run journal unavailable ({exc.__class__.__name__})"]

    candidates: List[Path] = []
    if run_dirs is not None:
        for ref in run_dirs:
            try:
                candidates.append(journal.resolve_run_dir(ref))
            except Exception:
                problems.append(f"no such run folder: {ref}")
    elif search_journal:
        try:
            candidates = [entry["dir"] for entry in journal.recent_runs(journal_limit)]
        except Exception as exc:
            problems.append(
                f"could not read the run journal ({exc.__class__.__name__})")

    for run_dir in candidates:
        manifest: Dict[str, Any] = {}
        settings: Dict[str, Any] = {}
        try:
            raw = (Path(run_dir) / "manifest.json").read_text()
            loaded = json.loads(raw)
            manifest = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            manifest = {}
        try:
            settings = journal.load_run_settings(Path(run_dir)) or {}
        except Exception:
            settings = {}
        if run_dirs is None and not _settings_point_at(settings, src):
            continue
        records.append({
            "dir": Path(run_dir),
            "manifest": manifest,
            "settings": settings if isinstance(settings, dict) else {},
            "app_key": manifest.get("app_key", "?"),
            "status": manifest.get("status", "?"),
            "start_utc": manifest.get("start_utc", ""),
            "elapsed_s": manifest.get("elapsed_s"),
            "env": manifest.get("env") if isinstance(manifest.get("env"), dict) else {},
        })
    records.sort(key=lambda r: str(r.get("start_utc") or ""), reverse=True)
    return records, problems


def _settings_point_at(settings: Dict[str, Any], src: Path) -> bool:
    """True when a run's ``src`` setting names ``src``.

    ``src`` is a string in most apps and a list in the ones that accept
    several plates, so both shapes are checked.
    """
    if not isinstance(settings, dict):
        return False
    value = settings.get("src")
    values: Iterable[Any]
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str) and value.strip().startswith("["):
        # A CSV round-trip turns a list of plates into its repr.
        try:
            import ast
            parsed = ast.literal_eval(value)
            values = parsed if isinstance(parsed, (list, tuple)) else [value]
        except (ValueError, SyntaxError):
            values = [value]
    else:
        values = [value]
    return any(_same_path(v, src) for v in values if v)


# ---------------------------------------------------------------------------
# Section: run status  (first, always, because failure must not be buried)
# ---------------------------------------------------------------------------

def _read_stamps(paths: Sequence[Path]) -> Tuple[List[Tuple[Path, Dict[str, Any]]], List[str]]:
    """Read every :meth:`spacr.errors.RunLedger.stamp` on ``paths``."""
    stamps: List[Tuple[Path, Dict[str, Any]]] = []
    problems: List[str] = []
    try:
        from .errors import read_run_status
    except Exception as exc:
        return stamps, [f"spacr.errors unavailable ({exc.__class__.__name__})"]
    for path in paths:
        try:
            for record in read_run_status(path) or []:
                if isinstance(record, dict):
                    stamps.append((path, record))
        except Exception as exc:
            problems.append(f"{path.name}: unreadable run status "
                            f"({exc.__class__.__name__})")
    return stamps, problems


def _collect_run_status(src: Path, artifacts: Dict[str, Any],
                        runs: Sequence[Dict[str, Any]]) -> Tuple[Section, str, str]:
    """Build the run-status section and decide the report's overall verdict.

    :returns: ``(section, status, status_detail)``.
    """
    section = Section(title=SECTION_TITLES["run_status"], key="run_status")
    lines: List[str] = []

    if not src.exists():
        section.status = STATUS_PROBLEM
        section.body_html = (
            f"<p class='bad'>The folder <code>{_esc(src)}</code> does not "
            f"exist. Nothing could be read.</p>")
        section.text_lines = [f"The folder {src} does not exist."]
        return section, "empty", _STATUS_LABELS["empty"]

    stamp_paths = list(artifacts.get("databases") or []) + list(artifacts.get("sidecars") or [])
    stamps, problems = _read_stamps(stamp_paths)
    section.notes.extend(problems)

    failed_runs = [r for r in runs if str(r.get("status")) == "failed"]
    n_failed_items = sum(int(rec.get("n_failed") or 0) for _, rec in stamps)
    partial = [
        (path, rec) for path, rec in stamps
        if int(rec.get("n_failed") or 0) > 0 or str(rec.get("status")) == "partial"
    ]

    if failed_runs:
        status = "failed"
    elif partial:
        status = "partial"
    elif stamps:
        status = "complete"
    else:
        status = "unknown"
    detail = _STATUS_LABELS[status]

    body: List[str] = [f"<p class='verdict {status}'>{_esc(detail)}</p>"]
    lines.append(detail)

    if failed_runs:
        body.append("<p>Journalled runs that raised before finishing:</p><ul>")
        for run in failed_runs:
            trace = str((run.get("manifest") or {}).get("traceback") or "").strip()
            last = trace.splitlines()[-1] if trace else "no traceback recorded"
            body.append(
                f"<li><code>{_esc(run['dir'].name)}</code> "
                f"({_esc(run.get('app_key'))}) — {_esc(last)}</li>")
            lines.append(f"  FAILED run {run['dir'].name} ({run.get('app_key')}): {last}")
        body.append("</ul>")

    if stamps:
        rows = []
        for path, rec in stamps:
            rows.append([
                path.name,
                str(rec.get("name") or "-"),
                str(rec.get("status") or "-"),
                str(rec.get("n_attempted") or 0),
                str(rec.get("n_succeeded") or 0),
                str(rec.get("n_failed") or 0),
                _fmt_time(rec.get("stamped_utc")),
            ])
        section.table = Table(
            columns=["artifact", "step", "status", "attempted", "succeeded",
                     "failed", "stamped"],
            rows=rows,
            caption=f"{len(stamps)} run-status stamp(s) found under {src.name}",
        )
        lines.append(f"  {len(stamps)} run-status stamp(s); "
                     f"{n_failed_items} failed item(s) in total.")
    else:
        section.status = STATUS_MISSING
        body.append(
            "<p><strong>No run-status stamp was found.</strong> spaCR stamps "
            "its outputs with a <code>run_status</code> record (a table inside "
            "a database, or a <code>.run_status.json</code> sidecar) saying how "
            "many items it attempted and how many failed. Nothing here carries "
            "one, so this report cannot tell you whether every field was "
            "processed. Absence of a stamp is not evidence of success.</p>")
        lines.append("  No run-status stamp was found — completeness is unknown.")

    if partial:
        body.append("<p class='bad'>Failures recorded, worst first:</p><ul>")
        for path, rec in sorted(partial, key=lambda pr: -int(pr[1].get("n_failed") or 0))[:20]:
            summary = str(rec.get("summary") or "").strip()
            body.append(
                f"<li><code>{_esc(path.name)}</code> · "
                f"{_esc(rec.get('name') or 'step')}: "
                f"{_esc(rec.get('n_failed'))} of {_esc(rec.get('n_attempted'))} "
                f"item(s) failed"
                + (f" — {_esc(summary)}" if summary else "") + "</li>")
            lines.append(
                f"  {path.name} · {rec.get('name')}: {rec.get('n_failed')} of "
                f"{rec.get('n_attempted')} item(s) failed")
            for failure in (rec.get("failures") or [])[:5]:
                if isinstance(failure, dict):
                    body.append(
                        f"<li class='sub'><code>{_esc(failure.get('item'))}</code> "
                        f"[{_esc(failure.get('stage'))}] {_esc(failure.get('error'))}</li>")
        body.append("</ul>")
        section.status = STATUS_PROBLEM

    section.body_html = "\n".join(body)
    section.text_lines = lines
    return section, status, detail


# ---------------------------------------------------------------------------
# Section: provenance + versions
# ---------------------------------------------------------------------------

_ENV_ORDER = ("spacr", "spacr_git", "python", "platform", "torch", "torchvision",
              "cellpose", "numpy", "scipy", "pandas", "scikit_image",
              "scikit_learn", "pyside6")


def _collect_provenance(src: Path, runs: Sequence[Dict[str, Any]],
                        problems: Sequence[str]) -> Section:
    """Which pipeline ran, when, and against which package versions."""
    section = Section(title=SECTION_TITLES["provenance"], key="provenance")
    body: List[str] = []
    lines: List[str] = [f"source folder : {src}"]
    body.append(f"<p>Source folder <code>{_esc(src)}</code></p>")
    section.notes.extend(problems)

    if runs:
        rows = []
        for run in runs:
            rows.append([
                run["dir"].name,
                str(run.get("app_key") or "?"),
                str(run.get("status") or "?"),
                _fmt_time(run.get("start_utc")),
                _fmt_elapsed(run.get("elapsed_s")),
                str((run.get("env") or {}).get("spacr") or "-"),
                str(len(run.get("settings") or {})),
            ])
            lines.append(
                f"  {run['dir'].name}  {run.get('app_key')}  "
                f"{run.get('status')}  {_fmt_time(run.get('start_utc'))}  "
                f"{_fmt_elapsed(run.get('elapsed_s'))}")
        section.table = Table(
            columns=["run id", "pipeline", "status", "started", "elapsed",
                     "spaCR", "settings"],
            rows=rows,
            caption=f"{len(runs)} journalled run(s) recorded against this folder",
        )
        env = runs[0].get("env") or {}
        source_of_env = f"recorded by run {runs[0]['dir'].name}"
    else:
        section.status = STATUS_MISSING
        body.append(
            "<p><strong>No journalled run was found for this folder.</strong> "
            "spaCR records every pipeline invocation under "
            "<code>~/.spacr/runs</code>; none of those records names this "
            "folder as its source. The versions below therefore describe the "
            "machine that generated <em>this report</em>, not the machine that "
            "produced the data.</p>")
        lines.append("  No journalled run was found for this folder.")
        env = {}
        source_of_env = "the machine that generated this report"

    if not env:
        try:
            from .version import get_version_info
            info = get_version_info()
            env = {
                "spacr": info.get("spacr_version", "unknown"),
                "python": info.get("python_version", "unknown"),
                "platform": info.get("platform", "unknown"),
                "torch": info.get("torch_version", "unknown"),
            }
        except Exception:
            env = {}

    if env:
        items = [(k, env[k]) for k in _ENV_ORDER if k in env]
        items += [(k, v) for k, v in sorted(env.items()) if k not in _ENV_ORDER]
        body.append(f"<p class='muted'>Versions, {_esc(source_of_env)}:</p>")
        body.append("<dl class='env'>")
        lines.append(f"  versions ({source_of_env}):")
        for key, value in items:
            body.append(f"<dt>{_esc(key)}</dt><dd>{_esc(value)}</dd>")
            lines.append(f"    {key:<14} {value}")
        body.append("</dl>")

    models = {}
    for run in runs:
        models.update((run.get("manifest") or {}).get("model_hashes") or {})
    if models:
        body.append("<p class='muted'>Model checkpoints fingerprinted "
                    "during the run:</p><ul>")
        lines.append("  model checkpoints:")
        for name, digest in sorted(models.items()):
            body.append(f"<li><code>{_esc(name)}</code> → {_esc(digest)}</li>")
            lines.append(f"    {name} -> {digest}")
        body.append("</ul>")

    section.body_html = "\n".join(body)
    section.text_lines = lines
    return section


# ---------------------------------------------------------------------------
# Section: segmentation QC
# ---------------------------------------------------------------------------

def _field_qcs_from_csv(path: Path) -> Tuple[List[Any], Optional[str]]:
    """Rebuild :class:`spacr.seg_qc.FieldQC` objects from a scorecard CSV.

    The report re-feeds these to :func:`spacr.seg_qc.summarize_qc` — the
    function that printed the verdict at run time — rather than deriving a
    second opinion from the same rows.

    :param path: ``<src>/qc/segmentation_qc_<object>.csv``.
    :returns: ``(field_qcs, error)``.
    """
    try:
        from .seg_qc import FieldQC
    except Exception as exc:
        return [], f"spacr.seg_qc unavailable ({exc.__class__.__name__})"

    reserved = {"field", "object_type", "n_objects", "severity", "flags", "note"}
    out: List[Any] = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if any(
                    "\x00" in str(value)
                    for value in row.values()
                    if value is not None
                ):
                    return [], (
                        f"{path.name} is not readable as CSV "
                        "(embedded NUL byte)"
                    )
                metrics: Dict[str, float] = {}
                for key, value in row.items():
                    if key in reserved or key is None:
                        continue
                    try:
                        metrics[key] = float(value)
                    except (TypeError, ValueError):
                        metrics[key] = float("nan")
                try:
                    n_objects = int(float(row.get("n_objects") or 0))
                except (TypeError, ValueError):
                    n_objects = 0
                flags = [f for f in str(row.get("flags") or "").split(";") if f]
                out.append(FieldQC(
                    field=str(row.get("field") or ""),
                    object_type=str(row.get("object_type") or ""),
                    n_objects=n_objects,
                    flags=flags,
                    metrics=metrics,
                    severity=str(row.get("severity") or "ok"),
                    note=str(row.get("note") or ""),
                ))
    except OSError as exc:
        return [], f"{path.name} unreadable ({exc.__class__.__name__})"
    except csv.Error as exc:
        # A damaged scorecard is reported as unreadable rather than
        # summarised from the rows that did parse: a plate verdict derived
        # from half a scorecard is a different verdict, and this module
        # does not invent one.
        return [], f"{path.name} is not readable as CSV ({exc})"
    return out, None


def _collect_segmentation_qc(src: Path, artifacts: Dict[str, Any],
                             max_rows: int) -> Section:
    """The plate verdict :mod:`spacr.seg_qc` reached, read back off disk."""
    section = Section(title=SECTION_TITLES["segmentation_qc"], key="segmentation_qc")
    cards = list(artifacts.get("qc_csv") or [])
    if not cards:
        section.status = STATUS_MISSING
        section.body_html = (
            "<p><strong>Segmentation QC: not run.</strong> No "
            "<code>qc/segmentation_qc_*.csv</code> scorecard exists under this "
            "folder, so no mask has been scored. That is not the same as "
            "clean — the masks may be perfect or may have collapsed; nothing "
            "here has looked. Set <code>seg_qc='report'</code> before the mask "
            "step, or run <code>spacr.seg_qc.run_segmentation_qc()</code> "
            "against the mask stack.</p>")
        section.text_lines = [
            "Segmentation QC: NOT RUN — no scorecard exists under qc/.",
            "  This is not the same as clean; nothing has scored the masks.",
        ]
        return section

    try:
        from .seg_qc import summarize_qc
    except Exception as exc:
        section.status = STATUS_PROBLEM
        section.body_html = (
            f"<p class='bad'>Scorecards exist but spacr.seg_qc could not be "
            f"imported ({_esc(exc.__class__.__name__)}).</p>")
        section.text_lines = ["Segmentation QC: scorecards found but unreadable."]
        return section

    body: List[str] = []
    lines: List[str] = []
    summary_rows: List[List[str]] = []
    worst_rows: List[List[str]] = []
    n_worst_total = 0
    any_fail = False

    verdicts: Dict[str, str] = {}
    for flags_path in artifacts.get("qc_flags") or []:
        try:
            payload = json.loads(flags_path.read_text())
            if isinstance(payload, dict) and payload.get("object_type"):
                verdicts[str(payload["object_type"])] = str(payload.get("verdict") or "")
        except (OSError, ValueError):
            continue

    for card in cards:
        field_qcs, error = _field_qcs_from_csv(card)
        if error:
            section.notes.append(error)
            continue
        if not field_qcs:
            section.notes.append(f"{card.name} is empty")
            continue
        summary = summarize_qc(field_qcs)
        obj = summary.get("object_type") or "object"
        verdict = str(summary.get("verdict") or "?")
        any_fail = any_fail or verdict == "fail"
        summary_rows.append([
            obj,
            str(summary.get("n_fields", 0)),
            str(summary.get("n_ok", 0)),
            str(summary.get("n_warn", 0)),
            str(summary.get("n_fail", 0)),
            verdict.upper(),
            str(summary.get("message") or ""),
        ])
        lines.append(f"  {obj}: {summary.get('n_fields', 0)} fields, "
                     f"{summary.get('n_ok', 0)} ok / {summary.get('n_warn', 0)} warn / "
                     f"{summary.get('n_fail', 0)} fail — {verdict.upper()}")
        lines.append(f"    {summary.get('message')}")
        tally = summary.get("flag_counts") or {}
        if tally:
            rendered = ", ".join(f"{k} {v}" for k, v in
                                 sorted(tally.items(), key=lambda kv: -kv[1]))
            body.append(f"<p class='muted'>{_esc(obj)} — flags raised: "
                        f"{_esc(rendered)}</p>")
            lines.append(f"    flags raised: {rendered}")

        bad = [q for q in field_qcs if q.severity != "ok"]
        n_worst_total += len(bad)
        order = {"fail": 0, "warn": 1, "ok": 2}
        bad.sort(key=lambda q: (order.get(q.severity, 3), q.field))
        for qc in bad[: max(0, max_rows - len(worst_rows))]:
            worst_rows.append([
                qc.field, qc.object_type, str(qc.n_objects), qc.severity,
                ", ".join(qc.flags) or "-", qc.note,
            ])

    if not summary_rows:
        section.status = STATUS_PROBLEM
        section.body_html = ("<p class='bad'>Scorecard CSVs were found but none "
                             "could be read.</p>")
        section.text_lines = ["Segmentation QC: scorecards found but none readable."]
        return section

    section.table = Table(
        columns=["object", "fields", "ok", "warn", "fail", "verdict", "message"],
        rows=summary_rows,
        caption="Plate verdict per object type, as spacr.seg_qc recorded it",
    )
    if verdicts:
        body.append("<p class='muted'>Recorded verdicts: " + ", ".join(
            f"{_esc(k)} = {_esc(v)}" for k, v in sorted(verdicts.items())) + "</p>")
    if worst_rows:
        body.append(_table_html(Table(
            columns=["field", "object", "objects", "severity", "flags", "note"],
            rows=worst_rows,
            caption="Fields that are not clean",
            n_total_rows=n_worst_total,
        )))
        lines.append(f"  {n_worst_total} field(s) flagged; "
                     f"{len(worst_rows)} listed.")
    else:
        body.append("<p>Every scored field is clean.</p>")
        lines.append("  Every scored field is clean.")

    if any_fail:
        section.status = STATUS_PROBLEM
    section.body_html = "\n".join(body)
    section.text_lines = lines
    return section


# ---------------------------------------------------------------------------
# Section: plate QC / edge effects
# ---------------------------------------------------------------------------

_LAYOUT_MARKERS = frozenset({"ring", "is_edge"})


def _collect_plate_qc(src: Path, artifacts: Dict[str, Any],
                      max_rows: int) -> Section:
    """Persisted plate-layout exports, if any exist.

    :mod:`spacr.plate_qc` runs on demand from the Plate Viewer, and its
    verdict is a Mann-Whitney p-value plus a Cliff's delta. This report
    will not compute one: a statistic the report invents is not a result
    the pipeline produced, and a collaborator has no way to tell the two
    apart. Only a layout the user exported into this folder is reported.
    """
    section = Section(title=SECTION_TITLES["plate_qc"], key="plate_qc")
    layouts = []
    for path in artifacts.get("layout_csv") or []:
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as handle:
                header = next(csv.reader(handle), [])
        except (OSError, csv.Error):
            continue
        if _LAYOUT_MARKERS.issubset({str(c) for c in header}):
            layouts.append(path)

    if not layouts:
        section.status = STATUS_MISSING
        section.body_html = (
            "<p><strong>Plate QC: not run.</strong> No exported plate layout "
            "was found under <code>qc/</code>. spaCR tests plates for edge "
            "artefacts and row/column gradients on demand — Tools &rarr; Plate "
            "Viewer, or <code>spacr.plate_qc.detect_edge_effect()</code> — and "
            "the result is only written where you ask for it. This report does "
            "not run that test itself: it reports what the pipeline produced, "
            "and a p-value it computed on the spot would look identical to one "
            "the analysis produced.</p>")
        section.text_lines = [
            "Plate QC: NOT RUN — no exported plate layout under qc/.",
            "  Run Tools > Plate Viewer, or spacr.plate_qc.detect_edge_effect().",
        ]
        return section

    body: List[str] = []
    lines: List[str] = []
    rows: List[List[str]] = []
    for path in layouts:
        columns, head, n_total = _read_csv_head(path, max_rows)
        n_edge = 0
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    flag = str(row.get("is_edge", "")).strip().lower()
                    if flag in ("1", "true", "yes"):
                        n_edge += 1
        except (OSError, csv.Error):
            # Damaged past some row: the wells counted so far are real,
            # and a layout export that cannot be parsed to the end must
            # not take the whole report down with it.
            pass
        rows.append([path.name, str(n_total), str(n_edge),
                     _fmt_bytes(path.stat().st_size if path.exists() else 0)])
        lines.append(f"  {path.name}: {n_total} well(s), {n_edge} on the outer ring")
        body.append(_table_html(Table(
            columns=columns, rows=head,
            caption=f"{path.name} — first {len(head)} of {n_total} well(s)",
            n_total_rows=n_total)))

    section.table = Table(
        columns=["layout export", "wells", "edge wells", "size"], rows=rows,
        caption="Plate layouts exported into this folder")
    body.insert(0, "<p class='muted'>These are exported well grids. The "
                   "edge-effect verdict itself (Cliff's delta, Mann-Whitney p) "
                   "is not stored alongside them; re-open the layout in Tools "
                   "&rarr; Plate Viewer to see it.</p>")
    section.body_html = "\n".join(body)
    section.text_lines = lines
    return section


# ---------------------------------------------------------------------------
# Section: figures
# ---------------------------------------------------------------------------

def _embed_figure(path: Path, max_px: int) -> Tuple[Optional[bytes], str, str]:
    """Read a raster figure, downscale it, and return PNG bytes.

    :param path: image file.
    :param max_px: longest edge after downscaling.
    :returns: ``(data, mime, reason)``. ``data`` is None when the figure
        could not be embedded, and ``reason`` says why.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, "", f"unreadable ({exc.__class__.__name__})"
    if not raw:
        return None, "", "empty file"
    try:
        from PIL import Image
    except Exception:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return raw, mime, ""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            width, height = image.size
            longest = max(width, height)
            if longest <= max_px and path.suffix.lower() == ".png":
                return raw, "image/png", ""
            if longest > max_px and longest > 0:
                scale = max_px / float(longest)
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.LANCZOS)
            if image.mode not in ("RGB", "RGBA", "L"):
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue(), "image/png", ""
    except Exception as exc:
        return None, "", f"could not decode ({exc.__class__.__name__})"


def _figure_title(path: Path, src: Path) -> str:
    """A caption: the path relative to ``src`` when possible."""
    try:
        return str(path.relative_to(src))
    except ValueError:
        return path.name


def _collect_figures(src: Path, artifacts: Dict[str, Any], max_figures: int,
                     max_px: int) -> Section:
    """Embed up to ``max_figures`` raster figures; count what was left out."""
    section = Section(title=SECTION_TITLES["figures"], key="figures")
    raster = list(artifacts.get("raster") or [])
    vector = list(artifacts.get("vector") or [])
    n_found = len(raster) + len(vector)

    if not n_found:
        section.status = STATUS_MISSING
        section.body_html = (
            "<p><strong>No figures were found.</strong> spaCR writes plots into "
            "<code>&lt;src&gt;/results</code> and <code>&lt;src&gt;/figure</code>; "
            "neither holds an image here. Either the run had "
            "<code>plot=False</code>, or it did not reach the plotting step.</p>")
        section.text_lines = ["Figures: none found under results/ or figure/."]
        return section

    body: List[str] = []
    lines: List[str] = []
    embedded = 0
    skipped: List[Tuple[Path, str]] = []
    for path in raster:
        if embedded >= max(0, int(max_figures)):
            skipped.append((path, "figure cap reached"))
            continue
        data, mime, reason = _embed_figure(path, max_px)
        try:
            n_bytes = path.stat().st_size
        except OSError:
            n_bytes = 0
        figure = Figure(path=path, title=_figure_title(path, src), mime=mime or "image/png",
                        data=data, reason=reason, n_bytes=n_bytes)
        if data:
            section.figures.append(figure)
            embedded += 1
        else:
            skipped.append((path, reason or "not embeddable"))

    n_omitted = n_found - embedded
    headline = (f"Embedded {embedded} of {n_found} figure(s) found; "
                f"{n_omitted} omitted.")
    body.append(f"<p>{_esc(headline)}</p>")
    lines.append(f"  {headline}")
    if n_omitted:
        body.append(f"<p class='muted'>The figure cap is {int(max_figures)}; "
                    f"raise <code>max_figures</code> to embed more. Every "
                    f"omitted figure is named below so nothing looks like it "
                    f"was never produced.</p>")

    if vector:
        body.append(
            f"<p class='muted'>{len(vector)} figure(s) are vector PDFs. spaCR "
            f"writes most of its plots as PDF "
            f"(<code>spacr.io._save_figure</code>), and spaCR has no PDF "
            f"rasteriser among its dependencies, so these are listed rather "
            f"than shown. Open them from the run folder.</p>")
        lines.append(f"  {len(vector)} vector figure(s) listed, not embedded "
                     f"(no PDF rasteriser available).")

    omitted_rows = [[_figure_title(p, src), r] for p, r in skipped]
    omitted_rows += [[_figure_title(p, src), "vector — not embeddable"] for p in vector]
    if omitted_rows:
        section.table = Table(
            columns=["figure", "why it is not shown"],
            rows=omitted_rows[:200],
            caption=f"{len(omitted_rows)} figure(s) present in the run folder "
                    f"but not embedded",
            n_total_rows=len(omitted_rows))
        for name, reason in omitted_rows[:20]:
            lines.append(f"    not shown: {name} ({reason})")

    section.body_html = "\n".join(body)
    section.text_lines = lines
    return section


# ---------------------------------------------------------------------------
# Section: statistics / result tables
# ---------------------------------------------------------------------------

def _collect_statistics(src: Path, artifacts: Dict[str, Any],
                        max_rows: int, max_files: int = 12) -> Section:
    """Preview the result CSVs the pipeline wrote. Nothing is recomputed."""
    section = Section(title=SECTION_TITLES["statistics"], key="statistics")
    csvs = list(artifacts.get("result_csv") or [])
    dbs = list(artifacts.get("databases") or [])

    if not csvs and not dbs:
        section.status = STATUS_MISSING
        section.body_html = (
            "<p><strong>No result tables were found.</strong> Neither a "
            "measurements database nor a result CSV exists under "
            "<code>&lt;src&gt;/measurements</code> or "
            "<code>&lt;src&gt;/results</code>. Nothing downstream of "
            "segmentation has produced a table here.</p>")
        section.text_lines = ["Statistics: no result CSV and no database found."]
        return section

    body: List[str] = []
    lines: List[str] = []
    index_rows: List[List[str]] = []

    for path in dbs:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        tables = _sqlite_table_counts(path)
        index_rows.append([
            _figure_title(path, src), "sqlite",
            str(sum(n for _, n in tables)), _fmt_bytes(size)])
        lines.append(f"  {path.name}: {len(tables)} table(s), "
                     f"{sum(n for _, n in tables)} row(s) total")
        if tables:
            body.append(_table_html(Table(
                columns=["table", "rows"],
                rows=[[name, str(n)] for name, n in tables],
                caption=f"{_figure_title(path, src)} — tables and row counts")))

    shown = 0
    for path in csvs:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        try:
            columns, rows, n_total = _read_csv_head(path, max_rows)
        except OSError as exc:
            section.notes.append(f"{path.name} unreadable ({exc.__class__.__name__})")
            continue
        index_rows.append([_figure_title(path, src), f"{len(columns)} columns",
                           str(n_total), _fmt_bytes(size)])
        lines.append(f"  {path.name}: {n_total} row(s) x {len(columns)} column(s)")
        if shown < max_files and rows:
            body.append(_table_html(Table(
                columns=columns, rows=rows,
                caption=f"{_figure_title(path, src)} — first {len(rows)} of "
                        f"{n_total} row(s)",
                n_total_rows=n_total)))
            shown += 1

    if len(csvs) > shown:
        body.append(f"<p class='muted'>{len(csvs) - shown} further result CSV(s) "
                    f"are listed above but not previewed.</p>")

    section.table = Table(
        columns=["file", "shape", "rows", "size"], rows=index_rows,
        caption="Result tables produced by this run")
    body.insert(0, "<p class='muted'>These tables are reproduced as the "
                   "pipeline wrote them. Nothing on this page was recomputed, "
                   "aggregated or re-tested.</p>")
    section.body_html = "\n".join(body)
    section.text_lines = lines
    return section


def _sqlite_table_counts(path: Path, max_tables: int = 40) -> List[Tuple[str, int]]:
    """Return ``(table, n_rows)`` for a SQLite file, read-only, best effort."""
    import sqlite3
    from urllib.parse import quote as _quote
    uri = "file:" + _quote(str(path).replace("\\", "/"), safe="/:") + "?mode=ro"
    out: List[Tuple[str, int]] = []
    try:
        from .database_concurrency import connect as _connect_database

        conn = _connect_database(path, readonly=True)
    except sqlite3.Error:
        return out
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for name in names[:max_tables]:
            try:
                count = conn.execute(
                    f'SELECT COUNT(*) FROM "{name.replace(chr(34), chr(34) * 2)}"'
                ).fetchone()[0]
            except sqlite3.Error:
                count = -1
            out.append((name, int(count)))
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return out


# ---------------------------------------------------------------------------
# Section: settings
# ---------------------------------------------------------------------------

def _collect_settings(src: Path, artifacts: Dict[str, Any],
                      runs: Sequence[Dict[str, Any]],
                      max_rows: int = DEFAULT_MAX_SETTINGS_ROWS,
                      include_plan: bool = True) -> Section:
    """The exact settings the run used, from the journal and from ``src``."""
    section = Section(title=SECTION_TITLES["settings"], key="settings")
    settings_csvs = list(artifacts.get("settings_csv") or [])
    journal_settings = [r for r in runs if r.get("settings")]

    if not settings_csvs and not journal_settings:
        section.status = STATUS_MISSING
        section.body_html = (
            "<p><strong>No settings were found.</strong> spaCR saves a copy of "
            "every run's settings to <code>&lt;src&gt;/settings/*.csv</code> and "
            "to its run journal; neither is available for this folder, so the "
            "configuration that produced these results cannot be shown. The "
            "results below are therefore not reproducible from this document "
            "alone.</p>")
        section.text_lines = ["Settings: none found (not reproducible from this report)."]
        return section

    body: List[str] = []
    lines: List[str] = []

    if journal_settings:
        run = journal_settings[0]
        items = sorted((str(k), _render_setting(v))
                       for k, v in (run["settings"] or {}).items())
        section.table = Table(
            columns=["key", "value"],
            rows=[list(pair) for pair in items[:max_rows]],
            caption=f"Settings recorded by run {run['dir'].name} "
                    f"({run.get('app_key')})",
            n_total_rows=len(items))
        lines.append(f"  {len(items)} setting(s) recorded by run {run['dir'].name}")
        for key, value in items:
            lines.append(f"    {key:<32} {value}")
        if include_plan:
            plan = _describe_plan_safe(run.get("settings") or {},
                                       str(run.get("app_key") or ""))
            if plan:
                body.append(
                    "<details><summary>How spaCR reads these settings "
                    "today</summary><p class='muted'>Re-derived from the "
                    "settings against this folder's current contents, so it "
                    "describes the configuration, not the run's live "
                    "state.</p><pre>" + _esc(plan) + "</pre></details>")

    for path in settings_csvs:
        try:
            columns, rows, n_total = _read_csv_head(path, max_rows)
        except OSError as exc:
            section.notes.append(f"{path.name} unreadable ({exc.__class__.__name__})")
            continue
        body.append("<details><summary>" + _esc(_figure_title(path, src)) +
                    f" — {n_total} setting(s)</summary>" +
                    _table_html(Table(columns=columns, rows=rows,
                                      n_total_rows=n_total)) + "</details>")
        lines.append(f"  {_figure_title(path, src)}: {n_total} setting(s)")

    if not journal_settings:
        body.insert(0, "<p class='muted'>No journal entry matched this folder, "
                       "so the settings below come from the copies spaCR left "
                       "in <code>&lt;src&gt;/settings</code>.</p>")

    section.body_html = "\n".join(body)
    section.text_lines = lines
    return section


def _render_setting(value: Any, width: int = 160) -> str:
    """One-line rendering of a settings value."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _describe_plan_safe(settings: Dict[str, Any], app_key: str) -> str:
    """Call :func:`spacr.validate.describe_plan`, never raising."""
    try:
        from .validate import describe_plan
        return str(describe_plan(settings, app_key) or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Section: appendix
# ---------------------------------------------------------------------------

def _collect_appendix(src: Path, artifacts: Dict[str, Any],
                      max_dict_rows: int = DEFAULT_MAX_DICT_ROWS) -> Section:
    """Feature dictionary, annotation columns and a file inventory."""
    section = Section(title=SECTION_TITLES["appendix"], key="appendix")
    body: List[str] = []
    lines: List[str] = []
    have_something = False

    dbs = list(artifacts.get("databases") or [])
    if dbs:
        db = dbs[0]
        families, entries, n_total, error = _feature_dictionary(db, max_dict_rows)
        if error:
            section.notes.append(error)
        elif families:
            have_something = True
            body.append("<h3>Measured features</h3>")
            body.append(f"<p class='muted'>Every column of "
                        f"<code>{_esc(_figure_title(db, src))}</code>, described "
                        f"by <code>spacr.feature_dict</code>. "
                        f"{len(entries)} of {n_total} column(s) shown; use "
                        f"<code>spacr.feature_dict.export_dictionary()</code> "
                        f"for the full list.</p>")
            body.append(_table_html(Table(
                columns=["family", "columns"],
                rows=[[str(k), str(v)] for k, v in families],
                caption="Feature families")))
            body.append("<details><summary>Column descriptions</summary>" +
                        _table_html(Table(
                            columns=["table", "column", "family", "unit",
                                     "description"],
                            rows=entries, n_total_rows=n_total)) + "</details>")
            lines.append(f"  feature dictionary: {n_total} column(s) across "
                         f"{len(families)} family/families")

        columns, n_annotated, ann_error = _annotation_summary(db)
        if ann_error:
            section.notes.append(ann_error)
        elif columns:
            have_something = True
            body.append("<h3>Annotations</h3>")
            body.append("<p>Annotation column(s) present: " + ", ".join(
                f"<code>{_esc(c)}</code>" for c in columns) + ". "
                "Inter-annotator agreement (Cohen's / Fleiss' &kappa;) is not "
                "computed here — it is a statistic, not a stored result. Run "
                "Tools &rarr; Annotator Agreement, or "
                "<code>spacr.agreement.agreement_report()</code>.</p>")
            if n_annotated is not None:
                body.append(f"<p class='muted'>{_esc(n_annotated)} annotated "
                            f"row(s) in <code>png_list</code>.</p>")
            lines.append(f"  annotation columns: {', '.join(columns)}")

    rows, truncated = _file_inventory(src)
    if rows:
        have_something = True
        section.table = Table(
            columns=["folder", "files", "size"], rows=rows,
            caption="File inventory")
        lines.append("  file inventory:")
        for name, n_files, size in rows:
            lines.append(f"    {name:<28} {n_files:>8} files  {size}")
    if truncated:
        section.notes.append(
            f"The file inventory stopped after {WALK_BUDGET} entries — this "
            f"folder holds more files than that, so the counts are lower "
            f"bounds.")

    if not have_something:
        section.status = STATUS_MISSING
        body.append("<p><strong>Nothing to append.</strong> No measurements "
                    "database and no files were found under this folder.</p>")
        lines.append("Appendix: nothing found.")

    section.body_html = "\n".join(body)
    section.text_lines = lines
    return section


def _feature_dictionary(db: Path, max_rows: int
                        ) -> Tuple[List[Tuple[str, int]], List[List[str]], int, str]:
    """Describe a measurements database with :mod:`spacr.feature_dict`."""
    try:
        from .feature_dict import describe_database
    except Exception as exc:
        return [], [], 0, f"spacr.feature_dict unavailable ({exc.__class__.__name__})"
    try:
        frame = describe_database(db)
    except Exception as exc:
        return [], [], 0, (f"feature dictionary unavailable for {db.name} "
                           f"({exc.__class__.__name__})")
    try:
        n_total = int(len(frame))
        tally = frame["family"].fillna("unknown").value_counts()
        families = [(str(k), int(v)) for k, v in tally.items()]
        rows: List[List[str]] = []
        for _, row in frame.head(max_rows).iterrows():
            rows.append([
                str(row.get("table", "")),
                str(row.get("column", "")),
                str(row.get("family", "")),
                "" if row.get("unit") is None else str(row.get("unit")),
                "" if row.get("description") is None else str(row.get("description")),
            ])
    except Exception as exc:
        return [], [], 0, f"feature dictionary unreadable ({exc.__class__.__name__})"
    return families, rows, n_total, ""


def _annotation_summary(db: Path) -> Tuple[List[str], Optional[int], str]:
    """List a database's annotation columns without scoring them."""
    try:
        from .agreement import annotation_columns, PNG_TABLE
    except Exception as exc:
        return [], None, f"spacr.agreement unavailable ({exc.__class__.__name__})"
    try:
        columns = list(annotation_columns(str(db)))
    except Exception:
        return [], None, ""
    if not columns:
        return [], None, ""
    n_rows: Optional[int] = None
    for name, count in _sqlite_table_counts(db):
        if name == PNG_TABLE:
            n_rows = count
            break
    return columns, n_rows, ""


def _file_inventory(src: Path) -> Tuple[List[List[str]], bool]:
    """One row per immediate child of ``src``, plus its loose files."""
    rows: List[List[str]] = []
    truncated = False
    if not src.is_dir():
        return rows, truncated
    loose_files = 0
    loose_bytes = 0
    try:
        entries = sorted(os.scandir(src), key=lambda e: e.name)
    except OSError:
        return rows, truncated
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                n_files, total, cut = _dir_stats(Path(entry.path))
                truncated = truncated or cut
                rows.append([entry.name + "/", str(n_files), _fmt_bytes(total)])
            elif entry.is_file(follow_symlinks=False):
                loose_files += 1
                loose_bytes += entry.stat().st_size
        except OSError:
            continue
    if loose_files:
        rows.append(["(files at the top level)", str(loose_files),
                     _fmt_bytes(loose_bytes)])
    return rows, truncated


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_report(src: Any,
                   *,
                   title: Optional[str] = None,
                   max_figures: int = DEFAULT_MAX_FIGURES,
                   max_figure_px: int = DEFAULT_MAX_FIGURE_PX,
                   max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
                   run_dirs: Optional[Sequence[Any]] = None,
                   search_journal: bool = True,
                   journal_limit: int = DEFAULT_JOURNAL_LIMIT,
                   include_plan: bool = True) -> Report:
    """Gather everything reportable about the run folder ``src``.

    Headless and read-only: nothing is written, no pipeline is invoked, no
    statistic is computed. A folder that does not exist, or holds no spaCR
    output at all, yields a valid :class:`Report` that says so — this
    function does not raise for missing input.

    :param src: the plate / run folder.
    :param title: document title. Defaults to ``"spaCR report — <folder>"``.
    :param max_figures: raster figures embedded before the rest are only
        listed. See :data:`DEFAULT_MAX_FIGURES`.
    :param max_figure_px: longest edge a figure is downscaled to.
    :param max_table_rows: rows previewed from any one table.
    :param run_dirs: explicit run-journal folders to use instead of
        searching. Every one given is used, whether or not its ``src``
        setting matches — callers that pass this know which runs they mean.
    :param search_journal: when ``run_dirs`` is None, scan
        ``~/.spacr/runs`` for runs whose ``src`` is this folder.
    :param journal_limit: how many recent journal entries to consider.
    :param include_plan: render :func:`spacr.validate.describe_plan` into
        the settings section.
    :returns: a :class:`Report` holding one section per :data:`SECTION_KEYS`,
        plus any section contributed by a plugin, each inserted after the
        existing section it names — core, or one an earlier plugin added — or
        appended when that key is absent.

    Example:
        .. code-block:: python

            from spacr.report import collect_report, write_html
            report = collect_report('/data/plate1')
            print(report.status, report.missing_sections)
            write_html(report, '/tmp/plate1.html')
    """
    src_path = Path(str(src)).expanduser()
    try:
        src_path = src_path.resolve()
    except (OSError, RuntimeError):
        # RuntimeError is what pathlib raises for a symlink loop (ELOOP),
        # and this function promises never to raise for bad input: an
        # unresolvable folder is reported as "does not exist", not as a
        # traceback in the caller's face.
        pass

    artifacts = _find_artifacts(src_path)
    runs, journal_problems = _load_journal_runs(
        src_path, run_dirs, search_journal, journal_limit)

    status_section, status, detail = _collect_run_status(src_path, artifacts, runs)
    sections = [
        status_section,
        _collect_provenance(src_path, runs, journal_problems),
        _collect_segmentation_qc(src_path, artifacts, max_table_rows),
        _collect_plate_qc(src_path, artifacts, max_table_rows),
        _collect_figures(src_path, artifacts, max_figures, max_figure_px),
        _collect_statistics(src_path, artifacts, max_table_rows),
        _collect_settings(src_path, artifacts, runs, include_plan=include_plan),
        _collect_appendix(src_path, artifacts),
    ]

    # Third-party report chapters are inserted relative to stable core keys.
    # A failing builder becomes a visible problem chapter rather than silently
    # disappearing from the report.
    try:
        from .plugins import (
            ReportContext, load_object, record_diagnostic, report_sections,
        )
        context = ReportContext(
            src=src_path,
            artifacts=artifacts,
            runs=tuple(runs),
            options={
                "max_figures": max_figures,
                "max_figure_px": max_figure_px,
                "max_table_rows": max_table_rows,
                "include_plan": include_plan,
            },
        )
        existing_keys = {section.key for section in sections}
        for plugin_name, contribution in report_sections():
            try:
                if contribution.key in existing_keys:
                    raise ValueError(
                        f"report section key {contribution.key!r} already exists"
                    )
                builder = load_object(contribution.builder)
                if not callable(builder):
                    raise TypeError(f"{contribution.builder!r} is not callable")
                plugin_section = builder(context)
                if not isinstance(plugin_section, Section):
                    raise TypeError(
                        f"{contribution.builder!r} returned "
                        f"{type(plugin_section).__name__}, expected Section"
                    )
                if plugin_section.key not in ("", contribution.key):
                    raise ValueError(
                        f"builder returned key {plugin_section.key!r}; "
                        f"expected {contribution.key!r}"
                    )
                plugin_section.key = contribution.key
                plugin_section.title = plugin_section.title or contribution.title
            except Exception as exc:
                record_diagnostic(
                    plugin_name,
                    f"Report section {contribution.key!r} failed",
                    exc,
                )
                plugin_section = Section(
                    key=contribution.key,
                    title=contribution.title,
                    status=STATUS_PROBLEM,
                    body_html=(
                        "<p>This plugin report section could not be generated. "
                        f"<code>{_esc(exc)}</code></p>"
                    ),
                    text_lines=[
                        f"Plugin report section failed: {type(exc).__name__}: {exc}"
                    ],
                )
            insert_at = next(
                (index + 1 for index, section in enumerate(sections)
                 if section.key == contribution.after),
                len(sections),
            )
            sections.insert(insert_at, plugin_section)
            existing_keys.add(contribution.key)
    except Exception:
        LOG.exception("Could not initialise plugin report sections")

    if artifacts.get("truncated"):
        sections[0].notes.append(
            f"The scan of this folder stopped after {WALK_BUDGET} files. Some "
            f"outputs may not be listed.")

    figures_section = sections[4]
    n_found = len(artifacts.get("raster") or []) + len(artifacts.get("vector") or [])
    n_embedded = len(figures_section.figures)

    if status == "unknown" and src_path.is_dir():
        nothing = not any(artifacts.get(key) for key in
                          ("databases", "qc_csv", "result_csv", "settings_csv",
                           "raster", "vector", "sidecars"))
        if nothing and not runs:
            status = "empty"
            detail = _STATUS_LABELS["empty"]
            sections[0].body_html += (
                f"<p>No spaCR output was found under "
                f"<code>{_esc(src_path)}</code>: no database, no scorecard, no "
                f"figure, no settings copy. Either the run wrote elsewhere, or "
                f"it never wrote anything.</p>")
            sections[0].text_lines.append(
                "  No spaCR output was found under this folder.")

    try:
        from .version import get_version
        version = get_version()
    except Exception:
        version = "unknown"

    return Report(
        src=src_path,
        title=title or f"spaCR report — {src_path.name or src_path}",
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sections=sections,
        status=status,
        status_detail=detail,
        spacr_version=version,
        n_figures_found=n_found,
        n_figures_embedded=n_embedded,
    )


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 0 4rem 0;
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: #16202a; background: #f6f7f9;
}
main { max-width: 62rem; margin: 0 auto; padding: 0 1.25rem; }
header.doc {
  background: #16202a; color: #f6f7f9; padding: 2rem 1.25rem 1.5rem;
  margin-bottom: 1.5rem;
}
header.doc .inner { max-width: 62rem; margin: 0 auto; }
header.doc h1 { margin: 0 0 .35rem; font-size: 1.6rem; font-weight: 600; }
header.doc .meta { font-size: .82rem; opacity: .78; }
header.doc code { background: rgba(255,255,255,.12); color: inherit; }
.banner {
  margin: 0 auto 1.5rem; max-width: 62rem;
  padding: .85rem 1rem; border-radius: 6px; font-weight: 600;
  border-left: 5px solid #6b7785; background: #e9ecef; color: #16202a;
}
.banner.complete { border-color: #2e7d4f; background: #e6f4ec; }
.banner.partial, .banner.failed { border-color: #b3261e; background: #fbe9e7; }
.banner.unknown, .banner.empty { border-color: #a2751a; background: #fdf3e0; }
.banner .sub { display: block; font-weight: 400; font-size: .85rem; margin-top: .3rem; }
nav.toc { margin-bottom: 1.5rem; font-size: .9rem; }
nav.toc ol { margin: .4rem 0 0; padding-left: 1.3rem; }
nav.toc .missing { color: #8a939c; }
section.chapter {
  background: #fff; border: 1px solid #dfe3e8; border-radius: 8px;
  padding: 1.1rem 1.25rem 1.35rem; margin-bottom: 1.25rem;
}
section.chapter.missing { background: #fafbfc; border-style: dashed; }
section.chapter > h2 { margin: 0 0 .75rem; font-size: 1.15rem; font-weight: 600; }
section.chapter > h2 .tag {
  float: right; font-size: .68rem; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; padding: .2rem .5rem; border-radius: 999px;
  background: #e9ecef; color: #6b7785;
}
section.chapter > h2 .tag.problem { background: #fbe9e7; color: #b3261e; }
h3 { font-size: 1rem; margin: 1.2rem 0 .4rem; }
p { margin: .5rem 0; }
.muted { color: #6b7785; font-size: .88rem; }
.bad { color: #b3261e; font-weight: 600; }
.verdict { font-weight: 600; }
.verdict.partial, .verdict.failed { color: #b3261e; }
.verdict.unknown, .verdict.empty { color: #8a6100; }
.verdict.complete { color: #2e7d4f; }
code, pre { font-family: "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; }
code { background: #eef1f4; padding: .08em .3em; border-radius: 3px; font-size: .88em; }
pre {
  background: #16202a; color: #e6e9ec; padding: .8rem 1rem; border-radius: 6px;
  overflow-x: auto; font-size: .8rem; line-height: 1.45;
}
ul { margin: .5rem 0; padding-left: 1.3rem; }
li.sub { color: #6b7785; font-size: .88rem; list-style: circle; }
.tablewrap { overflow-x: auto; margin: .85rem 0; }
table { border-collapse: collapse; width: 100%; font-size: .84rem; }
caption { caption-side: top; text-align: left; color: #6b7785;
          font-size: .84rem; padding-bottom: .35rem; }
th, td { border-bottom: 1px solid #e4e8ec; padding: .35rem .55rem;
         text-align: left; vertical-align: top; }
th { background: #f2f4f6; font-weight: 600; white-space: nowrap; }
tbody tr:nth-child(even) td { background: #fafbfc; }
.figgrid { display: grid; grid-template-columns: 1fr; gap: 1.1rem; margin: 1rem 0; }
figure { margin: 0; }
figure img { max-width: 100%; height: auto; display: block;
             border: 1px solid #dfe3e8; border-radius: 6px; background: #fff; }
figcaption { font-size: .8rem; color: #6b7785; margin-top: .35rem;
             word-break: break-all; }
details { margin: .7rem 0; }
summary { cursor: pointer; font-size: .9rem; font-weight: 600; }
dl.env { display: grid; grid-template-columns: max-content 1fr;
         gap: .1rem .9rem; margin: .4rem 0; font-size: .85rem; }
dl.env dt { color: #6b7785; }
dl.env dd { margin: 0; word-break: break-all; }
footer.doc { max-width: 62rem; margin: 2rem auto 0; padding: 0 1.25rem;
             color: #8a939c; font-size: .8rem; }
@media (prefers-color-scheme: dark) {
  body { background: #10151a; color: #dfe3e8; }
  section.chapter { background: #171e25; border-color: #2a333c; }
  section.chapter.missing { background: #141a20; }
  code { background: #212b34; }
  th { background: #1d252d; }
  tbody tr:nth-child(even) td { background: #1b232a; }
  th, td { border-bottom-color: #2a333c; }
  .banner { background: #1d252d; color: #dfe3e8; }
  .banner.complete { background: #16261d; }
  .banner.partial, .banner.failed { background: #2a1917; }
  .banner.unknown, .banner.empty { background: #29220f; }
  figure img { border-color: #2a333c; background: #0d1115; }
  section.chapter > h2 .tag { background: #212b34; color: #8a939c; }
}
@media print {
  body { background: #fff; }
  section.chapter { break-inside: avoid; border-color: #ccc; }
  header.doc { background: #fff; color: #000; border-bottom: 2px solid #000; }
  nav.toc { display: none; }
}
"""


def _table_html(table: Optional[Table]) -> str:
    """Render a :class:`Table` as escaped HTML.

    :param table: the table, or None.
    :returns: an HTML fragment; empty string when there is nothing to show.
    """
    if table is None or (not table.columns and not table.rows):
        return ""
    parts = ["<div class='tablewrap'><table>"]
    if table.caption:
        parts.append(f"<caption>{_esc(table.caption)}</caption>")
    if table.columns:
        parts.append("<thead><tr>" + "".join(
            f"<th>{_esc(c)}</th>" for c in table.columns) + "</tr></thead>")
    parts.append("<tbody>")
    n_columns = len(table.columns) or max((len(r) for r in table.rows), default=0)
    for row in table.rows:
        cells = list(row) + [""] * (n_columns - len(row))
        parts.append("<tr>" + "".join(
            f"<td>{_esc(c)}</td>" for c in cells[:n_columns]) + "</tr>")
    parts.append("</tbody>")
    if table.n_omitted:
        parts.append(
            f"<tfoot><tr><td colspan='{max(1, n_columns)}'>"
            f"{_esc(table.n_omitted)} further row(s) not shown.</td></tr></tfoot>")
    parts.append("</table></div>")
    return "".join(parts)


def _figures_html(section: Section) -> str:
    """Render a section's embedded figures as base64 ``data:`` images."""
    if not section.figures:
        return ""
    parts = ["<div class='figgrid'>"]
    for figure in section.figures:
        if not figure.embedded:
            continue
        parts.append(
            "<figure><img alt=\"" + _esc(figure.title) + "\" src=\"" +
            figure.data_uri() + "\" /><figcaption>" + _esc(figure.title) +
            "</figcaption></figure>")
    parts.append("</div>")
    return "".join(parts)


def render_html(report: Report) -> str:
    """Render ``report`` as a single self-contained HTML document.

    The output has no external dependencies of any kind: the stylesheet is
    inline, every image is a base64 ``data:`` URI, and there is no
    JavaScript. It renders identically on a machine that has never heard of
    spaCR and has no network.

    :param report: a :class:`Report` from :func:`collect_report`.
    :returns: the complete HTML document as a string.
    """
    out: List[str] = []
    out.append("<!doctype html>")
    out.append("<html lang='en'><head><meta charset='utf-8' />")
    out.append("<meta name='viewport' content='width=device-width, "
               "initial-scale=1' />")
    out.append(f"<title>{_esc(report.title)}</title>")
    out.append("<style>" + _CSS + "</style>")
    out.append("</head><body>")

    out.append("<header class='doc'><div class='inner'>")
    out.append(f"<h1>{_esc(report.title)}</h1>")
    out.append(
        "<div class='meta'>"
        f"Source <code>{_esc(report.src)}</code> &middot; "
        f"generated {_esc(_fmt_time(report.generated_utc))} &middot; "
        f"spaCR {_esc(report.spacr_version)}</div>")
    out.append("</div></header>")

    out.append(f"<div class='banner {_esc(report.status)}'>"
               f"{_esc(report.status_detail)}")
    if report.n_figures_found:
        out.append(f"<span class='sub'>{_esc(report.n_figures_embedded)} of "
                   f"{_esc(report.n_figures_found)} figure(s) are embedded in "
                   f"this file.</span>")
    if report.missing_sections:
        names = ", ".join(SECTION_TITLES.get(k, k) for k in report.missing_sections)
        out.append(f"<span class='sub'>Not available for this run: "
                   f"{_esc(names)}. Each is listed below with what was looked "
                   f"for.</span>")
    out.append("</div>")

    out.append("<main>")
    out.append("<nav class='toc'><strong>Contents</strong><ol>")
    for section in report.sections:
        css = " class='missing'" if section.status == STATUS_MISSING else ""
        suffix = " — not available" if section.status == STATUS_MISSING else ""
        out.append(f"<li{css}><a href='#{_esc(section.key)}'>"
                   f"{_esc(section.title)}</a>{_esc(suffix)}</li>")
    out.append("</ol></nav>")

    for section in report.sections:
        classes = "chapter"
        if section.status == STATUS_MISSING:
            classes += " missing"
        out.append(f"<section class='{classes}' id='{_esc(section.key)}'>")
        tag = ""
        if section.status == STATUS_MISSING:
            tag = "<span class='tag'>not available</span>"
        elif section.status == STATUS_PROBLEM:
            tag = "<span class='tag problem'>attention</span>"
        out.append(f"<h2>{tag}{_esc(section.title)}</h2>")
        if section.body_html:
            out.append(section.body_html)
        if section.table is not None:
            out.append(_table_html(section.table))
        out.append(_figures_html(section))
        if section.notes:
            out.append("<ul class='muted'>")
            for note in section.notes:
                out.append(f"<li>{_esc(note)}</li>")
            out.append("</ul>")
        out.append("</section>")
    out.append("</main>")

    out.append("<footer class='doc'><p>Generated by spaCR "
               f"{_esc(report.spacr_version)}. This file is self-contained: "
               "images are embedded, there is no JavaScript, and nothing here "
               "is loaded from the network.</p></footer>")
    out.append("</body></html>")
    return "\n".join(out)


def render_text(report: Report) -> str:
    """Render ``report`` as plain text.

    This is what the PDF transcribes, and what the GUI shows as a preview.

    :param report: a :class:`Report`.
    :returns: a multi-line string.
    """
    return "\n".join(_text_lines(report))


def _text_lines(report: Report) -> List[str]:
    """Plain-text body of the report, one string per line."""
    rule = "=" * 78
    lines = [rule, report.title, rule,
             f"source     : {report.src}",
             f"generated  : {_fmt_time(report.generated_utc)}",
             f"spaCR      : {report.spacr_version}",
             f"status     : {report.status_detail}", ""]
    if report.n_figures_found:
        lines.append(f"figures    : {report.n_figures_embedded} of "
                     f"{report.n_figures_found} embedded")
    if report.missing_sections:
        names = ", ".join(SECTION_TITLES.get(k, k) for k in report.missing_sections)
        lines.append(f"not available: {names}")
    lines.append("")
    for i, section in enumerate(report.sections, start=1):
        suffix = "  [NOT AVAILABLE]" if section.status == STATUS_MISSING else ""
        lines.append("-" * 78)
        lines.append(f"{i}. {section.title}{suffix}")
        lines.append("-" * 78)
        lines.extend(section.text_lines or ["  (nothing recorded)"])
        if section.table is not None:
            lines.append("")
            lines.extend(_table_text(section.table))
        for note in section.notes:
            lines.append(f"  ! {note}")
        lines.append("")
    return lines


def _table_text(table: Table, width: int = 76) -> List[str]:
    """Render a :class:`Table` as fixed-width text."""
    if not table.columns and not table.rows:
        return []
    out: List[str] = []
    if table.caption:
        out.append(f"  {table.caption}")
    n_columns = len(table.columns) or max((len(r) for r in table.rows), default=0)
    grid = [list(table.columns) + [""] * (n_columns - len(table.columns))]
    for row in table.rows:
        grid.append([str(c) for c in row] + [""] * (n_columns - len(row)))
    widths = [min(28, max(len(str(r[i])) for r in grid)) for i in range(n_columns)]
    # Squeeze the widest column when the row would overflow the page.
    while sum(widths) + 2 * n_columns > width and max(widths) > 8:
        widths[widths.index(max(widths))] -= 1
    for j, row in enumerate(grid):
        cells = []
        for i in range(n_columns):
            text = str(row[i])
            if len(text) > widths[i]:
                text = text[: max(1, widths[i] - 1)] + "…"
            cells.append(text.ljust(widths[i]))
        out.append("  " + "  ".join(cells).rstrip())
        if j == 0:
            out.append("  " + "  ".join("-" * w for w in widths))
    if table.n_omitted:
        out.append(f"  … and {table.n_omitted} further row(s) not shown.")
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_html(report: Report, path: Any) -> Path:
    """Write ``report`` as a single self-contained HTML file.

    :param report: a :class:`Report`.
    :param path: destination file. Parent directories are created.
    :returns: the path written.
    """
    out = Path(str(path)).expanduser()
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(report), encoding="utf-8")
    return out


#: Text lines per PDF page. A4 portrait at 8 pt monospace fits about 78;
#: 66 leaves margin for the header rule and a short overhang.
_PDF_LINES_PER_PAGE = 66
_PDF_WRAP = 104


def _pdf_page_specs(report: Report) -> List[Tuple[str, Any]]:
    """Plan the PDF: a list of ``('text', lines)`` / ``('figure', Figure)``.

    Separated from the writing so the page count can be known — and
    tested — without rendering.
    """
    pages: List[Tuple[str, Any]] = []
    wrapped: List[str] = []
    for line in _text_lines(report):
        if len(line) <= _PDF_WRAP:
            wrapped.append(line)
            continue
        indent = " " * (len(line) - len(line.lstrip()))
        chunks = textwrap.wrap(line, width=_PDF_WRAP,
                               subsequent_indent=indent + "  ",
                               break_long_words=True, break_on_hyphens=False)
        wrapped.extend(chunks or [line[:_PDF_WRAP]])
    for start in range(0, max(len(wrapped), 1), _PDF_LINES_PER_PAGE):
        pages.append(("text", wrapped[start:start + _PDF_LINES_PER_PAGE]))
    for section in report.sections:
        for figure in section.figures:
            if figure.embedded:
                pages.append(("figure", figure))
    return pages


def pdf_page_count(report: Report) -> int:
    """How many pages :func:`write_pdf` will produce for ``report``.

    :param report: a :class:`Report`.
    :returns: the page count.
    """
    return len(_pdf_page_specs(report))


def write_pdf(report: Report, path: Any) -> Path:
    """Write ``report`` as a PDF composed with matplotlib.

    spaCR depends on matplotlib and on nothing that converts HTML to PDF,
    and this module will not add such a dependency for one feature. The PDF
    is therefore a **monospace transcription** of the same content — the
    same sections, the same tables, the same "not run" statements — plus one
    page per embedded figure. Compared with the HTML it loses colour
    emphasis, table borders, collapsible detail blocks and clickable
    contents, and long cells are truncated to the page width. Send the HTML
    when the recipient can open one.

    :param report: a :class:`Report`.
    :param path: destination file. Parent directories are created.
    :returns: the path written.
    """
    # `import matplotlib` does NOT bind `matplotlib.image`; reaching it that
    # way raises AttributeError, which the per-figure guard below would
    # swallow into "could not be drawn" on *every* page. Import it by name.
    import matplotlib.image as mpimage
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure as MplFigure

    out = Path(str(path)).expanduser()
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)

    specs = _pdf_page_specs(report)
    with PdfPages(str(out)) as pdf:
        for kind, payload in specs:
            figure = MplFigure(figsize=(8.27, 11.69))
            if kind == "text":
                figure.text(0.06, 0.965, "\n".join(payload), family="monospace",
                            fontsize=8, va="top", ha="left", linespacing=1.35)
            else:
                axes = figure.add_subplot(111)
                axes.set_axis_off()
                axes.set_title(payload.title, fontsize=8, loc="left", wrap=True)
                try:
                    image = mpimage.imread(io.BytesIO(payload.data),
                                           format="png")
                    axes.imshow(image)
                except Exception:
                    axes.text(0.5, 0.5, f"[{payload.title} could not be drawn]",
                              ha="center", va="center", fontsize=9)
            pdf.savefig(figure)
        if not specs:
            figure = MplFigure(figsize=(8.27, 11.69))
            figure.text(0.06, 0.95, "Empty report", family="monospace",
                        fontsize=10, va="top")
            pdf.savefig(figure)
    return out


def build_report(src: Any, out: Any, fmt: str = "html", **kwargs: Any) -> List[Path]:
    """Collect and write a report for ``src`` in one call.

    :param src: the plate / run folder to report on.
    :param out: destination. A path ending in ``.html`` or ``.pdf`` names
        the file; anything else is treated as a folder and the files are
        named after ``src`` and the current time.
    :param fmt: ``"html"``, ``"pdf"`` or ``"both"``.
    :param kwargs: forwarded to :func:`collect_report`.
    :returns: the paths written, in the order html, pdf.
    :raises ValueError: for an unknown ``fmt``.

    Example:
        .. code-block:: python

            from spacr.report import build_report
            paths = build_report('/data/plate1', '/tmp/reports', fmt='both')
    """
    formats = str(fmt or "html").strip().lower()
    if formats not in ("html", "pdf", "both"):
        raise ValueError(
            f"fmt must be 'html', 'pdf' or 'both', not {fmt!r}")

    report = collect_report(src, **kwargs)
    destination = Path(str(out)).expanduser()
    suffix = destination.suffix.lower()

    written: List[Path] = []
    if suffix in (".html", ".htm", ".pdf") and formats != "both":
        target = destination
        if formats == "html":
            written.append(write_html(report, target.with_suffix(".html")
                                      if suffix == ".pdf" else target))
        else:
            written.append(write_pdf(report, target.with_suffix(".pdf")
                                     if suffix != ".pdf" else target))
        return written

    if suffix in (".html", ".htm", ".pdf"):
        stem_dir = destination.parent
        stem = destination.stem
    else:
        stem_dir = destination
        name = Path(str(src)).expanduser().name or "spacr"
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        stem = f"spacr_report_{name}_{stamp}"

    if formats in ("html", "both"):
        written.append(write_html(report, stem_dir / f"{stem}.html"))
    if formats in ("pdf", "both"):
        written.append(write_pdf(report, stem_dir / f"{stem}.pdf"))
    return written
