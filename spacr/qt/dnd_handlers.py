"""
Per-module drop handlers.

Each pipeline app has different expectations for what a "source"
means. This module encodes those policies as :class:`DropHandler`
subclasses that the AppScreen wires up at construction time.

Handler map (also read by ``get_handler``):

+-----------------+-------------------------------------------------------+
| App             | Accepts                                               |
+=================+=======================================================+
| mask            | folder w/ images (auto-parses regex + preview)        |
| measure         | folder named ``merged`` OR one containing merged/     |
| external_masks  | mixed image/label files or folders; assignment table  |
| annotate        | folder with ``measurements/measurements.db``          |
| classify        | folder with ``data/`` or ``measurements/``            |
| make_masks      | folder with images + optional masks/                  |
| map_barcodes    | folder with FASTQ; also a raw .fastq.gz drop          |
| umap            | folder with ``measurements/measurements.db``          |
| ml_analyze      | ditto                                                 |
| regression      | ditto — the database attaches to a PLATE ROW          |
| recruitment     | folder with per-well recruitment CSVs                 |
| activation      | folder with saved activation maps or the CV model dir |
| analyze_plaques | folder with plaque images                             |
| train_cellpose  | folder with image+mask pairs                          |
| cellpose_masks  | folder with images                                    |
| cellpose_all    | ditto                                                 |
| other modules   | an existing source folder or supported data file      |
+-----------------+-------------------------------------------------------+

Every handler falls back to CSV settings-import via :mod:`spacr.qt.dnd`
so users can also drop a settings CSV on any screen to load it.
"""
from __future__ import annotations

import logging
import os
from itertools import chain, islice
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from PySide6.QtCore import QEvent, QObject

from .. import chaining as _ch
from .. import ports as _kinds
from .dnd import (
    DropHandler, IMAGE_EXTS, find_image_folders_nearby, has_images_in,
    _report_drop_problem,
)
# Two extension sets, deliberately: IMAGE_EXTS (above) is what the filename
# preview will *sample*, containers included; RASTER_EXTS is what counts as
# one image on disk.
from .folder_metadata import IMAGE_EXTS as RASTER_EXTS
from .job_runner import JobRunner

LOG = logging.getLogger("spacr.qt.dnd_handlers")


# ---------------------------------------------------------------------------
# Shared setter — every AppScreen exposes the src widget through
# _settings_model._widgets["src"]; AnnotateScreen / MakeMasksScreen
# have their own _open_source / _open_folder methods.
# ---------------------------------------------------------------------------

def _set_src_on(screen, path) -> bool:
    """Best-effort set the screen's source path.

    Tries three shapes:
      1. ``screen._open_source(path)``          — AnnotateScreen
      2. ``screen._open_folder(path)``          — MakeMasksScreen
      3. ``screen._settings_model._widgets["src"].setText(path)`` — AppScreen
    """
    if hasattr(screen, "_open_source"):
        try:
            screen._open_source(path); return True
        except Exception:
            pass
    if hasattr(screen, "_open_folder"):
        try:
            screen._open_folder(path); return True
        except Exception:
            pass
    if hasattr(screen, "_settings_model"):
        try:
            model = screen._settings_model
            if model.set_value_for_key("src", path):
                return True
        except Exception:
            pass
        try:
            widget = screen._settings_model._widgets.get("src")
            if hasattr(widget, "setText"):
                if isinstance(path, (list, tuple)):
                    text = str(path[0]) if len(path) == 1 else repr(list(path))
                else:
                    text = str(path)
                widget.setText(text)
                return True
        except Exception:
            pass
    return False


def _log(screen, msg: str) -> None:
    """Put one line on the screen's console, if it has one.

    Every caller also logs through :data:`LOG`, so a console that refuses the
    line loses a convenience, not the message. Swallowed for that reason and
    that reason only: this runs inside Qt's drop-event dispatch, where an
    exception is a crash rather than an error dialog.
    """
    if hasattr(screen, "_console"):
        try:
            screen._console.append_stdout(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scanning a dropped folder without freezing the window
#
# A drop is delivered inside Qt's event dispatch, so everything a handler
# does happens on the GUI thread with the event loop stopped. Reading a
# directory is not "a bit of I/O": a user dropped a 100 000-file plate folder
# and the window froze for over a second -- three separate recursive walks of
# the same tree, one to detect a folder layout and two more inside the
# extraction planner, which called the detector again.
#
# So the walking moves to a worker via :class:`spacr.qt.job_runner.JobRunner`,
# and only the walking. ``handler.apply`` still runs -- and returns --
# synchronously; what changes is that the answer arrives a moment later,
# through a completion handler that runs back on the GUI thread and is the
# only place allowed to touch a widget.
# ---------------------------------------------------------------------------

#: How many image files the folder-layout guess looks at. The layout repeats,
#: so a probe is enough -- and stopping here means a folder with no layout to
#: find is never fully walked.
_FOLDER_PROBE = 30


def _is_alive(obj) -> bool:
    """True unless ``obj``'s C++ half has been destroyed underneath it.

    The Qt equivalent of "is this still there". PySide6 keeps handing out the
    Python wrapper after Qt has deleted the object it wraps, and touching it
    then raises ``RuntimeError: Internal C++ object already deleted`` -- from
    inside a slot, where there is no Python caller to catch it. Same shape as
    the ``getattr(self, "_target", None)`` guard in
    :meth:`spacr.qt.dnd._DropzoneFilter.eventFilter`: a destroyed wrapper is
    the answer, not an error.
    """
    if not isinstance(obj, QObject):
        return True          # a plain Python screen cannot be half-deleted
    try:
        from shiboken6 import isValid
    except Exception:
        try:
            obj.objectName()
        except RuntimeError:
            return False
        return True
    try:
        return bool(isValid(obj))
    except Exception:
        return False


class _DropScanner(QObject):
    """The one background walker a screen uses for its dropped folders.

    Parked on the **screen**, not on :class:`spacr.qt.dnd._DropzoneFilter`.
    The filter is a bare QObject with no ``closeEvent`` and no lifecycle hook
    of any kind, so a runner living there could never be told the screen is
    going away; the screen, being a widget, gets a Close event this can watch
    for. When the screen is not a QObject at all (a test double, a small
    controller object) the runner simply has no Qt parent and dies with the
    attribute that holds it.
    """

    def __init__(self, screen) -> None:
        # Assigned BEFORE super().__init__: parenting can deliver a ChildAdded
        # event synchronously, and this object is an event filter, so it must
        # already be able to answer for itself. (The same race that put the
        # assignment first in ``_DropzoneFilter.__init__``.)
        self._screen = screen
        parent = screen if isinstance(screen, QObject) else None
        super().__init__(parent)
        self._runner = JobRunner(self, app_key="folder scan")
        if parent is not None:
            parent.installEventFilter(self)

    # -- lifecycle --------------------------------------------------------
    def eventFilter(self, obj, event):     # noqa: N802  (Qt naming)
        # Every event delivered to the screen comes through here, so the
        # cheap discriminator goes first and the attribute lookup second.
        # ``getattr`` rather than ``self._screen`` for the reason spelled out
        # in ``_DropzoneFilter.eventFilter``: Qt keeps delivering events to a
        # filter after PySide6 has emptied its wrapper's __dict__, and an
        # AttributeError raised there has no Python caller to catch it.
        if (event.type() == QEvent.Close
                and obj is getattr(self, "_screen", None)):
            self.shutdown()
        return False                        # never consume the event

    def shutdown(self) -> None:
        """Drop the results in flight and wait briefly for their threads.

        Qt aborts the process if a running QThread is destroyed, so leaving a
        screen mid-scan has to be answered here rather than by hoping.
        """
        runner = getattr(self, "_runner", None)
        if runner is None:
            return
        try:
            runner.shutdown()
        except RuntimeError:
            pass

    # -- work -------------------------------------------------------------
    def submit(self, fn: Callable[[], Any],
               on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` on a worker thread, then ``on_done`` on the GUI thread."""
        return self._runner.submit(
            fn, lambda result: self._deliver(on_done, result))

    def _deliver(self, on_done: Callable[[Any], None], result) -> None:
        """Hand a finished scan to its handler, unless the screen has gone."""
        screen = getattr(self, "_screen", None)
        if screen is None or not _is_alive(screen) or not _is_alive(self):
            return
        on_done(result)

    # -- state (used by tests and by anything that wants to wait) ----------
    def is_busy(self) -> bool:
        runner = getattr(self, "_runner", None)
        return bool(runner is not None and runner.is_busy())

    def active_jobs(self) -> int:
        runner = getattr(self, "_runner", None)
        return 0 if runner is None else runner.active_jobs()


def _scanner_for(screen) -> Optional[_DropScanner]:
    """Return ``screen``'s scanner, creating it on the first drop.

    ``None`` when there is nowhere to keep one — a screen that refuses new
    attributes has nothing to hold a thread alive, and the caller runs the
    scan inline instead of leaking one.
    """
    scanner = getattr(screen, "_dnd_scanner", None)
    if scanner is not None and _is_alive(scanner):
        return scanner
    try:
        scanner = _DropScanner(screen)
        screen._dnd_scanner = scanner
    except Exception:
        LOG.debug("no background scanner for %r", type(screen), exc_info=True)
        return None
    return scanner


def _scan_then(screen, fn: Callable[[], Any],
               on_done: Callable[[Any], None]) -> bool:
    """Scan off the GUI thread; report back on it.

    :param fn: the filesystem work. Runs on a worker thread, so it may not
        touch a single widget — it returns plain data instead.
    :param on_done: given ``fn``'s return value, on the GUI thread. This is
        where logging, dialogs and settings widgets belong.
    :returns: True when the scan was dispatched to a thread, False when it
        had to run inline (no owner to hold one).
    """
    scanner = _scanner_for(screen)
    if scanner is not None:
        try:
            scanner.submit(fn, on_done)
            return True
        except Exception:
            # Qt refused to start a thread. Better a stall than no report.
            LOG.debug("falling back to an inline folder scan", exc_info=True)
    try:
        result = fn()
    except Exception:
        LOG.debug("folder scan failed", exc_info=True)
        return False
    on_done(result)
    return False


def scan_is_busy(screen) -> bool:
    """True while a dropped folder is still being walked for ``screen``."""
    scanner = getattr(screen, "_dnd_scanner", None)
    return bool(scanner is not None and _is_alive(scanner)
                and scanner.is_busy())


def active_scan_jobs(screen) -> int:
    """How many folder-scan threads ``screen`` still owns."""
    scanner = getattr(screen, "_dnd_scanner", None)
    if scanner is None or not _is_alive(scanner):
        return 0
    return scanner.active_jobs()


# -- the scans themselves. Worker-thread code: no Qt, no widgets, data out. --

def scan_mask_folder(path, sample: int = 20) -> Dict[str, Any]:
    """List the top level of a dropped folder once. Worker-safe.

    Returns the filenames the regex preview samples plus the total image
    count the report quotes. Both used to come from two separate listings of
    the same directory, taken on the GUI thread.
    """
    root = Path(path)
    if not root.is_dir():
        return {"names": [], "total": 0}
    try:
        entries = sorted(p for p in root.iterdir() if p.is_file())
    except OSError:
        return {"names": [], "total": 0}
    names = [p.name for p in entries
             if p.suffix.lower() in IMAGE_EXTS][:sample]
    # The count deliberately uses the narrower raster set, as it always has:
    # it is quoted as "N of M total sampled" beside a filename-regex preview,
    # and one .nd2 container is not M images yet.
    total = sum(1 for p in entries if p.suffix.lower() in RASTER_EXTS)
    return {"names": names, "total": total}


def scan_folder_structure(path) -> Dict[str, Any]:
    """Walk a dropped folder ONCE and return the folder-metadata report.

    Worker-safe: plain data in, plain data out, nothing Qt. The single walk
    is the point. This replaced three walks of the same tree —
    ``detect_folder_metadata`` did one, ``plan_folder_extraction`` did another
    and called ``detect_folder_metadata`` again for a third — all on the GUI
    thread, all inside the drop event.

    The probe is pulled off the *same* lazy generator the planner then drains,
    so a folder whose layout is not recognisable costs 30 files, not a
    traversal.

    :returns: ``{"labels": (...), "rows": [...], "error": ""}``. Empty
        ``labels`` means no folder layout was recognised and there is nothing
        to report.
    """
    from . import folder_metadata as fm
    from . import ingest_preview as ip

    out: Dict[str, Any] = {"labels": (), "rows": [], "error": ""}
    try:
        walk = fm.iter_image_files(path)
        probe = list(islice(walk, _FOLDER_PROBE))
        # Reached through the module, not a from-import, so that patching
        # ``spacr.qt.folder_metadata.detect_folder_metadata`` still works.
        template = fm.detect_folder_metadata(path, files=probe)
    except Exception:
        return out
    labels = getattr(template, "depth_labels", None) if template else None
    if not labels:
        return out                    # the rest of the tree is never walked
    out["labels"] = tuple(labels)
    try:
        out["rows"] = ip.plan_folder_extraction(
            path, files=chain(probe, walk), template=template)
    except Exception as exc:
        out["error"] = str(exc) or exc.__class__.__name__
    return out


# ---------------------------------------------------------------------------
# Mask — the star handler with regex-preview canvas
# ---------------------------------------------------------------------------

class MaskDropHandler(DropHandler):
    """Accept a folder of raw microscopy images and preview its filename
    regex parse. Multi-drop is supported."""

    def accepts_multiple(self) -> bool:
        return True

    def can_accept(self, path: Path) -> bool:
        if path.is_dir():
            return has_images_in(path)
        if path.is_file():
            from .multi_format import describe_file
            return path.suffix.lower() in (
                ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".czi",
                ".nd2", ".lif", ".npy", ".npz",
            ) and describe_file(path) is not None
        return False

    def suggest_alternatives(self, path: Path) -> List[Path]:
        if path.is_dir():
            return find_image_folders_nearby(path)
        return []

    def error_message(self, path: Path) -> str:
        return ("The mask module needs a folder of microscopy images "
                "(.tif / .png / .czi / .nd2 / .lif) at the top level.")

    def apply(self, path: Path, screen) -> None:
        src = path.parent if path.is_file() else path
        _set_src_on(screen, str(src))
        _log(screen, f"[drop] mask src = {src}\n")
        if path.is_dir():
            # Read the folder on a worker thread and render the report when
            # it comes back.
            #
            # What this replaced: ``QTimer.singleShot(50, ...)``, commented
            # "asynchronously so the UI doesn't stall". It is not
            # asynchronous. A single-shot timer defers to the next turn of
            # the event loop and then runs everything ON the GUI thread, with
            # the loop stopped — the freeze just started 50 ms later than the
            # drop, which is why it was never traced back to here.
            _scan_then(screen,
                       lambda: scan_mask_folder(path),
                       lambda scan: _render_mask_report(path, screen, scan))
            return
        # A single container file. Describing it reads a header, not a tree,
        # and the branch has to set widgets and open a table — so it stays on
        # the GUI thread, one turn later so the src field paints first.
        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: _report_regex_on_mask(path, screen))
        except Exception:
            pass


def _report_regex_on_mask(path: Path, screen) -> None:
    """Sample filenames, apply / auto-detect the metadata regex, and
    write a tabular report into the AppScreen's Console.

    Synchronous — it reads the folder inline. :meth:`MaskDropHandler.apply`
    does **not** call it that way: a drop scans on a worker thread and calls
    :func:`_render_mask_report` with the result. This entry point is for
    callers that already know the folder is small (and for tests that want
    the whole report in one call).

    On a good match: prints an aligned column table of up to 10
    randomly-sampled records + a ``✓ All required fields captured``
    footer.

    On a partial / no match: prints a warning list AND opens the
    :class:`RegexEditorDialog` so the user can edit the regex or
    click "Auto detect" for a smarter guess. Saved regex is pushed
    back into the ``custom_regex`` settings widget.

    Handles two kinds of drops:
      * A folder of image files (existing default).
      * A single dataset-in-a-file drop (``.npz`` / ``.lif`` /
        ``.nd2`` / multi-page tiff / big ``.npy``) — reported via
        :mod:`spacr.qt.multi_format`.
    """
    from . import multi_format as mf

    # ── Folder path ───────────────────────────────────────────────
    if not path.is_file():
        _render_mask_report(path, screen, scan_mask_folder(path))
        return

    _log(screen, "\n")

    # ── Single-file dataset path ──────────────────────────────────
    desc = mf.describe_file(path)
    if desc is None:
        _log(screen, f"[drop] dropped file {path.name} — unrecognised "
                     f"single-file dataset format.\n")
        return
    # Container formats (nd2/czi/lif/multi-page tiff/npz) are expanded
    # to the canonical Yokogawa layout by the pipeline's auto converter.
    # Set metadata_type='auto' so that conversion actually runs, and
    # point src at the containing folder.
    _set_screen_setting(screen, "metadata_type", "auto")
    _log(screen,
         f"[drop] single-file dataset: {desc.summary()}\n"
         f"       Set metadata_type = 'auto' — spaCR will auto-extract "
         f"every image (channels/z/fields) from this container into the "
         f"canonical filename structure on the first Run, and write a "
         f"filename_map.csv linking each generated file back to it.\n")
    # Preview the planned extraction and let the user edit the
    # plate/well/field/channel assignment before committing.
    try:
        from . import ingest_preview as ip
        rows = ip.plan_container_extraction(desc)
        if rows:
            _log(screen, f"[drop] planned extraction — "
                         f"{ip.summarize_rows(rows)}\n")
            _open_metadata_table(rows, path.parent, screen)
    except Exception as e:
        _log(screen, f"[drop] metadata preview unavailable: {e}\n")


def _render_mask_report(path: Path, screen, scan: Dict[str, Any]) -> None:
    """Write the filename-regex report for a finished :func:`scan_mask_folder`.

    The GUI-thread half of a folder drop: it reads and writes settings
    widgets and can open the regex editor, so it must run here — while
    everything that touched the filesystem already happened on the worker
    that produced ``scan``.
    """
    from . import regex_detect as rd

    _log(screen, "\n")
    filenames = list(scan.get("names") or ())
    if not filenames:
        _log(screen, "[drop] no images found in the top level of "
                     f"{Path(path).name} — nothing to preview.\n")
        return
    total_images = int(scan.get("total") or 0)

    # Read the user's current custom_regex (may be empty)
    custom = ""
    try:
        w = screen._settings_model._widgets.get("custom_regex")
        if w is not None and hasattr(w, "text"):
            custom = (w.text() or "").strip()
    except Exception:
        pass

    # Auto-detect if the user has no custom regex or if it fails
    if custom:
        records, missed = rd.apply_regex(filenames, custom)
        pattern, label = custom, "custom"
        n_matches = len(records)
    else:
        pattern, label, n_matches = rd.auto_detect_regex(filenames)
        records, missed = ([], filenames[:]) \
                          if pattern is None \
                          else rd.apply_regex(filenames, pattern)

    _log(screen,
         f"[drop] mask · folder = {path}\n"
         f"[drop] regex ({label}) — matched {n_matches}/"
         f"{len(filenames)} sampled filenames\n"
         f"[drop] {len(filenames)} of {total_images} total sampled "
         f"— showing up to 10 rows:\n\n")

    if records:
        table = rd.tabulate_records(records, max_rows=10)
        _log(screen, table + "\n")

    warnings = rd.validate_records(records, multichannel=True)
    if warnings:
        for w in warnings:
            _log(screen, f"⚠ {w}\n")
        # Offer folder-structure metadata as an alternative to a filename regex
        # (useful when the plate/well/field/channel live in directory names).
        _report_folder_structure(path, screen)
        _log(screen, "→ Opening the regex editor so you can enter a custom "
                     "pattern that matches your filenames live. Use the "
                     "Auto-detect button or edit the pattern manually — or use "
                     "the folder-structure option above.\n")
        _open_regex_editor(filenames, pattern or "", screen)
    else:
        _log(screen, "✓ All required fields captured "
                     "(wellID / fieldID, chanID).\n")
        # Confirm even when nothing looks wrong. A regex that captures every
        # required field can still be capturing the WRONG field -- a well ID
        # read as a field ID validates perfectly and silently mislabels the
        # whole plate. The check that catches that is a person reading the
        # parsed columns, which only happens if they are shown.
        #
        # Previously this branch pushed the pattern with no prompt, so the
        # editor appeared only when validation failed. That made the common
        # case (a naming dialect that fits) the one case nobody ever
        # verified, and made the prompt read as an error rather than a step.
        _log(screen, "→ Confirm the parsed columns above match your naming "
                     "before running. Edit the pattern if a column is "
                     "holding the wrong value.\n")
        _open_regex_editor(filenames, pattern or "", screen,
                           confirming=True, fallback=pattern)


def _set_screen_setting(screen, key: str, value) -> bool:
    """Set a settings widget's value on the screen (combo or line edit)."""
    try:
        w = screen._settings_model._widgets.get(key)
        if w is None:
            return False
        from PySide6.QtWidgets import QComboBox, QLineEdit
        if isinstance(w, QComboBox):
            idx = w.findText(str(value))
            if idx >= 0:
                w.setCurrentIndex(idx)
                return True
            w.setEditText(str(value))
            return True
        if isinstance(w, QLineEdit):
            w.setText(str(value))
            return True
        if hasattr(w, "setText"):
            w.setText(str(value))
            return True
    except Exception:
        pass
    return False


def _report_folder_structure(path, screen) -> None:
    """Detect metadata from the folder structure and report it as an
    alternative to a filename regex (folder_metadata is otherwise unwired).

    Returns as soon as the walk is **dispatched**. The walk itself is
    :func:`scan_folder_structure` on a worker thread, and
    :func:`_render_folder_structure` writes the report when it lands — that
    split is the whole fix for "I dropped a big folder in and it froze".
    Wait for it with :func:`scan_is_busy`.
    """
    _scan_then(screen,
               lambda: scan_folder_structure(path),
               lambda result: _render_folder_structure(path, screen, result))


def _render_folder_structure(path, screen, result: Dict[str, Any]) -> None:
    """Report a finished :func:`scan_folder_structure`. GUI thread only."""
    labels = (result or {}).get("labels") or ()
    if not labels:
        return
    _log(screen,
         "\n[drop] folder-structure alternative — detected metadata from the "
         "directory layout:\n"
         f"       path depth → {' / '.join(str(l) for l in labels)}\n"
         "       If your images are organised by folder (e.g. plate/well/"
         "field) rather than by filename, this can be used instead of a "
         "filename regex.\n")
    error = result.get("error") or ""
    if error:
        _log(screen, f"[drop] folder-structure preview unavailable: {error}\n")
        return
    # Make the detection actionable: the preview of how each image would be
    # named opens in the editable metadata table so the user can accept or
    # correct it, writing a filename_map.csv the pipeline consumes.
    rows = result.get("rows") or []
    if not rows:
        return
    from . import ingest_preview as ip
    _log(screen, f"[drop] folder-structure plan — "
                 f"{ip.summarize_rows(rows)}\n")
    _open_metadata_table(rows, path, screen)


#: Fallback owner for metadata dialogs whose screen cannot hold a reference.
#: Entries are removed again when the dialog emits ``finished``.
_ORPHAN_DIALOGS: List = []


def _open_metadata_table(rows, dst, screen) -> None:
    """Open the editable metadata table so the user can review/correct the
    inferred plate/well/field/channel assignment before extraction.

    On Apply it writes ``filename_map.csv`` into ``dst`` and logs the path.
    Fails quietly (and never blocks) if Qt/dialog construction is
    unavailable — e.g. in a headless context.
    """
    def _on_apply(csv_path):
        _log(screen, f"[drop] wrote metadata map → {csv_path}\n")

    # ``dst`` arrives as the FOLDER the data lives in, but the dialog hands
    # it straight to folder_metadata.save_filename_map(), which treats its
    # argument as the CSV file to open() for writing. Passing a directory
    # made every Apply raise IsADirectoryError and silently write nothing.
    dst = Path(dst)
    if dst.is_dir() or dst.suffix.lower() != ".csv":
        dst = dst / "filename_map.csv"

    try:
        from .widgets.metadata_table import MetadataTableDialog
    except Exception:
        return
    try:
        parent = screen if hasattr(screen, "window") else None
        dlg = MetadataTableDialog(rows, dst, on_apply=_on_apply, parent=parent)
    except Exception as e:
        _log(screen, f"[drop] could not open metadata table: {e}\n")
        return
    # Show modeless (never exec()) so the drop handler never blocks — a
    # blocking modal would hang headless/offscreen runs. Keep a reference on
    # the screen so the dialog isn't garbage-collected while open.
    try:
        holder = getattr(screen, "_metadata_dialogs", None)
        if holder is None:
            holder = []
            try:
                screen._metadata_dialogs = holder
            except Exception:
                # The screen refuses new attributes (__slots__, proxy, …).
                # Park the reference module-side: the dialog is parentless
                # here, so without SOME live reference it is collected the
                # moment this function returns and the user never sees it.
                holder = _ORPHAN_DIALOGS
        holder.append(dlg)
        dlg.finished.connect(lambda *_: holder.remove(dlg)
                             if dlg in holder else None)
        dlg.setModal(False)
        dlg.show()
    except Exception:
        # Non-interactive / headless — leave the console report in place.
        pass


def _open_regex_editor(filenames: list, initial: str, screen,
                       confirming: bool = False,
                       fallback: Optional[str] = None) -> None:
    """Show the regex editor, either to fix a bad match or to confirm a good one.

    :param confirming: the pattern already validated; this is a review step,
        so dismissing the dialog keeps ``fallback`` rather than leaving the
        screen with no regex at all.
    :param fallback: pattern to keep when a confirmation is dismissed.
    """
    try:
        from .regex_editor import RegexEditorDialog
    except Exception:
        # No editor available. A validated pattern is still better than none,
        # so a confirmation that cannot be shown must not lose it.
        if confirming and fallback:
            _push_regex_to_screen(fallback, screen)
        return
    try:
        from PySide6.QtWidgets import QDialog
        dlg = RegexEditorDialog(filenames, initial_regex=initial,
                                 multichannel=True, parent=screen)
        # QDialog.Accepted, not dlg.Accepted: PySide6 exposes the enum on the
        # class, not on instances. This only ever ran when validation failed,
        # so the AttributeError sat here until the editor started opening on
        # every import to confirm a good match.
        if dlg.exec() == QDialog.Accepted and dlg.regex:
            _push_regex_to_screen(dlg.regex, screen)
            _log(screen, f"[drop] saved custom regex: {dlg.regex}\n")
        elif confirming and fallback:
            _push_regex_to_screen(fallback, screen)
            _log(screen, f"[drop] kept the detected regex: {fallback}\n")
    except Exception as e:
        _log(screen, f"[drop] regex editor failed: {e}\n")
        if confirming and fallback:
            _push_regex_to_screen(fallback, screen)


def _push_regex_to_screen(pattern: Optional[str], screen) -> None:
    if not pattern:
        return
    try:
        w = screen._settings_model._widgets.get("custom_regex")
        if w is not None and hasattr(w, "setText"):
            w.setText(pattern)
    except Exception:
        pass


# NOTE: ``_count_images`` used to live here, and the report called it right
# after ``sample_image_names`` -- two listings of the same directory, both on
# the GUI thread. :func:`scan_mask_folder` produces the sample and the count
# from one listing, on a worker.


# ---------------------------------------------------------------------------
# Measure — must be `merged` or contain merged/
# ---------------------------------------------------------------------------

class MeasureDropHandler(DropHandler):
    """Accept the ``merged`` folder produced by the mask module, or a
    parent folder that contains one."""

    def can_accept(self, path: Path) -> bool:
        if path.is_file():
            return path.suffix.lower() in (".npy", ".tif", ".tiff")
        if not path.is_dir():
            return False
        # Direct: dropped `merged` folder itself
        if path.name == "merged" and has_images_in(path, exts=(".tif", ".tiff", ".npy")):
            return True
        # Contains: dropped a plate parent that HAS merged/
        merged = path / "merged"
        return merged.is_dir()

    def suggest_alternatives(self, path: Path) -> List[Path]:
        hits: List[Path] = []
        # Look for merged/ under nearby folders
        if path.is_dir():
            for child in path.iterdir():
                if child.is_dir() and (child / "merged").is_dir():
                    hits.append(child / "merged")
            if path.parent and path.parent.is_dir():
                for sib in path.parent.iterdir():
                    if sib.is_dir() and (sib / "merged").is_dir():
                        hits.append(sib / "merged")
        return hits

    def error_message(self, path: Path) -> str:
        return ("Measure needs the ``merged`` folder produced by the "
                "mask module. Drop the folder called `merged` (or a "
                "plate folder that contains one).")

    def apply(self, path: Path, screen) -> None:
        # The plate folder, not ``merged/`` inside it.
        #
        # This used to drill *into* ``merged``, and auto-chaining fills the
        # same field with the plate — so dropping a folder and letting the
        # chain fill it produced two different strings for one project. Both
        # run (``spacr.ports.project_root`` hops a trailing ``merged``), which
        # is exactly why the disagreement survived: it only showed up when a
        # settings CSV written by one was compared against the other.
        #
        # :func:`spacr.chaining.resolve_drop` is the single answer now, and it
        # asks the registry first, so a plate whose merged arrays were written
        # somewhere unusual resolves to where the producer says they are.
        resolved = _resolve_for(self, "measure", path)
        target = resolved.target_for(_kinds.MERGED_ARRAYS) if resolved else None
        if target is not None:
            _set_src_on(screen, str(target.value))
            _log(screen, f"[drop] measure src = {target.value}\n"
                         f"[drop] merged arrays → {target.location} "
                         f"(from the {target.source})\n")
            return
        if path.is_file():
            path = path.parent
        if path.name == "merged":
            path = path.parent
        _set_src_on(screen, str(path))
        _log(screen, f"[drop] measure src = {path}\n")


# ---------------------------------------------------------------------------
# Annotate — expects a measurements DB
# ---------------------------------------------------------------------------

class AnnotateDropHandler(DropHandler):
    """Accept a plate folder with ``measurements/measurements.db`` or
    the .db file itself."""

    def can_accept(self, path: Path) -> bool:
        if path.is_file() and path.suffix.lower() == ".db":
            return True
        if path.is_dir():
            return (path / "measurements" / "measurements.db").is_file()
        return False

    def error_message(self, path: Path) -> str:
        return ("Annotate needs a plate folder that has "
                "measurements/measurements.db (produced by the "
                "measure module).")

    def apply(self, path: Path, screen) -> None:
        # Drop-db: use its containing plate folder as src.
        # The canonical layout is <plate>/measurements/measurements.db, so
        # climb two levels ONLY when the db really sits in a measurements/
        # folder — a loose .db (which can_accept also allows) must resolve
        # to its own directory, not that directory's parent.
        if path.is_file() and path.suffix.lower() == ".db":
            path = (path.parent.parent
                    if path.parent.name == "measurements"
                    else path.parent)
        _set_src_on(screen, str(path))
        _log(screen, f"[drop] annotate src = {path}\n")


# ---------------------------------------------------------------------------
# Classify — same DB requirement as annotate, plus optional model dir
# ---------------------------------------------------------------------------

class ClassifyDropHandler(DropHandler):
    """Accept a plate folder with ``measurements/measurements.db`` or
    a folder produced by the annotate step."""

    def can_accept(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        return (path / "measurements" / "measurements.db").is_file() \
               or (path / "data").is_dir() \
               or (path / "train").is_dir()

    def error_message(self, path: Path) -> str:
        return ("Classify needs a plate folder with either "
                "measurements/measurements.db, a data/ crop folder, or an "
                "existing dataset root containing train/<class>/ folders. "
                "Run Measure with object-crop output enabled if the plate has "
                "no measurements database or crops yet.")

    def apply(self, path: Path, screen) -> None:
        paths = []
        try:
            current = screen._settings_model.collect().get("src")
            if isinstance(current, (list, tuple)):
                paths.extend(str(item) for item in current if str(item).strip())
            elif current and str(current).strip() not in ("", "path"):
                paths.append(str(current))
        except Exception:
            pass
        value = str(path)
        if value not in paths:
            paths.append(value)
        _set_src_on(screen, paths)
        _log(screen, f"[drop] classify src = {path}\n")
        if len(paths) > 1:
            _log(screen, f"[drop] classify plates = {paths}\n")


# ---------------------------------------------------------------------------
# Make Masks — image folder, optional companion masks/
# ---------------------------------------------------------------------------

class MakeMasksDropHandler(DropHandler):
    """Accept a folder with images (or image+mask pairs)."""

    def can_accept(self, path: Path) -> bool:
        return path.is_dir() and has_images_in(path)

    def suggest_alternatives(self, path: Path) -> List[Path]:
        if path.is_dir():
            return find_image_folders_nearby(path)
        return []

    def error_message(self, path: Path) -> str:
        return ("Make Masks needs a folder of images to fine-tune "
                "Cellpose against.")

    def apply(self, path: Path, screen) -> None:
        _set_src_on(screen, str(path))
        _log(screen, f"[drop] make_masks folder = {path}\n")


# ---------------------------------------------------------------------------
# Map Barcodes — fastq file OR folder with fastqs
# ---------------------------------------------------------------------------

class MapBarcodesDropHandler(DropHandler):
    """Accept a FASTQ file (``.fastq``/``.fastq.gz``) or a folder
    containing one."""

    _FQ_EXTS = (".fastq", ".fastq.gz", ".fq", ".fq.gz")

    def can_accept(self, path: Path) -> bool:
        if path.is_file():
            name = path.name.lower()
            return any(name.endswith(x) for x in self._FQ_EXTS)
        if path.is_dir():
            for child in path.iterdir():
                if child.is_file() and any(
                    child.name.lower().endswith(x) for x in self._FQ_EXTS
                ):
                    return True
        return False

    def error_message(self, path: Path) -> str:
        return ("Map Barcodes needs a FASTQ file (.fastq / .fastq.gz) "
                "or a folder that contains one.")

    def apply(self, path: Path, screen) -> None:
        # If a file: point src at the containing folder + fastq at the file
        if path.is_file():
            fq_path = str(path)
            src_path = str(path.parent)
        else:
            src_path = str(path)
            fq_path = None
        _set_src_on(screen, src_path)
        if fq_path and hasattr(screen, "_settings_model"):
            for key in ("fastq", "fastq_path", "fq"):
                w = screen._settings_model._widgets.get(key)
                if w is not None and hasattr(w, "setText"):
                    w.setText(fq_path); break
        _log(screen, f"[drop] map_barcodes src = {src_path}\n")


# ---------------------------------------------------------------------------
# Generic "measurements DB" downstream handler — UMAP / ML / regression
# ---------------------------------------------------------------------------

class MeasurementsDropHandler(DropHandler):
    """Accept a database, its measurements folder, or its plate folder.

    Where the screen takes its inputs ONE ROW PER PLATE -- today only
    Regression, through the ``paired_data`` table -- the database is attached
    to a plate row instead of setting ``src``. That is not an app-key special
    case: it follows the shape of the screen, so any panel that grows a
    per-plate input table gets it without a registry edit.

    It matters because the two gestures used to disagree. Dropping
    ``measurements.db`` ON the regression input table attaches it to a plate
    (instruction 130); dropping the same file two inches higher, on the
    screen around it, landed here and set ``src`` -- a key the regression
    panel does not even display. The drop reported success and changed
    nothing the user could see.
    """

    def can_accept(self, path: Path) -> bool:
        if path.is_file():
            return path.name == "measurements.db"
        if path.is_dir():
            return (
                (path / "measurements" / "measurements.db").is_file()
                or (path / "measurements.db").is_file()
            )
        return False

    def error_message(self, path: Path) -> str:
        return ("This module needs a plate folder with "
                "measurements/measurements.db.")

    @staticmethod
    def database_file(path: Path):
        """The database ``path`` names: itself, or the one under it.

        Returns ``None`` when there is no database file to be found, so a
        caller can fall back rather than hand a folder to something that
        expects to open a database.
        """
        if path.is_file():
            return path if _is_database_path(path) else None
        for candidate in (path / "measurements" / "measurements.db",
                          path / "measurements.db"):
            if candidate.is_file():
                return candidate
        return None

    def apply(self, path: Path, screen) -> None:
        # A screen whose inputs are one row per plate wants the database on a
        # PLATE ROW; `src` is not where its measurements live.
        widget = _paired_input_table(screen)
        attach = getattr(widget, "attach_database", None)
        database = self.database_file(path)
        if database is not None and callable(attach):
            message = attach(str(database))
            LOG.info("measurements drop: %s", message)
            _log(screen, f"[drop] {message}\n")
            return
        # Same resolution as auto-chaining, for the same reason as in
        # :meth:`MeasureDropHandler.apply`: the registry knows where the
        # producer actually wrote, and the declared layout answers when no
        # run was ever registered.
        app_key = str(getattr(screen, "app_key", "") or "")
        resolution = _resolve_for(self, app_key, path) if app_key else None
        target = (resolution.target_for(_kinds.MEASUREMENTS_DB)
                  if resolution is not None else None)
        if target is not None:
            _set_src_on(screen, str(target.value))
            _log(screen, f"[drop] src = {target.value}\n"
                         f"[drop] measurements → {target.location} "
                         f"(from the {target.source})\n")
            return
        if path.is_file():
            path = path.parent
        if path.name == "measurements" and (path / "measurements.db").is_file():
            path = path.parent
        _set_src_on(screen, str(path))
        _log(screen, f"[drop] src = {path}\n")


def _is_database_path(path) -> bool:
    """Is this a measurements database? Asked of the widgets' own rule.

    Imported here rather than at module scope for the reason the local import
    in :class:`SweepInputsDropHandler` gives: importing a widget module pulls
    in the whole widgets package, and this module is imported while the first
    window is still being built.
    """
    from .widgets.file_list import is_database_path
    return is_database_path(path)


def _paired_input_table(screen):
    """The screen's ``paired_data`` widget, or ``None`` if it has none."""
    model = getattr(screen, "_settings_model", None)
    widgets = getattr(model, "_widgets", None)
    return widgets.get("paired_data") if isinstance(widgets, dict) else None


class DatabaseDropHandler(DropHandler):
    """Open a dropped measurements database in the Database Browser."""

    def can_accept(self, path: Path) -> bool:
        return MeasurementsDropHandler().can_accept(path)

    def error_message(self, path: Path) -> str:
        return (
            "Database Browser accepts measurements.db, the measurements "
            "folder that contains it, or the parent run folder."
        )

    def apply(self, path: Path, screen) -> None:
        if not hasattr(screen, "set_database"):
            raise TypeError("This screen cannot open a database.")
        if not screen.set_database(str(path)):
            raise ValueError(
                getattr(screen, "last_error", "")
                or f"Could not open {path}."
            )


class SourceDropHandler(DropHandler):
    """General source-path policy for modules without a narrower contract.

    Every standard :class:`AppScreen` has a ``src`` field. Accepting a real
    directory here gives newer and less-specialised modules drag-and-drop
    support automatically instead of requiring a registry edit for every
    screen. A dropped file is only accepted for known data extensions and is
    normalised to its containing directory.

    Where the module declares ports, the value comes from
    :func:`spacr.chaining.resolve_drop` — the same answer auto-chaining would
    fill the field with — so dropping ``<plate>/measurements`` and letting the
    chain fill ``src`` cannot produce two different strings. A module with no
    declaration keeps the plain normalisation below, which is all there is to
    go on.
    """

    _DATA_EXTS = {
        ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".czi", ".nd2",
        ".lif", ".npy", ".npz", ".csv", ".db", ".sqlite", ".sqlite3",
        ".fastq", ".fq", ".gz", ".tar",
    }

    def can_accept(self, path: Path) -> bool:
        return path.is_dir() or (
            path.is_file() and path.suffix.lower() in self._DATA_EXTS
        )

    def error_message(self, path: Path) -> str:
        return "Drop an existing source folder or a supported data file."

    def apply(self, path: Path, screen) -> None:
        app_key = str(getattr(screen, "app_key", "") or "")
        resolution = (_resolve_for(self, app_key, path) if app_key else None)
        if resolution is not None and resolution.targets:
            target = resolution.targets[0]
            if not _set_src_on(screen, str(target.value)):
                raise TypeError(
                    "This module has no source field to receive the drop.")
            _log(screen,
                 f"[drop] src = {target.value}\n"
                 f"[drop] resolved {target.kind} → {target.location} "
                 f"(from the {target.source})\n")
            return
        if path.is_file():
            path = path.parent
        if path.name == "measurements" and (path / "measurements.db").is_file():
            path = path.parent
        if not _set_src_on(screen, str(path)):
            raise TypeError("This module has no source field to receive the drop.")
        _log(screen, f"[drop] src = {path}\n")


def _measurement_database(path: Path) -> Optional[Path]:
    """Return a canonical measurements database at or immediately below path."""
    if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        return path
    if not path.is_dir():
        return None
    for candidate in (
        path / "measurements.db",
        path / "measurements" / "measurements.db",
    ):
        if candidate.is_file():
            return candidate
    return None


class SweepInputsDropHandler(DropHandler):
    """Route dropped CSVs into Parameter Sweep's score and count lists.

    The sweep takes the same two inputs the regression does, but holds them in
    two separate list widgets rather than one paired table, so the side has to
    be decided before the file is added. It is decided the same way Regression
    decides it -- from the CSV header, via
    :func:`spacr.qt.widgets.file_list.side_for_header` -- because a count table
    filed as a score is not an error the user sees, it is a wrong sweep.

    A dropped FOLDER contributes the CSVs directly inside it, which is what
    makes "drop the plate folder" work for a screen whose whole point is
    running many plates at once.
    """

    def accepts_multiple(self) -> bool:
        return True

    @staticmethod
    def _tables(path: Path):
        """The CSVs ``path`` contributes: itself, or the ones it contains."""
        if path.is_dir():
            return sorted(p for p in path.iterdir()
                          if p.is_file() and p.suffix.lower() == ".csv")
        if path.is_file() and path.suffix.lower() == ".csv":
            return [path]
        return []

    def can_accept(self, path: Path) -> bool:
        return bool(self._tables(path))

    def error_message(self, path: Path) -> str:
        return ("Parameter Sweep accepts per-object score CSVs, gRNA count "
                "CSVs, or a folder holding them.")

    def apply(self, path: Path, screen) -> None:
        from .widgets.file_list import side_for_header

        score = getattr(screen, "score_data", None)
        count = getattr(screen, "count_data", None)
        if score is None or count is None:
            raise TypeError("Parameter Sweep has no score/count inputs.")
        tables = self._tables(path)
        if not tables:
            raise ValueError(self.error_message(path))
        for table in tables:
            side = side_for_header(table)
            target = count if side == "count" else score
            target.add_paths([str(table)])
            _log(screen, f"[drop] parameter_sweep {side} += {table}\n")


class ExplainCvInputsDropHandler(DropHandler):
    """Fill Explain CV's database or prediction input from one drop."""

    def accepts_multiple(self) -> bool:
        return True

    def can_accept(self, path: Path) -> bool:
        return bool(
            _measurement_database(path)
            or (path.is_file() and path.suffix.lower() == ".csv")
        )

    def error_message(self, path: Path) -> str:
        return (
            "Explain CV Model accepts measurements.db, its project folder, "
            "or an existing per-object prediction CSV."
        )

    def apply(self, path: Path, screen) -> None:
        panel = getattr(screen, "explain", None)
        if panel is None:
            raise TypeError("Explain CV Model has no input panel.")
        database = _measurement_database(path)
        if database is not None:
            panel.database.setText(str(database))
            _log(screen, f"[drop] explain_cv database = {database}\n")
            return
        panel.predictions.setText(str(path))
        panel._refresh_prediction_columns()
        _log(screen, f"[drop] explain_cv predictions = {path}\n")


class InvestigateHitInputsDropHandler(DropHandler):
    """Fill Investigate Hit's provenance inputs without guessing a hit."""

    def accepts_multiple(self) -> bool:
        return True

    def can_accept(self, path: Path) -> bool:
        return bool(
            _measurement_database(path)
            or path.is_dir()
            or (path.is_file() and path.suffix.lower() == ".csv")
        )

    def error_message(self, path: Path) -> str:
        return (
            "Investigate Hit accepts measurements.db, a prediction or "
            "well/guide-fraction CSV, or the exact regression-results folder."
        )

    @staticmethod
    def _looks_like_fractions(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
                header = stream.readline()
        except OSError:
            return False
        fields = {field.strip().casefold() for field in header.split(",")}
        has_guide = any("guide" in field or "grna" in field for field in fields)
        has_fraction = any("fraction" in field or "abundance" in field
                           for field in fields)
        return has_guide and has_fraction

    def apply(self, path: Path, screen) -> None:
        panel = getattr(screen, "investigate", None)
        if panel is None:
            raise TypeError("Investigate Hit has no input panel.")
        database = _measurement_database(path)
        if database is not None:
            panel.database.setText(str(database))
            _log(screen, f"[drop] investigate_hit database = {database}\n")
            return
        if path.is_dir():
            panel.regression_folder.setText(str(path))
            _log(screen, f"[drop] investigate_hit regression results = {path}\n")
            return
        if self._looks_like_fractions(path):
            panel.fractions.setText(str(path))
            _log(screen, f"[drop] investigate_hit guide fractions = {path}\n")
            return
        panel.predictions.setText(str(path))
        panel._refresh_prediction_columns()
        _log(screen, f"[drop] investigate_hit predictions = {path}\n")


class ExternalMasksDropHandler(DropHandler):
    """Append mixed intensity images and external label masks to the mapper."""

    def accepts_multiple(self) -> bool:
        return True

    def can_accept(self, path: Path) -> bool:
        if path.is_dir():
            return True
        return path.is_file() and path.name.lower().endswith(
            (".tif", ".tiff", ".ome.tif", ".ome.tiff",
             ".png", ".jpg", ".jpeg", ".bmp"))

    def error_message(self, path: Path) -> str:
        return (
            "Drop image or mask files, or folders containing TIFF, PNG, "
            "JPEG or BMP files.")

    def apply(self, path: Path, screen) -> None:
        try:
            model = screen._settings_model
            widget = model._widgets["inputs"]
            added = widget.add_paths([str(path)])
        except Exception as exc:
            raise TypeError(
                "The External Masks input-mapping table is unavailable."
            ) from exc
        if added <= 0:
            raise ValueError(f"No supported images were found under {path}.")
        destination = path if path.is_dir() else path.parent
        dst_widget = model._widgets.get("dst")
        if dst_widget is not None and not model._read_widget(dst_widget):
            model.set_value_for_key("dst", f"{destination}_spacr")
        _log(
            screen,
            f"[drop] detected {added} external image/mask file(s) from "
            f"{path}; review the assignments before Run.\n",
        )


# ---------------------------------------------------------------------------
# Tool and results screens — these are not SettingsWidgets/AppScreens, so
# each handler calls the screen's small public configuration API directly.
# ---------------------------------------------------------------------------

_MODEL_SUFFIXES = (
    ".cp_model", ".pth", ".pt", ".ckpt", ".onnx", ".h5", ".keras",
)
_IMAGE_SUFFIXES = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".czi", ".nd2",
    ".lif", ".npy", ".npz",
}


def _contains_suffix(folder: Path, suffixes, *, recursive: bool = False) -> bool:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    try:
        return any(
            child.is_file()
            and any(child.name.lower().endswith(ext) for ext in suffixes)
            for child in iterator
        )
    except OSError:
        return False


def _settings_files(path: Path) -> List[Path]:
    """Settings snapshots directly associated with a dropped plate."""
    roots = [path / "settings", path] if path.is_dir() else []
    found: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        found.extend(sorted(
            child for child in root.iterdir()
            if child.is_file() and child.suffix.lower() == ".csv"
            and "setting" in child.name.lower()
        ))
    return list(dict.fromkeys(found))


def _module_from_settings(path: Path, default: str = "mask") -> str:
    """Infer a runnable GUI module from spaCR's settings snapshot name."""
    stem = path.stem.lower()
    aliases = (
        ("measure_crop", "measure"),
        ("crop_measure", "measure"),
        ("gen_masks", "mask"),
        ("gen_mask", "mask"),
        ("train_test", "classify"),
        ("ml_analyze", "ml_analyze"),
        ("map_barcodes", "map_barcodes"),
        ("regression", "regression"),
        ("recruitment", "recruitment"),
        ("annotate", "annotate"),
        ("classify", "classify"),
        ("measure", "measure"),
        ("mask", "mask"),
    )
    for token, module in aliases:
        if token in stem:
            return module
    return default


class ForeignProjectDropHandler(DropHandler):
    """Populate Import Project from image folders, tables and mapping files."""

    def accepts_multiple(self) -> bool:
        return True

    def can_accept(self, path: Path) -> bool:
        return path.is_dir() or (
            path.is_file() and path.suffix.lower() in
            (_IMAGE_SUFFIXES | {".csv", ".tsv", ".xlsx", ".xls",
                               ".parquet", ".json"})
        )

    def error_message(self, path: Path) -> str:
        return ("Import Project accepts an image/mask folder, an image file, "
                "a CSV/TSV/Excel/Parquet measurement table, or a JSON mapping.")

    def apply(self, path: Path, screen) -> None:
        suffix = path.suffix.lower()
        is_mapping_csv = False
        if path.is_file() and suffix == ".csv":
            try:
                header = {
                    value.strip().lower()
                    for value in path.open(
                        "r", encoding="utf-8", errors="replace"
                    ).readline().split(",")
                }
                is_mapping_csv = {
                    "source", "target", "transform", "unit_in", "unit_out",
                    "note",
                }.issubset(header)
            except OSError:
                pass
        if (path.is_file() and (suffix == ".json" or is_mapping_csv)
                and hasattr(screen, "load_mapping")):
            if screen.load_mapping(str(path)) is False:
                raise ValueError(f"Could not load mapping {path}.")
            return
        if path.is_file() and suffix in {".csv", ".tsv", ".xlsx", ".xls",
                                        ".parquet"}:
            screen.set_measurements(str(path))
            return
        if path.is_dir() and any(
                token in path.name.lower()
                for token in ("mask", "label", "segmentation")):
            try:
                object_type = str(screen._object_box.currentData())
            except Exception:
                object_type = "cell"
            if screen.add_mask_folder(object_type, str(path)) is False:
                raise ValueError(f"Could not add mask folder {path}.")
            return
        source = path.parent if path.is_file() else path
        screen.set_images(str(source))


class AlignDropHandler(DropHandler):
    """Use a dropped image folder (or one tile) as Align & Stitch input."""

    def can_accept(self, path: Path) -> bool:
        return (path.is_dir() and _contains_suffix(path, _IMAGE_SUFFIXES)) or (
            path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)

    def error_message(self, path: Path) -> str:
        return "Align & Stitch needs a folder containing microscopy tiles."

    def apply(self, path: Path, screen) -> None:
        source = path.parent if path.is_file() else path
        screen.apply_settings({"src": str(source)})


class ConvertDropHandler(DropHandler):
    """Use a dropped microscopy container or folder as converter input."""

    def can_accept(self, path: Path) -> bool:
        return path.is_dir() or (
            path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)

    def error_message(self, path: Path) -> str:
        return ("Format Converter accepts a microscopy image/container or a "
                "folder containing ND2, CZI, LIF, OME-TIFF or image files.")

    def apply(self, path: Path, screen) -> None:
        screen.set_source(str(path.parent if path.is_file() else path))


class PlateQueueDropHandler(DropHandler):
    """Queue plate folders that carry spaCR settings snapshots."""

    def accepts_multiple(self) -> bool:
        return True

    def can_accept(self, path: Path) -> bool:
        return (
            path.is_file() and path.suffix.lower() == ".csv"
        ) or (
            path.is_dir() and bool(_settings_files(path))
        )

    def error_message(self, path: Path) -> str:
        return ("Plate Queue accepts a plate folder containing settings/*.csv "
                "snapshots, or a plate-list CSV with an src column.")

    def apply(self, path: Path, screen) -> None:
        if path.is_file():
            from .plate_queue import import_plates_from_csv
            items = import_plates_from_csv(path, base_settings={},
                                           app_key="mask")
            if not items:
                raise ValueError(
                    f"{path.name} contains no plate rows with an src value.")
            for item in items:
                screen.queue().add(item)
            screen._refresh_table()
            screen.queue_size_changed.emit(len(screen.queue()))
            return

        from spacr.utils import load_settings
        added = 0
        skipped: list = []
        snapshots = list(_settings_files(path))
        for settings_path in snapshots:
            try:
                settings = load_settings(
                    str(settings_path), setting_key="Key",
                    setting_value="Value")
            except Exception:
                # The two-column spelling is the one spaCR writes; the
                # single-argument call is the documented default and is what
                # a hand-made snapshot is likely to use. Only the SECOND
                # failure means the file is unreadable.
                try:
                    settings = load_settings(str(settings_path))
                except Exception as exc:
                    LOG.warning("plate queue: skipping %s — its settings "
                                "snapshot could not be read (%s)",
                                settings_path.name, exc)
                    skipped.append(settings_path.name)
                    continue
            if not isinstance(settings, dict):
                LOG.warning("plate queue: skipping %s — its settings "
                            "snapshot parsed to %s, not a settings dict",
                            settings_path.name, type(settings).__name__)
                skipped.append(settings_path.name)
                continue
            settings["src"] = str(path)
            screen.add_item(
                _module_from_settings(settings_path), settings)
            added += 1
        if not added:
            raise ValueError(f"No readable settings snapshots found in {path}.")
        if skipped:
            # A partial drop used to report plain success: one unreadable
            # snapshot among several meant that plate quietly never reached
            # the queue, and the user found out when the run they expected
            # was not in the list.
            _log(screen,
                 f"Queued {added} of {len(snapshots)} settings snapshots "
                 f"from {path.name}. Skipped: {', '.join(skipped)}.")


class BatchDropHandler(DropHandler):
    """Load queue files or add dropped settings snapshots as jobs."""

    def accepts_multiple(self) -> bool:
        return True

    def can_accept(self, path: Path) -> bool:
        if path.is_file():
            return path.suffix.lower() in {".csv", ".json", ".yaml", ".yml"}
        return path.is_dir() and bool(_settings_files(path))

    def error_message(self, path: Path) -> str:
        return ("Batch Runner accepts a saved JSON/YAML queue, a settings CSV, "
                "or a plate folder containing settings snapshots.")

    def apply(self, path: Path, screen) -> None:
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}:
            if not screen.load_queue_from(str(path)):
                raise ValueError(getattr(screen, "last_error", "")
                                 or f"Could not load {path}.")
            return
        candidates = [path] if path.is_file() else _settings_files(path)
        added = 0
        for settings_path in candidates:
            if screen.add_job(
                    module=_module_from_settings(settings_path),
                    settings=str(settings_path)):
                added += 1
        if not added:
            raise ValueError(f"No runnable settings jobs found in {path}.")


class ImageFieldsDropHandler(DropHandler):
    """Image-folder input shared by Model Compare."""

    def can_accept(self, path: Path) -> bool:
        return (path.is_dir() and _contains_suffix(path, _IMAGE_SUFFIXES)) or (
            path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)

    def error_message(self, path: Path) -> str:
        return "Drop a folder containing microscopy fields."

    def apply(self, path: Path, screen) -> None:
        source = path.parent if path.is_file() else path
        if screen.set_source(str(source)) is False:
            raise ValueError(getattr(screen, "last_error", "")
                             or f"Could not load fields from {source}.")


class ModelZooDropHandler(DropHandler):
    """Scan checkpoints, or use image-only folders as benchmark fields."""

    def can_accept(self, path: Path) -> bool:
        return path.is_dir() or (
            path.is_file()
            and (path.name.lower().endswith(_MODEL_SUFFIXES)
                 or path.suffix.lower() in _IMAGE_SUFFIXES)
        )

    def error_message(self, path: Path) -> str:
        return "Drop a model/checkpoint folder or a folder of test fields."

    def apply(self, path: Path, screen) -> None:
        source = path.parent if path.is_file() else path
        # The cheap answers first, inline: a dropped checkpoint, or a
        # checkpoint sitting at the top level of the dropped folder. That is
        # one directory listing which stops at the first hit — and it is what
        # a real model folder looks like, so the common drop stays fully
        # synchronous.
        if ((path.is_file() and path.name.lower().endswith(_MODEL_SUFFIXES))
                or _contains_suffix(source, _MODEL_SUFFIXES)):
            screen.scan(str(source))
            return
        # Nothing up top. Answering "is there one further down?" means
        # walking the entire tree, and it is the NO that costs — ``any()``
        # short-circuits on a hit but a negative answer visits every file.
        # That is exactly the "dropped a plate folder on the wrong screen"
        # case: 100 000 files, a second of dead window. Off the GUI thread it
        # goes, and the branch it decides goes with it.
        _scan_then(
            screen,
            lambda: _contains_suffix(source, _MODEL_SUFFIXES, recursive=True),
            lambda is_model: _apply_model_zoo_source(source, screen, is_model),
        )


def _apply_model_zoo_source(source: Path, screen, is_model: bool) -> None:
    """GUI-thread half of :meth:`ModelZooDropHandler.apply`."""
    if is_model:
        screen.scan(str(source))
        return
    if screen.set_fields_source(str(source)) is not False:
        return
    # ``apply`` returned long ago, so raising here would surface as an
    # unhandled exception in the Qt event loop instead of being caught by
    # ``dnd._on_drop`` — and the user would be told nothing at all. Report it
    # exactly as that handler would have.
    reason = (getattr(screen, "last_error", "")
              or f"Could not load fields from {source}.")
    _report_drop_problem(
        screen, Path(source), f"The drop handler failed: {reason}",
        "Check that the path is readable and that its contents match this "
        "module, then try again.",
    )


class ResultsDatabaseDropHandler(DatabaseDropHandler):
    """Database input for Plate Viewer and Annotator Agreement."""

    def apply(self, path: Path, screen) -> None:
        opener = getattr(screen, "set_database", None)
        if not callable(opener):
            opener = getattr(screen, "open_database", None)
        if not callable(opener):
            raise TypeError("This screen cannot open a database.")
        if opener(str(path)) is False:
            raise ValueError(getattr(screen, "last_error", "")
                             or f"Could not open {path}.")


class TrainingRunsDropHandler(DropHandler):
    """Accept a directory and asynchronously scan it for training runs."""

    def can_accept(self, path: Path) -> bool:
        return path.is_dir()

    def error_message(self, path: Path) -> str:
        return "Training Runs accepts a folder containing model training runs."

    def apply(self, path: Path, screen) -> None:
        if screen.scan(str(path)) is False:
            raise ValueError(getattr(screen, "last_error", "")
                             or f"Could not scan {path}.")


class ReportDropHandler(DropHandler):
    """Accept a completed spaCR run folder and scan its report inputs."""

    def can_accept(self, path: Path) -> bool:
        return path.is_dir()

    def error_message(self, path: Path) -> str:
        return "Report accepts a completed spaCR run folder."

    def apply(self, path: Path, screen) -> None:
        screen.set_source(str(path))
        if screen.scan() is False:
            raise ValueError(getattr(screen, "last_error", "")
                             or f"Could not scan {path}.")


# ---------------------------------------------------------------------------
# Layout-aware drops
#
# "It should be possible to drag-n-drop folders and files into every module,
# and every module should be aware of the spaCR folder structure": drop the
# project on a screen that reads a database and it finds
# ``measurements/measurements.db``; drop it on one that reads a table and it
# offers the tables in that database; drop the database itself and that still
# works.
#
# None of the layout knowledge lives here. :func:`spacr.chaining.resolve_drop`
# answers "where is the X in this project?" by asking the artifact registry
# first -- the same question, through the same call, that auto-chaining asks --
# and falling back to the declared paths in :data:`spacr.ports.PORTS`. Two
# answers to "where is the database" is how a screen and the run it launches
# come to disagree, so there is only one.
#
# What a handler adds is the last step: which of this screen's fields the
# answer goes into.
# ---------------------------------------------------------------------------

def _resolve_for(handler, app_key: str, path: Path):
    """Resolve ``path`` for ``app_key``, memoised for one drop.

    ``can_accept``, ``error_message``, ``suggest_alternatives`` and ``apply``
    are all called for the same path inside one drop, and each of them wants
    the same answer. Resolving four times would list the same directories four
    times while the user is still holding the mouse button down.

    The folder's mtime is part of the key, so the cache lasts exactly as long
    as the answer does: run Measure and drop the same plate again, and the
    database that appeared is found rather than remembered as absent.
    """
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        stamp = 0
    key = (app_key, str(path), stamp)
    cached = getattr(handler, "_last_resolution", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        resolution = _ch.resolve_drop(
            app_key, path, kinds=getattr(handler, "kinds", ()),
            form=getattr(handler, "form", _ch.PATH))
    except Exception:
        LOG.debug("could not resolve %s for %s", path, app_key, exc_info=True)
        return None
    handler._last_resolution = (key, resolution)
    return resolution


def table_names(path: Path) -> List[str]:
    """Return the tables in a SQLite file, or ``[]`` for anything else.

    One ``sqlite_master`` query; the table screens make the same one when they
    load. Kept here so the drop can *ask* which table rather than let
    ``load_path`` take the first one silently.

    Opened read-only, and through a *quoted* URI: a folder with a ``?`` or a
    ``#`` in its name would otherwise have everything after it read as query
    parameters, and the open would fail on a database that is perfectly fine.
    """
    import sqlite3
    from urllib.parse import quote

    if not str(path).lower().endswith(_ch.DB_SUFFIXES):
        return []
    try:
        connection = sqlite3.connect(
            f"file:{quote(str(path))}?mode=ro", uri=True, timeout=30)
    except sqlite3.Error:
        return []
    try:
        return [str(row[0]) for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "ORDER BY name")]
    except sqlite3.DatabaseError:
        return []
    finally:
        connection.close()


class LayoutDropHandler(DropHandler):
    """Drop policy for a screen that names what it wants, not where it is.

    A subclass says two things: the vocabulary terms it consumes
    (:attr:`kinds`, from :data:`spacr.ports.ALL_KINDS`) and what to do with
    the answer (:meth:`deliver`). Everything between — climbing from the
    dropped path to the project root, asking the registry, falling back to the
    declared layout, noticing that the answer is ambiguous — is
    :func:`spacr.chaining.resolve_drop`.

    Ambiguity is routed through the machinery :mod:`spacr.qt.dnd` already has:
    an ambiguous drop reports ``can_accept() is False`` and returns the
    candidates from :meth:`suggest_alternatives`, so the user gets the "did
    you mean…" chooser and :meth:`apply` is called again with their answer.
    A drop that resolves to nothing reports
    :attr:`spacr.chaining.DropResolution.reason`, which is
    :func:`spacr.ports.check_ready`'s own sentence about what is missing.
    """

    #: What this screen consumes, in the shared vocabulary. Empty means "the
    #: project folder itself".
    kinds: tuple = ()
    #: Whether the field wants the artifact (:data:`spacr.chaining.PATH`) or
    #: the project it belongs to (:data:`spacr.chaining.ROOT`).
    form: str = _ch.PATH
    #: Suffixes this screen can be handed directly, bypassing the layout walk.
    suffixes: tuple = ()
    #: What to call the screen in a message.
    label: str = ""

    def __init__(self, app_key: str = "") -> None:
        self.app_key = app_key or self.label or type(self).__name__
        self._last_resolution = None

    # -- the subclass hook -------------------------------------------------
    def deliver(self, screen, value: str, target) -> None:
        """Put ``value`` into the screen. Runs on the GUI thread.

        :param screen: the screen the drop landed on.
        :param value: the resolved path.
        :param target: the :class:`spacr.chaining.DropTarget` it came from,
            or None for a file the user dropped directly.
        """
        raise NotImplementedError

    # -- DropHandler -------------------------------------------------------
    def _direct(self, path: Path) -> bool:
        """True when the dropped file is already the artifact wanted."""
        return bool(path.is_file() and self.suffixes
                    and path.name.lower().endswith(self.suffixes))

    def resolve(self, path: Path):
        """Return the :class:`spacr.chaining.DropResolution` for ``path``."""
        return _resolve_for(self, self.app_key, path)

    def can_accept(self, path: Path) -> bool:
        if self._direct(path):
            return True
        if not path.is_dir():
            return False
        resolution = self.resolve(path)
        return bool(resolution is not None and resolution.ok
                    and not resolution.ambiguous)

    def suggest_alternatives(self, path: Path) -> List[Path]:
        resolution = self.resolve(path)
        if resolution is None:
            return []
        found: List[Path] = []
        for choice in resolution.choices:
            found.extend(Path(option) for option in choice.options)
        return found

    def error_message(self, path: Path) -> str:
        resolution = self.resolve(path)
        if resolution is None:
            return f"{self.app_key} cannot use {path.name!r}."
        return resolution.reason

    def apply(self, path: Path, screen) -> None:
        if self._direct(path):
            self.deliver(screen, str(path), None)
            _log(screen, f"[drop] {self.app_key} ← {path}\n")
            return
        resolution = self.resolve(path)
        if resolution is None or not resolution.targets:
            reason = (resolution.reason if resolution is not None
                      else f"{path} could not be read.")
            _report_drop_problem(
                screen, path, reason,
                f"Drop the folder or file this screen names, or run the step "
                f"that writes it into {getattr(resolution, 'root', path)}.")
            return
        target = resolution.targets[0]
        self.deliver(screen, str(target.location), target)
        _log(screen,
             f"[drop] {self.app_key} ← {target.location}\n"
             f"[drop] resolved {target.kind} in {resolution.root} "
             f"(from the {target.source})\n")


class ProjectFolderDropHandler(LayoutDropHandler):
    """A screen that takes a whole project, wherever inside it you drop.

    ``kinds`` is empty on purpose: there is no port for "the project", and
    inventing one would put it in the module graph. The layout walk still
    happens — dropping ``<plate>/measurements/measurements.db`` on the
    pipeline graph opens ``<plate>``.
    """

    form = _ch.ROOT
    #: The screen method that takes the project root, tried in order.
    setters: tuple = ("load_project", "set_project", "set_source",
                      "add_root", "set_src")

    def can_accept(self, path: Path) -> bool:
        resolution = self.resolve(path)
        return bool(resolution is not None and resolution.ok
                    and not resolution.ambiguous)

    def deliver(self, screen, value: str, target) -> None:
        for name in self.setters:
            setter = getattr(screen, name, None)
            if callable(setter):
                if setter(value) is False:
                    raise ValueError(getattr(screen, "last_error", "")
                                     or f"Could not open {value}.")
                return
        raise TypeError(
            f"{type(screen).__name__} has no way to receive a project folder.")


class DataManagerDropHandler(ProjectFolderDropHandler):
    """Data Manager: set the project, then measure it."""

    def deliver(self, screen, value: str, target) -> None:
        super().deliver(screen, value, target)
        scan = getattr(screen, "scan", None)
        if callable(scan):
            scan()


class ProjectRootsDropHandler(ProjectFolderDropHandler):
    """Project Browser: several folders at once, each becoming a root."""

    setters = ("add_root",)

    def accepts_multiple(self) -> bool:
        return True

    def deliver(self, screen, value: str, target) -> None:
        # ``add_root`` returns False for a root that is already listed, which
        # is not a failure and must not be reported as one: dropping a folder
        # the browser already watches should be a no-op, not an error dialog.
        screen.add_root(value)


class RunHistoryDropHandler(ProjectFolderDropHandler):
    """Run History: select the run a dropped run folder belongs to."""

    setters = ("select_run",)

    def can_accept(self, path: Path) -> bool:
        return path.is_dir() or path.is_file()

    def apply(self, path: Path, screen) -> None:
        folder = path if path.is_dir() else path.parent
        refresh = getattr(screen, "refresh", None)
        if callable(refresh):
            refresh()
        if screen.select_run(str(folder)) is False:
            raise ValueError(
                f"No run named {folder.name!r} is in the history. Drop the "
                "run folder spaCR wrote, or clear the filters above.")
        _log(screen, f"[drop] run_history ← {folder}\n")


class TableDropHandler(LayoutDropHandler):
    """A screen that reads one table: the explorers, the plotters, the gates.

    Drop the project and it finds ``measurements/measurements.db``; drop the
    database and it uses it; drop a CSV and it reads that. When the database
    holds more than one table the table is *asked* rather than taken —
    ``load_path`` picks the first one silently, which is fine for a file
    dialog where the user chose the file and wrong for a drop where they
    chose a folder.
    """

    kinds = (_kinds.MEASUREMENTS_DB,)
    form = _ch.PATH
    suffixes = _ch.DB_SUFFIXES + (".csv", ".tsv", ".parquet", ".txt")

    def deliver(self, screen, value: str, target) -> None:
        table = self._choose_table(screen, value)
        if table is False:                     # the chooser was cancelled
            return
        screen.load_path(value, table or None)

    def _choose_table(self, screen, value: str):
        """Return the table to read, ``""`` for "there is only one", or False.

        False means the user cancelled and nothing should be loaded.
        """
        names = table_names(Path(value))
        if len(names) <= 1:
            return names[0] if names else ""
        picked = _ask_for_one(
            screen, f"{Path(value).name} holds {len(names)} tables.",
            "Which one should be loaded?", names)
        return picked if picked is not None else False


class ScatterTableDropHandler(TableDropHandler):
    """Image Scatter: a path field, a table picker, then the read."""

    def deliver(self, screen, value: str, target) -> None:
        screen._db.setText(value)
        screen.open_source()


class LineageDropHandler(TableDropHandler):
    """Lineage: a database path field and one load."""

    def deliver(self, screen, value: str, target) -> None:
        screen._db.setText(value)
        screen.load()


class CoefficientsDropHandler(LayoutDropHandler):
    """Prediction Profiler: the regression coefficients under ``results/``."""

    kinds = (_kinds.REGRESSION_RESULTS,)
    suffixes = (".csv",)

    def deliver(self, screen, value: str, target) -> None:
        path = Path(value)
        if path.is_dir():
            candidates = [Path(p) for p in (target.paths if target else ())]
            if not candidates:
                candidates = sorted(path.glob("*.csv"))
            if not candidates:
                raise ValueError(f"No coefficient CSV was found in {path}.")
            if len(candidates) > 1:
                picked = _ask_for_one(
                    screen, f"{path.name} holds {len(candidates)} tables.",
                    "Which one holds the coefficients?",
                    [str(c) for c in candidates])
                if picked is None:
                    return
                path = Path(picked)
            else:
                path = candidates[0]
        screen.load_coefficients(str(path))


class ResultsFolderDropHandler(LayoutDropHandler):
    """Hit List: the ``results/`` folder a regression wrote."""

    kinds = (_kinds.REGRESSION_RESULTS,)
    suffixes = ()

    def deliver(self, screen, value: str, target) -> None:
        folder = Path(value)
        screen.load_folder(str(folder if folder.is_dir() else folder.parent))


class LabelMaskDropHandler(LayoutDropHandler):
    """Curate and Napari Bridge: one label mask, from wherever you drop.

    Dropping the project resolves ``masks/``; a folder of masks is not one
    mask, so the file is asked for rather than guessed at.
    """

    kinds = (_kinds.MASKS,)
    suffixes = (".tif", ".tiff", ".png", ".npy")

    def _one_mask(self, screen, value: str, target) -> Optional[str]:
        path = Path(value)
        if path.is_file():
            return str(path)
        candidates = [p for p in (target.paths if target else ())
                      if p.lower().endswith(self.suffixes)]
        if not candidates:
            candidates = [str(p) for p in sorted(path.iterdir())
                          if p.is_file()
                          and p.name.lower().endswith(self.suffixes)]
        if not candidates:
            raise ValueError(f"No label mask was found in {path}.")
        if len(candidates) == 1:
            return candidates[0]
        return _ask_for_one(
            screen, f"{path.name} holds {len(candidates)} masks.",
            "Which one should be opened?", candidates)

    def deliver(self, screen, value: str, target) -> None:
        mask = self._one_mask(screen, value, target)
        if mask is None:
            return
        if hasattr(screen, "set_paths"):
            screen.set_paths(mask=mask)
            return
        screen._mask_edit.setText(mask)
        screen.open_mask()


class LayerStackDropHandler(LabelMaskDropHandler):
    """Layer Viewer: the dropped array, added as an image or as labels.

    A viewer stacks layers, so a multi-drop of an image and its mask lands as
    two layers rather than as the first one.
    """

    suffixes = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy", ".npz")

    def accepts_multiple(self) -> bool:
        return True

    def deliver(self, screen, value: str, target) -> None:
        chosen = self._one_mask(screen, value, target)
        if chosen is None:
            return
        # A file that came out of ``masks/`` is a label array; anything else
        # the user dropped is the image they want to look at.
        as_labels = (target is not None and target.kind == _kinds.MASKS) or (
            "mask" in Path(chosen).parent.name.lower())
        if as_labels:
            screen.add_labels_file(chosen)
        else:
            screen.add_image_file(chosen)


class MethodsSourcesDropHandler(LayoutDropHandler):
    """Methods & Results: fill whichever of its four source fields fits."""

    form = _ch.ROOT

    def can_accept(self, path: Path) -> bool:
        return path.is_dir() or path.is_file()

    def accepts_multiple(self) -> bool:
        return True

    def apply(self, path: Path, screen) -> None:
        fields = getattr(screen, "_fields", {})
        name = path.name.lower()
        if path.is_file() and name.endswith(_MODEL_SUFFIXES):
            key = "model"
            value = str(path)
        elif path.is_dir() and name == "results":
            key, value = "results", str(path)
        else:
            resolution = self.resolve(path)
            root = resolution.root if resolution is not None else str(path)
            key, value = "project", root
        widget = fields.get(key)
        if widget is None:
            raise TypeError("Methods & Results has no field for this drop.")
        widget.setText(value)
        _log(screen, f"[drop] methods_export {key} = {value}\n")


class EvaluationBundleDropHandler(LayoutDropHandler):
    """Classifier Evaluation: the run folder holding the evaluation bundle."""

    kinds = (_kinds.MODEL_WEIGHTS,)
    form = _ch.ROOT
    suffixes = (".json", ".csv")

    def can_accept(self, path: Path) -> bool:
        return path.is_dir() or self._direct(path)

    def deliver(self, screen, value: str, target) -> None:
        screen._source.setText(value)
        screen.scan()

    def apply(self, path: Path, screen) -> None:
        resolution = self.resolve(path)
        value = str(path if path.is_dir() else path.parent)
        if resolution is not None and resolution.targets:
            value = str(resolution.targets[0].value)
        self.deliver(screen, value, None)
        _log(screen, f"[drop] classifier_evaluation ← {value}\n")


class SubmissionSettingsDropHandler(LayoutDropHandler):
    """Distributed Jobs: a settings snapshot to submit, or the plate with one."""

    suffixes = (".csv", ".json", ".yaml", ".yml")

    def can_accept(self, path: Path) -> bool:
        return self._direct(path) or (
            path.is_dir() and bool(_settings_files(path)))

    def error_message(self, path: Path) -> str:
        return ("Distributed Jobs needs a settings snapshot — a .csv or "
                ".json file, or a plate folder with settings/*.csv in it.")

    def apply(self, path: Path, screen) -> None:
        chosen = path
        if path.is_dir():
            snapshots = _settings_files(path)
            if not snapshots:
                raise ValueError(f"No settings snapshot was found in {path}.")
            if len(snapshots) > 1:
                picked = _ask_for_one(
                    screen, f"{path.name} holds {len(snapshots)} snapshots.",
                    "Which one should be submitted?",
                    [str(s) for s in snapshots])
                if picked is None:
                    return
                chosen = Path(picked)
            else:
                chosen = snapshots[0]
        screen._settings_path.setText(str(chosen))
        module = getattr(screen, "_module", None)
        if module is not None and hasattr(module, "setCurrentText"):
            module.setCurrentText(_module_from_settings(chosen))
        _log(screen, f"[drop] distributed_jobs settings = {chosen}\n")


def _ask_for_one(screen, headline: str, question: str,
                 options: Sequence[str]) -> Optional[str]:
    """Ask which of ``options`` was meant. ``None`` when nobody answered.

    A drop that silently takes the first of several is the failure this whole
    change exists to avoid, so there is no "sensible default" branch here.
    Headless (no QApplication, or a screen that is not a widget) is the one
    exception, and it declines rather than choosing.
    """
    from .dnd import choose_one_dialog

    try:
        return choose_one_dialog(screen, headline, question, list(options))
    except Exception:
        LOG.debug("could not ask which of %d options was meant", len(options),
                  exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_HANDLERS = {
    "mask":            MaskDropHandler,
    "measure":         MeasureDropHandler,
    "external_masks":  ExternalMasksDropHandler,
    "annotate":        AnnotateDropHandler,
    "classify":        ClassifyDropHandler,
    "make_masks":      MakeMasksDropHandler,
    "map_barcodes":    MapBarcodesDropHandler,
    "umap":            MeasurementsDropHandler,
    "ml_analyze":      MeasurementsDropHandler,
    "regression":      MeasurementsDropHandler,
    "recruitment":     MeasurementsDropHandler,
    "activation":      MeasurementsDropHandler,
    "invasion":        MeasurementsDropHandler,
    "analyze_plaques": MakeMasksDropHandler,      # plaque images
    "train_cellpose":  MakeMasksDropHandler,      # image + mask pairs
    "cellpose_masks":  MakeMasksDropHandler,
    "cellpose_all":    MakeMasksDropHandler,
    "db_browser":      DatabaseDropHandler,
    "foreign":         ForeignProjectDropHandler,
    "align":           AlignDropHandler,
    "convert":         ConvertDropHandler,
    "queue":           PlateQueueDropHandler,
    "batch":           BatchDropHandler,
    "model_compare":   ImageFieldsDropHandler,
    "model_zoo":       ModelZooDropHandler,
    "plate_view":      ResultsDatabaseDropHandler,
    "agreement":       ResultsDatabaseDropHandler,
    "train_compare":   TrainingRunsDropHandler,
    "report":          ReportDropHandler,

    # -- the layout-aware screens ------------------------------------------
    # One table out of a measurement database, or a CSV. All nine expose the
    # same ``load_path(path, table=None)``, which is why one handler covers
    # them: the difference between these screens is what they draw, not what
    # they read.
    "graph_builder":    TableDropHandler,
    "trellis":          TableDropHandler,
    "gate_editor":      TableDropHandler,
    "feature_explorer": TableDropHandler,
    "outliers":         TableDropHandler,
    "control_chart":    TableDropHandler,
    "dose_response":    TableDropHandler,
    "pca":              TableDropHandler,
    "tabulate":         TableDropHandler,
    "image_scatter":    ScatterTableDropHandler,
    "lineage":          LineageDropHandler,

    # A whole project, from anywhere inside it.
    "pipeline_graph":   ProjectFolderDropHandler,
    "run_compare":      ProjectFolderDropHandler,
    "qc_dashboard":     ProjectFolderDropHandler,
    "data_manager":     DataManagerDropHandler,
    "project_browser":  ProjectRootsDropHandler,
    "run_history":      RunHistoryDropHandler,
    "methods_export":   MethodsSourcesDropHandler,

    # One artifact out of the layout.
    "profiler":         CoefficientsDropHandler,
    "hit_list":         ResultsFolderDropHandler,
    "curate":           LabelMaskDropHandler,
    "napari_bridge":    LabelMaskDropHandler,
    "layer_viewer":     LayerStackDropHandler,
    "classifier_evaluation": EvaluationBundleDropHandler,
    "distributed_jobs": SubmissionSettingsDropHandler,
    "explain_cv":       ExplainCvInputsDropHandler,
    "investigate_hit":  InvestigateHitInputsDropHandler,
    "parameter_sweep":  SweepInputsDropHandler,
}

#: Screens where a drop is genuinely meaningless, recorded rather than left
#: to look like an oversight. None of them reads a path: Experiment Design
#: and Power compute a layout and a sample size from numbers typed into the
#: screen, and Feature Dictionary is a searchable glossary that ships with
#: spaCR. A drop target here would accept a folder and do nothing with it,
#: which is worse than no target at all.
NO_DROP_TARGET: Dict[str, str] = {
    "experiment_design": "designs a plate layout from typed numbers; it "
                         "reads no file",
    "power": "computes a sample size from typed numbers; it reads no file",
    "feature_dict": "is a glossary of spaCR's feature names, not a reader "
                    "of data",
}


def get_handler(app_key: str) -> DropHandler:
    """Return a fresh DropHandler for ``app_key``.

    Falls back to :class:`SourceDropHandler` so every conventional AppScreen
    can at least receive its source folder.
    """
    cls = _HANDLERS.get(app_key)
    if cls is not None and issubclass(cls, LayoutDropHandler):
        # The layout-aware handlers resolve against the module they are
        # installed on, so they need to be told which one that is.
        return cls(app_key)
    if cls is None:
        try:
            from spacr.plugins import get_app, load_object
            plugin_app = get_app(app_key)
            if plugin_app is not None and plugin_app.drop_handler:
                candidate = load_object(plugin_app.drop_handler)
                if not isinstance(candidate, type) or not issubclass(candidate, DropHandler):
                    raise TypeError(
                        f"{plugin_app.drop_handler} is not a DropHandler subclass"
                    )
                cls = candidate
        except Exception as exc:
            try:
                from spacr.plugins import record_diagnostic
                record_diagnostic(
                    app_key, "Could not load plugin drag-and-drop handler", exc
                )
            except Exception:
                pass
    if cls is not None and issubclass(cls, LayoutDropHandler):
        return cls(app_key)
    cls = cls or SourceDropHandler
    return cls()
    # NOTE: an ``accepts_multiple`` override used to sit here, after the
    # return, indented as if it were a method of this *function*. It was
    # unreachable in both readings, so nothing it claimed was ever true.
