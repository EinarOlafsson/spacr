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
| annotate        | folder with ``measurements/measurements.db``          |
| classify        | folder with ``data/`` or ``measurements/``            |
| make_masks      | folder with images + optional masks/                  |
| map_barcodes    | folder with FASTQ; also a raw .fastq.gz drop          |
| umap            | folder with ``measurements/measurements.db``          |
| ml_analyze      | ditto                                                 |
| regression      | ditto + ``scores.csv``                                |
| recruitment     | folder with per-well recruitment CSVs                 |
| activation      | folder with saved activation maps or the CV model dir |
| analyze_plaques | folder with plaque images                             |
| train_cellpose  | folder with image+mask pairs                          |
| cellpose_masks  | folder with images                                    |
| cellpose_all    | ditto                                                 |
+-----------------+-------------------------------------------------------+

Every handler falls back to CSV settings-import via :mod:`spacr.qt.dnd`
so users can also drop a settings CSV on any screen to load it.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .dnd import (
    DropHandler, find_image_folders_nearby, has_images_in,
    sample_image_names,
)


# ---------------------------------------------------------------------------
# Shared setter — every AppScreen exposes the src widget through
# _settings_model._widgets["src"]; AnnotateScreen / MakeMasksScreen
# have their own _open_source / _open_folder methods.
# ---------------------------------------------------------------------------

def _set_src_on(screen, path: str) -> bool:
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
            w = screen._settings_model._widgets.get("src")
            if w is not None and hasattr(w, "setText"):
                w.setText(path)
                return True
        except Exception:
            pass
    return False


def _log(screen, msg: str) -> None:
    if hasattr(screen, "_console"):
        try:
            screen._console.append_stdout(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Mask — the star handler with regex-preview canvas
# ---------------------------------------------------------------------------

class MaskDropHandler(DropHandler):
    """Accept a folder of raw microscopy images and preview its filename
    regex parse. Multi-drop is supported."""

    def accepts_multiple(self) -> bool:
        return True

    def can_accept(self, path: Path) -> bool:
        return path.is_dir() and has_images_in(path)

    def suggest_alternatives(self, path: Path) -> List[Path]:
        if path.is_dir():
            return find_image_folders_nearby(path)
        return []

    def error_message(self, path: Path) -> str:
        return ("The mask module needs a folder of microscopy images "
                "(.tif / .png / .czi / .nd2 / .lif) at the top level.")

    def apply(self, path: Path, screen) -> None:
        _set_src_on(screen, str(path))
        _log(screen, f"[drop] mask src = {path}\n")
        # Fire the console-based regex report asynchronously so the
        # UI doesn't stall while it reads image filenames + auto-
        # detects the regex.
        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(50, lambda: _report_regex_on_mask(path, screen))
        except Exception:
            pass


def _report_regex_on_mask(path: Path, screen) -> None:
    """Sample filenames, apply / auto-detect the metadata regex, and
    write a tabular report into the AppScreen's Console.

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
    from . import regex_detect as rd
    from . import multi_format as mf

    _log(screen, "\n")

    # ── Single-file dataset path ──────────────────────────────────
    if path.is_file():
        desc = mf.describe_file(path)
        if desc is not None:
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
                 f"filename_map.csv linking each generated file back to it. "
                 f"Review the extracted metadata in the run output; re-run with "
                 f"a custom regex if anything is off.\n")
            return
        _log(screen, f"[drop] dropped file {path.name} — unrecognised "
                     f"single-file dataset format.\n")
        return

    # ── Folder path ───────────────────────────────────────────────
    imgs = sample_image_names(path, n=20)
    if not imgs:
        _log(screen, "[drop] no images found in the top level of "
                     f"{path.name} — nothing to preview.\n")
        return
    filenames = [p.name for p in imgs]

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
         f"[drop] {len(imgs)} of {_count_images(path)} total sampled "
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
        _push_regex_to_screen(pattern, screen)


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
    alternative to a filename regex (folder_metadata is otherwise unwired)."""
    try:
        from . import folder_metadata as fm
        template = fm.detect_folder_metadata(str(path))
    except Exception:
        template = None
    if template is None:
        return
    labels = getattr(template, "depth_labels", None)
    if not labels:
        return
    _log(screen,
         "\n[drop] folder-structure alternative — detected metadata from the "
         "directory layout:\n"
         f"       path depth → {' / '.join(str(l) for l in labels)}\n"
         "       If your images are organised by folder (e.g. plate/well/"
         "field) rather than by filename, this can be used instead of a "
         "filename regex.\n")


def _open_regex_editor(filenames: list, initial: str, screen) -> None:
    try:
        from .regex_editor import RegexEditorDialog
    except Exception:
        return
    try:
        dlg = RegexEditorDialog(filenames, initial_regex=initial,
                                 multichannel=True, parent=screen)
        if dlg.exec() == dlg.Accepted and dlg.regex:
            _push_regex_to_screen(dlg.regex, screen)
            _log(screen, f"[drop] saved custom regex: {dlg.regex}\n")
    except Exception as e:
        _log(screen, f"[drop] regex editor failed: {e}\n")


def _push_regex_to_screen(pattern: Optional[str], screen) -> None:
    if not pattern:
        return
    try:
        w = screen._settings_model._widgets.get("custom_regex")
        if w is not None and hasattr(w, "setText"):
            w.setText(pattern)
    except Exception:
        pass


def _count_images(path: Path) -> int:
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    return sum(1 for c in path.iterdir()
                if c.is_file() and c.suffix.lower() in exts)


# ---------------------------------------------------------------------------
# Measure — must be `merged` or contain merged/
# ---------------------------------------------------------------------------

class MeasureDropHandler(DropHandler):
    """Accept the ``merged`` folder produced by the mask module, or a
    parent folder that contains one."""

    def can_accept(self, path: Path) -> bool:
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
        # Normalise: if user dropped the parent, drill into merged/
        if path.name != "merged" and (path / "merged").is_dir():
            path = path / "merged"
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
        # Drop-db: use its containing plate folder as src
        if path.is_file() and path.suffix.lower() == ".db":
            path = path.parent.parent
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
               or (path / "data").is_dir()

    def error_message(self, path: Path) -> str:
        return ("Classify needs a plate folder with either "
                "measurements/measurements.db or a data/ subfolder of "
                "cropped PNGs.")

    def apply(self, path: Path, screen) -> None:
        _set_src_on(screen, str(path))
        _log(screen, f"[drop] classify src = {path}\n")


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
    """Accept a plate folder containing a measurements DB."""

    def can_accept(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        return (path / "measurements" / "measurements.db").is_file()

    def error_message(self, path: Path) -> str:
        return ("This module needs a plate folder with "
                "measurements/measurements.db.")

    def apply(self, path: Path, screen) -> None:
        _set_src_on(screen, str(path))
        _log(screen, f"[drop] src = {path}\n")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_HANDLERS = {
    "mask":            MaskDropHandler,
    "measure":         MeasureDropHandler,
    "annotate":        AnnotateDropHandler,
    "classify":        ClassifyDropHandler,
    "make_masks":      MakeMasksDropHandler,
    "map_barcodes":    MapBarcodesDropHandler,
    "umap":            MeasurementsDropHandler,
    "ml_analyze":      MeasurementsDropHandler,
    "regression":      MeasurementsDropHandler,
    "recruitment":     MeasurementsDropHandler,
    "activation":      MeasurementsDropHandler,
    "analyze_plaques": MakeMasksDropHandler,      # plaque images
    "train_cellpose":  MakeMasksDropHandler,      # image + mask pairs
    "cellpose_masks":  MakeMasksDropHandler,
    "cellpose_all":    MakeMasksDropHandler,
}


def get_handler(app_key: str) -> DropHandler:
    """Return a fresh DropHandler for ``app_key``.

    Falls back to :class:`MeasurementsDropHandler` for unknown apps.
    """
    cls = _HANDLERS.get(app_key, MeasurementsDropHandler)
    return cls()
