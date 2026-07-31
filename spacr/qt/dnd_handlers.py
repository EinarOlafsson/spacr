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
| regression      | ditto + ``scores.csv``                                |
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
    # Make the detection actionable: build a preview of how each image would
    # be named and open the editable metadata table so the user can accept or
    # correct it, writing a filename_map.csv the pipeline consumes.
    try:
        from . import ingest_preview as ip
        rows = ip.plan_folder_extraction(path)
        if rows:
            _log(screen, f"[drop] folder-structure plan — "
                         f"{ip.summarize_rows(rows)}\n")
            _open_metadata_table(rows, path, screen)
    except Exception as e:
        _log(screen, f"[drop] folder-structure preview unavailable: {e}\n")


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
        if path.is_file():
            path = path.parent
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
    """Accept a database, its measurements folder, or its plate folder."""

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

    def apply(self, path: Path, screen) -> None:
        if path.is_file():
            path = path.parent
        if path.name == "measurements" and (path / "measurements.db").is_file():
            path = path.parent
        _set_src_on(screen, str(path))
        _log(screen, f"[drop] src = {path}\n")


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
        if path.is_file():
            path = path.parent
        if path.name == "measurements" and (path / "measurements.db").is_file():
            path = path.parent
        if not _set_src_on(screen, str(path)):
            raise TypeError("This module has no source field to receive the drop.")
        _log(screen, f"[drop] src = {path}\n")


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
        for settings_path in _settings_files(path):
            try:
                settings = load_settings(
                    str(settings_path), setting_key="Key",
                    setting_value="Value")
            except Exception:
                try:
                    settings = load_settings(str(settings_path))
                except Exception:
                    continue
            if not isinstance(settings, dict):
                continue
            settings["src"] = str(path)
            screen.add_item(
                _module_from_settings(settings_path), settings)
            added += 1
        if not added:
            raise ValueError(f"No readable settings snapshots found in {path}.")


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
        is_model = (
            path.is_file() and path.name.lower().endswith(_MODEL_SUFFIXES)
        ) or _contains_suffix(source, _MODEL_SUFFIXES, recursive=True)
        if is_model:
            screen.scan(str(source))
        elif screen.set_fields_source(str(source)) is False:
            raise ValueError(getattr(screen, "last_error", "")
                             or f"Could not load fields from {source}.")


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
}


def get_handler(app_key: str) -> DropHandler:
    """Return a fresh DropHandler for ``app_key``.

    Falls back to :class:`SourceDropHandler` so every conventional AppScreen
    can at least receive its source folder.
    """
    cls = _HANDLERS.get(app_key)
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
    cls = cls or SourceDropHandler
    return cls()
    def accepts_multiple(self) -> bool:
        return True
