"""Reversible, auditable exclusion of merged fields from measurement.

Measurement discovers work by enumerating ``merged/*.npy``.  Quarantine
therefore needs no exclusion database: moving one array to the sibling
``merged_quarantined/`` folder removes it from the next run, while leaving
every mask stack untouched.  A JSON ledger beside the moved array records
who made that decision, when, and which segmentation-QC flags prompted it.

The functions in this module deliberately have no Qt dependency.  They are
usable from the field browser, a notebook, or a headless audit script and can
be tested without constructing an application.
"""
from __future__ import annotations

import datetime as _datetime
import errno
import getpass
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

_PathValue = Union[os.PathLike, str]

__all__ = [
    "QUARANTINE_DIRNAME",
    "QuarantineError",
    "is_quarantined",
    "list_quarantined",
    "quarantine_dir_for",
    "quarantine_field",
    "quarantine_record_path",
    "resolve_field_path",
    "restore_field",
]


QUARANTINE_DIRNAME = "merged_quarantined"
_RECORD_SUFFIX = ".quarantine.json"


class QuarantineError(RuntimeError):
    """A field could not be quarantined or restored without losing data."""


def _field_stem(field: _PathValue) -> str:
    """Validate and return the filename stem used by a merged field.

    ``FieldQC.field`` is a stem, not a path.  Rejecting separators here keeps
    a malformed or hand-edited scorecard from moving a file outside the two
    directories this module owns.
    """
    text = os.fspath(field).strip()
    if not text or text in {".", ".."} or "\x00" in text:
        raise ValueError("field must be a non-empty merged-array name")
    if "/" in text or "\\" in text:
        raise ValueError("field must be a name, not a path")
    if text.lower().endswith(".npy"):
        text = text[:-4]
    if not text or text in {".", ".."}:
        raise ValueError("field must name a merged .npy array")
    return text


def _merged_dir(path: _PathValue) -> Path:
    """Resolve and return ``path`` after requiring a ``merged`` basename."""
    folder = Path(path).expanduser().resolve()
    if folder.name != "merged":
        raise ValueError(
            f"expected a plate's 'merged' folder, got {os.fspath(path)!r}")
    return folder


def _quarantine_dir(path: _PathValue) -> Path:
    """Resolve and return ``path`` after requiring the quarantine basename."""
    folder = Path(path).expanduser().resolve()
    if folder.name != QUARANTINE_DIRNAME:
        raise ValueError(
            f"expected a '{QUARANTINE_DIRNAME}' folder, got "
            f"{os.fspath(path)!r}")
    return folder


def quarantine_dir_for(merged_dir: _PathValue) -> Path:
    """Return ``<plate>/merged_quarantined`` for ``<plate>/merged``.

    :param merged_dir: validated plate ``merged`` directory.
    """
    merged = _merged_dir(merged_dir)
    return merged.parent / QUARANTINE_DIRNAME


def quarantine_record_path(
    quarantine_dir: _PathValue,
    field: _PathValue,
) -> Path:
    """Return the audit sidecar path for one quarantined field.

    :param quarantine_dir: plate ``merged_quarantined`` directory.
    :param field: merged-field stem, with an optional ``.npy`` suffix.
    """
    folder = _quarantine_dir(quarantine_dir)
    return folder / f"{_field_stem(field)}.npy{_RECORD_SUFFIX}"


def _field_path(folder: Path, field: _PathValue) -> Path:
    """Return the validated ``.npy`` path for ``field`` beneath ``folder``."""
    return folder / f"{_field_stem(field)}.npy"


def _now() -> str:
    """Return the current UTC time as a seconds-precision ISO 8601 string."""
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(
        timespec="seconds")


def _who(value: Optional[str]) -> str:
    """Return an explicit actor, the OS account, or ``"unknown"``."""
    if value is not None and str(value).strip():
        return str(value).strip()
    try:
        name = getpass.getuser().strip()
    except Exception:  # a platform account lookup failure is not an error here
        name = ""
    return name or "unknown"


def _read_record(path: Path) -> Dict[str, Any]:
    """Read a ledger object, retaining an explanation for malformed content."""
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        # A damaged old ledger must not make restoration impossible.  Keep
        # the fact that it was damaged in the replacement audit record.
        return {"prior_record_error": f"{type(exc).__name__}: {exc}"}
    return value if isinstance(value, dict) else {
        "prior_record_error": "the previous sidecar was not a JSON object"}


def _write_record(path: Path, record: Dict[str, Any]) -> None:
    """Atomically replace ``path`` with ``record`` in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise


def _move_without_overwrite(source: Path, destination: Path) -> None:
    """Move a regular file without replacing an existing destination."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite existing field {destination}")
    if source.is_symlink():
        raise QuarantineError(f"refusing to move symlink {source}")
    if not source.is_file():
        raise FileNotFoundError(source)
    # A plain POSIX rename silently REPLACES a destination created between
    # the check above and the syscall.  Link-then-unlink is the stdlib's
    # atomic no-replace move for sibling regular files: link fails when the
    # name already exists and both names address the same bytes until the
    # source is removed.  The bounded copy fallback covers filesystems that
    # do not support hard links while keeping O_EXCL's no-overwrite promise.
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as exc:
        unsupported = {
            errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.EPERM),
        }
        if exc.errno not in unsupported:
            raise
        descriptor = -1
        try:
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                source.stat().st_mode & 0o777,
            )
            with source.open("rb") as reader, os.fdopen(
                    descriptor, "wb") as writer:
                descriptor = -1
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            shutil.copystat(source, destination, follow_symlinks=False)
            os.unlink(source)
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(destination)
            except OSError:
                pass
            raise
        return
    try:
        os.unlink(source)
    except Exception:
        try:
            os.unlink(destination)
        except OSError as rollback:
            raise QuarantineError(
                f"could not remove source {source} or rollback destination "
                f"{destination}: {rollback}") from rollback
        raise


def quarantine_field(
    merged_dir: _PathValue,
    field: _PathValue,
    *,
    flags: Iterable[str] = (),
    who: Optional[str] = None,
) -> Path:
    """Move one merged array out of measurement and write its audit record.

    :param merged_dir: the plate's ``merged`` directory.
    :param field: a :class:`spacr.seg_qc.FieldQC` field stem (``.npy`` is
        accepted too).
    :param flags: the QC flags that motivated this decision.
    :param who: actor recorded in the ledger; defaults to the OS account.
    :returns: the new ``merged_quarantined/<field>.npy`` path.
    :raises FileNotFoundError: when the merged array is already gone.
    :raises FileExistsError: rather than overwriting an existing quarantine.

    If the sidecar cannot be written, the array is moved back before the
    exception is raised.  An unaudited quarantine is never reported as a
    successful operation.
    """
    merged = _merged_dir(merged_dir)
    stem = _field_stem(field)
    source = _field_path(merged, stem)
    quarantine = quarantine_dir_for(merged)
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = _field_path(quarantine, stem)
    sidecar = quarantine_record_path(quarantine, stem)
    actor = _who(who)
    timestamp = _now()
    clean_flags = sorted({str(flag).strip() for flag in flags
                          if str(flag).strip()})

    _move_without_overwrite(source, destination)
    previous = _read_record(sidecar)
    history = list(previous.get("events") or [])
    history.append({
        "action": "quarantined",
        "at": timestamp,
        "by": actor,
        "qc_flags": clean_flags,
    })
    record: Dict[str, Any] = {
        "version": 1,
        "field": stem,
        "source": str(source),
        "quarantined_path": str(destination),
        "quarantined_at": timestamp,
        "quarantined_by": actor,
        "qc_flags": clean_flags,
        "events": history,
    }
    if previous.get("prior_record_error"):
        record["prior_record_error"] = previous["prior_record_error"]
    try:
        _write_record(sidecar, record)
    except Exception as exc:
        try:
            _move_without_overwrite(destination, source)
        except Exception as rollback:
            raise QuarantineError(
                f"could not write {sidecar} and could not restore {source}: "
                f"{rollback}") from exc
        raise QuarantineError(
            f"could not write quarantine record {sidecar}; field restored") \
            from exc
    return destination


def restore_field(
    quarantine_dir: _PathValue,
    field: _PathValue,
    *,
    who: Optional[str] = None,
) -> Path:
    """Move one quarantined array back to its sibling ``merged`` folder.

    :param quarantine_dir: plate ``merged_quarantined`` directory.
    :param field: field stem, with an optional ``.npy`` suffix, to restore.

    The sidecar remains in ``merged_quarantined`` as the plate's audit trail
    and gains a restoration event.  As with quarantine, a ledger-write
    failure rolls the file move back.
    """
    quarantine = _quarantine_dir(quarantine_dir)
    stem = _field_stem(field)
    source = _field_path(quarantine, stem)
    merged = quarantine.parent / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    destination = _field_path(merged, stem)
    sidecar = quarantine_record_path(quarantine, stem)
    actor = _who(who)
    timestamp = _now()

    _move_without_overwrite(source, destination)
    previous = _read_record(sidecar)
    history = list(previous.get("events") or [])
    history.append({"action": "restored", "at": timestamp, "by": actor})
    record = dict(previous)
    record.update({
        "version": 1,
        "field": stem,
        "source": str(destination),
        "quarantined_path": str(source),
        "restored_at": timestamp,
        "restored_by": actor,
        "events": history,
    })
    try:
        _write_record(sidecar, record)
    except Exception as exc:
        try:
            _move_without_overwrite(destination, source)
        except Exception as rollback:
            raise QuarantineError(
                f"could not update {sidecar} and could not return {source}: "
                f"{rollback}") from exc
        raise QuarantineError(
            f"could not update quarantine record {sidecar}; field remains "
            "quarantined") from exc
    return destination


def is_quarantined(
    merged_dir: _PathValue,
    field: _PathValue,
) -> bool:
    """Return whether the sibling quarantine currently holds ``field``.

    :param merged_dir: plate ``merged`` directory whose quarantine is checked.
    :param field: field stem, with an optional ``.npy`` suffix, to locate.
    """
    quarantine = quarantine_dir_for(merged_dir)
    path = _field_path(quarantine, field)
    return path.is_file() and not path.is_symlink()


def list_quarantined(merged_dir: _PathValue) -> List[str]:
    """Return sorted field stems currently excluded from ``merged/*.npy``.

    :param merged_dir: plate ``merged`` directory whose quarantine is listed.
    """
    quarantine = quarantine_dir_for(merged_dir)
    try:
        entries = list(quarantine.iterdir())
    except OSError:
        return []
    return sorted(
        path.name[:-4] for path in entries
        if path.is_file() and not path.is_symlink()
        and path.name.lower().endswith(".npy")
    )


def resolve_field_path(
    merged_dir: _PathValue,
    field: _PathValue,
) -> Optional[Path]:
    """Locate a field in ``merged`` or its quarantine, active copy first.

    :param merged_dir: plate ``merged`` directory to search first.
    :param field: field stem, with an optional ``.npy`` suffix, to locate.
    """
    merged = _merged_dir(merged_dir)
    active = _field_path(merged, field)
    if active.is_file() and not active.is_symlink():
        return active
    quarantined = _field_path(quarantine_dir_for(merged), field)
    if quarantined.is_file() and not quarantined.is_symlink():
        return quarantined
    return None
