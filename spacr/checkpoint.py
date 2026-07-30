"""Atomic, signature-checked checkpoints for long spaCR workflows.

The workflow modules decide what a safe unit is: a field for conversion and
image processing, a trial (plus its completed adaptive round) for UMAP, and a
job/plate for Batch.  This module only supplies the small persistence contract
they share:

* checkpoint JSON is written to a temporary sibling and atomically replaced;
* a resume is refused when the workflow signature differs;
* completed units carry JSON payloads and optional NumPy artifacts;
* every write records the boundary and update time, making a checkpoint
  inspectable without importing the workflow that produced it.

It deliberately imports only the standard library.  Mask and Measure consult
resume state before loading torch/Cellpose, so checkpoint infrastructure must
never make those imports heavier.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .errors import ConfigurationError

__all__ = [
    "CHECKPOINT_VERSION",
    "CheckpointError",
    "CheckpointMismatch",
    "CheckpointStore",
    "fingerprint",
    "json_safe",
]


CHECKPOINT_VERSION = 1


class CheckpointError(ConfigurationError):
    """Base class for a checkpoint that cannot be read or written safely."""


class CheckpointMismatch(CheckpointError):
    """Raised when resume settings/input identity differ from the checkpoint."""


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Return ``value`` in a deterministic JSON-compatible form.

    Paths, sets, tuples, NumPy scalars and other scalar-like objects are
    normalised without importing NumPy.  Unknown objects fall back to their
    string representation; workflow signatures should still prefer explicit
    primitives for scientifically meaningful settings.

    :param value: object to normalise.
    :returns: JSON-compatible value with mapping keys sorted as strings.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [json_safe(item) for item in value]
        return sorted(converted, key=lambda item: repr(item))
    # NumPy scalar types expose item(); using it keeps this module NumPy-free.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def fingerprint(value: Any) -> str:
    """Return a SHA-256 digest of deterministic JSON for ``value``.

    :param value: settings, input identity, or another JSON-like structure.
    :returns: lowercase hexadecimal SHA-256 digest.
    """
    encoded = json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one JSON document to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(json_safe(payload), stream, indent=2, sort_keys=True,
                      ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class CheckpointStore:
    """One atomic checkpoint document plus optional array artifacts.

    :param path: JSON checkpoint path.
    :param workflow: stable workflow identifier, e.g. ``"umap_search"``.
    :param signature: digest or JSON-like identity of inputs and material
        settings. Non-digest values are passed through :func:`fingerprint`.
    :param boundary: human-readable unit such as ``"field"`` or ``"trial"``.
    :param resume: load compatible state when True; otherwise start a fresh
        document at the same path.
    :raises CheckpointMismatch: when ``resume`` is requested for a checkpoint
        from a different workflow/signature.
    :raises CheckpointError: when the document is corrupt or inaccessible.
    """

    def __init__(
        self,
        path: os.PathLike | str,
        *,
        workflow: str,
        signature: Any,
        boundary: str,
        resume: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.workflow = str(workflow)
        self.signature = (
            str(signature)
            if isinstance(signature, str) and len(signature) == 64
            else fingerprint(signature)
        )
        self.boundary = str(boundary)
        self.resumed = False
        if resume and self.path.is_file():
            self._document = self._read()
            self._validate()
            self.resumed = True
        else:
            now = _utc_now()
            self._document: Dict[str, Any] = {
                "version": CHECKPOINT_VERSION,
                "workflow": self.workflow,
                "signature": self.signature,
                "boundary": self.boundary,
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "meta": {},
                "completed": {},
            }
            self.flush()

    @property
    def artifact_dir(self) -> Path:
        """Directory holding large artifacts referenced by the JSON."""
        return self.path.parent / f"{self.path.name}.d"

    @property
    def completed(self) -> Dict[str, Any]:
        """Copy of completed-unit payloads keyed by unit id."""
        value = self._document.get("completed", {})
        return dict(value) if isinstance(value, Mapping) else {}

    @property
    def meta(self) -> Dict[str, Any]:
        """Copy of workflow-specific state."""
        value = self._document.get("meta", {})
        return dict(value) if isinstance(value, Mapping) else {}

    @property
    def status(self) -> str:
        """Current checkpoint status."""
        return str(self._document.get("status", "running"))

    def _read(self) -> Dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                f"Checkpoint {self.path} could not be read: {exc}. Keep it "
                "for diagnosis, then start without Resume to create a fresh "
                "checkpoint.") from exc
        if not isinstance(payload, dict):
            raise CheckpointError(
                f"Checkpoint {self.path} is not a JSON object.")
        return payload

    def _validate(self) -> None:
        version = self._document.get("version")
        if version != CHECKPOINT_VERSION:
            raise CheckpointMismatch(
                f"Checkpoint {self.path} uses version {version!r}, but this "
                f"spaCR build supports version {CHECKPOINT_VERSION}. Start a "
                "fresh run rather than mixing checkpoint formats.")
        actual_workflow = self._document.get("workflow")
        if actual_workflow != self.workflow:
            raise CheckpointMismatch(
                f"Checkpoint {self.path} belongs to {actual_workflow!r}, not "
                f"{self.workflow!r}. Choose the matching checkpoint or turn "
                "Resume off.")
        actual_signature = self._document.get("signature")
        if actual_signature != self.signature:
            raise CheckpointMismatch(
                f"Checkpoint {self.path} does not match the current inputs or "
                "material settings. spaCR will not combine units produced by "
                "different configurations; restore the original settings or "
                "start without Resume.")
        if not isinstance(self._document.get("completed", {}), dict):
            raise CheckpointError(
                f"Checkpoint {self.path} has an invalid completed-unit table.")
        if not isinstance(self._document.get("meta", {}), dict):
            raise CheckpointError(
                f"Checkpoint {self.path} has invalid workflow metadata.")

    def flush(self) -> None:
        """Atomically persist the current document."""
        self._document["updated_at"] = _utc_now()
        try:
            _atomic_json(self.path, self._document)
        except OSError as exc:
            raise CheckpointError(
                f"Checkpoint {self.path} could not be written: {exc}. The "
                "workflow stopped rather than pretending it can be resumed."
            ) from exc

    def mark(
        self,
        unit: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record one completed safe unit and atomically persist it.

        :param unit: stable unit id.
        :param payload: JSON-like result metadata for the unit.
        :param meta: workflow state to merge into the document metadata.
        """
        completed = self._document.setdefault("completed", {})
        completed[str(unit)] = json_safe(dict(payload or {}))
        if meta:
            self._document.setdefault("meta", {}).update(json_safe(dict(meta)))
        self._document["status"] = "running"
        self.flush()

    def update(
        self,
        *,
        meta: Optional[Mapping[str, Any]] = None,
        status: Optional[str] = None,
    ) -> None:
        """Persist workflow metadata or status without completing a unit."""
        if meta:
            self._document.setdefault("meta", {}).update(json_safe(dict(meta)))
        if status is not None:
            self._document["status"] = str(status)
        self.flush()

    def finish(self, *, meta: Optional[Mapping[str, Any]] = None) -> None:
        """Mark the workflow complete while retaining its inspectable state."""
        self.update(meta=meta, status="complete")

    def artifact_path(self, unit: str, suffix: str = ".npy") -> Path:
        """Return a collision-resistant artifact path for ``unit``.

        The directory is created lazily. ``unit`` itself is not used as a
        filename; its digest prevents paths/settings from becoming filesystem
        syntax.
        """
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        ending = suffix if str(suffix).startswith(".") else f".{suffix}"
        return self.artifact_dir / f"{fingerprint(str(unit))}{ending}"
