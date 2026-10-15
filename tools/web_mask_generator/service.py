"""Framework-independent N698 plaque service boundary and private job lifetime."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import MappingProxyType

from spacr.qt.i18n import tr

from .inputs import DEFAULT_MAX_FILE_BYTES, describe_upload
from .plaque import _load_model, run_plaque_download
from .registry import resolve_trusted_model


def _copy_upload(source, target, limit):
    remaining = limit
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        while True:
            chunk = incoming.read(min(1024 * 1024, remaining + 1))
            if not chunk:
                return
            if len(chunk) > remaining:
                raise ValueError(tr("The upload exceeds the configured file size limit."))
            outgoing.write(chunk)
            remaining -= len(chunk)


class PlaqueService:
    """Keep checkpoint paths and resource limits outside visitor requests.

    A yielded ZIP exists only for the lifetime of ``job``. The HTTP/UI layer
    must serve it within that lifetime or copy it into its own explicitly
    disclosed and bounded download cache before exiting the context.
    """

    def __init__(self, provisioned_models, *, job_parent=None):
        self.models = MappingProxyType({key: Path(path).resolve() for key, path in provisioned_models.items()})
        self.job_parent = Path(job_parent) if job_parent is not None else None
        if self.job_parent is not None:
            self.job_parent.mkdir(parents=True, exist_ok=True)

    def preload_default(self):
        """Explicit operator startup warm-up; never called by importing this module.

        HF ZeroGPU recommends placing models during startup. Digest-keyed
        caching lets the later GPU callback reuse those exact verified bytes.
        """
        resolved = resolve_trusted_model("toxoplasma_plaque", self.models)
        metadata = resolved["metadata"]
        _load_model(resolved["checkpoint"], metadata["sha256"])
        return metadata["key"]

    @contextmanager
    def job(self, upload, request):
        """Validate a visitor request, run the trusted adapter, and remove staging.

        Inputs are never retained here after success, failure or consumer
        exceptions. Model paths, size limits, download folders and backend
        callbacks cannot be supplied as visitor fields.
        """
        allowed = {"family", "model_key", "settings", "selection", "consent_upload"}
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError(tr("Unsupported request fields were supplied."))
        if request.get("consent_upload") is not True:
            raise ValueError(tr("Confirm that the selected image may be uploaded for processing."))
        if request.get("family") != "toxoplasma_plaque":
            raise ValueError(tr("This service currently implements the plaque model class only."))
        selection = request.get("selection", {})
        selection_keys = {"series", "indices", "channels", "page", "pdf_dpi"}
        if not isinstance(selection, dict) or set(selection) - selection_keys:
            raise ValueError(tr("Unsupported image selection fields were supplied."))
        resolved = resolve_trusted_model("toxoplasma_plaque", self.models, model_key=request.get("model_key"))
        metadata = resolved["metadata"]
        upload = Path(upload)
        describe_upload(upload, max_file_bytes=DEFAULT_MAX_FILE_BYTES)
        with tempfile.TemporaryDirectory(prefix="spacr-web-job-", dir=self.job_parent) as directory:
            stage = Path(directory)
            copied = stage / "input" / upload.name
            copied.parent.mkdir()
            _copy_upload(upload, copied, DEFAULT_MAX_FILE_BYTES)
            if copied.stat().st_size > DEFAULT_MAX_FILE_BYTES:
                raise ValueError(tr("The upload exceeds the configured file size limit."))
            output = run_plaque_download(
                stage / "result.zip", input_file=copied,
                checkpoint=resolved["checkpoint"], expected_sha256=metadata["sha256"],
                model_id=metadata["key"], model_version=metadata["version"],
                effective_settings=request.get("settings"), selection=selection)
            yield {"result_zip": output, "model": metadata}
