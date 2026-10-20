"""Framework-independent N698 model service boundary and private job lifetime."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import MappingProxyType

from spacr.qt.i18n import tr

from .inputs import DEFAULT_MAX_FILE_BYTES, describe_upload
from .plaque import _load_model, run_plaque_download
from .detector import run_well_download
from .masks import OBJECT_ROLES, run_mask_download
from .registry import model_families, resolve_trusted_model


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


class ModelService:
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
        """Validate configured identities and warm at most one segmentation model.

        Every operator-provisioned key and checksum is validated before any
        model is constructed. The current plaque default is warmed when
        provisioned; otherwise the first supported configured segmentation key
        in lexical order is warmed. No missing default is substituted for an
        inference request, and no model is downloaded here.

        :returns: the warmed key, or ``None`` for an empty or detector-only
            configuration. Detectors retain their request-scoped desktop
            loader. This bounded warm-up does not certify ZeroGPU placement.
        """
        catalogue = model_families()
        known = {version["key"]: family["family"]
                 for family in catalogue for version in family["versions"]}
        resolved_models = {}
        for key in self.models:
            if key not in known:
                raise ValueError(tr("Unknown provisioned model key: {key}", key=key))
            family = known[key]
            resolved_models[key] = (family, resolve_trusted_model(
                family, self.models, model_key=key))
        plaque_default = next((family["default"] for family in catalogue
                               if family["family"] == "toxoplasma_plaque"), None)
        candidates = sorted(key for key, (family, _) in resolved_models.items()
                            if family == "toxoplasma_plaque" or family in OBJECT_ROLES)
        if not candidates:
            return None
        key = plaque_default if plaque_default in candidates else candidates[0]
        family, resolved = resolved_models[key]
        metadata = resolved["metadata"]
        if family == "toxoplasma_plaque":
            _load_model(resolved["checkpoint"], metadata["sha256"])
        else:
            from .masks import _load_mask_model
            _load_mask_model(resolved["checkpoint"], metadata["sha256"],
                             OBJECT_ROLES[family])
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
        family = request.get("family")
        if family not in {"toxoplasma_plaque", "toxoplasma_well_detector", *OBJECT_ROLES}:
            raise ValueError(tr("Unknown web model class."))
        selection = request.get("selection", {})
        selection_keys = {"series", "indices", "channels", "page", "pdf_dpi"}
        if not isinstance(selection, dict) or set(selection) - selection_keys:
            raise ValueError(tr("Unsupported image selection fields were supplied."))
        resolved = resolve_trusted_model(family, self.models, model_key=request.get("model_key"))
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
            runner = run_plaque_download if family == "toxoplasma_plaque" else run_well_download if family == "toxoplasma_well_detector" else run_mask_download
            family_argument = {"family": family} if family in OBJECT_ROLES else {}
            output = runner(
                stage / "result.zip", input_file=copied,
                checkpoint=resolved["checkpoint"], expected_sha256=metadata["sha256"],
                model_id=metadata["key"], model_version=metadata["version"],
                effective_settings=request.get("settings"), selection=selection, **family_argument)
            yield {"result_zip": output, "model": metadata}


PlaqueService = ModelService
