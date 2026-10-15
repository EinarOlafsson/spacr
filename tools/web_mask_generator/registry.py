"""Trusted web-model identities derived from spaCR's existing catalogue.

No network, downloads or directory scans. Visitor model keys are resolved only
through an operator-supplied mapping, with exact bundled-checksum verification.
"""
from __future__ import annotations

from pathlib import Path

from spacr import model_zoo
from spacr.qt.i18n import tr
from tools.build_model_cards import families, published_models

from .results import _sha256


def model_families():
    """Public model identities/defaults, preserving desktop alpha/version rules.

    Detection families advertise boxes rather than fabricated segmentation
    outputs. Only the plaque family currently has a web inference adapter.
    Scientific score/provenance text comes from the existing catalogue, without
    reinterpreting it as a new benchmark or cross-dataset ranking.
    """
    records = [record for record in published_models()
               if record.get("kind") in ("cellpose", "detector")]
    result = []
    for family in families(records):
        versions = []
        for record in family["versions"]:
            entry = model_zoo._entry_from_mapping(record, source="bundled")
            versions.append({
                "key": entry.key, "name": entry.name, "kind": entry.kind,
                "version": entry.key.rsplit("_v", 1)[-1],
                "sha256": entry.sha256, "source_url": entry.uri,
                "repo_id": record.get("repo_id"), "repo_type": record.get("repo_type"),
                "status": family["status"][entry.key],
                "trained_on": entry.trained_on, "metrics": dict(entry.metrics),
                "display_name": record.get("display_name", entry.name),
                "architecture": record.get("architecture", ""),
                "versus_stock": record.get("versus_stock", ""),
                "notes": list(entry.notes),
                "output_kind": "boxes" if entry.kind == "detector" else "labels",
                "inference_adapter": "plaque" if family["stem"] == "toxoplasma_plaque" else None,
            })
        result.append({"family": family["stem"], "default": family["default"]["key"], "versions": versions})
    return result


def resolve_trusted_model(family, provisioned_paths, *, model_key=None):
    """Resolve one public key to operator-owned local weights; never substitute.

    Returned checkpoint paths are private server-side values and must not be
    sent in the public registry. An unavailable default raises rather than
    falling back silently to an older installed version.
    """
    matches = [entry for entry in model_families() if entry["family"] == family]
    if not matches:
        raise ValueError(tr("Unknown web model class: {family}", family=family))
    selected_family = matches[0]
    key = selected_family["default"] if model_key is None else model_key
    versions = [entry for entry in selected_family["versions"] if entry["key"] == key]
    if not versions:
        raise ValueError(tr("The selected model does not belong to this model class."))
    if key not in provisioned_paths:
        raise ValueError(tr("The selected model is not provisioned on this service."))
    metadata = versions[0]
    checkpoint = Path(provisioned_paths[key])
    if not checkpoint.is_file() or _sha256(checkpoint) != metadata["sha256"]:
        raise ValueError(tr("The provisioned checkpoint is missing or does not match the published SHA-256."))
    return {"checkpoint": checkpoint, "metadata": metadata}
