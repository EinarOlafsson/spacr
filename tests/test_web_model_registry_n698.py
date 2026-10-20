"""Current catalogue/default binding and private trusted checkpoint resolution."""
import copy
import hashlib

import pytest

from spacr import model_zoo
from tools.web_mask_generator import registry
from tools.build_model_cards import published_models


def test_required_model_classes_and_current_desktop_defaults_are_preserved():
    families = {entry["family"]: entry for entry in registry.model_families()}
    assert set(families) == {"toxoplasma_pv", "toxoplasma_plaque", "toxoplasma_well_detector", "toxoplasma_from_cellmask", "live_cell", "nuclei_from_cellmask", "cell_from_hoechst", "toxoplasma_from_hoechst"}
    assert families["toxoplasma_plaque"]["default"] == "toxoplasma_plaque_v5"
    assert families["toxoplasma_pv"]["default"] == "toxoplasma_pv_v4"
    assert families["toxoplasma_well_detector"]["default"] == "toxoplasma_well_detector_v3"
    for family in families.values():
        for entry in family["versions"]:
            source = next(record for record in published_models() if record["key"] == entry["key"])
            assert entry["sha256"] == source["sha256"]
            assert entry["metrics"] == source.get("metrics", {})
            assert entry["repo_type"] == source["repo_type"]
            assert "checkpoint" not in entry
            assert entry["output_kind"] == ("boxes" if source["kind"] == "detector" else "labels")
            assert entry["inference_adapter"] == ("plaque" if family["family"] == "toxoplasma_plaque" else "well_detector" if source["kind"] == "detector" else "masks")


def test_alternatives_never_silently_replace_current_default():
    family = next(item for item in registry.model_families() if item["family"] == "toxoplasma_pv")
    assert family["versions"][0]["key"] == "toxoplasma_pv_v5"
    assert family["versions"][0]["status"] == "alternative"
    assert family["default"] == "toxoplasma_pv_v4"


@pytest.fixture
def provisioned(tmp_path, monkeypatch):
    checkpoint = tmp_path / "trusted-model"
    checkpoint.write_bytes(b"controlled fake weights for identity only")
    records = copy.deepcopy(model_zoo.BUNDLED_REMOTE_MODELS)
    for entry in records:
        if entry["key"] == "toxoplasma_plaque_v5":
            entry["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS", records)
    return checkpoint, {"toxoplasma_plaque_v5": checkpoint}


def test_only_operator_mapping_and_exact_model_checksum_can_resolve(provisioned):
    checkpoint, paths = provisioned
    resolved = registry.resolve_trusted_model("toxoplasma_plaque", paths)
    assert resolved["checkpoint"] == checkpoint
    assert resolved["metadata"]["key"] == "toxoplasma_plaque_v5"
    assert registry.resolve_trusted_model("toxoplasma_plaque", paths, model_key="toxoplasma_plaque_v5") == resolved


@pytest.mark.parametrize("family,key", [("unknown", None), ("toxoplasma_plaque", "toxoplasma_pv_v4"), ("toxoplasma_plaque", "../../checkpoint"), ("toxoplasma_plaque", "toxoplasma_plaque_v3")])
def test_unknown_mismatched_and_unprovisioned_keys_refuse(provisioned, family, key):
    _, paths = provisioned
    with pytest.raises(ValueError):
        registry.resolve_trusted_model(family, paths, model_key=key)


def test_unavailable_default_never_falls_back_to_old_installed_model(provisioned):
    checkpoint, _ = provisioned
    with pytest.raises(ValueError, match="not provisioned"):
        registry.resolve_trusted_model("toxoplasma_plaque", {"toxoplasma_plaque_v3": checkpoint})


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_replaced_provisioned_weights_refuse(provisioned, missing):
    checkpoint, paths = provisioned
    if missing:
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b"replacement does not match published identity")
    with pytest.raises(ValueError, match="SHA-256"):
        registry.resolve_trusted_model("toxoplasma_plaque", paths)
