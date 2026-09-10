"""Regression tests for concise user-facing module descriptions."""

from spacr.settings import descriptions


def test_legacy_module_descriptions_do_not_reintroduce_promotional_boilerplate():
    """Module help must remain specific, current and free of legacy AI prose."""

    forbidden = (
        "Key Features:",
        "Comprehensive Analysis",
        "Works seamlessly",
        "Leverage Cellpose",
        "streamlined workflows",
        "state-of-the-art",
    )
    combined = "\n".join(descriptions.values())
    assert all(marker not in combined for marker in forbidden)


def test_legacy_module_descriptions_cover_the_registered_legacy_keys():
    """Removing a legacy description must be an explicit compatibility change."""

    legacy_keys = {
        "mask",
        "measure",
        "classify",
        "umap",
        "train_cellpose",
        "ml_analyze",
        "cellpose_masks",
        "cellpose_all",
        "map_barcodes",
        "regression",
        "activation",
        "analyze_plaques",
        "recruitment",
    }
    # Self-registering modules may append their own valid descriptions during
    # test collection.  This contract owns the legacy compatibility rows, not
    # the open-ended dynamic registry.
    assert legacy_keys <= set(descriptions)
