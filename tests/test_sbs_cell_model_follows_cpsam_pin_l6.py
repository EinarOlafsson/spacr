"""The item-1025 SBS cells load the Cellpose-SAM weights ``ops_cellpose_model`` pins (975).

Found at the L6 merge: the decode worker's mask settings left out
``ops_cellpose_model`` and the model cache ignored it, so a run set to
``cpsam_v1`` built its SBS cells with ``cpsam_v2``.
"""
from spacr import ops_engine


def _options(**extra):
    settings = {"ops_sbs_cell_method": "cellpose", **extra}
    return ops_engine._sbs_cell_options(settings)


def test_the_worker_settings_carry_the_pinned_weights():
    assert _options(ops_cellpose_model="cpsam_v1")["mask_settings"]["ops_cellpose_model"] == "cpsam_v1"


def test_each_pin_builds_its_own_model(monkeypatch):
    built = []

    def fake(settings, gpu, key):
        built.append(ops_engine.ops_setting(settings, "ops_cellpose_model"))
        return object()

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake)
    monkeypatch.setattr(ops_engine, "_SBS_CELL_MODELS", {})
    v1 = ops_engine._sbs_cell_model(_options(ops_cellpose_model="cpsam_v1"), False)
    v2 = ops_engine._sbs_cell_model(_options(ops_cellpose_model="cpsam_v2"), False)
    again = ops_engine._sbs_cell_model(_options(ops_cellpose_model="cpsam_v1"), False)
    assert built == ["cpsam_v1", "cpsam_v2"]
    assert v1 is again and v1 is not v2


def test_an_unset_pin_follows_the_default():
    options = _options()
    assert "ops_cellpose_model" not in options["mask_settings"]
    assert ops_engine.ops_setting(options["mask_settings"], "ops_cellpose_model") == "cpsam_v2"
