"""N684: the audit receipt regenerates Automatic's cost table, and spinn is timed moving."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "audit_animation_performance", ROOT / "tools" / "audit_animation_performance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cost_table_keeps_only_rows_drawn_entirely_on_the_gpu():
    report = {"rows": [
        {"theme": "a", "size": [10, 10], "density": 1.0,
         "cpu": {"median_ms": 2.0}, "gpu": {"median_ms": 1.0, "gpu_frames": 4, "frames": 4}},
        {"theme": "b", "size": [10, 10], "density": 1.0,
         "cpu": {"median_ms": 2.0}, "gpu": {"median_ms": 1.0, "gpu_frames": 0, "frames": 4}},
        {"theme": "c", "size": [10, 10], "density": 1.0, "cpu": {"median_ms": 2.0}},
    ]}
    namespace = {}
    exec(_tool().cost_table(report), namespace)
    assert namespace["_GRAPHICS_COST"] == {"a": ((100, 1.0, 2.0, 1.0),)}


def test_spinn_is_audited_under_a_moving_pointer():
    tool = _tool()
    assert "data_art_tissue_facets" in tool.POINTER_THEMES
    assert set(tool.GPU_COVERAGE) >= {"blobs", "drift", "aurora", "data_art_fungal_growth",
                                      "data_art_tissue_facets"}


def test_automatic_cost_table_is_the_newest_workstation_receipt():
    import json

    from spacr.qt.widgets import ambient

    receipts = sorted((ROOT / "features" / "data").glob("684_animation_gpu_audit_*/receipt.json"))
    receipt = receipts[-1]
    namespace = {}
    exec(_tool().cost_table(json.loads(receipt.read_text())), namespace)
    assert ambient._GRAPHICS_COST == namespace["_GRAPHICS_COST"]
