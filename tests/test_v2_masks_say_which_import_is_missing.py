"""Item 288: v2 mask streaming's refusals and defaults, made deterministic.

``stream_masks_from_stack`` needs two imports before it segments anything:
``spacr.object`` (which routes a model setting to its backend) and
Cellpose itself. Which one failed decides the sentence the user reads, and
which of the two a test reached used to depend on whether an earlier test
had already imported ``spacr.object``. Both are pinned here on purpose.

Also pinned: a plate whose every field the saved image-quality policy
rejects returns no stacks rather than segmenting nothing, and the backend
settings' default diameter follows the pass's own diameter when no
magnification is set.
"""
from __future__ import annotations

import builtins
import sys

import pytest

from spacr import pipeline_v2 as PV
from tests.test_coverage_fill_pipeline_v2 import _make_plate


@pytest.fixture
def stacks(tmp_path):
    plate = _make_plate(tmp_path)
    mapper = PV.FilenameMapper.discover(plate, metadata_type="cellvoyager")
    return PV.stream_originals_to_stack(plate, mapper, channels=(0, 1, 2))


def test_without_the_model_router_the_error_names_it(stacks, monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.object", None)
    with pytest.raises(RuntimeError) as excinfo:
        PV.stream_masks_from_stack(stacks, model_name="cyto")
    assert "spacr.object" in str(excinfo.value)


def test_without_cellpose_the_error_names_cellpose(stacks, monkeypatch):
    import spacr.object  # noqa: F401  (the router is importable)

    real = builtins.__import__

    def block(name, *args, **kwargs):
        if name == "cellpose" or name.startswith("cellpose."):
            raise ImportError("no cellpose")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(RuntimeError) as excinfo:
        PV.stream_masks_from_stack(stacks, model_name="cyto")
    assert str(excinfo.value) == "cellpose is required for v2 mask streaming"
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_a_plate_whose_every_field_is_rejected_returns_no_stacks(
        tmp_path, monkeypatch):
    import spacr.image_quality as image_quality

    plate = _make_plate(tmp_path)
    monkeypatch.setattr(image_quality, "screen_fields",
                        lambda src, settings, paths, *a:
                        [p.name for p in paths])
    out = PV.run_v2(plate, channels=(0, 1, 2), model_name="cyto",
                    metadata_type="cellvoyager")
    assert out["stacks"] == []
    assert out["dst"] == plate / "merged"


@pytest.mark.parametrize("diameter, expected", [(40, 40.0), (None, 30.0)])
def test_without_magnification_the_default_diameter_is_the_passes_own(
        diameter, expected):
    merged = PV._backend_mask_settings(
        {"cell_diameter": 17}, "cell", diameter=diameter, flow_threshold=0.4,
        cellprob_threshold=0.0)
    assert merged["_default_diameter"] == expected
    assert merged["cell_diameter"] == (diameter if diameter else 17)
