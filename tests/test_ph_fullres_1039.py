"""Item 1039: OPS segmentation at full resolution by default, and an audit.

Maintainer decision 2026-11-25 (Einar, final): phenotype nuclei and cells
at factor 1 with no diameter, so nothing is resized. ``ops_ph_downsample``
is the cell factor (1) and the new ``ops_ph_nucleus_downsample`` the
nucleus factor (``'same'``, the default, follows it). Settings schema 978
gives older files 2 and ``'same'``. A diameter is in full-resolution pixels
at every factor and is divided once, by its own object's factor, in
``_ops_diameter``.

Every OPS segmentation path is checked by recording what reaches the
model: the phenotype nuclei and cells (and the resident-batch probe), the
sequencing nuclei of the stitched windows, the item-1025 SBS cells and the
Make Masks kernel. By default each sees an image of the field's shape.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pytest.importorskip("pandas")

from spacr import ops_compose, ops_engine
from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS, ops_defaults
from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                            _fold_older_defaults)
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call
from tests.test_ops_objects_resume_958 import _CountingModel, _run as _run_sbs, stitched  # noqa: F401
from tests.test_ph_segment_flatfield_1035 import SETTINGS, SIZE, _run, models  # noqa: F401


class _ShapeModel:
    """A stand-in Cellpose model: records each input shape, labels nothing."""

    def __init__(self):
        self.shapes = []
        self.diameters = []

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        """Record the image shape and ``diameter``; return empty labels.

        The installed cellpose eval signature is written out, and
        ``channel_axis`` is checked against it (tests/conftest.py); the OPS
        nucleus call leaves the axis to Cellpose, as the other OPS doubles
        allow.
        """
        check_cellpose_eval_call(image, channel_axis, require_channel_axis=False)
        self.diameters.append(diameter)
        image = np.asarray(image)
        self.shapes.append(image.shape)
        height, width = image.shape[:2]
        labels = np.zeros((height, width), np.int32)
        flows = [np.zeros((height, width, 3), np.uint8),
                 np.zeros((2, height, width), np.float32),
                 np.zeros((height, width), np.float32)]
        return labels, flows, None


def _stack(height=101, width=87):
    """A two-channel phenotype field of odd size, so a downscale would show."""
    rng = np.random.default_rng(0)
    return rng.integers(100, 4000, (height, width, 2)).astype(np.float32)


def _settings(**extra):
    return ops_defaults({"ops_ph_channels": "DAPI,AF750", "ops_flatfield": "none", **extra})


def _segment(**extra):
    """Segment one odd-sized field; return the two recording models."""
    stack = _stack()
    nucleus, cell = _ShapeModel(), _ShapeModel()
    nuclei, cells, _counts = ops_engine._ph_segment_field(
        stack, ["DAPI", "AF750"], (nucleus, cell), _settings(**extra))
    assert nuclei.shape == cells.shape == stack.shape[:2]
    return nucleus, cell


def test_both_default_to_full_resolution_and_the_nucleus_follows_the_cell_factor():
    assert OPS_DEFAULTS["ops_ph_downsample"] == 1
    assert OPS_DEFAULTS["ops_ph_nucleus_downsample"] == "same"
    assert OPS_DEFAULTS["ops_ph_nucleus_diameter"] is None
    assert OPS_DEFAULTS["ops_ph_cell_diameter"] is None
    factors = ops_engine._ph_downsample_factors
    assert factors({}) == factors(ops_defaults({})) == {"nucleus": 1, "cell": 1}
    assert factors({"ops_ph_downsample": 2}) == {"nucleus": 2, "cell": 2}
    assert factors({"ops_ph_downsample": 1, "ops_ph_nucleus_downsample": 2}) == {
        "nucleus": 2, "cell": 1}
    assert factors({"ops_ph_nucleus_downsample": "2"}) == {"nucleus": 2, "cell": 1}
    assert factors({"ops_ph_downsample": 2, "ops_ph_nucleus_downsample": " Same "}) == {
        "nucleus": 2, "cell": 2}
    assert factors({"ops_ph_downsample": 0}) == {"nucleus": 1, "cell": 1}
    with pytest.raises(ValueError):
        factors({"ops_ph_nucleus_downsample": "half"})
    with pytest.raises(ValueError):
        ops_engine._ph_downsample_factor({}, "pathogen")


def test_files_before_978_replay_2_for_both():
    assert SETTINGS_SCHEMA_VERSION >= 978
    assert 977 not in DEFAULT_CHANGES
    assert DEFAULT_CHANGES[978] == {"ops_ph_downsample": 2,
                                    "ops_ph_nucleus_downsample": "same"}
    known = set(OPS_DEFAULTS)
    for version in (0, 958, 976):
        old_file = {"plate": "P1", "ops_ph_downsample": 1}
        applied = _fold_older_defaults(old_file, version, known=known)
        assert applied["ops_ph_nucleus_downsample"] == ("same", 978)
        assert ops_engine._ph_downsample_factors(ops_defaults(old_file)) == {
            "nucleus": 1, "cell": 1}
        unset = {"plate": "P1"}
        applied = _fold_older_defaults(unset, version, known=known)
        assert applied["ops_ph_downsample"] == (2, 978)
        assert ops_engine._ph_downsample_factors(ops_defaults(unset)) == {
            "nucleus": 2, "cell": 2}
    new_file = {"plate": "P1"}
    applied = _fold_older_defaults(new_file, 978, known=known)
    assert "ops_ph_downsample" not in applied
    assert "ops_ph_nucleus_downsample" not in applied
    assert ops_engine._ph_downsample_factors(ops_defaults(new_file)) == {
        "nucleus": 1, "cell": 1}
    explicit = {"plate": "P1", "ops_ph_nucleus_downsample": 1}
    _fold_older_defaults(explicit, 958, known=known)
    assert ops_engine._ph_downsample_factors(ops_defaults(explicit))["nucleus"] == 1


def test_the_default_feeds_both_models_the_whole_field_without_a_diameter():
    nucleus, cell = _segment()
    assert nucleus.shapes == [(101, 87)] and nucleus.diameters == [None]
    assert cell.shapes == [(101, 87, 2)] and cell.diameters == [None]


def test_factor_2_still_replays_the_old_block_average():
    nucleus, cell = _segment(ops_ph_downsample=2)
    assert nucleus.shapes == [(51, 44)]
    assert cell.shapes == [(51, 44, 2)]


def test_a_full_resolution_diameter_reaches_cellpose_divided_once_by_its_factor():
    """Arm (c): factor 1 + diameter d hands Cellpose d; factor 2 + d hands it d / 2."""
    nucleus, cell = _segment(ops_ph_downsample=1, ops_ph_nucleus_diameter=34,
                             ops_ph_cell_diameter=80)
    assert nucleus.diameters == [34.0] and cell.diameters == [80.0]
    nucleus, cell = _segment(ops_ph_downsample=2, ops_ph_nucleus_diameter=34,
                             ops_ph_cell_diameter=80)
    assert nucleus.diameters == [17.0] and cell.diameters == [40.0]
    nucleus, cell = _segment(ops_ph_downsample=1, ops_ph_nucleus_downsample=2,
                             ops_ph_nucleus_diameter=34, ops_ph_cell_diameter=80)
    assert nucleus.shapes == [(51, 44)] and nucleus.diameters == [17.0]
    assert cell.shapes == [(101, 87, 2)] and cell.diameters == [80.0]
    nucleus, cell = _segment(ops_ph_downsample=1)
    assert nucleus.diameters == [None] and cell.diameters == [None]


def test_mixed_factors_give_the_cell_model_dna_at_the_cell_factor():
    stack = _stack()
    dna, body, _extra, factors = ops_engine._ph_segmentation_inputs(
        stack, ["DAPI", "AF750"], _settings(ops_ph_downsample=1, ops_ph_nucleus_downsample=2))
    assert factors == {"nucleus": 2, "cell": 1}
    assert dna.shape == (51, 44)
    assert np.array_equal(body[..., 1], stack[..., 0])
    assert np.array_equal(body[..., 0], stack[..., 1])


def test_the_resident_batch_probe_sizes_the_cell_input_at_its_factor():
    stack = _stack()
    probe = _ShapeModel()
    model, body, extra = ops_engine._ph_probe_input(
        stack, channels=["DAPI", "AF750"],
        settings=_settings(ops_ph_downsample=1),
        models=[probe, probe], gpu=False, object_models={})
    assert model is probe and body.shape == stack.shape[:2] + (2,)
    assert extra == {"channel_axis": -1}


def test_the_ops_run_segments_full_fields_and_marks_the_factors(tmp_path, models, monkeypatch):
    """Through _ph_fields at the default: Cellpose sees SIZE x SIZE, the marker names both factors."""
    fed = []
    real = ops_engine._ops_mask_eval

    def record(model, image, settings, **keywords):
        fed.append(np.asarray(image).shape)
        return real(model, image, settings, **keywords)

    monkeypatch.setattr(ops_engine, "_ops_mask_eval", record)
    settings = {k: v for k, v in SETTINGS.items() if k != "ops_ph_downsample"}
    _report, masks, marker = _run(tmp_path, settings)
    assert fed and {shape[:2] for shape in fed} == {(SIZE, SIZE)}
    assert marker["nucleus_downsample"] == 1 and marker["cell_downsample"] == 1
    assert masks[0]["nuclei"].shape == (SIZE, SIZE)


def test_masks_of_another_factor_are_never_mixed(tmp_path, models):
    _r, _m, marker = _run(tmp_path, {**SETTINGS, "ops_ph_downsample": 2})
    assert "nucleus_downsample" not in marker and "cell_downsample" not in marker
    with pytest.raises(ValueError, match="parts made with"):
        _run(tmp_path, {**SETTINGS, "ops_ph_downsample": 2, "ops_ph_nucleus_downsample": 1})


def test_sbs_nuclei_see_each_composed_window_whole(stitched, tmp_path, monkeypatch):
    composed = []
    real = ops_compose.compose_window

    def spy(*args, **kwargs):
        image, coverage = real(*args, **kwargs)
        if (coverage > 0).any():
            composed.append(np.asarray(image).shape)
        return image, coverage

    monkeypatch.setattr(ops_compose, "compose_window", spy)

    class Recording(_CountingModel):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def eval(self, image, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True, rescale=None,
                 diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
                 do_3D=False, anisotropy=None, flow3D_smooth=0,
                 stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
                 niter=None, augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            """Record the window shape, then label it as the counting model."""
            check_cellpose_eval_call(image, channel_axis, require_channel_axis=False)
            self.shapes.append(np.asarray(image).shape)
            return super().eval(image, batch_size=batch_size, diameter=diameter,
                                min_size=min_size, normalize=normalize)

    model = Recording()
    _run_sbs(stitched, tmp_path, monkeypatch, model=model)
    assert model.shapes and sorted(model.shapes) == sorted(composed)


def test_sbs_cells_see_the_whole_field(monkeypatch):
    seen = []

    def fake_eval(model, image, settings, **kwargs):
        seen.append(np.asarray(image).shape)
        height, width = np.asarray(image).shape[:2]
        return (np.zeros((height, width), np.int32),
                [None, np.zeros((2, height, width), np.float32),
                 np.zeros((height, width), np.float32)], None)

    monkeypatch.setattr(ops_engine, "_ops_mask_eval", fake_eval)
    rng = np.random.default_rng(1)
    stack = rng.integers(0, 500, (3, 4, 75, 61)).astype(np.float32)
    seeds = np.zeros((75, 61), np.int32)
    seeds[30:36, 20:26] = 1
    cells = ops_engine._sbs_segment_cells(
        stack, seeds, {"method": "cellpose", "foreground": None, "mask_settings": {}},
        model=object(), nuclear=stack[0, 0])
    assert seen == [(75, 61, 2)]
    assert cells.shape == (75, 61)


def test_the_make_masks_kernel_passes_the_image_unchanged():
    from spacr.object import _mask_eval

    model = _ShapeModel()
    images = [np.zeros((123, 97, 3), np.float32), np.zeros((123, 97, 3), np.float32)]
    _mask_eval(model, images, site="mask_test", settings={}, batch_size=4, channel_axis=-1)
    assert model.shapes == [(2, 123, 97, 3)]


def test_the_tooltips_state_the_defaults():
    cell = OPS_TOOLTIPS["ops_ph_downsample"]
    nucleus = OPS_TOOLTIPS["ops_ph_nucleus_downsample"]
    assert cell.endswith("Default 1. Alpha.") and "ops_ph_nucleus_downsample" in cell
    assert nucleus.endswith("Default 'same'. Alpha.") and "full-resolution pixels" in nucleus
    assert max(len(cell), len(nucleus)) <= 600
