"""418: V2 filters saved objects using their own original-channel means."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spacr import pipeline_v2 as PV
from tests.test_pipeline_v2_illumination import _CaptureModel, _Session


@pytest.fixture(autouse=True)
def capture_cellpose(monkeypatch):
    """Reuse the existing Cellpose double with its strict eval signature."""
    from spacr import accelerator

    _CaptureModel.received = []
    monkeypatch.setattr("cellpose.models.CellposeModel", _CaptureModel)
    monkeypatch.setattr(accelerator, "_CACHED", accelerator._CPU)


def _raw_field(shape, object_mean):
    """Channel two owns the object; channel zero would reject every field."""
    rows, cols = np.indices(shape)
    raw = np.stack((1000 + rows + cols,
                    2000 + rows * 2 + cols,
                    1 + rows + cols), axis=-1).astype(np.uint16)
    raw[1:3, 1:3, 2] = object_mean
    return raw


def _write_stacks(root, raws):
    """Persist fields with the same channel metadata the streaming API uses."""
    merged = root / "merged"
    merged.mkdir(parents=True)
    names = ["nucleus", "pathogen", "cell"]
    (merged / "channel_order.json").write_text(json.dumps({
        "image_channels": names,
        "mask_channels": [],
    }))
    stacks = []
    for index, raw in enumerate(raws):
        field_id = f"A01_F{index + 1:03d}"
        path = merged / f"stack_{field_id}.npy"
        np.save(path, raw)
        stacks.append(PV.StackFile(field_id, path, raw.shape, names.copy()))
    return stacks


def _settings(minimum=0, maximum=0):
    """Disable other object operations while keeping real normalization."""
    return {
        "cell_min_intensity": minimum,
        "cell_max_intensity": maximum,
        "cell_min_area": 0,
        "cell_max_area": 0,
        "cell_perimeter_fraction": 0,
        "cell_remove_border_objects": False,
        "lower_percentile": 2,
        "cell_background": 100,
        "cell_signal_to_noise": 10,
        "remove_background_cell": False,
        "nucleus_background": 100,
        "nucleus_signal_to_noise": 10,
        "remove_background_nucleus": False,
    }


def _assert_saved(stack, raw, *, kept, label=1):
    """Check the complete persisted label plane and all original channels."""
    stored = np.load(stack.path)
    np.testing.assert_array_equal(stored[..., :3], raw)
    expected = np.zeros(raw.shape[:2], dtype=np.uint16)
    if kept:
        expected[1:3, 1:3] = label
    np.testing.assert_array_equal(stored[..., 3], expected)
    assert stored.shape == (*raw.shape[:2], 4)
    assert stack.shape == stored.shape
    assert stack.channels == ["nucleus", "pathogen", "cell", "mask"]


@pytest.mark.parametrize("minimum,maximum,kept", [
    (25, 0, (False, True, True)),
    (0, 60, (True, True, False)),
    (25, 60, (False, True, False)),
    (0, 0, (True, True, True)),
])
def test_saved_labels_use_raw_own_channel_on_heterogeneous_canvases(
        tmp_path, minimum, maximum, kept):
    raws = [_raw_field(shape, mean) for shape, mean in (
        ((4, 5), 10), ((8, 7), 40), ((6, 9), 90),
    )]
    stacks = _write_stacks(tmp_path, raws)

    PV.stream_masks_from_stack(
        stacks, channels_for_cellpose=(2, 0, 2), batch_fields=3,
        postprocess_settings=_settings(minimum, maximum), keep_npz=True,
    )

    # All model inputs still share the normalization canvas and are scaled.
    assert [image.shape for image in _CaptureModel.received] == [(8, 9, 2)] * 3
    assert all(np.isfinite(image).all() and 0 <= image.min() <= image.max() <= 1
               for image in _CaptureModel.received)
    label = 1 if minimum > 0 or maximum > 0 else 7
    for stack, raw, retained in zip(stacks, raws, kept):
        _assert_saved(stack, raw, kept=retained, label=label)
    with np.load(stacks[0].path.parent / "_scratch" / "batch_0000.npz") as batch:
        for stack, raw in zip(stacks, raws):
            np.testing.assert_array_equal(batch[stack.field_id], raw)


@pytest.mark.parametrize("minimum,maximum,kept", [
    (30, 0, False),
    (0, 60, True),
])
def test_illumination_corrects_model_inputs_but_not_filter_means(
        tmp_path, minimum, maximum, kept):
    raw = _raw_field((6, 7), 20)
    stack, = _write_stacks(tmp_path, [raw])
    session = _Session()
    session.stack_by_field[stack.field_id] = stack.path
    session.raw_by_field[stack.field_id] = raw
    settings = _settings(minimum, maximum)

    PV.stream_masks_from_stack(
        [stack], channels_for_cellpose=(2, 0, 2), batch_fields=1,
        postprocess_settings=settings, illumination_session=session,
        keep_npz=True,
    )

    selected = raw[..., [2, 0]]
    np.testing.assert_array_equal(session.raw_selected[stack.field_id], selected)
    from spacr.io import _normalize_img_batch

    normalization_settings = dict(settings, cell_channel=0, nucleus_channel=1)
    expected_model_input = _normalize_img_batch(
        (selected + 100)[None, ...], channels=(0, 1), save_dtype=np.float32,
        settings=normalization_settings,
    )[0]
    np.testing.assert_array_equal(_CaptureModel.received[0], expected_model_input)
    _assert_saved(stack, raw, kept=kept)
    assert session.events == [
        ("correct", stack.field_id, stack.path.name, (2, 0)),
        ("complete", stack.field_id),
        ("finish", (stack.field_id,)),
    ]


def test_zero_bounds_preserve_the_previous_model_and_saved_stack(tmp_path):
    raw = _raw_field((6, 7), 20)
    off_stack, = _write_stacks(tmp_path / "off", [raw])
    zero_stack, = _write_stacks(tmp_path / "zero", [raw])
    settings = _settings()
    del settings["cell_min_intensity"]
    del settings["cell_max_intensity"]

    PV.stream_masks_from_stack(
        [off_stack], channels_for_cellpose=(2, 0), postprocess_settings=settings,
    )
    off_model_input = _CaptureModel.received[0].copy()
    PV.stream_masks_from_stack(
        [zero_stack], channels_for_cellpose=(2, 0),
        postprocess_settings=_settings(),
    )

    np.testing.assert_array_equal(_CaptureModel.received[0], off_model_input)
    assert off_stack.path.read_bytes() == zero_stack.path.read_bytes()
    _assert_saved(zero_stack, raw, kept=True, label=7)


@pytest.mark.parametrize("key", ["cell_min_intensity", "cell_max_intensity"])
@pytest.mark.parametrize("invalid", [-1, np.nan, np.inf, -np.inf, "invalid"])
def test_invalid_bounds_fail_before_model_construction_with_working_counterpart(
        tmp_path, monkeypatch, key, invalid):
    raw = _raw_field((6, 7), 20)
    stack, = _write_stacks(tmp_path, [raw])
    original_bytes = stack.path.read_bytes()
    constructed = []

    class _CountedModel(_CaptureModel):
        """Count constructors while inheriting the existing strict eval."""

        def __init__(self, *, gpu, pretrained_model, device, use_bfloat16=True):
            super().__init__(gpu=gpu, pretrained_model=pretrained_model,
                             device=device, use_bfloat16=use_bfloat16)
            assert not gpu and str(device) == "cpu" and use_bfloat16 is False
            constructed.append(self)

    monkeypatch.setattr("cellpose.models.CellposeModel", _CountedModel)
    settings = _settings(None, None)
    settings[key] = invalid

    with pytest.raises(ValueError, match="Intensity bounds must be finite nonnegative"):
        PV.stream_masks_from_stack(
            [stack], channels_for_cellpose=(2, 0), postprocess_settings=settings,
        )

    assert constructed == []
    assert stack.path.read_bytes() == original_bytes
    assert not (stack.path.parent / "_scratch").exists()

    # Saved numeric strings are accepted; the opposite None bound is off.
    settings[key] = "20"
    PV.stream_masks_from_stack(
        [stack], channels_for_cellpose=(2, 0), postprocess_settings=settings,
    )
    assert len(constructed) == 1
    _assert_saved(stack, raw, kept=True)
