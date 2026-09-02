"""Illumination is a segmentation input in V2, never a stored intensity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spacr import pipeline_v2 as PV


class _CaptureModel:
    received = []

    def __init__(self, *args, **kwargs):
        self.pretrained_model = None

    def eval(self, images, **kwargs):
        type(self).received = [np.asarray(image).copy() for image in images]
        masks = []
        for image in images:
            mask = np.zeros(np.asarray(image).shape[:2], dtype=np.uint16)
            mask[1:3, 1:3] = 7
            masks.append(mask)
        return masks, None, None


class _ThresholdModel:
    def __init__(self, *args, **kwargs):
        self.pretrained_model = None

    def eval(self, images, **kwargs):
        from scipy import ndimage

        masks = []
        for image in images:
            plane = np.asarray(image)
            if plane.ndim == 3:
                plane = plane[..., 0]
            mask, _count = ndimage.label(plane >= 0.6)
            masks.append(mask.astype(np.uint16))
        return masks, None, None


class _Session:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.events = []
        self.raw_selected = {}
        self.stack_by_field = {}
        self.raw_by_field = {}

    def correct(self, field_id, image, context):
        field_id = str(field_id)
        self.events.append(("correct", field_id, context.file_name,
                            context.channels))
        self.raw_selected[field_id] = np.asarray(image).copy()
        if self.fail:
            raise RuntimeError("correction failed")
        return np.asarray(image) + 100

    def mark_completed(self, field_id):
        field_id = str(field_id)
        stack = np.load(self.stack_by_field[field_id])
        raw = self.raw_by_field[field_id]
        assert stack.shape[-1] == raw.shape[-1] + 1
        assert np.array_equal(stack[..., :raw.shape[-1]], raw)
        self.events.append(("complete", field_id))
        return True

    def finish(self, expected_fields):
        fields = tuple(sorted(str(item) for item in expected_fields))
        sidecar = next(iter(self.stack_by_field.values())).parent / \
            "channel_order.json"
        assert json.loads(sidecar.read_text())["mask_channels"] == ["mask"]
        self.events.append(("finish", fields))


def _raw():
    rows = np.arange(16, dtype=np.uint16).reshape(4, 4)
    return np.stack([rows + 10, rows + 30], axis=-1)


def _stack(tmp_path: Path):
    merged = tmp_path / "merged"
    merged.mkdir()
    path = merged / "stack_A01_F001.npy"
    np.save(path, _raw())
    (merged / "channel_order.json").write_text(json.dumps({
        "image_channels": ["cell", "nucleus"],
        "mask_channels": [],
    }))
    return PV.StackFile(
        field_id="A01_F001", path=path, shape=_raw().shape,
        channels=["cell", "nucleus"],
    )


def _vignette_model(tmp_path: Path):
    from spacr.illumination import IlluminationField, IlluminationModel

    rows, cols = np.indices((16, 16), dtype=np.float32)
    radius = np.sqrt((rows - 7.5) ** 2 + (cols - 7.5) ** 2)
    flat = np.clip(1.0 - 0.07 * radius, 0.30, 1.0).astype(np.float32)
    model = IlluminationModel(
        fields={
            "stack": IlluminationField(
                plate="stack", channels=(0,), flatfield=flat[None, ...],
                dark=np.zeros(1, dtype=np.float32), n_fields=20,
                estimator="polynomial", degree=2, bin_size=1,
            ),
        },
        meta={
            "application_contract_version": 1,
            "channel_index_space": "persisted-intensity-axis",
            "estimated_from_intensity_state": "raw",
        },
    )
    return flat, Path(model.save(tmp_path / "known_vignette.npz"))


def _write_vignette_plate(path: Path, flat: np.ndarray):
    import tifffile

    path.mkdir()
    truth = np.zeros(flat.shape, dtype=np.float32)
    truth[1:4, 1:4] = 1000
    truth[7:10, 7:10] = 1000
    raw = np.rint(truth * flat).astype(np.uint16)
    tifffile.imwrite(
        path / "plate1_A01_T01F01L01A01Z01C00.tif", raw)
    return truth, raw


@pytest.fixture
def capture_cellpose(monkeypatch):
    _CaptureModel.received = []
    monkeypatch.setattr("cellpose.models.CellposeModel", _CaptureModel)


def test_v2_refuses_a_field_with_no_persisted_intensity_axis():
    with pytest.raises(ValueError, match="no persisted intensity channels"):
        PV._cellpose_channel_indices((0,), 0)


def test_v2_corrects_a_private_selection_and_commits_after_the_stack(
        tmp_path, capture_cellpose):
    stack = _stack(tmp_path)
    session = _Session()
    session.stack_by_field[stack.field_id] = stack.path
    session.raw_by_field[stack.field_id] = _raw()

    PV.stream_masks_from_stack(
        [stack], channels_for_cellpose=(1, 0, 1), batch_fields=1,
        keep_npz=True, illumination_session=session,
    )

    selected = session.raw_selected[stack.field_id]
    assert np.array_equal(selected, _raw()[..., [1, 0]])
    corrected = selected + 100
    assert np.allclose(
        _CaptureModel.received[0], corrected / corrected.max())
    stored = np.load(stack.path)
    assert np.array_equal(stored[..., :2], _raw())
    scratch = np.load(stack.path.parent / "_scratch" / "batch_0000.npz")
    assert np.array_equal(scratch[stack.field_id], _raw())
    assert session.events == [
        ("correct", stack.field_id, stack.path.name, (1, 0)),
        ("complete", stack.field_id),
        ("finish", (stack.field_id,)),
    ]


def test_v2_correction_failure_preserves_raw_and_never_claims_completion(
        tmp_path, capture_cellpose):
    stack = _stack(tmp_path)
    session = _Session(fail=True)
    session.stack_by_field[stack.field_id] = stack.path
    session.raw_by_field[stack.field_id] = _raw()

    with pytest.raises(RuntimeError, match="correction failed"):
        PV.stream_masks_from_stack(
            [stack], batch_fields=1, illumination_session=session)

    assert np.array_equal(np.load(stack.path), _raw())
    assert [event[0] for event in session.events] == ["correct"]


def test_v2_atomic_stack_failure_never_marks_the_corrected_field(
        tmp_path, capture_cellpose, monkeypatch):
    import spacr.io as spacr_io

    stack = _stack(tmp_path)
    session = _Session()
    session.stack_by_field[stack.field_id] = stack.path
    session.raw_by_field[stack.field_id] = _raw()
    monkeypatch.setattr(
        spacr_io, "_save_array_atomic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("atomic replace refused")),
    )

    with pytest.raises(OSError, match="atomic replace refused"):
        PV.stream_masks_from_stack(
            [stack], batch_fields=1, illumination_session=session)

    assert np.array_equal(np.load(stack.path), _raw())
    assert [event[0] for event in session.events] == ["correct"]


def test_v2_off_path_writes_the_same_npy_bytes_as_the_previous_contract(
        tmp_path, capture_cellpose):
    stack = _stack(tmp_path)
    expected = tmp_path / "expected.npy"
    expected_mask = np.zeros((4, 4), dtype=np.uint16)
    expected_mask[1:3, 1:3] = 7
    np.save(expected, np.concatenate(
        [_raw(), expected_mask[..., None]], axis=-1).astype(np.uint16))

    PV.stream_masks_from_stack([stack], batch_fields=1)

    assert stack.path.read_bytes() == expected.read_bytes()


def test_run_v2_prepares_once_from_raw_persisted_axis_positions(
        tmp_path, capture_cellpose, monkeypatch):
    import tifffile

    import spacr.illumination as illumination

    plate = tmp_path / "plate1"
    plate.mkdir()
    for channel, value in enumerate((10, 30, 50)):
        tifffile.imwrite(
            plate / f"plate1_A01_T01F01L01A01Z01C0{channel}.tif",
            np.full((4, 4), value, dtype=np.uint16),
        )

    session = _Session()
    calls = []

    def prepare(settings, *, src, channels, pipeline_style, verbose=None):
        stacks = sorted(Path(src).glob("stack_*.npy"))
        assert len(stacks) == 1
        assert np.load(stacks[0]).shape == (4, 4, 3)
        calls.append((dict(settings), Path(src), tuple(channels),
                      pipeline_style))
        field_id = stacks[0].stem.removeprefix("stack_")
        session.stack_by_field[field_id] = stacks[0]
        session.raw_by_field[field_id] = np.load(stacks[0]).copy()
        return session

    monkeypatch.setattr(
        illumination, "prepare_segmentation_illumination", prepare)
    settings = {"illumination_correction": True, "illumination_qc": False}

    result = PV.run_v2(
        plate, channels=(0, 1, 2), channels_for_cellpose=(2, 0, 2),
        batch_fields=1, illumination_settings=settings,
    )

    assert calls == [(settings, plate / "merged", (2, 0), "v2")]
    assert len(result["stacks"]) == 1
    assert np.array_equal(
        np.load(result["stacks"][0].path)[..., :3],
        np.stack([
            np.full((4, 4), 10, dtype=np.uint16),
            np.full((4, 4), 30, dtype=np.uint16),
            np.full((4, 4), 50, dtype=np.uint16),
        ], axis=-1),
    )


def test_run_v2_does_not_prepare_when_illumination_is_off(
        tmp_path, capture_cellpose, monkeypatch):
    import tifffile

    import spacr.illumination as illumination

    plate = tmp_path / "plate1"
    plate.mkdir()
    tifffile.imwrite(
        plate / "plate1_A01_T01F01L01A01Z01C00.tif",
        np.ones((4, 4), dtype=np.uint16),
    )
    monkeypatch.setattr(
        illumination, "prepare_segmentation_illumination",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("off must not prepare")),
    )

    PV.run_v2(
        plate, channels=(0,), batch_fields=1,
        illumination_settings={"illumination_correction": False},
    )


def test_run_v2_real_session_finishes_exact_fields_and_keeps_raw_intensity(
        tmp_path, capture_cellpose):
    import hashlib

    import tifffile

    plate = tmp_path / "plate1"
    plate.mkdir()
    originals = []
    rows, cols = np.indices((8, 8))
    vignette = (100 + 3 * rows + 5 * cols).astype(np.uint16)
    for field, offset in ((1, 0), (2, 40)):
        image = vignette + offset
        originals.append(image)
        tifffile.imwrite(
            plate / f"plate1_A01_T01F0{field}L01A01Z01C00.tif", image)

    result = PV.run_v2(
        plate, channels=(0,), batch_fields=1,
        illumination_settings={
            "illumination_correction": True,
            "illumination_qc": False,
            "illumination_estimator": "polynomial",
            "illumination_degree": 1,
            "illumination_per_plate": True,
            "illumination_max_fields": 2,
            "illumination_dark": 0,
            "illumination_on_missing": "error",
            "verbose": False,
        },
    )

    stacks = result["stacks"]
    assert len(stacks) == 2
    for stack, original in zip(stacks, originals):
        assert np.array_equal(np.load(stack.path)[..., 0], original)
    record_path = plate / "illumination" / \
        "segmentation_application.json"
    record = json.loads(record_path.read_text())
    assert record["application_state"] == "complete"
    assert record["completed_fields"] == sorted(
        stack.field_id for stack in stacks)
    assert record["source_intensity_state"] == "raw"
    assert record["target_scope"] == "segmentation-input-only"
    assert record["correction_depth"] == 1
    assert record["raw_persisted_intensities_modified"] is False
    model_path = record_path.parent / record["model_path"]
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == \
        record["model_sha256"]


def test_known_vignette_recovers_the_dim_corner_object_before_segmentation(
        tmp_path, monkeypatch):
    flat, model_path = _vignette_model(tmp_path)
    off = tmp_path / "off"
    on = tmp_path / "on"
    _write_vignette_plate(off, flat)
    _write_vignette_plate(on, flat)
    monkeypatch.setattr("cellpose.models.CellposeModel", _ThresholdModel)

    off_result = PV.run_v2(
        off, channels=(0,), batch_fields=1,
        illumination_settings={"illumination_correction": False},
    )
    on_result = PV.run_v2(
        on, channels=(0,), batch_fields=1,
        illumination_settings={
            "illumination_correction": True,
            "illumination_model": str(model_path),
            "illumination_qc": False,
            "illumination_on_missing": "error",
            "verbose": False,
        },
    )

    off_mask = np.load(off_result["stacks"][0].path)[..., -1]
    on_mask = np.load(on_result["stacks"][0].path)[..., -1]
    assert int(off_mask.max()) == 1
    assert int(on_mask.max()) == 2


def test_segmentation_and_measure_share_a_model_without_squaring_its_gain(
        tmp_path, monkeypatch):
    from spacr.illumination import IlluminationCorrector, IlluminationModel
    from spacr.measure_hooks import PreprocessingContext

    flat, model_path = _vignette_model(tmp_path)
    plate = tmp_path / "plate"
    truth, raw = _write_vignette_plate(plate, flat)
    monkeypatch.setattr("cellpose.models.CellposeModel", _ThresholdModel)
    result = PV.run_v2(
        plate, channels=(0,), batch_fields=1,
        illumination_settings={
            "illumination_correction": True,
            "illumination_model": str(model_path),
            "illumination_qc": False,
            "illumination_on_missing": "error",
            "verbose": False,
        },
    )

    merged = np.load(result["stacks"][0].path)
    assert np.array_equal(merged[..., 0], raw)
    corrector = IlluminationCorrector(
        IlluminationModel.load(model_path), verbose=False)
    context = PreprocessingContext(
        file_name=result["stacks"][0].path.name, channels=(0,), settings={})
    measured_once = corrector(merged[..., :1], context)
    applied_twice = corrector(measured_once, context)
    object_pixels = truth[..., None] > 0
    assert np.max(np.abs(
        measured_once[object_pixels].astype(float) -
        truth[..., None][object_pixels])) <= 2
    assert np.max(np.abs(
        applied_twice[object_pixels].astype(float) -
        truth[..., None][object_pixels])) > 100
