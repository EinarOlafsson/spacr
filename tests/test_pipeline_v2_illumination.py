"""Illumination is a segmentation input in V2, never a stored intensity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call
from spacr import pipeline_v2 as PV


class _CaptureModel:
    received = []

    def __init__(self, *args, **kwargs):
        self.pretrained_model = None

    # THE WHOLE SIGNATURE, and no **kwargs. A double that accepts
    # everything cannot fail when spaCR passes an argument cellpose
    # has removed -- it swallows it and the suite reports green
    # against a call the real library would reject. channel_axis is
    # the one sanctioned deviation: MISSING_CHANNEL_AXIS rather than
    # None, because None is a legal value meaning auto-detect and so
    # cannot say "the caller omitted it". The list is
    # CELLPOSE_EVAL_PARAMETERS in tests/cellpose_api_contract.py.
    def eval(self, images, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
             normalize=True, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
             anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        check_cellpose_eval_call(images, channel_axis)
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

    # THE WHOLE SIGNATURE, and no **kwargs. A double that accepts
    # everything cannot fail when spaCR passes an argument cellpose
    # has removed -- it swallows it and the suite reports green
    # against a call the real library would reject. channel_axis is
    # the one sanctioned deviation: MISSING_CHANNEL_AXIS rather than
    # None, because None is a legal value meaning auto-detect and so
    # cannot say "the caller omitted it". The list is
    # CELLPOSE_EVAL_PARAMETERS in tests/cellpose_api_contract.py.
    def eval(self, images, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
             normalize=True, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
             anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        check_cellpose_eval_call(images, channel_axis)
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
        assert json.loads(sidecar.read_text(encoding="utf-8"))["mask_channels"] == ["mask"]
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
    }), encoding="utf-8")
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

    # The monkeypatch above turns "prepared anyway" into a failure, so what
    # is left to assert is that the run HAPPENED -- a run_v2 that returned
    # early without segmenting anything would also never call prepare, and
    # would pass this test while testing nothing.
    assert _CaptureModel.received, (
        "nothing was segmented, so 'it did not prepare' says nothing")


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
    record = json.loads(record_path.read_text(encoding="utf-8"))
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


from tests.test_spectral_unmixing import calibrated_plate


@pytest.fixture
def calibrated_v2_plate(calibrated_plate, tmp_path):
    """Repack measured controls as float TIFFs with physical channels 3 and 1."""
    import tifffile

    stack, settings, calibration_path, calibration, _, _, _, _, _, expected = calibrated_plate
    source = tmp_path / "originals"
    source.mkdir()
    for path in sorted(stack.glob("*.npy")):
        camera = np.load(path)
        for position, channel in enumerate((3, 1)):
            tifffile.imwrite(source / f"{path.stem}_C{channel}.tif",
                             camera[..., position].astype(np.float32))
    calibration["reference_channel"] = 3
    for entry, channel in zip(calibration["channels"], (3, 1)):
        entry["channel"] = channel
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    settings.update(unmix_controls="3:A01;1:B01", lower_percentile=0.,
                    cell_background=0., cell_signal_to_noise=0.,
                    remove_background_cell=False, nucleus_background=0.,
                    nucleus_signal_to_noise=0., remove_background_nucleus=False)
    merged = source / "merged"
    merged.mkdir()
    np.save(merged / "untouched.npy", np.array([[[11, 12, 13]]], dtype=np.uint16))
    (merged / "channel_order.json").write_text('{"original": true}', encoding="utf-8")
    (source / "filename_map.csv").write_text("original metadata\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    kwargs = dict(channels=(3, 1), channels_for_cellpose=(0, 1),
                  channel_names=("cell", "nucleus"), metadata_type="custom",
                  custom_regex=r"^(?P<plateID>plate\d+)_(?P<wellID>[A-Z]\d+)_(?P<fieldID>\d+)_C(?P<chanID>\d+)\.tif$",
                  batch_fields=3, postprocess_settings=settings)
    return source, kwargs, before, expected


def test_calibrated_v2_keeps_originals_and_commits_reference_canvas(
        calibrated_v2_plate, capture_cellpose, monkeypatch):
    """Real TIFF assembly transports ROI, physical IDs and durable artifacts."""
    import functools
    import hashlib

    source, kwargs, before, expected = calibrated_v2_plate
    reserved = source / "calibrated_segmentation"
    reserved.mkdir()
    sentinel = reserved / "keep.txt"
    sentinel.write_text("previous result", encoding="utf-8")
    original_eval = _CaptureModel.eval

    @functools.wraps(original_eval)
    def predict_with_boundary_id(self, *args, **options):
        masks, flows, styles = original_eval(self, *args, **options)
        for mask in masks:
            mask[0, 0] = 9
            mask[5, 5] = 9
        return masks, flows, styles

    monkeypatch.setattr(_CaptureModel, "eval", predict_with_boundary_id)
    result = PV.run_v2(source, **kwargs)
    root = result["output_root"]
    assert root == source / "calibrated_segmentation_0001"
    assert result["dst"] == root / "merged"
    assert sentinel.read_text(encoding="utf-8") == "previous result"
    assert all(path.read_bytes() == data for path, data in before.items())
    assert len(_CaptureModel.received) == 3
    assert all(image.shape == (44, 52, 2) and np.isfinite(image).all()
               and image.min() >= 0 for image in _CaptureModel.received)
    metadata = json.loads((root / "merged/channel_order.json").read_text())
    assert metadata["source_channels"] == [3, 1]
    assert metadata["image_dtype"] == "float32"
    assert metadata["quantitative"] is False
    assert metadata["mask_channels"] == ["mask"]
    record = json.loads((root / "psf/segmentation_application.json").read_text())
    assert record["complete"] is True
    assert record["configuration"]["coordinate_system"] == "calibrated_reference_canvas"
    assert record["configuration"]["physical_channel_order"] == [3, 1]
    assert record["configuration"]["source_dataset"] == str(source.resolve())
    assert len(record["fields"]) == 3
    for stack_file in result["stacks"]:
        combined = np.load(stack_file.path)
        signed_path = root / "signed" / stack_file.path.name
        signed = np.load(signed_path)
        assert combined.shape == (48, 56, 3) and combined.dtype == np.float32
        assert signed.shape == (44, 52, 2) and signed.dtype == np.float32
        outside = combined.copy()
        outside[2:46, 1:53] = 0
        assert not outside.any()
        np.testing.assert_array_equal(combined[2:46, 1:53, :2], np.maximum(signed, 0))
        mask = combined[..., 2]
        expected_mask = np.zeros((48, 56), dtype=np.float32)
        expected_mask[3:5, 2:4] = 7
        np.testing.assert_array_equal(mask, expected_mask)
        row = record["calibrated_fields"][stack_file.field_id]
        assert row["source_identity"]["plate"] == "plate1"
        assert row["source_identity"]["source_dataset"] == str(source.resolve())
        assert row["source_identity"]["field"] == 1
        assert row["source_identity"]["time"] == 1
        assert row["source_identity"]["z"] == 1
        assert row["excluded_boundary_label_ids"] == [9]
        assert row["measurement_schema_version"] == 1
        assert row["mask_role"] == "cell" and row["mask_channel_position"] == 2
        for key, path in (("display_mask", stack_file.path), ("signed_crop", signed_path)):
            assert root / row[key]["path"] == path
            assert row[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        if row["source_identity"]["well"] == "C01":
            np.testing.assert_allclose(signed, expected[2:46, 1:53], atol=5e-4, rtol=0)
    mapping = PV.FilenameMapper.load_csv(root / "filename_map.csv")
    assert {entry.channel for entry in mapping.records} == {3, 1}
    assert {entry.well for entry in mapping.records} == {"A01", "B01", "C01"}
    from spacr.measure import measure_calibrated_cells
    from spacr.tabular import read_table

    measured = measure_calibrated_cells(root, dst=source.parent / "signed_cells", channels=[1])
    cells = read_table(measured["csv_path"])
    assert measured["fields"] == measured["rows"] == 3
    assert set(cells["physical_channel_id"]) == {1}
    assert set(cells["label_id"]) == {7}
    assert set(cells["fieldID"]) == set(cells["timeID"]) == set(cells["sliceID"]) == {1}
    assert set(cells["wellID"]) == {"A01", "B01", "C01"}
    assert all(path.read_bytes() == data for path, data in before.items())


def test_calibrated_v2_invalid_model_labels_leave_incomplete_separate_output(
        calibrated_v2_plate, capture_cellpose, monkeypatch):
    """A fractional model label refuses before any success artifact is committed."""
    import functools

    source, kwargs, before, _ = calibrated_v2_plate
    original_eval = _CaptureModel.eval

    @functools.wraps(original_eval)
    def fractional_labels(self, *args, **options):
        masks, flows, styles = original_eval(self, *args, **options)
        masks = [mask.astype(np.float32) for mask in masks]
        masks[0][4, 4] = .5
        return masks, flows, styles

    monkeypatch.setattr(_CaptureModel, "eval", fractional_labels)
    with pytest.raises(ValueError, match="labels"):
        PV.run_v2(source, **kwargs)
    root = source / "calibrated_segmentation"
    record = json.loads((root / "psf/segmentation_application.json").read_text())
    assert record["complete"] is False and record["fields"] == []
    assert record["configuration"]["coordinate_system"] == "calibrated_reference_canvas"
    assert all("display_mask" not in row and "signed_crop" not in row
               for row in record["calibrated_fields"].values())
    assert not (root / "signed").exists()
    assert all(np.load(path).shape == (48, 56, 2)
               for path in (root / "merged").glob("*.npy"))
    assert all(path.read_bytes() == data for path, data in before.items())


def test_calibrated_v2_rejects_multiplane_source_before_model(
        calibrated_v2_plate, capture_cellpose):
    """A real multi-plane TIFF is never silently reduced to its first plane."""
    import tifffile

    source, kwargs, before, _ = calibrated_v2_plate
    path = source / "plate1_A01_1_C3.tif"
    tifffile.imwrite(path, np.ones((2, 48, 56), dtype=np.float32), metadata={"axes": "ZYX"})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="2D"):
        PV.run_v2(source, **kwargs)
    assert not _CaptureModel.received
    assert path.read_bytes() == original
    assert all(file.read_bytes() == data for file, data in before.items() if file != path)
    root = source / "calibrated_segmentation"
    record = json.loads((root / "psf/segmentation_application.json").read_text())
    assert record["complete"] is False
    assert record["configuration"]["coordinate_system"] == "calibrated_reference_canvas"
    assert not list((root / "merged").glob("*.npy"))


def test_calibrated_v2_caps_aggregate_batch_before_private_copies(
        calibrated_v2_plate, capture_cellpose, monkeypatch):
    """Real field headers bound the batch before private images or a model exist."""
    source, kwargs, before, _ = calibrated_v2_plate
    per_field_values = 48 * 56 * 2
    monkeypatch.setattr(PV, "_CALIBRATED_BATCH_VALUES", per_field_values + 1)
    constructed = []
    original_init = _CaptureModel.__init__

    def observed_init(self, *args, **options):
        constructed.append(self)
        original_init(self, *args, **options)

    monkeypatch.setattr(_CaptureModel, "__init__", observed_init)
    with pytest.raises(ValueError, match="aggregate batch-pixel") as caught:
        PV.run_v2(source, **kwargs)
    trace = caught.value.__traceback__
    while trace is not None and trace.tb_frame.f_code.co_name != "stream_masks_from_stack":
        trace = trace.tb_next
    assert trace is not None
    assert trace.tb_frame.f_locals["field_values"] == [per_field_values] * 3
    assert "loaded" not in trace.tb_frame.f_locals
    assert not constructed and not _CaptureModel.received
    root = source / "calibrated_segmentation"
    record = json.loads((root / "psf/segmentation_application.json").read_text())
    assert not record["complete"] and not record["fields"]
    assert all(path.read_bytes() == data for path, data in before.items())


def test_calibrated_v2_caps_orthogonal_roi_padding_before_allocation(
        calibrated_v2_plate, capture_cellpose, monkeypatch):
    """Skinny real TIFF fields cannot create an unbounded padded batch canvas."""
    import tifffile

    source, kwargs, _, _ = calibrated_v2_plate
    for well, shape in (("C01", (48, 8)), ("D01", (8, 56))):
        for channel in (3, 1):
            tifffile.imwrite(source / f"plate1_{well}_1_C{channel}.tif",
                             np.full(shape, 10.25, dtype=np.float32))
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    kwargs["batch_fields"] = 4
    monkeypatch.setattr(PV, "_CALIBRATED_BATCH_VALUES", 13000)
    with pytest.raises(ValueError, match="padded batch-pixel") as caught:
        PV.run_v2(source, **kwargs)
    trace = caught.value.__traceback__
    while trace is not None and trace.tb_frame.f_code.co_name != "stream_masks_from_stack":
        trace = trace.tb_next
    assert trace is not None
    state = trace.tb_frame.f_locals
    assert sum(state["field_values"]) == 12416
    assert [image.shape for image in state["selected_images"]] == [
        (44, 52, 2), (44, 52, 2), (44, 4, 2), (4, 52, 2)]
    assert "selected_stack" not in state and "normalized_stack" not in state
    assert not _CaptureModel.received
    root = source / "calibrated_segmentation"
    record = json.loads((root / "psf/segmentation_application.json").read_text())
    assert not record["complete"] and not record["fields"]
    assert not (root / "signed").exists()
    assert all(path.read_bytes() == data for path, data in before.items())


def test_calibrated_v2_rejects_empty_valid_roi_before_model(
        calibrated_v2_plate, capture_cellpose, monkeypatch):
    """Valid controls do not authorize model loading for an unusable target ROI."""
    import tifffile

    source, kwargs, _, _ = calibrated_v2_plate
    for channel in (3, 1):
        tifffile.imwrite(source / f"plate1_C01_1_C{channel}.tif",
                         np.full((3, 3), 5.25, dtype=np.float32))
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    constructed = []
    original_init = _CaptureModel.__init__

    def observed_init(self, *args, **options):
        constructed.append(self)
        original_init(self, *args, **options)

    monkeypatch.setattr(_CaptureModel, "__init__", observed_init)
    with pytest.raises(ValueError, match="valid"):
        PV.run_v2(source, **kwargs)
    assert not constructed and not _CaptureModel.received
    root = source / "calibrated_segmentation"
    record = json.loads((root / "psf/segmentation_application.json").read_text())
    assert not record["complete"] and not record["fields"]
    assert not (root / "signed").exists()
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("unsupported", [
    {"cell_perimiter_fraction": .5},
    {"image_qc_mode": "exclude"},
], ids=["legacy-perimeter", "quality-exclusion"])
def test_calibrated_v2_refuses_unsupported_output_policies_before_artifacts(
        calibrated_v2_plate, capture_cellpose, unsupported, monkeypatch):
    """Known unsupported policies cannot publish mixed-coordinate output trees."""
    source, kwargs, before, _ = calibrated_v2_plate
    kwargs["postprocess_settings"].update(unsupported)
    constructed = []
    original_init = _CaptureModel.__init__

    def observed_init(self, *args, **options):
        constructed.append(self)
        original_init(self, *args, **options)

    monkeypatch.setattr(_CaptureModel, "__init__", observed_init)
    with pytest.raises(ValueError):
        PV.run_v2(source, **kwargs)
    assert not constructed and not _CaptureModel.received
    assert not list(source.glob("calibrated_segmentation*"))
    assert {path for path in source.rglob("*") if path.is_file()} == set(before)
    assert all(path.read_bytes() == data for path, data in before.items())


def test_calibrated_stream_refuses_original_paths_before_any_write(
        calibrated_v2_plate, capture_cellpose, monkeypatch):
    """A real captured session cannot overwrite originals through an external path."""
    from functools import partial
    from spacr.psf_pipeline import _initialize_calibrated_output, _prepare_segmentation_psf

    source, kwargs, original_bytes, _ = calibrated_v2_plate
    settings = kwargs["postprocess_settings"]
    root = source / "calibrated_segmentation"
    _initialize_calibrated_output(root, settings)
    mapper = PV.FilenameMapper.discover(
        source, metadata_type="custom", custom_regex=kwargs["custom_regex"])
    stacks = PV.stream_originals_to_stack(
        source, mapper, channels=(3, 1), channel_names=("cell", "nucleus"),
        dst=root / "merged", metadata_root=root, calibrated=True)
    identities = {}
    for field_id, records in mapper.by_field().items():
        record = records[0]
        identity = dict(plate=record.plate, well=record.well, field=record.field,
                        time=record.time, z=record.z, source_dataset=str(source.resolve()))
        identities[field_id] = identity
        identities[f"stack_{field_id}.npy"] = identity
    session = _prepare_segmentation_psf(
        settings, root, (0, 1), pipeline_style="v2", stack_dir=root / "merged",
        load=partial(np.load, mmap_mode="r", allow_pickle=False),
        source_channels=(3, 1), field_identities=identities)
    original = source / "merged" / stacks[0].path.name
    np.save(original, np.load(stacks[0].path))
    alias = root / "merged/escaped.npy"
    alias.symlink_to(original)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in source.rglob("*") if path.is_file()}
    constructed = []
    original_init = _CaptureModel.__init__

    def observed_init(self, *args, **options):
        constructed.append(self)
        original_init(self, *args, **options)

    monkeypatch.setattr(_CaptureModel, "__init__", observed_init)
    for path in (original, alias):
        supplied = PV.StackFile(field_id=stacks[0].field_id, path=path,
                                shape=(48, 56, 2), channels=["cell", "nucleus"])
        with pytest.raises(ValueError, match="separate output merged"):
            PV.stream_masks_from_stack(
                [stacks[0], supplied], psf_session=session,
                postprocess_settings=settings, channels_for_cellpose=(0, 1))
    assert not constructed and not _CaptureModel.received
    assert all(path.read_bytes() == data and path.stat().st_mtime_ns == modified
               for path, (data, modified) in before.items())
    assert all(path.read_bytes() == data for path, data in original_bytes.items())
    (root / "merged").rename(root / "assembled")
    (root / "merged").symlink_to(original.parent, target_is_directory=True)
    before_redirect = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                       for path in source.rglob("*") if path.is_file()}
    redirected = PV.StackFile(field_id=stacks[0].field_id,
                              path=root / "merged" / original.name,
                              shape=(48, 56, 2), channels=["cell", "nucleus"])
    with pytest.raises(ValueError, match="separate output merged"):
        PV.stream_masks_from_stack(
            [redirected], psf_session=session, postprocess_settings=settings,
            channels_for_cellpose=(0, 1))
    assert not constructed and not _CaptureModel.received
    assert all(path.read_bytes() == data and path.stat().st_mtime_ns == modified
               for path, (data, modified) in before_redirect.items())
    assert all(path.read_bytes() == data for path, data in original_bytes.items())
    record = json.loads((root / "psf/segmentation_application.json").read_text())
    assert not record["complete"] and not record["fields"]
