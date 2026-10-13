"""418: V1 filters original own-channel values with resume and z alignment."""

from __future__ import annotations

import numpy as np
import pytest

import spacr.object as O
from spacr.utils import _filter_objects
from spacr import zstack as Z
from tests.test_cov_object_masks_sam import fake_model, fake_timelapse, force_cpu


def _settings(src, **overrides):
    """Raw cell channel two becomes dense NPZ channel one, after nucleus zero."""
    settings = {
        "src": str(src),
        "cell_channel": 2,
        "nucleus_channel": 0,
        "pathogen_channel": None,
        "organelle_channel": None,
        "cell_min_area": 0,
        "cell_max_area": 0,
        "cell_perimeter_fraction": 0,
        "cell_remove_border_objects": False,
        "cell_min_intensity": 10,
        "cell_max_intensity": 60,
        "magnification": 20,
        "batch_size": 3,
        "verbose": False,
        "plot": False,
        "save": True,
        "timelapse": False,
        "n_jobs": 1,
        "resume": True,
        "seg_qc": "off",
    }
    settings.update(overrides)
    return settings


def _raw_field(shape, means):
    """Both decoy channels are above the bounds; channel two owns the cells."""
    raw = np.full((*shape, 3), 1000, dtype=np.uint16)
    raw[..., 0] = 400
    raw[..., 2] = 2
    raw[2:8, 2:8, 2] = means[0]
    raw[12:18, 12:18, 2] = means[1]
    return raw


def _write_npz(src, data, filenames):
    src.mkdir(parents=True, exist_ok=True)
    np.savez(src / "batch.npz", data=data, filenames=np.asarray(filenames))


def _expected_mask(shape, objects):
    """The strict shared model returns these two fixed, non-touching cells."""
    expected = np.zeros(shape, dtype=np.uint16)
    regions = {1: (slice(2, 8), slice(2, 8)),
               2: (slice(12, 18), slice(12, 18))}
    for label, original in enumerate(objects, 1):
        expected[regions[original]] = label
    return expected


def test_resume_uses_surviving_raw_filenames_and_original_batch_canvas(
        tmp_path, fake_model, force_cpu):
    src = tmp_path / "masks"
    raw_dir = tmp_path / "stack"
    raw_dir.mkdir()
    filenames = [f"plate1_A01_{index}.npy" for index in (1, 2, 3)]
    raws = [_raw_field((32, 40), (5, 50)),
            _raw_field((48, 36), (50, 5))]
    normalized = np.zeros((3, 64, 64, 2), dtype=np.float32)
    normalized[0, ..., 0] = 0.75
    normalized[0, ..., 1] = 0.25
    for index, (filename, raw) in enumerate(zip(filenames[1:], raws), 1):
        np.save(raw_dir / filename, raw)
        normalized[index, :raw.shape[0], :raw.shape[1]] = (
            raw[..., [0, 2]].astype(np.float32) / 1000)
    _write_npz(src, normalized, filenames)
    output_dir = src / "cell_mask_stack"
    output_dir.mkdir()
    skipped_output = output_dir / filenames[0]
    np.save(skipped_output, np.full((64, 64), 17, dtype=np.uint16))
    skipped_bytes = skipped_output.read_bytes()
    originals = {filename: (raw_dir / filename).read_bytes()
                 for filename in filenames[1:]}
    assert not (raw_dir / filenames[0]).exists()

    O.generate_cellpose_masks_sam(str(src), _settings(src), "cell")

    model = fake_model["model"]
    assert model.gpu is False
    assert len(model.eval_inputs) == 1
    np.testing.assert_array_equal(
        np.asarray(model.eval_inputs[0]), normalized[1:][..., [1, 0]])
    assert model.eval_kwargs[0]["normalize"] is False
    assert model.eval_kwargs[0]["channel_axis"] == -1
    assert skipped_output.read_bytes() == skipped_bytes
    for filename, kept_object in zip(filenames[1:], (2, 1)):
        np.testing.assert_array_equal(
            np.load(output_dir / filename),
            _expected_mask((64, 64), (kept_object,)),
        )
        assert (raw_dir / filename).read_bytes() == originals[filename]
    assert not (raw_dir / filenames[0]).exists()


@pytest.mark.parametrize("disabled", [0, None])
def test_disabled_bounds_accept_npz_only_without_original_files(
        tmp_path, fake_model, force_cpu, disabled):
    src = tmp_path / "masks"
    filename = "plate1_A01_1.npy"
    normalized = np.full((1, 32, 32, 2), 0.25, dtype=np.float32)
    normalized[..., 1] = 0.75
    _write_npz(src, normalized, [filename])

    O.generate_cellpose_masks_sam(
        str(src), _settings(src, cell_min_intensity=disabled,
                            cell_max_intensity=disabled), "cell")

    np.testing.assert_array_equal(
        np.asarray(fake_model["model"].eval_inputs[0]),
        normalized[..., [1, 0]],
    )
    np.testing.assert_array_equal(
        np.load(src / "cell_mask_stack" / filename),
        _expected_mask((32, 32), (1, 2)),
    )
    assert not (tmp_path / "stack").exists()


def test_missing_raw_refuses_before_overwriting_a_resumed_output(
        tmp_path, fake_model, force_cpu):
    src = tmp_path / "masks"
    filename = "plate1_A01_1.npy"
    _write_npz(src, np.full((1, 32, 32, 2), 0.5, dtype=np.float32), [filename])
    output_dir = src / "cell_mask_stack"
    output_dir.mkdir()
    output = output_dir / filename
    # Resume should regenerate a truncated prior output, but only with raw data.
    output.write_bytes(b"unfinished npy output")
    previous_bytes = output.read_bytes()

    with pytest.raises(FileNotFoundError) as error:
        O.generate_cellpose_masks_sam(str(src), _settings(src), "cell")

    assert filename in str(error.value)
    assert output.read_bytes() == previous_bytes
    assert len(fake_model["model"].eval_inputs) == 1

    raw_dir = tmp_path / "stack"
    raw_dir.mkdir()
    raw_path = raw_dir / filename
    np.save(raw_path, _raw_field((32, 32), (5, 50)))
    original_bytes = raw_path.read_bytes()
    O.generate_cellpose_masks_sam(str(src), _settings(src), "cell")

    np.testing.assert_array_equal(np.load(output),
                                  _expected_mask((32, 32), (2,)))
    assert raw_path.read_bytes() == original_bytes


@pytest.mark.parametrize("axis_order", ["ZTYX", "TZYX"])
def test_time_archives_keep_raw_filenames_and_axes_across_batches_and_resume(
        tmp_path, fake_model, force_cpu, axis_order):
    """T != Z in both archives; a skipped timepoint has no raw volume."""
    src = tmp_path / "masks"
    src.mkdir()
    raw_dir = tmp_path / "stack"
    raw_dir.mkdir()
    output_dir = src / "cell_mask_stack"
    output_dir.mkdir()
    expected_inputs = []
    expected_masks = {}
    original_bytes = {}
    archive_bytes = {}
    skipped_filename = "plate1_A01_2.npy"
    skipped_output = output_dir / skipped_filename
    np.save(skipped_output, np.full((48, 48), 17, dtype=np.uint16))
    skipped_bytes = skipped_output.read_bytes()

    next_id = 1
    for archive_id, (n_t, n_z) in enumerate(((5, 3), (3, 4))):
        canonical = np.zeros((n_t, n_z, 48, 48, 2), dtype=np.float32)
        filenames = []
        for t in range(n_t):
            field_id = next_id
            next_id += 1
            filename = f"plate1_A01_{field_id}.npy"
            filenames.append(filename)
            for z in range(n_z):
                # Deliberately distinct normalized values identify each time
                # and z plane without being usable as absolute raw bounds.
                canonical[t, z, ..., 0] = 0.2 + field_id * 0.01
                canonical[t, z, ..., 1] = 0.5 + field_id * 0.01 + z * 0.001
            if filename == skipped_filename:
                continue
            kept_object = 1 if field_id % 2 else 2
            means = (50, 5) if kept_object == 1 else (5, 50)
            raw = np.stack([_raw_field((32, 40), means) for _ in range(n_z)])
            raw_path = raw_dir / filename
            np.save(raw_path, raw)
            original_bytes[filename] = raw_path.read_bytes()
            expected_masks[filename] = _expected_mask((48, 48), (kept_object,))
            expected_inputs.append(canonical[t].max(axis=0)[..., [1, 0]])
        stored = (np.moveaxis(canonical, 0, 1)
                  if axis_order == "ZTYX" else canonical)
        path = src / f"batch{archive_id}.npz"
        np.savez(path, data=stored, filenames=np.asarray(filenames))
        archive_bytes[path] = path.read_bytes()

    settings = _settings(src, t_stack=True, t_axis_order=axis_order,
                         z_segmentation_mode="project", z_projection="max",
                         batch_size=2)
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    model = fake_model["model"]
    assert len(model.eval_inputs) == 7
    assert all(len(images) == 1 for images in model.eval_inputs)
    actual_inputs = [images[0] for images in model.eval_inputs]
    # os.listdir does not promise archive order; each frame's own-channel
    # value uniquely identifies it, so compare independently of that order.
    by_own_channel = lambda image: float(image[0, 0, 0])
    for actual, expected in zip(sorted(actual_inputs, key=by_own_channel),
                                sorted(expected_inputs, key=by_own_channel)):
        np.testing.assert_array_equal(actual, expected)
    assert all(call["normalize"] is False for call in model.eval_kwargs)
    assert set(path.name for path in output_dir.iterdir()) == (
        set(expected_masks) | {skipped_filename})
    for filename, expected in expected_masks.items():
        np.testing.assert_array_equal(np.load(output_dir / filename), expected)
        assert (raw_dir / filename).read_bytes() == original_bytes[filename]
    assert skipped_output.read_bytes() == skipped_bytes
    assert not (raw_dir / skipped_filename).exists()
    assert settings["t_axis_order"] == axis_order
    for path, before in archive_bytes.items():
        assert path.read_bytes() == before


@pytest.mark.parametrize("batch_size", [2, 5])
def test_ztyx_frame_limits_select_time_before_resume_and_raw_loading(
        tmp_path, fake_model, fake_timelapse, force_cpu, batch_size):
    src = tmp_path / "masks"
    raw_dir = tmp_path / "stack"
    raw_dir.mkdir()
    filenames = [f"plate1_A01_{index}.npy" for index in range(1, 6)]
    canonical = np.zeros((5, 3, 48, 48, 2), dtype=np.float32)
    for t in range(5):
        canonical[t, ..., 0] = 0.2 + t * 0.01
        canonical[t, ..., 1] = 0.5 + t * 0.01
    _write_npz(src, np.moveaxis(canonical, 0, 1), filenames)
    output_dir = src / "cell_mask_stack"
    output_dir.mkdir()
    skipped_output = output_dir / filenames[2]
    np.save(skipped_output, np.full((48, 48), 17, dtype=np.uint16))
    skipped_bytes = skipped_output.read_bytes()
    originals = {}
    for t, means in ((1, (5, 50)), (3, (50, 5))):
        raw_path = raw_dir / filenames[t]
        np.save(raw_path, np.stack([_raw_field((32, 40), means)] * 3))
        originals[filenames[t]] = raw_path.read_bytes()

    O.generate_cellpose_masks_sam(
        str(src), _settings(src, t_stack=True, t_axis_order="ZTYX",
                            z_segmentation_mode="project", z_projection="mean",
                            timelapse=True, timelapse_objects=[],
                            timelapse_frame_limits=[1, 4], batch_size=batch_size),
        "cell",
    )

    model = fake_model["model"]
    assert len(model.eval_inputs) == 2
    for images, t in zip(model.eval_inputs, (1, 3)):
        np.testing.assert_array_equal(
            images, [canonical[t].mean(axis=0)[..., [1, 0]]])
    assert len(fake_timelapse["movie"]) == 1
    assert fake_timelapse["movie"][0]["n_frames"] == 2
    assert fake_timelapse["movie"][0]["filenames"] == [filenames[1], filenames[3]]
    assert set(path.name for path in output_dir.iterdir()) == set(filenames[1:4])
    assert skipped_output.read_bytes() == skipped_bytes
    for t, kept_object in ((1, 2), (3, 1)):
        np.testing.assert_array_equal(np.load(output_dir / filenames[t]),
                                      _expected_mask((48, 48), (kept_object,)))
        assert (raw_dir / filenames[t]).read_bytes() == originals[filenames[t]]
    for t in (0, 2, 4):
        assert not (raw_dir / filenames[t]).exists()


@pytest.mark.parametrize("bad_manifest", ["z_count", "too_many", "matrix"])
def test_time_filename_manifest_is_validated_before_inference_or_overwrite(
        tmp_path, fake_model, force_cpu, bad_manifest):
    src = tmp_path / "masks"
    raw_dir = tmp_path / "stack"
    raw_dir.mkdir()
    filenames = [f"plate1_A01_{index}.npy" for index in range(1, 4)]
    acquisition = np.full((2, 3, 32, 32, 2), 0.25, dtype=np.float32)  # Z != T
    invalid_filenames = {
        "z_count": filenames[:2],
        "too_many": filenames + ["plate1_A01_4.npy"],
        "matrix": np.asarray(filenames)[:, None],
    }[bad_manifest]
    _write_npz(src, acquisition, invalid_filenames)
    output_dir = src / "cell_mask_stack"
    output_dir.mkdir()
    existing = output_dir / filenames[0]
    np.save(existing, np.full((32, 32), 17, dtype=np.uint16))
    previous_bytes = existing.read_bytes()
    settings = _settings(src, t_stack=True, t_axis_order="ZTYX",
                         z_segmentation_mode="project", z_projection="max",
                         batch_size=2, resume=False)

    with pytest.raises(ValueError, match="one filename per timepoint"):
        O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert fake_model["model"].eval_inputs == []
    assert existing.read_bytes() == previous_bytes
    assert list(output_dir.iterdir()) == [existing]

    # The same acquisition is supported with a genuine timepoint manifest.
    # A valid existing output is skipped even when resume=False; that flag
    # controls repairing invalid outputs, not overwriting valid ones.
    _write_npz(src, acquisition, filenames)
    for filename in filenames:
        np.save(raw_dir / filename,
                np.stack([_raw_field((32, 32), (5, 50))] * 2))
    originals = {filename: (raw_dir / filename).read_bytes() for filename in filenames}
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(fake_model["model"].eval_inputs) == 2
    assert set(path.name for path in output_dir.iterdir()) == set(filenames)
    assert existing.read_bytes() == previous_bytes
    for filename in filenames[1:]:
        np.testing.assert_array_equal(np.load(output_dir / filename),
                                      _expected_mask((32, 32), (2,)))
    for filename in filenames:
        assert (raw_dir / filename).read_bytes() == originals[filename]


def _write_raw_volume(root, raw):
    raw_dir = root / "stack"
    raw_dir.mkdir()
    filename = "plate1_A01_1.npy"
    np.save(raw_dir / filename, raw)
    return str(root / "masks"), filename, raw_dir / filename


@pytest.mark.parametrize("mode,value", [("max", 10), ("mean", 17 / 3), ("sum", 17)])
@pytest.mark.parametrize("z_axis", [0, 1])
def test_raw_projection_preserves_absolute_values_and_padding(
        tmp_path, mode, value, z_axis):
    raw = np.full((3, 4, 5, 3), 500, dtype=np.uint16)
    raw[..., 2] = np.asarray([2, 5, 10], dtype=np.uint16)[:, None, None]
    model_input = np.full((3, 6, 7, 2), 0.25, dtype=np.float32)
    raw = np.moveaxis(raw, 0, z_axis)
    model_input = np.moveaxis(model_input, 0, z_axis)
    src, filename, raw_path = _write_raw_volume(tmp_path, raw)
    original_bytes = raw_path.read_bytes()
    mask = np.zeros((6, 7), dtype=np.uint16)
    mask[1:3, 1:3] = 7
    mask[4:6, 5:7] = 19  # wholly outside the original field: padded values are zero

    plane, = O._raw_filter_images(
        src, [filename], [model_input], [mask], 2,
        z_axis=z_axis, projection=mode,
    )

    expected = np.zeros(mask.shape, dtype=np.float64)
    expected[:4, :5] = value
    np.testing.assert_allclose(plane, expected, rtol=0, atol=1e-12)
    filtered = _filter_objects(mask.copy(), plane,
                               min_intensity=value - 0.01,
                               max_intensity=value + 0.01)
    np.testing.assert_array_equal(filtered, (mask == 7).astype(np.uint16))
    assert raw_path.read_bytes() == original_bytes


def test_best_focus_uses_the_model_slice_and_that_slices_original_values(tmp_path):
    checker = (np.indices((8, 8)).sum(axis=0) % 2).astype(np.uint16)
    raw = np.full((2, 8, 8, 3), 1000, dtype=np.uint16)
    raw[0, ..., 2] = checker * 1000 + 5
    raw[1, ..., 2] = 50
    model_input = np.full((2, 8, 8, 2), 0.2, dtype=np.float32)
    model_input[1, ..., 0] = 0.1
    model_input[1, ..., 1] = checker
    assert Z._best_focus_index(model_input) == 1
    assert Z._best_focus_index(raw[..., 2]) == 0
    src, filename, raw_path = _write_raw_volume(tmp_path, raw)
    original_bytes = raw_path.read_bytes()
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[1:3, 1:3] = 1
    mask[4:6, 4:6] = 2

    plane, = O._raw_filter_images(
        src, [filename], [model_input], [mask], 2,
        z_axis=0, projection="best_focus",
    )

    np.testing.assert_array_equal(plane, raw[1, ..., 2])
    np.testing.assert_array_equal(
        _filter_objects(mask.copy(), plane, min_intensity=10, max_intensity=60),
        mask,
    )
    assert raw_path.read_bytes() == original_bytes


def test_single_z_plane_matches_flat_labels_without_losing_raw_precision(tmp_path):
    exact = 2**24 + 1
    raw = np.zeros((1, 4, 5, 3), dtype=np.uint32)
    raw[..., 2] = exact
    model_input = np.full((1, 6, 7, 2), 0.5, dtype=np.float32)
    src, filename, _ = _write_raw_volume(tmp_path, raw)
    mask = np.zeros((6, 7), dtype=np.uint16)
    mask[1:3, 1:3] = 17

    plane, = O._raw_filter_images(
        src, [filename], [model_input], [mask], 2, z_axis=0,
    )

    assert plane.shape == mask.shape
    assert plane.dtype == np.uint32
    assert np.all(plane[:4, :5] == exact)
    assert not plane[4:].any() and not plane[:, 5:].any()
    np.testing.assert_array_equal(
        _filter_objects(mask.copy(), plane, min_intensity=exact, max_intensity=exact),
        (mask > 0).astype(np.uint16),
    )


def test_whole_volume_intensity_bounds_keep_each_label_together_across_z(tmp_path):
    raw = np.zeros((2, 4, 5, 3), dtype=np.uint16)
    raw[0, ..., 2] = 5
    raw[1, ..., 2] = 55
    raw[:, 1:3, 3:5, 2] = 5
    model_input = np.full((2, 6, 7, 2), 0.5, dtype=np.float32)
    src, filename, _ = _write_raw_volume(tmp_path, raw)
    mask = np.zeros((2, 6, 7), dtype=np.uint16)
    mask[:, 1:3, 1:3] = 7
    mask[:, 1:3, 3:5] = 19

    plane, = O._raw_filter_images(
        src, [filename], [model_input], [mask], 2,
        z_axis=0, projection="max",
    )

    assert plane.shape == mask.shape
    np.testing.assert_array_equal(plane[:, :4, :5], raw[..., 2])
    assert np.mean(plane[mask == 7]) == 30
    filtered = _filter_objects(mask.copy(), plane, min_intensity=10, max_intensity=40)
    np.testing.assert_array_equal(filtered, (mask == 7).astype(np.uint16))
    assert np.all(filtered[:, 1:3, 1:3] == 1)
