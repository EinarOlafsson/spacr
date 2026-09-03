"""V1 mask preprocessing applies illumination only to Cellpose inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class _Session:
    """Record correction/completion order and alter one spatial corner."""

    def __init__(self, stack: Path, output: Path):
        self.stack = stack
        self.output = output
        self.events = []
        self.completed = []
        self.raw_before = {
            path.stem: path.read_bytes() for path in stack.glob("*.npy")
        }

    def correct(self, field_id, selected, context):
        field_id = str(field_id)
        assert context.file_name == f"{field_id}.npy"
        assert tuple(context.channels) == (0, 1)
        assert (self.stack / context.file_name).read_bytes() == \
            self.raw_before[field_id]
        self.events.append(("correct", field_id))
        corrected = np.asarray(selected).copy()
        corrected[0, 0, :] = 0
        return corrected

    def mark_completed(self, field_id):
        # Completion is legal only after a durable, readable NPZ exists.
        written = sorted(self.output.glob("*.npz"))
        assert written
        with np.load(written[-1]) as archive:
            assert archive["data"].size
        self.events.append(("complete", str(field_id)))
        self.completed.append(str(field_id))
        return True

    def finish(self, expected_fields):
        fields = tuple(expected_fields)
        if set(fields) != set(self.completed):
            raise RuntimeError(
                f"incomplete illumination fields: expected={fields}, "
                f"completed={tuple(self.completed)}")
        self.events.append(("finish", fields))


def _stack(folder: Path):
    folder.mkdir(parents=True)
    rows, cols = np.indices((8, 8))
    for index in range(2):
        first = 20 + rows * 5 + cols * 3 + index * 7
        second = 40 + rows * 2 + cols * 6 + index * 11
        np.save(
            folder / f"plate1_A01_F00{index}.npy",
            np.stack([first, second], axis=-1).astype(np.uint16),
        )


def _settings(src: Path):
    from spacr.settings import set_default_settings_preprocess_generate_masks

    return set_default_settings_preprocess_generate_masks({
        "src": str(src.parent),
        "channels": [0, 1],
        "nucleus_channel": 0,
        "cell_channel": 1,
        "pathogen_channel": None,
        "randomize": False,
        "timelapse": False,
        "batch_size": 2,
        "plot": False,
        "lower_percentile": 2,
    })


def test_v1_corrects_private_inputs_then_atomically_completes_the_batch(
        tmp_path):
    """Raw stack bytes stay fixed while the normalised segmentation NPZ moves."""
    from spacr.io import concatenate_and_normalize

    corrected_stack = tmp_path / "corrected" / "stack"
    baseline_stack = tmp_path / "baseline" / "stack"
    _stack(corrected_stack)
    _stack(baseline_stack)
    session = _Session(corrected_stack, corrected_stack.parent / "masks")

    concatenate_and_normalize(
        str(corrected_stack), [0, 1], settings=_settings(corrected_stack),
        illumination_session=session,
    )
    concatenate_and_normalize(
        str(baseline_stack), [0, 1], settings=_settings(baseline_stack),
    )

    assert [event[0] for event in session.events] == [
        "correct", "correct", "complete", "complete", "finish"]
    corrected_ids = tuple(event[1] for event in session.events[:2])
    assert set(corrected_ids) == {"plate1_A01_F000", "plate1_A01_F001"}
    assert {event[1] for event in session.events[2:4]} == set(corrected_ids)
    assert session.events[-1] == ("finish", corrected_ids)
    assert {
        path.stem: path.read_bytes() for path in corrected_stack.glob("*.npy")
    } == session.raw_before
    corrected_npz = next((corrected_stack.parent / "masks").glob("*.npz"))
    baseline_npz = next((baseline_stack.parent / "masks").glob("*.npz"))
    with np.load(corrected_npz) as corrected, np.load(baseline_npz) as baseline:
        assert not np.array_equal(corrected["data"], baseline["data"])


def test_v1_off_path_keeps_the_existing_normalised_values(tmp_path):
    """Passing no session retains the exact pre-339 scientific output."""
    from spacr.io import concatenate_and_normalize

    implicit = tmp_path / "implicit" / "stack"
    explicit = tmp_path / "explicit" / "stack"
    _stack(implicit)
    _stack(explicit)

    concatenate_and_normalize(
        str(implicit), [0, 1], settings=_settings(implicit))
    concatenate_and_normalize(
        str(explicit), [0, 1], settings=_settings(explicit),
        illumination_session=None)

    implicit_npz = next((implicit.parent / "masks").glob("*.npz"))
    explicit_npz = next((explicit.parent / "masks").glob("*.npz"))
    with np.load(implicit_npz) as left, np.load(explicit_npz) as right:
        np.testing.assert_array_equal(left["data"], right["data"])
        np.testing.assert_array_equal(left["filenames"], right["filenames"])

    # Measured by running the same one-field fixture against the parent of
    # the V1 integration commit (3c1ca30bd), not derived from this code path.
    # The archive container carries timestamps, so the stable artifact is the
    # decompressed float payload plus its exact filename manifest.
    golden = tmp_path / "golden" / "stack"
    golden.mkdir(parents=True)
    rows, cols = np.indices((8, 8))
    first = 20 + rows * 5 + cols * 3
    second = 40 + rows * 2 + cols * 6
    np.save(
        golden / "plate1_A01_F000.npy",
        np.stack([first, second], axis=-1).astype(np.uint16),
    )
    golden_settings = _settings(golden)
    golden_settings["batch_size"] = 1
    concatenate_and_normalize(
        str(golden), [0, 1], settings=golden_settings)
    golden_npz = next((golden.parent / "masks").glob("*.npz"))
    with np.load(golden_npz) as archive:
        payload = np.ascontiguousarray(archive["data"]).tobytes()
        assert hashlib.sha256(payload).hexdigest() == (
            "35d7acd5f7717f07d55d35be6128371957b8e7420a9dc8c57cbda9ac18b4751f"
        )
        assert archive["filenames"].tolist() == ["plate1_A01_F000.npy"]


def test_v1_enabled_rerun_replaces_the_whole_published_npz_set(tmp_path):
    """A shorter rerun cannot leave an old uncorrected batch for Cellpose."""
    from spacr.io import concatenate_and_normalize

    stack = tmp_path / "plate" / "stack"
    _stack(stack)
    masks = stack.parent / "masks"
    masks.mkdir()
    np.savez_compressed(
        masks / "stale_batch_norm.npz",
        data=np.zeros((1, 8, 8, 2), dtype=np.float32),
        filenames=np.asarray(["stale_field.npy"]),
    )
    old_masks = masks / "cell_mask_stack"
    old_masks.mkdir()
    np.save(
        old_masks / "stale_field.npy",
        np.full((8, 8), 9, dtype=np.uint16),
    )
    merged = stack.parent / "merged"
    merged.mkdir()
    np.save(
        merged / "stale_field.npy",
        np.full((8, 8, 2), 9, dtype=np.uint16),
    )
    settings = _settings(stack)
    settings["resume"] = True

    class InspectingSession(_Session):
        def finish(self, expected_fields):
            # The durable record must not become complete before outputs made
            # from the superseded pixels have been invalidated.
            assert not old_masks.exists()
            assert not merged.exists()
            assert settings["resume"] is False
            return super().finish(expected_fields)

    session = InspectingSession(stack, masks)

    concatenate_and_normalize(
        str(stack), [0, 1], settings=settings,
        illumination_session=session,
    )

    assert [path.name for path in masks.glob("*.npz")] == [
        "stack_0_norm.npz"]
    assert session.events[-1][0] == "finish"
    assert "stale_field" not in session.events[-1][1]


def test_v1_enabled_timelapse_refuses_an_ungrouped_source_frame(tmp_path):
    """Every source NPY must enter the correction and completion inventory."""
    import pytest

    from spacr.io import concatenate_and_normalize

    stack = tmp_path / "plate" / "stack"
    stack.mkdir(parents=True)
    values = np.arange(32, dtype=np.uint16).reshape(4, 4, 2)
    np.save(stack / "plate1_A01_1_1.npy", values)
    np.save(stack / "malformed.npy", values)
    session = _Session(stack, stack.parent / "masks")
    settings = _settings(stack)
    settings["timelapse"] = True

    with pytest.raises(ValueError, match="could not group every source NPY"):
        concatenate_and_normalize(
            str(stack), [0, 1], settings=settings,
            illumination_session=session,
        )

    assert session.events == []
    assert not list((stack.parent / "masks").glob("*.npz"))
    assert not list(stack.parent.glob(".spacr_v1_npz_*"))


def test_v1_atomic_save_failure_never_marks_or_finishes(
        tmp_path, monkeypatch):
    """A failed durable replacement leaves provenance incomplete."""
    import spacr.io as IO

    stack = tmp_path / "plate" / "stack"
    _stack(stack)
    session = _Session(stack, stack.parent / "masks")
    monkeypatch.setattr(
        IO, "_save_npz_atomic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("atomic replacement refused")),
    )

    import pytest
    with pytest.raises(OSError, match="atomic replacement refused"):
        IO.concatenate_and_normalize(
            str(stack), [0, 1], settings=_settings(stack),
            illumination_session=session,
        )

    assert [event[0] for event in session.events] == ["correct", "correct"]
    assert not list((stack.parent / "masks").glob("*.npz"))
    assert not list(stack.parent.glob(".spacr_v1_npz_*"))
    assert {
        path.stem: path.read_bytes() for path in stack.glob("*.npy")
    } == session.raw_before


def test_atomic_npz_replace_preserves_previous_file_and_cleans_temp(
        tmp_path, monkeypatch):
    """The atomic helper never leaves a partial final or orphaned sibling."""
    import pytest

    import spacr.io as IO

    output = tmp_path / "batch.npz"
    output.write_bytes(b"previous complete archive")
    monkeypatch.setattr(
        IO.os, "replace",
        lambda *args: (_ for _ in ()).throw(OSError("disk refused replace")),
    )

    with pytest.raises(OSError, match="disk refused replace"):
        IO._save_npz_atomic(output, data=np.ones((2, 2)))

    assert output.read_bytes() == b"previous complete archive"
    assert not list(tmp_path.glob(".spacr_npz_*.npz"))


def test_v1_archive_set_publication_rolls_back_both_sides(
        tmp_path, monkeypatch):
    """A mid-publication failure restores the old set and retains the new."""
    import pytest

    import spacr.io as IO

    output = tmp_path / "masks"
    staging = tmp_path / ".spacr_v1_npz_stage"
    output.mkdir()
    staging.mkdir()
    np.savez_compressed(
        output / "old.npz", data=np.zeros((1, 1)),
        filenames=np.asarray(["old.npy"]),
    )
    for name in ("a.npz", "b.npz"):
        np.savez_compressed(
            staging / name, data=np.ones((1, 1)),
            filenames=np.asarray([f"{name}.npy"]),
        )
    real_replace = IO.os.replace

    def fail_on_second_new(source, destination):
        if Path(source) == staging / "b.npz":
            raise OSError("second publication refused")
        return real_replace(source, destination)

    monkeypatch.setattr(IO.os, "replace", fail_on_second_new)
    with pytest.raises(OSError, match="second publication refused"):
        IO._publish_v1_normalized_archives(staging, output)

    assert [path.name for path in output.glob("*.npz")] == ["old.npz"]
    assert sorted(path.name for path in staging.glob("*.npz")) == [
        "a.npz", "b.npz"]
    assert not list(tmp_path.glob(".spacr_previous_v1_npz_*"))


def test_v1_archive_set_publication_refuses_an_empty_stage(tmp_path):
    """No successful run can replace real batches with an empty set."""
    import pytest

    import spacr.io as IO

    output = tmp_path / "masks"
    staging = tmp_path / "stage"
    output.mkdir()
    staging.mkdir()

    with pytest.raises(ValueError, match="empty V1 normalized archive set"):
        IO._publish_v1_normalized_archives(staging, output)


def test_v1_missing_input_cannot_finalize_a_partial_field_set(
        tmp_path, monkeypatch):
    """The intended field inventory is independent of successful loads."""
    import pytest

    import spacr.io as IO

    stack = tmp_path / "plate" / "stack"
    _stack(stack)
    session = _Session(stack, stack.parent / "masks")
    missing = sorted(stack.glob("*.npy"))[-1]
    real_load = IO.np.load

    def load(path, *args, **kwargs):
        if Path(path) == missing:
            raise OSError("field became unreadable")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(IO.np, "load", load)
    with pytest.raises(RuntimeError, match="incomplete illumination fields"):
        IO.concatenate_and_normalize(
            str(stack), [0, 1], settings=_settings(stack),
            illumination_session=session,
        )

    assert set(session.completed) != {
        "plate1_A01_F000", "plate1_A01_F001"}
    assert not list(stack.parent.glob(".spacr_v1_npz_*"))


def test_v1_resume_inventory_is_exact_and_rejects_missing_manifests(tmp_path):
    """Every archive contributes field ids; unnamed archives fail closed."""
    import pytest

    import spacr.io as IO

    masks = tmp_path / "masks"
    masks.mkdir()
    with pytest.raises(FileNotFoundError, match="none were found"):
        IO._normalized_npz_field_ids(masks)

    np.savez_compressed(
        masks / "b.npz", data=np.zeros((1, 1)),
        filenames=np.asarray(["f2.npy", "f1.npy"]),
    )
    np.savez_compressed(
        masks / "a.npz", data=np.zeros((1, 1)),
        filenames=np.asarray(["f1.npy", "f0.npy"]),
    )
    assert IO._normalized_npz_field_ids(masks) == ("f0", "f1", "f2")

    np.savez_compressed(masks / "broken.npz", data=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="no filenames manifest"):
        IO._normalized_npz_field_ids(masks)


def test_v1_timelapse_correction_failure_is_not_reported_as_success(tmp_path):
    """Enabled correction errors escape the legacy timelapse print handler."""
    import pytest

    import spacr.io as IO

    stack = tmp_path / "plate" / "stack"
    stack.mkdir(parents=True)
    for timepoint in (1, 2):
        np.save(
            stack / f"plate1_A01_1_{timepoint}.npy",
            np.full((4, 4, 1), 10 + timepoint, dtype=np.uint16),
        )

    class FailingSession(_Session):
        def correct(self, field_id, selected, context):
            self.events.append(("correct", str(field_id)))
            raise RuntimeError("timelapse correction failed")

    session = FailingSession(stack, stack.parent / "masks")
    settings = _settings(stack)
    settings.update({
        "timelapse": True,
        "channels": [0],
        "nucleus_channel": 0,
        "cell_channel": None,
    })

    with pytest.raises(RuntimeError, match="timelapse correction failed"):
        IO.concatenate_and_normalize(
            str(stack), [0], settings=settings,
            illumination_session=session,
        )

    assert session.events == [("correct", "plate1_A01_1_1")]
    assert not list((stack.parent / "masks").glob("*.npz"))


def test_v1_timelapse_finishes_the_exact_frame_inventory(tmp_path):
    """Every frame is corrected before the grouped archive is committed."""
    import spacr.io as IO

    stack = tmp_path / "plate" / "stack"
    stack.mkdir(parents=True)
    for timepoint in (1, 2):
        values = np.arange(32, dtype=np.uint16).reshape(4, 4, 2)
        np.save(
            stack / f"plate1_A01_1_{timepoint}.npy",
            values + timepoint,
        )
    session = _Session(stack, stack.parent / "masks")
    settings = _settings(stack)
    settings["timelapse"] = True

    IO.concatenate_and_normalize(
        str(stack), [0, 1], settings=settings,
        illumination_session=session,
    )

    assert session.events == [
        ("correct", "plate1_A01_1_1"),
        ("correct", "plate1_A01_1_2"),
        ("complete", "plate1_A01_1_1"),
        ("complete", "plate1_A01_1_2"),
        ("finish", ("plate1_A01_1_1", "plate1_A01_1_2")),
    ]


def test_known_vignette_reaches_cellpose_and_recovers_the_dim_corner(
        tmp_path, monkeypatch):
    """The V1 adapter changes which synthetic objects Cellpose can find."""
    import torch
    from scipy import ndimage

    import spacr.object as O
    from spacr.illumination import IlluminationField, IlluminationModel
    from spacr.io import preprocess_img_data

    rows, cols = np.indices((16, 16), dtype=np.float32)
    radius = np.sqrt((rows - 7.5) ** 2 + (cols - 7.5) ** 2)
    flat = np.clip(1.0 - 0.07 * radius, 0.30, 1.0).astype(np.float32)
    model = IlluminationModel(
        fields={
            "plate1": IlluminationField(
                plate="plate1", channels=(0,), flatfield=flat[None, ...],
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
    model_path = model.save(tmp_path / "known_vignette.npz")

    raw = np.rint(np.full(flat.shape, 10, dtype=np.float32) * flat)
    truth = np.zeros(flat.shape, dtype=bool)
    truth[1:4, 1:4] = True
    truth[7:10, 7:10] = True
    raw[truth] = np.rint(1000 * flat[truth])
    raw = raw.astype(np.uint16)[..., None]

    class ThresholdModel:
        def __init__(self, **kwargs):
            self.pretrained_model = kwargs.get("pretrained_model")

        def eval(self, x, **kwargs):
            masks = []
            flows = []
            for image in x:
                plane = np.asarray(image)[..., 0]
                mask, _count = ndimage.label(plane >= 0.6)
                masks.append(mask.astype(np.uint16))
                flows.append((
                    np.zeros((*plane.shape, 3), dtype=np.float32),
                    np.zeros((2, *plane.shape), dtype=np.float32),
                    np.zeros(plane.shape, dtype=np.float32),
                    None,
                ))
            return masks, flows, None

    monkeypatch.setattr(
        O, "cp_models", type("Models", (), {"CellposeModel": ThresholdModel}))
    monkeypatch.setattr(
        O.accelerator, "cellpose_kwargs",
        lambda: {"gpu": False, "device": torch.device("cpu")},
    )

    counts = []
    for name, enabled in (("off", False), ("on", True)):
        stack = tmp_path / name / "stack"
        stack.mkdir(parents=True)
        np.save(stack / "plate1_A01_F001.npy", raw)
        if enabled:
            # A complete mask from an uncorrected prior run must not trigger
            # either skip guard after corrected inputs are published.
            (stack.parent / "source.tif").write_bytes(b"existing stack wins")
            old_masks = stack.parent / "masks" / "cell_mask_stack"
            old_masks.mkdir(parents=True)
            np.save(
                old_masks / "plate1_A01_F001.npy",
                np.full(flat.shape, 9, dtype=np.uint16),
            )
        settings = _settings(stack)
        settings.update({
            "cell_channel": 0,
            "nucleus_channel": None,
            "channels": [0],
            "batch_size": 1,
            "illumination_correction": enabled,
            "illumination_model": str(model_path) if enabled else "",
            "illumination_qc": False,
            "verbose": False,
        })
        settings, src = preprocess_img_data(settings)
        O.generate_cellpose_masks_sam(
            str(Path(src) / "masks"), settings, "cell")
        mask = np.load(
            Path(src) / "masks" / "cell_mask_stack" /
            "plate1_A01_F001.npy")
        counts.append(int(mask.max()))
        np.testing.assert_array_equal(np.load(stack / "plate1_A01_F001.npy"),
                                      raw)

    assert counts == [1, 2]
