"""Conversion guards preserve provenance and recover interrupted deletion safely."""

import json
from pathlib import Path

import numpy as np
import pytest

from spacr import image_store, ome_zarr_convert
from spacr.crops import MERGED_LAYOUT_SIDECAR, _write_measured_masks
from spacr.io import _save_array_atomic


def _source(root, *, dtype=np.uint16):
    """Create two tiny fields with sparse cell labels and explicit native axes."""
    merged = root / "merged"
    merged.mkdir(parents=True)
    (merged / MERGED_LAYOUT_SIDECAR).write_text(json.dumps({
        "version": 1, "intensity_channels": [2], "mask_plane_order": ["cell"],
        "mask_dims": {"cell": 1}}))
    array = np.zeros((4, 5, 2), dtype=dtype)
    array[..., 0] = np.arange(20).reshape(4, 5)
    array[1:3, 1:4, 1] = 23
    for number in (1, 2):
        _save_array_atomic(str(merged / f"p_A01_{number}.npy"), array)
    return root


def test_effective_edits_and_original_embedded_masks_both_round_trip(tmp_path):
    """Measured labels stay authoritative while legacy embedded planes remain reversible."""
    src = _source(tmp_path / "source")
    field = src / "merged" / "p_A01_1.npy"
    raw = image_store.read_image(field)
    edited = raw[..., 1].copy()
    edited[edited == 23] = 71
    _write_measured_masks(str(field), {1: edited})
    report = ome_zarr_convert.convert_project(src, what=("merged", "masks"))
    assert report["status"] == "verified"
    store = Path(report["destination"]) / "merged" / "p.ome.zarr"
    np.testing.assert_array_equal(image_store.read_labels(store, "cell", key=("A01", 1)), edited)
    np.testing.assert_array_equal(image_store.read_labels(store, "legacy_cell", key=("A01", 1)), raw[..., 1])
    np.testing.assert_array_equal(image_store.read_image(field), raw)


def test_stale_measured_mask_provenance_blocks_before_writing(tmp_path):
    """A stale sidecar cannot silently fall back to unmeasured embedded labels."""
    src = _source(tmp_path / "source")
    field = src / "merged" / "p_A01_1.npy"
    raw = image_store.read_image(field)
    edited = raw[..., 1].copy()
    edited[edited == 23] = 71
    _write_measured_masks(str(field), {1: edited})
    raw[0, 0, 1] = 3
    _save_array_atomic(str(field), raw)
    dst = tmp_path / "dest"
    plan = ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"), dry_run=True)
    assert plan["status"] == "blocked"
    assert any("stale" in problem for problem in plan["problems"])
    with pytest.raises(ValueError, match="stale"):
        ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"))
    assert not dst.exists()


def test_nan_payload_and_signed_zero_are_preserved_bit_exactly(tmp_path):
    """Floating intensities retain IEEE payload bits through native and legacy exports."""
    src = _source(tmp_path / "source", dtype=np.float32)
    field = src / "merged" / "p_A01_1.npy"
    raw = image_store.read_image(field)
    raw.view(np.uint32)[0, :3, 0] = [0x7FC00001, 0x7FC00002, 0x80000000]
    _save_array_atomic(str(field), raw)
    report = ome_zarr_convert.convert_project(src, what=("merged", "masks"))
    actual = image_store.read_image(Path(report["destination"]) / "merged" / "p.ome.zarr", key=("A01", 1))
    assert actual.tobytes() == np.moveaxis(raw[..., :1], -1, 0).tobytes()


@pytest.mark.parametrize("hazard", ["symlink", "case_collision", "reserved", "traversal", "overlap"])
def test_unsafe_paths_are_refused_without_output(tmp_path, hazard):
    """Discovery rejects aliases and paths that cannot safely move across platforms."""
    src = _source(tmp_path / "source")
    dst = tmp_path / "dest"
    if hazard == "symlink":
        (src / "alias").symlink_to(src / "merged", target_is_directory=True)
    elif hazard == "case_collision":
        (src / "ABC.txt").write_text("one")
        (src / "abc.txt").write_text("two")
    elif hazard == "reserved":
        (src / "CON.txt").write_text("reserved")
    elif hazard == "traversal":
        src = src / ".." / "source"
    else:
        dst = src / "destination"
    with pytest.raises(ValueError):
        ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"), dry_run=True)
    assert not dst.exists()


def test_changed_asset_selection_cannot_reuse_a_destination(tmp_path):
    """The immutable plan prevents relabeling a partial conversion as another selection."""
    src = _source(tmp_path / "source")
    dst = tmp_path / "dest"
    ome_zarr_convert.convert_project(src, dst, what=("merged",))
    with pytest.raises(ValueError, match="incompatible"):
        ome_zarr_convert.convert_project(src, dst, what=("masks",))


def test_deletion_resumes_after_one_original_was_removed(tmp_path, monkeypatch):
    """Missing sources are accepted only when the deletion journal explains them."""
    src = _source(tmp_path / "source")
    dst = tmp_path / "dest"
    original = Path.unlink
    removed = []

    def interrupted_unlink(path, *args, **kwargs):
        """Interrupt the second planned source unlink, leaving a durable pending entry."""
        if src in path.parents and path.suffix == ".npy":
            if removed:
                raise RuntimeError("interrupted deletion")
            removed.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted_unlink)
    with pytest.raises(RuntimeError, match="interrupted deletion"):
        ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"), delete_old=True)
    assert len(removed) == 1 and not removed[0].exists()
    monkeypatch.setattr(Path, "unlink", original)
    report = ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"), delete_old=True)
    assert set(report["deletions"].values()) == {"deleted"}
    assert not list((src / "merged").glob("*.npy"))
    assert (src / "merged" / MERGED_LAYOUT_SIDECAR).exists()
    assert ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"))["status"] == "verified"


def test_source_change_just_before_deletion_preserves_every_original(tmp_path, monkeypatch):
    """Verification cannot authorize deletion after the source changes again."""
    src = _source(tmp_path / "source")
    dst = tmp_path / "dest"
    delete = ome_zarr_convert._delete_sources

    def change_before_delete(source, destination, ledger):
        """Mutate the first eligible source between verification and unlinking."""
        path = source / "merged" / "p_A01_1.npy"
        array = image_store.read_image(path)
        array[0, 0, 0] += 1
        _save_array_atomic(str(path), array)
        return delete(source, destination, ledger)

    monkeypatch.setattr(ome_zarr_convert, "_delete_sources", change_before_delete)
    with pytest.raises(ValueError, match="changed before deletion"):
        ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"), delete_old=True)
    assert len(list((src / "merged").glob("*.npy"))) == 2


def test_changed_axis_metadata_fails_verification(tmp_path):
    """Correct bytes with a wrong axis interpretation are not a verified image."""
    src = _source(tmp_path / "source")
    dst = tmp_path / "dest"
    ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"))
    metadata_path = dst / "merged" / "p.ome.zarr" / "A" / "1" / "1" / "zarr.json"
    metadata = json.loads(metadata_path.read_text())
    axes = metadata["attributes"]["ome"]["multiscales"][0]["axes"]
    axes[1]["name"], axes[2]["name"] = axes[2]["name"], axes[1]["name"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata verification"):
        ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"))


def test_saved_channel_selection_and_pixel_calibration_are_preserved(tmp_path):
    """Known calibration and acquisition channel labels survive the merged mapping."""
    import pandas as pd
    from spacr.ome_zarr import read_ome_zarr
    from spacr.tabular import write_database

    src = _source(tmp_path / "source")
    db = src / "measurements" / "measurements.db"
    frame = pd.DataFrame({"setting_key": ["channel_names", "pixel_size_um"],
                          "setting_value": ["['unused0', 'unused1', 'DNA']", "0.325"]})
    write_database(frame, str(db), "settings", canonicalise=False)
    before = db.read_bytes()
    report = ome_zarr_convert.convert_project(src, what=("merged", "masks"))
    dst = Path(report["destination"])
    metadata = read_ome_zarr(dst / "merged" / "p.ome.zarr" / "A" / "1" / "1")
    assert metadata.channel_names == ("DNA",)
    assert metadata.spacing.scale == (0.325, 0.325)
    assert metadata.spacing.units == "um"
    assert db.read_bytes() == before
    assert (dst / "measurements" / "measurements.db").read_bytes() == before


def test_conflicting_saved_calibration_is_a_planning_blocker(tmp_path):
    """Different recorded image scales require provenance resolution before output."""
    from spacr.tabular import write_table
    import pandas as pd

    src = _source(tmp_path / "source")
    settings = src / "settings"
    settings.mkdir()
    for name, value in (("gen_mask_settings.csv", "0.325"), ("measure_crop_settings.csv", "0.5")):
        write_table(pd.DataFrame({"Key": ["pixel_size_um"], "Value": [value]}), settings / name)
    dst = tmp_path / "dest"
    report = ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"), dry_run=True)
    assert report["status"] == "blocked"
    assert any("calibration" in error and "conflict" in error for error in report["problems"])
    assert not dst.exists()


@pytest.mark.parametrize("empty_wal", [False, True])
def test_immutable_source_reads_keep_empty_wal_and_leftover_shm_unchanged(tmp_path, empty_wal):
    """Quiescent WAL databases need no source-side cleanup to read their settings."""
    import pandas as pd
    from contextlib import closing
    from spacr.database_concurrency import connect
    from spacr.tabular import write_database

    src = _source(tmp_path / "source")
    db = src / "measurements" / "measurements.db"
    write_database(pd.DataFrame({"setting_key": ["pixel_size_um", "channel_names"],
                                 "setting_value": ["None", "None"]}), db, "settings",
                   canonicalise=False)
    with closing(connect(db, journal_mode="WAL")) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shm = db.with_name(db.name + "-shm")
    shm.write_bytes(b"leftover shared-memory sidecar")
    if empty_wal:
        db.with_name(db.name + "-wal").write_bytes(b"")
    before = {str(p.relative_to(src)): p.read_bytes() for p in src.rglob("*") if p.is_file()}

    report = ome_zarr_convert.convert_project(src, what=("merged", "masks"))

    assert report["status"] == "verified"
    assert {str(p.relative_to(src)): p.read_bytes()
            for p in src.rglob("*") if p.is_file()} == before


def test_nonempty_wal_is_still_refused_before_immutable_reads(tmp_path):
    """Ignoring uncheckpointed source transactions must never become a conversion path."""
    src = _source(tmp_path / "source")
    folder = src / "measurements"
    folder.mkdir()
    (folder / "measurements.db-wal").write_bytes(b"uncheckpointed transaction")
    dst = tmp_path / "dest"
    with pytest.raises(ValueError, match="checkpoint"):
        ome_zarr_convert.convert_project(src, dst, dry_run=True)
    assert not dst.exists()


def test_unset_csv_metadata_uses_named_channels_and_no_physical_scale(tmp_path):
    """Blank or null saved settings mean unknown calibration, not malformed metadata."""
    import pandas as pd
    from spacr.ome_zarr import read_ome_zarr
    from spacr.tabular import write_table

    src = _source(tmp_path / "source")
    settings = src / "settings"
    settings.mkdir()
    write_table(pd.DataFrame({"Key": ["pixel_size_um", "channel_names"],
                              "Value": ["", "null"]}),
                settings / "measure_crop_settings.csv")

    report = ome_zarr_convert.convert_project(src, what=("merged", "masks"))

    assert report["status"] == "verified"
    assert report["notes"] == ["Physical pixel size is unknown; unitless coordinates are used."]
    metadata = read_ome_zarr(Path(report["destination"]) / "merged" / "p.ome.zarr" / "A" / "1" / "1")
    assert metadata.channel_names == ("channel_0",)
    assert metadata.spacing.units == "px"
    assert not metadata.units_declared
    assert all(field["pixel_size_um"] is None for field in report["fields"])


def test_missing_plane_layout_reports_one_actionable_blocker_for_all_fields(tmp_path):
    """An absent common sidecar must not cascade into per-field None errors."""
    src = _source(tmp_path / "source")
    (src / "merged" / MERGED_LAYOUT_SIDECAR).unlink()
    dst = tmp_path / "dest"

    report = ome_zarr_convert.convert_project(src, dst, dry_run=True)

    assert report["status"] == "blocked"
    assert report["problems"] == [
        "Merged conversion needs .spacr_plane_layout.json; recover the saved channel layout first."]
    assert not dst.exists()


def test_non_native_byte_order_is_reconstructed_in_legacy_export(tmp_path):
    """Zarr native scalar storage retains the original NPY byte-order contract."""
    src = _source(tmp_path / "source", dtype=np.dtype('>u2'))
    report = ome_zarr_convert.convert_project(src, what=("merged", "masks"))
    assert report["status"] == "verified"
    entry = report["fields"][0]
    assert entry["original"]["dtype"] == ">u2"
    pixels = image_store.read_image(Path(report["destination"]) / "merged" / "p.ome.zarr", key=("A01", 1))
    raw = image_store.read_image(src / "merged" / "p_A01_1.npy")
    assert pixels.dtype.kind == raw.dtype.kind and pixels.dtype.itemsize == raw.dtype.itemsize
    np.testing.assert_array_equal(pixels, np.moveaxis(raw[..., :1], -1, 0))


def test_equal_mask_ids_with_different_legacy_dtypes_are_both_preserved(tmp_path):
    """Separate label dtype and embedded merged dtype can coexist without a false conflict."""
    from spacr.tiff_io import write_tiff

    src = _source(tmp_path / "source")
    folder = src / "cell_mask"
    folder.mkdir()
    raw = image_store.read_image(src / "merged" / "p_A01_1.npy")
    write_tiff(folder / "p_A01_1.tif", raw[..., 1].astype(np.uint32))
    report = ome_zarr_convert.convert_project(src, what=("merged", "masks"))
    store = Path(report["destination"]) / "merged" / "p.ome.zarr"
    assert report["status"] == "verified"
    assert image_store.read_labels(store, "cell", key=("A01", 1)).dtype == np.uint32
    assert image_store.read_labels(store, "legacy_cell", key=("A01", 1)).dtype == np.uint16


def test_destination_plan_is_rechecked_after_the_writer_lock(tmp_path, monkeypatch):
    """A concurrent conversion cannot make a stale local plan overwrite its ledger."""
    from contextlib import contextmanager

    src = _source(tmp_path / "source")
    dst = tmp_path / "dest"
    lock = ome_zarr_convert._lock

    @contextmanager
    def competing_plan(destination):
        """Model another completed planner before the current writer obtains its lock."""
        with lock(destination):
            path = destination / "conversion.json"
            ledger = json.loads(path.read_text())
            ledger["plan"]["what"] = ["masks"]
            ome_zarr_convert._atomic_json(path, ledger)
            yield

    def refuse_write(*args, **kwargs):
        """Prove a stale plan is rejected before any array mutation."""
        raise AssertionError("stale plan reached a writer")

    monkeypatch.setattr(ome_zarr_convert, "_lock", competing_plan)
    monkeypatch.setattr(image_store.PlateWriter, "write_field", refuse_write)
    with pytest.raises(ValueError, match="(?i)plan.*changed|changed.*plan"):
        ome_zarr_convert.convert_project(src, dst, what=("merged", "masks"))
    assert not (dst / "merged").exists()
    assert len(list((src / "merged").glob("*.npy"))) == 2
