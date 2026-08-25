"""What the external-mask importer refuses, warns about, and reports.

Segmentation done outside spaCR arrives as a pile of folders, and almost
every interesting path through this module is a way that pile can be wrong:
a mask nothing pairs with, two masks for one field, a float image that would
lose precision in Measure's uint16 arrays. Each is driven with real files on
disk so the check being exercised is the real one.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest
import tifffile

from spacr import external_masks as em
from spacr.errors import ConfigurationError


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")
    return path


def _one_field(tmp_path, shape=(32, 32)):
    """One two-channel intensity field with a paired cell mask."""
    yy, xx = np.indices(shape)
    images = tmp_path / "images"
    cells = tmp_path / "cell_masks"
    _write(images / "fov001_C1.tif", (yy * 32 + xx).astype(np.uint16))
    _write(images / "fov001_C2.tif", ((xx * 17 + yy * 3) % 4096).astype(np.uint16))
    mask = np.zeros(shape, dtype=np.uint16)
    mask[3:-3, 3:-3] = 1
    _write(cells / "fov001_cell_mask.tif", mask)
    return images, cells


def _settings(tmp_path, *folders, **overrides):
    values = {
        "inputs": [group.to_dict() for group in em.detect_inputs(list(folders))],
        "dst": str(tmp_path / "project"),
        "layout": "flat",
    }
    values.update(overrides)
    return values


# --------------------------------------------------------------------------
# InputGroup


def test_an_already_reviewed_group_is_passed_through():
    """A round trip through the GUI hands back InputGroups, not mappings."""
    group = em.InputGroup(key="k", root="/r", paths=["/r/a.tif"], role="mask")

    assert em.InputGroup.from_value(group) is group


def test_an_input_that_is_neither_a_group_nor_a_mapping_is_refused():
    """A bare string in the reviewed list is a caller bug, and is named."""
    with pytest.raises(ConfigurationError, match="InputGroup or mapping"):
        em.InputGroup.from_value(["/some/path.tif"])


# --------------------------------------------------------------------------
# File collection and label detection


def test_a_single_supported_file_is_its_own_group(tmp_path):
    """Users drop one file as readily as a folder."""
    image = _write(tmp_path / "fov001_C1.tif", np.zeros((8, 8), np.uint16))

    groups = em.detect_inputs([image])

    assert [os.path.basename(p) for p in groups[0].paths] == ["fov001_C1.tif"]


def test_an_unsupported_file_contributes_nothing(tmp_path):
    """A README dropped alongside the images is not an image."""
    notes = tmp_path / "README.txt"
    notes.write_text("segmented in ilastik\n")

    assert em.detect_inputs([notes]) == []


def test_a_path_that_does_not_exist_contributes_nothing(tmp_path):
    """A stale path from a saved settings file must not raise."""
    assert em.detect_inputs([tmp_path / "gone"]) == []


def test_a_file_that_cannot_be_read_is_not_a_label_plane(tmp_path):
    """A truncated TIFF is reported as unsampled, not crashed on."""
    broken = tmp_path / "fov001.tif"
    broken.write_bytes(b"II*\x00 not really a tiff")

    likely, confidence, reason = em._label_likelihood(broken)

    assert not likely
    assert confidence == 0.0
    assert "could not sample pixels" in reason


def test_an_empty_export_is_not_a_label_plane(tmp_path):
    """A zero-size TIFF is a failed export, and says so with its shape."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tifffile.imwrite(tmp_path / "fov001.tif", np.zeros((0, 8), np.uint16))

    likely, confidence, reason = em._label_likelihood(tmp_path / "fov001.tif")

    assert not likely
    assert confidence == 0.0
    assert "not a 2-D label plane" in reason
    assert "(0, 8)" in reason


def test_a_float_image_is_not_a_label_plane(tmp_path):
    """Object IDs are integers; float pixels are intensities."""
    _write(tmp_path / "fov001.tif", np.zeros((8, 8), dtype=np.float32))

    likely, _confidence, reason = em._label_likelihood(tmp_path / "fov001.tif")

    assert not likely
    assert "integer label dtype" in reason


# --------------------------------------------------------------------------
# Coercing whatever the caller supplied into groups


def test_no_inputs_at_all_is_no_groups():
    """An unconfigured module has nothing selected, and does not raise."""
    assert em._coerce_groups(None, recursive=True) == []
    assert em._coerce_groups([], recursive=True) == []


def test_a_single_folder_may_be_given_without_a_list(tmp_path):
    """`inputs="/data/masks"` is what a hand-written script says."""
    images, _cells = _one_field(tmp_path)

    groups = em._coerce_groups(str(images), recursive=True)

    assert sum(len(group.paths) for group in groups) == 2


def test_several_folders_may_be_given_as_plain_paths(tmp_path):
    """A list of paths is detected; a list of mappings is taken as reviewed."""
    images, cells = _one_field(tmp_path)

    groups = em._coerce_groups([str(images), str(cells)], recursive=True)

    assert {group.role for group in groups} == {"image", "mask"}


# --------------------------------------------------------------------------
# Pairing and the preview a user reads before anything is written


def _two_fields(tmp_path):
    """Two two-channel fields, each with a paired cell mask."""
    yy, xx = np.indices((32, 32))
    images = tmp_path / "images"
    cells = tmp_path / "cell_masks"
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[3:29, 3:29] = 1
    for name in ("fov001", "fov002"):
        _write(images / f"{name}_C1.tif", (yy * 32 + xx).astype(np.uint16))
        _write(images / f"{name}_C2.tif",
               ((xx * 17 + yy * 3) % 4096).astype(np.uint16))
        _write(cells / f"{name}_cell_mask.tif", mask)
    return images, cells


def test_a_mask_no_field_claims_is_a_warning_not_a_failure(tmp_path):
    """One stray file in a mask folder must not block the whole import."""
    images, cells = _two_fields(tmp_path)
    _write(cells / "zzz999_cell_mask.tif",
           np.zeros((32, 32), dtype=np.uint16))

    plan = em.plan_external_masks(_settings(tmp_path, images, cells))

    assert plan.ok, plan.summary()
    assert any("no intensity field has the same" in message
               for message in plan.warnings)
    assert sorted(plan.masks["cell"]) == plan.stems


def test_two_masks_for_one_field_is_a_blocking_error(tmp_path):
    """spaCR will not silently pick one of two cell masks for a field."""
    images, cells = _two_fields(tmp_path)
    _write(cells / "fov001_cell_labels.tif",
           np.zeros((32, 32), dtype=np.uint16))

    plan = em.plan_external_masks(_settings(tmp_path, images, cells))

    assert not plan.ok
    assert any("both map to" in message for message in plan.errors)


def test_the_preview_lists_channels_masks_warnings_and_problems(tmp_path):
    """The summary is what the user reads before agreeing to the import."""
    images, cells = _two_fields(tmp_path)
    _write(cells / "zzz999_cell_mask.tif",
           np.zeros((32, 32), dtype=np.uint16))
    _write(cells / "fov001_cell_labels.tif",
           np.zeros((32, 32), dtype=np.uint16))

    text = em.plan_external_masks(
        _settings(tmp_path, images, cells)).summary()

    assert "nothing written" in text
    assert "intensity channels: 2" in text
    assert "mask types: cell" in text
    assert "cell: 2 paired mask(s), merged plane 2" in text
    assert "Warnings:" in text
    assert "Blocking problems:" in text
    assert "both map to" in text


def test_a_preview_with_no_masks_at_all_says_so(tmp_path):
    """'mask types: none' is more use than an empty line."""
    images, _cells = _one_field(tmp_path)

    text = em.plan_external_masks(_settings(tmp_path, images)).summary()

    assert "mask types: none" in text
    assert "No label-mask group is selected." in text


def test_intensity_only_inputs_are_refused(tmp_path):
    """Without masks there is nothing external to import."""
    images, _cells = _one_field(tmp_path)

    plan = em.plan_external_masks(_settings(tmp_path, images))

    assert "No label-mask group is selected." in plan.errors


def test_mask_only_inputs_are_refused(tmp_path):
    """Masks with no intensity data cannot be measured."""
    _images, cells = _one_field(tmp_path)

    plan = em.plan_external_masks(_settings(tmp_path, cells))

    assert "No intensity-image group is selected." in plan.errors


def test_an_unknown_role_is_named(tmp_path):
    """A settings file edited by hand must not quietly ignore a typo."""
    images, cells = _one_field(tmp_path)
    reviewed = [group.to_dict()
                for group in em.detect_inputs([images, cells])]
    reviewed[0]["role"] = "intensity"

    plan = em.plan_external_masks(_settings(
        tmp_path, images, cells, inputs=reviewed))

    assert any("Unknown input roles: intensity." in message
               for message in plan.errors)


def test_an_unknown_layout_is_named_and_falls_back_to_auto(tmp_path):
    """The import still previews, so the user can see what auto detected."""
    images, cells = _one_field(tmp_path)

    plan = em.plan_external_masks(
        _settings(tmp_path, images, cells, layout="hexagonal"))

    assert any("Unknown input layout 'hexagonal'" in message
               for message in plan.errors)
    assert plan.n_channels == 2


def test_an_image_group_with_nothing_selected_has_no_channels(tmp_path):
    """Deselecting every intensity file leaves nothing to measure."""
    images, cells = _one_field(tmp_path)
    reviewed = [group.to_dict()
                for group in em.detect_inputs([images, cells])]
    for entry in reviewed:
        if entry["role"] == "image":
            entry["paths"] = []

    plan = em.plan_external_masks(_settings(
        tmp_path, images, cells, inputs=reviewed))

    assert plan.n_channels == 0
    assert "No readable intensity channels were detected." in plan.errors


def test_an_existing_measurements_database_is_never_replaced(tmp_path):
    """Re-running into the same folder would destroy earlier measurements."""
    images, cells = _one_field(tmp_path)
    existing = tmp_path / "project" / "measurements" / "measurements.db"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"")

    plan = em.plan_external_masks(_settings(tmp_path, images, cells))

    assert not plan.ok
    assert any("already has measurements/measurements.db" in message
               for message in plan.errors)


def test_an_image_group_contributes_no_masks(tmp_path):
    """Pairing looks only at mask groups, whatever it is handed."""
    images, cells = _one_field(tmp_path)
    groups = em.detect_inputs([images, cells])
    image_plan = em.cv.plan([
        source for group in groups if group.role == "image"
        for source in em._scan_group(group, layout="flat")
    ])

    only_images = [group for group in groups if group.role == "image"]
    by_type, errors, warnings = em._pair_masks(
        image_plan, only_images, layout="flat")

    assert by_type == {}
    assert errors == []
    assert warnings == []


def test_a_database_that_does_not_exist_has_no_tables(tmp_path):
    """Measure may have failed before writing anything; that is not a crash."""
    assert em._tables(str(tmp_path / "measurements.db")) == []


# --------------------------------------------------------------------------
# What the writer refuses, once the user has agreed to the import


def _measure_writing(tables):
    """A stand-in for Measure that writes the tables it is told to."""
    def _measure_crop(measure_settings):
        project = os.path.dirname(measure_settings["src"])
        db = os.path.join(project, "measurements", "measurements.db")
        os.makedirs(os.path.dirname(db), exist_ok=True)
        with sqlite3.connect(db) as connection:
            for table in tables:
                connection.execute(
                    f'CREATE TABLE "{table}" (object_label INTEGER)')
        os.makedirs(os.path.join(project, "data", "plate1", "cell_png"),
                    exist_ok=True)

    return _measure_crop


def test_a_refused_plan_is_never_written(tmp_path):
    """`run` repeats every blocking problem instead of writing half a project."""
    _images, cells = _one_field(tmp_path)
    settings = _settings(tmp_path, cells)
    plan = em.plan_external_masks(settings)

    with pytest.raises(ConfigurationError) as failure:
        em.run_external_masks(plan, settings)

    assert "nothing was written" in str(failure.value)
    assert "No intensity-image group is selected." in str(failure.value)
    assert not os.path.exists(plan.destination)


def test_only_the_fields_the_plan_lists_are_written(tmp_path, monkeypatch):
    """A field dropped from the plan is skipped, not half-imported."""
    images, cells = _two_fields(tmp_path)
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)
    dropped = sorted(plan.masks["cell"])[1]
    del plan.masks["cell"][dropped]
    monkeypatch.setattr("spacr.measure.measure_crop",
                        _measure_writing(("cell", "cytoplasm", "png_list")))

    result = em.run_external_masks(plan, settings)

    assert [os.path.basename(path) for path in result.merged] == [
        f"{plan.stems[0]}.npy"]
    assert not os.path.exists(
        os.path.join(result.destination, "merged", f"{dropped}.npy"))


def test_a_field_missing_a_channel_is_refused(tmp_path, monkeypatch):
    """A one-channel field in a two-channel import would misalign every plane."""
    images, cells = _two_fields(tmp_path)
    (images / "fov002_C2.tif").unlink()
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)

    with pytest.raises(ConfigurationError, match="expected 2 channels, found 1"):
        em.run_external_masks(plan, settings)


def test_a_conversion_that_dropped_a_file_is_refused(tmp_path, monkeypatch):
    """If conversion silently lost a channel, the import stops there."""
    images, cells = _one_field(tmp_path)
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)
    real_convert = em.cv.convert

    def _lossy_convert(image_plan, destination, **kwargs):
        conversion = real_convert(image_plan, destination, **kwargs)
        conversion.written[:] = conversion.written[:1]
        conversion.existing[:] = []
        return conversion

    monkeypatch.setattr(em.cv, "convert", _lossy_convert)

    with pytest.raises(ConfigurationError,
                       match="converted intensity image is missing"):
        em.run_external_masks(plan, settings)


def test_channels_of_different_sizes_are_refused(tmp_path):
    """Two channels of one field must describe the same pixels."""
    images, cells = _one_field(tmp_path)
    grain = (np.arange(16 * 16).reshape(16, 16) * 251 % 65535)
    _write(images / "fov001_C3.tif", grain.astype(np.uint16))
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)

    with pytest.raises(ConfigurationError, match="inconsistent shapes"):
        em.run_external_masks(plan, settings)


def test_a_mask_of_the_wrong_size_is_refused(tmp_path):
    """A mask segmented from a downscaled copy does not describe this field."""
    images, cells = _one_field(tmp_path)
    _write(cells / "fov001_cell_mask.tif",
           np.zeros((16, 16), dtype=np.uint16))
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)

    with pytest.raises(ConfigurationError,
                       match="does not match intensity shape"):
        em.run_external_masks(plan, settings)


def test_a_mask_with_negative_labels_is_refused(tmp_path):
    """Some tools write -1 for 'unlabelled'; that is not an object ID."""
    images, cells = _one_field(tmp_path)
    mask = np.zeros((32, 32), dtype=np.int16)
    mask[3:10, 3:10] = -1
    mask[15:20, 15:20] = 1
    _write(cells / "fov001_cell_mask.tif", mask)
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)

    with pytest.raises(ConfigurationError, match="negative IDs"):
        em.run_external_masks(plan, settings)


def test_a_label_id_above_65535_is_refused(tmp_path):
    """Truncating a label ID would merge two objects into one."""
    images, cells = _one_field(tmp_path)
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[3:10, 3:10] = 70000
    _write(cells / "fov001_cell_mask.tif", mask)
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)

    with pytest.raises(ConfigurationError, match="70000 exceeds the maximum"):
        em.run_external_masks(plan, settings)


def test_intensity_data_with_a_nan_is_refused(tmp_path):
    """NaN casts to an arbitrary integer; the import stops instead."""
    images, cells = _one_field(tmp_path)
    values = np.indices((32, 32))[0].astype(np.float32)
    values[0, 0] = np.nan
    _write(images / "fov001_C2.tif", values)
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)

    with pytest.raises(ConfigurationError, match="NaN or infinity"):
        em.run_external_masks(plan, settings)


def test_fractional_intensity_data_is_refused_with_advice(tmp_path):
    """A normalised 0-1 float image would become an array of zeros."""
    images, cells = _one_field(tmp_path)
    yy, _xx = np.indices((32, 32))
    _write(images / "fov001_C2.tif", (yy / 32.0).astype(np.float32))
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)

    with pytest.raises(ConfigurationError, match="lose precision"):
        em.run_external_masks(plan, settings)


def test_intensity_data_outside_the_uint16_range_is_refused(tmp_path):
    """A 32-bit camera export must be rescaled before Measure sees it."""
    images, cells = _one_field(tmp_path)
    yy, xx = np.indices((32, 32))
    values = ((yy * 32 + xx) * 70).astype(np.int32)
    _write(images / "fov001_C2.tif", values)
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)

    with pytest.raises(ConfigurationError, match="must fit the Measure uint16"):
        em.run_external_masks(plan, settings)


def test_a_single_crop_mode_may_be_written_as_a_string(tmp_path, monkeypatch):
    """`crop_mode: cell` in a hand-written settings file is one crop mode."""
    images, cells = _one_field(tmp_path)
    settings = _settings(tmp_path, images, cells, crop_mode="cytoplasm")
    plan = em.plan_external_masks(settings)
    received = {}

    def _record(measure_settings):
        received.update(measure_settings)
        _measure_writing(("cell", "cytoplasm", "png_list"))(measure_settings)

    monkeypatch.setattr("spacr.measure.measure_crop", _record)
    em.run_external_masks(plan, settings)

    assert received["crop_mode"] == ["cytoplasm"]


def test_measure_finishing_without_its_table_is_an_error(tmp_path,
                                                        monkeypatch):
    """An empty measurements.db is a failed import, not a finished one."""
    images, cells = _one_field(tmp_path)
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)
    monkeypatch.setattr("spacr.measure.measure_crop",
                        _measure_writing(("png_list",)))

    with pytest.raises(ConfigurationError,
                       match="without required output table"):
        em.run_external_masks(plan, settings)


def test_the_result_says_what_was_written(tmp_path, monkeypatch):
    """One line a script can log after an import."""
    images, cells = _one_field(tmp_path)
    settings = _settings(tmp_path, images, cells)
    plan = em.plan_external_masks(settings)
    monkeypatch.setattr("spacr.measure.measure_crop",
                        _measure_writing(("cell", "cytoplasm", "png_list")))

    text = em.run_external_masks(plan, settings).summary()

    assert text.startswith("Prepared 1 field(s) in ")
    assert "cell, cytoplasm, png_list" in text
    assert "Annotation crops: " in text


# --------------------------------------------------------------------------
# The scripted entry point


def test_a_preview_only_run_prints_the_plan_and_writes_nothing(tmp_path,
                                                               capsys):
    """`preview_only` is the dry run a user asks for before committing."""
    images, cells = _one_field(tmp_path)
    settings = _settings(tmp_path, images, cells, preview_only=True)

    plan = em.prepare_external_masks(settings)

    assert isinstance(plan, em.ExternalMaskPlan)
    assert "nothing written" in capsys.readouterr().out
    assert not os.path.exists(plan.destination)


def test_a_full_run_prints_both_summaries(tmp_path, monkeypatch, capsys):
    """The preview and then what was actually written."""
    images, cells = _one_field(tmp_path)
    settings = _settings(tmp_path, images, cells)
    monkeypatch.setattr("spacr.measure.measure_crop",
                        _measure_writing(("cell", "cytoplasm", "png_list")))

    result = em.prepare_external_masks(settings)

    printed = capsys.readouterr().out
    assert isinstance(result, em.ExternalMaskResult)
    assert "nothing written" in printed
    assert "Prepared 1 field(s)" in printed
    assert os.path.isfile(result.db_path)


def test_a_mask_beside_its_images_pairs_on_the_normalised_name(tmp_path):
    """Masks exported next to the images share a plate, so the name decides."""
    yy, xx = np.indices((32, 32))
    plate = tmp_path / "plateA"
    _write(plate / "fov001_C1.tif", (yy * 32 + xx).astype(np.uint16))
    _write(plate / "fov002_C1.tif", ((xx * 17 + yy * 3) % 4096).astype(np.uint16))
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[4:28, 4:28] = 1
    _write(plate / "fov001_cell_mask.tif", mask)
    _write(plate / "fov002_cell_mask.tif", mask)

    plan = em.plan_external_masks(_settings(tmp_path, plate))

    assert plan.ok, plan.summary()
    assert {match.match for match in plan.masks["cell"].values()} == {
        "normalised"}


def test_a_field_with_no_mask_at_all_is_a_blocking_error(tmp_path):
    """Importing a field whose objects were never segmented measures nothing."""
    images, cells = _two_fields(tmp_path)
    (cells / "fov002_cell_mask.tif").unlink()

    plan = em.plan_external_masks(_settings(tmp_path, images, cells))

    assert not plan.ok
    assert any("intensity field(s) have no matching mask" in message
               for message in plan.errors)
