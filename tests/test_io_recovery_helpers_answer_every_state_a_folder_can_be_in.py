"""Item 288: spaCR's I/O recovery helpers against every state a folder can be in.

``spacr.io`` resumes a Mask run over what an earlier one left behind:
unfinished atomic writes, damaged archives, archives named twice, raw
images moved into ``orig/``, crop archives whose format marker is wrong.
Each helper below has one job and answers it for the awkward case as well
as the ordinary one -- which is what these tests pin, one helper at a time,
by what it leaves on disk or says.
"""
from __future__ import annotations

import io as stdio
import json
import os
import tarfile

import numpy as np
import pytest
import tifffile

from spacr import crops
from spacr import io as sio
from spacr.classification_pixels import DECLARED_UINT8, STORED_PIL


# ---------------------------------------------------------------------------
# crop archives and their format markers
# ---------------------------------------------------------------------------

def _tar(path, members):
    with tarfile.open(path, "w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, stdio.BytesIO(data))
    return str(path)


def _marker(fmt, **extra):
    return json.dumps({"spacr_crop_format": fmt, **extra}).encode()


SIDECAR = crops.CROP_FORMAT_SIDECAR


def test_a_marker_naming_an_unknown_format_is_refused_under_the_declared_policy(
        tmp_path):
    path = _tar(tmp_path / "c.tar", [(f"cells/{SIDECAR}", _marker(99)),
                                     ("cells/a.png", b"x")])
    with pytest.raises(ValueError, match="Invalid tar crop marker"):
        sio.TarImageDataset(path, crop_loading_policy=DECLARED_UINT8)
    legacy = sio.TarImageDataset(path, crop_loading_policy=STORED_PIL)
    assert [m.name for m in legacy.members] == ["cells/a.png"]


def test_two_markers_for_one_folder_are_refused(tmp_path):
    path = _tar(tmp_path / "c.tar", [(f"cells/{SIDECAR}", _marker(3)),
                                     (f"cells/{SIDECAR}", _marker(3)),
                                     ("cells/a.png", b"x")])
    with pytest.raises(ValueError, match="duplicate crop format marker"):
        sio.TarImageDataset(path)


def test_migration_state_decides_each_members_format(tmp_path):
    suffix = crops.CROP_MIGRATION_SUFFIX
    marker = _marker(3, migration={"from": 1, "unconverted": ["b.png"],
                                   "done_through": "zzz"})
    path = _tar(tmp_path / "c.tar", [
        (f"plate/{SIDECAR}", marker),
        ("plate/cells/a.png", b"x"),
        ("plate/cells/a.png" + suffix, b"x"),
        ("plate/cells/b.png", b"x"),
        ("plate/cells/c.png", b"x"),
    ])
    dataset = sio.TarImageDataset(path)
    assert [m.name for m in dataset.members] == [
        "plate/cells/a.png", "plate/cells/b.png", "plate/cells/c.png"]
    assert dataset._member_crop_format("plate/cells/a.png") == 1, (
        "a member with a migration twin was not converted")
    assert dataset._member_crop_format("plate/cells/b.png") == 1, (
        "a member listed as unconverted keeps the source format")
    assert dataset._member_crop_format("plate/cells/c.png") == 3
    assert dataset._member_crop_format("elsewhere/d.png") == \
        crops.CROP_FORMAT_LEGACY_BGR


# ---------------------------------------------------------------------------
# listing raw images and channel folders
# ---------------------------------------------------------------------------

def test_raw_images_are_listed_by_one_ending_or_several(tmp_path):
    for name in ("a.tif", "b.png", ".hidden.tif"):
        (tmp_path / name).write_bytes(b"")
    assert sio._raw_image_names(str(tmp_path), ".tif") == ["a.tif"]
    assert sio._raw_image_names(str(tmp_path / "missing")) == []
    assert sio._channel_folders(str(tmp_path / "missing")) == []


def test_raw_images_in_orig_are_keyed_to_the_plate_folder(tmp_path):
    plate = tmp_path / "plate7"
    (plate / "orig").mkdir(parents=True)
    for field in (1, 2):
        for channel in (1, 2):
            tifffile.imwrite(plate / "orig" / f"A01_s{field}_w{channel}.tif",
                             np.full((6, 6), field * 10 + channel, np.uint16))
    regex = r"(?P<wellID>[A-Z]\d+)_s(?P<fieldID>\d+)_w(?P<chanID>\d+)\.tif"
    sio._rename_and_organize_image_files(str(plate), regex, 10, "custom",
                                         [".tif"])
    stacks = sorted(os.listdir(plate / "stack"))
    assert len(stacks) == 2
    assert all(name.startswith("plate7_") for name in stacks), stacks


# ---------------------------------------------------------------------------
# atomic writes and their rollback
# ---------------------------------------------------------------------------

def test_an_atomic_write_steps_past_a_temporary_name_already_taken(
        tmp_path, monkeypatch):
    draws = iter([b"\x00" * 8, b"\x01" * 8])
    monkeypatch.setattr(sio.os, "urandom", lambda n: next(draws))
    taken = tmp_path / f".spacr_tmp_{'00' * 8}{sio._PARTIAL_SUFFIX}"
    taken.write_bytes(b"someone else's")
    out = sio._replace_atomically(str(tmp_path / "out.bin"),
                                  lambda handle: handle.write(b"ours"))
    assert open(out, "rb").read() == b"ours"
    assert taken.read_bytes() == b"someone else's"


def test_a_failed_publish_rolls_back_around_files_that_vanished(
        tmp_path, monkeypatch):
    staging, output = tmp_path / "staging", tmp_path / "masks"
    staging.mkdir()
    output.mkdir()
    for name in ("a.npz", "b.npz"):
        (staging / name).write_bytes(b"new " + name.encode())
        (output / name).write_bytes(b"old " + name.encode())
    real = os.replace
    calls = []

    def replace(src, dst):
        calls.append((os.path.basename(src), os.path.dirname(dst)))
        if os.path.dirname(dst) == str(output) and \
                os.path.basename(src) == "b.npz":
            os.remove(output / "a.npz")
            backups = [p for p in tmp_path.iterdir()
                       if p.name.startswith(".spacr_previous_v1_npz_")]
            os.remove(backups[0] / "b.npz")
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(sio.os, "replace", replace)
    with pytest.raises(OSError, match="disk full"):
        sio._publish_v1_normalized_archives(str(staging), str(output))
    assert sorted(os.listdir(output)) == ["a.npz"]
    assert (output / "a.npz").read_bytes() == b"old a.npz"
    assert not [p for p in tmp_path.iterdir()
                if p.name.startswith(".spacr_previous_v1_npz_")]


# ---------------------------------------------------------------------------
# what a killed run leaves behind
# ---------------------------------------------------------------------------

def test_a_long_list_of_names_says_how_many_it_left_out():
    assert sio._name_list([str(i) for i in range(10)], limit=3) == \
        "0, 1, 2 and 7 more"


def test_unfinished_writes_are_swept_and_a_locked_one_is_left(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    assert sio._sweep_partial_writes(str(tmp_path / "missing")) == []
    for name in ("a", "b"):
        (tmp_path / f".spacr_tmp_{name}{sio._PARTIAL_SUFFIX}").write_bytes(b"")
    (tmp_path / "keep.npy").write_bytes(b"")
    real = os.remove

    def remove(path):
        if ".spacr_tmp_b" in path:
            raise PermissionError("in use")
        real(path)

    monkeypatch.setattr(sio.os, "remove", remove)
    removed = sio._sweep_partial_writes(str(tmp_path))
    assert removed == [f".spacr_tmp_a{sio._PARTIAL_SUFFIX}"]
    assert "Removed 1 unfinished write(s)" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f".spacr_tmp_b{sio._PARTIAL_SUFFIX}", "keep.npy"]


def test_a_second_damaged_copy_gets_its_own_name(tmp_path):
    for _ in range(2):
        (tmp_path / "f.npy").write_bytes(b"x")
        sio._set_aside(str(tmp_path / "f.npy"))
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "f.npy.damaged", "f.npy.damaged.1"]


def test_a_stack_whose_dtype_cannot_be_sized_is_checked_by_mapping_it(
        tmp_path):
    structured = tmp_path / "s.npy"
    np.save(structured, np.zeros(3, dtype=[("a", "i4"), ("b", "f8")]))
    assert sio._npy_is_whole(str(structured)) == (True, "done")
    pickled = tmp_path / "o.npy"
    np.save(pickled, np.array([1, None], dtype=object), allow_pickle=True)
    assert sio._npy_is_whole(str(pickled)) == (False, "unreadable")


def test_the_next_archive_number_ignores_other_files(tmp_path):
    for name in ("stack_2_norm.npz", "notes.npz", "stack_x_norm.npz"):
        (tmp_path / name).write_bytes(b"")
    assert sio._next_archive_index(str(tmp_path)) == 3


def _archive(path, names, n_fields=None):
    count = len(names) if n_fields is None else n_fields
    np.savez(path, data=np.zeros((count, 4, 4, 2), np.float32),
             filenames=names)


def test_a_batch_whose_names_are_bytes_is_not_dispatched(tmp_path):
    _archive(tmp_path / "b.npz", np.array([b"f0.npy", b"f1.npy"]))
    with pytest.raises(ValueError, match="must be Unicode strings"):
        sio._mask_batch_manifest(str(tmp_path))


def test_a_batch_with_no_fields_is_not_dispatched(tmp_path):
    _archive(tmp_path / "b.npz", np.array([], dtype="<U8"))
    with pytest.raises(ValueError, match="batch contains no fields"):
        sio._mask_batch_manifest(str(tmp_path))


# ---------------------------------------------------------------------------
# resuming from an earlier run's masks/
# ---------------------------------------------------------------------------

def _plate_with_masks(tmp_path, names):
    plate = tmp_path / "plate1"
    (plate / "masks").mkdir(parents=True)
    (plate / "stack").mkdir()
    for name in ("f0", "f1"):
        np.save(plate / "stack" / f"{name}.npy", np.zeros((4, 4, 2),
                                                          np.uint16))
    _archive(plate / "masks" / "stack_0_norm.npz", names)
    return plate


def test_a_raw_rebuild_that_fails_is_reported_and_the_resume_goes_on(
        tmp_path, monkeypatch, capsys):
    plate = _plate_with_masks(tmp_path, np.array(["f0.npy", "f1.npy"]))

    def broken(settings, src):
        raise RuntimeError("raw images unreadable")

    monkeypatch.setattr(sio, "_rebuild_stacks_from_raw", broken)
    assert sio._resume_normalized_archives({}, str(plate), [0, 1]) is True
    assert ("Could not build missing field stacks from the raw images: "
            "RuntimeError: raw images unreadable") in capsys.readouterr().out


def test_archives_listing_fields_as_objects_are_not_second_guessed(
        tmp_path, capsys):
    plate = _plate_with_masks(
        tmp_path, np.array(["f0.npy", "f1.npy"], dtype=object))
    assert sio._resume_normalized_archives({}, str(plate), [0, 1]) is True
    out = capsys.readouterr().out
    assert "1 archive(s) list their fields as an object array" in out
    assert sorted(os.listdir(plate / "masks")) == ["stack_0_norm.npz"]


def test_test_mode_takes_the_first_stacks_when_not_asked_to_shuffle(
        tmp_path):
    plate = tmp_path / "plate1"
    (plate / "stack").mkdir(parents=True)
    for name in ("c", "a", "b"):
        np.save(plate / "stack" / f"{name}.npy", np.zeros((2, 2, 1)))
    chosen = sio._sample_stacks_for_test_mode(str(plate), str(tmp_path / "t"),
                                              2, random_test=False)
    assert chosen == ["a.npy", "b.npy"]


@pytest.mark.parametrize("folders, advice", [
    ({"orig": ["x.tif"]}, ""),
    ({"masks": ["a.npz"]}, "Neither orig/ nor stack/ holds anything"),
])
def test_a_processed_folder_is_described_with_its_database(tmp_path, folders,
                                                           advice):
    for folder, names in folders.items():
        (tmp_path / folder).mkdir()
        for name in names:
            (tmp_path / folder / name).write_bytes(b"")
    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements" / "measurements.db").write_bytes(b"")
    summary, said = sio._describe_processed_folder(str(tmp_path))
    assert summary.endswith("measurements/ holds measurements.db")
    assert said.strip().startswith(advice) if advice else said == ""


def test_the_no_stacks_error_for_a_chosen_folder_that_is_gone(tmp_path):
    error = sio._no_stacks_error(str(tmp_path), str(tmp_path / "gone"),
                                 "regex", "cellvoyager")
    assert isinstance(error, FileNotFoundError)
    assert "Test mode copies a sample of the raw images in" in str(error)
    assert str(tmp_path / "gone") in str(error)


class _Stop(Exception):
    pass


def test_test_mode_never_resumes_an_earlier_runs_masks(tmp_path,
                                                       monkeypatch, capsys):
    import spacr.settings as settings_module

    plate = tmp_path / "plate1"
    (plate / "masks").mkdir(parents=True)
    (plate / "masks" / "stack_0_norm.npz").write_bytes(b"an earlier run's")

    def stop(settings):
        raise _Stop()

    monkeypatch.setattr(sio, "_resume_normalized_archives", lambda *a: \
                        pytest.fail("test mode resumed the masks folder"))
    monkeypatch.setattr(settings_module,
                        "set_default_settings_preprocess_img_data", stop)
    with pytest.raises(_Stop):
        sio.preprocess_img_data({"src": str(plate), "test_mode": True,
                                 "cell_channel": 0})
    assert ("Found existing masks folder; test mode works on a sample in "
            "test/ and does not reuse it.") in capsys.readouterr().out
    assert (plate / "masks" / "stack_0_norm.npz").read_bytes() == \
        b"an earlier run's"


def test_a_crop_folder_stamped_with_a_named_format(tmp_path):
    folder = tmp_path / "crops"
    folder.mkdir()
    path = sio.mark_crop_output_folder(str(folder), fmt=2)
    assert path and os.path.isfile(path)
    assert crops.crop_folder_format(str(folder)) == 2
