"""Item 0aj P1c: no stack/ folder. Mask preprocessing writes each field straight into fov.ome.zarr.

The segmentation stages (normalisation, image QC, the merge) read the
channels back through the one field-stack reader
(:func:`spacr.image_store._read_stack`); no ``stack/*.npy`` is written. The
merge adds the masks as labels without touching the stored channels, and
an older project with ``stack/*.npy`` still reads.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from tests.test_a_rerun_trusts_nothing_a_killed_run_left import (  # noqa: E402,F401
    _RUN, _settings, fake_cellpose,
)


def _run(plate, **over):
    from spacr.core import preprocess_generate_masks

    run = dict(_RUN, keep_intermediate=False, preprocess=True)
    run.update(over)
    preprocess_generate_masks(_settings(plate, **run))


def _merged(project, store):
    """``{field name: merged array}`` read through the shared merged-field reader."""
    from spacr.image_store import _list_fields, _read_field

    folder = Path(project) / ("fov.ome.zarr" if store else "merged")
    return {name: _read_field(folder / name) for name in _list_fields(folder)}


@pytest.mark.parametrize("channels", [[0, 1], [1]])
def test_a_mask_run_writes_no_stack_folder_and_matches_the_npy_run(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch, tmp_path, channels):
    import spacr.io as IO
    from spacr.image_store import _image_entries, _stack_names, verify_project

    plate = yokogawa_cellvoyager_dir["src"]
    npy_plate = tmp_path / "npy" / plate.name
    shutil.copytree(plate, npy_plate)
    _run(npy_plate, mask_image_format="npy", channels=channels, keep_intermediate=True)

    seen = {}
    real = IO.concatenate_and_normalize

    def normalise(src, *args, **kwargs):
        # Normalisation reads the fields from fov.ome.zarr: there is no stack/.
        seen["stack_dir"] = Path(src).exists()
        seen["fields"] = _stack_names(src)
        return real(src, *args, **kwargs)

    monkeypatch.setattr(IO, "concatenate_and_normalize", normalise)
    _run(plate, channels=channels, keep_intermediate=True)
    assert seen["stack_dir"] is False and len(seen["fields"]) == 4
    assert not (plate / "stack").exists() and not (plate / "merged").exists()
    assert all(e["image_planes"] == "stack" for e in _image_entries(plate / "fov.ome.zarr").values())

    before, after = _merged(npy_plate, False), _merged(plate, True)
    assert sorted(before) == sorted(after) and len(after) == 4
    for name in before:
        assert after[name].dtype == before[name].dtype
        assert after[name].shape == before[name].shape
        assert np.array_equal(after[name], before[name])
    # The stack the old run kept equals what the store holds, field by field.
    from spacr.image_store import _read_stack
    for name in _stack_names(npy_plate / "stack"):
        old = np.load(npy_plate / "stack" / name)
        new = _read_stack(plate / "stack" / name)
        assert new.dtype == old.dtype and np.array_equal(new, old)
    report = verify_project(plate)
    assert report["mismatches"] == [] and report["originals_compared"] == 8


def test_the_labels_are_added_without_touching_the_stored_channels(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch):
    import spacr.image_store as IS

    plate = yokogawa_cellvoyager_dir["src"]
    rewritten = []
    real = IS.PlateWriter.write_field

    def write_field(self, well, field, image, labels=None, **kwargs):
        rewritten.append((well, field, labels is not None and bool(labels)))
        return real(self, well, field, image, labels, **kwargs)

    monkeypatch.setattr(IS.PlateWriter, "write_field", write_field)
    _run(plate)
    # Every field's channels were written once, by preprocessing, with no labels.
    assert len(rewritten) == 4 and not any(has for _w, _f, has in rewritten)
    entries = IS._store_entries(plate / "fov.ome.zarr")
    assert len(entries) == 4 and all(e["labels"] for e in entries.values())


def test_a_preprocessing_run_stopped_mid_plate_resumes_field_by_field(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch):
    import spacr.image_store as IS

    plate = yokogawa_cellvoyager_dir["src"]
    written = []
    real = IS._write_stack_field

    def stop_after_two(writer, name, stack, channels):
        if len(written) == 2:
            raise KeyboardInterrupt("stopped while preprocessing")
        written.append(name)
        return real(writer, name, stack, channels)

    with monkeypatch.context() as patch:
        patch.setattr(IS, "_write_stack_field", stop_after_two)
        with pytest.raises(KeyboardInterrupt):
            _run(plate, keep_original_images=False)
    assert sorted(IS._image_entries(plate / "fov.ome.zarr")) == sorted(written)
    # No original went before its field was committed and verified.
    assert len(list(plate.glob("*.tif"))) == 8
    _run(plate, keep_original_images=False, resume=True)
    entries = IS._store_entries(plate / "fov.ome.zarr")
    assert len(entries) == 4 and not (plate / "stack").exists()
    assert list(plate.glob("*.tif")) == [] and not (plate / "orig").exists()
    assert IS.verify_store(plate)["mismatches"] == []


def test_an_older_project_with_stack_npy_still_reads(yokogawa_cellvoyager_dir, fake_cellpose):
    from spacr.image_store import _read_stack, _stack_exists, _stack_header, _stack_names
    from spacr.io import preprocess_img_data

    plate = yokogawa_cellvoyager_dir["src"]
    # A direct preprocessing call (outside a Mask run) still writes stack/*.npy.
    preprocess_img_data(_settings(plate))
    names = _stack_names(plate / "stack")
    assert len(names) == 4 and all((plate / "stack" / n).is_file() for n in names)
    assert _stack_exists(plate / "stack")
    array = _read_stack(plate / "stack" / names[0])
    assert _stack_header(plate / "stack" / names[0]) == (array.shape, array.dtype)
    # A Mask run on it merges from those stacks as before.
    _run(plate, preprocess=False)
    from spacr.image_store import _store_entries
    assert len(_store_entries(plate / "fov.ome.zarr")) == 4


def test_streamed_crops_read_the_selected_channels_of_a_stack_field(
        yokogawa_cellvoyager_dir, fake_cellpose):
    from spacr.crop_store import _field_windows
    from spacr.image_store import _read_field, _store_entries, _store_field

    plate = yokogawa_cellvoyager_dir["src"]
    _run(plate, channels=[1])
    store = plate / "fov.ome.zarr"
    name = sorted(_store_entries(store))[0]
    path = str(store / name)
    merged = _read_field(path)
    windows = _field_windows(path, *_store_field(path))
    box = windows.channels([0], 5, 40, 7, 30)
    assert np.array_equal(box[0], merged[5:40, 7:30, 0])
