"""The crop paths that only a real dataset's awkward corners reach.

A crop is cut from a memory-mapped field, so every window that runs off the
edge, every plane that is not on disk, and every recorded path written by a
different machine has to come back as data rather than as an exception. This
file drives the ones the ordinary happy-path tests never reach:

* a window that misses the field completely -- which is what a stale bounding
  box on a re-cropped field produces -- must read back as background, not as
  an index error;
* the merged-field cache has to be able to *measure itself* while it holds a
  label index, because that measurement is what the memory sweep evicts on;
* a colour left empty in ``png_channel_mapping`` has to become a zero plane,
  and two empty colours have to share one;
* a recorded field name that does not end in a number must not have its last
  segment stripped off as if it were a crop's object suffix.

The module is also loaded here the way a dependency-light consumer loads it --
straight off its file, with no package around it -- because the fallback that
makes that work has to reach the *same* role vocabulary rather than a second
copy of it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import crops                                          # noqa: E402
from spacr.crops import (CropSpec, MergedCropSource,             # noqa: E402
                         build_png_channels, open_merged_field,
                         reconcile_merged_mask_dims)

CELL_DIM, NUC_DIM, PATH_DIM = 4, 5, 6
MASK_DIMS = {"cell": CELL_DIM, "nucleus": NUC_DIM, "pathogen": PATH_DIM}


@pytest.fixture(autouse=True)
def _empty_field_cache():
    """No test may inherit or leave a cached merged field."""
    crops.clear_field_cache()
    yield
    crops.clear_field_cache()


def _field(h=32, w=40, n_channels=4, seed=0):
    """A merged array: intensity planes, then cell / nucleus / pathogen."""
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 4000, size=(h, w, n_channels + 3)).astype(np.uint16)
    for dim in (CELL_DIM, NUC_DIM, PATH_DIM):
        data[:, :, dim] = 0
    data[8:16, 10:20, CELL_DIM] = 1
    data[10:14, 12:18, NUC_DIM] = 1
    data[11:13, 13:15, PATH_DIM] = 1
    return data


def _write(tmp_path, data, name="plate1_A01_1.npy"):
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    path = merged / name
    np.save(path, data)
    return str(path)


# ---------------------------------------------------------------------------
# The plane manifest
# ---------------------------------------------------------------------------

def test_a_merged_folder_without_a_manifest_leaves_the_mask_dims_alone(tmp_path):
    """A legacy folder has no manifest, so nothing may be reconciled away."""
    folder = tmp_path / "merged"
    folder.mkdir()
    settings = {"cell_mask_dim": 4, "nucleus_mask_dim": 5, "src": "somewhere"}

    out = reconcile_merged_mask_dims(settings, str(folder),
                                     explicit_keys=("cell_mask_dim",))

    assert out == settings
    assert out is not settings, "the settings must be copied, not mutated"


def test_a_manifest_replaces_the_default_mask_dims(tmp_path):
    """With a manifest present the folder's own plane indices win."""
    folder = tmp_path / "merged"
    folder.mkdir()
    (folder / crops.MERGED_LAYOUT_SIDECAR).write_text(json.dumps({
        "version": 1,
        "intensity_channels": [0, 1],
        "mask_plane_order": ["cell", "nucleus"],
        "mask_dims": {"cell": 2, "nucleus": 3},
    }), encoding="utf-8")

    out = reconcile_merged_mask_dims({"cell_mask_dim": None}, str(folder))

    assert out["cell_mask_dim"] == 2
    assert out["nucleus_mask_dim"] == 3


# ---------------------------------------------------------------------------
# Windows that miss the field
# ---------------------------------------------------------------------------

def test_a_window_entirely_off_the_field_reads_back_as_zeros(tmp_path):
    """A window with no overlap is all padding, at the requested shape."""
    field = open_merged_field(_write(tmp_path, _field(seed=1)), MASK_DIMS)

    off = field.read_window(-40, -30, -50, -40, (0, 1))

    assert off.shape == (10, 10, 2)
    assert off.dtype == np.uint16
    assert not off.any()

    # A window that does overlap is not all zeros, so the assertion above is
    # about the missing overlap and not about an empty field.
    overlapping = field.read_window(0, 10, 0, 10, (0, 1))
    assert overlapping.any()


def test_a_mask_window_off_the_field_is_all_background(tmp_path):
    """Reading a label plane past the edge yields background, not an error."""
    field = open_merged_field(_write(tmp_path, _field(seed=2)), MASK_DIMS)

    beyond = field.read_mask_window("cell", 100, 108, 100, 112)
    before = field.read_mask_window("cell", -20, -12, -30, -20)

    assert beyond.shape == (8, 12)
    assert not beyond.any()
    assert before.shape == (8, 10)
    assert not before.any()
    # The same plane inside the field does hold the object.
    assert field.read_mask_window("cell", 8, 16, 10, 20).max() == 1


def test_a_cytoplasm_window_off_the_field_is_all_background(tmp_path):
    """The derived cytoplasm plane pads the same way the on-disk planes do."""
    field = open_merged_field(_write(tmp_path, _field(seed=3)), MASK_DIMS)

    beyond = field.read_mask_window("cytoplasm", -20, -12, -30, -20)

    assert beyond.shape == (8, 10)
    assert not beyond.any()
    # Inside the field, cytoplasm is cell minus nucleus and pathogen.
    inside = field.read_mask_window("cytoplasm", 8, 16, 10, 20)
    assert inside.max() == 1
    assert inside[10 - 8:14 - 8, 12 - 10:18 - 10].max() == 0


# ---------------------------------------------------------------------------
# The merged-field cache measures itself
# ---------------------------------------------------------------------------

def test_the_cache_budget_counts_the_label_index_it_built(tmp_path):
    """A cached field reports the mapping AND the index scanned out of it."""
    path = _write(tmp_path, _field(seed=4))
    field = open_merged_field(path, MASK_DIMS)
    bare = crops.cache_budget_entries()
    assert len(bare) == 1
    mapping_only = bare[0][1]
    assert mapping_only >= field.array.nbytes

    index = field.label_index("cell")
    field.mask_plane("cytoplasm")

    rows = crops.cache_budget_entries()
    assert len(rows) == 1
    key, measured, last_used, in_use = rows[0]
    assert key in crops._FIELD_CACHE
    assert in_use is False
    assert isinstance(last_used, float)
    # The index arrays and the derived cytoplasm plane are now addressable
    # through the cache, so the budget has to have grown by at least them.
    index_bytes = sum(getattr(index, name).nbytes
                      for name in ("labels", "ymin", "ymax", "xmin", "xmax",
                                   "count"))
    assert measured >= mapping_only + index_bytes


def test_an_empty_label_plane_is_measured_without_double_counting(tmp_path):
    """An index over a plane with no objects shares one empty array per slot."""
    data = _field(seed=14)
    data[:, :, PATH_DIM] = 0                      # nothing was segmented here
    path = _write(tmp_path, data)
    field = open_merged_field(path, MASK_DIMS)

    empty = field.label_index("pathogen")
    assert empty.labels.size == 0
    # The empty index aliases one zero-length array across five of its slots.
    assert empty.ymin is empty.ymax is empty.count

    measured = crops.cache_budget_entries()[0][1]
    assert measured >= field.array.nbytes


def test_a_field_whose_array_cannot_size_itself_does_not_stop_the_sweep(
        tmp_path):
    """One unmeasurable field counts as zero; the rest are still measured."""

    class _Proxy:
        """A stand-in array: shape, dtype, ndim and indexing, no ``nbytes``."""

        def __init__(self, data):
            self._data = data
            self.shape = data.shape
            self.dtype = data.dtype
            self.ndim = data.ndim

        def __getitem__(self, item):
            return self._data[item]

    real_path = _write(tmp_path, _field(seed=15))
    real = open_merged_field(real_path, MASK_DIMS)
    proxied = crops.MergedField("proxy.npy", array=_Proxy(_field(seed=16)),
                                mask_dims=MASK_DIMS)
    crops._FIELD_CACHE[("proxy.npy", 0, 0)] = proxied

    rows = {key: measured for key, measured, _used, _busy
            in crops.cache_budget_entries()}

    assert len(rows) == 2
    assert rows[("proxy.npy", 0, 0)] == 0
    real_key = next(k for k in rows if k != ("proxy.npy", 0, 0))
    assert rows[real_key] >= real.array.nbytes


def test_dropping_a_cached_field_says_whether_it_was_there(tmp_path):
    """The evictor reports what it actually removed, and forgets its clock."""
    path = _write(tmp_path, _field(seed=5))
    open_merged_field(path, MASK_DIMS)
    key = next(iter(crops._FIELD_CACHE))

    assert crops.drop_cache_budget_entry(key) is True
    assert key not in crops._FIELD_CACHE
    assert key not in crops._FIELD_CACHE_USED
    assert crops.cache_budget_entries() == []

    # Evicting it a second time is not an error, and says nothing was there.
    assert crops.drop_cache_budget_entry(key) is False


def test_caching_a_new_field_arms_the_budget_sweep_once(tmp_path, monkeypatch):
    """The sweep is armed when a field enters the cache, not on a cache hit."""
    installs = []
    stub = types.ModuleType("spacr.qt.resource_cleanup")
    stub.install_budget_sweep = lambda: installs.append(1)
    monkeypatch.setitem(sys.modules, "spacr.qt.resource_cleanup", stub)

    path = _write(tmp_path, _field(seed=6))
    open_merged_field(path, MASK_DIMS)
    assert len(installs) == 1

    open_merged_field(path, MASK_DIMS)          # served from the cache
    assert len(installs) == 1

    second = _write(tmp_path, _field(seed=7), "plate1_A01_2.npy")
    open_merged_field(second, MASK_DIMS)
    assert len(installs) == 2


def test_a_cleanup_module_without_the_sweep_is_not_an_error(tmp_path,
                                                            monkeypatch):
    """An older resource_cleanup has no sweep to install; opening still works."""
    stub = types.ModuleType("spacr.qt.resource_cleanup")
    monkeypatch.setitem(sys.modules, "spacr.qt.resource_cleanup", stub)

    field = open_merged_field(_write(tmp_path, _field(seed=8)), MASK_DIMS)

    assert field.labels("cell") == [1]


# ---------------------------------------------------------------------------
# Empty colour slots
# ---------------------------------------------------------------------------

def test_two_empty_colours_become_two_blank_planes(tmp_path):
    """Green and blue left unmapped are written as zeros, not as red again."""
    data = _field(seed=9)

    out = build_png_channels(data, {"r": 0, "g": None, "b": None})

    assert out.shape == (data.shape[0], data.shape[1], 3)
    np.testing.assert_array_equal(out[:, :, 0], data[:, :, 0])
    assert not out[:, :, 1].any()
    assert not out[:, :, 2].any()


def test_one_empty_colour_leaves_the_others_alone(tmp_path):
    """Only the unmapped slot is blanked; the mapped ones keep their source."""
    data = _field(seed=10)

    out = build_png_channels(data, {"r": 2, "g": None, "b": 0})

    np.testing.assert_array_equal(out[:, :, 0], data[:, :, 2])
    assert not out[:, :, 1].any()
    np.testing.assert_array_equal(out[:, :, 2], data[:, :, 0])


# ---------------------------------------------------------------------------
# Reading a row
# ---------------------------------------------------------------------------

def test_a_null_label_column_falls_through_to_the_next_one(tmp_path):
    """A row carrying object_label=NULL is answered by its cell_id instead."""
    path = _write(tmp_path, _field(seed=11))
    source = MergedCropSource(
        CropSpec(merged_path="", channels=(0, 1, 2), size=(12, 12),
                 mask_dims=MASK_DIMS),
        merged_root=os.path.dirname(path))

    spec = source.spec_for({"path_name": path, "object_label": None,
                            "cell_id": 1})

    assert spec.label == 1
    assert spec.merged_path == path
    crop = source.get_array({"path_name": path, "object_label": None,
                             "cell_id": 1})
    assert crop.shape == (12, 12, 3)
    assert crop.any(), "the object's pixels must survive the crop"


def test_a_field_name_that_does_not_end_in_a_number_is_not_shortened(tmp_path):
    """Only a trailing object suffix is dropped, never a real name segment."""
    root = tmp_path / "merged"
    root.mkdir()
    # A decoy the shortening rule would land on if it dropped any last
    # segment rather than only a numeric one.
    np.save(root / "plate1_A01.npy", _field(seed=18))
    source = MergedCropSource(merged_root=str(root))

    resolved = source.resolve_path({"file_name": "plate1_A01_top.tif"})

    assert resolved == os.path.join(str(root), "plate1_A01_top.npy")


def test_a_crop_row_drops_its_object_suffix_to_find_the_field(tmp_path):
    """A png_list name carries the object label; the field name does not."""
    root = tmp_path / "merged"
    root.mkdir()
    np.save(root / "plate1_A01_17_1.npy", _field(seed=12))
    source = MergedCropSource(merged_root=str(root))

    resolved = source.resolve_path({"file_name": "plate1_A01_17_1_2.png"})

    assert resolved == os.path.join(str(root), "plate1_A01_17_1.npy")


# ---------------------------------------------------------------------------
# The invariant behind the unmasked-crop branch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_bounding_box", [False, True])
@pytest.mark.parametrize("dilate,ratio", [(False, 0.2), (True, 0.0),
                                          (True, 0.2)])
def test_every_crop_is_masked_to_its_object(tmp_path, use_bounding_box,
                                            dilate, ratio):
    """A crop always carries a region, so its far corners are background.

    ``_crop_from_field`` still has an unmasked path for a region that covers
    the whole field -- what ``binary_dilation(..., iterations=0)`` used to
    produce before the radius was guarded. No setting reaches it any more,
    and this is the invariant that says so: whatever the dilation asks for,
    the window is wider than the region and its corners stay zero.
    """
    path = _write(tmp_path, _field(h=64, w=64, seed=17))
    crop = crops.extract_crop(
        path, "cell", 1, channels=(0, 1, 2), size=(60, 60),
        mask_dims=MASK_DIMS, use_bounding_box=use_bounding_box,
        dilate=dilate, dilate_ratio=ratio)

    assert crop.shape == (60, 60, 3)
    assert crop.any(), "the object itself must survive"
    assert not crop[0, 0].any(), "the far corner is outside the region"
    assert not crop[-1, -1].any()


# ---------------------------------------------------------------------------
# Loading the module without a package around it
# ---------------------------------------------------------------------------

def test_loading_the_module_off_its_file_reaches_the_same_role_vocabulary():
    """The dependency-light load must share the role registry, not copy it."""
    path = os.path.abspath(crops.__file__)
    spec = importlib.util.spec_from_file_location("_crops_standalone_probe",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)

        assert module.__name__ == "_crops_standalone_probe"
        assert module.ALL_ROLES == crops.ALL_ROLES
        assert module.SEGMENTED_ROLES == crops.SEGMENTED_ROLES
        assert module.ORGANELLE_ROLES == crops.ORGANELLE_ROLES
        # And the crop it cuts is the crop the package cuts.
        assert module.DEFAULT_MASK_DIMS == crops.DEFAULT_MASK_DIMS
    finally:
        sys.modules.pop(spec.name, None)
        sys.modules.pop("_spacr_crops_schema", None)


def test_the_standalone_module_cuts_the_same_pixels_as_the_package(tmp_path):
    """Same file, no package: the crop has to be identical, not merely valid."""
    path = _write(tmp_path, _field(seed=13))
    spec_kwargs = dict(object_type="cell", label=1, channels=(0, 1, 2),
                       size=(16, 16), mask_dims=MASK_DIMS)

    module_spec = importlib.util.spec_from_file_location(
        "_crops_standalone_probe", os.path.abspath(crops.__file__))
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    try:
        module_spec.loader.exec_module(module)
        standalone = module.extract_crop(path, **spec_kwargs)
    finally:
        sys.modules.pop(module_spec.name, None)
        sys.modules.pop("_spacr_crops_schema", None)

    packaged = crops.extract_crop(path, **spec_kwargs)

    assert standalone.shape == (16, 16, 3)
    np.testing.assert_array_equal(standalone, packaged)
