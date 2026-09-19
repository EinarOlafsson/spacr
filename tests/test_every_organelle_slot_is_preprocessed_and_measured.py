"""Every organelle slot is preprocessed and measured the way the first one is.

Items 364 and 76. The maintainer, asked on 2026-09-19 whether io.py's
per-channel preprocessing should loop over every organelle slot instead of
handling the first one only, answered "Yes, all slots".

Two things were measured on origin/nightly (92ea1ba00) before this change:

* ``spacr.io._normalize_img_batch`` gave a nucleus, cell, pathogen or
  first-organelle channel that object's own background floor and
  signal-to-noise anchor. It gave every other organelle slot's channel the
  generic ``background`` / ``Signal_to_noise`` pair and ignored
  ``organelleb_background`` and ``organelleb_signal_to_noise``, which the
  settings declare for every slot.
* A second slot already reached the database: raw TIFFs with two organelle
  channels, through the Mask pipeline and Measure, wrote an ``organelleb``
  table whose ``cell_id`` named the containing cell. What Measure received was
  wrong in another way. ``spacr.crops.reconcile_merged_mask_dims`` set
  ``<slot>_mask_dim`` for all 702 slots, ``None`` for the 700 the run never
  had. The settings factory then counted all 702 as declared, and the saved
  settings carried 2,100 rows for the 700 slots the run never had: 2,170
  settings rows where the same run now saves 70.

The end-to-end test stubs only the Cellpose forward pass for the cell and
nucleus. Preprocessing, the organelle segmenter for both slots, the merge
and Measure are the real ones.
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


def _normalise(settings, stack):
    from spacr.io import _normalize_img_batch

    base = {"lower_percentile": 2}
    base.update(settings)
    return _normalize_img_batch(
        stack=stack.copy(), channels=range(stack.shape[-1]),
        save_dtype=np.float32, settings=base)


def _ramp_stack():
    """One field, five channels; channel 4 holds the values 1..4096."""
    ramp = np.arange(1, 4097, dtype=np.float32).reshape(64, 64)
    stack = np.zeros((1, 64, 64, 5), dtype=np.float32)
    for channel in range(5):
        stack[0, :, :, channel] = ramp
    return stack


class TestASecondSlotIsNormalisedWithItsOwnValues:
    """The slot's background floor, anchor and switch reach its channel."""

    OWN = {"background": 300, "signal_to_noise": 2, "remove": True}

    def _slot(self, role):
        return {
            f"{role}_channel": 4,
            f"{role}_background": self.OWN["background"],
            f"{role}_signal_to_noise": self.OWN["signal_to_noise"],
            f"remove_background_{role}": self.OWN["remove"],
        }

    def test_the_second_slot_is_treated_exactly_as_the_first(self):
        stack = _ramp_stack()
        as_first = _normalise(self._slot("organelle"), stack)
        as_second = _normalise(
            {"organelle_channel": 3, **self._slot("organelleb")}, stack)
        np.testing.assert_array_equal(as_second[..., 4], as_first[..., 4])

    def test_its_own_values_change_the_result(self):
        stack = _ramp_stack()
        generic = _normalise({"organelle_channel": 3}, stack)
        as_second = _normalise(
            {"organelle_channel": 3, **self._slot("organelleb")}, stack)
        assert not np.array_equal(as_second[..., 4], generic[..., 4])
        below_floor = stack[0, :, :, 4] < self.OWN["background"]
        assert np.all(as_second[0][below_floor, 4] == 0)
        assert np.any(generic[0][below_floor, 4] > 0)

    def test_a_slot_the_run_does_not_enable_changes_nothing(self):
        stack = _ramp_stack()
        generic = _normalise({"organelle_channel": 3}, stack)
        disabled = _normalise(
            {"organelle_channel": 3, **self._slot("organelleb"),
             "organelleb_channel": None}, stack)
        np.testing.assert_array_equal(disabled, generic)

    def test_the_twentieth_slot_is_reached_too(self):
        from spacr.object_roles import ORGANELLE_ROLES

        role = ORGANELLE_ROLES[19]
        stack = _ramp_stack()
        as_first = _normalise(self._slot("organelle"), stack)
        as_twentieth = _normalise(
            {"organelle_channel": 3, **self._slot(role)}, stack)
        np.testing.assert_array_equal(as_twentieth[..., 4], as_first[..., 4])


def test_a_crop_cut_on_demand_takes_a_second_slots_plane_from_the_run(
        tmp_path):
    """``open_crop_source`` forwards every slot's mask plane, not the first's.

    On origin/nightly both lists that forward a run's crop-shaping settings
    -- ``spacr.io.CROP_SHAPE_KEYS`` and the one inside
    ``spacr.crops.resolve_crop_source`` -- named ``organelle_mask_dim`` and
    no other slot, so a run could not say where Organelle 2's plane is.
    """
    from spacr.io import open_crop_source

    merged = tmp_path / "plate1" / "merged"
    merged.mkdir(parents=True)
    np.save(merged / "plate1_A01_1.npy", np.zeros((8, 8, 7), np.uint16))
    source = open_crop_source(
        {"src": str(tmp_path / "plate1"), "crop_source": "merged",
         "cell_mask_dim": 2, "organelle_mask_dim": 5,
         "organelleb_mask_dim": 6},
        object_type="organelleb", verbose=False)
    assert source is not None and source.kind == "merged"
    assert source.spec.mask_dims["organelleb"] == 6
    assert source.spec.mask_dims["organelle"] == 5


SIZE = 96
LEFT_CELL = (slice(8, 88), slice(4, 44))
RIGHT_CELL = (slice(8, 88), slice(52, 92))
NUCLEI = ((slice(40, 56), slice(16, 32)), (slice(40, 56), slice(64, 80)))
SPOTS = {
    "organelle": [((14, 10), 5), ((70, 12), 5), ((74, 34), 5),
                  ((16, 60), 5)],
    "organelleb": [((24, 20), 6), ((70, 60), 6), ((20, 80), 6)],
}


def _channels():
    """Nucleus, cell, organelle 1 and organelle 2 images of one field."""
    nucleus = np.full((SIZE, SIZE), 50, np.uint16)
    cell = np.full((SIZE, SIZE), 50, np.uint16)
    for box in (LEFT_CELL, RIGHT_CELL):
        cell[box] = 800
    for box in NUCLEI:
        nucleus[box] = 900
    organelles = []
    for role in ("organelle", "organelleb"):
        image = np.full((SIZE, SIZE), 50, np.uint16)
        for (row, col), side in SPOTS[role]:
            image[row:row + side, col:col + side] = 3000
        organelles.append(image)
    return [nucleus, cell, *organelles]


def _label_the_painted_squares(src, settings, object_type):
    """Stand-in for the Cellpose pass: label the painted cells or nuclei."""
    from scipy import ndimage

    from spacr.io import _listdir_visible

    out = os.path.join(src, f"{object_type}_mask_stack")
    os.makedirs(out, exist_ok=True)
    position = settings[f"cellpose_{object_type}_channel"]
    for name in sorted(_listdir_visible(src)):
        if not name.endswith(".npz"):
            continue
        with np.load(os.path.join(src, name), allow_pickle=True) as archive:
            data, files = archive["data"], archive["filenames"]
        for index, filename in enumerate(files):
            labels, _count = ndimage.label(data[index, :, :, position] > 0.5)
            np.save(os.path.join(out, str(filename)), labels.astype(np.uint16))


@pytest.fixture(scope="module")
def two_organelle_plate(tmp_path_factory):
    """A plate run once through Mask and Measure with two organelle slots."""
    plate = tmp_path_factory.mktemp("two_organelles") / "plate1"
    plate.mkdir()
    for well in ("A01", "A02"):
        for channel, image in enumerate(_channels(), start=1):
            tifffile.imwrite(
                plate / f"plate1_{well}_T0001F001L01A01Z01C0{channel}.tif",
                image)
    with pytest.MonkeyPatch.context() as patch:
        _run_mask_and_measure(plate, patch)
    return plate


def _run_mask_and_measure(plate, patch):
    import spacr.measure as measure
    import spacr.object as spacr_object
    from spacr.core import preprocess_generate_masks
    from spacr.measure import measure_crop

    patch.setattr(spacr_object, "generate_cellpose_masks_sam",
                  _label_the_painted_squares)
    patch.setattr(
        measure, "_load_zernike_moments",
        lambda: (_ for _ in ()).throw(ImportError("not needed here")))
    patch.setattr(measure, "_ZERNIKE_AVAILABLE", None)

    preprocess_generate_masks({
        "src": str(plate), "metadata_type": "cellvoyager",
        "custom_regex": None, "channels": [0, 1, 2, 3],
        "nucleus_channel": 0, "cell_channel": 1, "pathogen_channel": None,
        "organelle_channel": 2, "organelleb_channel": 3,
        "number_of_organelles": 2,
        "organelle_method": "otsu", "organelleb_method": "otsu",
        "preprocess": True, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False, "save": True,
        "batch_size": 2, "randomize": False, "normalize": True,
    })
    merged = plate / "merged"
    measure_crop({
        "src": str(merged), "timelapse": False, "channels": [0, 1, 2, 3],
        "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
        "save_png": False, "save_arrays": False, "plot": False,
        "save_measurements": True, "n_jobs": 1, "verbose": False,
        "radial_dist": False, "homogeneity": False,
        "calculate_correlation": False, "experiment": "two-organelles",
    })


def _prcf(merged_name):
    well = merged_name.split("_")[1]
    return f"plate1_r1_c{int(well[1:])}_f1"


def _expected_parents(plate, role):
    """``{(prcf, object label): containing cell label}`` from the geometry.

    The cell a spot belongs to is read off the painted layout -- left or
    right half -- and the cell label at that cell's centre, not from any
    parent mapping spaCR computed.
    """
    import json

    merged = plate / "merged"
    layout = json.loads((merged / ".spacr_plane_layout.json").read_text())
    dims = layout["mask_dims"]
    expected = {}
    for name in sorted(os.listdir(merged)):
        if not name.endswith(".npy"):
            continue
        field = np.load(merged / name)
        cells = field[..., dims["cell"]].astype(int)
        left, right = cells[48, 24], cells[48, 72]
        assert left and right and left != right
        plane = field[..., dims[role]].astype(int)
        for label in np.unique(plane[plane > 0]):
            _rows, cols = np.nonzero(plane == label)
            expected[(_prcf(name), int(label))] = int(
                left if cols.mean() < SIZE / 2 else right)
    return expected


@pytest.mark.parametrize("role", ["organelle", "organelleb"])
def test_each_slot_reaches_the_database_with_its_containing_cell(
        two_organelle_plate, role):
    expected = _expected_parents(two_organelle_plate, role)
    assert len(expected) == 2 * len(SPOTS[role]), (
        f"{role}: the segmenter found {len(expected)} objects in two fields, "
        f"{len(SPOTS[role])} were painted in each")
    db = two_organelle_plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as connection:
        rows = connection.execute(
            f'SELECT prcf, object_label, cell_id FROM "{role}"').fetchall()
    measured = {(prcf, int(label)): int(cell) for prcf, label, cell in rows}
    assert measured == expected


def test_the_relationships_table_links_both_slots_to_their_cells(
        two_organelle_plate):
    from spacr.filters import build_relationships_frame

    db = two_organelle_plate / "measurements" / "measurements.db"
    frame = build_relationships_frame(str(db))
    for role in ("organelle", "organelleb"):
        rows = frame[frame["object_type"] == role]
        assert len(rows) == 2 * len(SPOTS[role])
        assert set(rows["parent_type"]) == {"cell"}
        assert rows["parent_label"].notna().all()


def test_measure_is_given_the_two_slots_the_run_has_and_no_others(
        two_organelle_plate):
    db = two_organelle_plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as connection:
        keys = [row[0] for row in connection.execute(
            "SELECT setting_key FROM settings")]
    slot_dims = sorted(key for key in keys
                       if key.startswith("organelle")
                       and key.endswith("_mask_dim"))
    assert slot_dims == ["organelle_mask_dim", "organelleb_mask_dim"]
    assert not [key for key in keys
                if key.startswith("organellec")], (
        "the saved settings carry a third slot the run never had")
